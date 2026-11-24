"""Per-session dashboard context (section 14.2).

The dashboard is a separate process from the engine. It only needs:

* a SQLAlchemy ``sessionmaker`` bound to the shared trading DB (``Secrets().database_url``),
* the immutable ``HardCaps`` (to clamp inputs and display caps),
* a ``Clock`` (``SystemClock`` - IST wall time),
* the logged-in user and the batch selected in the sidebar.

``get_ctx()`` returns a :class:`DashboardContext` cached in ``st.session_state``. The DB engine
is cached per process (``st.cache_resource``) keyed by the DB URL, so every browser session
shares one connection pool. ``core.guard.startup_safety_check`` runs once per process before
anything else touches the DB.

The dashboard NEVER imports ``agents/``, ``paper_engine/``, ``risk/``, ``scheduler/`` or engine code.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.loader import load_hard_caps
from config.schema import HardCaps
from config.secrets import Secrets
from core.clock import Clock, SystemClock
from core.guard import startup_safety_check
from core.logging import get_log_dir
from storage.batches import get_current_batch
from storage.db import init_db, make_engine, make_session_factory

#: ``st.session_state`` keys shared by app.py, auth.py and the pages.
CTX_KEY = "_dashboard_ctx"
AUTH_USER_KEY = "auth_user"
AUTH_LAST_ACTIVITY_KEY = "auth_last_activity"
SELECTED_BATCH_KEY = "selected_batch_id"

_safety_lock = threading.Lock()
_safety_done = False


def ensure_startup_safety(caps: HardCaps) -> None:
    """Run the paper-only startup assertion + order guards once per process."""
    global _safety_done
    with _safety_lock:
        if _safety_done:
            return
        patched = startup_safety_check(caps)
        logger.info(f"Dashboard paper-only guard installed ({len(patched)} targets patched)")
        _safety_done = True


@dataclass
class DashboardContext:
    """Everything a dashboard page needs. Cached per browser session.

    Attributes:
        session_factory: sessionmaker bound to the trading DB.
        engine: the SQLAlchemy engine (disposed after a DB restore).
        caps: hard caps from ``config/hard_caps.yaml`` (read-only).
        clock: IST clock. Never call ``datetime.now()`` in dashboard code.
        secrets: ``.env`` values (DB URL, auth file path, Ollama host override...).
        user: logged-in username (None until login).
        batch_id: batch selected in the sidebar (defaults to the current batch).
        runtime_dir: writable runtime directory (backups go to ``runtime/backups``).
        log_dir: JSONL log root used by the log viewer.
    """

    session_factory: sessionmaker[Session]
    engine: Engine
    caps: HardCaps
    clock: Clock
    secrets: Secrets
    db_url: str
    user: str | None = None
    batch_id: int | None = None
    runtime_dir: Path = field(default_factory=lambda: Path("runtime"))
    log_dir: Path = field(default_factory=get_log_dir)

    def session(self) -> Session:
        """Open a new session (use as a context manager; commit explicitly for writes)."""
        return self.session_factory()

    @property
    def actor(self) -> str:
        """Audit actor name for actions taken from the dashboard."""
        return self.user or "dashboard"

    @property
    def backup_dir(self) -> Path:
        return self.runtime_dir / "backups"


@st.cache_resource(show_spinner=False)
def _engine_for(url: str) -> tuple[Engine, sessionmaker[Session]]:
    """One engine + session factory per DB URL per process (tables created idempotently)."""
    engine = make_engine(url)
    init_db(engine)
    return engine, make_session_factory(engine)


def build_context(secrets: Secrets | None = None, clock: Clock | None = None) -> DashboardContext:
    """Build a fresh context (no Streamlit session involved)."""
    secrets = secrets or Secrets()
    caps = load_hard_caps()
    ensure_startup_safety(caps)
    engine, factory = _engine_for(secrets.database_url)
    return DashboardContext(
        session_factory=factory,
        engine=engine,
        caps=caps,
        clock=clock or SystemClock(),
        secrets=secrets,
        db_url=secrets.database_url,
        runtime_dir=Path(secrets.runtime_dir),
        log_dir=get_log_dir(),
    )


def _default_batch_id(ctx: DashboardContext) -> int | None:
    try:
        with ctx.session() as session:
            batch = get_current_batch(session)
            return batch.id if batch is not None else None
    except Exception as exc:  # noqa: BLE001 - dashboard must still render
        logger.error(f"Could not determine current batch: {exc}")
        return None


def get_ctx() -> DashboardContext:
    """Return the session's :class:`DashboardContext`, creating it on first use.

    ``user`` and ``batch_id`` are refreshed from ``st.session_state`` on every call so the
    sidebar selection and login state are always current.
    """
    ctx: Any = st.session_state.get(CTX_KEY)
    secrets_url = Secrets().database_url
    if not isinstance(ctx, DashboardContext) or ctx.db_url != secrets_url:
        ctx = build_context()
        st.session_state[CTX_KEY] = ctx
    ctx.user = st.session_state.get(AUTH_USER_KEY)
    if SELECTED_BATCH_KEY not in st.session_state:
        st.session_state[SELECTED_BATCH_KEY] = _default_batch_id(ctx)
    ctx.batch_id = st.session_state.get(SELECTED_BATCH_KEY)
    return ctx


def set_selected_batch(batch_id: int | None) -> None:
    """Change the sidebar batch filter."""
    st.session_state[SELECTED_BATCH_KEY] = batch_id
