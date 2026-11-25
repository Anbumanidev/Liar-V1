"""Database engine / session helpers.

The engine process and the dashboard process share one SQLite file, so SQLite runs in
WAL mode with a busy timeout. Switch to PostgreSQL by setting ``DATABASE_URL`` in ``.env``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from storage.models import Base


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=OFF")
    cur.execute("PRAGMA busy_timeout=30000")   # engine + dashboards share the file; wait, don't fail
    cur.close()


def make_engine(url: str, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine. ``sqlite://`` (no path) = shared in-memory DB for tests."""
    if url.startswith("sqlite"):
        if url in ("sqlite://", "sqlite:///:memory:"):
            engine = create_engine(
                "sqlite://",
                echo=echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            db_path = url.split("sqlite:///", 1)[-1]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(
                url, echo=echo, connect_args={"check_same_thread": False, "timeout": 30}
            )
        event.listen(engine, "connect", _sqlite_pragmas)
        return engine
    return create_engine(url, echo=echo, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory with ``expire_on_commit=False`` so returned rows stay readable."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error, always close."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_for_url(url: str) -> sessionmaker[Session]:
    """Convenience: engine + tables + session factory for ``url``."""
    engine = make_engine(url)
    init_db(engine)
    return make_session_factory(engine)
