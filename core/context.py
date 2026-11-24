"""Application context: the dependency-injection container passed to services.

Built once per process by ``main.py`` (engine / replay / analyze) or the dashboard.
Services receive the context (or the specific pieces they need) instead of reaching for
globals, which keeps them testable and replay-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from core.clock import Clock, SystemClock

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm import Session, sessionmaker

    from config.schema import HardCaps, Settings
    from config.secrets import Secrets

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppContext:
    """Runtime container.

    Attributes:
        settings: active ``config.schema.Settings`` (latest settings_versions row).
        settings_version_id: id of that settings_versions row (None in pure unit tests).
        caps: ``config.schema.HardCaps`` loaded from disk (immutable at runtime).
        clock: ``SystemClock`` live, ``SimulatedClock`` in replay/tests.
        session_factory: SQLAlchemy ``sessionmaker`` bound to the trading DB.
        secrets: ``config.secrets.Secrets`` (.env values).
        batch_id: id of the ACTIVE batch (None until a batch exists).
        data: ``data.service.MarketDataService`` (set by bootstrap).
        llm: ``agents.llm.LLMClient`` (set by bootstrap).
        notifier: ``notifications.service.Notifier`` (set by bootstrap).
        is_replay: True in REPLAY / walk-forward mode.
        runtime_dir: writable directory for DB, caches, CA bundle.
        extras: free-form registry for late-bound services (engine, broker, ...).
    """

    settings: "Settings"
    caps: "HardCaps"
    session_factory: "sessionmaker[Session]"
    clock: Clock = field(default_factory=SystemClock)
    settings_version_id: int | None = None
    secrets: "Secrets | None" = None
    batch_id: int | None = None
    data: Any = None
    llm: Any = None
    notifier: Any = None
    is_replay: bool = False
    runtime_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runtime")
    extras: dict[str, Any] = field(default_factory=dict)

    def session(self) -> "Session":
        """Open a new SQLAlchemy session (caller manages commit/close, or use storage.db.session_scope)."""
        return self.session_factory()

    def reload_settings(self, loader: Callable[["AppContext"], tuple[int | None, "Settings"]]) -> None:
        """Replace settings using ``loader(ctx) -> (version_id, settings)``."""
        version_id, settings = loader(self)
        self.settings_version_id = version_id
        self.settings = settings
