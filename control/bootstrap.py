"""Process bootstrap: builds every service once and wires them together.

Used by the engine (``main.py engine``), REPLAY / walk-forward and ``main.py analyze``.
The dashboard never calls this (it only reads the DB and writes control commands).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session, sessionmaker

from config.loader import load_hard_caps
from config.schema import HardCaps, Settings
from config.secrets import Secrets, load_secrets
from config.store import get_active_settings
from core.clock import Clock, SystemClock
from core.context import AppContext
from core.guard import startup_safety_check
from core.logging import set_log_dir, setup_logging
from core.net import configure_ssl
from core.types import MarketEvent
from storage.db import create_all_for_url


@dataclass
class Services:
    """All long-lived services of one process."""

    ctx: AppContext
    calendar: Any
    data: Any
    universe: Any
    nse: Any | None
    events: Any | None
    fundamentals: Any | None
    news: Any | None
    sentiment: Any | None
    features: Any
    llm: Any
    prompts: Any | None
    notifier: Any | None
    capital: Any
    risk: Any
    broker: Any
    monitor: Any
    shadow: Any
    queue: Any
    manual: Any
    pipeline: Any
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def settings(self) -> Settings:
        return self.ctx.settings

    @property
    def caps(self) -> HardCaps:
        return self.ctx.caps

    @property
    def clock(self) -> Clock:
        return self.ctx.clock

    @property
    def session_factory(self) -> sessionmaker[Session]:
        return self.ctx.session_factory

    def session(self) -> Session:
        return self.ctx.session_factory()

    def reload_settings(self) -> int | None:
        """Load the latest settings version and push it into every service."""
        with self.session() as session:
            version_id, settings = get_active_settings(session, self.caps, clock=self.clock)
            session.commit()
        self.apply_settings(settings, version_id)
        return version_id

    def apply_settings(self, settings: Settings, version_id: int | None = None) -> None:
        old = self.ctx.settings
        if (old.schedule.holiday_overrides != settings.schedule.holiday_overrides
                or old.schedule.special_sessions != settings.schedule.special_sessions):
            self._rebuild_calendar(settings)
        self.ctx.settings = settings
        self.ctx.settings_version_id = version_id
        for svc in (self.capital, self.risk, self.broker, self.monitor, self.shadow, self.queue, self.manual,
                    self.pipeline, self.notifier):
            fn = getattr(svc, "update_settings", None)
            if callable(fn):
                fn(settings)
        logger.info(f"[bootstrap] settings version {version_id} applied")

    def _rebuild_calendar(self, settings: Settings) -> None:
        """Holiday overrides / special sessions changed: rebuild the calendar everywhere."""
        from data.calendar import load_calendar

        cal = load_calendar(settings.schedule, self.ctx.runtime_dir / "cache", fetch=False)
        self.calendar = cal
        for svc in (self.data, self.broker, self.monitor, self.shadow, self.queue, self.manual,
                    getattr(self.pipeline, "s", None), self.features):
            if svc is not None and hasattr(svc, "calendar"):
                svc.calendar = cal
        logger.info("[bootstrap] NSE calendar rebuilt (holiday overrides changed)")

    def events_for_window(self, symbol: str, mode: Any, now: datetime) -> list[MarketEvent]:
        """Events inside the holding window of a new ``mode`` position opened at ``now``."""
        if self.events is None or self.calendar is None:
            return []
        from core.enums import Mode

        m = Mode(mode)
        return self.events.events_in_window(symbol, m, now.date(), self.settings.mode(m).max_holding_days or 1,
                                            self.calendar)


def bootstrap(
    *,
    process: str = "engine",
    clock: Clock | None = None,
    db_url: str | None = None,
    primary_provider: Any | None = None,
    is_replay: bool = False,
    llm_backend: str | None = None,
    with_notifier: bool = True,
    runtime_dir: Path | None = None,
    log_dir: Path | None = None,
    settings_override: Settings | None = None,
    secrets: Secrets | None = None,
    offline: bool = False,
    console_log: bool = True,
) -> Services:
    """Build the full service graph for a process.

    Args:
        process: log file name (engine / replay / analyze).
        clock: SystemClock (live) or SimulatedClock (replay / tests).
        db_url: database URL (defaults to ``Secrets.database_url``).
        primary_provider: override the market data provider (e.g. ``ReplayDataProvider``).
        is_replay: replay mode (no live network services, synchronous email disabled).
        llm_backend: override ``settings.engine.llm_backend`` ("ollama" | "rules").
        with_notifier: build the email notifier.
        settings_override: use these settings instead of the stored active version.
        offline: never touch the network (tests / replay with preloaded data).
    """
    secrets = secrets or load_secrets()
    runtime_dir = Path(runtime_dir or secrets.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir or secrets.log_dir)
    setup_logging(process_name=process, log_dir=log_dir, console=console_log)
    set_log_dir(log_dir)
    caps = load_hard_caps()
    patched = startup_safety_check(caps)
    logger.info(f"[bootstrap] PAPER TRADING ONLY - order guards installed on {len(patched)} targets")
    clock = clock or SystemClock()
    if not offline:
        configure_ssl(runtime_dir, use_os_store=True, ca_bundle=secrets.ca_bundle_path)

    session_factory = create_all_for_url(db_url or secrets.database_url)
    with session_factory() as session:
        version_id, settings = get_active_settings(session, caps, clock=clock)
        session.commit()
    if settings_override is not None:
        settings = settings_override
    if llm_backend:
        from config.loader import update_settings

        settings = update_settings(settings, {"engine.llm_backend": llm_backend})
    if secrets.ollama_host:
        from config.loader import update_settings

        settings = update_settings(settings, {"models.ollama_host": secrets.ollama_host})

    ctx = AppContext(settings=settings, caps=caps, session_factory=session_factory, clock=clock,
                     settings_version_id=version_id, secrets=secrets, is_replay=is_replay, runtime_dir=runtime_dir)
    cache_dir = runtime_dir / "cache"

    from data.calendar import load_calendar
    from data.events import EventCalendar
    from data.service import build_market_data_service
    from data.universe import UniverseService

    calendar = load_calendar(settings.schedule, cache_dir, fetch=not (offline or is_replay))
    data = build_market_data_service(settings, secrets, clock, calendar, cache_dir, session_factory=session_factory,
                                     primary=primary_provider)
    universe = UniverseService()
    nse = None
    if not (offline or is_replay):
        from data.nse import NSEClient

        nse = NSEClient(cache_dir, clock, session_factory=session_factory)
    events = EventCalendar(session_factory, nse, clock)
    fundamentals = news = sentiment = None
    if not (offline or is_replay):
        from data.fundamentals import FundamentalsService
        from data.news import NewsService

        fundamentals = FundamentalsService(clock, nse, cache_dir, universe, events=events)
        news = NewsService(clock, cache_dir, universe, nse=nse)
        from features.sentiment import FinBERTScorer

        sentiment = FinBERTScorer(settings.models.finbert_model, settings.models.finbert_device)

    from features.builder import FeatureBuilder

    features = FeatureBuilder(data, calendar)

    from agents.llm import build_llm
    from agents.prompt_store import PromptStore

    llm = build_llm(settings)
    prompts = PromptStore(session_factory, clock=clock)
    try:
        seeded = prompts.seed()
        if seeded:
            logger.info(f"[bootstrap] seeded {seeded} prompt templates")
    except Exception as exc:  # noqa: BLE001 - prompts fall back to files; never block startup
        logger.error(f"[bootstrap] prompt seeding failed: {exc}")

    notifier = None
    if with_notifier:
        from notifications.service import Notifier

        notifier = Notifier(session_factory, settings, secrets, clock, async_send=not is_replay)

    from approvals.manual import ManualTrades
    from approvals.queue import ApprovalQueue
    from capital.accounts import CapitalManager
    from paper_engine.broker import PaperBroker
    from paper_engine.monitor import PositionMonitor
    from risk.manager import RiskManager
    from shadow.tracker import ShadowTracker

    capital = CapitalManager(settings, clock)
    risk = RiskManager(settings, caps)
    broker = PaperBroker(settings, caps, clock, capital, data, calendar, notifier)
    shadow = ShadowTracker(settings, clock, calendar, data)
    queue = ApprovalQueue(settings, clock, calendar, risk, broker, capital, shadow, data=data, notifier=notifier)

    services_ref: dict[str, Services] = {}

    def _events_window(symbol: str, mode: Any, now: datetime) -> list[MarketEvent]:
        return services_ref["s"].events_for_window(symbol, mode, now)

    manual = ManualTrades(settings, clock, calendar, data, risk, broker, capital, sector_fn=universe.sector_of,
                          events_fn=_events_window)
    monitor = PositionMonitor(settings, clock, calendar, data, broker, capital,
                              events_fn=lambda sym, a, b: events.events_for(sym, a, b), notifier=notifier)

    from agents.graph import AnalysisPipeline, PipelineServices

    pipeline = AnalysisPipeline(PipelineServices(
        settings=settings, caps=caps, clock=clock, session_factory=session_factory, calendar=calendar, data=data,
        features=features, llm=llm, prompts=prompts, risk=risk, capital=capital, broker=broker, queue=queue,
        shadow=shadow, universe=universe, events=events, fundamentals=fundamentals, news=news, sentiment=sentiment,
        nse=nse, notifier=notifier,
    ))
    services = Services(ctx=ctx, calendar=calendar, data=data, universe=universe, nse=nse, events=events,
                        fundamentals=fundamentals, news=news, sentiment=sentiment, features=features, llm=llm,
                        prompts=prompts, notifier=notifier, capital=capital, risk=risk, broker=broker,
                        monitor=monitor, shadow=shadow, queue=queue, manual=manual, pipeline=pipeline)
    services_ref["s"] = services

    from control.batches import ensure_current_batch

    with session_factory() as session:
        batch = ensure_current_batch(session, services, actor="system")
        session.commit()
        ctx.batch_id = batch.id
    if notifier is not None:
        notifier.batch_id = ctx.batch_id
    logger.info(f"[bootstrap] ready: batch {ctx.batch_id}, settings v{version_id}, data={data.providers[0].name}, "
                f"llm={settings.engine.llm_backend}, replay={is_replay}")
    return services
