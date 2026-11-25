"""Light, read-mostly process context for the web dashboard.

The web process must NOT call ``control.bootstrap.bootstrap()`` (it loads LLM/FinBERT and writes
batches). It only needs: a session factory on the shared DB, the hard caps, a clock, the
active settings (cached per settings version), the NSE calendar and - lazily, on first chart or
quote request - a ``MarketDataService`` for candles/quotes.

Everything is injectable so tests can pass a temp DB, a fake data service and a simulated clock.
"""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from config.schema import HardCaps, Settings
from core.clock import Clock, SystemClock
from storage.models import SettingsVersion

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================================== quote cache


class QuoteCache:
    """~10 s in-process quote cache with single-flight background refresh and a time budget.

    ``get_many`` never blocks a request for longer than ``wait`` seconds: missing/stale symbols
    are fetched on a worker thread and whatever is cached when the budget runs out is returned
    (callers fall back to ``Position.last_price``).
    """

    def __init__(self, data_getter: Callable[[], Any], ttl: float = 10.0, max_workers: int = 2) -> None:
        self._data_getter = data_getter
        self.ttl = ttl
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, Any]] = {}
        self._inflight: dict[frozenset[str], Future[None]] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="webui-quotes")

    def _fetch(self, symbols: list[str]) -> None:
        try:
            data = self._data_getter()
        except Exception as exc:  # noqa: BLE001 - quotes are best effort
            logger.warning(f"[webui] market data unavailable: {exc}")
            return
        if data is None:
            return
        got: dict[str, Any] = {}
        try:
            if hasattr(data, "quotes"):
                got = dict(data.quotes(symbols) or {})
            else:
                for s in symbols:
                    q = data.quote(s)
                    if q is not None:
                        got[s] = q
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[webui] quote fetch failed for {symbols[:5]}: {exc}")
        now = _time.monotonic()
        with self._lock:
            for s in symbols:
                q = got.get(s) or got.get(s.upper())
                if q is not None:
                    self._cache[s] = (now, q)

    def get_many(self, symbols: Iterable[str], wait: float = 2.5) -> dict[str, Any]:
        wanted = sorted({s.strip().upper() for s in symbols if s and s.strip()})
        if not wanted:
            return {}
        now = _time.monotonic()
        with self._lock:
            stale = [s for s in wanted if s not in self._cache or now - self._cache[s][0] >= self.ttl]
            fut: Future[None] | None = None
            if stale:
                key = frozenset(stale)
                fut = self._inflight.get(key)
                if fut is None or fut.done():
                    fut = self._pool.submit(self._fetch, stale)
                    self._inflight[key] = fut
                    fut.add_done_callback(lambda _f, k=key: self._drop_inflight(k))
        if fut is not None and wait > 0:
            try:
                fut.result(timeout=wait)
            except Exception:  # noqa: BLE001 - timeout or fetch error: use what we have
                pass
        with self._lock:
            return {s: self._cache[s][1] for s in wanted if s in self._cache}

    def _drop_inflight(self, key: frozenset[str]) -> None:
        with self._lock:
            self._inflight.pop(key, None)

    def age(self, symbol: str) -> float | None:
        with self._lock:
            hit = self._cache.get(symbol.upper())
        return None if hit is None else _time.monotonic() - hit[0]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# =========================================================================== context


@dataclass
class WebContext:
    """Everything the web API needs (all injectable for tests).

    Attributes:
        session_factory: sessionmaker bound to the shared trading DB (SQLite WAL).
        caps: immutable hard caps (``allow_full_auto`` gates FULL_AUTO / HYBRID).
        clock: IST clock (``SystemClock`` live, ``SimulatedClock`` in tests).
        log_dir: root of ``logs/YYYY-MM-DD/`` (engine.log + JSONL categories).
        auth_file: owner credentials YAML (``scripts/set_password.py``).
        secret_key_path: cookie signing key, generated once (``runtime/web_secret.key``).
        runtime_dir: writable runtime dir (cache dir for the calendar / candles).
        data: injected market data service (tests); otherwise built lazily by ``data_factory``.
        data_factory: ``(settings, calendar) -> MarketDataService``.
        settings_provider: ``(session) -> (version_id, Settings)``; default = config.store.
        calendar: injected ``NSECalendar``; otherwise loaded lazily (static file, no network).
        allowed_hosts: Host header allow-list (DNS-rebinding protection).
        streamlit_url: link to the advanced Streamlit console.
        stream_interval / stream_heartbeat / stream_max_iterations: SSE loop tuning (tests use a
            short interval and a bounded number of iterations).
    """

    session_factory: sessionmaker[Session]
    caps: HardCaps
    clock: Clock = field(default_factory=SystemClock)
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    auth_file: Path = field(default_factory=lambda: PROJECT_ROOT / "secrets" / "dashboard_auth.yaml")
    secret_key_path: Path = field(default_factory=lambda: PROJECT_ROOT / "runtime" / "web_secret.key")
    runtime_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runtime")
    data: Any | None = None
    data_factory: Callable[[Settings, Any], Any] | None = None
    settings_provider: Callable[[Session], tuple[int, Settings]] | None = None
    calendar: Any | None = None
    allowed_hosts: list[str] = field(default_factory=lambda: ["127.0.0.1", "localhost", "::1"])
    streamlit_url: str = "http://127.0.0.1:8501"
    secure_cookies: bool = False
    quote_ttl: float = 10.0
    quote_wait: float = 2.5
    candle_ttl: float = 20.0
    stream_interval: float = 2.0
    stream_heartbeat: float = 15.0
    stream_max_iterations: int | None = None

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _settings_cache: tuple[int, Settings] | None = field(default=None, init=False, repr=False)
    _settings_checked: float = field(default=0.0, init=False, repr=False)
    _data_failed_at: float | None = field(default=None, init=False, repr=False)
    _quotes: QuoteCache | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ db
    def session(self) -> Session:
        return self.session_factory()

    # ------------------------------------------------------------------ settings
    def settings(self, session: Session | None = None) -> Settings:
        return self.settings_with_version(session)[1]

    def settings_with_version(self, session: Session | None = None) -> tuple[int, Settings]:
        """Active settings, cached per version id (re-checked at most every 2 s)."""
        now = _time.monotonic()
        with self._lock:
            cached = self._settings_cache
            if cached is not None and now - self._settings_checked < 2.0:
                return cached
        own = session is None
        s = session or self.session_factory()
        try:
            if self.settings_provider is not None:
                result = self.settings_provider(s)
            else:
                latest = s.execute(select(SettingsVersion.id).order_by(SettingsVersion.id.desc()).limit(1)
                                   ).scalar_one_or_none()
                if cached is not None and latest == cached[0]:
                    result = cached
                else:
                    from config.store import get_active_settings

                    result = get_active_settings(s, self.caps, clock=self.clock)
                    if latest is None:  # seeded version 1 on an empty DB
                        s.commit()
        finally:
            if own:
                s.close()
        with self._lock:
            self._settings_cache = result
            self._settings_checked = now
        return result

    def invalidate_settings(self) -> None:
        with self._lock:
            self._settings_cache = None
            self._settings_checked = 0.0

    # ------------------------------------------------------------------ calendar / data
    def get_calendar(self) -> Any:
        with self._lock:
            if self.calendar is not None:
                return self.calendar
        from data.calendar import load_calendar

        try:
            settings = self.settings()
            schedule = settings.schedule
        except Exception as exc:  # noqa: BLE001 - weekends + static holidays still work
            logger.warning(f"[webui] settings unavailable for calendar: {exc}")
            schedule = None
        cal = load_calendar(schedule, self.runtime_dir / "cache", fetch=False)
        with self._lock:
            self.calendar = cal
        return cal

    def get_data(self) -> Any | None:
        """The market data service (built lazily; None if it cannot be built - retried after 60 s)."""
        with self._lock:
            if self.data is not None:
                return self.data
            if self.data_factory is None:
                return None
            if self._data_failed_at is not None and _time.monotonic() - self._data_failed_at < 60:
                return None
        try:
            built = self.data_factory(self.settings(), self.get_calendar())
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[webui] could not build market data service: {exc}")
            with self._lock:
                self._data_failed_at = _time.monotonic()
            return None
        with self._lock:
            if self.data is None:
                self.data = built
            return self.data

    @property
    def quotes(self) -> QuoteCache:
        with self._lock:
            if self._quotes is None:
                self._quotes = QuoteCache(self.get_data, ttl=self.quote_ttl)
            return self._quotes

    def close(self) -> None:
        if self._quotes is not None:
            self._quotes.shutdown()


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_context(*, host: str = "127.0.0.1", port: int = 8600, db_url: str | None = None) -> WebContext:
    """Production context: secrets, DB (tables created idempotently), caps, TLS bundle, lazy data.

    Mirrors the light parts of ``control.bootstrap`` (secrets, DB, settings, SSL, calendar,
    market data service) without LLMs, notifier, engine services or batch creation.
    """
    from config.loader import load_hard_caps
    from config.secrets import load_secrets
    from core.guard import startup_safety_check
    from core.net import configure_ssl
    from storage.db import create_all_for_url

    secrets = load_secrets()
    caps = load_hard_caps()
    startup_safety_check(caps)  # paper-only assertion + order guards (defence in depth)
    runtime_dir = _resolve(secrets.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        configure_ssl(runtime_dir, use_os_store=True, ca_bundle=secrets.ca_bundle_path)
    except Exception as exc:  # noqa: BLE001 - never block startup on TLS bundle building
        logger.warning(f"[webui] TLS bundle setup failed: {exc}")
    factory = create_all_for_url(db_url or secrets.database_url)
    clock = SystemClock()

    def _data_factory(settings: Settings, calendar: Any) -> Any:
        from data.service import build_market_data_service

        return build_market_data_service(settings, secrets, clock, calendar, runtime_dir / "cache",
                                         session_factory=None)

    allowed = ["127.0.0.1", "localhost", "::1"]
    if host not in ("0.0.0.0", "::", "") and host not in allowed:
        allowed.append(host)
    st_host = secrets.dashboard_host or "127.0.0.1"
    if st_host in ("0.0.0.0", "::"):
        st_host = "127.0.0.1"
    return WebContext(
        session_factory=factory,
        caps=caps,
        clock=clock,
        log_dir=_resolve(secrets.log_dir),
        auth_file=_resolve(secrets.dashboard_auth_file),
        secret_key_path=runtime_dir / "web_secret.key",
        runtime_dir=runtime_dir,
        data_factory=_data_factory,
        allowed_hosts=allowed,
        streamlit_url=f"http://{st_host}:{secrets.dashboard_port or 8501}",
    )
