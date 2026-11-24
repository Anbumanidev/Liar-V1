"""MarketDataService: the single entry point every module uses for candles and quotes.

Guarantees
----------
* **Closed bars only** - ``candles()`` drops every bar whose close time is after
  ``clock.now()`` (``data.calendar.closed_bars``), whatever the provider returned.
* **Fallback** - providers are tried in order ``[provider, *fallbacks]``; the first non-empty
  result wins. Total failure returns an empty frame, sets ``last_error`` and records an error
  (``storage.audit.record_error``) - it never raises.
* **Resampling** - a timeframe a provider does not serve natively (e.g. yfinance weekly) is
  built from the finest suitable timeframe it does serve (``data.bars.RESAMPLE_SOURCES``),
  session-aligned at 09:15.
* **Caching** - in-memory per ``(symbol, timeframe)``; an entry stays valid until the next
  bar boundary (when a new bar can close) or a max TTL, whichever is first. Daily history is
  also cached on disk per closed session (``<cache_dir>/candles/``) to survive restarts.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from core.clock import Clock, to_ist
from core.enums import TIMEFRAME_MINUTES, Timeframe
from core.errors import DataUnavailableError
from core.types import Quote, empty_ohlcv
from data.bars import RESAMPLE_SOURCES, resample_ohlcv
from data.base import DataProvider
from data.calendar import INTRADAY_TIMEFRAMES, NSECalendar, as_timeframe, bar_close_time, closed_bars

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm import Session, sessionmaker

    from config.schema import Settings
    from config.secrets import Secrets

#: Upper bound on how long a candle cache entry may live, per timeframe.
MAX_CACHE_TTL: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.H1: timedelta(minutes=60),
    Timeframe.D1: timedelta(hours=6),
    Timeframe.W1: timedelta(hours=6),
}
#: Retry delay when a just-closed intraday bar is not published by the vendor yet.
STALE_RETRY = timedelta(seconds=15)


@dataclass
class _CacheEntry:
    df: pd.DataFrame
    start: datetime
    valid_until: datetime
    source: str


class MarketDataService:
    """Candles/quotes with closed-bar filtering, caching, fallback and resampling.

    Args:
        provider: primary ``DataProvider``.
        clock: time source (``SimulatedClock`` in replay).
        calendar: ``NSECalendar``.
        fallbacks: further providers tried in order when the primary fails.
        cache_dir: optional disk cache root (``runtime/cache``).
        session_factory: optional SQLAlchemy sessionmaker for ``errors`` rows.
        quote_ttl_seconds: in-memory quote cache lifetime (clock seconds).
    """

    def __init__(
        self,
        provider: DataProvider,
        clock: Clock,
        calendar: NSECalendar,
        fallbacks: Sequence[DataProvider] = (),
        cache_dir: Path | None = None,
        *,
        session_factory: "sessionmaker[Session] | None" = None,
        quote_ttl_seconds: float = 5.0,
    ) -> None:
        self.provider = provider
        self.clock = clock
        self.calendar = calendar
        self.fallbacks: tuple[DataProvider, ...] = tuple(p for p in fallbacks if p is not provider)
        self.cache_dir = cache_dir
        self._session_factory = session_factory
        self._quote_ttl = timedelta(seconds=quote_ttl_seconds)
        self._cache: dict[tuple[str, Timeframe], _CacheEntry] = {}
        self._quotes: dict[str, tuple[datetime, Quote]] = {}
        self._lock = threading.RLock()
        self._last_error: str | None = None
        self._consecutive_failures = 0
        self.last_source: dict[tuple[str, Timeframe], str] = {}

    # ------------------------------------------------------------------ status
    @property
    def last_error(self) -> str | None:
        """Message of the most recent total failure (None once a later call succeeds)."""
        return self._last_error

    @property
    def providers(self) -> tuple[DataProvider, ...]:
        """Primary followed by fallbacks."""
        return (self.provider, *self.fallbacks)

    def healthy(self) -> bool:
        """False after a total data failure until the next successful call."""
        return self._consecutive_failures == 0

    def check_health(self) -> dict[str, bool]:
        """Run every provider's (possibly networked) healthcheck."""
        out: dict[str, bool] = {}
        for p in self.providers:
            try:
                out[p.name] = bool(p.healthcheck())
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[data] healthcheck of {p.name} raised: {exc}")
                out[p.name] = False
        return out

    def clear_cache(self) -> None:
        """Drop all in-memory candle/quote caches."""
        with self._lock:
            self._cache.clear()
            self._quotes.clear()

    # ------------------------------------------------------------------ candles
    def lookback_start(self, timeframe: Timeframe | str, lookback_bars: int, now: datetime | None = None) -> datetime:
        """Start datetime that yields at least ``lookback_bars`` closed bars (plus margin)."""
        tf = as_timeframe(timeframe)
        now = to_ist(now or self.clock.now())
        today = now.date()
        bars = max(int(lookback_bars), 1)
        if tf in INTRADAY_TIMEFRAMES:
            per_day = math.ceil(375 / TIMEFRAME_MINUTES[tf])
            days = math.ceil(bars / per_day) + 1
            return self.calendar.market_open_dt(self.calendar.add_trading_days(today, -days))
        if tf == Timeframe.D1:
            return self.calendar.market_open_dt(self.calendar.add_trading_days(today, -(bars + 2)))
        return self.calendar.market_open_dt(today - timedelta(weeks=bars + 2))

    def candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        lookback_bars: int = 300,
        start: datetime | None = None,
    ) -> pd.DataFrame:
        """CLOSED candles of ``symbol`` as of ``clock.now()``.

        Args:
            symbol: plain NSE ticker or index/global pseudo-symbol.
            timeframe: ``Timeframe`` or its string value.
            lookback_bars: number of most-recent closed bars to return (ignored if ``start``).
            start: return every closed bar from ``start`` instead.

        Returns:
            Contract-compliant frame (possibly empty on total failure - never raises).
        """
        tf = as_timeframe(timeframe)
        sym = symbol.strip().upper()
        now = to_ist(self.clock.now())
        req_start = to_ist(start) if start is not None else self.lookback_start(tf, lookback_bars, now)
        try:
            raw = self._get_raw(sym, tf, req_start, now)
        except DataUnavailableError as exc:
            self._fail(f"candles {sym} {tf.value}: {exc}", {"symbol": sym, "timeframe": tf.value})
            return empty_ohlcv()
        out = closed_bars(raw, tf, now, self.calendar)
        lo = pd.Timestamp(req_start.replace(hour=0, minute=0)) if tf in (Timeframe.D1, Timeframe.W1) \
            else pd.Timestamp(req_start)
        out = out.loc[out.index >= lo]
        if start is None:
            out = out.tail(lookback_bars)
        self._ok()
        return out.copy()

    def _get_raw(self, symbol: str, tf: Timeframe, start: datetime, now: datetime) -> pd.DataFrame:
        """All bars (closed or forming) from ``start``, via cache / providers / resampling."""
        key = (symbol, tf)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and entry.start <= start and now < entry.valid_until:
                return entry.df
        errors: list[str] = []
        for provider in self.providers:
            try:
                df = self._fetch_from(provider, symbol, tf, start, now)
            except Exception as exc:  # noqa: BLE001 - try the next provider
                errors.append(f"{provider.name}: {exc}")
                logger.debug(f"[data] {provider.name} failed for {symbol} {tf.value}: {exc}")
                continue
            if df is None or df.empty:
                errors.append(f"{provider.name}: empty")
                continue
            with self._lock:
                self._cache[key] = _CacheEntry(df, start, self._valid_until(tf, df, now), provider.name)
                self.last_source[key] = provider.name
            if provider is not self.provider:
                logger.info(f"[data] {symbol} {tf.value} served by fallback {provider.name}")
            return df
        raise DataUnavailableError("; ".join(errors) or "no providers")

    def _fetch_from(
        self, provider: DataProvider, symbol: str, tf: Timeframe, start: datetime, now: datetime
    ) -> pd.DataFrame:
        if provider.supports(tf):
            if tf == Timeframe.D1:
                return self._daily_with_disk_cache(provider, symbol, start, now)
            return provider.get_candles(symbol, tf, start, now)
        for src_tf in RESAMPLE_SOURCES[tf]:
            if provider.supports(src_tf):
                src_start = start
                if tf == Timeframe.W1:  # include the whole first week
                    src_start = start - timedelta(days=start.weekday())
                if src_tf == Timeframe.D1:
                    src = self._daily_with_disk_cache(provider, symbol, src_start, now)
                else:
                    src = provider.get_candles(symbol, src_tf, src_start, now)
                return resample_ohlcv(src, tf)
        raise DataUnavailableError(f"{provider.name} cannot serve or resample {tf.value}")

    # ------------------------------------------------------------------ cache helpers
    def _next_boundary(self, tf: Timeframe, t: datetime) -> datetime:
        """First instant after ``t`` at which a new ``tf`` bar can close."""
        cal = self.calendar
        day = t.date()
        if tf in INTRADAY_TIMEFRAMES:
            step = timedelta(minutes=TIMEFRAME_MINUTES[tf])
            if cal.is_trading_day(day) and t < cal.market_close_dt(day):
                open_dt = cal.market_open_dt(day)
                if t < open_dt:
                    return open_dt + step
                k = int((t - open_dt) // step) + 1
                return min(open_dt + k * step, cal.market_close_dt(day))
            nd = cal.next_trading_day(day)
            return min(cal.market_open_dt(nd) + step, cal.market_close_dt(nd))
        if cal.is_trading_day(day) and t < cal.market_close_dt(day):
            return cal.market_close_dt(day)
        return cal.market_close_dt(cal.next_trading_day(day))

    def _valid_until(self, tf: Timeframe, df: pd.DataFrame, now: datetime) -> datetime:
        until = min(now + MAX_CACHE_TTL[tf], self._next_boundary(tf, now))
        if tf in INTRADAY_TIMEFRAMES and self.calendar.is_market_open(now):
            closed = closed_bars(df, tf, now, self.calendar)
            step = timedelta(minutes=TIMEFRAME_MINUTES[tf])
            open_dt = self.calendar.market_open_dt(now.date())
            if now >= open_dt + step:  # at least one bar should have closed today
                expected = open_dt + int((now - open_dt) // step) * step
                last_close = bar_close_time(closed.index[-1], tf, self.calendar) if not closed.empty else None
                if last_close is None or last_close < expected:
                    until = min(until, now + STALE_RETRY)
        return until

    def _disk_dir(self, provider: DataProvider) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / "candles" / provider.name

    @staticmethod
    def _safe_name(symbol: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in symbol)

    def _daily_with_disk_cache(
        self, provider: DataProvider, symbol: str, start: datetime, now: datetime
    ) -> pd.DataFrame:
        """Daily candles; history up to the last closed session is cached on disk.

        Files: ``<cache_dir>/candles/<provider>/<SYMBOL>_1d_<session>_<start>.parquet`` - reused
        when ``session`` is the last closed session at ``now`` and ``start`` <= the request.
        """
        folder = self._disk_dir(provider)
        if folder is None or provider.name == "replay":
            return provider.get_candles(symbol, Timeframe.D1, start, now)
        session_day = self.calendar.last_closed_session(now)
        prefix = f"{self._safe_name(symbol)}_1d_{session_day:%Y%m%d}_"
        wanted = to_ist(start).date()
        if folder.exists():
            for path in sorted(folder.glob(f"{prefix}*.parquet")):
                try:
                    cached_start = datetime.strptime(path.stem[len(prefix):], "%Y%m%d").date()
                    if cached_start > wanted:
                        continue
                    cached = pd.read_parquet(path)
                    cached.index = pd.DatetimeIndex(cached.index).tz_convert("Asia/Kolkata")
                    cached.index.name = "ts"
                    return cached
                except Exception as exc:  # noqa: BLE001 - corrupt cache: ignore and refetch
                    logger.warning(f"[data] daily cache unreadable {path}: {exc}")
        df = provider.get_candles(symbol, Timeframe.D1, start, now)
        if not df.empty:
            path = folder / f"{prefix}{wanted:%Y%m%d}.parquet"
            try:
                folder.mkdir(parents=True, exist_ok=True)
                closed_bars(df, Timeframe.D1, now, self.calendar).to_parquet(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[data] could not write daily cache {path}: {exc}")
        return df

    # ------------------------------------------------------------------ quotes
    def quote(self, symbol: str) -> Quote | None:
        """Latest quote (primary, then fallbacks). Quotes stamped in the future are rejected."""
        sym = symbol.strip().upper()
        now = to_ist(self.clock.now())
        with self._lock:
            hit = self._quotes.get(sym)
            if hit is not None and timedelta(0) <= now - hit[0] < self._quote_ttl:
                return hit[1]
        for provider in self.providers:
            try:
                q = provider.get_quote(sym)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[data] quote via {provider.name} failed for {sym}: {exc}")
                continue
            if q is None:
                continue
            if to_ist(q.timestamp) > now:
                logger.warning(f"[data] {provider.name} quote for {sym} is in the future ({q.timestamp}); ignored")
                continue
            with self._lock:
                self._quotes[sym] = (now, q)
            self._ok()
            return q
        self._fail(f"quote {sym}: no provider returned a quote", {"symbol": sym})
        return None

    def quotes(self, symbols: Iterable[str]) -> dict[str, Quote]:
        """Quotes for several symbols; missing ones are retried on fallbacks."""
        wanted = [s.strip().upper() for s in symbols]
        now = to_ist(self.clock.now())
        out: dict[str, Quote] = {}
        with self._lock:
            for s in wanted:
                hit = self._quotes.get(s)
                if hit is not None and timedelta(0) <= now - hit[0] < self._quote_ttl:
                    out[s] = hit[1]
        for provider in self.providers:
            missing = [s for s in wanted if s not in out]
            if not missing:
                break
            try:
                got = provider.get_quotes(missing)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[data] bulk quotes via {provider.name} failed: {exc}")
                continue
            for s, q in got.items():
                if q is not None and to_ist(q.timestamp) <= now:
                    out[s.upper()] = q
                    with self._lock:
                        self._quotes[s.upper()] = (now, q)
        missing = [s for s in wanted if s not in out]
        if missing and len(missing) == len(wanted):
            self._fail(f"quotes: none of {len(wanted)} symbols available", {"symbols": missing[:20]})
        elif out:
            self._ok()
        return out

    # ------------------------------------------------------------------ error bookkeeping
    def _ok(self) -> None:
        self._consecutive_failures = 0
        self._last_error = None

    def _fail(self, message: str, context: dict[str, Any]) -> None:
        self._consecutive_failures += 1
        self._last_error = message
        session = None
        try:
            if self._session_factory is not None:
                session = self._session_factory()
            from storage.audit import record_error

            record_error(session, "data", message, context, severity="WARNING", clock=self.clock)
            if session is not None:
                session.commit()
        except Exception as exc:  # noqa: BLE001 - error reporting must never break data access
            logger.error(f"[data] could not record data error: {exc}")
        finally:
            if session is not None:
                session.close()


# =========================================================================== factories


def build_provider(
    name: str,
    secrets: "Secrets | None" = None,
    *,
    clock: Clock | None = None,
    cache_dir: Path | None = None,
) -> DataProvider:
    """Construct a live provider by name: ``yfinance`` | ``angelone`` | ``upstox``.

    ``replay`` needs preloaded frames - use ``ReplayDataProvider.from_provider`` instead.
    """
    key = name.strip().lower()
    if key == "yfinance":
        from data.providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider(clock=None)  # Yahoo limits are relative to wall time
    if key == "angelone":
        from data.providers.angelone_provider import AngelOneProvider

        return AngelOneProvider(secrets, cache_dir=cache_dir, clock=clock)
    if key == "upstox":
        from data.providers.upstox_provider import UpstoxProvider

        return UpstoxProvider(secrets, cache_dir=cache_dir, clock=clock)
    if key == "replay":
        raise ValueError("replay provider needs preloaded frames: use ReplayDataProvider.from_provider(...)")
    raise ValueError(f"unknown data provider {name!r}")


def build_market_data_service(
    settings: "Settings",
    secrets: "Secrets | None",
    clock: Clock,
    calendar: NSECalendar,
    cache_dir: Path | None = None,
    *,
    session_factory: "sessionmaker[Session] | None" = None,
    primary: DataProvider | None = None,
) -> MarketDataService:
    """Service from ``settings.engine.data_provider`` + ``fallback_providers``.

    Pass ``primary`` (e.g. a ``ReplayDataProvider``) to override the configured primary.
    """
    if primary is None:
        primary = build_provider(settings.engine.data_provider, secrets, clock=clock, cache_dir=cache_dir)
    fallbacks: list[DataProvider] = []
    if primary.name != "replay":
        for name in settings.engine.fallback_providers:
            if name == primary.name or name == "replay" or any(f.name == name for f in fallbacks):
                continue
            try:
                fallbacks.append(build_provider(name, secrets, clock=clock, cache_dir=cache_dir))
            except ValueError as exc:
                logger.warning(f"[data] skipping fallback provider {name}: {exc}")
    return MarketDataService(primary, clock, calendar, fallbacks, cache_dir, session_factory=session_factory)
