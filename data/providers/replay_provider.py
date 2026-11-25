"""REPLAY data provider: serves preloaded history strictly as of ``clock.now()`` (section 16).

No look-ahead guarantees (tested exhaustively):

* ``get_candles`` NEVER returns a bar whose close time (``data.calendar.bar_close_time``) is
  after ``clock.now()``; the requested ``end`` is clamped to ``now``.
* A timeframe that is not preloaded is resampled from a finer preloaded frame, using ONLY the
  source bars already closed at ``now`` - and the result is filtered again, so a partially
  built 1h/daily/weekly bar is never visible.
* ``get_quote`` is the close of the last CLOSED bar of the finest preloaded timeframe; its
  timestamp is that bar's close time (never a future instant).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import datetime, time, timedelta

import pandas as pd
from loguru import logger

from core.clock import IST, Clock, to_ist
from core.enums import TIMEFRAME_MINUTES, Timeframe
from core.errors import DataUnavailableError
from core.types import Quote, empty_ohlcv
from data.bars import RESAMPLE_SOURCES, normalize_ohlcv, resample_ohlcv
from data.base import DataProvider
from data.calendar import NSECalendar, bar_close_time, bar_close_times, closed_bars


class ReplayDataProvider(DataProvider):
    """In-memory, clock-gated provider for REPLAY / walk-forward runs.

    Args:
        frames: ``{(symbol, Timeframe): candle frame}`` (complete history; normalised here).
        clock: usually a ``SimulatedClock``.
        calendar: ``NSECalendar`` used for bar close times.
    """

    name = "replay"

    def __init__(
        self,
        frames: dict[tuple[str, Timeframe], pd.DataFrame],
        clock: Clock,
        calendar: NSECalendar,
    ) -> None:
        self._clock = clock
        self._calendar = calendar
        self._lock = threading.RLock()
        self._frames: dict[tuple[str, Timeframe], pd.DataFrame] = {}
        self._close_times: dict[tuple[str, Timeframe], pd.DatetimeIndex] = {}
        for (symbol, tf), df in frames.items():
            self.add_frame(symbol, tf, df)

    # ------------------------------------------------------------------ setup
    def add_frame(self, symbol: str, timeframe: Timeframe | str, df: pd.DataFrame) -> None:
        """Add / replace the full history of ``symbol`` on ``timeframe``."""
        tf = Timeframe(timeframe)
        norm = normalize_ohlcv(df, tf)
        key = (symbol.upper(), tf)
        with self._lock:
            self._frames[key] = norm
            self._close_times[key] = bar_close_times(pd.DatetimeIndex(norm.index), tf, self._calendar)

    @property
    def symbols(self) -> set[str]:
        """Symbols with any preloaded data."""
        return {s for s, _ in self._frames}

    def timeframes_for(self, symbol: str) -> list[Timeframe]:
        """Preloaded timeframes of ``symbol``, finest first."""
        tfs = [tf for s, tf in self._frames if s == symbol.upper()]
        return sorted(tfs, key=lambda t: TIMEFRAME_MINUTES[t])

    @classmethod
    def from_provider(
        cls,
        provider: DataProvider,
        symbols: Iterable[str],
        timeframes: Iterable[Timeframe | str],
        start: datetime,
        end: datetime,
        clock: Clock,
        calendar: NSECalendar,
    ) -> "ReplayDataProvider":
        """Preload ``[start, end]`` history from a live provider (failures logged and skipped).

        Weekly frames are resampled from daily data rather than fetched.
        """
        frames: dict[tuple[str, Timeframe], pd.DataFrame] = {}
        tfs = [Timeframe(t) for t in timeframes]
        for sym in dict.fromkeys(s.strip().upper() for s in symbols):
            for tf in tfs:
                fetch_tf = tf
                if not provider.supports(tf):
                    fetch_tf = next((s for s in RESAMPLE_SOURCES[tf] if provider.supports(s)), tf)
                try:
                    df = provider.get_candles(sym, fetch_tf, start, end)
                except Exception as exc:  # noqa: BLE001 - one symbol failing must not abort the preload
                    logger.warning(f"[replay] preload failed for {sym} {tf.value}: {exc}")
                    continue
                if df is None or df.empty:
                    logger.warning(f"[replay] no {tf.value} history for {sym}")
                    continue
                frames[(sym, tf)] = df if fetch_tf == tf else resample_ohlcv(df, tf)
        logger.info(f"[replay] preloaded {len(frames)} frames for {len(set(s for s, _ in frames))} symbols")
        return cls(frames, clock, calendar)

    # ------------------------------------------------------------------ DataProvider
    @property
    def supported_timeframes(self) -> frozenset[Timeframe]:  # type: ignore[override]
        """Every timeframe that is preloaded or can be resampled (possibly in several hops, e.g.
        5m -> 1d -> 1wk) from a preloaded one."""
        out = {tf for _, tf in self._frames}
        changed = True
        while changed:
            changed = False
            for tf, sources in RESAMPLE_SOURCES.items():
                if tf not in out and any(s in out for s in sources):
                    out.add(tf)
                    changed = True
        return frozenset(out)

    def _visible(self, symbol: str, tf: Timeframe, now: datetime) -> pd.DataFrame | None:
        key = (symbol.upper(), tf)
        with self._lock:
            df = self._frames.get(key)
            close = self._close_times.get(key)
        if df is None or close is None:
            return None
        return df.loc[(close <= pd.Timestamp(now))]

    def _closed_frame(
        self, symbol: str, tf: Timeframe, now: datetime, floor: pd.Timestamp | None = None
    ) -> pd.DataFrame | None:
        """All bars of ``(symbol, tf)`` closed at ``now``; built from a finer frame when ``tf`` is not
        preloaded (a directly preloaded source first, else a multi-hop build such as 5m -> 1d -> 1wk).

        Sources are cut to their CLOSED bars before resampling and the result is filtered again, so
        a partially built bar is never visible. ``floor`` (a bucket boundary: start of a day, or the
        Monday of a week) limits the resampling work to the requested window - bars at or after it
        are identical with or without it. ``None`` if nothing can produce ``tf``.
        """
        df = self._visible(symbol, tf, now)
        if df is not None:
            return df
        sources = RESAMPLE_SOURCES[tf]
        for source_tf in sources:
            src = self._visible(symbol, source_tf, now)
            if src is not None:
                return self._resampled(src, tf, now, floor)
        for source_tf in sources:
            src = self._closed_frame(symbol, source_tf, now, floor)
            if src is not None:
                return self._resampled(src, tf, now, floor)
        return None

    def _resampled(self, src: pd.DataFrame, tf: Timeframe, now: datetime, floor: pd.Timestamp | None) -> pd.DataFrame:
        if floor is not None:
            src = src.loc[src.index >= floor]
        if src.empty:
            return empty_ohlcv()
        return closed_bars(resample_ohlcv(src, tf), tf, now, self._calendar)

    @staticmethod
    def _bucket_floor(tf: Timeframe, start: datetime) -> pd.Timestamp:
        """Earliest source time that can belong to a ``tf`` bar opening at/after ``start``'s bucket."""
        day = start.date()
        if tf == Timeframe.W1:
            day -= timedelta(days=day.weekday())
        return pd.Timestamp(datetime.combine(day, time(0, 0), tzinfo=IST))

    def get_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        tf = Timeframe(timeframe)
        now = to_ist(self._clock.now())
        end = min(to_ist(end), now)
        start = to_ist(start)
        df = self._closed_frame(symbol, tf, now, self._bucket_floor(tf, start))
        if df is None:
            raise DataUnavailableError(f"replay has no {tf.value} (or finer) data for {symbol}")
        if df.empty:
            return empty_ohlcv()
        lo = pd.Timestamp(start.replace(hour=0, minute=0)) if tf in (Timeframe.D1, Timeframe.W1) else pd.Timestamp(start)
        return df.loc[(df.index >= lo) & (df.index <= pd.Timestamp(end))].copy()

    def _last_close(self, symbol: str, tf: Timeframe, now: datetime) -> pd.Timestamp | None:
        """Close time of the last bar of ``(symbol, tf)`` closed at ``now`` (None if none)."""
        with self._lock:
            close = self._close_times.get((symbol.upper(), tf))
        if close is None:
            return None
        visible = close[close <= pd.Timestamp(now)]
        return visible.max() if len(visible) else None

    def get_quote(self, symbol: str) -> Quote | None:
        """Close of the most recent CLOSED bar; among timeframes whose last closed bar is equally
        recent the finest wins (a finer frame with a data gap never yields a stale quote)."""
        now = to_ist(self._clock.now())
        best: tuple[pd.Timestamp, Timeframe] | None = None
        for tf in self.timeframes_for(symbol):  # finest first; strict '>' keeps the finest on ties
            last_close = self._last_close(symbol, tf, now)
            if last_close is not None and (best is None or last_close > best[0]):
                best = (last_close, tf)
        df = self._visible(symbol, best[1], now) if best is not None else None
        if best is None or df is None or df.empty:
            logger.debug(f"[replay] no closed bar for {symbol} at {now}")
            return None
        tf = best[1]
        last_ts = df.index[-1]
        last = df.iloc[-1]
        prev = df.loc[df.index.normalize() < pd.Timestamp(last_ts).normalize()]
        today = df.loc[df.index.normalize() == pd.Timestamp(last_ts).normalize()]
        intraday = tf not in (Timeframe.D1, Timeframe.W1)
        return Quote(
            symbol=symbol.upper(),
            ltp=float(last["close"]),
            timestamp=bar_close_time(last_ts.to_pydatetime(), tf, self._calendar),
            open=float(today["open"].iloc[0]) if intraday else float(last["open"]),
            high=float(today["high"].max()) if intraday else float(last["high"]),
            low=float(today["low"].min()) if intraday else float(last["low"]),
            prev_close=float(prev["close"].iloc[-1]) if not prev.empty else None,
            volume=float(today["volume"].sum()) if intraday else float(last["volume"]),
            source=self.name,
        )

    def healthcheck(self) -> bool:
        return bool(self._frames)
