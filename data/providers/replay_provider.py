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
from datetime import datetime

import pandas as pd
from loguru import logger

from core.clock import Clock, to_ist
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
        for sym in symbols:
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
                frames[(sym.upper(), tf)] = df if fetch_tf == tf else resample_ohlcv(df, tf)
        logger.info(f"[replay] preloaded {len(frames)} frames for {len(set(s for s, _ in frames))} symbols")
        return cls(frames, clock, calendar)

    # ------------------------------------------------------------------ DataProvider
    @property
    def supported_timeframes(self) -> frozenset[Timeframe]:  # type: ignore[override]
        """Every timeframe that is preloaded or can be resampled from a preloaded one."""
        have = {tf for _, tf in self._frames}
        out = set(have)
        for tf, sources in RESAMPLE_SOURCES.items():
            if any(s in have for s in sources):
                out.add(tf)
        return frozenset(out)

    def _visible(self, symbol: str, tf: Timeframe, now: datetime) -> pd.DataFrame | None:
        key = (symbol.upper(), tf)
        with self._lock:
            df = self._frames.get(key)
            close = self._close_times.get(key)
        if df is None or close is None:
            return None
        return df.loc[(close <= pd.Timestamp(now))]

    def get_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        tf = Timeframe(timeframe)
        now = to_ist(self._clock.now())
        end = min(to_ist(end), now)
        start = to_ist(start)
        df = self._visible(symbol, tf, now)
        if df is None:
            source_tf = next((s for s in RESAMPLE_SOURCES[tf] if (symbol.upper(), s) in self._frames), None)
            if source_tf is None:
                raise DataUnavailableError(f"replay has no {tf.value} (or finer) data for {symbol}")
            src = self._visible(symbol, source_tf, now)
            df = resample_ohlcv(src, tf) if src is not None and not src.empty else empty_ohlcv()
            df = closed_bars(df, tf, now, self._calendar)
        if df.empty:
            return empty_ohlcv()
        lo = pd.Timestamp(start.replace(hour=0, minute=0)) if tf in (Timeframe.D1, Timeframe.W1) else pd.Timestamp(start)
        return df.loc[(df.index >= lo) & (df.index <= pd.Timestamp(end))].copy()

    def get_quote(self, symbol: str) -> Quote | None:
        now = to_ist(self._clock.now())
        for tf in self.timeframes_for(symbol):
            df = self._visible(symbol, tf, now)
            if df is None or df.empty:
                continue
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
        logger.debug(f"[replay] no closed bar for {symbol} at {now}")
        return None

    def healthcheck(self) -> bool:
        return bool(self._frames)
