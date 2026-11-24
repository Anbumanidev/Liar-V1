"""Test doubles for the paper engine: a clock-aware fake market data service."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd

from core.clock import IST, Clock
from core.enums import TIMEFRAME_MINUTES, Timeframe
from core.types import Quote, empty_ohlcv


def bars(day: date, start: str, rows: list[tuple[float, float, float, float]], freq_min: int = 1,
         volume: float = 1000.0) -> pd.DataFrame:
    """Consecutive bars from ``start`` (HH:MM) on ``day``: rows of (open, high, low, close)."""
    hh, mm = (int(x) for x in start.split(":"))
    t0 = datetime.combine(day, time(hh, mm), tzinfo=IST)
    idx = [t0 + timedelta(minutes=i * freq_min) for i in range(len(rows))]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.DatetimeIndex(idx, name="ts"))
    df["volume"] = volume
    return df.astype("float64")


class FakeData:
    """Serves only CLOSED bars as of ``clock.now()`` (like MarketDataService)."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.frames: dict[tuple[str, Timeframe], pd.DataFrame] = {}

    def add(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> None:
        key = (symbol, Timeframe(timeframe))
        cur = self.frames.get(key)
        self.frames[key] = df if cur is None else pd.concat([cur, df]).sort_index()

    def _closed(self, symbol: str, tf: Timeframe) -> pd.DataFrame:
        df = self.frames.get((symbol, tf))
        if df is None or df.empty:
            return empty_ohlcv()
        now = self.clock.now()
        if tf in (Timeframe.D1, Timeframe.W1):
            close_ts = [datetime.combine(ts.date(), time(15, 30), tzinfo=IST) for ts in df.index]
        else:
            close_ts = [ts + timedelta(minutes=TIMEFRAME_MINUTES[tf]) for ts in df.index]
        mask = [c <= now for c in close_ts]
        return df[mask]

    def candles(self, symbol: str, timeframe: Any, lookback_bars: int = 300, start: datetime | None = None) -> pd.DataFrame:
        tf = Timeframe(timeframe)
        df = self._closed(symbol, tf)
        if start is not None:
            return df[df.index >= start]
        return df.tail(lookback_bars)

    def quote(self, symbol: str) -> Quote | None:
        df = self._closed(symbol, Timeframe.M1)
        if df.empty:
            return None
        ts = df.index[-1] + timedelta(minutes=1)
        return Quote(symbol=symbol, ltp=float(df["close"].iloc[-1]), timestamp=ts.to_pydatetime(), source="fake")
