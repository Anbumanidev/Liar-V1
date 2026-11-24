"""Shared test helpers: deterministic synthetic OHLCV generators.

Candle contract (see ``core.types``): tz-aware IST DatetimeIndex of bar OPEN times,
columns open/high/low/close/volume. Intraday bars span 09:15-15:30. Daily bars are
indexed at 09:15 IST of their date; weekly bars at 09:15 IST of the week's first day.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from core.clock import IST


def trading_days(start: date, n: int) -> list[date]:
    """``n`` consecutive weekdays starting at ``start`` (ignores exchange holidays)."""
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _walk(n: int, start_price: float, seed: int, drift: float, vol: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    return start_price * np.exp(np.cumsum(rets))


def _ohlc_from_closes(closes: np.ndarray, seed: int, vol: float, base_volume: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    wick = np.abs(rng.normal(0, vol / 2, size=len(closes))) * closes
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick
    volume = base_volume * (1 + np.abs(rng.normal(0, 0.5, size=len(closes))))
    return {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume}


def make_intraday_candles(
    days: list[date],
    freq_minutes: int = 5,
    start_price: float = 100.0,
    seed: int = 0,
    drift: float = 0.0,
    vol: float = 0.002,
    base_volume: float = 10_000.0,
) -> pd.DataFrame:
    """Intraday candles for each date in ``days`` (09:15 to 15:30, bar open times)."""
    index: list[datetime] = []
    for d in days:
        t = datetime.combine(d, time(9, 15), tzinfo=IST)
        end = datetime.combine(d, time(15, 30), tzinfo=IST)
        while t < end:
            index.append(t)
            t += timedelta(minutes=freq_minutes)
    closes = _walk(len(index), start_price, seed, drift, vol)
    data = _ohlc_from_closes(closes, seed, vol, base_volume)
    df = pd.DataFrame(data, index=pd.DatetimeIndex(index, name="ts"))
    return df.astype("float64")


def make_daily_candles(
    end: date,
    n: int = 300,
    start_price: float = 100.0,
    seed: int = 0,
    drift: float = 0.0005,
    vol: float = 0.015,
    base_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """``n`` daily candles ending on ``end`` (weekdays), indexed at 09:15 IST."""
    days: list[date] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    index = [datetime.combine(x, time(9, 15), tzinfo=IST) for x in days]
    closes = _walk(n, start_price, seed, drift, vol)
    data = _ohlc_from_closes(closes, seed, vol, base_volume)
    return pd.DataFrame(data, index=pd.DatetimeIndex(index, name="ts")).astype("float64")


def ist(y: int, m: int, d: int, hh: int = 9, mm: int = 15) -> datetime:
    """Shorthand for an IST datetime."""
    return datetime(y, m, d, hh, mm, tzinfo=IST)
