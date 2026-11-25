"""Chart data: timeframes (incl. 10m/20m/30m/5h resampled), EMA/VWAP overlays, trade markers.

Native timeframes come straight from ``MarketDataService.candles`` (CLOSED bars, IST bar-open
index). 10m/20m/30m are built from 5m and 5h from 1h, anchored at the 09:15 IST session open of
each day - a bucket never spans two sessions (the last bucket of a day may be shorter).

Chart ``time`` values are IST wall-clock epoch seconds (UTC epoch + 19 800 s) because
Lightweight Charts renders timestamps as UTC; see ``webui.util.IST_OFFSET_S``.
"""

from __future__ import annotations

import math
import threading
import time as _time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.clock import IST, to_ist
from storage.models import Order, Position, Trade
from webui.queries import exit_label
from webui.util import IST_OFFSET_S, iso, r2


@dataclass(frozen=True)
class TfSpec:
    key: str
    source: str          # native timeframe requested from the data service
    minutes: int | None  # resample bucket size (None = native)
    intraday: bool
    bar_minutes: int     # nominal bar length (for snapping markers)


TIMEFRAMES: dict[str, TfSpec] = {
    "1m": TfSpec("1m", "1m", None, True, 1),
    "5m": TfSpec("5m", "5m", None, True, 5),
    "10m": TfSpec("10m", "5m", 10, True, 10),
    "15m": TfSpec("15m", "15m", None, True, 15),
    "20m": TfSpec("20m", "5m", 20, True, 20),
    "30m": TfSpec("30m", "5m", 30, True, 30),
    "1h": TfSpec("1h", "1h", None, True, 60),
    "5h": TfSpec("5h", "1h", 300, True, 300),
    "1d": TfSpec("1d", "1d", None, False, 375),
    "1wk": TfSpec("1wk", "1wk", None, False, 375 * 5),
}
#: Source bars per regular session (09:15-15:30) for the intraday source timeframes.
_SOURCE_BARS_PER_DAY = {"1m": 375, "5m": 75, "15m": 25, "1h": 7}

_NS_MIN = 60 * 1_000_000_000
_NS_DAY = 1_440 * _NS_MIN
_IST_NS = IST_OFFSET_S * 1_000_000_000
_OPEN_MIN = 9 * 60 + 15
_SESSION_MIN = 375


def tf_spec(tf: str) -> TfSpec:
    key = (tf or "").strip().lower()
    aliases = {"1w": "1wk", "w": "1wk", "d": "1d", "60m": "1h", "300m": "5h"}
    key = aliases.get(key, key)
    if key not in TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {tf!r}; use one of {', '.join(TIMEFRAMES)}")
    return TIMEFRAMES[key]


def resample_session(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample intraday bars into ``minutes`` buckets anchored at 09:15 IST of each session.

    Rows outside 09:15-15:30 are ignored; buckets never cross days; a leading partial bucket
    (source starting mid-bucket) is dropped.
    """
    if df is None or df.empty:
        return df.iloc[0:0] if df is not None else pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    src = df.sort_index()
    o, h, lo, c = (src[col].to_numpy(dtype="float64") for col in ("open", "high", "low", "close"))
    vol = np.nan_to_num(src["volume"].to_numpy(dtype="float64")) if "volume" in src else np.zeros(len(src))
    idx = pd.DatetimeIndex(src.index)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    local = idx.as_unit("ns").asi8 + _IST_NS
    day = local // _NS_DAY * _NS_DAY
    offset = (local - day) // _NS_MIN - _OPEN_MIN
    keep = (offset >= 0) & (offset < _SESSION_MIN) & ~(np.isnan(o) | np.isnan(h) | np.isnan(lo) | np.isnan(c))
    o, h, lo, c, vol, day, offset = (a[keep] for a in (o, h, lo, c, vol, day, offset))
    if len(o) == 0:
        return src.iloc[0:0]
    if offset[0] % minutes != 0:  # first bucket would be partial
        first_key = offset[0] // minutes
        drop = (day == day[0]) & (offset // minutes == first_key)
        o, h, lo, c, vol, day, offset = (a[~drop] for a in (o, h, lo, c, vol, day, offset))
        if len(o) == 0:
            return src.iloc[0:0]
    keys = day + (_OPEN_MIN + offset // minutes * minutes) * _NS_MIN
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)] - 1
    index = pd.to_datetime(keys[starts] - _IST_NS, unit="ns", utc=True).tz_convert(IST)
    out = pd.DataFrame({
        "open": o[starts],
        "high": np.fmax.reduceat(h, starts),
        "low": np.fmin.reduceat(lo, starts),
        "close": c[ends],
        "volume": np.add.reduceat(vol, starts),
    }, index=pd.DatetimeIndex(index, name="ts"))
    return out.astype("float64")


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP reset at each IST session (NaN where volume is missing)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0.0)
    days = pd.DatetimeIndex(df.index).tz_convert(IST).date
    pv = (tp * vol).groupby(days).cumsum()
    cv = vol.groupby(days).cumsum()
    return (pv / cv.replace(0.0, np.nan)).astype("float64")


def _times(index: pd.DatetimeIndex) -> np.ndarray:
    idx = index if index.tz is not None else index.tz_localize(IST)
    return idx.as_unit("ns").asi8 // 1_000_000_000 + IST_OFFSET_S


def _series(times: np.ndarray, values: pd.Series, skip: int = 0) -> list[dict[str, Any]]:
    out = []
    for i, (t, v) in enumerate(zip(times, values.to_numpy(dtype="float64"), strict=False)):
        if i < skip or math.isnan(v):
            continue
        out.append({"time": int(t), "value": round(float(v), 2)})
    return out


class CandleCache:
    """Per (symbol, tf, bars, start, end) payload cache (~20 s) with a small LRU bound."""

    def __init__(self, ttl: float = 20.0, max_entries: int = 96) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._data: OrderedDict[tuple[Any, ...], tuple[float, dict[str, Any]]] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        with self._lock:
            hit = self._data.get(key)
            if hit is None or _time.monotonic() - hit[0] >= self.ttl:
                return None
            self._data.move_to_end(key)
            return hit[1]

    def put(self, key: tuple[Any, ...], value: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = (_time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)


def build_frame(data: Any, symbol: str, spec: TfSpec, bars: int, start: datetime | None,
                end: datetime | None) -> pd.DataFrame:
    """Fetch closed source bars and resample to ``spec``. Never raises on empty data."""
    if spec.minutes is not None:
        per_day = _SOURCE_BARS_PER_DAY.get(spec.source, 75)
        src_minutes = 5 if spec.source == "5m" else 60
        factor = spec.minutes / src_minutes
        src_bars = int(math.ceil(bars * factor)) + per_day
    else:
        src_bars = bars
    if start is not None:
        lead = timedelta(minutes=(spec.minutes or spec.bar_minutes))
        df = data.candles(symbol, spec.source, lookback_bars=src_bars, start=start - lead)
    else:
        df = data.candles(symbol, spec.source, lookback_bars=src_bars)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if spec.minutes is not None:
        df = resample_session(df, spec.minutes)
    if end is not None:
        df = df.loc[pd.DatetimeIndex(df.index) <= pd.Timestamp(to_ist(end))]
    if start is None:
        df = df.tail(bars)
    return df


def frame_payload(df: pd.DataFrame, spec: TfSpec) -> dict[str, Any]:
    if df is None or df.empty:
        return {"candles": [], "volume_max": 0, "ema9": [], "ema21": [], "vwap": [] if spec.intraday else None,
                "first": None, "last": None}
    df = df.astype("float64")
    times = _times(pd.DatetimeIndex(df.index))
    candles = []
    for t, o, h, lo, c, v in zip(times, df["open"], df["high"], df["low"], df["close"], df["volume"], strict=False):
        candles.append({"time": int(t), "open": round(float(o), 2), "high": round(float(h), 2),
                        "low": round(float(lo), 2), "close": round(float(c), 2),
                        "volume": 0 if math.isnan(v) else int(round(float(v)))})
    close = df["close"]
    out = {
        "candles": candles,
        "ema9": _series(times, ema(close, 9), skip=8),
        "ema21": _series(times, ema(close, 21), skip=20),
        "vwap": _series(times, session_vwap(df)) if spec.intraday else None,
        "first": iso(pd.Timestamp(df.index[0]).to_pydatetime()),
        "last": iso(pd.Timestamp(df.index[-1]).to_pydatetime()),
    }
    return out


def _snap(times: list[int], t: int, bar_seconds: int) -> int | None:
    """Bar time containing ``t`` (last bar opening at or before it); None if outside the data."""
    if not times or t < times[0]:
        return None
    i = bisect_right(times, t) - 1
    if i == len(times) - 1 and t > times[-1] + 2 * bar_seconds:
        return None
    return times[i]


def trade_overlays(session: Session, symbol: str, times: list[int], spec: TfSpec,
                   position_id: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Markers (entries/exits) inside the candle range + price lines for open positions."""
    markers: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    bar_s = spec.bar_minutes * 60 if spec.intraday else 86_400 * (7 if spec.key == "1wk" else 1)
    if times:
        lo_t = datetime.fromtimestamp(times[0] - IST_OFFSET_S, tz=IST) - timedelta(days=1)
        stmt = select(Order).where(Order.symbol == symbol, Order.status == "FILLED", Order.filled_at >= lo_t)
        if position_id is not None:
            stmt = stmt.where(Order.position_id == position_id)
        for o in session.execute(stmt).scalars():
            if o.purpose != "ENTRY" or o.filled_at is None:
                continue
            t = _snap(times, int(to_ist(o.filled_at).timestamp()) + IST_OFFSET_S, bar_s)
            if t is None:
                continue
            long_ = o.side == "BUY"
            markers.append({"time": t, "position": "belowBar" if long_ else "aboveBar",
                            "shape": "arrowUp" if long_ else "arrowDown", "color": "buy" if long_ else "sell",
                            "text": f"{'BUY' if long_ else 'SHORT'} {o.qty} @ {o.fill_price:,.2f}",
                            "position_id": o.position_id, "kind": "entry"})
        t_stmt = select(Trade).where(Trade.symbol == symbol, Trade.exit_time >= lo_t)
        if position_id is not None:
            t_stmt = t_stmt.where(Trade.position_id == position_id)
        for tr in session.execute(t_stmt).scalars():
            t = _snap(times, int(to_ist(tr.exit_time).timestamp()) + IST_OFFSET_S, bar_s)
            if t is None:
                continue
            long_ = tr.direction == "LONG"
            label = exit_label(tr.exit_reason, direction=tr.direction, exit_price=tr.exit_price,
                               target_2=tr.target_2, partial=bool(tr.is_partial))
            markers.append({"time": t, "position": "aboveBar" if long_ else "belowBar",
                            "shape": "arrowDown" if long_ else "arrowUp",
                            "color": "exit_win" if tr.net_pnl > 0 else "exit_loss",
                            "text": f"{label} @ {tr.exit_price:,.2f} ({tr.net_pnl:+,.2f})",
                            "position_id": tr.position_id, "kind": "exit"})
    markers.sort(key=lambda m: m["time"])
    p_stmt = select(Position).where(Position.symbol == symbol, Position.status == "OPEN")
    if position_id is not None:
        p_stmt = select(Position).where(Position.id == position_id, Position.status == "OPEN")
    for p in session.execute(p_stmt).scalars():
        tag = f"#{p.id}"
        lines.append({"price": r2(p.avg_entry), "kind": "entry", "title": f"Entry {tag}", "position_id": p.id})
        lines.append({"price": r2(p.stop_loss), "kind": "sl", "title": f"SL {tag}", "position_id": p.id})
        if p.target_1 is not None:
            lines.append({"price": r2(p.target_1), "kind": "t1", "title": f"T1 {tag}", "position_id": p.id})
        if p.target_2 is not None:
            lines.append({"price": r2(p.target_2), "kind": "t2", "title": f"T2 {tag}", "position_id": p.id})
    return markers, lines
