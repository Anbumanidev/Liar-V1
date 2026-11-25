"""Chart data: timeframes (incl. 10m/20m/30m/5h resampled), EMA/VWAP overlays, trade markers.

Native timeframes come straight from ``MarketDataService.candles`` (CLOSED bars, IST bar-open
index). 10m/20m/30m are built from 5m and 5h from 1h, anchored at the 09:15 IST session open of
each day - a bucket never spans two sessions (the last bucket of a day may be shorter).

Chart ``time`` values are IST wall-clock epoch seconds (UTC epoch + 19 800 s) because
Lightweight Charts renders timestamps as UTC; see ``webui.util.IST_OFFSET_S``.

History: :func:`load_window` serves the latest ``bars``, a named ``range`` (1D ... MAX, clamped to
what the provider keeps for the timeframe - :data:`HISTORY_DAYS`) or the ``bars`` ending
``before`` a chart time (scroll-back pagination), and reports ``has_more`` /
``earliest_available`` so the chart knows when the start of the available history is reached.
"""

from __future__ import annotations

import math
import threading
import time as _time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
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


# =========================================================================== history window (range / before)

#: How far back the data provider keeps each timeframe, in calendar days (Yahoo; ``MarketDataService``
#: clamps older requests automatically). Resampled timeframes inherit their source's limit.
HISTORY_DAYS: dict[str, int] = {"1m": 7, "5m": 59, "10m": 59, "15m": 59, "20m": 59, "30m": 59, "1h": 729, "5h": 729,
                                "1d": 4383, "1wk": 10957}
HISTORY_NOTE: dict[str, str] = {
    "1m": "Yahoo keeps 1-minute candles for ~7 days",
    "5m": "Yahoo keeps 5m for ~59 days", "10m": "10m is built from 5m; Yahoo keeps 5m for ~59 days",
    "15m": "Yahoo keeps 15m for ~59 days", "20m": "20m is built from 5m; Yahoo keeps 5m for ~59 days",
    "30m": "30m is built from 5m; Yahoo keeps 5m for ~59 days",
    "1h": "Yahoo keeps hourly candles for ~2 years", "5h": "5h is built from 1h; Yahoo keeps 1h for ~2 years",
    "1d": "daily history is shown for ~12 years", "1wk": "weekly candles cover the full listed history",
}
RANGES = ("1D", "5D", "1M", "3M", "6M", "1Y", "5Y", "MAX")
_RANGE_DAYS = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365, "5Y": 1826}
#: Upper bound of bars per request (range responses are trimmed to the newest MAX_BARS).
MAX_BARS = 5000
_OPEN = time(9, 15)


def _at_open(d: date) -> datetime:
    return datetime.combine(d, _OPEN, tzinfo=IST)


def earliest_available(spec: TfSpec, now: datetime) -> datetime:
    """Oldest bar the provider keeps for ``spec`` (intraday: to the minute; daily/weekly: that day's open)."""
    start = to_ist(now) - timedelta(days=HISTORY_DAYS[spec.key])
    if not spec.intraday:
        return _at_open(start.date())
    return start.replace(second=0, microsecond=0) + timedelta(minutes=1)


def _last_session(now: datetime, calendar: Any) -> date:
    today = to_ist(now).date()
    try:
        if calendar.is_trading_day(today) and now >= calendar.market_open_dt(today):
            return today
        return calendar.prev_trading_day(today)
    except Exception:  # noqa: BLE001 - calendar unavailable: weekdays only
        d = today if now.time() >= _OPEN else today - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d


def _sessions_back(day: date, n: int, calendar: Any) -> date:
    """The trading day ``n`` sessions before ``day`` (weekday fallback without a calendar)."""
    try:
        return calendar.add_trading_days(day, -n) if n else day
    except Exception:  # noqa: BLE001
        d = day
        while n > 0:
            d -= timedelta(days=1)
            if d.weekday() < 5:
                n -= 1
        return d


def range_start(key: str, spec: TfSpec, now: datetime, calendar: Any) -> datetime:
    """Unclamped start of a named range (1D/5D = session opens; months/years = that day's open)."""
    if key == "MAX":
        return earliest_available(spec, now)
    if key in ("1D", "5D"):
        last = _last_session(now, calendar)
        return _at_open(_sessions_back(last, 0 if key == "1D" else 4, calendar))
    return _at_open(to_ist(now).date() - timedelta(days=_RANGE_DAYS[key]))


def start_for_bars(spec: TfSpec, end: datetime, bars: int, calendar: Any) -> datetime:
    """A start that yields at least ``bars`` bars before ``end`` (one spare session / a few spare bars)."""
    end_day = to_ist(end).date()
    if spec.intraday:
        per_day = max(1, math.ceil(_SESSION_MIN / spec.bar_minutes))
        return _at_open(_sessions_back(end_day, math.ceil(bars / per_day) + 1, calendar))
    if spec.key == "1d":
        return _at_open(_sessions_back(end_day, bars + 3, calendar))
    return _at_open(end_day - timedelta(weeks=bars + 2))


def _fetch_from(data: Any, symbol: str, spec: TfSpec, start: datetime) -> pd.DataFrame:
    """Every closed bar of ``spec`` from ``start`` up to now (resampled timeframes session-anchored)."""
    lead = timedelta(minutes=spec.minutes) if spec.minutes else timedelta(0)
    df = data.candles(symbol, spec.source, lookback_bars=MAX_BARS, start=start - lead)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if spec.minutes is not None:
        df = resample_session(df, spec.minutes)
    return df


def _gap_tolerance(spec: TfSpec) -> timedelta:
    """How far the first bar may sit after the requested start and still count as 'data goes further back'."""
    return timedelta(days=5) if spec.intraday else timedelta(days=8 if spec.key == "1d" else 22)


def load_window(data: Any, symbol: str, spec: TfSpec, now: datetime, calendar: Any, *, bars: int,
                start: datetime | None = None, end: datetime | None = None, range_key: str | None = None,
                before: datetime | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one chart window. Precedence: ``before`` > ``range_key`` > ``start``/``end`` > latest ``bars``.

    Returns ``(frame, info)`` where ``info`` has ``has_more`` (older bars exist), ``earliest_available``,
    ``clamped`` (the request reached past the provider's history) and the effective ``start``.
    """
    earliest = earliest_available(spec, now)
    info: dict[str, Any] = {"range": None, "clamped": False, "start": None, "before": None}
    if before is not None:
        want = start_for_bars(spec, before, bars, calendar)
        clamped = want <= earliest
        eff = max(want, earliest)
        df = _fetch_from(data, symbol, spec, eff)
        df = df.loc[pd.DatetimeIndex(df.index) < pd.Timestamp(to_ist(before))]
        has_more = len(df) > bars or (len(df) == bars and not clamped)
        df = df.tail(bars)
        info |= {"clamped": clamped, "start": iso(eff), "before": iso(before)}
    elif range_key is not None:
        want = range_start(range_key, spec, now, calendar)
        clamped = range_key == "MAX" or want < earliest
        eff = max(want, earliest)
        df = _fetch_from(data, symbol, spec, eff)
        if end is not None:
            df = df.loc[pd.DatetimeIndex(df.index) <= pd.Timestamp(to_ist(end))]
        first = pd.Timestamp(df.index[0]).to_pydatetime() if len(df) else None
        has_more = not clamped and first is not None and first - eff <= _gap_tolerance(spec)
        if len(df) > MAX_BARS:
            df, has_more = df.tail(MAX_BARS), True
        info |= {"range": range_key, "clamped": clamped, "start": iso(eff)}
    else:
        df = build_frame(data, symbol, spec, bars, start, end)
        if start is not None:
            first = pd.Timestamp(df.index[0]).to_pydatetime() if len(df) else None
            has_more = first is not None and start > earliest and first - start <= _gap_tolerance(spec)
            info |= {"clamped": start <= earliest, "start": iso(start)}
        else:
            has_more = len(df) >= bars
    info |= {"has_more": bool(has_more), "earliest_available": iso(earliest),
             "history_days": HISTORY_DAYS[spec.key], "history_note": HISTORY_NOTE[spec.key]}
    return df, info


def _snap(times: list[int], t: int, bar_seconds: int) -> int | None:
    """Bar time containing ``t`` (last bar opening at or before it); None if outside the data."""
    if not times or t < times[0]:
        return None
    i = bisect_right(times, t) - 1
    if i == len(times) - 1 and t > times[-1] + 2 * bar_seconds:
        return None
    return times[i]


def trade_overlays(session: Session, symbol: str, times: list[int], spec: TfSpec,
                   position_id: int | None = None, hi_time: int | None = None
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Markers (entries/exits) inside the candle range + price lines for open positions.

    ``hi_time`` (chart time, exclusive) drops fills at/after it - used by scroll-back pages so a fill that
    belongs to the newer, already loaded bars is never snapped onto this page's last bar.
    """
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
            raw = int(to_ist(o.filled_at).timestamp()) + IST_OFFSET_S
            if hi_time is not None and raw >= hi_time:
                continue
            t = _snap(times, raw, bar_s)
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
            raw = int(to_ist(tr.exit_time).timestamp()) + IST_OFFSET_S
            if hi_time is not None and raw >= hi_time:
                continue
            t = _snap(times, raw, bar_s)
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
