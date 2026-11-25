"""OHLCV normalisation and session-aligned resampling.

Candle contract (``core.types``): tz-aware IST ``DatetimeIndex`` of bar OPEN times (named
``ts``), ascending, unique, float64 columns ``open, high, low, close, volume``.

Resampling is aligned to the NSE session that starts at 09:15 IST - NOT to the clock hour:

* intraday ``N``-minute bars open at 09:15 + k*N (``1h`` -> 09:15, 10:15, ..., 15:15, the last
  one only 15 minutes long because the session ends at 15:30);
* daily bars are indexed at 09:15 IST of their date;
* weekly bars (Monday-Sunday weeks) are indexed at 09:15 IST of the first day present in the week.

Rows outside the regular session are ignored when building intraday bars.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
from loguru import logger

from core.clock import IST
from core.enums import TIMEFRAME_MINUTES, Timeframe
from core.types import OHLCV_COLUMNS, empty_ohlcv
from data.calendar import INTRADAY_TIMEFRAMES, SESSION_MINUTES, as_timeframe

_SESSION_START = pd.Timedelta(hours=9, minutes=15)

#: Columns we accept as the timestamp when the frame has a RangeIndex.
_TS_COLUMNS = ("ts", "datetime", "date", "timestamp", "time")
_ALIASES = {"vol": "volume", "volumes": "volume", "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "last": "close", "ltp": "close"}

#: Which finer timeframes can build a coarser one (in preference order).
RESAMPLE_SOURCES: dict[Timeframe, tuple[Timeframe, ...]] = {
    Timeframe.M1: (),
    Timeframe.M5: (Timeframe.M1,),
    Timeframe.M15: (Timeframe.M5, Timeframe.M1),
    Timeframe.H1: (Timeframe.M15, Timeframe.M5, Timeframe.M1),
    Timeframe.D1: (Timeframe.H1, Timeframe.M15, Timeframe.M5, Timeframe.M1),
    Timeframe.W1: (Timeframe.D1,),
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        rename[col] = _ALIASES.get(key, key)
    return df.rename(columns=rename)


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    for col in _TS_COLUMNS:
        if col in df.columns:
            return df.set_index(pd.DatetimeIndex(pd.to_datetime(df.pop(col), utc=False))).copy()
    return df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))


def _all_midnight(idx: pd.DatetimeIndex) -> bool:
    return bool(((idx.hour == 0) & (idx.minute == 0) & (idx.second == 0)).all())


def _is_daily_like(idx: pd.DatetimeIndex) -> bool:
    """All timestamps at midnight - in their own timezone or in IST (daily/weekly vendor output)."""
    if len(idx) == 0:
        return False
    return _all_midnight(idx) or (idx.tz is not None and _all_midnight(idx.tz_convert(IST)))


def _align_to_session_open(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map each daily/weekly timestamp to 09:15 IST of its trading date.

    * A timestamp at midnight in its OWN timezone is a vendor date label: its local calendar date
      is kept (``2026-11-23 00:00 UTC`` or ``00:00 America/New_York`` -> 23-Nov).
    * Any other tz-aware timestamp is an instant, converted to IST first
      (``2026-11-22 18:30 UTC`` == 23-Nov 00:00 IST -> 23-Nov; ``03:45 UTC`` -> same day).
    * Naive timestamps are IST wall time.
    """
    if idx.tz is None:
        local = idx
    else:
        wall = idx.tz_localize(None)
        ist_wall = idx.tz_convert(IST).tz_localize(None)
        local = wall.where(wall == wall.normalize(), ist_wall)
    return pd.DatetimeIndex(pd.DatetimeIndex(local).normalize() + _SESSION_START).tz_localize(IST)


def normalize_ohlcv(df: pd.DataFrame | None, timeframe: Timeframe | str | None = None) -> pd.DataFrame:
    """Return a contract-compliant candle frame.

    * lower-cases columns, maps aliases, drops extras (``adj close``, dividends ...);
    * converts the index to IST (naive timestamps are assumed to be IST);
    * daily/weekly bars (``timeframe`` D1/W1, or every timestamp at local midnight when
      ``timeframe`` is None) are re-indexed to 09:15 IST of their LOCAL calendar date, so a
      vendor's ``2026-11-23 00:00 UTC`` or ``00:00 Asia/Tokyo`` stays on 23-Nov;
    * sorts, de-duplicates (last wins), drops rows with missing/non-positive prices,
      fills missing volume with 0, casts to float64.
    """
    if df is None or len(df) == 0:
        return empty_ohlcv()
    out = _to_datetime_index(_clean_columns(df.copy()))
    missing = [c for c in ("open", "high", "low", "close") if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing price columns {missing}: {list(out.columns)}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out = out[list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce").astype("float64")

    idx = pd.DatetimeIndex(out.index)
    tf = as_timeframe(timeframe) if timeframe is not None else None
    if tf in (Timeframe.D1, Timeframe.W1) or (tf is None and _is_daily_like(idx)):
        idx = _align_to_session_open(idx)
    elif idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)
    out.index = idx.as_unit("ns") if hasattr(idx, "as_unit") else idx
    out.index.name = "ts"

    out["volume"] = out["volume"].fillna(0.0)
    bad = out[["open", "high", "low", "close"]].isna().any(axis=1) | (out["close"] <= 0)
    if bad.any():
        logger.debug(f"[bars] dropping {int(bad.sum())} rows with missing/non-positive prices")
        out = out.loc[~bad]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


#: IST is a fixed UTC+05:30 offset (Asia/Kolkata has no DST), so bucket keys are integer maths.
_IST_OFFSET_NS = 19_800 * 1_000_000_000
_NS_PER_MIN = 60 * 1_000_000_000
_NS_PER_DAY = 1_440 * _NS_PER_MIN
_OPEN_MIN = 9 * 60 + 15


def _run_starts(values: np.ndarray) -> np.ndarray:
    """Positions where a run of equal consecutive ``values`` begins."""
    return np.flatnonzero(np.r_[True, values[1:] != values[:-1]])


def resample_ohlcv(df: pd.DataFrame, timeframe: Timeframe | str) -> pd.DataFrame:
    """Resample a (finer) candle frame to ``timeframe``, session-aligned at 09:15 IST.

    The source must be at least as fine as the target (e.g. 5m -> 15m/1h/1d, 1d -> 1wk).
    Incomplete trailing buckets are kept - callers apply ``data.calendar.closed_bars`` to
    drop bars that have not closed yet. Rows with a missing price are ignored.

    Candle frames are sorted, so every bucket is a contiguous run of rows: bucket keys are
    computed with int64 maths on IST wall-clock nanoseconds and aggregated with numpy
    ``reduceat`` (open first, high max, low min, close last, volume sum). pandas ``groupby`` costs
    ~50x more per call, which matters in REPLAY where bars are rebuilt as the clock advances.
    """
    tf = as_timeframe(timeframe)
    if df is None or df.empty:
        return empty_ohlcv()
    conforming = (isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None
                  and df.index.is_monotonic_increasing and all(c in df.columns for c in OHLCV_COLUMNS))
    src = df if conforming else normalize_ohlcv(df)
    o, h, lo, c = (src[col].to_numpy(dtype="float64") for col in ("open", "high", "low", "close"))
    vol = np.nan_to_num(src["volume"].to_numpy(dtype="float64"))
    local = pd.DatetimeIndex(src.index).as_unit("ns").asi8 + _IST_OFFSET_NS   # IST wall clock, ns
    day = local // _NS_PER_DAY * _NS_PER_DAY
    keep = ~(np.isnan(o) | np.isnan(h) | np.isnan(lo) | np.isnan(c))

    if tf in INTRADAY_TIMEFRAMES:
        minutes = TIMEFRAME_MINUTES[tf]
        offset = (local - day) // _NS_PER_MIN - _OPEN_MIN
        keep &= (offset >= 0) & (offset < SESSION_MINUTES)        # regular session only
        keys = day + (_OPEN_MIN + offset // minutes * minutes) * _NS_PER_MIN
    else:
        keys = day + _OPEN_MIN * _NS_PER_MIN
    if not keep.all():
        o, h, lo, c, vol, day, keys = (a[keep] for a in (o, h, lo, c, vol, day, keys))
    if len(keys) == 0:
        return empty_ohlcv()
    if tf == Timeframe.W1:  # Monday-Sunday weeks, labelled with the first day present
        monday = day - ((day // _NS_PER_DAY + 3) % 7) * _NS_PER_DAY             # 1970-01-01 was a Thursday
        week = np.cumsum(np.r_[True, monday[1:] != monday[:-1]]) - 1
        keys = day[_run_starts(monday)][week] + _OPEN_MIN * _NS_PER_MIN

    starts = _run_starts(keys)
    last = np.r_[starts[1:], len(keys)] - 1
    index = pd.to_datetime(keys[starts] - _IST_OFFSET_NS, unit="ns", utc=True).tz_convert(IST)
    out = pd.DataFrame(
        {
            "open": o[starts],
            "high": np.fmax.reduceat(h, starts),
            "low": np.fmin.reduceat(lo, starts),
            "close": c[last],
            "volume": np.add.reduceat(vol, starts),
        },
        index=pd.DatetimeIndex(index, name="ts"),
    )
    return out.astype("float64")


def session_time(ts: pd.Timestamp) -> time:
    """Wall-clock IST time of a timestamp (helper for filters)."""
    return ts.tz_convert(IST).time() if ts.tzinfo is not None else ts.time()
