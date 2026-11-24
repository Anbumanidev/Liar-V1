"""Deterministic technical indicators (pure pandas / numpy, no TA library dependency).

Conventions
-----------
* Inputs follow the OHLCV contract in :mod:`core.types` (IST DatetimeIndex of bar OPEN
  times, float ``open/high/low/close/volume``). Only CLOSED bars should be passed in; every
  indicator here is causal (value at bar ``t`` uses bars ``<= t`` only).
* Smoothing matches TA-Lib so values are comparable with any charting tool:

  - ``ema``: SMA of the first ``period`` values as seed, then the recursive EMA
    ``ewm(span=period, adjust=False)`` (identical to TA-Lib ``EMA``).
  - ``rsi`` / ``atr`` / ``adx``: Wilder smoothing (alpha = 1/period) seeded with a simple
    average (TA-Lib compatible).
  - ``bollinger``: SMA(20) +/- 2 * population std (ddof=0), like TA-Lib ``BBANDS``.

* Values that cannot be computed yet (warm-up) are ``NaN`` in the frames;
  :func:`latest_values` converts them to ``None`` for :class:`core.types.TimeframeFeatures`.

Documented custom rules
-----------------------
* **VWAP** is session-anchored: typical price ``(H+L+C)/3`` x volume, cumulated from the
  first bar of each trading day (resets daily). Only meaningful for intraday frames; for
  daily/weekly frames the ``vwap`` / ``vwap_dev_pct`` columns are NaN (-> None).
* **Bollinger squeeze** (``bb_squeeze`` = 1): the current ``bb_width_pct`` is at or below the
  ``SQUEEZE_QUANTILE`` (10th) percentile of the trailing ``SQUEEZE_WINDOW`` (120) bars
  (at least ``SQUEEZE_MIN_PERIODS`` widths required), OR below an optional absolute
  ``squeeze_threshold_pct`` (disabled by default because a sensible absolute width differs
  wildly between 5m and weekly bars).
* **ATR expansion** = ATR(14) / mean(ATR(14) over the last 20 bars, current included).
* **rvol**: intraday frames -> volume / mean volume of the SAME time-of-day bar over the
  prior ``rvol_sessions`` sessions (needs ``rvol_min_sessions`` prior sessions, else NaN).
  Daily / weekly frames -> volume / mean volume of the PRIOR 20 bars (current excluded so a
  spike does not dilute itself). ``avg_volume_20`` is the plain 20-bar SMA of volume
  (current bar included).
* **trend_structure**, **rsi_bull_div**, **rsi_bear_div**: see :mod:`features.swings`
  (fractal swings with left=right=2, only confirmed swings are used).
* **inside_bar** = 1 when ``high <= prev high`` and ``low >= prev low``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from core.enums import Timeframe
from features.swings import rsi_divergence_series, trend_structure_series

#: Keys of ``TimeframeFeatures.values`` (see core/types.py) - also the columns returned by
#: :func:`compute_indicators`, in this order.
VALUE_KEYS: tuple[str, ...] = (
    "ema_9", "ema_20", "ema_21", "ema_50", "ema_200",
    "vwap",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "supertrend", "supertrend_dir",
    "adx_14", "plus_di", "minus_di",
    "bb_upper", "bb_mid", "bb_lower", "bb_width_pct", "bb_squeeze",
    "atr_14", "atr_pct", "atr_expansion",
    "volume", "avg_volume_20", "rvol",
    "vwap_dev_pct",
    "close", "high", "low", "open",
    "trend_structure", "inside_bar", "rsi_bull_div", "rsi_bear_div",
)

INTRADAY_TIMEFRAMES: frozenset[str] = frozenset(
    {Timeframe.M1.value, Timeframe.M5.value, Timeframe.M15.value, Timeframe.H1.value}
)

SQUEEZE_WINDOW = 120
SQUEEZE_QUANTILE = 0.10
SQUEEZE_MIN_PERIODS = 40
ATR_EXPANSION_WINDOW = 20
SWING_LEFT = 2
SWING_RIGHT = 2


# --------------------------------------------------------------------------- helpers


def is_intraday_timeframe(timeframe: str | Timeframe | None) -> bool | None:
    """True for 1m/5m/15m/1h, False for 1d/1wk, None when unknown."""
    if timeframe is None:
        return None
    tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
    if tf in INTRADAY_TIMEFRAMES:
        return True
    if tf in (Timeframe.D1.value, Timeframe.W1.value):
        return False
    return None


def infer_intraday(df: pd.DataFrame) -> bool:
    """Infer whether a candle frame is intraday (more than one bar on some date)."""
    if len(df) < 2 or not isinstance(df.index, pd.DatetimeIndex):
        return False
    return bool(df.index.normalize().duplicated().any())


def _f(series: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(series, dtype="float64")


def _wilder(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing seeded with the simple mean of the first ``period`` valid values.

    The seed is placed at ``first_valid + period - 1``; afterwards
    ``avg = (prev * (period - 1) + x) / period``. NaNs after the seed carry the previous value.
    """
    n = len(x)
    out = np.full(n, np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if valid.size == 0:
        return out
    first = int(valid[0])
    seed_idx = first + period - 1
    if seed_idx >= n:
        return out
    window = x[first:seed_idx + 1]
    if np.isnan(window).any():
        window = window[~np.isnan(window)]
        if window.size == 0:
            return out
    prev = float(window.mean())
    out[seed_idx] = prev
    for i in range(seed_idx + 1, n):
        xi = x[i]
        if not np.isnan(xi):
            prev = (prev * (period - 1) + xi) / period
        out[i] = prev
    return out


# --------------------------------------------------------------------------- moving averages


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average (NaN until ``period`` values are available)."""
    return series.astype("float64").rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, TA-Lib compatible.

    Seeds with the SMA of the first ``period`` valid values, then applies
    ``ewm(span=period, adjust=False)``. NaN before the seed bar.
    """
    values = _f(series).copy()
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size == 0:
        return pd.Series(out, index=series.index)
    first = int(valid[0])
    seed_idx = first + period - 1
    if seed_idx >= len(values):
        return pd.Series(out, index=series.index)
    seeded = values.copy()
    seeded[:seed_idx] = np.nan
    seeded[seed_idx] = float(np.nanmean(values[first:seed_idx + 1]))
    res = pd.Series(seeded).ewm(span=period, adjust=False).mean().to_numpy(copy=True)
    res[:seed_idx] = np.nan
    return pd.Series(res, index=series.index)


# --------------------------------------------------------------------------- oscillators


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (TA-Lib compatible). A completely flat window returns 50."""
    c = _f(close)
    diff = np.full(len(c), np.nan)
    if len(c) > 1:
        diff[1:] = np.diff(c)
    gain = np.where(np.isnan(diff), np.nan, np.clip(diff, 0, None))
    loss = np.where(np.isnan(diff), np.nan, np.clip(-diff, 0, None))
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    total = avg_gain + avg_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(total > 0, 100.0 * avg_gain / total, 50.0)
    out = np.where(np.isnan(avg_gain) | np.isnan(avg_loss), np.nan, out)
    return pd.Series(out, index=close.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal (EMA of MACD) and histogram. Columns ``macd, macd_signal, macd_hist``.

    Like TA-Lib, the fast EMA's SMA seed is aligned to end on the same bar as the slow EMA's
    seed (bars before ``slow - fast`` are ignored for the fast EMA), so values match TA-Lib.
    """
    offset = max(0, slow - fast)
    fast_src = close.astype("float64").copy()
    if offset:
        fast_src.iloc[:offset] = np.nan
    line = ema(fast_src, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig}, index=close.index)


# --------------------------------------------------------------------------- volatility / trend strength


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range; the first bar is NaN (no previous close), like TA-Lib ``TRANGE``."""
    h, lo, c = _f(df["high"]), _f(df["low"]), _f(df["close"])
    prev_c = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum.reduce([h - lo, np.abs(h - prev_c), np.abs(lo - prev_c)])
    if len(tr):
        tr[0] = np.nan
    return pd.Series(tr, index=df.index)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing (TA-Lib compatible, first value at bar ``period``)."""
    return pd.Series(_wilder(true_range(df).to_numpy(), period), index=df.index)


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX, +DI and -DI (Wilder; TA-Lib compatible).

    Returns columns ``adx, plus_di, minus_di``. DI values start at bar ``period``; ADX at bar
    ``2 * period - 1``.
    """
    h, lo, c = _f(df["high"]), _f(df["low"]), _f(df["close"])
    n = len(h)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    adx_out = np.full(n, np.nan)
    if n <= period:
        return pd.DataFrame({"adx": adx_out, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)

    def _dm_tr(i: int) -> tuple[float, float, float]:
        up = h[i] - h[i - 1]
        down = lo[i - 1] - lo[i]
        pdm = up if (up > 0 and up > down) else 0.0
        mdm = down if (down > 0 and down > up) else 0.0
        tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
        return pdm, mdm, tr

    sm_p = sm_m = sm_tr = 0.0
    for i in range(1, period):
        pdm, mdm, tr = _dm_tr(i)
        sm_p += pdm
        sm_m += mdm
        sm_tr += tr

    dx = np.full(n, np.nan)
    for i in range(period, n):
        pdm, mdm, tr = _dm_tr(i)
        sm_p = sm_p - sm_p / period + pdm
        sm_m = sm_m - sm_m / period + mdm
        sm_tr = sm_tr - sm_tr / period + tr
        if sm_tr > 0:
            p = 100.0 * sm_p / sm_tr
            m = 100.0 * sm_m / sm_tr
        else:
            p = m = 0.0
        plus_di[i] = p
        minus_di[i] = m
        s = p + m
        dx[i] = 100.0 * abs(p - m) / s if s > 0 else 0.0

    first_adx = 2 * period - 1
    if first_adx < n:
        prev = float(np.mean(dx[period:first_adx + 1]))
        adx_out[first_adx] = prev
        for i in range(first_adx + 1, n):
            prev = (prev * (period - 1) + dx[i]) / period
            adx_out[i] = prev
    return pd.DataFrame({"adx": adx_out, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Bollinger bands: columns ``bb_upper, bb_mid, bb_lower, bb_width_pct`` (population std)."""
    mid = sma(close, period)
    std = close.astype("float64").rolling(period, min_periods=period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower) / mid.where(mid != 0) * 100.0
    return pd.DataFrame({"bb_upper": upper, "bb_mid": mid, "bb_lower": lower, "bb_width_pct": width},
                        index=close.index)


def bollinger_squeeze(
    width_pct: pd.Series,
    window: int = SQUEEZE_WINDOW,
    quantile: float = SQUEEZE_QUANTILE,
    min_periods: int = SQUEEZE_MIN_PERIODS,
    threshold_pct: float | None = None,
) -> pd.Series:
    """Squeeze flag (1/0, NaN when undeterminable). See module docstring for the rule."""
    q = width_pct.rolling(window, min_periods=min_periods).quantile(quantile)
    flag = (width_pct <= q)
    if threshold_pct is not None:
        flag = flag | (width_pct < threshold_pct)
    out = flag.astype("float64")
    if threshold_pct is None:
        undetermined = width_pct.isna() | q.isna()
    else:
        # Without enough history the absolute threshold can still assert a squeeze.
        undetermined = width_pct.isna() | (q.isna() & ~(width_pct < threshold_pct))
    return out.mask(undetermined)


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Standard Supertrend on Wilder ATR. Columns ``supertrend`` (line) and ``supertrend_dir`` (+1/-1).

    Band ratchet: the final upper band only moves down (unless the previous close broke above
    it), the final lower band only moves up (unless the previous close broke below it).
    Direction flips to +1 when the close crosses above the final upper band and to -1 when it
    crosses below the final lower band. The line is the lower band in an uptrend and the upper
    band in a downtrend. Initial direction: +1 if close >= hl2 else -1.
    """
    h, lo, c = _f(df["high"]), _f(df["low"]), _f(df["close"])
    n = len(c)
    a = atr(df, period).to_numpy()
    hl2 = (h + lo) / 2.0
    basic_up = hl2 + mult * a
    basic_lo = hl2 - mult * a
    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    fu = fl = np.nan
    d = 0
    started = False
    for i in range(n):
        if np.isnan(a[i]):
            continue
        if not started:
            fu, fl = basic_up[i], basic_lo[i]
            d = 1 if c[i] >= hl2[i] else -1
            started = True
        else:
            prev_c = c[i - 1]
            new_fu = basic_up[i] if (basic_up[i] < fu or prev_c > fu) else fu
            new_fl = basic_lo[i] if (basic_lo[i] > fl or prev_c < fl) else fl
            if d == -1 and c[i] > new_fu:
                d = 1
            elif d == 1 and c[i] < new_fl:
                d = -1
            fu, fl = new_fu, new_fl
        line[i] = fl if d == 1 else fu
        direction[i] = float(d)
    return pd.DataFrame({"supertrend": line, "supertrend_dir": direction}, index=df.index)


# --------------------------------------------------------------------------- volume


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP (resets on each IST trading date). NaN while cumulative volume is 0."""
    tp = (df["high"].astype("float64") + df["low"].astype("float64") + df["close"].astype("float64")) / 3.0
    vol = df["volume"].astype("float64").fillna(0.0)
    day = df.index.normalize()
    cum_pv = (tp * vol).groupby(day).cumsum()
    cum_v = vol.groupby(day).cumsum()
    return (cum_pv / cum_v.where(cum_v > 0)).astype("float64")


def rvol(
    df: pd.DataFrame,
    intraday: bool | None = None,
    sessions: int = 10,
    min_sessions: int = 2,
    bars: int = 20,
) -> pd.Series:
    """Relative volume (see module docstring for the exact rule)."""
    vol = df["volume"].astype("float64")
    if intraday is None:
        intraday = infer_intraday(df)
    if intraday:
        tod = pd.Series(df.index.hour * 60 + df.index.minute, index=df.index)
        prior = vol.groupby(tod.to_numpy()).transform(
            lambda s: s.shift(1).rolling(sessions, min_periods=min_sessions).mean()
        )
    else:
        prior = vol.shift(1).rolling(bars, min_periods=bars).mean()
    return vol / prior.where(prior > 0)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume (starts at 0 on the first bar)."""
    c = df["close"].astype("float64")
    sign = np.sign(c.diff().fillna(0.0))
    return (sign * df["volume"].astype("float64")).cumsum()


# --------------------------------------------------------------------------- aggregate


def compute_indicators(
    df: pd.DataFrame,
    timeframe: str | Timeframe | None = None,
    *,
    rvol_sessions: int = 10,
    rvol_min_sessions: int = 2,
    squeeze_threshold_pct: float | None = None,
) -> pd.DataFrame:
    """Compute every indicator for a candle frame.

    Args:
        df: OHLCV frame (closed bars only).
        timeframe: ``"5m"``, ``"1d"`` ... used to decide intraday-only indicators (VWAP,
            time-of-day rvol). Inferred from the index when omitted.
        rvol_sessions / rvol_min_sessions: intraday rvol lookback (sessions).
        squeeze_threshold_pct: optional absolute Bollinger width threshold (see module doc).

    Returns:
        DataFrame on ``df.index`` with exactly the :data:`VALUE_KEYS` columns (NaN where not
        computable).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame({k: pd.Series(dtype="float64") for k in VALUE_KEYS},
                            index=df.index if df is not None else None)
    intraday = is_intraday_timeframe(timeframe)
    if intraday is None:
        intraday = infer_intraday(df)
    o = df["open"].astype("float64")
    h = df["high"].astype("float64")
    lo = df["low"].astype("float64")
    c = df["close"].astype("float64")
    v = df["volume"].astype("float64")

    out: dict[str, Any] = {}
    for p in (9, 20, 21, 50, 200):
        out[f"ema_{p}"] = ema(c, p)
    vw = vwap(df) if intraday else pd.Series(np.nan, index=df.index)
    out["vwap"] = vw
    r = rsi(c, 14)
    out["rsi_14"] = r
    m = macd(c)
    out["macd"], out["macd_signal"], out["macd_hist"] = m["macd"], m["macd_signal"], m["macd_hist"]
    st = supertrend(df, 10, 3.0)
    out["supertrend"], out["supertrend_dir"] = st["supertrend"], st["supertrend_dir"]
    ad = adx(df, 14)
    out["adx_14"], out["plus_di"], out["minus_di"] = ad["adx"], ad["plus_di"], ad["minus_di"]
    bb = bollinger(c, 20, 2.0)
    for col in ("bb_upper", "bb_mid", "bb_lower", "bb_width_pct"):
        out[col] = bb[col]
    out["bb_squeeze"] = bollinger_squeeze(bb["bb_width_pct"], threshold_pct=squeeze_threshold_pct)
    a = atr(df, 14)
    out["atr_14"] = a
    out["atr_pct"] = a / c.where(c != 0) * 100.0
    atr_avg = a.rolling(ATR_EXPANSION_WINDOW, min_periods=ATR_EXPANSION_WINDOW).mean()
    out["atr_expansion"] = a / atr_avg.where(atr_avg > 0)
    out["volume"] = v
    out["avg_volume_20"] = sma(v, 20)
    out["rvol"] = rvol(df, intraday=intraday, sessions=rvol_sessions, min_sessions=rvol_min_sessions)
    out["vwap_dev_pct"] = (c - vw) / vw.where(vw != 0) * 100.0
    out["close"], out["high"], out["low"], out["open"] = c, h, lo, o
    out["trend_structure"] = pd.Series(
        trend_structure_series(h.to_numpy(), lo.to_numpy(), SWING_LEFT, SWING_RIGHT), index=df.index)
    inside = (h <= h.shift(1)) & (lo >= lo.shift(1))
    ib = inside.astype("float64")
    if len(ib):
        ib.iloc[0] = np.nan
    out["inside_bar"] = ib
    bull, bear = rsi_divergence_series(h.to_numpy(), lo.to_numpy(), r.to_numpy(), SWING_LEFT, SWING_RIGHT)
    out["rsi_bull_div"] = pd.Series(bull, index=df.index)
    out["rsi_bear_div"] = pd.Series(bear, index=df.index)

    frame = pd.DataFrame({k: pd.Series(out[k], index=df.index).astype("float64") for k in VALUE_KEYS},
                         index=df.index)
    return frame


def _clean(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def latest_values(ind: pd.DataFrame) -> dict[str, float | None]:
    """Last row of :func:`compute_indicators` as a ``{key: float | None}`` dict (all VALUE_KEYS)."""
    if ind is None or len(ind) == 0:
        return {k: None for k in VALUE_KEYS}
    row = ind.iloc[-1]
    return {k: _clean(row[k]) if k in row.index else None for k in VALUE_KEYS}
