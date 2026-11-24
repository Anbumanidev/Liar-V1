"""Pivot / CPR levels, swing points and clustered support / resistance.

Formulas (classic floor pivots, H/L/C of the reference period):

* ``P = (H + L + C) / 3``; ``R1 = 2P - L``; ``S1 = 2P - H``; ``R2 = P + (H - L)``;
  ``S2 = P - (H - L)``; ``R3 = H + 2(P - L)``; ``S3 = L - 2(H - P)``.
* CPR: ``BC = (H + L) / 2``, ``TC = 2P - BC``; stored ordered so ``cpr_bc <= cpr_tc``;
  ``cpr_width_pct = (TC - BC) / P * 100``.

Reference period used by :func:`compute_levels` (relative to the session being analysed):

* intraday timeframes -> the previous trading session (daily bar),
* ``1d`` -> the previous calendar (ISO) week,
* ``1wk`` -> the previous calendar month.

Support / resistance: confirmed fractal swing highs and lows of the timeframe frame are
clustered (levels within ``cluster_pct`` % of each other merge into their mean); supports are
clusters below price (nearest first), resistances clusters above price (nearest first).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.enums import Timeframe
from core.types import Levels
from features.indicators import infer_intraday, is_intraday_timeframe
from features.swings import fractal_flags

DEFAULT_CLUSTER_PCT = 0.5
DEFAULT_MAX_SWINGS = 5


def pivots(prev_high: float, prev_low: float, prev_close: float) -> Levels:
    """Classic pivots + CPR from the reference period's high / low / close."""
    h, lo, c = float(prev_high), float(prev_low), float(prev_close)
    if h < lo:
        raise ValueError(f"prev_high {h} < prev_low {lo}")
    p = (h + lo + c) / 3.0
    bc_raw = (h + lo) / 2.0
    tc_raw = 2.0 * p - bc_raw
    bc, tc = min(bc_raw, tc_raw), max(bc_raw, tc_raw)
    width = (tc - bc) / p * 100.0 if p else None
    return Levels(
        pivot=p,
        r1=2.0 * p - lo,
        s1=2.0 * p - h,
        r2=p + (h - lo),
        s2=p - (h - lo),
        r3=h + 2.0 * (p - lo),
        s3=lo - 2.0 * (h - p),
        cpr_bc=bc,
        cpr_tc=tc,
        cpr_width_pct=width,
        prev_high=h,
        prev_low=lo,
        prev_close=c,
    )


def swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> tuple[pd.Series, pd.Series]:
    """Confirmed fractal swing highs and lows (chronological ``Series`` of price by bar time).

    A swing needs ``right`` closed bars after it, so the last ``right`` bars never qualify
    (no look-ahead).
    """
    if df is None or len(df) == 0:
        empty = pd.Series(dtype="float64")
        return empty, empty.copy()
    h = df["high"].to_numpy(dtype="float64")
    lo = df["low"].to_numpy(dtype="float64")
    is_h, is_l = fractal_flags(h, lo, left, right)
    return (pd.Series(h[is_h], index=df.index[is_h], dtype="float64"),
            pd.Series(lo[is_l], index=df.index[is_l], dtype="float64"))


def cluster_levels(levels: list[float], cluster_pct: float = DEFAULT_CLUSTER_PCT) -> list[tuple[float, int]]:
    """Greedy 1-D clustering. Returns ``(mean level, touches)`` sorted ascending by level.

    Sorted levels join the current cluster while within ``cluster_pct`` % of the cluster's
    first (lowest) member.
    """
    vals = sorted(float(x) for x in levels if x is not None and np.isfinite(x))
    clusters: list[list[float]] = []
    for v in vals:
        if clusters and clusters[-1][0] > 0 and (v - clusters[-1][0]) / clusters[-1][0] * 100.0 <= cluster_pct:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(float(np.mean(c)), len(c)) for c in clusters]


def support_resistance(
    df: pd.DataFrame,
    price: float,
    n: int = 3,
    *,
    left: int = 2,
    right: int = 2,
    cluster_pct: float = DEFAULT_CLUSTER_PCT,
) -> tuple[list[float], list[float]]:
    """Nearest ``n`` supports (below ``price``, descending) and resistances (above, ascending)."""
    highs, lows = swing_points(df, left, right)
    clusters = cluster_levels(list(highs.to_numpy()) + list(lows.to_numpy()), cluster_pct)
    supports = sorted((lvl for lvl, _ in clusters if lvl < price), reverse=True)[:n]
    resistances = sorted(lvl for lvl, _ in clusters if lvl > price)[:n]
    return supports, resistances


# --------------------------------------------------------------------------- reference periods


def _daily_from_intraday(df: pd.DataFrame) -> pd.DataFrame:
    day = df.index.normalize()
    g = df.groupby(day)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    })
    out.index.name = "ts"
    return out


def _next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def reference_hlc(
    daily_df: pd.DataFrame,
    session_date: date,
    period: str,
) -> tuple[float, float, float] | None:
    """H/L/C of the period preceding ``session_date``.

    Args:
        daily_df: daily candles (closed bars).
        session_date: the session being analysed / traded (weekends roll to Monday).
        period: ``"day"`` (previous session), ``"week"`` (previous ISO week) or ``"month"``.
    """
    if daily_df is None or len(daily_df) == 0:
        return None
    session_date = _next_weekday(session_date)
    dates = np.array(daily_df.index.date)
    if period == "day":
        sel = daily_df[dates < session_date]
        if sel.empty:
            return None
        last = sel.iloc[-1]
        return float(last["high"]), float(last["low"]), float(last["close"])
    if period == "week":
        week_start = session_date - timedelta(days=session_date.weekday())
        prev_start = week_start - timedelta(days=7)
        mask = (dates >= prev_start) & (dates < week_start)
    elif period == "month":
        month_start = session_date.replace(day=1)
        prev_start = (month_start - timedelta(days=1)).replace(day=1)
        mask = (dates >= prev_start) & (dates < month_start)
    else:
        raise ValueError(f"unknown period {period!r}")
    sel = daily_df[mask]
    if sel.empty:
        return None
    return float(sel["high"].max()), float(sel["low"].min()), float(sel["close"].iloc[-1])


def compute_levels(
    daily_df: pd.DataFrame | None,
    tf_df: pd.DataFrame | None,
    price: float,
    *,
    timeframe: str | Timeframe | None = None,
    session_date: date | None = None,
    n: int = 3,
    max_swings: int = DEFAULT_MAX_SWINGS,
    cluster_pct: float = DEFAULT_CLUSTER_PCT,
) -> Levels:
    """Pivots/CPR for the timeframe's reference period + swings + S/R from ``tf_df``.

    Args:
        daily_df: daily candles (for the pivot reference period). When empty and ``tf_df``
            is intraday, daily bars are aggregated from ``tf_df``.
        tf_df: candles of the timeframe being analysed (swings and S/R come from it).
        price: current price (splits supports from resistances).
        timeframe: timeframe label (inferred from ``tf_df`` when omitted).
        session_date: session being analysed; defaults to the date of the last ``tf_df`` bar
            (or the last daily bar).
        n: number of supports / resistances to keep.
        max_swings: number of most recent swing highs / lows kept (chronological order).
    """
    tf = timeframe.value if isinstance(timeframe, Timeframe) else timeframe
    intraday = is_intraday_timeframe(tf)
    if intraday is None:
        intraday = infer_intraday(tf_df) if tf_df is not None and len(tf_df) else False
    if session_date is None:
        if tf_df is not None and len(tf_df):
            session_date = tf_df.index[-1].date()
        elif daily_df is not None and len(daily_df):
            session_date = daily_df.index[-1].date()

    ref_daily = daily_df
    if (ref_daily is None or len(ref_daily) == 0) and intraday and tf_df is not None and len(tf_df):
        ref_daily = _daily_from_intraday(tf_df)

    if intraday:
        period = "day"
    elif tf == Timeframe.W1.value:
        period = "month"
    else:
        period = "week"

    levels = Levels()
    if session_date is not None:
        hlc = reference_hlc(ref_daily, session_date, period) if ref_daily is not None else None
        if hlc is not None:
            levels = pivots(*hlc)

    if tf_df is not None and len(tf_df):
        highs, lows = swing_points(tf_df)
        levels.swing_highs = [float(x) for x in highs.to_numpy()[-max_swings:]]
        levels.swing_lows = [float(x) for x in lows.to_numpy()[-max_swings:]]
        sup, res = support_resistance(tf_df, price, n, cluster_pct=cluster_pct)
        levels.supports = sup
        levels.resistances = res
    return levels


def inside_cpr(price: float | None, levels: Levels | None) -> bool:
    """True when ``price`` lies within the CPR band (inclusive)."""
    if price is None or levels is None or levels.cpr_bc is None or levels.cpr_tc is None:
        return False
    return levels.cpr_bc <= price <= levels.cpr_tc
