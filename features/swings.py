"""Swing-point (fractal) primitives shared by indicators, patterns and levels.

Definition (Bill Williams style fractal with configurable arms):
bar ``i`` is a **swing high** when ``high[i]`` is strictly greater than every high of the
``left`` bars before it and greater than or equal to every high of the ``right`` bars after
it. Swing lows mirror this on ``low``. Ties therefore resolve to the EARLIEST bar of a
flat top/bottom.

No look-ahead: a swing at bar ``i`` needs the ``right`` bars after it, so it only becomes
*known* at bar ``i + right``. Every per-bar series built here (``trend_structure_series``,
divergence flags) respects that confirmation delay; bars without a full right arm are never
flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fractal_flags(high: np.ndarray, low: np.ndarray, left: int = 2, right: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean arrays ``(is_swing_high, is_swing_low)`` (same length as input).

    Args:
        high: bar highs.
        low: bar lows.
        left: bars required on the left of the pivot (>= 1).
        right: bars required on the right of the pivot (>= 1).
    """
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")
    h = pd.Series(np.asarray(high, dtype="float64"))
    lo = pd.Series(np.asarray(low, dtype="float64"))
    n = len(h)
    if n < left + right + 1:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    left_max = h.shift(1).rolling(left, min_periods=left).max()
    right_max = h[::-1].shift(1).rolling(right, min_periods=right).max()[::-1]
    left_min = lo.shift(1).rolling(left, min_periods=left).min()
    right_min = lo[::-1].shift(1).rolling(right, min_periods=right).min()[::-1]

    is_high = ((h > left_max) & (h >= right_max)).to_numpy(dtype=bool, na_value=False)
    is_low = ((lo < left_min) & (lo <= right_min)).to_numpy(dtype=bool, na_value=False)
    return is_high, is_low


def structure_from_pivots(highs: list[float], lows: list[float]) -> int:
    """Classify market structure from chronological swing highs / lows.

    Returns ``+1`` for higher-high AND higher-low, ``-1`` for lower-high AND lower-low,
    ``0`` otherwise (mixed structure or fewer than two pivots of either kind).
    """
    if len(highs) < 2 or len(lows) < 2:
        return 0
    hh = highs[-1] > highs[-2]
    hl = lows[-1] > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1] < lows[-2]
    if hh and hl:
        return 1
    if lh and ll:
        return -1
    return 0


def trend_structure_series(high: np.ndarray, low: np.ndarray, left: int = 2, right: int = 2) -> np.ndarray:
    """Per-bar trend structure (+1 HH/HL, -1 LH/LL, 0) using only swings confirmed by that bar."""
    h = np.asarray(high, dtype="float64")
    lo = np.asarray(low, dtype="float64")
    n = len(h)
    out = np.zeros(n, dtype="float64")
    is_high, is_low = fractal_flags(h, lo, left, right)
    highs: list[float] = []
    lows: list[float] = []
    for t in range(n):
        j = t - right  # the pivot that becomes confirmed at bar t
        if j >= 0:
            if is_high[j]:
                highs.append(float(h[j]))
            if is_low[j]:
                lows.append(float(lo[j]))
        out[t] = structure_from_pivots(highs, lows)
    return out


def rsi_divergence_series(
    high: np.ndarray,
    low: np.ndarray,
    rsi: np.ndarray,
    left: int = 2,
    right: int = 2,
    max_age: int = 5,
    min_separation: int = 3,
    max_separation: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Regular RSI divergence flags per bar (no look-ahead).

    Rule (documented, deliberately simple):

    * **Bullish** at bar ``t``: the most recent swing LOW confirmed by ``t`` is at bar ``j``
      with ``t - j <= max_age + right``; the previous swing low ``k`` satisfies
      ``min_separation <= j - k <= max_separation``; price made a lower low
      (``low[j] < low[k]``) while RSI made a higher low (``rsi[j] > rsi[k]``).
    * **Bearish** mirrors it on swing HIGHS: higher high in price, lower high in RSI.

    Returns float arrays of 0/1 (NaN where RSI is not yet available).
    """
    h = np.asarray(high, dtype="float64")
    lo = np.asarray(low, dtype="float64")
    r = np.asarray(rsi, dtype="float64")
    n = len(h)
    bull = np.zeros(n, dtype="float64")
    bear = np.zeros(n, dtype="float64")
    is_high, is_low = fractal_flags(h, lo, left, right)
    sh: list[int] = []
    sl: list[int] = []
    for t in range(n):
        j = t - right
        if j >= 0:
            if is_high[j] and not np.isnan(r[j]):
                sh.append(j)
            if is_low[j] and not np.isnan(r[j]):
                sl.append(j)
        if np.isnan(r[t]):
            bull[t] = np.nan
            bear[t] = np.nan
            continue
        if len(sl) >= 2:
            j1, k1 = sl[-1], sl[-2]
            if (t - j1) <= max_age + right and min_separation <= j1 - k1 <= max_separation:
                if lo[j1] < lo[k1] and r[j1] > r[k1]:
                    bull[t] = 1.0
        if len(sh) >= 2:
            j2, k2 = sh[-1], sh[-2]
            if (t - j2) <= max_age + right and min_separation <= j2 - k2 <= max_separation:
                if h[j2] > h[k2] and r[j2] < r[k2]:
                    bear[t] = 1.0
    return bull, bear
