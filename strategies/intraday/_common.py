"""Helpers shared by the intraday strategies (primary timeframe 5m, context 15m / 1h)."""

from __future__ import annotations

from core.enums import Direction
from core.types import StockFeatures
from strategies.base import collect_levels, sign, val

TF5 = "5m"
TF15 = "15m"
TF1H = "1h"
INTRADAY_TFS: tuple[str, ...] = (TF5, TF15, TF1H)
DIRECTIONS: tuple[Direction, Direction] = (Direction.LONG, Direction.SHORT)


def rvol_of(features: StockFeatures, timeframe: str = TF5) -> float | None:
    """Relative volume: StockFeatures.rvol (same-time average) falling back to the bar's rvol."""
    if features.rvol is not None:
        return float(features.rvol)
    return val(features, timeframe, "rvol")


def atr_of(features: StockFeatures, timeframe: str = TF5) -> float | None:
    """ATR of the primary timeframe (``features.atr``) or of ``timeframe``."""
    if features.atr is not None and features.atr > 0:
        return float(features.atr)
    a = val(features, timeframe, "atr_14")
    return a if a is not None and a > 0 else None


def above(a: float | None, b: float | None, direction: Direction) -> bool:
    """``a`` is beyond ``b`` in ``direction`` (a > b for LONG, a < b for SHORT). False if unknown."""
    if a is None or b is None:
        return False
    return (a - b) * sign(direction) > 0


def ema_trend(features: StockFeatures, timeframe: str, direction: Direction) -> bool | None:
    """EMA 9 vs EMA 21 alignment on ``timeframe`` (None when unknown)."""
    e9, e21 = val(features, timeframe, "ema_9"), val(features, timeframe, "ema_21")
    if e9 is None or e21 is None:
        return None
    return (e9 - e21) * sign(direction) > 0


def target_levels(features: StockFeatures) -> list[float]:
    """All intraday-relevant levels (pivots/CPR/S-R on every intraday timeframe + prev day)."""
    lv = collect_levels(features, [t for t in INTRADAY_TFS if features.tf(t) is not None])
    for x in (features.day_high, features.day_low):
        if x is not None:
            lv.append(float(x))
    return lv
