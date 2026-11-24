"""Helpers shared by the daily/weekly (BTST, SWING, POSITIONAL) strategies. LONG only."""

from __future__ import annotations

from core.enums import Signal
from core.types import MarketContext, StockFeatures
from strategies.base import collect_levels, val

TFD = "1d"
TFW = "1wk"


def atr_daily(features: StockFeatures) -> float | None:
    """Daily ATR: ``features.atr`` when the primary timeframe is daily, else the 1d ATR(14)."""
    if features.primary_timeframe == TFD and features.atr is not None and features.atr > 0:
        return float(features.atr)
    a = val(features, TFD, "atr_14")
    if a is not None and a > 0:
        return a
    return float(features.atr) if features.atr is not None and features.atr > 0 else None


def sector_of(features: StockFeatures) -> str | None:
    """Sector name from ``features.extra['sector']`` (None when unknown)."""
    s = features.extra.get("sector")
    return str(s) if s else None


def sector_strength(features: StockFeatures, market: MarketContext | None) -> float | None:
    """Sector strength (% change / RS) of the stock's sector from the market context."""
    sector = sector_of(features)
    if market is None or sector is None:
        return None
    v = market.sector_strength.get(sector)
    return float(v) if v is not None else None


def sector_rank(features: StockFeatures, market: MarketContext | None) -> int | None:
    """1-based rank of the stock's sector by strength (1 = strongest). None when unknown."""
    sector = sector_of(features)
    if market is None or sector is None or sector not in market.sector_strength:
        return None
    ordered = sorted(market.sector_strength.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ordered].index(sector) + 1


def market_not_bearish(market: MarketContext | None) -> bool:
    """True unless the NIFTY trend is BEARISH (unknown market counts as not bearish)."""
    return market is None or market.nifty_trend != Signal.BEARISH


def daily_levels(features: StockFeatures) -> list[float]:
    """Daily + weekly levels and 52-week extremes."""
    return collect_levels(features, [t for t in (TFD, TFW) if features.tf(t) is not None])


def base_range(features: StockFeatures) -> tuple[float | None, float | None]:
    """Consolidation base high/low.

    ``extra['range_high_20']`` / ``extra['range_low_20']`` (prior 20 daily bars, excluding the
    last closed bar) when provided; else the max/min of the daily swing highs/lows.
    """
    hi = features.extra.get("range_high_20")
    lo = features.extra.get("range_low_20")
    tf = features.tf(TFD)
    if (hi is None or lo is None) and tf is not None and tf.levels is not None:
        if hi is None and tf.levels.swing_highs:
            hi = max(tf.levels.swing_highs)
        if lo is None and tf.levels.swing_lows:
            lo = min(tf.levels.swing_lows)
    return (float(hi) if hi is not None else None, float(lo) if lo is not None else None)


def fundamental_score(features: StockFeatures) -> float | None:
    """``features.extra['fundamental_score']`` (0-100) when present."""
    v = features.extra.get("fundamental_score")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
