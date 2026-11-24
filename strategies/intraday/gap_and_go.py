"""Gap-and-Go (INTRADAY, long + short).

Rules:
* opening gap between ``min_gap_pct`` and ``max_gap_pct`` (absolute, vs previous close);
* the gap held through the opening range (OR low stayed above the previous close for a gap
  up; OR high below it for a gap down);
* price breaks beyond the opening range in the gap direction, not more than
  ``max_extension_atr`` x ATR past it, on the right side of VWAP, with rvol >= ``min_rvol``.
Entry = LTP (MARKET). SL = other side of the OR (or OR midpoint / VWAP when the OR is wide).
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies.base import ParamSpec, Setup, Strategy, pick_stop, sign, strength_from, val
from strategies.intraday._common import TF5, above, atr_of, rvol_of, target_levels
from strategies.registry import register


@register
class GapAndGo(Strategy):
    name = "gap_and_go"
    description = "Gap that holds the opening range and continues in the gap direction."
    modes = (Mode.INTRADAY,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.UNDEFINED, Regime.HIGH_VOLATILITY})
    windows = {Mode.INTRADAY: [("09:30", "10:45")]}
    param_specs = (
        ParamSpec("min_gap_pct", 1.0, 0.3, 5.0, "float", "Minimum absolute gap in %"),
        ParamSpec("max_gap_pct", 5.0, 1.0, 10.0, "float", "Maximum absolute gap in % (exhaustion risk)"),
        ParamSpec("min_rvol", 1.5, 1.0, 5.0, "float", "Minimum relative volume"),
        ParamSpec("max_extension_atr", 1.0, 0.2, 3.0, "float", "Skip if price ran further past the OR (ATR)"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        gap, pc = features.gap_pct, features.prev_close
        orh, orl = features.opening_range_high, features.opening_range_low
        atr = atr_of(features)
        if None in (gap, pc, orh, orl, atr) or orh <= orl:  # type: ignore[operator]
            return None
        if not (params["min_gap_pct"] <= abs(gap) <= params["max_gap_pct"]):
            return None
        rvol = rvol_of(features)
        if rvol is None or rvol < params["min_rvol"]:
            return None
        d = Direction.LONG if gap > 0 else Direction.SHORT
        s = sign(d)
        edge, other = (orh, orl) if d == Direction.LONG else (orl, orh)
        if (other - pc) * s <= 0:
            return None  # gap was (partly) filled inside the opening range
        ltp = features.ltp
        if (ltp - edge) * s <= 0 or (ltp - edge) * s > params["max_extension_atr"] * atr:
            return None
        vwap = val(features, TF5, "vwap")
        if vwap is not None and not above(ltp, vwap, d):
            return None
        stop = pick_stop(d, ltp, [other, (orh + orl) / 2, vwap], atr, min_atr=0.3, max_atr=2.5, fallback_atr=1.5)
        if stop is None:
            return None
        return Setup(
            direction=d,
            entry=ltp,
            stop=stop,
            entry_type=EntryType.MARKET,
            target_levels=target_levels(features),
            strength=strength_from((rvol, 1.2, 3.0, 35.0), (abs(gap), params["min_gap_pct"], 3.0, 15.0)),
            notes=[f"gap {gap:+.2f}% held, OR {orl:.2f}-{orh:.2f} broken"],
        )
