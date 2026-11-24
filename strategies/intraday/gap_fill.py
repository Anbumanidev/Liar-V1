"""Gap-Fill (INTRADAY, long + short) - fade a failing opening gap back to the previous close.

Rules:
* opening gap between ``min_gap_pct`` and ``max_gap_pct`` (absolute);
* the gap is failing: price trades beyond the opposite side of the opening range (below the
  OR low after a gap up -> SHORT; above the OR high after a gap down -> LONG) and on the
  fill side of VWAP (when known);
* the previous close (the fill target) is still at least the mode's minimum R:R away.
Entry = LTP (LIMIT). SL = beyond the OR extreme (or day extreme) + ``sl_buffer_atr`` x ATR.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies.base import ParamSpec, Setup, Strategy, pick_stop, sign, strength_from, val
from strategies.intraday._common import TF5, above, atr_of, rvol_of, target_levels
from strategies.registry import register


@register
class GapFill(Strategy):
    name = "gap_fill"
    description = "Fade a gap that fails the opening range, targeting the previous close."
    modes = (Mode.INTRADAY,)
    regimes = frozenset({Regime.RANGING, Regime.UNDEFINED, Regime.TRENDING_UP, Regime.TRENDING_DOWN})
    countertrend = True
    windows = {Mode.INTRADAY: [("09:30", "11:30")]}
    param_specs = (
        ParamSpec("min_gap_pct", 0.8, 0.3, 5.0, "float", "Minimum absolute gap in %"),
        ParamSpec("max_gap_pct", 4.0, 1.0, 8.0, "float", "Maximum absolute gap in %"),
        ParamSpec("sl_buffer_atr", 0.2, 0.0, 1.0, "float", "SL buffer beyond the OR extreme in ATR"),
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
        # fade direction is opposite to the gap
        d = Direction.SHORT if gap > 0 else Direction.LONG
        s = sign(d)
        ltp = features.ltp
        failed_edge = orl if d == Direction.SHORT else orh
        far_edge = orh if d == Direction.SHORT else orl
        if (ltp - failed_edge) * s <= 0:
            return None  # gap not failing yet
        if (pc - ltp) * s <= 0:
            return None  # already filled
        vwap = val(features, TF5, "vwap")
        if vwap is not None and not above(ltp, vwap, d):
            return None
        day_ext = features.day_high if d == Direction.SHORT else features.day_low
        stop = pick_stop(
            d,
            ltp,
            [far_edge - s * params["sl_buffer_atr"] * atr, day_ext],
            atr,
            min_atr=0.3,
            max_atr=2.5,
            fallback_atr=1.5,
        )
        if stop is None:
            return None
        return Setup(
            direction=d,
            entry=ltp,
            stop=stop,
            entry_type=EntryType.LIMIT,
            target_levels=target_levels(features),
            natural_target=pc,
            strength=strength_from((rvol_of(features), 0.8, 2.0, 20.0), (abs(gap), params["min_gap_pct"], 3.0, 20.0)),
            notes=[f"gap {gap:+.2f}% failing; fill target prev close {pc:.2f}"],
        )
