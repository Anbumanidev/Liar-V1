"""Opening Range Breakout (INTRADAY, long + short).

Rules (5m primary):
* opening range (09:15-09:30) known and not wider than ``max_or_width_atr`` x ATR;
* last closed 5m bar closed beyond the OR high (low) by ``buffer_atr`` x ATR;
* price has not run more than ``max_extension_atr`` x ATR past the range (no chasing);
* relative volume >= ``min_rvol`` and price on the breakout side of VWAP (if known).
Entry = LTP (MARKET). SL = opposite side of the OR, or the OR midpoint when the full range
would exceed ``max_sl_atr`` x ATR. Targets via mode minimum R:R / next levels.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies.base import ParamSpec, Setup, Strategy, pick_stop, sign, strength_from, val
from strategies.intraday._common import DIRECTIONS, TF5, above, atr_of, rvol_of, target_levels
from strategies.registry import register


@register
class OpeningRangeBreakout(Strategy):
    name = "opening_range_breakout"
    description = "Breakout of the 09:15-09:30 opening range with volume, VWAP-aligned."
    modes = (Mode.INTRADAY,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.UNDEFINED, Regime.HIGH_VOLATILITY})
    windows = {Mode.INTRADAY: [("09:30", "11:30")]}
    param_specs = (
        ParamSpec("min_rvol", 1.5, 1.0, 5.0, "float", "Minimum relative volume"),
        ParamSpec("buffer_atr", 0.05, 0.0, 0.5, "float", "Close beyond the range by this many ATR"),
        ParamSpec("max_or_width_atr", 4.0, 1.0, 8.0, "float", "Skip if the opening range is wider (ATR)"),
        ParamSpec("max_extension_atr", 1.0, 0.2, 3.0, "float", "Skip if price ran further past the range (ATR)"),
        ParamSpec("max_sl_atr", 2.0, 0.5, 3.0, "float", "Use the OR midpoint when the full range is wider (ATR)"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        orh, orl = features.opening_range_high, features.opening_range_low
        atr = atr_of(features)
        close = val(features, TF5, "close")
        if orh is None or orl is None or atr is None or close is None or orh <= orl:
            return None
        if (orh - orl) > params["max_or_width_atr"] * atr:
            return None
        rvol = rvol_of(features)
        if rvol is None or rvol < params["min_rvol"]:
            return None
        vwap = val(features, TF5, "vwap")
        ltp = features.ltp
        for d in DIRECTIONS:
            s = sign(d)
            edge, other = (orh, orl) if d == Direction.LONG else (orl, orh)
            if (close - edge) * s < params["buffer_atr"] * atr or (ltp - edge) * s <= 0:
                continue
            if (ltp - edge) * s > params["max_extension_atr"] * atr:
                continue
            if vwap is not None and not above(ltp, vwap, d):
                continue
            mid = (orh + orl) / 2
            stop = pick_stop(
                d, ltp, [other, mid], atr, min_atr=0.3, max_atr=params["max_sl_atr"], fallback_atr=params["max_sl_atr"]
            )
            if stop is None:
                continue
            return Setup(
                direction=d,
                entry=ltp,
                stop=stop,
                entry_type=EntryType.MARKET,
                target_levels=target_levels(features),
                strength=strength_from((rvol, 1.0, 3.0, 35.0), (val(features, "15m", "adx_14"), 15, 35, 25.0)),
                notes=[f"OR {orl:.2f}-{orh:.2f} broken {'up' if s > 0 else 'down'}, rvol {rvol:.2f}"],
            )
        return None
