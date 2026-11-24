"""Mean Reversion (INTRADAY, long + short) - RANGING regime only.

Rules (5m primary):
* regime is RANGING (strict: silent in every other regime);
* last closed bar pierced the lower (upper) Bollinger band and RSI(14) is oversold
  (overbought) beyond ``rsi_extreme``;
* the bar closed back inside the band in the reversal direction;
* the Bollinger midline (the mean) is at least the mode's minimum R:R away.
Entry = LTP (LIMIT). SL = beyond the bar extreme by ``sl_buffer_atr`` x ATR.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies.base import ParamSpec, Setup, Strategy, sign, strength_from, val
from strategies.intraday._common import DIRECTIONS, TF5, above, atr_of, target_levels
from strategies.registry import register


@register
class MeanReversion(Strategy):
    name = "mean_reversion"
    description = "Bollinger band + RSI extreme fade back to the mean in a ranging market."
    modes = (Mode.INTRADAY,)
    regimes = frozenset({Regime.RANGING})
    strict_regime = True
    countertrend = True
    windows = {Mode.INTRADAY: [("10:00", "14:15")]}
    param_specs = (
        ParamSpec("rsi_oversold", 30.0, 15.0, 40.0, "float", "RSI at or below = oversold (longs)"),
        ParamSpec("rsi_overbought", 70.0, 60.0, 85.0, "float", "RSI at or above = overbought (shorts)"),
        ParamSpec("sl_buffer_atr", 0.3, 0.05, 1.0, "float", "SL buffer beyond the bar extreme in ATR"),
        ParamSpec("t2_r", 2.5, 2.0, 5.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        atr = atr_of(features)
        rsi = val(features, TF5, "rsi_14")
        upper, mid, lower = (val(features, TF5, k) for k in ("bb_upper", "bb_mid", "bb_lower"))
        o, h, low, c = (val(features, TF5, k) for k in ("open", "high", "low", "close"))
        if atr is None or rsi is None or None in (upper, mid, lower, o, h, low, c):
            return None
        for d in DIRECTIONS:
            s = sign(d)
            band = lower if d == Direction.LONG else upper
            extreme = low if d == Direction.LONG else h
            if d == Direction.LONG and rsi > params["rsi_oversold"]:
                continue
            if d == Direction.SHORT and rsi < params["rsi_overbought"]:
                continue
            if (extreme - band) * s > 0:
                continue  # band not pierced
            if not above(c, band, d) or not above(c, o, d):
                continue  # no close back inside in the reversal direction
            if not above(mid, features.ltp, d):
                continue  # already at/through the mean
            stop = extreme - s * params["sl_buffer_atr"] * atr
            return Setup(
                direction=d,
                entry=features.ltp,
                stop=stop,
                entry_type=EntryType.LIMIT,
                target_levels=target_levels(features),
                natural_target=mid,
                strength=strength_from(
                    (abs(rsi - 50), 20, 35, 35.0), (val(features, TF5, "bb_width_pct"), 0.5, 3.0, 15.0)
                ),
                notes=[f"BB {'lower' if s > 0 else 'upper'} band fade, RSI {rsi:.1f}, mean {mid:.2f}"],
            )
        return None
