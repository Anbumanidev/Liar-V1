"""EMA 9/21 Trend Pullback (INTRADAY, long + short).

Rules (5m primary, 15m confirmation):
* 5m EMA 9 beyond EMA 21 and 15m EMA 9 beyond EMA 21 (both in the trade direction);
* last closed 5m bar dipped to within ``touch_atr`` x ATR of EMA 9 (or into the 9-21 zone)
  without closing beyond EMA 21, and closed in the trend direction;
* ADX(14) on 5m or 15m >= ``min_adx`` (when known).
Entry = LTP (LIMIT). SL = EMA 21 -/+ ``sl_buffer_atr`` x ATR, or the bar extreme if farther.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime, TrailingMethod
from core.types import MarketContext, StockFeatures
from strategies.base import ParamSpec, Setup, Strategy, sign, strength_from, val
from strategies.intraday._common import DIRECTIONS, TF5, TF15, above, atr_of, ema_trend, target_levels
from strategies.registry import register


@register
class Ema921Pullback(Strategy):
    name = "ema_9_21_pullback"
    description = "Pullback into the 9/21 EMA zone in an aligned 5m/15m trend."
    modes = (Mode.INTRADAY,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.UNDEFINED})
    windows = {Mode.INTRADAY: [("09:45", "14:30")]}
    param_specs = (
        ParamSpec("touch_atr", 0.2, 0.0, 1.0, "float", "Bar must come within this many ATR of EMA 9"),
        ParamSpec("sl_buffer_atr", 0.3, 0.05, 1.5, "float", "SL buffer beyond EMA 21 in ATR"),
        ParamSpec("min_adx", 20.0, 10.0, 40.0, "float", "Minimum ADX on 5m/15m (ignored if unknown)"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        atr = atr_of(features)
        e9, e21 = val(features, TF5, "ema_9"), val(features, TF5, "ema_21")
        o, h, low, c = (val(features, TF5, k) for k in ("open", "high", "low", "close"))
        if atr is None or e9 is None or e21 is None or None in (o, h, low, c):
            return None
        adx = max([x for x in (val(features, TF5, "adx_14"), val(features, TF15, "adx_14")) if x is not None],
                  default=None)
        if adx is not None and adx < params["min_adx"]:
            return None
        for d in DIRECTIONS:
            s = sign(d)
            if not ema_trend(features, TF5, d) or not ema_trend(features, TF15, d):
                continue
            extreme = low if d == Direction.LONG else h
            if (extreme - e9) * s > params["touch_atr"] * atr:
                continue  # no pullback to the 9 EMA
            if not above(c, e21, d) or not above(c, o, d) or not above(features.ltp, e21, d):
                continue
            stop = min(e21 - s * params["sl_buffer_atr"] * atr, extreme - s * 0.1 * atr, key=lambda x: x * s)
            return Setup(
                direction=d,
                entry=features.ltp,
                stop=stop,
                entry_type=EntryType.LIMIT,
                target_levels=target_levels(features),
                strength=strength_from((adx, 20, 40, 35.0), (val(features, TF5, "rvol"), 0.8, 2.0, 15.0)),
                notes=[f"pullback to EMA9 {e9:.2f} (EMA21 {e21:.2f})"],
                trailing_rule=TrailingMethod.EMA9_5M,
            )
        return None
