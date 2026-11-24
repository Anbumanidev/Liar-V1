"""VWAP Pullback (INTRADAY, long + short).

Rules (5m primary, 15m confirmation):
* trend: 5m EMA 9 beyond EMA 21 and price on the trend side of VWAP; 15m EMAs not opposing;
* pullback: the last closed 5m bar traded back to within ``touch_atr`` x ATR of VWAP ...
* ... and closed back on the trend side of VWAP with a candle in the trend direction;
* ADX(14) on 15m >= ``min_adx`` (when known).
Entry = LTP (LIMIT). SL = beyond the bar extreme / ``VWAP -/+ sl_atr`` x ATR (whichever is
farther). Trailing: 9 EMA on 5m.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime, TrailingMethod
from core.types import MarketContext, StockFeatures
from strategies.base import ParamSpec, Setup, Strategy, sign, strength_from, val
from strategies.intraday._common import DIRECTIONS, TF5, TF15, above, atr_of, ema_trend, rvol_of, target_levels
from strategies.registry import register


@register
class VwapPullback(Strategy):
    name = "vwap_pullback"
    description = "Trend-day pullback to VWAP that holds, entered in the trend direction."
    modes = (Mode.INTRADAY,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.UNDEFINED})
    windows = {Mode.INTRADAY: [("09:45", "14:30")]}
    param_specs = (
        ParamSpec("touch_atr", 0.3, 0.05, 1.0, "float", "Bar must come within this many ATR of VWAP"),
        ParamSpec("sl_atr", 0.5, 0.1, 1.5, "float", "SL distance beyond VWAP in ATR"),
        ParamSpec("min_adx", 18.0, 10.0, 40.0, "float", "Minimum 15m ADX (ignored if unknown)"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        atr = atr_of(features)
        vwap = val(features, TF5, "vwap")
        o, h, low, c = (val(features, TF5, k) for k in ("open", "high", "low", "close"))
        if atr is None or vwap is None or None in (o, h, low, c):
            return None
        adx15 = val(features, TF15, "adx_14")
        if adx15 is not None and adx15 < params["min_adx"]:
            return None
        for d in DIRECTIONS:
            s = sign(d)
            if not ema_trend(features, TF5, d) or ema_trend(features, TF15, d) is False:
                continue
            if not (above(c, vwap, d) and above(features.ltp, vwap, d) and above(c, o, d)):
                continue
            extreme = low if d == Direction.LONG else h
            if (extreme - vwap) * s > params["touch_atr"] * atr:
                continue  # never came back to VWAP
            if (c - vwap) * s > 1.5 * atr:
                continue  # closed too far away: not a pullback entry any more
            stop = min(extreme - 0.1 * atr * s, vwap - params["sl_atr"] * atr * s, key=lambda x: x * s)
            return Setup(
                direction=d,
                entry=features.ltp,
                stop=stop,
                entry_type=EntryType.LIMIT,
                target_levels=target_levels(features),
                strength=strength_from((adx15, 18, 35, 30.0), (rvol_of(features), 0.8, 2.0, 20.0)),
                notes=[f"pullback to VWAP {vwap:.2f} held"],
                trailing_rule=TrailingMethod.EMA9_5M,
            )
        return None
