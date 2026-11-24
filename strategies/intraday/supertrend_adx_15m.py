"""Supertrend + ADX on 15m (INTRADAY, long + short).

Rules (15m):
* Supertrend(10, 3) direction +1 (-1) and price beyond the Supertrend line;
* ADX(14) >= ``min_adx`` with +DI beyond -DI in the trade direction;
* price not more than ``max_dist_atr`` x ATR(15m) away from the Supertrend line (no chasing);
* 5m price on the trend side of VWAP (when known).
Entry = LTP (MARKET). SL = Supertrend line -/+ ``sl_buffer_atr`` x ATR(15m), falling back to
an ATR stop when the line is too far. Trailing: ATR.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime, TrailingMethod
from core.types import MarketContext, StockFeatures
from strategies.base import ParamSpec, Setup, Strategy, pick_stop, sign, strength_from, val
from strategies.intraday._common import DIRECTIONS, TF5, TF15, above, atr_of, target_levels
from strategies.registry import register


@register
class SupertrendAdx15m(Strategy):
    name = "supertrend_adx_15m"
    description = "15m Supertrend direction confirmed by a strong ADX and DI alignment."
    modes = (Mode.INTRADAY,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.UNDEFINED})
    windows = {Mode.INTRADAY: [("09:45", "14:30")]}
    param_specs = (
        ParamSpec("min_adx", 25.0, 15.0, 45.0, "float", "Minimum ADX(14) on 15m"),
        ParamSpec("max_dist_atr", 2.0, 0.5, 4.0, "float", "Max distance from the Supertrend line (15m ATR)"),
        ParamSpec("sl_buffer_atr", 0.1, 0.0, 1.0, "float", "SL buffer beyond the Supertrend line (15m ATR)"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        st, st_dir = val(features, TF15, "supertrend"), val(features, TF15, "supertrend_dir")
        adx, pdi, mdi = (val(features, TF15, k) for k in ("adx_14", "plus_di", "minus_di"))
        atr15 = val(features, TF15, "atr_14")
        if atr15 is None or atr15 <= 0:
            atr15 = atr_of(features)
        if None in (st, st_dir, adx, pdi, mdi, atr15) or adx < params["min_adx"]:  # type: ignore[operator]
            return None
        ltp = features.ltp
        vwap = val(features, TF5, "vwap")
        for d in DIRECTIONS:
            s = sign(d)
            if st_dir * s <= 0 or (pdi - mdi) * s <= 0 or not above(ltp, st, d):
                continue
            if (ltp - st) * s > params["max_dist_atr"] * atr15:
                continue
            if vwap is not None and not above(ltp, vwap, d):
                continue
            # SL bounds are checked against the primary (5m) ATR by the base class; use 15m ATR
            # for the structural buffer only.
            atr5 = atr_of(features) or atr15
            stop = pick_stop(
                d, ltp, [st - s * params["sl_buffer_atr"] * atr15], atr5, min_atr=0.3, max_atr=3.0, fallback_atr=1.5
            )
            if stop is None:
                continue
            return Setup(
                direction=d,
                entry=ltp,
                stop=stop,
                entry_type=EntryType.MARKET,
                target_levels=target_levels(features),
                strength=strength_from((adx, params["min_adx"], 45, 35.0), (abs(pdi - mdi), 5, 25, 15.0)),
                notes=[f"15m Supertrend {st:.2f} dir {int(st_dir):+d}, ADX {adx:.1f}"],
                trailing_rule=TrailingMethod.ATR,
            )
        return None
