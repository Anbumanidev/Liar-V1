"""Relative Strength Momentum (INTRADAY, long + short).

Rules:
* the stock outperforms (underperforms) NIFTY by at least ``min_rs_pct`` (``rs_vs_nifty``);
* price on the trend side of VWAP with EMA 9/21 aligned on 5m and 15m;
* price is near the day's high (low): within ``near_extreme_atr`` x ATR;
* relative volume >= ``min_rvol``; the market is not strongly against the trade (NIFTY trend
  not opposite, when known).
Entry = LTP (MARKET). SL = ``sl_atr`` x ATR, tightened to just beyond VWAP when VWAP is closer
(but never inside the minimum SL distance).
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime, Signal
from core.types import MarketContext, StockFeatures
from strategies.base import ParamSpec, Setup, Strategy, pick_stop, sign, strength_from, val
from strategies.intraday._common import DIRECTIONS, TF5, TF15, above, atr_of, ema_trend, rvol_of, target_levels
from strategies.registry import register


@register
class RelativeStrengthMomentum(Strategy):
    name = "relative_strength_momentum"
    description = "Momentum continuation in stocks out/under-performing NIFTY, near the day extreme."
    modes = (Mode.INTRADAY,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.UNDEFINED})
    windows = {Mode.INTRADAY: [("10:00", "14:00")]}
    param_specs = (
        ParamSpec("min_rs_pct", 1.0, 0.2, 5.0, "float", "Minimum % out/under-performance vs NIFTY"),
        ParamSpec("near_extreme_atr", 0.75, 0.1, 2.0, "float", "Max distance from day high/low in ATR"),
        ParamSpec("min_rvol", 1.2, 0.8, 5.0, "float", "Minimum relative volume"),
        ParamSpec("sl_atr", 1.2, 0.5, 2.5, "float", "Fallback SL distance in ATR"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        rs, atr = features.rs_vs_nifty, atr_of(features)
        vwap = val(features, TF5, "vwap")
        if rs is None or atr is None or abs(rs) < params["min_rs_pct"]:
            return None
        rvol = rvol_of(features)
        if rvol is None or rvol < params["min_rvol"]:
            return None
        d = Direction.LONG if rs > 0 else Direction.SHORT
        if d not in DIRECTIONS:
            return None
        s = sign(d)
        ltp = features.ltp
        if market is not None:
            opposite = Signal.BEARISH if d == Direction.LONG else Signal.BULLISH
            if market.nifty_trend == opposite:
                return None
        if vwap is not None and not above(ltp, vwap, d):
            return None
        if not ema_trend(features, TF5, d) or ema_trend(features, TF15, d) is False:
            return None
        extreme = features.day_high if d == Direction.LONG else features.day_low
        if extreme is None or (extreme - ltp) * s > params["near_extreme_atr"] * atr:
            return None
        vwap_stop = vwap - s * 0.2 * atr if vwap is not None else None
        stop = pick_stop(
            d, ltp, [vwap_stop], atr, min_atr=0.5, max_atr=params["sl_atr"], fallback_atr=params["sl_atr"]
        )
        if stop is None:
            return None
        return Setup(
            direction=d,
            entry=ltp,
            stop=stop,
            entry_type=EntryType.MARKET,
            target_levels=target_levels(features),
            strength=strength_from((abs(rs), params["min_rs_pct"], 3.0, 30.0), (rvol, 1.0, 2.5, 25.0)),
            notes=[f"RS vs NIFTY {rs:+.2f}%, near day {'high' if s > 0 else 'low'}"],
        )
