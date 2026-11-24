"""Swing relative-strength leader (SWING, LONG only).

Leaders vs NIFTY:
* ``rs_vs_nifty`` >= ``min_rs_pct`` (% outperformance over the lookback);
* daily trend: close > EMA 20 > EMA 50, ADX(14) >= ``min_adx`` (when known);
* RSI(14) between ``rsi_min`` and ``rsi_max`` (strong but not blown off);
* close within ``near_high_pct`` % of the 52-week high (leaders make new highs);
* the NIFTY trend is not BEARISH (when known).
Entry = close (LIMIT). SL = EMA 20 - ``sl_buffer_atr`` x ATR when within bounds, else
``sl_atr`` x ATR.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies._daily import TFD, atr_daily, daily_levels, market_not_bearish
from strategies.base import CNC_SCAN_WINDOWS, ParamSpec, Setup, Strategy, pick_stop, strength_from, val
from strategies.registry import register


@register
class SwingRsLeader(Strategy):
    name = "swing_rs_leader"
    description = "Relative-strength leaders vs NIFTY trading near their highs in a daily uptrend."
    modes = (Mode.SWING,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.UNDEFINED})
    windows = {Mode.SWING: CNC_SCAN_WINDOWS}
    param_specs = (
        ParamSpec("min_rs_pct", 5.0, 1.0, 20.0, "float", "Minimum % outperformance vs NIFTY"),
        ParamSpec("min_adx", 20.0, 10.0, 40.0, "float", "Minimum daily ADX(14) (ignored if unknown)"),
        ParamSpec("rsi_min", 55.0, 45.0, 65.0, "float", "Minimum RSI(14)"),
        ParamSpec("rsi_max", 75.0, 65.0, 85.0, "float", "Maximum RSI(14)"),
        ParamSpec("near_high_pct", 5.0, 1.0, 15.0, "float", "Close within this % of the 52-week high"),
        ParamSpec("sl_buffer_atr", 0.5, 0.1, 2.0, "float", "SL buffer below EMA 20 in ATR"),
        ParamSpec("sl_atr", 2.0, 1.0, 3.0, "float", "Fallback SL distance in ATR"),
        ParamSpec("max_holding_days", 10, 2, 10, "int", "Maximum holding period (trading days)"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        rs = features.rs_vs_nifty
        e20, e50 = val(features, TFD, "ema_20"), val(features, TFD, "ema_50")
        rsi, adx = val(features, TFD, "rsi_14"), val(features, TFD, "adx_14")
        c = val(features, TFD, "close") or features.ltp
        atr = atr_daily(features)
        hi52 = features.week52_high
        if None in (rs, e20, e50, rsi, atr, hi52) or rs < params["min_rs_pct"]:  # type: ignore[operator]
            return None
        if not (c > e20 > e50):  # type: ignore[operator]
            return None
        if adx is not None and adx < params["min_adx"]:
            return None
        if not (params["rsi_min"] <= rsi <= params["rsi_max"]):  # type: ignore[operator]
            return None
        if (hi52 - c) / hi52 * 100 > params["near_high_pct"]:  # type: ignore[operator]
            return None
        if not market_not_bearish(market):
            return None
        entry = features.ltp
        stop = pick_stop(
            Direction.LONG,
            entry,
            [e20 - params["sl_buffer_atr"] * atr],  # type: ignore[operator]
            atr,  # type: ignore[arg-type]
            min_atr=0.75,
            max_atr=params["sl_atr"],
            fallback_atr=params["sl_atr"],
        )
        if stop is None:
            return None
        return Setup(
            direction=Direction.LONG,
            entry=entry,
            stop=stop,
            entry_type=EntryType.LIMIT,
            target_levels=daily_levels(features),
            strength=strength_from((rs, params["min_rs_pct"], 20.0, 30.0), (adx, 20, 40, 20.0)),
            notes=[f"RS vs NIFTY {rs:+.1f}%, {((hi52 - c) / hi52 * 100):.1f}% below 52w high"],  # type: ignore[operator]
        )
