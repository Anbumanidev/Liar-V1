"""Positional 52-week-high breakout (POSITIONAL, LONG only).

* the daily close is at / within ``near_high_pct`` % of the 52-week high (the 52-week high
  includes the latest closed bar, so a breakout day closes at or right below it);
* relative volume >= ``min_rvol``;
* trend: close > EMA 50 and EMA 50 > EMA 200 (when known);
* not over-extended: close no more than ``max_ext_ema50_atr`` x ATR above EMA 50.
Entry = close (LIMIT). SL = ``sl_atr`` x daily ATR (or the previous base high when known and
within bounds). No overhead resistance, so targets are R multiples unless weekly levels exist.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies._daily import TFD, atr_daily, base_range, daily_levels
from strategies.base import CNC_SCAN_WINDOWS, ParamSpec, Setup, Strategy, pick_stop, strength_from, val
from strategies.registry import register


@register
class Positional52wBreakout(Strategy):
    name = "positional_52w_breakout"
    description = "Close at a new 52-week high on strong volume in a long-term uptrend."
    modes = (Mode.POSITIONAL,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.UNDEFINED, Regime.HIGH_VOLATILITY})
    windows = {Mode.POSITIONAL: CNC_SCAN_WINDOWS}
    param_specs = (
        ParamSpec("near_high_pct", 1.0, 0.0, 3.0, "float", "Close within this % of the 52-week high"),
        ParamSpec("min_rvol", 1.5, 1.0, 5.0, "float", "Minimum relative volume"),
        ParamSpec("max_ext_ema50_atr", 6.0, 2.0, 12.0, "float", "Max distance above EMA 50 in ATR"),
        ParamSpec("sl_atr", 2.0, 1.0, 3.0, "float", "SL distance in daily ATR"),
        ParamSpec("max_holding_days", 63, 10, 63, "int", "Maximum holding period (trading days)"),
        ParamSpec("t2_r", 4.0, 2.5, 8.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        hi52 = features.week52_high
        c = val(features, TFD, "close") or features.ltp
        e50, e200 = val(features, TFD, "ema_50"), val(features, TFD, "ema_200")
        atr = atr_daily(features)
        if hi52 is None or atr is None or e50 is None or hi52 <= 0:
            return None
        if (hi52 - c) / hi52 * 100 > params["near_high_pct"]:
            return None
        rvol = features.rvol if features.rvol is not None else val(features, TFD, "rvol")
        if rvol is None or rvol < params["min_rvol"]:
            return None
        if c <= e50 or (e200 is not None and e50 <= e200):
            return None
        if (c - e50) > params["max_ext_ema50_atr"] * atr:
            return None
        base_hi, _ = base_range(features)
        entry = features.ltp
        cands = [base_hi - 0.5 * atr] if base_hi is not None and base_hi < entry else []
        stop = pick_stop(
            Direction.LONG, entry, cands, atr, min_atr=1.0, max_atr=params["sl_atr"], fallback_atr=params["sl_atr"]
        )
        if stop is None:
            return None
        return Setup(
            direction=Direction.LONG,
            entry=entry,
            stop=stop,
            entry_type=EntryType.LIMIT,
            target_levels=[x for x in daily_levels(features) if abs(x - hi52) > 1e-6],
            strength=strength_from((rvol, 1.2, 3.0, 35.0), (features.rs_vs_nifty, 0, 20, 15.0)),
            notes=[f"52-week high breakout ({hi52:.2f}), rvol {rvol:.2f}"],
        )
