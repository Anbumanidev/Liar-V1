"""BTST strong close (BTST, LONG only).

Buy-today-sell-tomorrow candidates scanned 14:30-15:20 (section 10):
* the stock closes near the day's high: ``(ltp - day_low) / (day_high - day_low) >= min_close_pos``;
* day change vs previous close between ``min_change_pct`` and ``max_change_pct`` (avoid stocks
  near the upper circuit);
* relative volume >= ``min_rvol``;
* delivery % >= ``min_delivery_ratio`` x its 20-day average (when known; required only if
  ``require_delivery``) - note that live delivery data is usually the previous session's;
* sector strength >= ``min_sector_change_pct`` (when the sector is known);
* positive market: NIFTY change >= ``min_market_change_pct`` (or, if unknown, NIFTY trend not
  bearish). A market context is required.
Entry = LTP (LIMIT). SL = ``sl_atr`` x daily ATR below the entry, never below the day low.
Exit next trading day (max_holding_days = 1), no trailing.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime, Signal, TrailingMethod
from core.types import MarketContext, StockFeatures
from strategies._daily import TFD, atr_daily, daily_levels, sector_strength
from strategies.base import ParamSpec, Setup, Strategy, pick_stop, strength_from, val
from strategies.registry import register


@register
class BtstStrongClose(Strategy):
    name = "btst_strong_close"
    description = "Close near the day high on above-average volume/delivery in a strong sector and market."
    modes = (Mode.BTST,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.UNDEFINED, Regime.RANGING})
    windows = {Mode.BTST: [("14:30", "15:20")]}
    param_specs = (
        ParamSpec("min_close_pos", 0.8, 0.5, 0.98, "float", "Min close position in the day range (0-1)"),
        ParamSpec("min_change_pct", 1.0, 0.2, 5.0, "float", "Min % change vs previous close"),
        ParamSpec("max_change_pct", 6.0, 2.0, 15.0, "float", "Max % change (circuit/exhaustion guard)"),
        ParamSpec("min_rvol", 1.3, 1.0, 5.0, "float", "Minimum relative volume"),
        ParamSpec("min_delivery_ratio", 1.0, 0.5, 3.0, "float", "Delivery % vs its 20-day average"),
        ParamSpec("require_delivery", False, kind="bool", description="Skip when delivery data is missing"),
        ParamSpec("min_sector_change_pct", 0.0, -2.0, 3.0, "float", "Min sector strength (%)"),
        ParamSpec("min_market_change_pct", 0.0, -1.0, 2.0, "float", "Min NIFTY % change"),
        ParamSpec("sl_atr", 0.6, 0.3, 1.5, "float", "SL distance in daily ATR"),
        ParamSpec("t2_r", 3.0, 2.0, 5.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        if market is None:
            return None
        hi, lo, pc, ltp = features.day_high, features.day_low, features.prev_close, features.ltp
        atr = atr_daily(features)
        if None in (hi, lo, pc, atr) or hi <= lo or pc <= 0:  # type: ignore[operator]
            return None
        close_pos = (ltp - lo) / (hi - lo)
        change = (ltp - pc) / pc * 100
        if close_pos < params["min_close_pos"]:
            return None
        if not (params["min_change_pct"] <= change <= params["max_change_pct"]):
            return None
        rvol = features.rvol if features.rvol is not None else val(features, TFD, "rvol")
        if rvol is None or rvol < params["min_rvol"]:
            return None
        notes = [f"close at {close_pos:.0%} of range, {change:+.2f}%, rvol {rvol:.2f}"]
        dp, dpa = features.delivery_pct, features.delivery_pct_avg_20
        if dp is not None and dpa is not None and dpa > 0:
            if dp < params["min_delivery_ratio"] * dpa:
                return None
            notes.append(f"delivery {dp:.1f}% vs avg {dpa:.1f}%")
        elif params["require_delivery"]:
            return None
        else:
            notes.append("delivery % unknown")
        sec = sector_strength(features, market)
        if sec is not None and sec < params["min_sector_change_pct"]:
            return None
        if market.nifty_change_pct is not None:
            if market.nifty_change_pct < params["min_market_change_pct"]:
                return None
        elif market.nifty_trend == Signal.BEARISH:
            return None
        stop = pick_stop(
            Direction.LONG, ltp, [], atr, min_atr=0.3, max_atr=3.0, fallback_atr=params["sl_atr"]
        )
        if stop is None:
            return None
        stop = max(stop, lo - 0.05 * atr) if lo < ltp - 0.3 * atr else stop
        return Setup(
            direction=Direction.LONG,
            entry=ltp,
            stop=stop,
            entry_type=EntryType.LIMIT,
            target_levels=daily_levels(features),
            strength=strength_from((close_pos, params["min_close_pos"], 1.0, 25.0), (rvol, 1.0, 2.5, 20.0),
                                   (sec, 0.0, 2.0, 15.0)),
            notes=notes,
            trailing_rule=TrailingMethod.NONE,
        )
