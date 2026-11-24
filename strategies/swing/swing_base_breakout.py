"""Swing base breakout (SWING, LONG only).

Daily breakout from a base / consolidation with volume:
* base = prior 20-day range (``extra['range_high_20'/'range_low_20']``) or daily swing extremes;
* base depth ``(high - low) / high`` <= ``max_base_depth_pct`` (a real consolidation);
* last daily close above the base high by >= ``buffer_pct`` %, but not more than
  ``max_extension_atr`` x ATR past it (no chasing extended moves);
* relative volume >= ``min_rvol``; close above EMA 50 (uptrend context, when known).
Entry = close (LIMIT for the next session). SL = base high - ``sl_atr`` x ATR, not below the
base low. Targets: levels above / measured move (base high + base height) / R multiples.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies._daily import TFD, atr_daily, base_range, daily_levels
from strategies.base import CNC_SCAN_WINDOWS, ParamSpec, Setup, Strategy, pick_stop, strength_from, val
from strategies.registry import register


@register
class SwingBaseBreakout(Strategy):
    name = "swing_base_breakout"
    description = "Daily close breaking out of a tight base on above-average volume."
    modes = (Mode.SWING,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.RANGING, Regime.UNDEFINED})
    windows = {Mode.SWING: CNC_SCAN_WINDOWS}
    param_specs = (
        ParamSpec("max_base_depth_pct", 15.0, 4.0, 30.0, "float", "Maximum base depth in %"),
        ParamSpec("buffer_pct", 0.3, 0.0, 3.0, "float", "Close above the base high by this %"),
        ParamSpec("max_extension_atr", 1.5, 0.3, 4.0, "float", "Skip if close is further past the base (ATR)"),
        ParamSpec("min_rvol", 1.5, 1.0, 5.0, "float", "Minimum relative volume"),
        ParamSpec("sl_atr", 1.0, 0.3, 3.0, "float", "SL distance below the base high in ATR"),
        ParamSpec("max_holding_days", 10, 2, 10, "int", "Maximum holding period (trading days)"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        hi, lo = base_range(features)
        atr = atr_daily(features)
        close = val(features, TFD, "close") or features.ltp
        if hi is None or lo is None or atr is None or hi <= lo:
            return None
        depth = (hi - lo) / hi * 100
        if depth > params["max_base_depth_pct"]:
            return None
        if close < hi * (1 + params["buffer_pct"] / 100) or (close - hi) > params["max_extension_atr"] * atr:
            return None
        rvol = features.rvol if features.rvol is not None else val(features, TFD, "rvol")
        if rvol is None or rvol < params["min_rvol"]:
            return None
        ema50 = val(features, TFD, "ema_50")
        if ema50 is not None and close <= ema50:
            return None
        entry = features.ltp
        stop = pick_stop(
            Direction.LONG,
            entry,
            [max(hi - params["sl_atr"] * atr, lo)],
            atr,
            min_atr=0.5,
            max_atr=3.0,
            fallback_atr=2.0,
        )
        if stop is None:
            return None
        measured = hi + (hi - lo)
        return Setup(
            direction=Direction.LONG,
            entry=entry,
            stop=stop,
            entry_type=EntryType.LIMIT,
            target_levels=[*daily_levels(features), measured],
            strength=strength_from((rvol, 1.2, 3.0, 30.0), (-depth, -params["max_base_depth_pct"], -5.0, 20.0)),
            notes=[f"breakout above base {lo:.2f}-{hi:.2f} (depth {depth:.1f}%), rvol {rvol:.2f}"],
        )
