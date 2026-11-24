"""Swing reversal with RSI divergence (SWING, LONG only).

Bullish reversal at support:
* daily bullish RSI divergence (``rsi_bull_div == 1``) and RSI(14) <= ``rsi_max``;
* the last bar's low is at or within ``near_support_atr`` x ATR of a known support level;
* reversal confirmation: a bullish candlestick pattern on the latest daily bars, or a bullish
  close (close > open) in the upper half of the bar's range.
Entry = close (LIMIT). SL = below min(bar low, support) by ``sl_buffer_atr`` x ATR.
Counter-trend by design (fits RANGING / TRENDING_DOWN regimes).
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime, Signal
from core.types import MarketContext, StockFeatures
from strategies._daily import TFD, atr_daily, daily_levels
from strategies.base import (
    CNC_SCAN_WINDOWS,
    ParamSpec,
    Setup,
    Strategy,
    latest_patterns,
    levels_beyond,
    pick_stop,
    strength_from,
    val,
)
from strategies.registry import register


def _next_res_note(levels: list[float], entry: float) -> str:
    above = levels_beyond(levels, entry, Direction.LONG)
    return f"next resistance {above[0]:.2f}" if above else "no resistance level known above entry"


@register
class SwingReversalDivergence(Strategy):
    name = "swing_reversal_divergence"
    description = "Bullish RSI divergence reversal at a daily support level."
    modes = (Mode.SWING,)
    regimes = frozenset({Regime.RANGING, Regime.TRENDING_DOWN, Regime.UNDEFINED})
    countertrend = True
    windows = {Mode.SWING: CNC_SCAN_WINDOWS}
    param_specs = (
        ParamSpec("rsi_max", 45.0, 30.0, 60.0, "float", "Maximum RSI(14) on the signal bar"),
        ParamSpec("near_support_atr", 1.0, 0.2, 3.0, "float", "Low within this many ATR of support"),
        ParamSpec("sl_buffer_atr", 0.5, 0.1, 2.0, "float", "SL buffer below the low/support in ATR"),
        ParamSpec("max_holding_days", 10, 2, 10, "int", "Maximum holding period (trading days)"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        div = val(features, TFD, "rsi_bull_div")
        rsi = val(features, TFD, "rsi_14")
        o, h, low, c = (val(features, TFD, k) for k in ("open", "high", "low", "close"))
        atr = atr_daily(features)
        if div != 1 or rsi is None or atr is None or None in (o, h, low, c):
            return None
        if rsi > params["rsi_max"]:
            return None
        tf = features.tf(TFD)
        supports: list[float] = []
        if tf is not None and tf.levels is not None:
            lv = tf.levels
            supports = [x for x in [*lv.supports, *lv.swing_lows, lv.s1, lv.s2, lv.prev_low] if x is not None]
        if features.week52_low is not None:
            supports.append(features.week52_low)
        near = [x for x in supports if abs(low - x) <= params["near_support_atr"] * atr]  # type: ignore[operator]
        if not near:
            return None
        support = min(near, key=lambda x: abs(low - x))  # type: ignore[operator]
        bull_patterns = latest_patterns(tf, Signal.BULLISH, lookback_bars=2)
        bullish_close = c > o and (c - low) >= 0.5 * (h - low)  # type: ignore[operator]
        if not bull_patterns and not bullish_close:
            return None
        entry = features.ltp
        stop = pick_stop(
            Direction.LONG,
            entry,
            [min(low, support) - params["sl_buffer_atr"] * atr],  # type: ignore[type-var]
            atr,
            min_atr=0.5,
            max_atr=3.0,
            fallback_atr=2.0,
        )
        if stop is None:
            return None
        levels = daily_levels(features)
        e50 = val(features, TFD, "ema_50")
        if e50 is not None:
            levels.append(e50)
        return Setup(
            direction=Direction.LONG,
            entry=entry,
            stop=stop,
            entry_type=EntryType.LIMIT,
            target_levels=levels,
            strength=strength_from((max(bull_patterns, default=0), 0, 100, 25.0),
                                   (-rsi, -params["rsi_max"], -25.0, 15.0), (features.rvol, 0.8, 2.0, 10.0)),
            notes=[f"bullish RSI divergence at support {support:.2f}, RSI {rsi:.1f}", _next_res_note(levels, entry)],
        )
