"""Swing EMA pullback (SWING, LONG only).

Pullback to the 20/50 EMA in an established daily uptrend:
* EMA 20 > EMA 50 (and EMA 50 > EMA 200 when known), close above EMA 50;
* the last daily bar's low came within ``touch_pct`` % of EMA 20 or EMA 50 (or pierced it)
  and the bar closed back above that EMA as a bullish candle;
* RSI(14) between ``rsi_min`` and ``rsi_max`` (a pullback, not a breakdown / blow-off).
Entry = close (LIMIT). SL = below the bar low / the touched EMA by ``sl_buffer_atr`` x ATR.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies._daily import TFD, atr_daily, daily_levels
from strategies.base import CNC_SCAN_WINDOWS, ParamSpec, Setup, Strategy, pick_stop, strength_from, val
from strategies.registry import register


@register
class SwingEmaPullback(Strategy):
    name = "swing_ema_pullback"
    description = "Bullish pullback to the daily 20/50 EMA inside an uptrend."
    modes = (Mode.SWING,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.UNDEFINED, Regime.RANGING})
    windows = {Mode.SWING: CNC_SCAN_WINDOWS}
    param_specs = (
        ParamSpec("touch_pct", 1.0, 0.1, 3.0, "float", "Low within this % of the EMA counts as a touch"),
        ParamSpec("rsi_min", 40.0, 25.0, 55.0, "float", "Minimum RSI(14)"),
        ParamSpec("rsi_max", 65.0, 55.0, 80.0, "float", "Maximum RSI(14)"),
        ParamSpec("sl_buffer_atr", 0.5, 0.1, 2.0, "float", "SL buffer below the bar low / EMA in ATR"),
        ParamSpec("max_holding_days", 10, 2, 10, "int", "Maximum holding period (trading days)"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        e20, e50, e200 = (val(features, TFD, k) for k in ("ema_20", "ema_50", "ema_200"))
        o, low, c = (val(features, TFD, k) for k in ("open", "low", "close"))
        rsi = val(features, TFD, "rsi_14")
        atr = atr_daily(features)
        if None in (e20, e50, o, low, c, rsi, atr):
            return None
        if not (e20 > e50 and c > e50) or (e200 is not None and e50 <= e200):  # type: ignore[operator]
            return None
        if not (params["rsi_min"] <= rsi <= params["rsi_max"]) or c <= o:  # type: ignore[operator]
            return None
        touched = None
        for ema in (e20, e50):
            if low <= ema * (1 + params["touch_pct"] / 100) and c > ema:  # type: ignore[operator]
                touched = ema
                break
        if touched is None:
            return None
        entry = features.ltp
        buf = params["sl_buffer_atr"] * atr  # type: ignore[operator]
        stop = pick_stop(
            Direction.LONG, entry, [min(low, touched) - buf], atr, min_atr=0.5, max_atr=3.0, fallback_atr=2.0  # type: ignore[arg-type]
        )
        if stop is None:
            return None
        return Setup(
            direction=Direction.LONG,
            entry=entry,
            stop=stop,
            entry_type=EntryType.LIMIT,
            target_levels=daily_levels(features),
            strength=strength_from((val(features, TFD, "adx_14"), 15, 35, 25.0),
                                   (features.rvol, 0.7, 1.5, 15.0), ((e20 - e50) / e50 * 100, 0.5, 5.0, 10.0)),  # type: ignore[operator]
            notes=[f"pullback to EMA{'20' if touched == e20 else '50'} {touched:.2f}, RSI {rsi:.1f}"],
        )
