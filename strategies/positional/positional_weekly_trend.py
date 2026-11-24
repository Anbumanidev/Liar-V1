"""Positional weekly trend (POSITIONAL, LONG only).

Weekly trend + fundamentals:
* weekly EMA 20 > EMA 50 and weekly close > EMA 20;
* weekly Supertrend direction +1 or weekly trend structure +1 (HH/HL), weekly ADX >=
  ``min_adx`` (when known);
* daily close > daily EMA 50 (trend intact on the entry timeframe);
* fundamentals score (``extra['fundamental_score']``, 0-100) >= ``min_fundamental_score``
  when present; required only if ``require_fundamentals``.
Entry = close (LIMIT). SL = weekly EMA 20 / daily EMA 50 structure when within
``sl_max_atr`` daily ATR, else ``sl_atr`` x daily ATR. Trailing: ATR on daily close.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies._daily import TFD, TFW, atr_daily, daily_levels, fundamental_score
from strategies.base import CNC_SCAN_WINDOWS, ParamSpec, Setup, Strategy, pick_stop, strength_from, val
from strategies.registry import register


@register
class PositionalWeeklyTrend(Strategy):
    name = "positional_weekly_trend"
    description = "Weekly uptrend (EMA/Supertrend/structure) with supportive fundamentals."
    modes = (Mode.POSITIONAL,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.UNDEFINED, Regime.RANGING})
    windows = {Mode.POSITIONAL: CNC_SCAN_WINDOWS}
    param_specs = (
        ParamSpec("min_adx", 20.0, 10.0, 40.0, "float", "Minimum weekly ADX(14) (ignored if unknown)"),
        ParamSpec("min_fundamental_score", 60.0, 0.0, 100.0, "float", "Minimum fundamentals score"),
        ParamSpec("require_fundamentals", False, kind="bool", description="Skip when no fundamentals score"),
        ParamSpec("sl_max_atr", 3.0, 1.0, 3.0, "float", "Max structural SL distance in daily ATR"),
        ParamSpec("sl_atr", 2.5, 1.0, 3.0, "float", "Fallback SL distance in daily ATR"),
        ParamSpec("max_holding_days", 63, 10, 63, "int", "Maximum holding period (trading days)"),
        ParamSpec("t2_r", 4.0, 2.5, 8.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        w20, w50, wc = val(features, TFW, "ema_20"), val(features, TFW, "ema_50"), val(features, TFW, "close")
        st_dir, struct, wadx = (val(features, TFW, k) for k in ("supertrend_dir", "trend_structure", "adx_14"))
        d50 = val(features, TFD, "ema_50")
        dc = val(features, TFD, "close") or features.ltp
        atr = atr_daily(features)
        if None in (w20, w50, wc, atr):
            return None
        if not (w20 > w50 and wc > w20):  # type: ignore[operator]
            return None
        if not ((st_dir is not None and st_dir > 0) or (struct is not None and struct > 0)):
            return None
        if wadx is not None and wadx < params["min_adx"]:
            return None
        if d50 is not None and dc <= d50:
            return None
        fs = fundamental_score(features)
        notes = [f"weekly EMA20 {w20:.2f} > EMA50 {w50:.2f}"]
        if fs is None:
            if params["require_fundamentals"]:
                return None
            notes.append("fundamentals score unavailable")
        elif fs < params["min_fundamental_score"]:
            return None
        else:
            notes.append(f"fundamentals score {fs:.0f}")
        entry = features.ltp
        stop = pick_stop(
            Direction.LONG,
            entry,
            [w20 - 0.25 * atr, (d50 - 0.25 * atr) if d50 is not None else None],  # type: ignore[operator]
            atr,  # type: ignore[arg-type]
            min_atr=1.0,
            max_atr=params["sl_max_atr"],
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
            strength=strength_from((wadx, 20, 40, 25.0), (fs, 50, 90, 25.0)),
            notes=notes,
        )
