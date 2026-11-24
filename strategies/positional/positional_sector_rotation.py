"""Positional sector rotation leaders (POSITIONAL, LONG only).

* the stock's sector (``extra['sector']``) ranks in the top ``top_n_sectors`` of
  ``MarketContext.sector_strength`` and its strength >= ``min_sector_strength``;
* the stock leads its market: ``rs_vs_nifty`` >= ``min_rs_pct``;
* trend: daily close > EMA 20 > EMA 50; weekly close > weekly EMA 20 (when known);
* RSI(14) <= ``rsi_max`` (not blown off).
Requires a market context with sector strengths. Entry = close (LIMIT). SL = daily EMA 50
structure when within bounds, else ``sl_atr`` x ATR.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import MarketContext, StockFeatures
from strategies._daily import TFD, TFW, atr_daily, daily_levels, sector_of, sector_rank, sector_strength
from strategies.base import CNC_SCAN_WINDOWS, ParamSpec, Setup, Strategy, pick_stop, strength_from, val
from strategies.registry import register


@register
class PositionalSectorRotation(Strategy):
    name = "positional_sector_rotation"
    description = "Relative-strength leaders inside the currently strongest sectors."
    modes = (Mode.POSITIONAL,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.UNDEFINED, Regime.RANGING})
    windows = {Mode.POSITIONAL: CNC_SCAN_WINDOWS}
    param_specs = (
        ParamSpec("top_n_sectors", 3, 1, 6, "int", "Sector must rank within the top N"),
        ParamSpec("min_sector_strength", 0.5, -1.0, 5.0, "float", "Minimum sector strength (%)"),
        ParamSpec("min_rs_pct", 2.0, 0.0, 15.0, "float", "Minimum % outperformance vs NIFTY"),
        ParamSpec("rsi_max", 75.0, 60.0, 85.0, "float", "Maximum daily RSI(14)"),
        ParamSpec("sl_atr", 2.5, 1.0, 3.0, "float", "Fallback SL distance in daily ATR"),
        ParamSpec("max_holding_days", 40, 10, 63, "int", "Maximum holding period (trading days)"),
        ParamSpec("t2_r", 4.0, 2.5, 8.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        rank = sector_rank(features, market)
        strength = sector_strength(features, market)
        if rank is None or strength is None:
            return None
        if rank > params["top_n_sectors"] or strength < params["min_sector_strength"]:
            return None
        rs = features.rs_vs_nifty
        if rs is None or rs < params["min_rs_pct"]:
            return None
        e20, e50, rsi = val(features, TFD, "ema_20"), val(features, TFD, "ema_50"), val(features, TFD, "rsi_14")
        c = val(features, TFD, "close") or features.ltp
        atr = atr_daily(features)
        if None in (e20, e50, atr) or not (c > e20 > e50):  # type: ignore[operator]
            return None
        if rsi is not None and rsi > params["rsi_max"]:
            return None
        w20, wc = val(features, TFW, "ema_20"), val(features, TFW, "close")
        if w20 is not None and wc is not None and wc <= w20:
            return None
        entry = features.ltp
        stop = pick_stop(
            Direction.LONG,
            entry,
            [e50 - 0.25 * atr],  # type: ignore[operator]
            atr,  # type: ignore[arg-type]
            min_atr=1.0,
            max_atr=3.0,
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
            strength=strength_from((strength, 0.0, 3.0, 25.0), (rs, params["min_rs_pct"], 15.0, 25.0),
                                   (-rank, -params["top_n_sectors"], -1, 10.0)),
            notes=[f"sector {sector_of(features)} rank #{rank} ({strength:+.2f}%), RS {rs:+.1f}%"],
        )
