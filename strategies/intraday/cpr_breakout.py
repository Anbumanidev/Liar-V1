"""CPR / Pivot Breakout (INTRADAY, long + short).

Rules (5m primary):
* today's Central Pivot Range is narrow (``cpr_width_pct <= max_cpr_width_pct``) - narrow CPR
  days tend to trend;
* last closed 5m bar closed beyond the whole CPR (above TC for longs / below BC for shorts)
  by ``buffer_atr`` x ATR, and price has not run more than ``max_extension_atr`` x ATR past it;
* relative volume >= ``min_rvol``.
Entry = LTP (MARKET). SL = beyond the opposite side of the CPR (or the breakout bar extreme
when the CPR-based stop is too wide). Targets: R1/R2 (S1/S2) or minimum R:R.
"""

from __future__ import annotations

from typing import Any

from core.enums import Direction, EntryType, Mode, Regime
from core.types import Levels, MarketContext, StockFeatures
from strategies.base import ParamSpec, Setup, Strategy, pick_stop, sign, strength_from, val
from strategies.intraday._common import DIRECTIONS, INTRADAY_TFS, TF5, atr_of, rvol_of, target_levels
from strategies.registry import register


def _cpr_levels(features: StockFeatures) -> Levels | None:
    for name in (*INTRADAY_TFS, "1d"):
        tf = features.tf(name)
        if tf is not None and tf.levels is not None and tf.levels.cpr_tc is not None and tf.levels.cpr_bc is not None:
            return tf.levels
    return None


@register
class CprBreakout(Strategy):
    name = "cpr_breakout"
    description = "Breakout beyond a narrow Central Pivot Range with volume."
    modes = (Mode.INTRADAY,)
    regimes = frozenset({Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.UNDEFINED, Regime.RANGING})
    windows = {Mode.INTRADAY: [("09:30", "13:00")]}
    param_specs = (
        ParamSpec("max_cpr_width_pct", 0.5, 0.05, 2.0, "float", "Maximum CPR width in % of price"),
        ParamSpec("buffer_atr", 0.05, 0.0, 0.5, "float", "Close beyond the CPR by this many ATR"),
        ParamSpec("max_extension_atr", 1.2, 0.2, 3.0, "float", "Skip if price ran further past the CPR (ATR)"),
        ParamSpec("min_rvol", 1.2, 0.8, 5.0, "float", "Minimum relative volume"),
        ParamSpec("sl_buffer_atr", 0.2, 0.0, 1.0, "float", "SL buffer beyond the CPR in ATR"),
        ParamSpec("t2_r", 3.0, 2.0, 6.0, "float", "Second target in R when no farther level exists"),
    )

    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        lv = _cpr_levels(features)
        atr = atr_of(features)
        close = val(features, TF5, "close")
        if lv is None or atr is None or close is None:
            return None
        tc, bc = max(lv.cpr_tc, lv.cpr_bc), min(lv.cpr_tc, lv.cpr_bc)  # type: ignore[type-var]
        width = lv.cpr_width_pct
        if width is None and features.ltp > 0:
            width = (tc - bc) / features.ltp * 100
        if width is None or width > params["max_cpr_width_pct"]:
            return None
        rvol = rvol_of(features)
        if rvol is None or rvol < params["min_rvol"]:
            return None
        ltp = features.ltp
        for d in DIRECTIONS:
            s = sign(d)
            edge, other = (tc, bc) if d == Direction.LONG else (bc, tc)
            if (close - edge) * s < params["buffer_atr"] * atr or (ltp - edge) * s <= 0:
                continue
            if (ltp - edge) * s > params["max_extension_atr"] * atr:
                continue
            bar_extreme = val(features, TF5, "low" if d == Direction.LONG else "high")
            stop = pick_stop(
                d,
                ltp,
                [other - s * params["sl_buffer_atr"] * atr, bar_extreme],
                atr,
                min_atr=0.3,
                max_atr=2.5,
                fallback_atr=1.5,
            )
            if stop is None:
                continue
            return Setup(
                direction=d,
                entry=ltp,
                stop=stop,
                entry_type=EntryType.MARKET,
                target_levels=target_levels(features),
                strength=strength_from((rvol, 1.0, 2.5, 30.0), (-width, -0.5, -0.1, 20.0)),
                notes=[f"close beyond CPR {bc:.2f}-{tc:.2f} (width {width:.2f}%)"],
            )
        return None
