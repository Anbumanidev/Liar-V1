"""Crafted bullish / contrary feature sets for every strategy (shared by the strategy tests)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.enums import Mode, Regime, Signal
from core.types import Levels, MarketContext, StockFeatures
from tests.strategies.strategies_kit import at, features, market, pattern, tf

# ----------------------------------------------------------------------------- intraday


def intraday_levels() -> Levels:
    return Levels(
        pivot=99.85, r1=101.0, r2=102.0, r3=103.0, s1=98.7, s2=97.5, s3=96.5,
        cpr_bc=99.8, cpr_tc=99.9, cpr_width_pct=0.1,
        prev_high=100.8, prev_low=98.6, prev_close=None,
        supports=[99.5, 99.0, 98.5], resistances=[101.0, 102.0, 103.0],
    )


def intraday_bull(v5: dict[str, Any] | None = None, v15: dict[str, Any] | None = None,
                  regime: Regime = Regime.TRENDING_UP, **kw: Any) -> StockFeatures:
    """Bullish 5m/15m/1h snapshot around 100 (ATR 5m = 0.4)."""
    base5: dict[str, Any] = dict(open=99.9, high=100.3, low=99.75, close=100.2, ema_9=100.0, ema_21=99.8,
                                 vwap=99.7, atr_14=0.4, rvol=2.0, adx_14=28.0, rsi_14=60.0,
                                 bb_upper=100.6, bb_mid=99.9, bb_lower=99.2, bb_width_pct=1.4,
                                 supertrend_dir=1.0, trend_structure=1.0)
    base15: dict[str, Any] = dict(open=99.8, high=100.3, low=99.6, close=100.2, ema_9=100.0, ema_21=99.5,
                                  adx_14=30.0, plus_di=30.0, minus_di=15.0, supertrend=99.6, supertrend_dir=1.0,
                                  atr_14=0.7, trend_structure=1.0)
    base5.update(v5 or {})
    base15.update(v15 or {})
    top: dict[str, Any] = dict(atr=0.4, gap_pct=1.5, opening_range_high=100.0, opening_range_low=99.0,
                               day_high=100.3, day_low=98.9, prev_close=98.0, rvol=2.0, rs_vs_nifty=2.0,
                               regime=regime)
    top.update(kw)
    return features(
        Mode.INTRADAY, top.pop("ltp", 100.25),
        [tf("5m", levels=intraday_levels(), **base5),
         tf("15m", **base15),
         tf("1h", ema_9=100.0, ema_21=99.0, close=100.2, supertrend_dir=1.0)],
        "5m", **top,
    )


def mean_reversion_bull(**kw: Any) -> StockFeatures:
    v5 = dict(open=98.1, high=98.35, low=97.9, close=98.3, bb_lower=98.0, bb_mid=99.5, bb_upper=101.0, rsi_14=27.0,
              ema_9=98.6, ema_21=98.8, vwap=98.9)
    top = dict(ltp=98.3, regime=Regime.RANGING, day_high=99.6, day_low=97.9)
    top.update(kw)
    return intraday_bull(v5=v5, **top)


def cpr_wide() -> StockFeatures:
    """Bullish snapshot but with a wide (non-trending) CPR."""
    f = intraday_bull()
    t5 = f.timeframes["5m"].model_copy(update={"levels": intraday_levels().model_copy(update={"cpr_width_pct": 1.0})})
    return f.model_copy(update={"timeframes": {**f.timeframes, "5m": t5}})


# ----------------------------------------------------------------------------- daily

def daily_levels(resistances: list[float], supports: list[float], **kw: Any) -> Levels:
    return Levels(resistances=resistances, supports=supports, **kw)


def btst_bull(**kw: Any) -> StockFeatures:
    top: dict[str, Any] = dict(atr=2.5, day_high=104.8, day_low=101.0, prev_close=102.0, rvol=1.8,
                               delivery_pct=55.0, delivery_pct_avg_20=45.0, extra={"sector": "IT"},
                               regime=Regime.TRENDING_UP, week52_high=120.0, week52_low=80.0)
    top.update(kw)
    return features(
        Mode.BTST, top.pop("ltp", 104.5),
        [tf("15m", ema_9=104.0, ema_21=103.0, close=104.5),
         tf("1h", ema_9=103.8, ema_21=102.5, close=104.5),
         tf("1d", levels=daily_levels([106.0, 110.0], [100.0, 97.0]), close=104.5, ema_20=101.0, ema_50=98.0,
            atr_14=2.5, rvol=1.8)],
        "1d", **top,
    )


def swing(daily: dict[str, Any], lv: Levels | None = None, hourly: dict[str, Any] | None = None,
          **kw: Any) -> StockFeatures:
    top: dict[str, Any] = dict(atr=2.0, rvol=1.2, regime=Regime.TRENDING_UP, week52_high=130.0, week52_low=70.0)
    top.update(kw)
    close = daily.get("close", 101.5)
    return features(
        Mode.SWING, top.pop("ltp", close),
        [tf("1h", **(hourly or dict(ema_9=101.0, ema_21=100.0, close=close))),
         tf("1d", levels=lv or daily_levels([112.0, 118.0], [96.0, 92.0]), atr_14=2.0, **daily)],
        "1d", **top,
    )


def swing_breakout_bull(**kw: Any) -> StockFeatures:
    extra = {"range_high_20": 100.0, "range_low_20": 92.0}
    top = dict(rvol=2.0, extra=extra)
    top.update(kw)
    return swing(dict(open=100.0, high=102.0, low=99.6, close=101.5, ema_20=98.0, ema_50=95.0), **top)


def swing_pullback_bull(**daily_kw: Any) -> StockFeatures:
    d = dict(open=100.2, high=101.8, low=99.8, close=101.5, ema_20=100.0, ema_50=96.0, ema_200=90.0,
             rsi_14=55.0, adx_14=24.0)
    d.update(daily_kw)
    return swing(d)


def swing_divergence_bull(**daily_kw: Any) -> StockFeatures:
    d = dict(open=95.0, high=97.0, low=94.2, close=96.8, rsi_14=38.0, rsi_bull_div=1.0, ema_50=101.0)
    d.update(daily_kw)
    lv = daily_levels([100.0, 104.0, 108.0], [94.0, 90.0])
    f = swing(d, lv=lv, regime=Regime.RANGING)
    t = f.timeframes["1d"].model_copy(update={"patterns": [pattern("hammer", "1d", Signal.BULLISH, 100, at(9, 15))]})
    return f.model_copy(update={"timeframes": {**f.timeframes, "1d": t}})


def swing_rs_bull(**kw: Any) -> StockFeatures:
    top = dict(rs_vs_nifty=8.0, week52_high=105.0)
    top.update(kw)
    return swing(dict(open=101.5, high=103.5, low=101.0, close=103.0, ema_20=100.5, ema_50=95.0, rsi_14=65.0,
                      adx_14=28.0), lv=daily_levels([105.0, 110.0], [100.0, 96.0]), **top)


def positional(daily: dict[str, Any], weekly: dict[str, Any], lv: Levels | None = None, **kw: Any) -> StockFeatures:
    top: dict[str, Any] = dict(atr=2.5, rvol=1.2, regime=Regime.TRENDING_UP, week52_high=125.0, week52_low=70.0)
    top.update(kw)
    close = daily.get("close", 104.0)
    return features(
        Mode.POSITIONAL, top.pop("ltp", close),
        [tf("1d", levels=lv or daily_levels([112.0, 120.0], [98.0, 94.0]), atr_14=2.5, **daily),
         tf("1wk", **weekly)],
        "1d", **top,
    )


def weekly_trend_bull(fscore: float | None = 72.0, **kw: Any) -> StockFeatures:
    extra = {"fundamental_score": fscore} if fscore is not None else {}
    return positional(dict(close=104.0, ema_20=101.0, ema_50=98.0),
                      dict(close=104.0, ema_20=95.0, ema_50=88.0, supertrend_dir=1.0, adx_14=25.0),
                      extra=extra, **kw)


def high52_bull(**kw: Any) -> StockFeatures:
    top = dict(week52_high=104.5, rvol=2.0, extra={"range_high_20": 101.0, "range_low_20": 94.0}, rs_vs_nifty=6.0)
    top.update(kw)
    return positional(dict(close=104.0, ema_20=101.0, ema_50=98.0, ema_200=90.0),
                      dict(close=104.0, ema_20=96.0, ema_50=90.0), lv=daily_levels([], [101.0, 98.0]), **top)


def sector_rotation_bull(sector: str = "IT", **kw: Any) -> StockFeatures:
    top = dict(rs_vs_nifty=6.0, extra={"sector": sector})
    top.update(kw)
    return positional(dict(close=104.0, ema_20=100.0, ema_50=96.0, rsi_14=62.0),
                      dict(close=104.0, ema_20=95.0, ema_50=90.0), **top)


SECTORS = {"IT": 2.0, "PHARMA": 1.0, "AUTO": 0.8, "METAL": -0.5}


# ----------------------------------------------------------------------------- case table


@dataclass(frozen=True)
class Case:
    strategy: str
    mode: Mode
    now: datetime
    bull: Callable[[], StockFeatures]
    contrary: Callable[[], StockFeatures]
    market: Callable[[], MarketContext | None]
    outside: datetime
    contrary_market: Callable[[], MarketContext | None] | None = None

    def cmarket(self) -> MarketContext | None:
        return (self.contrary_market or self.market)()


def _mk() -> MarketContext:
    return market(Signal.BULLISH, 0.6, SECTORS, 1.4)


CASES: list[Case] = [
    Case("opening_range_breakout", Mode.INTRADAY, at(10, 0), intraday_bull,
         lambda: intraday_bull(rvol=1.0), _mk, at(12, 0)),
    Case("vwap_pullback", Mode.INTRADAY, at(10, 30), intraday_bull,
         lambda: intraday_bull(v5={"low": 100.1}), _mk, at(9, 35)),
    Case("ema_9_21_pullback", Mode.INTRADAY, at(10, 30), intraday_bull,
         lambda: intraday_bull(v15={"ema_9": 99.0}), _mk, at(14, 40)),
    Case("cpr_breakout", Mode.INTRADAY, at(10, 0), intraday_bull, cpr_wide, _mk, at(13, 30)),
    Case("gap_and_go", Mode.INTRADAY, at(10, 0), intraday_bull,
         lambda: intraday_bull(gap_pct=0.2), _mk, at(11, 0)),
    Case("gap_fill", Mode.INTRADAY, at(10, 0), lambda: intraday_bull(gap_pct=-1.5, prev_close=102.0),
         lambda: intraday_bull(gap_pct=-1.5, prev_close=102.0, ltp=99.8), _mk, at(12, 0)),
    Case("relative_strength_momentum", Mode.INTRADAY, at(10, 30), intraday_bull,
         lambda: intraday_bull(rs_vs_nifty=0.3), _mk, at(9, 45)),
    Case("mean_reversion", Mode.INTRADAY, at(11, 0), mean_reversion_bull,
         lambda: mean_reversion_bull(regime=Regime.TRENDING_UP), _mk, at(9, 45)),
    Case("supertrend_adx_15m", Mode.INTRADAY, at(10, 30), intraday_bull,
         lambda: intraday_bull(v15={"adx_14": 15.0}), _mk, at(14, 45)),
    Case("btst_strong_close", Mode.BTST, at(15, 0), btst_bull, lambda: btst_bull(ltp=102.5), _mk, at(13, 0)),
    Case("swing_base_breakout", Mode.SWING, at(16, 30), swing_breakout_bull,
         lambda: swing_breakout_bull(rvol=0.9), _mk, at(12, 0)),
    Case("swing_ema_pullback", Mode.SWING, at(16, 30), swing_pullback_bull,
         lambda: swing_pullback_bull(ema_20=95.0), _mk, at(12, 0)),
    Case("swing_reversal_divergence", Mode.SWING, at(16, 30), swing_divergence_bull,
         lambda: swing_divergence_bull(rsi_bull_div=0.0), _mk, at(12, 0)),
    Case("swing_rs_leader", Mode.SWING, at(16, 30), swing_rs_bull, swing_rs_bull, _mk, at(12, 0),
         contrary_market=lambda: market(Signal.BEARISH, -0.8, SECTORS, 0.7)),
    Case("positional_weekly_trend", Mode.POSITIONAL, at(16, 30), weekly_trend_bull,
         lambda: weekly_trend_bull(fscore=40.0), _mk, at(12, 0)),
    Case("positional_52w_breakout", Mode.POSITIONAL, at(16, 30), high52_bull,
         lambda: high52_bull(rvol=1.0), _mk, at(12, 0)),
    Case("positional_sector_rotation", Mode.POSITIONAL, at(16, 30), sector_rotation_bull,
         lambda: sector_rotation_bull("METAL"), _mk, at(12, 0)),
]
