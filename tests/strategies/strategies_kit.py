"""Builders for crafted StockFeatures used by the strategy / confluence tests.

``mirror_features`` reflects every price around a pivot (p' = 2P - p) and flips every
directional value, turning a bullish setup into the equivalent bearish one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.clock import IST
from core.enums import Mode, Regime, Signal
from core.types import Levels, MarketContext, PatternHit, StockFeatures, TimeframeFeatures

NOW_DAY = (2026, 11, 24)


def at(hh: int, mm: int) -> datetime:
    """2026-11-24 hh:mm IST."""
    return datetime(*NOW_DAY, hh, mm, tzinfo=IST)


def tf(name: str, levels: Levels | None = None, patterns: list[PatternHit] | None = None,
       regime: Regime = Regime.UNDEFINED, **values: float | None) -> TimeframeFeatures:
    return TimeframeFeatures(timeframe=name, as_of=at(10, 0), bars=200, values=dict(values),
                             levels=levels, patterns=patterns or [], regime=regime)


def features(mode: Mode, ltp: float, tfs: list[TimeframeFeatures], primary: str, **kw: Any) -> StockFeatures:
    return StockFeatures(symbol="TEST", mode=mode, as_of=at(10, 0), ltp=ltp, primary_timeframe=primary,
                         timeframes={t.timeframe: t for t in tfs}, **kw)


def market(trend: Signal = Signal.BULLISH, change: float | None = 0.6, sectors: dict[str, float] | None = None,
           adr: float | None = 1.4) -> MarketContext:
    return MarketContext(as_of=at(10, 0), nifty_trend=trend, nifty_change_pct=change,
                         sector_strength=sectors or {}, advance_decline_ratio=adr)


def pattern(name: str, tfname: str, direction: Signal, strength: int, when: datetime | None = None) -> PatternHit:
    return PatternHit(name=name, timeframe=tfname, direction=direction, strength=strength, bar_time=when or at(9, 55))


# --------------------------------------------------------------------------- mirroring

_PRICE_KEYS = {
    "ema_9", "ema_20", "ema_21", "ema_50", "ema_200", "vwap", "supertrend", "bb_mid", "close", "open",
}
_SWAP_KEYS = {("high", "low"), ("bb_upper", "bb_lower"), ("plus_di", "minus_di"), ("rsi_bull_div", "rsi_bear_div")}
_NEG_KEYS = {"supertrend_dir", "trend_structure", "macd", "macd_signal", "macd_hist", "vwap_dev_pct"}


def _m(p: float | None, pivot: float) -> float | None:
    return None if p is None else 2 * pivot - p


def _mirror_values(values: dict[str, float | None], pivot: float) -> dict[str, float | None]:
    out = dict(values)
    for k in _PRICE_KEYS:
        if k in values:
            out[k] = _m(values[k], pivot)
    for a, b in _SWAP_KEYS:
        va, vb = values.get(a), values.get(b)
        if a in ("high", "bb_upper"):
            va, vb = _m(va, pivot), _m(vb, pivot)
        if a in values or b in values:
            out[a], out[b] = vb, va
    for k in _NEG_KEYS:
        if values.get(k) is not None:
            out[k] = -values[k]  # type: ignore[operator]
    if values.get("rsi_14") is not None:
        out["rsi_14"] = 100 - values["rsi_14"]  # type: ignore[operator]
    return out


def _mirror_levels(lv: Levels | None, pivot: float) -> Levels | None:
    if lv is None:
        return None
    m = lambda x: _m(x, pivot)  # noqa: E731
    ml = lambda xs: [2 * pivot - x for x in xs]  # noqa: E731
    return Levels(
        pivot=m(lv.pivot), r1=m(lv.s1), r2=m(lv.s2), r3=m(lv.s3), s1=m(lv.r1), s2=m(lv.r2), s3=m(lv.r3),
        cpr_bc=m(lv.cpr_tc), cpr_tc=m(lv.cpr_bc), cpr_width_pct=lv.cpr_width_pct,
        prev_high=m(lv.prev_low), prev_low=m(lv.prev_high), prev_close=m(lv.prev_close),
        supports=ml(lv.resistances), resistances=ml(lv.supports),
        swing_highs=ml(lv.swing_lows), swing_lows=ml(lv.swing_highs),
    )


def _flip(sig: Signal) -> Signal:
    return {Signal.BULLISH: Signal.BEARISH, Signal.BEARISH: Signal.BULLISH}.get(sig, sig)


def _flip_regime(r: Regime) -> Regime:
    return {Regime.TRENDING_UP: Regime.TRENDING_DOWN, Regime.TRENDING_DOWN: Regime.TRENDING_UP}.get(r, r)


def mirror_tf(t: TimeframeFeatures, pivot: float) -> TimeframeFeatures:
    return t.model_copy(update={
        "values": _mirror_values(t.values, pivot),
        "levels": _mirror_levels(t.levels, pivot),
        "patterns": [p.model_copy(update={"direction": _flip(p.direction), "strength": -p.strength})
                     for p in t.patterns],
        "regime": _flip_regime(t.regime),
    })


def mirror_features(f: StockFeatures, pivot: float) -> StockFeatures:
    m = lambda x: _m(x, pivot)  # noqa: E731
    neg = lambda x: None if x is None else -x  # noqa: E731
    return f.model_copy(update={
        "ltp": m(f.ltp),
        "timeframes": {k: mirror_tf(v, pivot) for k, v in f.timeframes.items()},
        "gap_pct": neg(f.gap_pct),
        "opening_range_high": m(f.opening_range_low),
        "opening_range_low": m(f.opening_range_high),
        "day_high": m(f.day_low),
        "day_low": m(f.day_high),
        "prev_close": m(f.prev_close),
        "week52_high": m(f.week52_low),
        "week52_low": m(f.week52_high),
        "rs_vs_nifty": neg(f.rs_vs_nifty),
        "regime": _flip_regime(f.regime),
    })


def mirror_market(mk: MarketContext) -> MarketContext:
    return mk.model_copy(update={
        "nifty_trend": _flip(mk.nifty_trend),
        "nifty_change_pct": None if mk.nifty_change_pct is None else -mk.nifty_change_pct,
        "sector_strength": {k: -v for k, v in mk.sector_strength.items()},
        "advance_decline_ratio": None if not mk.advance_decline_ratio else 1 / mk.advance_decline_ratio,
    })
