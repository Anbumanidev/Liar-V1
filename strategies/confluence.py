"""Deterministic confluence score (section 7.3).

Components and their maximum points (before normalisation):

============== ===== =============================================================
component      max   rule
============== ===== =============================================================
trend          25    per mode timeframe: EMA fast/slow order, close vs slow EMA,
                     Supertrend direction, trend structure -> score in [-1, 1];
                     points = 25 x mean(max(0, score x direction))
volume         15    relative volume 0.8 -> 0 .. 2.0 -> full; for non-intraday modes
                     10 of the 15 come from rvol and 5 from delivery % vs its 20-day
                     average (0.8x -> 0 .. 1.3x -> full) when delivery data exists
regime         10    signal.regime_fit -> 10 (6 when the regime is UNDEFINED), misfit 0;
                     without a signal: aligned trend regime 10, ranging/undefined 5,
                     high volatility 3, opposite trend 0
market         15    NIFTY trend 7 (aligned 7 / neutral or unknown 3.5 / opposite 0),
                     sector strength 5 (-1 % -> 0 .. +1 % -> 5 in trade direction,
                     unknown 2.5), breadth 3 (A/D 0.8 -> 0 .. 1.5 -> 3, inverted for
                     shorts, unknown 1.5)
candles        10    best recent pattern in trade direction (strength/100 x 10) minus
                     half of the best opposing pattern, clamped to [0, 10]
room           10    distance to the nearest opposing level in R: <= 0.5R -> 0,
                     >= min_rr + 0.5 R -> 10; no level known -> 8
sentiment      5     (sentiment x direction + 1) / 2 x 5; unknown -> 2.5
fundamentals   10    SWING/POSITIONAL only: score/100 x 10 (inverted for shorts);
                     unknown -> 5
============== ===== =============================================================

``score = sum(points) / sum(max points of applicable components) x 100`` minus an
``EVENT_RISK_PENALTY`` (15) when ``event_risk`` is True, clamped to 0..100.
Grades: A+ >= 85, A >= 75, B >= 60, else C.

The score is symmetric: mirroring every price feature around a pivot (bullish <-> bearish)
gives a SHORT the same score as the corresponding LONG.
"""

from __future__ import annotations

import math

from config.schema import ModeConfig
from core.enums import Direction, Grade, Mode, Regime, Signal
from core.types import ConfluenceResult, MarketContext, StockFeatures, StrategySignal, TimeframeFeatures
from strategies._daily import fundamental_score as extra_fundamental_score
from strategies._daily import sector_strength
from strategies.base import collect_levels, latest_patterns, nearest_opposing_level, sign
from strategies.registry import get_strategy

COMPONENT_MAX: dict[str, float] = {
    "trend": 25.0,
    "volume": 15.0,
    "regime": 10.0,
    "market": 15.0,
    "candles": 10.0,
    "room": 10.0,
    "sentiment": 5.0,
    "fundamentals": 10.0,
}
EVENT_RISK_PENALTY: float = 15.0
FUNDAMENTAL_MODES: frozenset[Mode] = frozenset({Mode.SWING, Mode.POSITIONAL})
_INTRADAY_TFS = frozenset({"1m", "5m", "15m", "1h"})


def grade_for(score: float) -> Grade:
    """A+ >= 85, A >= 75, B >= 60, else C."""
    if score >= 85:
        return Grade.A_PLUS
    if score >= 75:
        return Grade.A
    if score >= 60:
        return Grade.B
    return Grade.C


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _lin(x: float | None, lo: float, hi: float) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return _clip01((x - lo) / (hi - lo))


def _sgn(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def tf_trend_score(tf: TimeframeFeatures) -> float | None:
    """Trend score of one timeframe in [-1, 1] (None when nothing is known)."""
    fast_k, slow_k = ("ema_9", "ema_21") if tf.timeframe in _INTRADAY_TFS else ("ema_20", "ema_50")
    fast, slow, close = tf.v(fast_k), tf.v(slow_k), tf.v("close")
    votes: list[float] = []
    if fast is not None and slow is not None:
        votes.append(_sgn(fast - slow))
    if close is not None and slow is not None:
        votes.append(_sgn(close - slow))
    st = tf.v("supertrend_dir")
    if st is not None:
        votes.append(_sgn(st))
    ts = tf.v("trend_structure")
    if ts is not None:
        votes.append(max(-1.0, min(1.0, ts)))
    return sum(votes) / len(votes) if votes else None


def _trend_points(features: StockFeatures, timeframes: list[str], s: int, notes: list[str]) -> float:
    scores: list[float] = []
    for name in timeframes:
        tf = features.tf(name)
        if tf is None:
            continue
        sc = tf_trend_score(tf)
        if sc is not None:
            scores.append(_clip01(sc * s))
    if not scores:
        notes.append("trend: no timeframe data")
        return 0.0
    return COMPONENT_MAX["trend"] * sum(scores) / len(scores)


def _volume_points(features: StockFeatures, mode: Mode, notes: list[str]) -> float:
    rvol = features.rvol
    if rvol is None:
        tf = features.tf(features.primary_timeframe)
        rvol = tf.v("rvol") if tf is not None else None
    rv = _lin(rvol, 0.8, 2.0)
    if rv is None:
        notes.append("volume: rvol unknown")
        rv = 0.0
    dp, dpa = features.delivery_pct, features.delivery_pct_avg_20
    if mode != Mode.INTRADAY and dp is not None and dpa is not None and dpa > 0:
        dv = _lin(dp / dpa, 0.8, 1.3) or 0.0
        return 10.0 * rv + 5.0 * dv
    return COMPONENT_MAX["volume"] * rv


def _regime_points(features: StockFeatures, signal: StrategySignal | None, s: int) -> float:
    regime = features.regime
    if signal is not None:
        if not signal.regime_fit:
            return 0.0
        return 6.0 if regime == Regime.UNDEFINED else COMPONENT_MAX["regime"]
    aligned = Regime.TRENDING_UP if s > 0 else Regime.TRENDING_DOWN
    opposite = Regime.TRENDING_DOWN if s > 0 else Regime.TRENDING_UP
    if regime == aligned:
        return COMPONENT_MAX["regime"]
    if regime == opposite:
        return 0.0
    if regime == Regime.HIGH_VOLATILITY:
        return 3.0
    return 5.0


def _market_points(features: StockFeatures, market: MarketContext | None, s: int) -> float:
    pts = 0.0
    trend = market.nifty_trend if market is not None else Signal.NEUTRAL
    aligned = Signal.BULLISH if s > 0 else Signal.BEARISH
    if trend == aligned:
        pts += 7.0
    elif trend == Signal.NEUTRAL:
        pts += 3.5
    sec = sector_strength(features, market)
    pts += 2.5 if sec is None else 5.0 * _clip01((sec * s + 1.0) / 2.0)
    adr = market.advance_decline_ratio if market is not None else None
    if adr is None or adr <= 0:
        pts += 1.5
    else:
        breadth = adr if s > 0 else 1.0 / adr
        pts += 3.0 * (_lin(breadth, 0.8, 1.5) or 0.0)
    return pts


def _candle_points(features: StockFeatures, timeframes: list[str], s: int) -> float:
    aligned = Signal.BULLISH if s > 0 else Signal.BEARISH
    opposed = Signal.BEARISH if s > 0 else Signal.BULLISH
    best_a, best_o = 0, 0
    for name in timeframes:
        tf = features.tf(name)
        best_a = max([best_a, *latest_patterns(tf, aligned, lookback_bars=2)])
        best_o = max([best_o, *latest_patterns(tf, opposed, lookback_bars=2)])
    pts = COMPONENT_MAX["candles"] * (min(best_a, 100) / 100) - 0.5 * COMPONENT_MAX["candles"] * (min(best_o, 100) / 100)
    return max(0.0, min(COMPONENT_MAX["candles"], pts))


def _room_points(
    features: StockFeatures, signal: StrategySignal | None, direction: Direction, min_rr: float,
    timeframes: list[str], notes: list[str],
) -> float:
    entry = signal.entry_price if signal is not None else features.ltp
    risk = abs(signal.entry_price - signal.stop_loss) if signal is not None else (features.atr or 0.0)
    if risk <= 0:
        risk = entry * 0.01
    lv = nearest_opposing_level(features, entry, direction, timeframes)
    if lv is None:
        has_any = bool(collect_levels(features, timeframes))
        notes.append("room: no opposing level" + (" (clear path)" if has_any else " (no level data)"))
        return 8.0
    room_r = abs(lv - entry) / risk
    notes.append(f"room: nearest opposing level {lv:.2f} = {room_r:.2f}R")
    return COMPONENT_MAX["room"] * (_lin(room_r, 0.5, min_rr + 0.5) or 0.0)


def _score_direction(
    features: StockFeatures,
    market: MarketContext | None,
    signal: StrategySignal | None,
    mode: Mode,
    mode_cfg: ModeConfig,
    direction: Direction,
    sentiment: float | None,
    fundamental_score: float | None,
    event_risk: bool,
) -> ConfluenceResult:
    s = sign(direction)
    notes: list[str] = [f"direction {direction.value}"]
    tfs = [str(t.value if hasattr(t, "value") else t) for t in mode_cfg.timeframes]
    comp: dict[str, float] = {
        "trend": _trend_points(features, tfs, s, notes),
        "volume": _volume_points(features, mode, notes),
        "regime": _regime_points(features, signal, s),
        "market": _market_points(features, market, s),
        "candles": _candle_points(features, tfs, s),
        "room": _room_points(features, signal, direction, mode_cfg.min_rr, tfs, notes),
        "sentiment": 2.5 if sentiment is None else COMPONENT_MAX["sentiment"] * _clip01((sentiment * s + 1) / 2),
    }
    if mode in FUNDAMENTAL_MODES:
        fs = fundamental_score if fundamental_score is not None else extra_fundamental_score(features)
        if fs is None:
            comp["fundamentals"] = 5.0
            notes.append("fundamentals: unknown")
        else:
            frac = _clip01(fs / 100.0)
            comp["fundamentals"] = COMPONENT_MAX["fundamentals"] * (frac if s > 0 else 1 - frac)
    max_total = sum(COMPONENT_MAX[k] for k in comp)
    raw = sum(comp.values())
    score = raw / max_total * 100.0
    if event_risk:
        comp["event_risk"] = -EVENT_RISK_PENALTY
        score -= EVENT_RISK_PENALTY
        notes.append(f"event risk penalty -{EVENT_RISK_PENALTY:g}")
    score = round(max(0.0, min(100.0, score)), 2)
    comp = {k: round(v, 3) for k, v in comp.items()}
    notes.append(f"raw {raw:.2f} / {max_total:g} max points")
    return ConfluenceResult(score=score, grade=grade_for(score), components=comp, notes=notes)


def compute_confluence(
    features: StockFeatures,
    market: MarketContext | None,
    signal: StrategySignal | None,
    mode: Mode,
    mode_cfg: ModeConfig,
    *,
    sentiment: float | None = None,
    fundamental_score: float | None = None,
    event_risk: bool = False,
) -> ConfluenceResult:
    """Deterministic 0-100 confluence for ``features`` in ``mode``.

    Args:
        features: stock features for the mode.
        market: market context (None -> neutral market points).
        signal: the strategy signal being graded (direction / entry / SL / regime fit come
            from it). Without a signal the score is computed for LONG (CNC modes) or for the
            better of LONG/SHORT (INTRADAY).
        mode: trading mode.
        mode_cfg: the mode's configuration (timeframes, min R:R).
        sentiment: aggregate FinBERT score -1..+1 (None = unknown).
        fundamental_score: 0-100 (SWING/POSITIONAL; falls back to ``extra['fundamental_score']``).
        event_risk: an event falls inside the holding window -> penalty.
    """
    m = Mode(mode)
    if signal is not None:
        res = _score_direction(features, market, signal, m, mode_cfg, signal.direction, sentiment,
                               fundamental_score, event_risk)
        try:
            res.notes.insert(0, f"strategy {signal.strategy} ({get_strategy(signal.strategy).description})")
        except KeyError:
            res.notes.insert(0, f"strategy {signal.strategy}")
        return res
    dirs = [Direction.LONG, Direction.SHORT] if m == Mode.INTRADAY else [Direction.LONG]
    results = [
        _score_direction(features, market, None, m, mode_cfg, d, sentiment, fundamental_score, event_risk)
        for d in dirs
    ]
    return max(results, key=lambda r: r.score)
