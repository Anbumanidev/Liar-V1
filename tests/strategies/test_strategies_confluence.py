"""Confluence: grades, bounds, monotonic sanity, SHORT symmetry, mode-specific components."""

from __future__ import annotations

import pytest

from config.schema import Settings
from core.enums import Grade, Mode, Regime, Signal
from strategies.confluence import COMPONENT_MAX, EVENT_RISK_PENALTY, compute_confluence, grade_for
from strategies.registry import get_strategy, resolve_params
from tests.strategies.strategies_cases import (
    CASES,
    SECTORS,
    intraday_bull,
    swing_pullback_bull,
)
from tests.strategies.strategies_kit import at, market, mirror_features, mirror_market, pattern


def _signal(name: str, settings: Settings, feats=None, mk=None):
    case = next(c for c in CASES if c.strategy == name)
    return get_strategy(name).generate(
        feats or case.bull(), mk or case.market(), resolve_params(name, settings), case.now,
        min_rr=settings.mode(case.mode).min_rr,
    )


@pytest.mark.parametrize(
    ("score", "grade"),
    [(100, Grade.A_PLUS), (85, Grade.A_PLUS), (84.99, Grade.A), (75, Grade.A), (74.99, Grade.B),
     (60, Grade.B), (59.99, Grade.C), (0, Grade.C)],
)
def test_grade_boundaries(score: float, grade: Grade) -> None:
    assert grade_for(score) == grade


def test_bullish_setup_scores_high_and_components_bounded(settings: Settings) -> None:
    f = intraday_bull()
    f = f.model_copy(update={"timeframes": {**f.timeframes, "5m": f.timeframes["5m"].model_copy(
        update={"patterns": [pattern("engulfing", "5m", Signal.BULLISH, 100, at(9, 55))]})}, "extra": {"sector": "IT"}})
    mk = market(Signal.BULLISH, 0.6, SECTORS, 1.6)
    sig = _signal("opening_range_breakout", settings, f, mk)
    res = compute_confluence(f, mk, sig, Mode.INTRADAY, settings.mode(Mode.INTRADAY), sentiment=0.6)
    assert 0 <= res.score <= 100
    assert res.score >= 85 and res.grade == Grade.A_PLUS
    for k, v in res.components.items():
        assert 0 <= v <= COMPONENT_MAX[k] + 1e-9
    assert "fundamentals" not in res.components  # intraday: not applicable


def test_bearish_market_lowers_long_score(settings: Settings) -> None:
    f = intraday_bull()
    sig = _signal("opening_range_breakout", settings, f)
    cfg = settings.mode(Mode.INTRADAY)
    good = compute_confluence(f, market(Signal.BULLISH, 0.5, {}, 1.5), sig, Mode.INTRADAY, cfg, sentiment=0.5)
    bad = compute_confluence(f, market(Signal.BEARISH, -0.5, {}, 0.6), sig, Mode.INTRADAY, cfg, sentiment=-0.5)
    assert bad.score < good.score
    assert bad.components["market"] < good.components["market"]
    assert bad.components["sentiment"] < good.components["sentiment"]


def test_monotonic_in_rvol_patterns_and_event_risk(settings: Settings) -> None:
    cfg = settings.mode(Mode.INTRADAY)
    mk = market()
    scores = []
    for rvol in (0.5, 1.0, 1.5, 2.0, 3.0):
        f = intraday_bull(rvol=rvol)
        scores.append(compute_confluence(f, mk, None, Mode.INTRADAY, cfg).score)
    assert scores == sorted(scores) and scores[0] < scores[-1]

    f = intraday_bull()
    base = compute_confluence(f, mk, None, Mode.INTRADAY, cfg)
    with_pat = f.model_copy(update={"timeframes": {**f.timeframes, "15m": f.timeframes["15m"].model_copy(
        update={"patterns": [pattern("engulfing", "15m", Signal.BULLISH, 100)]})}})
    against = f.model_copy(update={"timeframes": {**f.timeframes, "15m": f.timeframes["15m"].model_copy(
        update={"patterns": [pattern("shooting_star", "15m", Signal.BEARISH, -100)]})}})
    assert compute_confluence(with_pat, mk, None, Mode.INTRADAY, cfg).score > base.score
    assert compute_confluence(against, mk, None, Mode.INTRADAY, cfg).components["candles"] == 0
    ev = compute_confluence(f, mk, None, Mode.INTRADAY, cfg, event_risk=True)
    assert ev.score == pytest.approx(max(0.0, base.score - EVENT_RISK_PENALTY), abs=0.01)
    assert ev.components["event_risk"] == -EVENT_RISK_PENALTY


def test_trend_alignment_counts(settings: Settings) -> None:
    cfg = settings.mode(Mode.INTRADAY)
    up = intraday_bull()
    flat = intraday_bull(v5={"ema_9": 99.8, "ema_21": 99.8, "supertrend_dir": None, "trend_structure": 0.0},
                         v15={"ema_9": 99.5, "ema_21": 99.5, "supertrend_dir": None, "trend_structure": 0.0})
    sig = _signal("opening_range_breakout", settings, up)
    a = compute_confluence(up, market(), sig, Mode.INTRADAY, cfg)
    b = compute_confluence(flat, market(), sig, Mode.INTRADAY, cfg)
    assert a.components["trend"] > b.components["trend"]


@pytest.mark.parametrize("name", [c.strategy for c in CASES if c.mode == Mode.INTRADAY])
def test_short_symmetry(name: str, settings: Settings) -> None:
    case = next(c for c in CASES if c.strategy == name)
    cfg = settings.mode(Mode.INTRADAY)
    f_long = case.bull()
    mk_long = market(Signal.BULLISH, 0.5, SECTORS, 1.3)
    f_long = f_long.model_copy(update={"extra": {"sector": "AUTO"}})
    f_short = mirror_features(f_long, 100.0)
    mk_short = mirror_market(mk_long)
    s_long = _signal(name, settings, f_long, mk_long)
    s_short = _signal(name, settings, f_short, mk_short)
    assert s_long is not None and s_short is not None
    r_long = compute_confluence(f_long, mk_long, s_long, Mode.INTRADAY, cfg, sentiment=0.3)
    r_short = compute_confluence(f_short, mk_short, s_short, Mode.INTRADAY, cfg, sentiment=-0.3)
    assert r_short.score == pytest.approx(r_long.score, abs=0.5)
    assert r_short.grade == r_long.grade or abs(r_short.score - r_long.score) < 0.5
    # bullish evidence must NOT help a short
    wrong = compute_confluence(f_long, mk_long, s_short, Mode.INTRADAY, cfg, sentiment=0.3)
    assert wrong.score < r_long.score


def test_no_signal_picks_better_direction_intraday(settings: Settings) -> None:
    cfg = settings.mode(Mode.INTRADAY)
    f = mirror_features(intraday_bull(), 100.0)
    res = compute_confluence(f, mirror_market(market()), None, Mode.INTRADAY, cfg)
    assert res.notes[0] == "direction SHORT"
    swing_cfg = settings.mode(Mode.SWING)
    res2 = compute_confluence(swing_pullback_bull(), market(), None, Mode.SWING, swing_cfg)
    assert res2.notes[0] == "direction LONG"


def test_swing_fundamentals_and_regime(settings: Settings) -> None:
    cfg = settings.mode(Mode.SWING)
    f = swing_pullback_bull()
    sig = _signal("swing_ema_pullback", settings, f)
    hi = compute_confluence(f, market(), sig, Mode.SWING, cfg, fundamental_score=90)
    lo = compute_confluence(f, market(), sig, Mode.SWING, cfg, fundamental_score=10)
    unk = compute_confluence(f, market(), sig, Mode.SWING, cfg)
    assert hi.score > unk.score > lo.score
    assert hi.components["fundamentals"] == pytest.approx(9.0)
    # extra['fundamental_score'] used as a fallback
    f2 = f.model_copy(update={"extra": {"fundamental_score": 90}})
    assert compute_confluence(f2, market(), sig, Mode.SWING, cfg).score == pytest.approx(hi.score)
    # regime misfit signal scores lower
    misfit = sig.model_copy(update={"regime_fit": False})
    assert compute_confluence(f, market(), misfit, Mode.SWING, cfg, fundamental_score=90).score < hi.score


def test_room_to_target(settings: Settings) -> None:
    cfg = settings.mode(Mode.INTRADAY)
    f = intraday_bull()
    sig = _signal("opening_range_breakout", settings, f)
    def with_resistance(level: float):  # type: ignore[no-untyped-def]
        return f.model_copy(update={"timeframes": {**f.timeframes, "5m": f.timeframes["5m"].model_copy(
            update={"levels": f.timeframes["5m"].levels.model_copy(update={"resistances": [level, 101.0]})})}})

    a = compute_confluence(f, market(), sig, Mode.INTRADAY, cfg)
    blocked = compute_confluence(with_resistance(sig.entry_price + 0.5 * f.atr), market(), sig, Mode.INTRADAY, cfg)
    assert blocked.components["room"] < a.components["room"]           # a real resistance just above cuts room
    tested = compute_confluence(with_resistance(sig.entry_price + 0.1 * f.atr), market(), sig, Mode.INTRADAY, cfg)
    assert tested.components["room"] == a.components["room"]           # a level the price is testing is ignored


def test_empty_features_do_not_crash(settings: Settings) -> None:
    from tests.strategies.strategies_kit import features

    f = features(Mode.SWING, 100.0, [], "1d")
    res = compute_confluence(f, None, None, Mode.SWING, settings.mode(Mode.SWING), event_risk=True)
    assert 0 <= res.score <= 100 and res.grade == Grade.C
    f = f.model_copy(update={"regime": Regime.HIGH_VOLATILITY})
    assert compute_confluence(f, None, None, Mode.SWING, settings.mode(Mode.SWING)).components["regime"] == 3.0
