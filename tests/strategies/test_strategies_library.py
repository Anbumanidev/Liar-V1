"""Every strategy: fires on a crafted bullish snapshot (and bearish mirror for intraday), stays
silent on a contrary one, respects windows, the mode's minimum R:R and the LONG-only rule."""

from __future__ import annotations

import pytest

from config.schema import Settings
from core.enums import Direction, Mode, Regime
from core.errors import OvernightShortError
from strategies.base import Setup, TICK
from strategies.registry import get_strategy, resolve_params, run_strategies
from tests.strategies.strategies_cases import CASES, Case
from tests.strategies.strategies_kit import at, mirror_features, mirror_market

IDS = [c.strategy for c in CASES]


def _gen(case: Case, settings: Settings, feats=None, mk=None, now=None, min_rr: float | None = None):
    strat = get_strategy(case.strategy)
    params = resolve_params(case.strategy, settings)
    return strat.generate(
        feats if feats is not None else case.bull(),
        mk if mk is not None else case.market(),
        params,
        now or case.now,
        min_rr=min_rr if min_rr is not None else settings.mode(case.mode).min_rr,
        sl_min_atr=settings.risk.sl_min_atr,
        sl_max_atr=settings.risk.sl_max_atr,
    )


def _check_signal(sig, case: Case, direction: Direction, min_rr: float, atr: float | None) -> None:
    assert sig is not None, f"{case.strategy} did not fire"
    assert sig.strategy == case.strategy and sig.mode == case.mode
    assert sig.direction == direction
    s = 1 if direction == Direction.LONG else -1
    assert (sig.entry_price - sig.stop_loss) * s >= TICK - 1e-9, "SL must be on the correct side, non-zero"
    assert (sig.target_1 - sig.entry_price) * s > 0
    assert sig.target_2 is not None and (sig.target_2 - sig.target_1) * s > 0
    assert sig.rr >= min_rr - 1e-9, f"R:R {sig.rr} below {min_rr}"
    if atr:
        dist = abs(sig.entry_price - sig.stop_loss)
        assert dist <= 3.0 * atr + TICK and dist >= 0.3 * atr - TICK
    for p in (sig.entry_price, sig.stop_loss, sig.target_1, sig.target_2):
        assert abs(p / TICK - round(p / TICK)) < 1e-6, f"{p} not on tick"
    if case.mode == Mode.INTRADAY:
        assert sig.max_holding_days == 0 and sig.valid_until is not None and sig.valid_until > case.now
    elif case.mode == Mode.BTST:
        assert sig.max_holding_days == 1
    else:
        assert sig.max_holding_days is not None and sig.max_holding_days >= 2


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_fires_long_on_bullish(case: Case, settings: Settings) -> None:
    feats = case.bull()
    sig = _gen(case, settings, feats)
    _check_signal(sig, case, Direction.LONG, settings.mode(case.mode).min_rr, feats.atr)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_silent_on_contrary(case: Case, settings: Settings) -> None:
    assert _gen(case, settings, case.contrary(), case.cmarket()) is None


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_outside_time_window_is_silent(case: Case, settings: Settings) -> None:
    assert _gen(case, settings, now=case.outside) is None


@pytest.mark.parametrize("case", [c for c in CASES if c.mode == Mode.INTRADAY], ids=lambda c: c.strategy)
def test_intraday_fires_short_on_mirror(case: Case, settings: Settings) -> None:
    pivot = 100.0
    feats = mirror_features(case.bull(), pivot)
    mk = case.market()
    sig = _gen(case, settings, feats, mirror_market(mk) if mk else None)
    _check_signal(sig, case, Direction.SHORT, settings.mode(case.mode).min_rr, feats.atr)
    # the mirrored contrary snapshot must stay silent as well
    cm = case.cmarket()
    assert _gen(case, settings, mirror_features(case.contrary(), pivot), mirror_market(cm) if cm else None) is None


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_short_setup_is_mirror_of_long(case: Case, settings: Settings) -> None:
    """Mirrored snapshot -> mirrored levels (intraday); CNC strategies never emit SHORT."""
    pivot = 100.0
    long_sig = _gen(case, settings)
    mk = case.market()
    short_sig = _gen(case, settings, mirror_features(case.bull(), pivot), mirror_market(mk) if mk else None)
    if case.mode != Mode.INTRADAY:
        assert short_sig is None
        return
    assert long_sig is not None and short_sig is not None
    assert short_sig.entry_price == pytest.approx(2 * pivot - long_sig.entry_price, abs=TICK)
    assert abs(short_sig.rr - long_sig.rr) < 0.25


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_respects_higher_min_rr(case: Case, settings: Settings) -> None:
    sig = _gen(case, settings, min_rr=3.0)
    if sig is not None:  # natural-target strategies may legitimately decline
        assert sig.rr >= 3.0 - 1e-9


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_run_strategies_includes_signal(case: Case, settings: Settings) -> None:
    sigs = run_strategies(case.bull(), case.market(), case.mode, settings, case.now)
    names = [s.strategy for s in sigs]
    assert case.strategy in names
    for s in sigs:
        assert s.mode == case.mode
        assert s.rr >= settings.mode(case.mode).min_rr - 1e-9
        if case.mode != Mode.INTRADAY:
            assert s.direction == Direction.LONG
    # disabled (plain name and name:MODE) -> excluded
    assert case.strategy not in [s.strategy for s in run_strategies(
        case.bull(), case.market(), case.mode, settings, case.now, disabled={case.strategy})]
    assert case.strategy not in [s.strategy for s in run_strategies(
        case.bull(), case.market(), case.mode, settings, case.now, disabled={f"{case.strategy}:{case.mode.value}"})]


@pytest.mark.parametrize("mode", [Mode.BTST, Mode.SWING, Mode.POSITIONAL])
def test_cnc_modes_never_short(mode: Mode, settings: Settings) -> None:
    """build_signal refuses a SHORT setup in CNC modes (hard-coded guard)."""
    case = next(c for c in CASES if c.mode == mode)
    strat = get_strategy(case.strategy)
    feats = case.bull()
    setup = Setup(direction=Direction.SHORT, entry=feats.ltp, stop=feats.ltp + 2.0)
    with pytest.raises(OvernightShortError):
        strat.build_signal(setup, feats, mode, case.now, strat.default_params(), min_rr=2.0)
    # and run_strategies on mirrored (bearish) data emits nothing short
    mk = case.market()
    sigs = run_strategies(mirror_features(feats, 100.0), mirror_market(mk) if mk else None, mode, settings, case.now)
    assert all(s.direction == Direction.LONG for s in sigs)


def test_mean_reversion_only_in_ranging(settings: Settings) -> None:
    case = next(c for c in CASES if c.strategy == "mean_reversion")
    for regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.HIGH_VOLATILITY, Regime.UNDEFINED):
        assert _gen(case, settings, case.bull().model_copy(update={"regime": regime})) is None
    assert _gen(case, settings) is not None


def test_regime_misfit_flagged(settings: Settings) -> None:
    case = next(c for c in CASES if c.strategy == "opening_range_breakout")
    sig = _gen(case, settings, case.bull().model_copy(update={"regime": Regime.TRENDING_DOWN}))
    assert sig is not None and sig.regime_fit is False
    fit = _gen(case, settings)
    assert fit is not None and fit.regime_fit is True and fit.strength > sig.strength


def test_btst_needs_market_context(settings: Settings) -> None:
    case = next(c for c in CASES if c.strategy == "btst_strong_close")
    strat = get_strategy(case.strategy)
    assert strat.generate(case.bull(), None, strat.default_params(), case.now) is None


def test_btst_negative_market_blocks(settings: Settings) -> None:
    from core.enums import Signal
    from tests.strategies.strategies_kit import market

    case = next(c for c in CASES if c.strategy == "btst_strong_close")
    assert _gen(case, settings, mk=market(Signal.BEARISH, -0.5, {"IT": 1.0})) is None
    assert _gen(case, settings, mk=market(Signal.BULLISH, 0.5, {"IT": -1.0})) is None  # weak sector


def test_wrong_mode_is_silent(settings: Settings) -> None:
    case = next(c for c in CASES if c.strategy == "opening_range_breakout")
    strat = get_strategy(case.strategy)
    assert strat.generate(case.bull(), case.market(), {}, case.now, mode=Mode.SWING) is None


def test_targets_prefer_next_level(settings: Settings) -> None:
    """ORB: min T1 = 1.5R (101.375); the 102.0 resistance (2.33R) is within the cap -> T1 = 102."""
    case = next(c for c in CASES if c.strategy == "opening_range_breakout")
    sig = _gen(case, settings)
    assert sig is not None
    assert sig.stop_loss == pytest.approx(99.5)
    assert sig.target_1 == pytest.approx(102.0)
    assert sig.target_2 == pytest.approx(103.0)


def test_unknown_or_bad_params_fall_back(settings: Settings) -> None:
    case = next(c for c in CASES if c.strategy == "opening_range_breakout")
    strat = get_strategy(case.strategy)
    sig = strat.generate(case.bull(), case.market(), {"min_rvol": "abc", "bogus": 3}, case.now)
    assert sig is not None and sig.params["min_rvol"] == 1.5 and "bogus" not in sig.params


def test_signal_window_boundaries(settings: Settings) -> None:
    case = next(c for c in CASES if c.strategy == "opening_range_breakout")
    assert _gen(case, settings, now=at(9, 30)) is not None
    assert _gen(case, settings, now=at(11, 30)) is not None
    assert _gen(case, settings, now=at(11, 31)) is None
    assert _gen(case, settings, now=at(9, 29)) is None
