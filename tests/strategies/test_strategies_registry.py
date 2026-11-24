"""Registry: declarations, parameter resolution/clamping, enable rules, fault isolation, YAML."""

from __future__ import annotations

import pytest
from loguru import logger

from config.loader import load_default_settings
from config.schema import Settings, StrategyConfig
from core.clock import parse_hhmm
from core.enums import Direction, Mode
from strategies.base import ParamSpec, build_targets, in_windows, pick_stop, round_tick
from strategies.registry import (
    all_strategies,
    get_strategy,
    resolve_params,
    run_strategies,
    strategies_for,
    strategy_enabled,
)
from tests.strategies.strategies_cases import CASES
from tests.strategies.strategies_kit import at

EXPECTED = {
    Mode.INTRADAY: {"opening_range_breakout", "vwap_pullback", "ema_9_21_pullback", "cpr_breakout", "gap_and_go",
                    "gap_fill", "relative_strength_momentum", "mean_reversion", "supertrend_adx_15m"},
    Mode.BTST: {"btst_strong_close"},
    Mode.SWING: {"swing_base_breakout", "swing_ema_pullback", "swing_reversal_divergence", "swing_rs_leader"},
    Mode.POSITIONAL: {"positional_weekly_trend", "positional_52w_breakout", "positional_sector_rotation"},
}


@pytest.fixture
def warnings_log():
    msgs: list[str] = []
    hid = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    yield msgs
    logger.remove(hid)


def _with_params(settings: Settings, name: str, **params) -> Settings:
    s = settings.model_copy(deep=True)
    s.strategies.strategies[name] = StrategyConfig(enabled=True, modes=list(get_strategy(name).modes), params=params)
    return s


def test_all_strategies_registered_per_mode() -> None:
    for mode, names in EXPECTED.items():
        assert {s.name for s in strategies_for(mode)} == names
    assert len(all_strategies()) == sum(len(v) for v in EXPECTED.values())


def test_declarations_complete() -> None:
    for s in all_strategies():
        assert s.description and s.regimes and s.param_specs
        for m in s.modes:
            for start, end in s.windows[m]:
                assert parse_hhmm(start) < parse_hhmm(end)
            if m == Mode.INTRADAY:  # inside the intraday entry window (risk 09:30-14:45)
                for start, end in s.windows[m]:
                    assert parse_hhmm(start) >= parse_hhmm("09:30") and parse_hhmm(end) <= parse_hhmm("14:45")
        for p in s.param_specs:
            assert p.clamp(p.default) == p.default


def test_yaml_matches_registry(settings: Settings) -> None:
    cfg = load_default_settings().strategies.strategies
    assert set(cfg) == {s.name for s in all_strategies()}
    for s in all_strategies():
        c = cfg[s.name]
        assert c.enabled is True
        assert set(c.modes) == set(s.modes)
        assert c.params == s.default_params()
        assert resolve_params(s.name, settings) == s.default_params()


def test_resolve_params_overlay_and_clamp(settings: Settings, warnings_log: list[str]) -> None:
    s = _with_params(settings, "opening_range_breakout", min_rvol=99, buffer_atr=-1, max_or_width_atr="3.5",
                     unknown_param=7)
    p = resolve_params("opening_range_breakout", s)
    assert p["min_rvol"] == 5.0           # clamped to max
    assert p["buffer_atr"] == 0.0         # clamped to min
    assert p["max_or_width_atr"] == 3.5   # string coerced
    assert "unknown_param" not in p
    assert any("unknown parameter" in m for m in warnings_log)
    assert any("clamped" in m for m in warnings_log)


def test_resolve_params_invalid_value_uses_default(settings: Settings, warnings_log: list[str]) -> None:
    s = _with_params(settings, "btst_strong_close", min_rvol="lots", require_delivery="yes")
    p = resolve_params("btst_strong_close", s)
    assert p["min_rvol"] == 1.3
    assert p["require_delivery"] is True
    assert any("invalid value" in m for m in warnings_log)


def test_int_params_are_ints(settings: Settings) -> None:
    s = _with_params(settings, "swing_rs_leader", max_holding_days=7.6)
    assert resolve_params("swing_rs_leader", s)["max_holding_days"] == 8
    s = _with_params(settings, "swing_rs_leader", max_holding_days=50)
    assert resolve_params("swing_rs_leader", s)["max_holding_days"] == 10


def test_paramspec_validation() -> None:
    with pytest.raises(ValueError):
        ParamSpec("x", 5.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        ParamSpec("x", 0.5, 1.0, 0.0)
    with pytest.raises(ValueError):
        ParamSpec("x", 1, 0, 2, kind="str")
    spec = ParamSpec("b", False, kind="bool")
    assert spec.clamp("true") is True and spec.clamp(0) is False
    with pytest.raises(ValueError):
        spec.coerce("maybe")
    with pytest.raises(ValueError):
        ParamSpec("f", 1.0, 0.0, 2.0).coerce(float("nan"))


def test_strategy_enabled_rules(settings: Settings) -> None:
    assert strategy_enabled("vwap_pullback", Mode.INTRADAY, settings)
    assert not strategy_enabled("vwap_pullback", Mode.SWING, settings)          # unsupported mode
    assert not strategy_enabled("nope", Mode.INTRADAY, settings)                # unknown
    assert not strategy_enabled("vwap_pullback", Mode.INTRADAY, settings, {"vwap_pullback"})
    assert not strategy_enabled("vwap_pullback", Mode.INTRADAY, settings, {"vwap_pullback:INTRADAY"})
    assert strategy_enabled("vwap_pullback", Mode.INTRADAY, settings, {"vwap_pullback:SWING"})
    s = settings.model_copy(deep=True)
    s.strategies.strategies["vwap_pullback"].enabled = False
    assert not strategy_enabled("vwap_pullback", Mode.INTRADAY, s)
    s = settings.model_copy(deep=True)
    s.strategies.strategies["vwap_pullback"].modes = [Mode.SWING]
    assert not strategy_enabled("vwap_pullback", Mode.INTRADAY, s)
    s = settings.model_copy(deep=True)
    del s.strategies.strategies["vwap_pullback"]
    assert strategy_enabled("vwap_pullback", Mode.INTRADAY, s)  # absent from settings -> coded defaults


def test_run_strategies_isolates_failures(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    case = next(c for c in CASES if c.strategy == "opening_range_breakout")
    broken = get_strategy("vwap_pullback")

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(broken, "evaluate", boom)
    sigs = run_strategies(case.bull(), case.market(), Mode.INTRADAY, settings, case.now)
    names = {s.strategy for s in sigs}
    assert "vwap_pullback" not in names and "opening_range_breakout" in names


def test_run_strategies_drops_invalid_signal(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    case = next(c for c in CASES if c.strategy == "opening_range_breakout")
    strat = get_strategy("opening_range_breakout")
    real = strat.generate

    def bad(*a, **k):
        sig = real(*a, **k)
        return sig.model_copy(update={"target_1": sig.entry_price + 0.1}) if sig else sig

    monkeypatch.setattr(strat, "generate", bad)
    sigs = run_strategies(case.bull(), case.market(), Mode.INTRADAY, settings, case.now)
    assert "opening_range_breakout" not in {s.strategy for s in sigs}


def test_run_strategies_sorted_and_without_settings() -> None:
    case = next(c for c in CASES if c.strategy == "opening_range_breakout")
    sigs = run_strategies(case.bull(), case.market(), Mode.INTRADAY, None, case.now)
    assert sigs and [s.strength for s in sigs] == sorted((s.strength for s in sigs), reverse=True)
    assert all(s.rr >= 1.5 - 1e-9 for s in sigs)


def test_run_strategies_respects_settings_min_rr(settings: Settings) -> None:
    s = settings.model_copy(deep=True)
    s.modes.modes[Mode.INTRADAY].min_rr = 2.5
    case = next(c for c in CASES if c.strategy == "opening_range_breakout")
    sigs = run_strategies(case.bull(), case.market(), Mode.INTRADAY, s, case.now)
    assert sigs and all(x.rr >= 2.5 - 1e-9 for x in sigs)


# ------------------------------------------------------------------ base helpers


def test_round_tick() -> None:
    assert round_tick(100.03) == 100.05
    assert round_tick(100.07, "down") == 100.05
    assert round_tick(100.01, "up") == 100.05
    assert round_tick(100.05, "up") == 100.05 and round_tick(100.05, "down") == 100.05


def test_in_windows() -> None:
    w = [("09:30", "11:30"), ("13:00", "14:00")]
    assert in_windows(at(9, 30), w) and in_windows(at(13, 30), w)
    assert not in_windows(at(12, 0), w) and not in_windows(at(9, 29), w)


def test_pick_stop() -> None:
    # structural candidate within bounds
    assert pick_stop(Direction.LONG, 100, [99.0], 1.0, min_atr=0.3, max_atr=3, fallback_atr=1.5) == 99.0
    # too tight -> widened to min
    assert pick_stop(Direction.LONG, 100, [99.9], 1.0, min_atr=0.3, max_atr=3, fallback_atr=1.5) == pytest.approx(99.7)
    # too wide -> fallback; wrong side ignored
    assert pick_stop(Direction.LONG, 100, [95.0, 101.0], 1.0, min_atr=0.3, max_atr=3, fallback_atr=1.5) == 98.5
    assert pick_stop(Direction.SHORT, 100, [101.0], 1.0, min_atr=0.3, max_atr=3, fallback_atr=1.5) == 101.0
    assert pick_stop(Direction.LONG, 100, [], 0.0, min_atr=0.3, max_atr=3, fallback_atr=1.5) is None


def test_build_targets() -> None:
    # no levels -> min_rr and t2_r multiples
    t1, t2 = build_targets(Direction.LONG, 100, 99, 1.5, [], t2_r=3.0)
    assert (t1, t2) == (101.5, 103.0)
    # nearer level ignored, farther level within cap used
    t1, t2 = build_targets(Direction.LONG, 100, 99, 1.5, [101.0, 102.5, 104.0], t2_r=3.0)
    assert t1 == 102.5 and t2 == 104.0
    # level beyond cap -> min R:R
    t1, _ = build_targets(Direction.SHORT, 100, 101, 2.0, [90.0], t2_r=3.0)
    assert t1 == 98.0
    with pytest.raises(ValueError):
        build_targets(Direction.LONG, 100, 100, 1.5, [])
