"""Config, hard caps and settings versioning tests (Phase 1)."""

from __future__ import annotations

import pytest

from config.loader import (
    clamp_to_caps,
    clamp_value,
    get_cap,
    load_default_settings,
    update_settings,
    validate_against_caps,
)
from config.schema import HardCaps, Settings
from config.store import (
    apply_changes,
    diff_settings,
    get_active_settings,
    list_versions,
    revert_to,
)
from core.enums import ExecutionPolicy, Mode
from core.errors import HardCapViolationError
from storage.models import AuditLog


def test_default_settings_load(settings: Settings) -> None:
    assert settings.capital.starting_capital == 1000
    assert settings.mode(Mode.INTRADAY).allocation_pct == 50
    assert settings.mode(Mode.SWING).allocation_pct == 50
    assert settings.mode(Mode.BTST).allocation_pct == 0
    assert settings.mode(Mode.POSITIONAL).allocation_pct == 0
    for m in Mode:
        assert settings.mode(m).execution_policy == ExecutionPolicy.MANUAL_APPROVE
    assert settings.risk.max_loss_pct == 10 and settings.risk.hard_stop_pct == 9
    assert settings.mode(Mode.INTRADAY).min_rr == 1.5
    assert settings.mode(Mode.SWING).min_rr == 2.0


def test_default_settings_within_caps(settings: Settings, caps: HardCaps) -> None:
    assert validate_against_caps(settings, caps) == []


def test_allocations_over_100_rejected(settings: Settings) -> None:
    with pytest.raises(ValueError):
        update_settings(settings, {"modes.modes.BTST.allocation_pct": 10})


def test_hard_caps_invariants_hard_coded() -> None:
    for bad in ({"allow_overnight_shorts": True}, {"sl_tighten_only": False},
                {"allow_averaging_down": True}, {"max_loss_pct": 11}):
        with pytest.raises(ValueError):
            HardCaps(**bad)


def test_cap_violation_detected_and_clamped(settings: Settings, caps: HardCaps) -> None:
    bad = update_settings(settings, {"risk.mis_leverage": 20.0, "risk.risk_per_trade_pct": 3.0})
    violations = validate_against_caps(bad, caps)
    assert any("mis_leverage" in v for v in violations)
    assert any("risk_per_trade_pct" in v for v in violations)
    fixed, adjustments = clamp_to_caps(bad, caps)
    assert fixed.risk.mis_leverage == caps.max_mis_leverage
    assert fixed.risk.risk_per_trade_pct == caps.max_risk_per_trade_pct
    assert len(adjustments) == 2
    assert validate_against_caps(fixed, caps) == []


def test_clamp_value_and_get_cap(caps: HardCaps) -> None:
    assert get_cap("risk.mis_leverage", caps) == ("max", caps.max_mis_leverage)
    assert clamp_value("risk.mis_leverage", 99, caps) == caps.max_mis_leverage
    assert clamp_value("modes.modes.INTRADAY.min_rr", 0.5, caps) == caps.min_rr_floor
    assert get_cap("capital.profit_lock_pct", caps) is None


def test_max_loss_pct_cannot_exceed_10() -> None:
    s = load_default_settings()
    with pytest.raises(ValueError):
        update_settings(s, {"risk.max_loss_pct": 12})


def test_hard_stop_must_be_below_max_loss(settings: Settings) -> None:
    with pytest.raises(ValueError):
        update_settings(settings, {"risk.hard_stop_pct": 10})


def test_diff_settings() -> None:
    d = diff_settings({"a": 1, "b": {"c": 2, "d": 3}}, {"a": 1, "b": {"c": 5, "d": 3}, "e": 1})
    assert {"path": "b.c", "old": 2, "new": 5} in d
    assert {"path": "e", "old": None, "new": 1} in d
    assert len(d) == 2


def test_versioning_seed_change_revert(session, caps: HardCaps, sim_clock) -> None:  # type: ignore[no-untyped-def]
    v1, s1 = get_active_settings(session, caps, clock=sim_clock)
    assert s1.capital.profit_lock_pct == 30
    row2 = apply_changes(session, {"capital.profit_lock_pct": 40}, caps, author="owner", note="lock more",
                         clock=sim_clock)
    assert row2.id != v1
    assert row2.diff_json == [{"path": "capital.profit_lock_pct", "old": 30.0, "new": 40.0}]
    v2, s2 = get_active_settings(session, caps)
    assert v2 == row2.id and s2.capital.profit_lock_pct == 40
    # no-op change does not create a version
    same = apply_changes(session, {"capital.profit_lock_pct": 40}, caps, author="owner")
    assert same.id == row2.id
    # cap violation rejected, nothing saved
    with pytest.raises(HardCapViolationError):
        apply_changes(session, {"risk.mis_leverage": 50.0}, caps, author="owner")
    # revert to v1
    row3 = revert_to(session, v1, caps, author="owner", clock=sim_clock)
    v3, s3 = get_active_settings(session, caps)
    assert v3 == row3.id and row3.reverted_from_id == v1
    assert s3.capital.profit_lock_pct == 30
    assert len(list_versions(session)) == 3
    actions = [a.action for a in session.query(AuditLog).all()]
    assert actions.count("SETTINGS_CHANGE") == 1 and "SETTINGS_REVERT" in actions and "SETTINGS_SEED" in actions


def test_model_for_agent(settings: Settings) -> None:
    assert settings.models.model_for("meta") == "qwen3:8b"
    assert settings.models.model_for("technical") == "qwen3:4b"
    s = update_settings(settings, {"models.assignments": {"technical": "qwen3:8b"}})
    assert s.models.model_for("technical") == "qwen3:8b"
