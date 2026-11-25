"""Recommended settings upgrade: one audited, idempotent, revertible version."""

from __future__ import annotations

from config.schema import HardCaps
from config.store import ensure_seeded, get_active_settings, update_settings
from control.upgrades import RECOMMENDED_2026_11_25, apply_recommended, pending_changes


def test_upgrade_applies_once_and_is_idempotent(session, caps: HardCaps) -> None:  # type: ignore[no-untyped-def]
    seeded = ensure_seeded(session, caps)
    # simulate a database seeded before the post-mortem (old profile / stop distance)
    _, cur = get_active_settings(session, caps)
    old = update_settings(cur, {"models.profiles.low.reasoning": "qwen3:8b", "risk.sl_min_atr": 0.3,
                                "modes.modes.INTRADAY.reentry_cooldown_minutes": 0})
    from config.store import save_settings

    save_settings(session, old, caps, author="old")
    todo = pending_changes(session, caps)
    assert todo["risk.sl_min_atr"] == 0.8 and todo["modes.modes.INTRADAY.reentry_cooldown_minutes"] == 60
    assert set(todo) <= set(RECOMMENDED_2026_11_25)
    vid, applied = apply_recommended(session, caps, author="test")
    assert vid is not None and vid > seeded.id and applied == todo
    _, now = get_active_settings(session, caps)
    assert now.models.model_for("meta") == "qwen3:4b" and now.risk.sl_min_atr == 0.8
    assert now.mode("INTRADAY").max_signal_age_minutes == 6
    assert apply_recommended(session, caps) == (None, {})
