"""Recommended settings upgrades, applied as ONE audited settings version (revertible).

Settings live in the database as versions; editing config/*.yaml only affects a fresh
database. ``python main.py upgrade-settings`` applies the values below to the active version
so the running system gets them, with a note explaining why. Revert from the dashboard
(Settings history) if you disagree with any of them.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from config.schema import HardCaps
from config.store import apply_changes, get_active_settings
from core.clock import Clock

#: 2026-11-25 post-mortem: 3/3 INTRADAY losses from stale, chased, quorum-less entries.
RECOMMENDED_2026_11_25: dict[str, Any] = {
    # one 4B model on a 6 GB GPU: no model swapping, no timeouts
    "models.profiles": {
        "low": {"reasoning": "qwen3:4b", "sub_agent": "qwen3:4b"},
        "balanced": {"reasoning": "qwen3:8b", "sub_agent": "qwen3:4b"},
        "high": {"reasoning": "qwen3:14b", "sub_agent": "qwen3:8b"},
    },
    "models.active_profile": "low",
    "models.num_ctx": 6144,
    "models.request_timeout_seconds": 90,
    "models.timeout_retries": 1,
    "models.max_concurrent_requests": 2,
    "models.finbert_device": "cpu",
    # only the best setups reach the LLM, inside the cycle
    "engine.max_llm_candidates_per_cycle": 2,
    "engine.cycle_llm_budget_seconds": 240,
    # execution-quality gates
    "risk.sl_min_atr": 0.8,
    "risk.max_failed_agents_pct": 34.0,
    "risk.debate_veto_enabled": True,
    "risk.max_debate_deficit": 10,
    "risk.max_entry_drift_atr": 1.0,
    "risk.live_rr_tolerance": 0.8,
    "risk.min_room_r": 1.0,
    "risk.fill_rr_min": 1.0,
    "modes.modes.INTRADAY.entry_windows": [{"start": "09:30", "end": "11:45"}, {"start": "13:15", "end": "14:45"}],
    "modes.modes.INTRADAY.reentry_cooldown_minutes": 60,
    "modes.modes.INTRADAY.max_signal_age_minutes": 6,
    "modes.modes.BTST.reentry_cooldown_minutes": 1440,
    "modes.modes.BTST.max_signal_age_minutes": 15,
    "modes.modes.SWING.reentry_cooldown_minutes": 2880,
    "modes.modes.POSITIONAL.reentry_cooldown_minutes": 7200,
}

NOTE_2026_11_25 = ("2026-11-25 post-mortem fixes: one qwen3:4b model (6 GB GPU), 2 concurrent LLM calls, top-2 "
                   "setups per cycle, stale/chased/quorum/debate/room/re-entry gates, SL >= 0.8 ATR, "
                   "INTRADAY skips 11:45-13:15")


def pending_changes(session: Session, caps: HardCaps, changes: dict[str, Any] | None = None, *,
                    clock: Clock | None = None) -> dict[str, Any]:
    """Subset of ``changes`` that differs from the active settings."""
    from config.loader import settings_to_dict

    changes = changes or RECOMMENDED_2026_11_25
    _vid, current = get_active_settings(session, caps, clock=clock)
    data = settings_to_dict(current)
    out: dict[str, Any] = {}
    for path, value in changes.items():
        node: Any = data
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node != value:
            out[path] = value
    return out


def apply_recommended(session: Session, caps: HardCaps, *, author: str = "upgrade",
                      clock: Clock | None = None) -> tuple[int | None, dict[str, Any]]:
    """Apply the recommended values; returns (new version id or None, applied changes). Caller commits."""
    todo = pending_changes(session, caps, clock=clock)
    if not todo:
        return None, {}
    row = apply_changes(session, todo, caps, author=author, note=NOTE_2026_11_25, clock=clock)
    return row.id, todo


__all__ = ["NOTE_2026_11_25", "RECOMMENDED_2026_11_25", "apply_recommended", "pending_changes"]
