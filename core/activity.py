"""Human-readable live activity feed: what the engine and the AI models are doing right now.

Every call writes ONE line to the terminal / ``engine.log`` (loguru) and one JSON object to
``logs/<day>/activity.jsonl``, which the web dashboard streams as its live log.

Event schema (always present: ``ts``, ``category="activity"``, ``stage``, ``level``, ``message``)::

    stage   extra fields
    engine  state, next_job, next_at                 start / stop / idle status
    cycle   job, mode, symbols, actions, seconds      a scheduled scan starting / finishing
    screen  symbol, mode, result, strategy, direction, confluence, grade, reason
            result: no_setup | filtered | candidate
    agents  symbol, mode                               the 10 specialist agents starting / finished (vote)
    llm     symbol, mode, agent, model, event (start|done), status, signal, confidence, latency_ms, error
    behaviour symbol, mode, verdict, range_pos, range_vs_adr   price-behaviour check before the risk manager
    plan    next-day plan (15:40) and opening check (09:20) messages
    watchlist follow-up list changes
    debate  symbol, mode, bull, bear, rounds, latency_ms
    meta    symbol, mode, decision, confidence, strategy, why_not, latency_ms
    risk    symbol, mode, approved, rule, qty
    route   symbol, mode, final_action, decision_id
    command command_id, type, status

Never raises (logging must never break trading).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from core.clock import Clock
from core.logging import log_event

_LEVELS = {"debug": "DEBUG", "info": "INFO", "success": "SUCCESS", "warning": "WARNING", "error": "ERROR"}


def activity(stage: str, message: str, *, clock: Clock | None = None, level: str = "info",
             terminal: bool = True, **fields: Any) -> None:
    """Record one activity event (terminal line + ``activity.jsonl``)."""
    try:
        lvl = _LEVELS.get(level, "INFO")
        clean = {k: v for k, v in fields.items() if v is not None}
        log_event("activity", {"stage": stage, "level": lvl.lower(), "message": message, **clean}, clock=clock)
        if terminal:
            logger.opt(depth=1).log(lvl, f"[{stage}] {message}")
    except Exception:  # noqa: BLE001
        pass


__all__ = ["activity"]
