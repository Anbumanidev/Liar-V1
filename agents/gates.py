"""Deterministic pre-decision gates shared by the debate and the meta agent.

``trade_gate`` returns the reason why NO trade is possible for a stock/mode regardless of any
LLM opinion (code verdicts, rule 0.3/0.4), or None. The meta agent turns a gate into
``NO_TRADE`` without calling the model and the debate is SKIPPED, so no reasoning-model time
is spent on a setup the code already rules out:

* no strategy signal for the mode (or none whose direction the mode allows - no overnight
  shorts in BTST/SWING/POSITIONAL);
* the fundamental agent's TRADEABLE=false verdict (also when that agent FAILED);
* a blocking event inside the holding window (event-risk agent code verdict, only produced
  when ``settings.risk.avoid_events_in_window`` is on);
* no confluence score, or a confluence grade / score below the mode minimum.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from agents.aggregate import VoteSummary, code_facts
from core.enums import AgentStatus, Grade
from core.types import AgentReport, StockContext, StrategySignal
from modes.base import direction_allowed

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

#: Grade ordering (C = below every minimum).
GRADE_RANK: dict[Grade, int] = {Grade.C: 0, Grade.B: 1, Grade.A: 2, Grade.A_PLUS: 3}


def grade_at_least(grade: Grade, minimum: Grade) -> bool:
    """True if ``grade`` meets ``minimum`` (A+ > A > B > C)."""
    return GRADE_RANK[Grade(grade)] >= GRADE_RANK[Grade(minimum)]


def usable_signals(ctx: StockContext) -> tuple[list[int], dict[int, str]]:
    """Indices of ``ctx.strategy_signals`` usable in ``ctx.mode`` and the excluded ones with reasons."""
    usable: list[int] = []
    excluded: dict[int, str] = {}
    for i, s in enumerate(ctx.strategy_signals):
        if s.mode != ctx.mode:
            excluded[i] = f"signal is for {s.mode.value}, not {ctx.mode.value}"
        elif not direction_allowed(ctx.mode, s.direction):
            excluded[i] = f"{s.direction.value} is not allowed in {ctx.mode.value} (no overnight shorts)"
        else:
            usable.append(i)
    return usable, excluded


def strongest_usable(ctx: StockContext) -> tuple[int, StrategySignal] | None:
    """(index, signal) of the strongest usable signal (ties -> higher R:R, then name)."""
    usable, _ = usable_signals(ctx)
    if not usable:
        return None
    i = sorted(usable, key=lambda k: (-ctx.strategy_signals[k].strength, -ctx.strategy_signals[k].rr,
                                      ctx.strategy_signals[k].strategy))[0]
    return i, ctx.strategy_signals[i]


def not_tradeable_reasons(reports: Sequence[AgentReport], vote: VoteSummary | None) -> list[str]:
    """Reasons from the fundamental agent's TRADEABLE=false code verdict (empty = tradeable)."""
    reasons: list[str] = []
    if vote is not None and not vote.tradeable:
        reasons.extend(vote.tradeable_reasons or ["fundamental: not tradeable"])
    for r in reports:
        if r.agent == "fundamental" and r.tradeable is False and not reasons:
            reasons.extend(code_facts(r).get("blocking_flags") or ["fundamental: not tradeable"])
    return reasons


def blocking_events(reports: Sequence[AgentReport], vote: VoteSummary | None) -> list[str]:
    """Code-owned blocking events in the holding window (event-risk agent)."""
    events: list[str] = list(vote.blocking_events) if vote is not None else []
    for r in reports:
        if r.agent == "event_risk" and r.status != AgentStatus.SKIPPED:
            for e in code_facts(r).get("blocking_events") or []:
                if str(e) not in events:
                    events.append(str(e))
    return events


def trade_gate(ctx: StockContext, reports: Sequence[AgentReport], vote: VoteSummary | None,
               settings: "Settings") -> str | None:
    """Reason why no trade is possible (see module docstring), or None."""
    if not ctx.strategy_signals:
        return "no strategy signal"
    usable, excluded = usable_signals(ctx)
    if not usable:
        return "no usable strategy signal: " + "; ".join(sorted(set(excluded.values())))[:300]
    reasons = not_tradeable_reasons(reports, vote)
    if reasons:
        return "fundamental TRADEABLE=false: " + "; ".join(reasons[:3])
    events = blocking_events(reports, vote)
    if events:
        return "blocking event(s) in the holding window: " + "; ".join(events[:3])
    cfg = settings.mode(ctx.mode)
    conf = ctx.confluence
    if conf is None:
        return "no confluence score available"
    if not grade_at_least(conf.grade, cfg.min_grade):
        return f"confluence grade {conf.grade.value} is below the {ctx.mode.value} minimum {cfg.min_grade.value}"
    if conf.score < cfg.min_confluence:
        return f"confluence {conf.score:.1f} is below the {ctx.mode.value} minimum {cfg.min_confluence:.1f}"
    return None
