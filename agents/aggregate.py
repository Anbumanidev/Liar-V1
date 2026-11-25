"""Deterministic weighted vote over sub-agent reports (section 4, rule 0.4).

score = 100 * sum(w_i * s_i * c_i / 100) / sum(w_i)   over VOTING agents with weight w_i > 0

where s_i = +1 BULLISH, -1 BEARISH, 0 NEUTRAL and c_i the confidence.

* FAILED agents vote NEUTRAL with confidence 0 and **stay in the denominator** (they dilute
  the score) - a broken agent can never push a trade through (rule 0.4).
* SKIPPED agents (not applicable to the mode, or disabled from the dashboard) are NEUTRAL and
  **do not vote**: they are excluded from the denominator, so disabling an agent removes it
  from the panel instead of silently making every trade harder to reach the thresholds.
* Code-owned facts survive an LLM failure: a FAILED report keeps its deterministic facts
  under ``extra['fallback']`` (see ``agents.base.SubAgent.failed_report``), so
  ``tradeable=False`` (fundamental), the fundamental score and the event-risk flag /
  blocking events are read from there too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from core.enums import AgentStatus, Signal
from core.types import AgentReport

#: Weight for an agent missing from the weights mapping.
DEFAULT_WEIGHT: float = 1.0
#: |score| below which the vote is NEUTRAL.
SIGNAL_THRESHOLD: float = 10.0

_SIGN: dict[Signal, int] = {Signal.BULLISH: 1, Signal.BEARISH: -1, Signal.NEUTRAL: 0}


class VoteSummary(BaseModel):
    """Result of ``weighted_vote``."""

    score: float = Field(ge=-100, le=100)   # -100 (all bearish, max confidence) .. +100
    bullish: int = 0
    bearish: int = 0
    neutral: int = 0                        # includes FAILED and SKIPPED agents
    failed: int = 0
    skipped: int = 0
    tradeable: bool = True
    total_weight: float = 0.0               # sum of weights of the voting (non-SKIPPED) agents
    contributions: dict[str, float] = Field(default_factory=dict)  # agent -> signed points
    tradeable_reasons: list[str] = Field(default_factory=list)
    event_risk: bool = False                # event_risk agent: any event / gap risk in the holding window
    blocking_events: list[str] = Field(default_factory=list)  # code-owned blocking events (risk manager vetoes)
    fundamental_score: float | None = None  # SWING/POSITIONAL fundamental score, if any
    failed_agents: list[str] = Field(default_factory=list)

    @property
    def signal(self) -> Signal:
        """Majority direction of the score (|score| < 10 -> NEUTRAL)."""
        if self.score >= SIGNAL_THRESHOLD:
            return Signal.BULLISH
        if self.score <= -SIGNAL_THRESHOLD:
            return Signal.BEARISH
        return Signal.NEUTRAL


def code_facts(rep: AgentReport) -> dict[str, Any]:
    """Code-owned facts of a report: ``extra``, or the deterministic fallback of a FAILED report."""
    if rep.status == AgentStatus.FAILED:
        fb = rep.extra.get("fallback")
        return fb if isinstance(fb, dict) else {}
    return rep.extra


def weighted_vote(reports: Sequence[AgentReport], weights: Mapping[str, float] | None = None) -> VoteSummary:
    """Aggregate reports with per-agent weights (e.g. ``settings.mode(m).agent_weights``)."""
    weights = weights or {}
    num = 0.0
    den = 0.0
    bullish = bearish = neutral = failed = skipped = 0
    contributions: dict[str, float] = {}
    tradeable = True
    reasons: list[str] = []
    event_risk = False
    blocking_events: list[str] = []
    fundamental_score: float | None = None
    failed_agents: list[str] = []

    for rep in reports:
        if rep.status == AgentStatus.FAILED:
            failed += 1
            failed_agents.append(rep.agent)
        elif rep.status == AgentStatus.SKIPPED:
            skipped += 1
        sig = rep.effective_signal
        if sig == Signal.BULLISH:
            bullish += 1
        elif sig == Signal.BEARISH:
            bearish += 1
        else:
            neutral += 1

        facts = code_facts(rep)
        if rep.agent == "fundamental":
            if rep.tradeable is False:
                tradeable = False
                reasons.extend(facts.get("blocking_flags") or rep.key_findings[:2]
                               or ["fundamental: not tradeable"])
            if facts.get("fundamental_score") is not None:
                fundamental_score = float(facts["fundamental_score"])
        if rep.agent == "event_risk" and rep.status != AgentStatus.SKIPPED:
            if facts.get("event_risk"):
                event_risk = True
            blocking_events.extend(str(b) for b in (facts.get("blocking_events") or []))

        if rep.status == AgentStatus.SKIPPED:
            continue  # not part of the panel for this mode
        w = float(weights.get(rep.agent, DEFAULT_WEIGHT))
        if w <= 0:
            continue
        pts = w * _SIGN[sig] * (rep.confidence if rep.status == AgentStatus.OK else 0) / 100.0
        contributions[rep.agent] = round(pts, 4)
        num += pts
        den += w

    score = 0.0 if den <= 0 else max(-100.0, min(100.0, 100.0 * num / den))
    return VoteSummary(
        score=round(score, 2), bullish=bullish, bearish=bearish, neutral=neutral, failed=failed,
        skipped=skipped, tradeable=tradeable, total_weight=round(den, 4), contributions=contributions,
        tradeable_reasons=reasons, event_risk=event_risk, blocking_events=blocking_events,
        fundamental_score=fundamental_score, failed_agents=failed_agents,
    )
