"""Deterministic weighted vote over sub-agent reports (section 4, rule 0.4).

score = 100 * sum(w_i * s_i * c_i / 100) / sum(w_i)   over agents with weight w_i > 0

where s_i = +1 BULLISH, -1 BEARISH, 0 NEUTRAL and c_i the confidence. FAILED and SKIPPED
agents count as NEUTRAL (they stay in the denominator, diluting the score) - a broken agent
can never push a trade through. ``tradeable`` is False when the fundamental agent says so
(also when that agent FAILED: the TRADEABLE verdict is a code fact, see ``agents.base``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from core.enums import AgentStatus, Signal
from core.types import AgentReport

#: Weight for an agent missing from the weights mapping.
DEFAULT_WEIGHT: float = 1.0

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
    total_weight: float = 0.0
    contributions: dict[str, float] = Field(default_factory=dict)  # agent -> signed points
    tradeable_reasons: list[str] = Field(default_factory=list)
    event_risk: bool = False                # event_risk agent flagged a blocking event
    fundamental_score: float | None = None  # SWING/POSITIONAL fundamental score, if any

    @property
    def signal(self) -> Signal:
        """Majority direction of the score (|score| < 10 -> NEUTRAL)."""
        if self.score >= 10:
            return Signal.BULLISH
        if self.score <= -10:
            return Signal.BEARISH
        return Signal.NEUTRAL


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
    fundamental_score: float | None = None

    for rep in reports:
        if rep.status == AgentStatus.FAILED:
            failed += 1
        elif rep.status == AgentStatus.SKIPPED:
            skipped += 1
        sig = rep.effective_signal
        if sig == Signal.BULLISH:
            bullish += 1
        elif sig == Signal.BEARISH:
            bearish += 1
        else:
            neutral += 1

        if rep.agent == "fundamental" and rep.tradeable is False:
            tradeable = False
            reasons.extend(rep.extra.get("blocking_flags") or rep.key_findings[:2] or ["fundamental: not tradeable"])
        if rep.agent == "fundamental" and rep.extra.get("fundamental_score") is not None:
            fundamental_score = float(rep.extra["fundamental_score"])
        if rep.agent == "event_risk" and rep.status != AgentStatus.SKIPPED and rep.extra.get("event_risk"):
            event_risk = True

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
        tradeable_reasons=reasons, event_risk=event_risk, fundamental_score=fundamental_score,
    )
