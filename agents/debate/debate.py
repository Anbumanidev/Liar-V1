"""Layer-2 debate (section 4): Bull argues FOR the trade, Bear AGAINST it.

Flow (``run_debate``):

1. Gates: debate disabled (``settings.agents.debate_enabled`` False or the ``bull`` / ``bear``
   agent disabled) -> ``SKIPPED``. Nothing to debate (``agents.gates.trade_gate``: no usable
   strategy setup, TRADEABLE=false, blocking event, confluence below the mode minimum) ->
   ``SKIPPED`` too: the meta agent returns NO_TRADE in those cases without asking a model,
   so no reasoning-model time is spent.
2. Deterministic facts (``debate_facts``): the reference setup, confluence, weighted vote,
   condensed sub-agent reports, market backdrop and - for the Bear - the costs and risks it
   must weigh: round-trip charges for ONE share
   (``paper_engine.charges.estimate_round_trip(product, direction, 1, entry, target_1)``) vs
   the reward per share, and for BTST/SWING/POSITIONAL the overnight gap buffer
   (``settings.risk.gap_buffer_pct``), events in the holding window and the holding period.
   Every number is computed here; the LLMs only argue (rule 0.3).
3. Opening statements (round 0, independent), then ``settings.agents.debate_rounds``
   rebuttal rounds (0 allowed): Bull answers the Bear's latest turn, then the Bear answers
   the Bull's. Every turn is a Pydantic-validated JSON answer (``DebateTurnOutput``, <= the
   client's retries) appended to the transcript ``[{round, speaker, phase, content, ...}]``.
4. Result ``DebateResult(bull=DebateSide, bear=DebateSide, rounds, transcript)``.

Any failure (LLM failed after retries, prompt error, unexpected exception) -> ``FAILED``
with empty (None) sides and the partial transcript kept for audit; ``run_debate`` never
raises. With the ``rules`` backend a deterministic debate is produced instead
(``deterministic_debate``: convictions from the vote / confluence / risk facts, no rebuttals).
Models: ``settings.models.model_for('bull'|'bear')`` with ``think=settings.models.reasoning_think``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.aggregate import VoteSummary, code_facts
from agents.base import DIRECTION_SIGNAL, current_price, mode_rules_text, r2
from agents.gates import strongest_usable, trade_gate
from agents.schemas import clamp_int, output_schema_summary, truncate_list, truncate_text
from core.clock import to_ist
from core.enums import AgentStatus, Direction, Signal
from core.logging import log_event, to_jsonable
from core.types import AgentReport, DebateResult, DebateSide, StockContext, StrategySignal
from modes.base import is_overnight, product_for
from paper_engine.charges import estimate_round_trip

if TYPE_CHECKING:  # pragma: no cover
    from datetime import datetime

    from agents.llm import LLMClient
    from agents.prompt_store import PromptStore
    from config.schema import Settings

BULL = "bull"
BEAR = "bear"
OPENING = "opening"
REBUTTAL = "rebuttal"

MAX_ARGUMENTS = 5
MAX_REBUTTALS_PER_TURN = 4
MAX_REBUTTALS_PER_SIDE = 8
MAX_POINT_CHARS = 240
MAX_STANCE_CHARS = 240
#: Turns of earlier transcript shown to a speaker (keeps the context small for local models).
TRANSCRIPT_WINDOW = 4
#: Findings per sub-agent included in the facts.
FINDINGS_PER_AGENT = 2


class DebateFailure(RuntimeError):
    """A debate turn could not be produced (LLM failed after retries)."""


class DebateTurnOutput(BaseModel):
    """LLM answer for one debate turn (Bull or Bear)."""

    model_config = ConfigDict(extra="ignore")

    stance: str = Field(description="One sentence: your position on taking this trade")
    arguments: list[str] = Field(
        description=f"Your {MAX_ARGUMENTS} strongest arguments at most, strongest first, each quoting the facts")
    rebuttals: list[str] = Field(
        default_factory=list,
        description=f"Up to {MAX_REBUTTALS_PER_TURN} direct answers to the opponent's latest points "
                    "(empty list in the opening statement)")
    conviction: int = Field(ge=0, le=100, description="0-100: how strong your side's case is after this turn")

    @field_validator("stance", mode="before")
    @classmethod
    def _v_stance(cls, v: Any) -> Any:
        return truncate_text(v, MAX_STANCE_CHARS) if isinstance(v, str) else v

    @field_validator("arguments", mode="before")
    @classmethod
    def _v_args(cls, v: Any) -> Any:
        out = truncate_list(v, MAX_ARGUMENTS, MAX_POINT_CHARS)
        if isinstance(out, list) and not out:
            raise ValueError("arguments must contain at least one argument")
        return out

    @field_validator("rebuttals", mode="before")
    @classmethod
    def _v_rebuttals(cls, v: Any) -> Any:
        if v is None:
            return []
        return truncate_list(v, MAX_REBUTTALS_PER_TURN, MAX_POINT_CHARS)

    @field_validator("conviction", mode="before")
    @classmethod
    def _v_conv(cls, v: Any) -> Any:
        return clamp_int(v, 0, 100)


class _FixedClock:
    """Clock frozen at the analysis instant (JSONL lands in the analysed day's folder)."""

    def __init__(self, at: "datetime") -> None:
        self._at = to_ist(at)

    def now(self) -> "datetime":
        return self._at


# --------------------------------------------------------------------------- deterministic facts


def setup_costs(setup: StrategySignal, settings: "Settings") -> dict[str, Any]:
    """Round-trip charges for ONE share vs the setup's reward (section 5 charges check)."""
    product = product_for(setup.mode)
    risk = abs(setup.entry_price - setup.stop_loss)
    reward = abs(setup.target_1 - setup.entry_price)
    charges = estimate_round_trip(product, setup.direction, 1, setup.entry_price, setup.target_1, settings.charges)
    net = reward - charges
    return {
        "product": product.value,
        "qty_basis": 1,
        "round_trip_charges_per_share": r2(charges, 4),
        "charges_pct_of_entry_value": r2(charges / setup.entry_price * 100, 3) if setup.entry_price > 0 else None,
        "risk_per_share": r2(risk, 4),
        "reward_per_share_at_target_1": r2(reward, 4),
        "charges_pct_of_reward": r2(charges / reward * 100, 1) if reward > 0 else None,
        "net_reward_per_share_after_charges": r2(net, 4),
        "net_rr_after_charges": r2(net / risk, 2) if risk > 0 else None,
        "charges_exceed_reward": bool(net <= 0),
        "note": "charges for a 1-share trade; flat brokerage makes small trades relatively expensive",
    }


def overnight_risk(ctx: StockContext, setup: StrategySignal | None, reports: Sequence[AgentReport],
                   vote: VoteSummary | None, settings: "Settings") -> dict[str, Any]:
    """Gap / event / holding-period facts for BTST, SWING and POSITIONAL (empty for INTRADAY)."""
    mode = ctx.mode
    if not is_overnight(mode):
        return {"holds_overnight": False, "square_off": settings.risk.intraday_squareoff,
                "no_new_entries_after": settings.risk.intraday_no_entry_after,
                "time_stop_minutes": settings.mode(mode).time_stop_minutes}
    cfg = settings.mode(mode)
    gap_pct = float(settings.risk.gap_buffer_pct)
    out: dict[str, Any] = {
        "holds_overnight": True,
        "holding_days": {"min": cfg.min_holding_days, "max": cfg.max_holding_days},
        "gap_buffer_pct": gap_pct,
        "no_overnight_shorts": True,
    }
    if setup is not None:
        sl_risk = abs(setup.entry_price - setup.stop_loss)
        gap_risk = setup.entry_price * gap_pct / 100.0
        out.update({
            "sl_risk_per_share": r2(sl_risk, 4),
            "gap_risk_per_share": r2(gap_risk, 4),
            "effective_risk_per_share": r2(max(sl_risk, gap_risk), 4),
            "gap_risk_exceeds_sl_risk": bool(gap_risk > sl_risk),
            "gap_risk_multiple_of_sl_risk": r2(gap_risk / sl_risk, 2) if sl_risk > 0 else None,
        })
    ev = next((r for r in reports if r.agent == "event_risk" and r.status != AgentStatus.SKIPPED), None)
    facts = code_facts(ev) if ev is not None else {}
    out["blocking_events"] = list(facts.get("blocking_events") or (vote.blocking_events if vote else []))
    out["soft_event_risks"] = list(facts.get("soft_risks") or [])
    out["events_in_context"] = [
        {"date": e.event_date.isoformat(), "type": e.event_type.value, "title": e.title[:100],
         "stock_specific": e.symbol is not None}
        for e in sorted(ctx.events, key=lambda e: e.event_date)[:8]
    ]
    out["avoid_events_in_window"] = settings.risk.avoid_events_in_window
    return out


def agent_digest(reports: Sequence[AgentReport]) -> list[dict[str, Any]]:
    """Condensed sub-agent reports (status, vote, top findings)."""
    out: list[dict[str, Any]] = []
    for r in reports:
        if r.status == AgentStatus.SKIPPED:
            continue
        item: dict[str, Any] = {"agent": r.agent, "status": r.status.value, "signal": r.effective_signal.value,
                                "confidence": r.confidence if r.status == AgentStatus.OK else 0,
                                "findings": [truncate_text(f, 160) for f in r.key_findings[:FINDINGS_PER_AGENT]]}
        if r.tradeable is not None:
            item["tradeable"] = r.tradeable
        out.append(item)
    return out


def vote_digest(vote: VoteSummary | None) -> dict[str, Any] | None:
    """The weighted vote in prompt form."""
    if vote is None:
        return None
    return {"score": vote.score, "signal": vote.signal.value, "bullish": vote.bullish, "bearish": vote.bearish,
            "neutral": vote.neutral, "failed": vote.failed, "tradeable": vote.tradeable,
            "tradeable_reasons": vote.tradeable_reasons[:4], "event_risk": vote.event_risk,
            "blocking_events": vote.blocking_events[:4], "fundamental_score": vote.fundamental_score}


def setup_digest(s: StrategySignal) -> dict[str, Any]:
    """A strategy signal in prompt form (R:R pre-computed)."""
    return {"strategy": s.strategy, "mode": s.mode.value, "direction": s.direction.value,
            "entry_price": r2(s.entry_price), "entry_type": s.entry_type.value, "stop_loss": r2(s.stop_loss),
            "target_1": r2(s.target_1), "target_2": r2(s.target_2), "rr_to_target_1": r2(s.rr),
            "strength": s.strength, "regime_fit": s.regime_fit, "notes": [n[:120] for n in s.notes[:3]]}


def market_digest(ctx: StockContext) -> dict[str, Any] | None:
    m = ctx.market
    if m is None:
        return None
    return {"nifty_trend": m.nifty_trend.value, "nifty_change_pct": r2(m.nifty_change_pct),
            "india_vix": r2(m.india_vix), "vix_elevated": m.vix_elevated, "market_regime": m.regime.value,
            "stock_sector": ctx.sector,
            "sector_strength": r2(m.sector_strength.get(ctx.sector)) if ctx.sector else None}


def debate_facts(ctx: StockContext, reports: Sequence[AgentReport], vote: VoteSummary | None,
                 settings: "Settings") -> dict[str, Any]:
    """Every deterministic fact both debaters see (JSON-serialisable)."""
    best = strongest_usable(ctx)
    setup = best[1] if best else None
    cfg = settings.mode(ctx.mode)
    conf = ctx.confluence
    facts: dict[str, Any] = {
        "symbol": ctx.symbol,
        "mode": ctx.mode.value,
        "ltp": r2(current_price(ctx)),
        "regime": ctx.regime.value,
        "setup": setup_digest(setup) if setup else None,
        "other_setups": [setup_digest(s) for s in ctx.strategy_signals if s is not setup][:3],
        "confluence": ({"score": r2(conf.score, 1), "grade": conf.grade.value, "min_grade": cfg.min_grade.value,
                        "min_confluence": cfg.min_confluence,
                        "top_components": dict(sorted(conf.components.items(), key=lambda kv: -kv[1])[:6])}
                       if conf else None),
        "mode_rules": {"min_rr": cfg.min_rr, "confidence_threshold": cfg.confidence_threshold},
        "weighted_vote": vote_digest(vote),
        "agents": agent_digest(reports),
        "market": market_digest(ctx),
        "costs": setup_costs(setup, settings) if setup else None,
        "overnight_and_event_risk": overnight_risk(ctx, setup, reports, vote, settings),
    }
    return to_jsonable(facts)


# --------------------------------------------------------------------------- deterministic debate


def _bull_conviction(ctx: StockContext, setup: StrategySignal, vote: VoteSummary | None) -> int:
    """Average of the confluence score and the direction-aligned vote mapped to 0-100."""
    c = ctx.confluence.score if ctx.confluence else 50.0
    sign = 1.0 if setup.direction == Direction.LONG else -1.0
    aligned = sign * (vote.score if vote else 0.0)
    return int(max(0, min(100, round(0.5 * c + 0.5 * (aligned + 100.0) / 2.0))))


def deterministic_debate(ctx: StockContext, reports: Sequence[AgentReport], vote: VoteSummary | None,
                         settings: "Settings", facts: dict[str, Any] | None = None) -> DebateResult:
    """Rules-backend debate: convictions from vote / confluence / risk facts, openings only."""
    t0 = time.perf_counter()
    best = strongest_usable(ctx)
    if best is None:
        return DebateResult(status=AgentStatus.SKIPPED, error="no strategy setup: nothing to debate", rounds=0)
    setup = best[1]
    facts = facts if facts is not None else debate_facts(ctx, reports, vote, settings)
    want = DIRECTION_SIGNAL[setup.direction]
    against = Signal.BEARISH if want == Signal.BULLISH else Signal.BULLISH
    weight = settings.mode(ctx.mode).agent_weights

    def _ranked(signal: Signal) -> list[AgentReport]:
        rs = [r for r in reports if r.status == AgentStatus.OK and r.signal == signal]
        return sorted(rs, key=lambda r: -(r.confidence * weight.get(r.agent, 1.0)))

    bull_args = [
        f"{setup.strategy} {setup.direction.value}: entry {setup.entry_price:.2f}, stop {setup.stop_loss:.2f}, "
        f"target {setup.target_1:.2f} ({setup.rr:.2f}R)",
    ]
    if ctx.confluence is not None:
        bull_args.append(f"confluence {ctx.confluence.score:.1f} (grade {ctx.confluence.grade.value})")
    if vote is not None:
        bull_args.append(f"weighted vote {vote.score:+.1f}: {vote.bullish} bullish / {vote.bearish} bearish agents")
    bull_args += [f"{r.agent} ({r.confidence}): {r.key_findings[0]}" for r in _ranked(want)[:3] if r.key_findings]

    bear_conv = 100 - _bull_conviction(ctx, setup, vote)
    bear_args: list[str] = []
    for r in _ranked(against)[:3]:
        bear_args.append(f"{r.agent} ({r.confidence}) disagrees: {r.key_findings[0] if r.key_findings else r.signal.value}")
        if r.confidence >= 60:
            bear_conv += 5
    costs = facts.get("costs") or {}
    pct = costs.get("charges_pct_of_reward")
    if pct is not None:
        bear_args.append(f"round-trip charges {costs['round_trip_charges_per_share']} per share = {pct}% of the "
                         f"reward at target 1")
        bear_conv += 20 if pct >= 50 else 10 if pct >= 25 else 0
    risk = facts.get("overnight_and_event_risk") or {}
    if risk.get("holds_overnight"):
        if risk.get("gap_risk_exceeds_sl_risk"):
            bear_args.append(f"a {risk['gap_buffer_pct']}% overnight gap ({risk['gap_risk_per_share']} per share) "
                             f"exceeds the stop risk ({risk['sl_risk_per_share']} per share)")
            bear_conv += 10
        if risk.get("blocking_events"):
            bear_args.append("blocking event in the holding window: " + "; ".join(risk["blocking_events"][:2]))
            bear_conv += 25
        elif risk.get("soft_event_risks"):
            bear_args.append("event risk in the holding window: " + "; ".join(risk["soft_event_risks"][:2]))
            bear_conv += 5
    if vote is not None and vote.failed:
        bear_args.append(f"{vote.failed} agent(s) FAILED: {', '.join(vote.failed_agents[:4])}")
        bear_conv += min(15, 3 * vote.failed)
    if ctx.market is not None and ctx.market.vix_elevated:
        bear_args.append("India VIX is elevated (risk-off)")
        bear_conv += 5
    if not bear_args:
        bear_args.append(f"no major red flags; the main risk is the stop at {setup.stop_loss:.2f} "
                         f"({abs(setup.entry_price - setup.stop_loss):.2f} per share)")
    bull_conv = _bull_conviction(ctx, setup, vote)
    bear_conv = int(max(0, min(100, bear_conv)))
    bull = DebateSide(stance=f"FOR: take the {setup.direction.value} {setup.strategy} setup",
                      top_arguments=truncate_list(bull_args, MAX_ARGUMENTS, MAX_POINT_CHARS), conviction=bull_conv)
    bear = DebateSide(stance="AGAINST: costs, risks and dissenting evidence outweigh the edge"
                      if bear_conv > bull_conv else "AGAINST: risks noted but not decisive",
                      top_arguments=truncate_list(bear_args, MAX_ARGUMENTS, MAX_POINT_CHARS), conviction=bear_conv)
    transcript = [_entry(0, BULL, OPENING, bull.stance, bull.top_arguments, [], bull.conviction, "rules", None),
                  _entry(0, BEAR, OPENING, bear.stance, bear.top_arguments, [], bear.conviction, "rules", None)]
    return DebateResult(bull=bull, bear=bear, rounds=0, transcript=transcript, status=AgentStatus.OK,
                        latency_ms=int((time.perf_counter() - t0) * 1000))


# --------------------------------------------------------------------------- LLM debate


def _content(stance: str, arguments: Sequence[str], rebuttals: Sequence[str], conviction: int) -> str:
    lines = [f"STANCE: {stance}", "ARGUMENTS:"] + [f"- {a}" for a in arguments]
    if rebuttals:
        lines += ["REBUTTALS:"] + [f"- {b}" for b in rebuttals]
    lines.append(f"CONVICTION: {conviction}")
    return "\n".join(lines)


def _entry(rnd: int, speaker: str, phase: str, stance: str, arguments: Sequence[str], rebuttals: Sequence[str],
           conviction: int, model: str | None, prompt_version: int | None, **kw: Any) -> dict[str, Any]:
    """One transcript entry ``{round, speaker, phase, content, ...}``."""
    return {"round": rnd, "speaker": speaker, "phase": phase,
            "content": _content(stance, arguments, rebuttals, conviction), "stance": stance,
            "arguments": list(arguments), "rebuttals": list(rebuttals), "conviction": conviction,
            "model": model, "prompt_version": prompt_version, **kw}


def _turn_json(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "null"
    keep = {k: entry[k] for k in ("round", "speaker", "stance", "arguments", "rebuttals", "conviction")}
    return json.dumps(keep, ensure_ascii=False)


class _Debate:
    """One LLM debate (state holder so a failure can keep the partial transcript)."""

    def __init__(self, ctx: StockContext, settings: "Settings", llm: "LLMClient", prompts: "PromptStore",
                 facts: dict[str, Any]) -> None:
        self.ctx = ctx
        self.settings = settings
        self.llm = llm
        self.prompts = prompts
        self.facts_json = json.dumps(facts, ensure_ascii=False, indent=1)
        self.transcript: list[dict[str, Any]] = []
        self.rounds_done = 0
        self.skipped = False  # backend answered "skipped" (rules client)

    def turn(self, speaker: str, phase: str, rnd: int, opponent_last: dict[str, Any] | None) -> dict[str, Any] | None:
        """Run one turn; returns its transcript entry (None when the backend skipped)."""
        s = self.settings
        model = s.models.model_for(speaker)
        tpl = self.prompts.get(speaker, s.agents.prompt_versions.get(speaker))
        history = [{k: e[k] for k in ("round", "speaker", "stance", "arguments", "rebuttals", "conviction")}
                   for e in self.transcript[-TRANSCRIPT_WINDOW:]]
        variables = {
            "agent": speaker, "symbol": self.ctx.symbol, "mode": self.ctx.mode.value,
            "as_of": self.ctx.as_of.isoformat(), "mode_rules": mode_rules_text(self.ctx.mode, s),
            "output_schema": output_schema_summary(DebateTurnOutput), "facts_json": self.facts_json,
            "phase": phase, "round": rnd, "total_rounds": s.agents.debate_rounds,
            "opponent": BEAR if speaker == BULL else BULL, "opponent_last_json": _turn_json(opponent_last),
            "transcript_json": json.dumps(history, ensure_ascii=False),
        }
        system, user = self.prompts.render_prompt(tpl, **variables)
        res = self.llm.generate(speaker, model, system, user, DebateTurnOutput,
                                temperature=s.models.temperature, think=s.models.reasoning_think)
        if res.skipped:
            self.skipped = True
            return None
        if not res.ok or res.parsed is None:
            raise DebateFailure(f"{speaker} {phase} (round {rnd}): {res.error or 'no parsed output'}")
        out: DebateTurnOutput = res.parsed
        rebuttals = list(out.rebuttals) if phase == REBUTTAL else []
        entry = _entry(rnd, speaker, phase, out.stance, out.arguments, rebuttals, out.conviction,
                       res.model or model, tpl.version, latency_ms=res.latency_ms, attempts=res.attempts,
                       tokens_in=res.tokens_in, tokens_out=res.tokens_out)
        self.transcript.append(entry)
        return entry

    def run(self) -> DebateResult | None:
        """Openings + rebuttal rounds. None when the backend skipped (use the rules debate)."""
        bull = self.turn(BULL, OPENING, 0, None)
        if bull is None:
            return None
        bear = self.turn(BEAR, OPENING, 0, None)
        if bear is None:
            return None
        for rnd in range(1, self.settings.agents.debate_rounds + 1):
            bull = self.turn(BULL, REBUTTAL, rnd, bear)
            if bull is None:
                return None
            bear = self.turn(BEAR, REBUTTAL, rnd, bull)
            if bear is None:
                return None
            self.rounds_done = rnd
        return DebateResult(bull=self.side(BULL), bear=self.side(BEAR), rounds=self.rounds_done,
                            transcript=self.transcript, status=AgentStatus.OK)

    def side(self, speaker: str) -> DebateSide:
        """Final position of ``speaker``: last stance / arguments / conviction, all rebuttals."""
        turns = [e for e in self.transcript if e["speaker"] == speaker]
        last = turns[-1]
        rebuttals: list[str] = []
        for e in turns:
            for b in e["rebuttals"]:
                if b not in rebuttals:
                    rebuttals.append(b)
        return DebateSide(stance=last["stance"], top_arguments=list(last["arguments"] or turns[0]["arguments"]),
                          rebuttals=rebuttals[:MAX_REBUTTALS_PER_SIDE], conviction=int(last["conviction"]))


# --------------------------------------------------------------------------- entry point


def debate_skip_reason(ctx: StockContext, reports: Sequence[AgentReport], vote: VoteSummary | None,
                       settings: "Settings") -> str | None:
    """Why the debate is not run (None = run it)."""
    if not settings.agents.debate_enabled:
        return "debate disabled in settings"
    disabled = [a for a in (BULL, BEAR) if not settings.agents.enabled.get(a, True)]
    if disabled:
        return f"debate agent(s) disabled: {', '.join(disabled)}"
    gate = trade_gate(ctx, reports, vote, settings)
    if gate is not None:
        return f"nothing to debate: {gate}"
    return None


def _log(ctx: StockContext, result: DebateResult) -> None:
    log_event("debates", {
        "symbol": ctx.symbol, "mode": ctx.mode.value, "run_id": ctx.run_id, "batch_id": ctx.batch_id,
        "status": result.status.value, "rounds": result.rounds, "error": result.error,
        "bull_conviction": result.bull.conviction if result.bull else None,
        "bear_conviction": result.bear.conviction if result.bear else None,
        "latency_ms": result.latency_ms, "transcript": result.transcript,
    }, _FixedClock(ctx.as_of))


def run_debate(
    ctx: StockContext,
    reports: Sequence[AgentReport],
    vote: VoteSummary | None,
    settings: "Settings",
    llm: "LLMClient",
    prompts: "PromptStore | None",
) -> DebateResult:
    """Bull vs Bear debate for ``ctx`` (see module docstring). NEVER raises."""
    t0 = time.perf_counter()

    def _done(result: DebateResult) -> DebateResult:
        result = result.model_copy(update={"latency_ms": int((time.perf_counter() - t0) * 1000)})
        try:
            _log(ctx, result)
        except Exception as exc:  # noqa: BLE001 - logging must never break the pipeline
            logger.error(f"Could not log debate for {ctx.symbol}: {exc}")
        return result

    state: _Debate | None = None
    try:
        why = debate_skip_reason(ctx, reports, vote, settings)
        if why is not None:
            logger.debug(f"Debate skipped for {ctx.symbol}/{ctx.mode.value}: {why}")
            return _done(DebateResult(status=AgentStatus.SKIPPED, error=why, rounds=0))
        facts = debate_facts(ctx, reports, vote, settings)
        if getattr(llm, "backend", None) == "rules":
            return _done(deterministic_debate(ctx, reports, vote, settings, facts))
        if prompts is None:
            raise DebateFailure("no PromptStore available for the LLM backend")
        state = _Debate(ctx, settings, llm, prompts, facts)
        result = state.run()
        if result is None:  # backend skipped (rules-like client): deterministic debate
            return _done(deterministic_debate(ctx, reports, vote, settings, facts))
        logger.info(f"Debate {ctx.symbol}/{ctx.mode.value}: bull {result.bull.conviction if result.bull else '-'} "
                    f"vs bear {result.bear.conviction if result.bear else '-'} after {result.rounds} round(s)")
        return _done(result)
    except Exception as exc:  # noqa: BLE001 - the debate must never crash the stock's analysis
        if isinstance(exc, DebateFailure):
            logger.warning(f"Debate FAILED for {ctx.symbol}/{ctx.mode.value}: {exc}")
        else:
            logger.exception(f"Debate FAILED for {ctx.symbol}/{ctx.mode.value}: {exc}")
        transcript = state.transcript if state is not None else []
        rounds = state.rounds_done if state is not None else 0
        return _done(DebateResult(bull=None, bear=None, rounds=rounds, transcript=transcript,
                                  status=AgentStatus.FAILED, error=truncate_text(f"{type(exc).__name__}: {exc}", 1000)))
