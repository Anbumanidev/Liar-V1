"""run_sub_agents (parallel, isolation, JSONL latency log), weighted_vote math, persistence."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.aggregate import VoteSummary, weighted_vote
from agents.base import SubAgent
from agents.llm import FakeLLMClient, LLMClient, RulesClient
from agents.persist import link_reports_to_decision, save_agent_reports, save_debate
from agents.prompt_store import PromptStore
from agents.runner import run_sub_agents
from config.schema import AGENT_NAMES, Settings
from core.enums import AgentStatus, Mode, Signal
from core.logging import read_events
from core.types import AgentReport, DebateResult, DebateSide, StockContext
from storage.models import AgentReportRow, DebateRow
from tests.agents.agents_kit import NOW, good_answers, intraday_ctx, report


class _Probe(SubAgent):
    """Unregistered test agent: waits on a barrier (proves parallelism) or raises out of run()."""

    def __init__(self, name: str, barrier: threading.Barrier | None = None, explode: bool = False) -> None:
        self._name = name
        self.barrier = barrier
        self.explode = explode

    @property  # type: ignore[override]
    def name(self) -> str:  # noqa: D401 - instance-specific name
        return self._name

    def build_payload(self, ctx: StockContext, settings: Settings | None = None) -> dict[str, Any]:
        return {}

    def deterministic_report(self, ctx: StockContext, settings: Settings | None = None) -> AgentReport:
        return self.make_report(ctx, signal=Signal.BULLISH, confidence=50)

    def run(self, ctx: StockContext, llm: LLMClient, settings: Settings, prompts: PromptStore | None) -> AgentReport:
        if self.explode:
            raise RuntimeError(f"{self._name} exploded outside its own guard")
        if self.barrier is not None:
            self.barrier.wait(timeout=5)  # BrokenBarrierError (-> FAILED) unless all run concurrently
        return self.deterministic_report(ctx, settings).model_copy(update={"latency_ms": 7})


# --------------------------------------------------------------------------- runner


def test_run_sub_agents_all_agents_in_registry_order_and_logged(settings: Settings, prompts: PromptStore) -> None:
    ctx = intraday_ctx()
    reps = run_sub_agents(ctx, settings, FakeLLMClient(good_answers()), prompts)
    assert [r.agent for r in reps] == list(AGENT_NAMES)
    by = {r.agent: r for r in reps}
    assert by["event_risk"].status == AgentStatus.SKIPPED
    assert all(r.status == AgentStatus.OK for n, r in by.items() if n != "event_risk")
    events = read_events(NOW.date(), "agents")
    per_agent = [e for e in events if e["agent"] != "_summary"]
    assert {e["agent"] for e in per_agent} == set(AGENT_NAMES)
    assert all("latency_ms" in e and e["symbol"] == "RELIANCE" and e["run_id"] == 7 for e in per_agent)
    summary = next(e for e in events if e["agent"] == "_summary")
    assert summary["OK"] == 9 and summary["SKIPPED"] == 1 and summary["wall_ms"] >= 0


def test_runner_runs_agents_in_parallel(settings: Settings) -> None:
    barrier = threading.Barrier(4)
    probes = [_Probe(f"p{i}", barrier) for i in range(4)]
    reps = run_sub_agents(intraday_ctx(), settings, RulesClient(), None, max_workers=4, agents=probes)
    assert [r.status for r in reps] == [AgentStatus.OK] * 4


def test_one_agent_raising_does_not_crash_the_others(settings: Settings) -> None:
    probes = [_Probe("good1"), _Probe("bad", explode=True), _Probe("good2")]
    reps = run_sub_agents(intraday_ctx(), settings, RulesClient(), None, max_workers=2, agents=probes)
    assert [r.agent for r in reps] == ["good1", "bad", "good2"]
    assert [r.status for r in reps] == [AgentStatus.OK, AgentStatus.FAILED, AgentStatus.OK]
    assert "exploded" in (reps[1].error or "") and reps[1].signal == Signal.NEUTRAL


def test_runner_warns_when_slow(settings: Settings, log_messages: list[str]) -> None:
    settings.models.latency_warn_seconds = 1e-9
    run_sub_agents(intraday_ctx(), settings, RulesClient(), None, agents=[_Probe("p")])
    assert any("took" in m and "slowest" in m for m in log_messages)


# --------------------------------------------------------------------------- weighted vote


def test_weighted_vote_math_failed_and_skipped_are_neutral() -> None:
    reps = [
        report("technical", Signal.BULLISH, 80),
        report("price_action", Signal.BEARISH, 40),
        report("fundamental", Signal.BULLISH, 90, status=AgentStatus.FAILED),
        report("event_risk", Signal.BEARISH, 90, status=AgentStatus.SKIPPED),
        report("news_sentiment", Signal.NEUTRAL, 50),
        report("stop_loss", Signal.BULLISH, 100),
    ]
    weights = {"technical": 1.5, "price_action": 1.0, "fundamental": 0.5, "event_risk": 1.0,
               "news_sentiment": 0.6, "stop_loss": 0.0}
    v = weighted_vote(reps, weights)
    # (1.5*0.8 - 1.0*0.4) / (1.5 + 1.0 + 0.5 + 0.6): FAILED dilutes, SKIPPED and weight 0 do not count
    assert v.score == pytest.approx(round(100 * 0.8 / 3.6, 2))
    assert (v.bullish, v.bearish, v.neutral, v.failed, v.skipped) == (2, 1, 3, 1, 1)
    assert v.total_weight == pytest.approx(3.6) and v.failed_agents == ["fundamental"]
    assert v.contributions == {"technical": 1.2, "price_action": -0.4, "fundamental": 0.0, "news_sentiment": 0.0}
    assert v.tradeable is True and v.signal == Signal.BULLISH


def test_weighted_vote_default_weight_extremes_and_signal_threshold() -> None:
    assert weighted_vote([]).score == 0.0
    assert weighted_vote([report("technical", Signal.BEARISH, 100)]).score == -100.0
    v = weighted_vote([report("technical", Signal.BULLISH, 9)])  # default weight 1.0
    assert v.score == 9.0 and v.signal == Signal.NEUTRAL
    assert VoteSummary(score=10).signal == Signal.BULLISH and VoteSummary(score=-10).signal == Signal.BEARISH
    assert weighted_vote([report("x", Signal.BULLISH, 50, status=AgentStatus.SKIPPED)]).score == 0.0


def test_tradeable_false_and_event_facts_propagate_even_when_failed() -> None:
    fundamental_failed = report("fundamental", status=AgentStatus.FAILED, tradeable=False,
                                extra={"fallback": {"blocking_flags": ["stock is in the GSM list"],
                                                    "fundamental_score": 31.0}})
    event_failed = report("event_risk", status=AgentStatus.FAILED,
                          extra={"fallback": {"event_risk": True, "blocking_events": ["E1: RESULTS 2026-11-27"]}})
    v = weighted_vote([fundamental_failed, event_failed, report("technical", Signal.BULLISH, 90)])
    assert v.tradeable is False and v.tradeable_reasons == ["stock is in the GSM list"]
    assert v.fundamental_score == 31.0
    assert v.event_risk is True and v.blocking_events == ["E1: RESULTS 2026-11-27"]
    ok = weighted_vote([report("fundamental", Signal.BEARISH, 60, tradeable=False,
                               extra={"blocking_flags": ["results announced today"]})])
    assert ok.tradeable is False and ok.tradeable_reasons == ["results announced today"]
    skipped_event = report("event_risk", status=AgentStatus.SKIPPED, extra={"event_risk": True})
    assert weighted_vote([skipped_event]).event_risk is False


def test_vote_over_real_reports_respects_mode_weights(settings: Settings, prompts: PromptStore) -> None:
    reps = run_sub_agents(intraday_ctx(), settings, RulesClient(), prompts)
    v = weighted_vote(reps, settings.mode(Mode.INTRADAY).agent_weights)
    assert "event_risk" not in v.contributions and v.skipped == 1
    assert v.total_weight == pytest.approx(sum(w for a, w in settings.mode(Mode.INTRADAY).agent_weights.items()
                                               if a != "event_risk"))
    assert v.signal == Signal.BULLISH and v.tradeable


# --------------------------------------------------------------------------- persistence


def test_save_agent_reports_roundtrip(session: Session, settings: Settings, prompts: PromptStore) -> None:
    reps = run_sub_agents(intraday_ctx(), settings, FakeLLMClient(good_answers()), prompts)
    rows = save_agent_reports(session, reps, run_id=7, batch_id=1)
    assert len(rows) == len(reps) and all(r.id for r in rows)
    stored = session.scalars(select(AgentReportRow).order_by(AgentReportRow.id)).all()
    assert [r.agent for r in stored] == list(AGENT_NAMES)
    tech = next(r for r in stored if r.agent == "technical")
    assert (tech.signal, tech.status, tech.mode, tech.prompt_version) == ("BULLISH", "OK", "INTRADAY", 1)
    assert tech.model == settings.models.model_for("technical") and tech.created_at == NOW
    assert AgentReport.model_validate(tech.report_json).agent == "technical"
    assert tech.decision_id is None
    link_reports_to_decision(session, rows, 42)
    assert {r.decision_id for r in session.scalars(select(AgentReportRow))} == {42}


def test_save_debate_roundtrip(session: Session) -> None:
    result = DebateResult(
        bull=DebateSide(stance="for", top_arguments=["a"], conviction=70),
        bear=DebateSide(stance="against", top_arguments=["b"], rebuttals=["r"], conviction=40),
        rounds=1, transcript=[{"round": 0, "speaker": "bull", "content": "x", "model": "qwen3:8b"},
                              {"round": 0, "speaker": "bear", "content": "y", "model": "qwen3:8b"}],
        latency_ms=12,
    )
    row = save_debate(session, result, symbol="RELIANCE", mode=Mode.INTRADAY, created_at=NOW, run_id=7, batch_id=1)
    got = session.scalars(select(DebateRow)).one()
    assert got.id == row.id and got.rounds == 1 and got.status == "OK" and got.model == "qwen3:8b"
    assert got.bull_json["conviction"] == 70 and got.bear_json["rebuttals"] == ["r"]
    assert len(got.transcript_json) == 2 and got.latency_ms == 12
    failed = save_debate(session, DebateResult(status=AgentStatus.FAILED, error="x"), symbol="RELIANCE",
                         mode=Mode.SWING, created_at=NOW)
    assert failed.bull_json is None and failed.status == "FAILED" and failed.model is None
