"""Layer-2 debate: rounds + transcript, Bear cost/overnight facts, SKIPPED / FAILED paths, rules backend."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

from agents.aggregate import VoteSummary, weighted_vote
from agents.debate.debate import DebateTurnOutput, debate_facts, deterministic_debate, run_debate
from agents.llm import FakeLLMClient, LLMResult, RulesClient
from agents.prompt_store import PromptStore
from agents.runner import run_sub_agents
from config.schema import Settings
from core.enums import AgentStatus, Direction, EventType, Mode, Product
from core.logging import read_events
from core.types import AgentReport, MarketEvent, StockContext
from paper_engine.charges import estimate_round_trip
from tests.agents.agents_kit import (
    NOW,
    SWING_NOW,
    clean_fundamentals,
    intraday_ctx,
    market_ctx,
    swing_ctx,
    swing_signal,
)


def analyse(ctx: StockContext, settings: Settings, prompts: PromptStore) -> tuple[list[AgentReport], VoteSummary]:
    reps = run_sub_agents(ctx, settings, RulesClient(), prompts)
    return reps, weighted_vote(reps, settings.mode(ctx.mode).agent_weights)


def debater(side: str, conviction: dict[int, int] | None = None, fail_round: int | None = None
            ) -> Callable[[str, str], Any]:
    """Scripted debater: answers depend on the round found in the prompt."""
    def _answer(system: str, user: str) -> Any:
        m = re.search(r"\(round (\d+) of", user)
        rnd = int(m.group(1)) if m else 0
        if fail_round is not None and rnd == fail_round:
            return "I refuse to answer in JSON"
        return {"stance": f"{side} stance r{rnd}", "arguments": [f"{side} arg r{rnd}", f"{side} arg2 r{rnd}"],
                "rebuttals": [f"{side} rebuttal r{rnd}"], "conviction": (conviction or {}).get(rnd, 50)}
    return _answer


@pytest.fixture
def intraday(settings: Settings, prompts: PromptStore) -> tuple[StockContext, list[AgentReport], VoteSummary]:
    ctx = intraday_ctx()
    reps, vote = analyse(ctx, settings, prompts)
    return ctx, reps, vote


def test_debate_rounds_transcript_and_sides(settings: Settings, prompts: PromptStore, intraday: Any) -> None:
    ctx, reps, vote = intraday
    settings.models.reasoning_think = True
    fake = FakeLLMClient({"bull": debater("bull", {0: 70, 1: 66, 2: 61}), "bear": debater("bear", {0: 40, 2: 55})})
    res = run_debate(ctx, reps, vote, settings, fake, prompts)

    assert res.status == AgentStatus.OK and res.error is None and res.rounds == 2 == settings.agents.debate_rounds
    assert [(t["round"], t["speaker"], t["phase"]) for t in res.transcript] == [
        (0, "bull", "opening"), (0, "bear", "opening"), (1, "bull", "rebuttal"), (1, "bear", "rebuttal"),
        (2, "bull", "rebuttal"), (2, "bear", "rebuttal")]
    assert all({"round", "speaker", "content"} <= set(t) for t in res.transcript)
    assert "STANCE: bull stance r0" in res.transcript[0]["content"]
    assert res.transcript[0]["rebuttals"] == []  # openings never carry rebuttals
    assert res.bull is not None and res.bear is not None
    assert res.bull.stance == "bull stance r2" and res.bull.conviction == 61
    assert res.bull.top_arguments == ["bull arg r2", "bull arg2 r2"]
    assert res.bull.rebuttals == ["bull rebuttal r1", "bull rebuttal r2"]
    assert res.bear.conviction == 55 and res.latency_ms is not None
    calls = fake.calls
    assert len(calls) == 6
    assert all(c["model"] == settings.models.model_for(c["agent"]) == settings.models.profiles["low"].reasoning
               for c in calls)
    assert all(c["think"] is True and c["schema"] == "DebateTurnOutput" for c in calls)
    assert all(t["prompt_version"] == 1 and t["model"] == settings.models.profiles["low"].reasoning for t in res.transcript)


def test_rebuttals_see_the_opponents_latest_turn(settings: Settings, prompts: PromptStore, intraday: Any) -> None:
    ctx, reps, vote = intraday
    fake = FakeLLMClient({"bull": debater("bull"), "bear": debater("bear")})
    run_debate(ctx, reps, vote, settings, fake, prompts)
    bull_calls, bear_calls = fake.calls_for("bull"), fake.calls_for("bear")
    assert "OPENING statement" in bull_calls[0]["user"] and "latest turn" not in bull_calls[0]["user"]
    assert "bear stance" not in bear_calls[0]["user"]  # independent openings
    assert "bear stance r0" in bull_calls[1]["user"]
    assert "bull stance r1" in bear_calls[1]["user"]  # bear answers the bull's same-round rebuttal
    assert "bear stance r1" in bull_calls[2]["user"] and "(round 2 of 2)" in bull_calls[2]["user"]
    assert "case FOR" in bull_calls[0]["system"] and "case AGAINST" in bear_calls[0]["system"]


def test_zero_rounds_openings_only(settings: Settings, prompts: PromptStore, intraday: Any) -> None:
    ctx, reps, vote = intraday
    settings.agents.debate_rounds = 0
    res = run_debate(ctx, reps, vote, settings, FakeLLMClient({"*": debater("x")}), prompts)
    assert res.status == AgentStatus.OK and res.rounds == 0 and len(res.transcript) == 2
    assert res.bull is not None and res.bull.rebuttals == [] and res.bear is not None and res.bear.rebuttals == []


def test_bear_gets_round_trip_charges(settings: Settings, prompts: PromptStore, intraday: Any) -> None:
    ctx, reps, vote = intraday
    facts = debate_facts(ctx, reps, vote, settings)
    expected = estimate_round_trip(Product.MIS, Direction.LONG, 1, 100.0, 102.0, settings.charges)
    costs = facts["costs"]
    assert costs["round_trip_charges_per_share"] == round(expected, 4) and costs["qty_basis"] == 1
    assert costs["reward_per_share_at_target_1"] == 2.0 and costs["risk_per_share"] == 1.2
    assert costs["charges_pct_of_reward"] == round(expected / 2.0 * 100, 1)
    assert facts["overnight_and_event_risk"]["holds_overnight"] is False
    fake = FakeLLMClient({"*": debater("x")})
    run_debate(ctx, reps, vote, settings, fake, prompts)
    assert '"round_trip_charges_per_share"' in fake.calls_for("bear")[0]["user"]


def test_bear_gets_overnight_gap_and_event_facts_for_swing(settings: Settings, prompts: PromptStore) -> None:
    macro = MarketEvent(symbol=None, event_type=EventType.MACRO, event_date=date(2026, 11, 26), title="US Fed")
    ctx = swing_ctx(events=[macro], strategy_signals=[swing_signal(stop_loss=490.0)])
    reps, vote = analyse(ctx, settings, prompts)
    risk = debate_facts(ctx, reps, vote, settings)["overnight_and_event_risk"]
    assert risk["holds_overnight"] is True and risk["gap_buffer_pct"] == settings.risk.gap_buffer_pct
    assert risk["gap_risk_per_share"] == 500.0 * settings.risk.gap_buffer_pct / 100
    assert risk["sl_risk_per_share"] == 10.0 and risk["gap_risk_exceeds_sl_risk"] is True
    assert risk["holding_days"] == {"min": 2, "max": 10} and risk["blocking_events"] == []
    assert any("MACRO" in s for s in risk["soft_event_risks"])
    fake = FakeLLMClient({"*": debater("x")})
    res = run_debate(ctx, reps, vote, settings, fake, prompts)
    assert res.status == AgentStatus.OK
    bear_prompt = fake.calls_for("bear")[0]["user"]
    assert '"gap_risk_exceeds_sl_risk": true' in bear_prompt and "US Fed" in bear_prompt


@pytest.mark.parametrize("case", ["disabled", "bear_off", "no_setup", "not_tradeable", "blocking_event"])
def test_debate_skipped(case: str, settings: Settings, prompts: PromptStore) -> None:
    ctx = intraday_ctx()
    if case == "disabled":
        settings.agents.debate_enabled = False
    elif case == "bear_off":
        settings.agents.enabled["bear"] = False
    elif case == "no_setup":
        ctx = intraday_ctx(strategy_signals=[])
    elif case == "not_tradeable":
        ctx = intraday_ctx(fundamentals=clean_fundamentals(in_gsm=True))
    else:
        ctx = swing_ctx()  # RESULTS inside the holding window
    reps, vote = analyse(ctx, settings, prompts)
    fake = FakeLLMClient({"*": debater("x")})
    res = run_debate(ctx, reps, vote, settings, fake, prompts)
    assert res.status == AgentStatus.SKIPPED and res.bull is None and res.bear is None and res.rounds == 0
    assert res.error and fake.calls == []


def test_failure_in_opening_gives_failed_with_empty_sides(settings: Settings, prompts: PromptStore,
                                                          intraday: Any) -> None:
    ctx, reps, vote = intraday
    fake = FakeLLMClient({"bull": debater("bull"), "bear": ["not json"]})
    res = run_debate(ctx, reps, vote, settings, fake, prompts)
    assert res.status == AgentStatus.FAILED and res.bull is None and res.bear is None
    assert "bear opening (round 0)" in (res.error or "") and "LLM failed after 3" in (res.error or "")
    assert [t["speaker"] for t in res.transcript] == ["bull"]  # partial transcript kept for audit
    assert len(fake.calls_for("bear")) == 3  # retried twice


def test_failure_in_last_round_keeps_completed_rounds(settings: Settings, prompts: PromptStore,
                                                      intraday: Any) -> None:
    ctx, reps, vote = intraday
    fake = FakeLLMClient({"bull": debater("bull"), "bear": debater("bear", fail_round=2)})
    res = run_debate(ctx, reps, vote, settings, fake, prompts)
    assert res.status == AgentStatus.FAILED and res.rounds == 1 and len(res.transcript) == 5
    assert res.bull is None and res.bear is None


class _RaisingClient:
    backend = "broken"

    def generate(self, *a: Any, **k: Any) -> LLMResult:
        raise RuntimeError("client blew up")

    def healthcheck(self) -> bool:
        return False

    def list_models(self) -> list[str]:
        return []


def test_debate_never_raises(settings: Settings, prompts: PromptStore, intraday: Any) -> None:
    ctx, reps, vote = intraday
    res = run_debate(ctx, reps, vote, settings, _RaisingClient(), prompts)
    assert res.status == AgentStatus.FAILED and "client blew up" in (res.error or "")
    res = run_debate(ctx, reps, vote, settings, FakeLLMClient({"*": debater("x")}), None)
    assert res.status == AgentStatus.FAILED and "PromptStore" in (res.error or "")
    prompts.save_version("bull", "{{ round + 'x' }}", "{{ facts_json }}", "owner")  # TypeError at render time
    res = run_debate(ctx, reps, vote, settings, FakeLLMClient({"*": debater("x")}), prompts)
    assert res.status == AgentStatus.FAILED and res.bull is None


def test_rules_backend_deterministic_debate(settings: Settings, prompts: PromptStore, intraday: Any) -> None:
    ctx, reps, vote = intraday
    res = run_debate(ctx, reps, vote, settings, RulesClient(), prompts)
    again = run_debate(ctx, reps, vote, settings, RulesClient(), None)
    assert res.status == AgentStatus.OK and res.rounds == 0 and len(res.transcript) == 2
    assert res.model_dump(exclude={"latency_ms"}) == again.model_dump(exclude={"latency_ms"})
    assert all(t["model"] == "rules" for t in res.transcript)
    expected_bull = round(0.5 * ctx.confluence.score + 0.5 * (vote.score + 100) / 2)
    assert res.bull is not None and res.bull.conviction == expected_bull
    assert res.bear is not None and res.bear.conviction < res.bull.conviction
    assert any("charges" in a for a in res.bear.top_arguments)


def test_rules_bear_conviction_rises_with_risk(settings: Settings, prompts: PromptStore, intraday: Any) -> None:
    ctx, reps, vote = intraday
    calm = deterministic_debate(ctx, reps, vote, settings)
    risky_ctx = intraday_ctx(market=market_ctx(vix_elevated=True))
    failed = reps + [AgentReport(agent="news_sentiment", symbol="RELIANCE", mode=Mode.INTRADAY, timestamp=NOW,
                                 status=AgentStatus.FAILED)]
    risky_vote = vote.model_copy(update={"failed": 2, "failed_agents": ["news_sentiment", "technical"]})
    risky = deterministic_debate(risky_ctx, failed, risky_vote, settings)
    assert risky.bear is not None and calm.bear is not None and risky.bear.conviction > calm.bear.conviction
    assert any("VIX" in a for a in risky.bear.top_arguments) or any("FAILED" in a for a in risky.bear.top_arguments)


def test_debate_logged_to_jsonl(settings: Settings, prompts: PromptStore, intraday: Any) -> None:
    ctx, reps, vote = intraday
    run_debate(ctx, reps, vote, settings, RulesClient(), prompts)
    events = read_events(NOW.date(), "debates")
    assert events and events[-1]["status"] == "OK" and events[-1]["symbol"] == "RELIANCE"
    assert len(events[-1]["transcript"]) == 2


def test_turn_schema_cleans_cosmetics() -> None:
    out = DebateTurnOutput.model_validate({"stance": "x" * 500, "arguments": ["a"] * 9, "rebuttals": None,
                                           "conviction": "87%"})
    assert len(out.stance) <= 240 and len(out.arguments) == 5 and out.rebuttals == [] and out.conviction == 87
    with pytest.raises(ValueError):
        DebateTurnOutput.model_validate({"stance": "x", "arguments": [], "conviction": 50})
    assert SWING_NOW.date() == date(2026, 11, 24)
