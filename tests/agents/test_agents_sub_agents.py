"""Layer-1 sub-agents: SubAgent.run (OK / FAILED / SKIPPED / rules) and finalize code validation."""

from __future__ import annotations

from typing import Any

import pytest

from agents.base import SubAgent
from agents.llm import FakeLLMClient, RulesClient
from agents.prompt_store import PromptStore
from agents.registry import agents_for_mode, all_agents, get_agent
from config.schema import AGENT_NAMES, Settings
from core.enums import AgentStatus, Direction, Mode, Signal, SLMethod
from core.types import AgentReport, SLCandidate, StockContext
from tests.agents.agents_kit import (
    answer,
    clean_fundamentals,
    good_answers,
    intraday_ctx,
    intraday_signal,
    market_ctx,
    swing_ctx,
)


def ctx_for(agent: str) -> StockContext:
    """event_risk only applies to overnight modes and needs events: use the swing context."""
    return swing_ctx() if agent == "event_risk" else intraday_ctx()


def run(agent: str, ctx: StockContext, settings: Settings, prompts: PromptStore,
        answers: dict[str, Any] | None = None) -> tuple[AgentReport, FakeLLMClient]:
    fake = FakeLLMClient(answers if answers is not None else good_answers())
    return get_agent(agent).run(ctx, fake, settings, prompts), fake


def test_registry_has_all_agents_in_order(settings: Settings) -> None:
    assert [a.name for a in all_agents()] == list(AGENT_NAMES)
    assert "event_risk" not in [a.name for a in agents_for_mode(Mode.INTRADAY, settings)]
    assert "event_risk" in [a.name for a in agents_for_mode(Mode.SWING, settings)]


# --------------------------------------------------------------------------- OK path

EXPECTED: dict[str, tuple[Signal, int]] = {
    "fundamental": (Signal.NEUTRAL, 50), "technical": (Signal.BULLISH, 72), "price_action": (Signal.BULLISH, 66),
    "volume_liquidity": (Signal.BULLISH, 64), "market_context": (Signal.BULLISH, 60),
    "news_sentiment": (Signal.BULLISH, 58), "event_risk": (Signal.BEARISH, 85),
    "entry_timing": (Signal.BULLISH, 62), "stop_loss": (Signal.BULLISH, 60), "exit_target": (Signal.BULLISH, 61),
}


@pytest.mark.parametrize("agent", AGENT_NAMES)
def test_ok_path_every_agent(agent: str, settings: Settings, prompts: PromptStore) -> None:
    ctx = ctx_for(agent)
    rep, fake = run(agent, ctx, settings, prompts)

    assert rep.status == AgentStatus.OK and rep.error is None
    assert (rep.signal, rep.confidence) == EXPECTED[agent]
    assert rep.agent == agent and rep.symbol == ctx.symbol and rep.mode == ctx.mode and rep.timestamp == ctx.as_of
    assert rep.model == settings.models.model_for(agent) and rep.prompt_version == 1
    assert rep.data_used and rep.latency_ms is not None and rep.tokens_in and rep.tokens_out
    assert rep.extra["attempts"] == 1
    call = fake.calls_for(agent)[0]
    assert call["think"] is False and call["temperature"] == settings.models.temperature
    assert ctx.symbol in call["user"] and "FACTS" in call["user"]


def test_ok_path_agent_specific_extras(settings: Settings, prompts: PromptStore) -> None:
    ctx = intraday_ctx()
    assert run("fundamental", ctx, settings, prompts)[0].tradeable is True
    assert run("price_action", ctx, settings, prompts)[0].extra["key_pattern"] == "CDLENGULFING"
    assert run("volume_liquidity", ctx, settings, prompts)[0].extra["liquidity_ok"] is True
    assert run("market_context", ctx, settings, prompts)[0].extra["risk_off"] is False
    en = run("entry_timing", ctx, settings, prompts)[0].extra
    assert en["entry_type"] == "MARKET" and en["entry_price"] == 100.0 and en["candidate_index"] == 0
    sl = run("stop_loss", ctx, settings, prompts)[0].extra
    assert sl["stop_loss"] == 98.8 and sl["candidate_index"] == 3 and sl["validation_notes"] == []
    ex = run("exit_target", ctx, settings, prompts)[0].extra
    assert ex["target_1"] == 101.8 and ex["rr_1"] == 1.5 and ex["trailing_rule"] == "ATR" and ex["max_holding_days"] == 0
    ev = run("event_risk", swing_ctx(), settings, prompts)[0].extra
    assert ev["event_risk"] is True and ev["blocking_events"][0].startswith("E1: RESULTS")


# --------------------------------------------------------------------------- FAILED / SKIPPED / rules


@pytest.mark.parametrize("agent", AGENT_NAMES)
def test_failed_path_votes_neutral_with_zero_confidence(agent: str, settings: Settings, prompts: PromptStore) -> None:
    rep, fake = run(agent, ctx_for(agent), settings, prompts, {"*": ["not json at all"]})
    assert rep.status == AgentStatus.FAILED
    assert rep.signal == Signal.NEUTRAL and rep.effective_signal == Signal.NEUTRAL and rep.confidence == 0
    assert "LLM failed after 3 attempt(s)" in (rep.error or "")
    assert rep.extra["attempts"] == 3 and len(fake.calls) == 3
    assert rep.model == settings.models.model_for(agent) and rep.prompt_version == 1
    assert "deterministic" in rep.extra and "fallback" in rep.extra  # code facts kept for audit


def test_failed_on_exception_and_missing_prompt_store(settings: Settings, prompts: PromptStore) -> None:
    class Boom(SubAgent):  # deliberately NOT registered
        name = "boom_test"

        def build_payload(self, ctx: StockContext, settings: Settings | None = None) -> dict[str, Any]:
            raise RuntimeError("kaboom")

        def deterministic_report(self, ctx: StockContext, settings: Settings | None = None) -> AgentReport:
            return self.make_report(ctx)

    rep = Boom().run(intraday_ctx(), FakeLLMClient({}), settings, prompts)
    assert rep.status == AgentStatus.FAILED and rep.error == "RuntimeError: kaboom" and rep.confidence == 0
    rep = get_agent("technical").run(intraday_ctx(), FakeLLMClient(good_answers()), settings, None)
    assert rep.status == AgentStatus.FAILED and "no PromptStore" in (rep.error or "")


def test_skipped_when_disabled_or_not_applicable(settings: Settings, prompts: PromptStore) -> None:
    ctx = intraday_ctx()
    rep, fake = run("event_risk", ctx, settings, prompts)
    assert rep.status == AgentStatus.SKIPPED and rep.extra["skip_reason"] == "not applicable in INTRADAY"
    settings.agents.enabled["technical"] = False
    rep2, _ = run("technical", ctx, settings, prompts)
    assert rep2.status == AgentStatus.SKIPPED and rep2.extra["skip_reason"] == "agent disabled in settings"
    settings.mode(Mode.INTRADAY).agents_enabled["price_action"] = False
    rep3, _ = run("price_action", ctx, settings, prompts)
    assert rep3.status == AgentStatus.SKIPPED and "INTRADAY" in rep3.extra["skip_reason"]
    for r in (rep, rep2, rep3):
        assert r.signal == Signal.NEUTRAL and r.confidence == 0
    assert fake.calls == []


@pytest.mark.parametrize("agent", AGENT_NAMES)
def test_rules_backend_is_deterministic(agent: str, settings: Settings, prompts: PromptStore) -> None:
    ctx = ctx_for(agent)
    a = get_agent(agent)
    rep1 = a.run(ctx, RulesClient(), settings, prompts)
    rep2 = a.run(ctx, RulesClient(), settings, None)  # no prompt store needed
    det = a.precheck(ctx, settings) or a.deterministic_report(ctx, settings)
    assert rep1.status == AgentStatus.OK
    assert (rep1.signal, rep1.confidence, rep1.key_findings) == (det.signal, det.confidence, det.key_findings)
    assert rep1.model_dump(exclude={"latency_ms"}) == rep2.model_dump(exclude={"latency_ms"})
    if not a.precheck(ctx, settings):
        assert rep1.model == "rules"


def test_skipped_llm_answer_falls_back_to_rules(settings: Settings, prompts: PromptStore) -> None:
    proxy = RulesClient()
    proxy.backend = "proxy"  # not recognised as 'rules' up front: exercises the res.skipped branch
    rep = get_agent("technical").run(intraday_ctx(), proxy, settings, prompts)
    det = get_agent("technical").deterministic_report(intraday_ctx(), settings)
    assert rep.status == AgentStatus.OK and rep.model == "rules" and rep.signal == det.signal


# --------------------------------------------------------------------------- setup agents without a setup


@pytest.mark.parametrize("agent", ["entry_timing", "stop_loss", "exit_target"])
def test_setup_agents_without_strategy_signal_skip_the_llm(agent: str, settings: Settings,
                                                           prompts: PromptStore) -> None:
    rep, fake = run(agent, intraday_ctx(strategy_signals=[]), settings, prompts)
    assert rep.status == AgentStatus.OK and rep.signal == Signal.NEUTRAL and rep.confidence == 0
    assert rep.extra.get("no_setup") is True and "no setup" in rep.key_findings[0]
    assert fake.calls == []


# --------------------------------------------------------------------------- stop-loss validation


def test_stop_loss_rejects_invented_price_and_falls_back(settings: Settings, prompts: PromptStore) -> None:
    ans = {"stop_loss": [answer("BULLISH", 80, candidate_index=99, price=97.77)]}
    rep, _ = run("stop_loss", intraday_ctx(), settings, prompts, ans)
    ex = rep.extra
    valid_prices = {c["price"] for c in ex["candidates"] if c["valid"]}
    assert ex["stop_loss"] in valid_prices and ex["stop_loss"] == 98.4  # nearest valid to the invented price
    assert any("97.77 is not a deterministic candidate" in n for n in ex["validation_notes"])
    assert any("does not exist" in n for n in ex["validation_notes"])
    assert rep.confidence <= 50


def test_stop_loss_uses_candidate_price_not_llm_price(settings: Settings, prompts: PromptStore) -> None:
    rep, _ = run("stop_loss", intraday_ctx(), settings, prompts,
                 {"stop_loss": [answer("BULLISH", 70, candidate_index=2, price=98.55)]})
    assert rep.extra["stop_loss"] == 98.9 and rep.confidence == 70
    assert any("candidate price used" in n for n in rep.extra["validation_notes"])


def test_stop_loss_invalid_candidate_too_wide_falls_back(settings: Settings, prompts: PromptStore) -> None:
    rep, _ = run("stop_loss", intraday_ctx(), settings, prompts,
                 {"stop_loss": [answer("BULLISH", 70, candidate_index=6)]})  # 97.5 = 3.1 ATR
    assert rep.extra["stop_loss"] == 98.4
    assert any("too wide" in n for n in rep.extra["validation_notes"])


def test_stop_loss_rule_prefers_strategy_stop_even_when_it_matches_another_candidate(
        settings: Settings, prompts: PromptStore) -> None:
    # 98.8 is both the strategy stop and the 1.5x ATR candidate: the rule must pick it (regression).
    rep = get_agent("stop_loss").run(intraday_ctx(), RulesClient(), settings, prompts)
    assert rep.extra["stop_loss"] == 98.8


def test_stop_loss_short_setup_only_accepts_stops_above_entry(settings: Settings, prompts: PromptStore) -> None:
    sig = intraday_signal(direction=Direction.SHORT, stop_loss=101.2, target_1=98.0, target_2=97.0)
    ctx = intraday_ctx(strategy_signals=[sig])
    rep, _ = run("stop_loss", ctx, settings, prompts, {"stop_loss": [answer("BEARISH", 70, candidate_index=0)]})
    assert rep.extra["stop_loss"] == 101.2 and rep.signal == Signal.BEARISH
    assert any("not above the SHORT entry" in n for n in rep.extra["validation_notes"])


def test_setup_agent_cannot_vote_against_its_setup(settings: Settings, prompts: PromptStore) -> None:
    rep, _ = run("stop_loss", intraday_ctx(), settings, prompts,
                 {"stop_loss": [answer("BEARISH", 90, candidate_index=3)]})
    assert rep.signal == Signal.NEUTRAL


# --------------------------------------------------------------------------- fundamental / event / liquidity verdicts


def test_fundamental_tradeable_false_cannot_be_overridden(settings: Settings, prompts: PromptStore) -> None:
    ctx = intraday_ctx(fundamentals=clean_fundamentals(in_asm=True))
    rep, _ = run("fundamental", ctx, settings, prompts,
                 {"fundamental": [answer("BULLISH", 90, tradeable=True)]})
    assert rep.status == AgentStatus.OK and rep.tradeable is False and rep.signal != Signal.BULLISH
    assert rep.confidence <= 60  # quality-filter cap
    assert any("ignored" in n for n in rep.extra["validation_notes"])
    assert any("ASM" in f for f in rep.key_findings)
    failed, _ = run("fundamental", ctx, settings, prompts, {"fundamental": ["garbage"]})
    assert failed.status == AgentStatus.FAILED and failed.tradeable is False  # code verdict survives a failure


def test_fundamental_scored_signal_in_swing(settings: Settings, prompts: PromptStore) -> None:
    rep = get_agent("fundamental").run(swing_ctx(), RulesClient(), settings, prompts)
    assert rep.extra["fundamental_score"] is not None and rep.extra["role"] == "SCORED_SIGNAL"


def test_event_risk_blocking_event_forces_bearish(settings: Settings, prompts: PromptStore) -> None:
    rep, _ = run("event_risk", swing_ctx(), settings, prompts,
                 {"event_risk": [answer("BULLISH", 90, event_risk=False, blocking_events=["E9: invented"])]})
    assert rep.signal == Signal.BEARISH and rep.confidence >= 85
    assert rep.extra["event_risk"] is True and len(rep.extra["blocking_events"]) == 1
    notes = " ".join(rep.extra["validation_notes"])
    assert "cannot vote BULLISH" in notes and "E9" in notes


def test_event_risk_without_events_skips_llm(settings: Settings, prompts: PromptStore) -> None:
    rep, fake = run("event_risk", swing_ctx(events=[]), settings, prompts)
    assert rep.status == AgentStatus.OK and rep.signal == Signal.NEUTRAL and rep.extra["event_risk"] is False
    assert fake.calls == []


def test_liquidity_verdict_and_risk_off_are_code_owned(settings: Settings, prompts: PromptStore) -> None:
    ctx = intraday_ctx(features=intraday_ctx().features.model_copy(update={"avg_daily_turnover_20": 1e6}))
    rep, _ = run("volume_liquidity", ctx, settings, prompts,
                 {"volume_liquidity": [answer("BULLISH", 90, liquidity_ok=True)]})
    assert rep.extra["liquidity_ok"] is False and rep.confidence == 30
    rep2, _ = run("market_context", intraday_ctx(market=market_ctx(vix_elevated=True)), settings, prompts,
                  {"market_context": [answer("BULLISH", 80, risk_off=False)]})
    assert rep2.extra["risk_off"] is True and rep2.confidence == 60


def test_price_action_drops_undetected_pattern(settings: Settings, prompts: PromptStore) -> None:
    rep, _ = run("price_action", intraday_ctx(), settings, prompts,
                 {"price_action": [answer("BULLISH", 60, key_pattern="CDLMADEUP")]})
    assert rep.extra["key_pattern"] is None and rep.extra["validation_notes"]


def test_crosscheck_caps_llm_contradicting_confident_rules(settings: Settings, prompts: PromptStore) -> None:
    rep, _ = run("technical", intraday_ctx(), settings, prompts,
                 {"technical": [answer("BEARISH", 90, trend_alignment="ALIGNED_DOWN")]})
    assert rep.signal == Signal.BEARISH and rep.confidence == 50 and "crosscheck" in rep.extra


# --------------------------------------------------------------------------- exit / entry validation


def test_exit_target_enforces_mode_min_rr(settings: Settings, prompts: PromptStore) -> None:
    settings.mode(Mode.INTRADAY).min_rr = 2.0
    rep, _ = run("exit_target", intraday_ctx(), settings, prompts,
                 {"exit_target": [answer("BULLISH", 70, target_1_index=0, target_2_index=0,
                                         trailing_rule="EMA20_DAILY", max_holding_days=5)]})
    ex = rep.extra
    assert all(c["rr"] >= 2.0 for c in ex["candidates"])
    assert any(o["label"].endswith("target_1") for o in ex["obstacles"])  # strategy T1 (1.67R) is an obstacle
    assert ex["rr_1"] >= 2.0
    assert ex["target_2"] is not None and ex["rr_2"] >= ex["rr_1"] + 0.5  # LLM's t2 = t1 replaced by a farther one
    assert ex["trailing_rule"] == "ATR" and ex["max_holding_days"] == 0
    notes = " ".join(ex["validation_notes"])
    assert "not allowed in INTRADAY" in notes and "clamped" in notes


def test_exit_target_out_of_range_index_uses_nearest_valid(settings: Settings, prompts: PromptStore) -> None:
    rep, _ = run("exit_target", intraday_ctx(), settings, prompts,
                 {"exit_target": [answer("BULLISH", 70, target_1_index=42, trailing_rule="NONE",
                                         max_holding_days=0)]})
    assert rep.extra["target_1_index"] == 0 and rep.extra["rr_1"] >= settings.mode(Mode.INTRADAY).min_rr


def test_exit_target_never_offers_non_positive_prices(settings: Settings, prompts: PromptStore) -> None:
    sig = intraday_signal(direction=Direction.SHORT, stop_loss=140.0, target_1=30.0, target_2=None)
    rep = get_agent("exit_target").run(intraday_ctx(strategy_signals=[sig], sl_candidates=[]), RulesClient(),
                                       settings, prompts)
    assert rep.extra["candidates"] and all(c["price"] > 0 for c in rep.extra["candidates"])


def test_entry_timing_only_accepts_candidate_zones(settings: Settings, prompts: PromptStore) -> None:
    ctx = intraday_ctx()
    base, _ = run("entry_timing", ctx, settings, prompts)
    prices = {c["price"] for c in base.extra["candidates"]}
    limit_ok = {c["index"] for c in base.extra["candidates"] if c["limit_ok"]}
    assert limit_ok
    rep, _ = run("entry_timing", ctx, settings, prompts,
                 {"entry_timing": [answer("BULLISH", 70, entry_type="LIMIT", candidate_index=0)]})
    assert rep.extra["entry_type"] == "LIMIT" and rep.extra["candidate_index"] in limit_ok
    assert rep.extra["entry_price"] in prices and rep.extra["validation_notes"]
    rep, _ = run("entry_timing", ctx, settings, prompts,
                 {"entry_timing": [answer("BULLISH", 70, entry_type="market", candidate_index=3)]})
    assert rep.extra["candidate_index"] == 0 and rep.extra["entry_price"] == 100.0
    rep, _ = run("entry_timing", ctx, settings, prompts,
                 {"entry_timing": [answer("BULLISH", 70, entry_type="WAIT", candidate_index=77)]})
    assert rep.signal == Signal.NEUTRAL and rep.extra["candidate_index"] != 0
    assert rep.extra["entry_price"] in prices and any("does not exist" in n for n in rep.extra["validation_notes"])


def test_sl_candidates_are_recomputed_against_setup_entry(settings: Settings, prompts: PromptStore) -> None:
    far = SLCandidate(method=SLMethod.ATR, price=99.0, distance=5.0, distance_atr=9.9, note="stale distance")
    rep = get_agent("stop_loss").run(intraday_ctx(sl_candidates=[far]), RulesClient(), settings, prompts)
    cand = next(c for c in rep.extra["candidates"] if c["price"] == 99.0)
    assert cand["distance_atr"] == 1.25 and cand["valid"]
