"""Layer-3 meta agent: option mapping, code validation, gates, AUTO mode, failures, rules backend."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest

from agents.aggregate import VoteSummary, weighted_vote
from agents.debate.debate import run_debate
from agents.llm import FakeLLMClient, LLMResult, RulesClient
from agents.meta.meta import (
    MetaOptions,
    MetaOutput,
    build_options,
    deterministic_confidence,
    deterministic_meta,
    run_meta,
    time_validity_minutes,
)
from agents.prompt_store import PromptStore
from agents.runner import run_sub_agents
from config.schema import Settings
from core.clock import IST
from core.enums import (
    AgentStatus,
    Direction,
    EntryType,
    EventType,
    Grade,
    Mode,
    SelectionPolicy,
    Signal,
    SLMethod,
    TradeDecision,
    TrailingMethod,
)
from core.logging import read_events
from core.types import (
    AgentReport,
    ConfluenceResult,
    DebateResult,
    MarketEvent,
    MetaDecision,
    Quote,
    SLCandidate,
    StockContext,
)
from tests.agents.agents_kit import (
    NOW,
    clean_fundamentals,
    intraday_ctx,
    intraday_signal,
    report,
    swing_ctx,
    swing_signal,
)

ENABLED = [Mode.INTRADAY, Mode.SWING]


def pick(options: list[Any], price: float) -> int:
    return next(o.index for o in options if abs(o.price - price) < 0.011)


def answer(opts: MetaOptions, **kw: Any) -> dict[str, Any]:
    """A valid BUY answer for the intraday kit (MARKET at 100, stop 98.8, targets 102 / 103)."""
    base: dict[str, Any] = {
        "decision": "BUY", "strategy_index": 0, "entry_type": "MARKET", "recommended_mode": None,
        "recommended_mode_reason": None, "confidence": 80, "reasons_for": ["trend and volume aligned"],
        "reasons_against_considered": ["bear: charges are small"], "why_not": None,
    }
    for key, options, price in (("stop_loss_index", opts.stops, 98.8), ("target_1_index", opts.targets, 102.0),
                                ("target_2_index", opts.targets, 103.0)):
        if key not in kw:
            base[key] = pick(options, price)
    base.update(kw)
    return base


class Case:
    """Analysed context: reports, vote, debate and options."""

    def __init__(self, ctx: StockContext, settings: Settings, prompts: PromptStore,
                 reports: list[AgentReport] | None = None) -> None:
        self.ctx = ctx
        self.reports = reports if reports is not None else run_sub_agents(ctx, settings, RulesClient(), prompts)
        self.vote = weighted_vote(self.reports, settings.mode(ctx.mode).agent_weights)
        self.debate = run_debate(ctx, self.reports, self.vote, settings, RulesClient(), prompts)
        self.opts = build_options(ctx, self.reports)
        self.settings = settings
        self.prompts = prompts

    def meta(self, llm: Any, *, vote: VoteSummary | None = None, prompts: Any = "default",
             enabled: list[Mode] | None = None) -> MetaDecision:
        return run_meta(self.ctx, self.reports, vote if vote is not None else self.vote, self.debate, self.settings,
                        llm, self.prompts if prompts == "default" else prompts,
                        enabled_modes=enabled if enabled is not None else ENABLED)


@pytest.fixture
def case(settings: Settings, prompts: PromptStore) -> Case:
    return Case(intraday_ctx(), settings, prompts)


# --------------------------------------------------------------------------- happy paths


def test_llm_choices_are_mapped_to_code_prices(case: Case, settings: Settings) -> None:
    fake = FakeLLMClient({"meta": [answer(case.opts, quantity=500)]})
    d = case.meta(fake)
    assert d.decision == TradeDecision.BUY and d.status == AgentStatus.OK and d.mode == Mode.INTRADAY
    assert (d.entry_type, d.entry_price, d.stop_loss, d.target_1, d.target_2) == (
        EntryType.MARKET, 100.0, 98.8, 102.0, 103.0)
    assert d.strategy == "opening_range_breakout" and d.confidence == 80 and d.why_not is None
    assert d.trailing_rule == TrailingMethod.ATR and d.max_holding_days == 0
    assert d.time_validity_minutes == 20  # strategy valid_until
    assert d.reasons_for == ["trend and volume aligned"] and d.reasons_against_considered
    assert d.model == settings.models.model_for("meta") and d.prompt_version == 1 and d.latency_ms is not None
    assert d.recommended_mode_reason is None
    # quantity is never part of the decision (the risk manager sizes trades)
    assert not any("qty" in f or "quant" in f for f in list(MetaDecision.model_fields) + list(MetaOutput.model_fields))
    call = fake.calls_for("meta")[0]
    assert call["think"] == settings.models.reasoning_think and call["model"] == settings.models.profiles["low"].reasoning
    assert "OPTIONS (choose by index only)" in call["user"] and '"suggested_choice"' in call["user"]


def test_limit_entry_uses_entry_timing_agents_validated_price(settings: Settings, prompts: PromptStore) -> None:
    base = Case(intraday_ctx(), settings, prompts)
    timing = report("entry_timing", Signal.BULLISH, 60, extra={
        "entry_type": "LIMIT", "entry_price": 99.5, "setup_strategy": "opening_range_breakout"})
    reps = [r for r in base.reports if r.agent != "entry_timing"] + [timing]
    c = Case(intraday_ctx(), settings, prompts, reports=reps)
    d = c.meta(FakeLLMClient({"meta": answer(c.opts, entry_type="LIMIT")}))
    assert d.decision == TradeDecision.BUY and d.entry_type == EntryType.LIMIT and d.entry_price == 99.5


def test_retry_after_invalid_choice_then_success(case: Case) -> None:
    fake = FakeLLMClient({"meta": [answer(case.opts, stop_loss_index=99), answer(case.opts)]})
    d = case.meta(fake)
    assert d.decision == TradeDecision.BUY and len(fake.calls) == 2
    corrective = fake.calls[1]["messages"][-1]["content"]
    assert "stop_loss_index 99 does not exist" in corrective


def test_llm_no_trade_keeps_reason_and_shadow_prices(case: Case) -> None:
    fake = FakeLLMClient({"meta": {"decision": "no trade", "confidence": 35, "why_not": "bear case stronger",
                                   "strategy_index": "null"}})
    d = case.meta(fake)
    assert d.decision == TradeDecision.NO_TRADE and d.status == AgentStatus.OK and d.why_not == "bear case stronger"
    assert d.confidence == 35 and d.model == case.settings.models.profiles["low"].reasoning
    assert (d.strategy, d.stop_loss, d.target_1) == ("opening_range_breakout", 98.8, 102.0)  # for shadow tracking


# --------------------------------------------------------------------------- validation rejections


def _failed(d: MetaDecision) -> None:
    assert d.decision == TradeDecision.NO_TRADE and d.status == AgentStatus.FAILED
    assert d.error and d.why_not and d.why_not.startswith("meta agent FAILED")


def test_invented_price_index_rejected(case: Case) -> None:
    fake = FakeLLMClient({"meta": answer(case.opts, target_1_index=57)})
    d = case.meta(fake)
    _failed(d)
    assert len(fake.calls) == 3 and "target_1_index 57 does not exist" in (d.error or "")
    assert d.model == case.settings.models.profiles["low"].reasoning and d.prompt_version == 1


def test_bad_rr_rejected(case: Case) -> None:
    d = case.meta(FakeLLMClient({"meta": answer(case.opts, stop_loss_index=pick(case.opts.stops, 98.4),
                                                target_1_index=pick(case.opts.targets, 101.8))}))
    _failed(d)
    assert "below the INTRADAY minimum 1.5" in (d.error or "")


def test_stop_on_wrong_side_rejected(settings: Settings, prompts: PromptStore) -> None:
    ctx = intraday_ctx(quote=Quote(symbol="RELIANCE", ltp=98.7, timestamp=NOW))
    c = Case(ctx, settings, prompts)
    d = c.meta(FakeLLMClient({"meta": answer(c.opts, entry_type="MARKET")}))
    _failed(d)
    assert "is not below the MARKET entry 98.70" in (d.error or "")


def test_stop_outside_atr_limits_rejected(case: Case, settings: Settings) -> None:
    settings.risk.sl_max_atr = 1.4  # strategy stop 98.8 is 1.5 ATR from entry
    d = case.meta(FakeLLMClient({"meta": answer(case.opts)}))
    _failed(d)
    assert "too wide" in (d.error or "")


def test_sell_short_on_long_setup_rejected(case: Case) -> None:
    d = case.meta(FakeLLMClient({"meta": answer(case.opts, decision="SELL_SHORT")}))
    _failed(d)
    assert "does not match the LONG setup" in (d.error or "")


def test_sell_short_in_swing_rejected(settings: Settings, prompts: PromptStore) -> None:
    c = Case(swing_ctx(events=[]), settings, prompts)
    d = c.meta(FakeLLMClient({"meta": answer(c.opts, decision="SELL_SHORT", entry_type="LIMIT",
                                             stop_loss_index=pick(c.opts.stops, 485.0),
                                             target_1_index=pick(c.opts.targets, 530.0), target_2_index=None)}))
    _failed(d)
    assert "SELL_SHORT only in INTRADAY" in (d.error or "")


def test_short_signal_in_swing_is_never_offered(settings: Settings, prompts: PromptStore) -> None:
    short = swing_signal(direction=Direction.SHORT, stop_loss=515.0, target_1=470.0, target_2=None)
    c = Case(swing_ctx(events=[], strategy_signals=[short]), settings, prompts)
    fake = FakeLLMClient({"meta": {"decision": "SELL_SHORT", "strategy_index": 0, "confidence": 99}})
    d = c.meta(fake)
    assert d.decision == TradeDecision.NO_TRADE and d.status == AgentStatus.OK
    assert "no overnight shorts" in (d.why_not or "") and fake.calls == []


def test_low_confidence_is_no_trade_not_failure(case: Case) -> None:
    d = case.meta(FakeLLMClient({"meta": answer(case.opts, confidence=60)}))
    assert d.decision == TradeDecision.NO_TRADE and d.status == AgentStatus.OK and d.error is None
    assert d.why_not == "confidence 60 is below the INTRADAY threshold 70" and d.confidence == 60
    assert (d.entry_price, d.stop_loss, d.target_1, d.target_2) == (100.0, 98.8, 102.0, 103.0)


def test_confidence_capped_when_vote_opposes(case: Case) -> None:
    bearish = case.vote.model_copy(update={"score": -30.0})
    d = case.meta(FakeLLMClient({"meta": answer(case.opts, confidence=95)}), vote=bearish)
    assert d.decision == TradeDecision.NO_TRADE and d.confidence == 50 and "below" in (d.why_not or "")


# --------------------------------------------------------------------------- deterministic gates (no LLM call)


@pytest.mark.parametrize("case_name, expected", [
    ("not_tradeable", "TRADEABLE=false"),
    ("no_signal", "no strategy signal"),
    ("grade_c", "grade C is below"),
    ("low_score", "below the INTRADAY minimum 80.0"),
    ("no_confluence", "no confluence"),
])
def test_gates_return_no_trade_without_llm(case_name: str, expected: str, settings: Settings,
                                           prompts: PromptStore) -> None:
    ctx = intraday_ctx()
    if case_name == "not_tradeable":
        ctx = intraday_ctx(fundamentals=clean_fundamentals(promoter_pledge_pct=60.0))
    elif case_name == "no_signal":
        ctx = intraday_ctx(strategy_signals=[])
    elif case_name == "grade_c":
        ctx = intraday_ctx(confluence=ConfluenceResult(score=55.0, grade=Grade.C))
    elif case_name == "low_score":
        settings.mode(Mode.INTRADAY).min_confluence = 80.0
    else:
        ctx = intraday_ctx(confluence=None)
    c = Case(ctx, settings, prompts)
    fake = FakeLLMClient({"meta": {"decision": "BUY", "strategy_index": 0, "confidence": 99}})
    d = c.meta(fake)
    assert d.decision == TradeDecision.NO_TRADE and d.status == AgentStatus.OK and expected in (d.why_not or "")
    assert fake.calls == []


def test_blocking_event_gate(settings: Settings, prompts: PromptStore) -> None:
    c = Case(swing_ctx(), settings, prompts)  # RESULTS on 2026-11-27 inside the window
    d = c.meta(FakeLLMClient({}))
    assert d.decision == TradeDecision.NO_TRADE and "blocking event" in (d.why_not or "")
    assert d.stop_loss == 485.0 and d.strategy == "swing_pullback_ema"  # shadow prices


# --------------------------------------------------------------------------- AUTO selection policy


def test_auto_policy_recommends_enabled_mode(case: Case, settings: Settings) -> None:
    settings.modes.selection_policy = SelectionPolicy.AUTO
    ans = answer(case.opts, recommended_mode="swing", recommended_mode_reason="daily trend supports a multi-day hold",
                 target_1_index=pick(case.opts.targets, 102.4), target_2_index=pick(case.opts.targets, 103.0))
    fake = FakeLLMClient({"meta": ans})
    d = case.meta(fake)
    assert d.decision == TradeDecision.BUY and d.mode == Mode.SWING
    assert d.recommended_mode_reason == "daily trend supports a multi-day hold"
    assert d.target_1 == 102.4 and d.trailing_rule == TrailingMethod.EMA20_DAILY and d.max_holding_days == 10
    assert '"enabled_modes"' in fake.calls[0]["user"] and "Selection policy: AUTO" in fake.calls[0]["user"]


@pytest.mark.parametrize("kw, error", [
    ({"recommended_mode": "POSITIONAL", "recommended_mode_reason": "x"}, "POSITIONAL is not enabled"),
    ({"recommended_mode": "SWING", "recommended_mode_reason": ""}, "recommended_mode_reason is required"),
    ({"recommended_mode": "SWING", "recommended_mode_reason": "x"}, "below the SWING minimum 2.0"),  # T1 = 1.67R
])
def test_auto_policy_rejects_invalid_recommendations(kw: dict[str, Any], error: str, case: Case,
                                                    settings: Settings) -> None:
    settings.modes.selection_policy = SelectionPolicy.AUTO
    d = case.meta(FakeLLMClient({"meta": answer(case.opts, **kw)}))
    _failed(d)
    assert error in (d.error or "")


def test_auto_policy_rejects_short_into_overnight_mode(settings: Settings, prompts: PromptStore) -> None:
    settings.modes.selection_policy = SelectionPolicy.AUTO
    sig = intraday_signal(direction=Direction.SHORT, stop_loss=101.2, target_1=97.6, target_2=None)
    stops = [SLCandidate(method=SLMethod.ATR, price=101.2, distance=1.2, distance_atr=1.5)]
    reps = [report("technical", Signal.BEARISH, 80), report("price_action", Signal.BEARISH, 70)]
    c = Case(intraday_ctx(strategy_signals=[sig], sl_candidates=stops), settings, prompts, reports=reps)
    ans = {"decision": "SELL_SHORT", "strategy_index": 0, "entry_type": "MARKET", "confidence": 90,
           "stop_loss_index": pick(c.opts.stops, 101.2), "target_1_index": pick(c.opts.targets, 97.6),
           "recommended_mode": "SWING", "recommended_mode_reason": "hold it"}
    d = c.meta(FakeLLMClient({"meta": ans}))
    _failed(d)
    assert "does not allow SHORT" in (d.error or "")
    ok = c.meta(FakeLLMClient({"meta": {**ans, "recommended_mode": None}}))
    assert ok.decision == TradeDecision.SELL_SHORT and ok.stop_loss == 101.2 and ok.target_1 == 97.6


def test_fixed_policy_ignores_recommendation(case: Case) -> None:
    d = case.meta(FakeLLMClient({"meta": answer(case.opts, recommended_mode="SWING",
                                                recommended_mode_reason="would be nice")}))
    assert d.decision == TradeDecision.BUY and d.mode == Mode.INTRADAY and d.recommended_mode_reason is None


# --------------------------------------------------------------------------- failures


class _RaisingClient:
    backend = "broken"

    def generate(self, *a: Any, **k: Any) -> LLMResult:
        raise ConnectionError("ollama is down")

    def healthcheck(self) -> bool:
        return False

    def list_models(self) -> list[str]:
        return []


def test_failures_degrade_to_failed_no_trade(case: Case) -> None:
    d = case.meta(FakeLLMClient({"meta": ["{not json"]}))
    _failed(d)
    assert "LLM failed after 3 attempt(s)" in (d.error or "") and d.target_1 == 102.0
    _failed(case.meta(_RaisingClient()))
    _failed(case.meta(FakeLLMClient({"meta": answer(case.opts)}), prompts=None))


def test_meta_disabled_uses_deterministic_decision(case: Case, settings: Settings) -> None:
    settings.agents.enabled["meta"] = False
    fake = FakeLLMClient({"meta": answer(case.opts)})
    d = case.meta(fake)
    assert d.model == "rules" and fake.calls == []


def test_meta_logged_to_jsonl(case: Case) -> None:
    case.meta(FakeLLMClient({"meta": answer(case.opts)}))
    events = read_events(NOW.date(), "meta")
    assert events and events[-1]["decision"]["decision"] == "BUY" and events[-1]["symbol"] == "RELIANCE"


# --------------------------------------------------------------------------- rules backend


def test_rules_backend_deterministic_buy(case: Case, settings: Settings) -> None:
    d = case.meta(RulesClient())
    again = deterministic_meta(case.ctx, case.reports, case.vote, case.debate, settings, enabled_modes=ENABLED)
    assert d.decision == TradeDecision.BUY and d.model == "rules" and d.status == AgentStatus.OK
    assert (d.entry_type, d.entry_price, d.stop_loss, d.target_1, d.target_2) == (
        EntryType.MARKET, 100.0, 98.8, 102.0, 103.0)
    sig = case.ctx.strategy_signals[0]
    assert d.confidence == deterministic_confidence(case.ctx, sig, case.vote, case.debate)
    assert d.model_dump(exclude={"latency_ms"}) == again.model_dump(exclude={"latency_ms"})


def test_deterministic_confidence_formula(case: Case) -> None:
    sig = case.ctx.strategy_signals[0]
    no_debate = deterministic_confidence(case.ctx, sig, case.vote, None)
    assert no_debate == round(0.6 * 78.0 + 0.4 * (case.vote.score + 100) / 2)
    tilted = DebateResult(bull=case.debate.bull.model_copy(update={"conviction": 90}),
                          bear=case.debate.bear.model_copy(update={"conviction": 10}))
    assert deterministic_confidence(case.ctx, sig, case.vote, tilted) == min(100, round(
        0.6 * 78.0 + 0.4 * (case.vote.score + 100) / 2 + 8))


def test_rules_no_trade_when_vote_not_aligned(case: Case, settings: Settings) -> None:
    flat = case.vote.model_copy(update={"score": 5.0})
    d = deterministic_meta(case.ctx, case.reports, flat, case.debate, settings)
    assert d.decision == TradeDecision.NO_TRADE and "not aligned" in (d.why_not or "") and d.model == "rules"


def test_rules_no_trade_when_grade_below_minimum(settings: Settings, prompts: PromptStore) -> None:
    settings.mode(Mode.INTRADAY).min_grade = Grade.A_PLUS
    c = Case(intraday_ctx(), settings, prompts)
    d = c.meta(RulesClient())
    assert d.decision == TradeDecision.NO_TRADE and "grade A is below the INTRADAY minimum A+" in (d.why_not or "")


def test_rules_sell_short_allowed_intraday(settings: Settings, prompts: PromptStore) -> None:
    sig = intraday_signal(direction=Direction.SHORT, stop_loss=101.2, target_1=98.0, target_2=97.0)
    stops = [SLCandidate(method=SLMethod.ATR, price=100.8, distance=0.8, distance_atr=1.0)]
    reps = [report("technical", Signal.BEARISH, 90), report("price_action", Signal.BEARISH, 90)]
    c = Case(intraday_ctx(strategy_signals=[sig], sl_candidates=stops), settings, prompts, reports=reps)
    d = c.meta(RulesClient())
    assert d.decision == TradeDecision.SELL_SHORT and (d.stop_loss, d.target_1, d.target_2) == (101.2, 98.0, 97.0)


# --------------------------------------------------------------------------- schema / helpers


def test_meta_output_normalises_cosmetics() -> None:
    out = MetaOutput.model_validate({"decision": "Sell Short", "entry_type": "market", "strategy_index": "1",
                                     "stop_loss_index": "null", "recommended_mode": "swing", "confidence": "88%",
                                     "reasons_for": "single reason", "why_not": "none"})
    assert out.decision == TradeDecision.SELL_SHORT and out.entry_type == EntryType.MARKET
    assert out.strategy_index == 1 and out.stop_loss_index is None and out.recommended_mode == Mode.SWING
    assert out.confidence == 88 and out.reasons_for == ["single reason"] and out.why_not is None
    assert MetaOutput.model_validate({"decision": "buy", "confidence": 1}).decision == TradeDecision.BUY
    assert MetaOutput.model_validate({"decision": "NO_TRADE", "confidence": 1,
                                      "recommended_mode": "None"}).recommended_mode is None
    with pytest.raises(ValueError):
        MetaOutput.model_validate({"decision": "MAYBE", "confidence": 50})


def test_time_validity_defaults() -> None:
    plain = intraday_signal(valid_until=None)
    assert time_validity_minutes(Mode.INTRADAY, plain, NOW) == 15
    assert time_validity_minutes(Mode.BTST, plain, datetime(2026, 11, 24, 14, 50, tzinfo=IST)) == 30
    assert time_validity_minutes(Mode.SWING, swing_signal(), NOW) is None
    assert time_validity_minutes(Mode.INTRADAY, intraday_signal(), NOW) == 20


def test_options_include_agent_picks_and_strategy_prices(case: Case) -> None:
    opts = case.opts
    assert opts.usable == [0] and opts.reference == 0 and opts.ltp == 100.0 and opts.atr == 0.8
    stop = opts.stops[pick(opts.stops, 98.8)]
    assert any("stop_loss agent choice" in s for s in stop.sources) and any("strategy" in s for s in stop.sources)
    assert all(o.price < 100.0 for o in opts.stops)  # only protective candidates for a LONG
    assert {102.0, 103.0} <= {o.price for o in opts.targets}
    assert opts.agent_trailing == TrailingMethod.ATR and opts.agent_holding_days == 0
    events = [MarketEvent(symbol="RELIANCE", event_type=EventType.RESULTS, event_date=date(2026, 11, 24))]
    assert build_options(intraday_ctx(events=events), case.reports).usable == [0]
