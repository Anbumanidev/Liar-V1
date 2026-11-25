"""End-to-end LangGraph pipeline: context -> strategies -> agents -> debate -> meta -> risk -> route."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import func, select

from agents.graph import AnalysisPipeline, PipelineServices, open_run
from agents.llm import build_llm
from approvals.queue import ApprovalQueue
from approvals.router import FINAL_ANALYZED, FINAL_FILTERED
from capital.accounts import CapitalManager
from config.loader import update_settings
from config.schema import HardCaps, Settings
from core.clock import IST, SimulatedClock
from core.enums import Direction, EntryType, Grade, Mode, RunType, Timeframe, TrailingMethod
from core.types import ConfluenceResult, StrategySignal
from data.calendar import load_calendar
from data.universe import UniverseService
from features.builder import FeatureBuilder
from paper_engine.broker import PaperBroker
from risk.manager import RiskManager
from shadow.tracker import ShadowTracker
from storage.models import AgentReportRow, ApprovalItem, Batch, DecisionRow, ShadowTrade, StockAnalyzed
from tests.helpers import make_daily_candles, make_intraday_candles
from tests.paper_engine.paper_fakes import FakeData

DAY = date(2026, 11, 25)
DAYS = [date(2026, 11, 18), date(2026, 11, 19), date(2026, 11, 20), date(2026, 11, 23), DAY]


@pytest.fixture
def pipe(settings: Settings, caps: HardCaps, session_factory):  # type: ignore[no-untyped-def]
    s = update_settings(settings, {"engine.llm_backend": "rules", "engine.slippage_pct": 0.0})
    clock = SimulatedClock(datetime(2026, 11, 25, 11, 0, tzinfo=IST))
    cal = load_calendar()
    data = FakeData(clock)
    for sym, px, seed in (("INFY", 100.0, 1), ("NIFTY50", 24000.0, 2), ("BANKNIFTY", 52000.0, 3)):
        data.add(sym, Timeframe.M5, make_intraday_candles(DAYS, 5, px, seed=seed, drift=0.0003))
        data.add(sym, Timeframe.M15, make_intraday_candles(DAYS, 15, px, seed=seed, drift=0.0006))
        data.add(sym, Timeframe.H1, make_intraday_candles(DAYS, 60, px, seed=seed, drift=0.001))
        data.add(sym, Timeframe.M1, make_intraday_candles(DAYS[-1:], 1, px, seed=seed))
        data.add(sym, Timeframe.D1, make_daily_candles(date(2026, 11, 23), 260, px * 0.8, seed=seed))
    data.add("INDIAVIX", Timeframe.D1, make_daily_candles(date(2026, 11, 23), 60, 13.0, seed=9, drift=0.0))
    capital = CapitalManager(s, clock)
    with session_factory() as session:
        batch = Batch(name="P", starting_capital=1000.0, status="ACTIVE", created_at=clock.now())
        session.add(batch)
        session.flush()
        capital.initialize_batch(session, batch.id, 1000.0)
        run = open_run(session, RunType.INTRADAY_CYCLE, batch_id=batch.id, mode=Mode.INTRADAY, clock=clock)
        session.commit()
        batch_id, run_id = batch.id, run.id
    risk = RiskManager(s, caps)
    broker = PaperBroker(s, caps, clock, capital, data, cal)
    shadow = ShadowTracker(s, clock, cal, data)
    queue = ApprovalQueue(s, clock, cal, risk, broker, capital, shadow, data=data)
    services = PipelineServices(
        settings=s, caps=caps, clock=clock, session_factory=session_factory, calendar=cal, data=data,
        features=FeatureBuilder(data, cal), llm=build_llm(s), prompts=None, risk=risk, capital=capital,
        broker=broker, queue=queue, shadow=shadow, universe=UniverseService(),
    )
    return AnalysisPipeline(services), batch_id, run_id, session_factory


def _signal_factory(direction: Direction = Direction.LONG):  # type: ignore[no-untyped-def]
    def fake_run_strategies(features, market, mode, settings, now, disabled=frozenset()):  # type: ignore[no-untyped-def]
        ltp, atr = features.ltp, features.atr or features.ltp * 0.005
        sign = 1 if direction == Direction.LONG else -1
        return [StrategySignal(strategy="vwap_pullback", mode=mode, direction=direction, entry_price=round(ltp, 2),
                               entry_type=EntryType.LIMIT, stop_loss=round(ltp - sign * atr, 2),
                               target_1=round(ltp + sign * 2 * atr, 2), target_2=round(ltp + sign * 3 * atr, 2),
                               strength=85, trailing_rule=TrailingMethod.ATR)]
    return fake_run_strategies


def test_analyze_now_full_path_without_routing(pipe) -> None:  # type: ignore[no-untyped-def]
    pipeline, batch_id, run_id, sf = pipe
    res = pipeline.analyze("INFY", Mode.INTRADAY, batch_id=batch_id, run_id=run_id, force=True, route=False)
    assert res.error is None, res.error
    assert res.decision_id is not None and res.final_action == FINAL_ANALYZED
    with sf() as s:
        row = s.get(DecisionRow, res.decision_id)
        assert row.symbol == "INFY" and row.final_action == FINAL_ANALYZED
        assert row.context_json is not None and "signals" in row.context_json
        n_reports = s.scalar(select(func.count(AgentReportRow.id)).where(AgentReportRow.decision_id == row.id))
        assert n_reports == 10                       # every layer-1 agent reported (OK / SKIPPED / FAILED)
        assert s.scalar(select(func.count(StockAnalyzed.id))) == 1


def test_no_setup_is_filtered_without_llm(pipe, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pipeline, batch_id, run_id, sf = pipe
    monkeypatch.setattr("strategies.registry.run_strategies", lambda *a, **k: [])
    res = pipeline.analyze("INFY", Mode.INTRADAY, batch_id=batch_id, run_id=run_id)
    assert res.final_action == FINAL_FILTERED and res.filtered_reason == "no strategy setup"
    with sf() as s:
        assert s.scalar(select(func.count(AgentReportRow.id))) == 0
        assert s.get(DecisionRow, res.decision_id).decision == "NO_TRADE"


def test_low_confluence_filtered_and_shadowed(pipe, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pipeline, batch_id, run_id, sf = pipe
    monkeypatch.setattr("strategies.registry.run_strategies", _signal_factory())
    monkeypatch.setattr("strategies.confluence.compute_confluence",
                        lambda *a, **k: ConfluenceResult(score=30.0, grade=Grade.C))
    res = pipeline.analyze("INFY", Mode.INTRADAY, batch_id=batch_id, run_id=run_id)
    assert res.final_action == FINAL_FILTERED and "confluence" in (res.filtered_reason or "")
    with sf() as s:
        sh = s.execute(select(ShadowTrade)).scalar_one()
        assert sh.rejection_source == "FILTER" and sh.decision_id == res.decision_id


def test_setup_runs_agents_and_routes_consistently(pipe, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pipeline, batch_id, run_id, sf = pipe
    monkeypatch.setattr("strategies.registry.run_strategies", _signal_factory())
    monkeypatch.setattr("strategies.confluence.compute_confluence",
                        lambda *a, **k: ConfluenceResult(score=88.0, grade=Grade.A_PLUS))
    res = pipeline.analyze("INFY", Mode.INTRADAY, batch_id=batch_id, run_id=run_id)
    assert res.error is None, res.error
    assert res.meta is not None
    with sf() as s:
        row = s.get(DecisionRow, res.decision_id)
        assert s.scalar(select(func.count(AgentReportRow.id)).where(AgentReportRow.decision_id == row.id)) == 10
        if res.meta.decision.value == "NO_TRADE":
            assert row.final_action == "NO_TRADE" and res.verdict is None
            assert s.execute(select(ShadowTrade)).scalar_one().rejection_source == "META"
        else:
            assert res.verdict is not None
            if res.verdict.approved:
                assert row.final_action == "QUEUED"
                item = s.execute(select(ApprovalItem)).scalar_one()
                assert item.decision_id == row.id and item.qty == res.verdict.qty
            else:
                assert row.final_action == "RISK_REJECTED" and row.risk_rule == res.verdict.rule.value


def test_feature_failure_degrades_to_no_trade(pipe, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pipeline, batch_id, run_id, sf = pipe

    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("feature engine exploded")

    monkeypatch.setattr(pipeline.s.features, "build", boom)
    res = pipeline.analyze("INFY", Mode.INTRADAY, batch_id=batch_id, run_id=run_id)
    assert res.final_action == FINAL_FILTERED and "feature build failed" in (res.filtered_reason or "")
    with sf() as s:
        assert s.get(DecisionRow, res.decision_id).decision == "NO_TRADE"


# --------------------------------------------------------------------------- two-phase cycle


def _add_symbol(pipeline, sym: str, px: float, seed: int) -> None:  # type: ignore[no-untyped-def]
    data = pipeline.s.data
    data.add(sym, Timeframe.M5, make_intraday_candles(DAYS, 5, px, seed=seed, drift=0.0003))
    data.add(sym, Timeframe.M15, make_intraday_candles(DAYS, 15, px, seed=seed, drift=0.0006))
    data.add(sym, Timeframe.H1, make_intraday_candles(DAYS, 60, px, seed=seed, drift=0.001))
    data.add(sym, Timeframe.M1, make_intraday_candles(DAYS[-1:], 1, px, seed=seed))
    data.add(sym, Timeframe.D1, make_daily_candles(date(2026, 11, 23), 260, px * 0.8, seed=seed))


def _cycle_env(pipe, monkeypatch, **settings_updates):  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    pipeline, batch_id, run_id, sf = pipe
    _add_symbol(pipeline, "TCS", 100.0, 4)
    _add_symbol(pipeline, "WIPRO", 100.0, 5)
    monkeypatch.setattr("strategies.registry.run_strategies", _signal_factory())
    scores = {"INFY": 70.0, "TCS": 90.0, "WIPRO": 80.0}
    monkeypatch.setattr("strategies.confluence.compute_confluence",
                        lambda features, *a, **k: ConfluenceResult(score=scores[features.symbol], grade=Grade.A))
    llm_calls: list[str] = []
    import agents.runner as runner

    real = runner.run_sub_agents

    def spy(ctx, *a, **k):  # type: ignore[no-untyped-def]
        llm_calls.append(ctx.symbol)
        return real(ctx, *a, **k)

    monkeypatch.setattr(runner, "run_sub_agents", spy)
    s = update_settings(pipeline.s.settings, settings_updates)
    pipeline.update_settings(s)
    svc = SimpleNamespace(settings=s, pipeline=pipeline, risk=pipeline.s.risk, clock=pipeline.s.clock)
    return svc, batch_id, run_id, sf, llm_calls


def test_cycle_sends_only_the_best_setups_to_the_llm(pipe, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from scheduler.jobs import _analyse_chunk

    svc, batch_id, run_id, sf, llm_calls = _cycle_env(pipe, monkeypatch, **{"engine.max_llm_candidates_per_cycle": 1})
    actions = _analyse_chunk(svc, Mode.INTRADAY, ["INFY", "TCS", "WIPRO"], batch_id=batch_id, run_id=run_id,
                             check_timing=False)
    assert llm_calls == ["TCS"]                                   # highest confluence only
    assert actions.get(FINAL_FILTERED) == 2 and sum(actions.values()) == 3
    with sf() as s:
        rows = {r.symbol: r for r in s.execute(select(DecisionRow)).scalars()}
        assert set(rows) == {"INFY", "TCS", "WIPRO"}
        for sym in ("INFY", "WIPRO"):
            assert rows[sym].final_action == FINAL_FILTERED
            assert "top 1 by confluence" in rows[sym].context_json["filtered_reason"]
        assert rows["TCS"].final_action != FINAL_FILTERED
        # skipped setups are still shadow-tracked
        assert {sh.symbol for sh in s.execute(select(ShadowTrade)).scalars()} >= {"INFY", "WIPRO"}


def test_cycle_spends_no_llm_time_outside_the_entry_window(pipe, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from scheduler.jobs import _analyse_chunk

    svc, batch_id, run_id, sf, llm_calls = _cycle_env(pipe, monkeypatch)
    svc.clock.set(datetime(2026, 11, 25, 12, 30, tzinfo=IST))     # midday lull: no INTRADAY entries
    _analyse_chunk(svc, Mode.INTRADAY, ["INFY", "TCS"], batch_id=batch_id, run_id=run_id, check_timing=True)
    assert llm_calls == []
    with sf() as s:
        reasons = [r.context_json["filtered_reason"] for r in s.execute(select(DecisionRow)).scalars()]
        assert reasons and all("outside the entry window" in r for r in reasons)


def test_proposal_carries_execution_quality_inputs(pipe, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pipeline, batch_id, run_id, sf = pipe
    monkeypatch.setattr("strategies.registry.run_strategies", _signal_factory())
    monkeypatch.setattr("strategies.confluence.compute_confluence",
                        lambda *a, **k: ConfluenceResult(score=88.0, grade=Grade.A_PLUS, room_r=3.2))
    res = pipeline.analyze("INFY", Mode.INTRADAY, batch_id=batch_id, run_id=run_id)
    if res.verdict is None:
        pytest.skip("rules meta chose NO_TRADE for this synthetic data")
    with sf() as s:
        prop = s.get(DecisionRow, res.decision_id).context_json["proposal"]
    assert prop["failed_agents_pct"] == 0.0                       # rules backend: nobody failed
    assert prop["room_r"] == 3.2 and "valid_until" in prop      # (the fake signal sets no validity)
    assert prop["debate_bull"] is not None and prop["debate_bear"] is not None
