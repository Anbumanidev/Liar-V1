"""Approval queue, routing, manual trades and shadow tracking (phase 8)."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import select

from approvals.manual import ManualTrades
from approvals.queue import ApprovalQueue
from approvals.router import route_decision
from capital.accounts import CapitalManager
from config.loader import update_settings
from config.schema import HardCaps, Settings
from control.payloads import ManualTradePayload, ModifyApprovePayload
from core.clock import IST, SimulatedClock
from core.enums import (
    ApprovalStatus,
    BatchStatus,
    Direction,
    EntryType,
    Grade,
    Mode,
    OrderStatus,
    RejectionSource,
    RiskRule,
    ShadowOutcome,
    Timeframe,
)
from core.types import TradeProposal
from data.calendar import load_calendar
from paper_engine.broker import PaperBroker
from risk.manager import RiskManager
from risk.state import build_risk_state
from shadow.tracker import ShadowTracker
from storage.models import ApprovalItem, Batch, DecisionRow, Order, ShadowTrade
from tests.paper_engine.paper_fakes import FakeData, bars

DAY = date(2026, 11, 25)
NEXT = date(2026, 11, 26)


def at(d: date, hh: int, mm: int) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=IST)


@pytest.fixture
def env(settings: Settings, caps: HardCaps, session):  # type: ignore[no-untyped-def]
    s = update_settings(settings, {"engine.slippage_pct": 0.0})
    clock = SimulatedClock(at(DAY, 10, 0))
    cal = load_calendar()
    data = FakeData(clock)
    capital = CapitalManager(s, clock)
    batch = Batch(name="A", starting_capital=1000.0, status=BatchStatus.ACTIVE.value, created_at=clock.now())
    session.add(batch)
    session.flush()
    capital.initialize_batch(session, batch.id, 1000.0)
    risk = RiskManager(s, caps)
    broker = PaperBroker(s, caps, clock, capital, data, cal)
    shadow = ShadowTracker(s, clock, cal, data)
    queue = ApprovalQueue(s, clock, cal, risk, broker, capital, shadow, data=data)
    manual = ManualTrades(s, clock, cal, data, risk, broker, capital)
    return SimpleNamespace(s=s, clock=clock, cal=cal, data=data, capital=capital, risk=risk, broker=broker,
                           shadow=shadow, queue=queue, manual=manual, session=session, batch_id=batch.id)


def proposal(env, **kw) -> TradeProposal:  # type: ignore[no-untyped-def]
    base = dict(batch_id=env.batch_id, symbol="INFY", mode=Mode.INTRADAY, direction=Direction.LONG,
                entry_price=100.0, entry_type=EntryType.LIMIT, stop_loss=99.0, target_1=101.5, target_2=103.0,
                strategy="vwap_pullback", confidence=80, confluence_score=75.0, grade=Grade.A, atr=1.0,
                sector="IT", created_at=env.clock.now())
    base.update(kw)
    return TradeProposal(**base)


def queued(env, **kw) -> ApprovalItem:  # type: ignore[no-untyped-def]
    p = proposal(env, **kw)
    st = build_risk_state(env.session, batch_id=env.batch_id, capital_manager=env.capital, clock=env.clock,
                          calendar=env.cal)
    v = env.risk.evaluate(p, st, check_timing=False)
    assert v.approved, v.message
    return env.queue.enqueue(env.session, p, v, summary={"bull": ["a"], "bear": ["b"]})


# ---------------------------------------------------------------- deadlines


def test_deadlines_per_mode(env) -> None:  # type: ignore[no-untyped-def]
    assert env.queue.deadline(Mode.INTRADAY, at(DAY, 10, 0)) == at(DAY, 10, 3)
    assert env.queue.deadline(Mode.BTST, at(DAY, 14, 50)) == at(DAY, 15, 0)
    assert env.queue.deadline(Mode.SWING, at(DAY, 16, 30)) == at(NEXT, 9, 45)
    assert env.queue.deadline(Mode.SWING, at(NEXT, 8, 0)) == at(NEXT, 9, 45)
    # Friday evening -> Monday (skips the weekend)
    assert env.queue.deadline(Mode.SWING, at(date(2026, 11, 27), 17, 0)) == at(date(2026, 11, 30), 9, 45)


# ---------------------------------------------------------------- approve / modify / reject / timeout


def test_approve_submits_order(env) -> None:  # type: ignore[no-untyped-def]
    item = queued(env)
    res = env.queue.approve(env.session, item.id, actor="owner")
    assert res.ok, res.message
    assert item.status == ApprovalStatus.APPROVED.value and item.order_id is not None
    order = env.session.get(Order, item.order_id)
    assert order.status == OrderStatus.PENDING.value and order.approval_id == item.id and order.qty == item.qty


def test_risk_manager_vetoes_owner_approval(env) -> None:  # type: ignore[no-untyped-def]
    item = queued(env)
    env.risk.update_settings(update_settings(env.s, {"modes.modes.INTRADAY.enabled": False}))
    res = env.queue.approve(env.session, item.id, actor="owner")
    assert not res.ok and res.rule == RiskRule.MODE_DISABLED
    assert item.status == ApprovalStatus.PENDING.value and item.block_rule == "MODE_DISABLED"


def test_modify_qty_capped_by_risk_max(env) -> None:  # type: ignore[no-untyped-def]
    item = queued(env)
    res = env.queue.modify_and_approve(env.session, ModifyApprovePayload(approval_id=item.id, qty=item.max_qty + 5),
                                       actor="owner")
    assert not res.ok and res.rule == RiskRule.QTY_ABOVE_MAX
    res = env.queue.modify_and_approve(env.session, ModifyApprovePayload(approval_id=item.id, qty=1, stop_loss=99.2,
                                                                         reason="smaller"), actor="owner")
    assert res.ok, res.message
    assert item.status == ApprovalStatus.MODIFIED.value
    assert item.user_edits_json["qty"] == {"old": item.qty, "new": 1}
    assert item.user_edits_json["stop_loss"]["new"] == 99.2


def test_modify_to_short_in_swing_is_blocked(env) -> None:  # type: ignore[no-untyped-def]
    item = queued(env, direction=Direction.SHORT, stop_loss=101.0, target_1=98.5, target_2=97.0)
    res = env.queue.modify_and_approve(env.session, ModifyApprovePayload(approval_id=item.id, mode=Mode.SWING),
                                       actor="owner")
    assert not res.ok and res.rule == RiskRule.NO_OVERNIGHT_SHORT


def test_reject_creates_user_shadow(env) -> None:  # type: ignore[no-untyped-def]
    item = queued(env)
    res = env.queue.reject(env.session, item.id, actor="owner", reason="don't like it")
    assert res.ok and item.status == ApprovalStatus.REJECTED.value
    sh = env.session.execute(select(ShadowTrade)).scalar_one()
    assert sh.rejection_source == RejectionSource.USER.value and sh.approval_id == item.id


def test_timeout_expires_and_shadows(env) -> None:  # type: ignore[no-untyped-def]
    item = queued(env)
    env.clock.set(at(DAY, 10, 4))
    assert env.queue.approve(env.session, item.id, actor="owner").ok is False
    assert item.status == ApprovalStatus.TIMEOUT.value
    other = queued(env, symbol="TCS")
    env.clock.set(at(DAY, 10, 10))
    expired = env.queue.expire(env.session)
    assert [i.id for i in expired] == [other.id]
    sources = sorted(r.rejection_source for r in env.session.execute(select(ShadowTrade)).scalars())
    assert sources == ["TIMEOUT", "TIMEOUT"]


def test_bulk_reject(env) -> None:  # type: ignore[no-untyped-def]
    a, b = queued(env), queued(env, symbol="TCS")
    res = env.queue.bulk_reject(env.session, [a.id, b.id, 999], actor="owner", reason="bulk")
    assert res.data["rejected"] == [a.id, b.id] and res.data["failed"] == [999]


# ---------------------------------------------------------------- routing


def _decision(env) -> DecisionRow:  # type: ignore[no-untyped-def]
    row = DecisionRow(batch_id=env.batch_id, symbol="INFY", mode="INTRADAY", decision="BUY", confidence=80,
                      created_at=env.clock.now())
    env.session.add(row)
    env.session.flush()
    return row


@pytest.mark.parametrize("policy,final", [("AUTO", "EXECUTED"), ("MANUAL_APPROVE", "QUEUED"),
                                          ("SIGNALS_ONLY", "SIGNAL_ONLY")])
def test_route_by_policy(env, policy: str, final: str) -> None:  # type: ignore[no-untyped-def]
    s = update_settings(env.s, {"modes.modes.INTRADAY.execution_policy": policy})
    p = proposal(env)
    st = build_risk_state(env.session, batch_id=env.batch_id, capital_manager=env.capital, clock=env.clock,
                          calendar=env.cal)
    v = env.risk.evaluate(p, st)
    assert v.approved, v.message
    d = _decision(env)
    out = route_decision(env.session, settings=s, decision=d, proposal=p, verdict=v, broker=env.broker,
                         queue=env.queue, shadow=env.shadow)
    assert out == final and d.final_action == final
    if final == "SIGNAL_ONLY":
        assert env.session.execute(select(ShadowTrade)).scalar_one().rejection_source == "SIGNALS_ONLY"


def test_route_risk_rejection_shadows(env) -> None:  # type: ignore[no-untyped-def]
    p = proposal(env, confidence=10)
    st = build_risk_state(env.session, batch_id=env.batch_id, capital_manager=env.capital, clock=env.clock,
                          calendar=env.cal)
    v = env.risk.evaluate(p, st)
    d = _decision(env)
    out = route_decision(env.session, settings=env.s, decision=d, proposal=p, verdict=v, broker=env.broker,
                         queue=env.queue, shadow=env.shadow)
    assert out == "RISK_REJECTED"
    sh = env.session.execute(select(ShadowTrade)).scalar_one()
    assert sh.rejection_source == "RISK" and sh.rejection_rule == "CONFIDENCE_BELOW_MIN"


# ---------------------------------------------------------------- manual trades


def test_manual_trade_blocked_shows_rule(env) -> None:  # type: ignore[no-untyped-def]
    res = env.manual.submit(env.session, ManualTradePayload(symbol="INFY", mode=Mode.INTRADAY, direction=Direction.LONG,
                                                            entry_type=EntryType.LIMIT, entry_price=100, qty=2,
                                                            stop_loss=100.5, target_1=102),
                            actor="owner", batch_id=env.batch_id)
    assert not res.ok and res.rule == RiskRule.SL_WRONG_SIDE


def test_manual_trade_submitted(env) -> None:  # type: ignore[no-untyped-def]
    res = env.manual.submit(env.session, ManualTradePayload(symbol="INFY", mode=Mode.INTRADAY, direction=Direction.LONG,
                                                            entry_type=EntryType.LIMIT, entry_price=100, qty=2,
                                                            stop_loss=99, target_1=102),
                            actor="owner", batch_id=env.batch_id)
    assert res.ok, res.message
    order = env.session.get(Order, res.data["order_id"])
    assert order.source == "MANUAL" and order.qty == 2


def test_manual_swing_short_blocked(env) -> None:  # type: ignore[no-untyped-def]
    res = env.manual.submit(env.session, ManualTradePayload(symbol="INFY", mode=Mode.SWING, direction=Direction.SHORT,
                                                            entry_type=EntryType.LIMIT, entry_price=100, qty=1,
                                                            stop_loss=103, target_1=94),
                            actor="owner", batch_id=env.batch_id)
    assert not res.ok and res.rule == RiskRule.NO_OVERNIGHT_SHORT


# ---------------------------------------------------------------- shadow simulation


def _row(**kw) -> ShadowTrade:  # type: ignore[no-untyped-def]
    base = dict(symbol="X", mode="INTRADAY", direction="LONG", entry_price=100.0, stop_loss=99.0, target_1=101.5,
                target_2=None, qty=2, rejection_source="USER", created_at=at(DAY, 10, 0),
                expires_at=at(DAY, 15, 15), entered_at=None, status="OPEN", outcome="OPEN")
    base.update(kw)
    return ShadowTrade(**base)


def _df(rows: list[tuple[float, float, float, float]], start: str = "10:01") -> pd.DataFrame:
    return bars(DAY, start, rows)


def test_shadow_outcomes() -> None:
    now = at(DAY, 12, 0)
    sim = ShadowTracker.simulate
    assert sim(_row(), _df([(100.2, 100.3, 99.95, 100.1), (100.0, 100.1, 98.9, 99.0)]), now).outcome == ShadowOutcome.SL
    assert sim(_row(), _df([(100.2, 100.3, 99.95, 100.1), (100.2, 101.6, 100.0, 101.5)]), now).outcome \
        == ShadowOutcome.TARGET_1
    r = sim(_row(target_2=103.0), _df([(100.0, 101.6, 99.95, 101.5), (101.5, 103.1, 101.4, 103.0)]), now)
    assert r.outcome == ShadowOutcome.TARGET_2 and r.exit_price == 103.0
    # target 1 then back to breakeven -> TARGET_1 credited
    r = sim(_row(target_2=103.0), _df([(100.0, 101.6, 99.95, 101.5), (101.5, 101.6, 99.9, 100.0)]), now)
    assert r.outcome == ShadowOutcome.TARGET_1
    # never touched the limit entry -> NOT_TRIGGERED after expiry
    late = at(DAY, 15, 30)
    assert sim(_row(), _df([(100.5, 100.8, 100.3, 100.6)]), late).outcome == ShadowOutcome.NOT_TRIGGERED
    # entered, neither stop nor target before expiry -> TIME
    r = sim(_row(), _df([(100.2, 100.3, 99.95, 100.1), (100.1, 100.4, 99.6, 100.3)]), late)
    assert r.outcome == ShadowOutcome.TIME and r.exit_price == 100.3
    # same bar touches stop and target -> stop first (conservative)
    assert sim(_row(entered_at=at(DAY, 10, 0)), _df([(100.0, 102.0, 98.5, 101.0)]), now).outcome == ShadowOutcome.SL


def test_shadow_update_closes_rows(env) -> None:  # type: ignore[no-untyped-def]
    row = env.shadow.create(env.session, batch_id=env.batch_id, symbol="SBIN", mode=Mode.INTRADAY,
                            direction=Direction.LONG, entry=100.0, stop_loss=99.0, target_1=101.5, qty=3,
                            source=RejectionSource.META, entry_type=EntryType.MARKET)
    assert row is not None and row.expires_at == at(DAY, 15, 15)
    env.data.add("SBIN", Timeframe.M1, bars(DAY, "10:00", [(100.0, 100.2, 99.8, 100.1), (100.1, 101.6, 100.0, 101.5)]))
    env.clock.set(at(DAY, 10, 2))
    assert env.shadow.update(env.session) == 1
    assert row.outcome == "TARGET_1" and row.r_multiple == pytest.approx(1.5) and row.hypothetical_net_pnl > 0
    assert env.shadow.create(env.session, batch_id=env.batch_id, symbol="BAD", mode=Mode.INTRADAY,
                             direction=Direction.LONG, entry=100, stop_loss=101, target_1=102,
                             source=RejectionSource.META) is None


@pytest.mark.parametrize("autonomy,grade,conf,final", [
    ("FULL_AUTO", Grade.B, 70, "EXECUTED"),
    ("HUMAN_IN_THE_LOOP", Grade.A_PLUS, 99, "QUEUED"),
    ("SIGNALS_ONLY", Grade.A_PLUS, 99, "SIGNAL_ONLY"),
    ("HYBRID", Grade.A_PLUS, 90, "EXECUTED"),
    ("HYBRID", Grade.A, 90, "QUEUED"),
])
def test_global_autonomy_overrides_mode_policy(env, autonomy: str, grade: Grade, conf: int, final: str) -> None:  # type: ignore[no-untyped-def]
    s = update_settings(env.s, {"modes.autonomy": autonomy, "modes.modes.INTRADAY.execution_policy": "MANUAL_APPROVE",
                                "modes.modes.INTRADAY.min_grade": "B"})
    p = proposal(env, grade=grade, confidence=conf)
    st = build_risk_state(env.session, batch_id=env.batch_id, capital_manager=env.capital, clock=env.clock,
                          calendar=env.cal)
    v = env.risk.evaluate(p, st)
    assert v.approved, v.message
    d = _decision(env)
    out = route_decision(env.session, settings=s, decision=d, proposal=p, verdict=v, broker=env.broker,
                         queue=env.queue, shadow=env.shadow)
    assert out == final
    assert d.context_json["autonomy"] == autonomy
