"""Owner funding across batches: top-up to the starting capital, withdrawals, overall P&L."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import delete, select

from capital.funding import funding_summary, topup_preview
from control.bootstrap import bootstrap
from control.commands import enqueue_command, get_command
from control.processor import CommandProcessor
from core.clock import IST, SimulatedClock
from core.enums import BatchStatus, CommandType
from data.calendar import load_calendar
from data.providers.replay_provider import ReplayDataProvider
from storage.models import Batch, CapitalAccount, FundingEvent


@pytest.fixture
def svc(tmp_path: Path):  # type: ignore[no-untyped-def]
    clock = SimulatedClock(datetime(2026, 11, 25, 16, 0, tzinfo=IST))
    provider = ReplayDataProvider({}, clock, load_calendar())
    return bootstrap(process="test", clock=clock, db_url=f"sqlite:///{(tmp_path / 'f.db').as_posix()}",
                     primary_provider=provider, llm_backend="rules", with_notifier=False, offline=True,
                     runtime_dir=tmp_path / "rt", log_dir=tmp_path / "logs", console_log=False)


def _set_equity(svc, cash: float) -> None:  # type: ignore[no-untyped-def]
    with svc.session() as s:
        acct = s.get(CapitalAccount, svc.ctx.batch_id)
        acct.trading_cash = cash
        s.commit()


def _new_batch(svc, **payload) -> dict:  # type: ignore[no-untyped-def,type-arg]
    with svc.session() as s:
        row = enqueue_command(s, CommandType.NEW_BATCH, {"confirm_text": "CONFIRM", **payload}, actor="owner",
                              batch_id=svc.ctx.batch_id, clock=svc.clock)
        s.commit()
        cid = row.id
    CommandProcessor(svc).process_pending()
    with svc.session() as s:
        row = get_command(s, cid)
        return {"status": row.status, **(row.result_json or {})}


def test_first_batch_is_the_initial_deposit(svc) -> None:  # type: ignore[no-untyped-def]
    with svc.session() as s:
        ev = s.execute(select(FundingEvent)).scalar_one()
        assert ev.kind == "INITIAL" and ev.amount == 1000.0 and ev.prev_batch_id is None
        summ = funding_summary(s)
    assert summ["net_invested"] == 1000.0 and summ["current_capital"] == 1000.0 and summ["overall_pnl"] == 0.0


def test_new_batch_after_a_loss_records_the_top_up(svc) -> None:  # type: ignore[no-untyped-def]
    _set_equity(svc, 995.28)                                   # what batch 1 ended with on 2026-11-25
    with svc.session() as s:
        pv = topup_preview(s, 1000.0)
    assert pv["action"] == "ADD" and pv["top_up"] == 4.72 and "Add ₹4.72" in pv["message"]

    r = _new_batch(svc)                                        # no name, no capital: defaults
    assert r["ok"], r
    assert r["data"]["top_up"] == 4.72 and r["data"]["carried_in"] == 995.28
    assert "added ₹4.72" in r["message"]
    with svc.session() as s:
        new = s.get(Batch, r["data"]["batch_id"])
        assert new.name == "Batch 2" and new.starting_capital == 1000.0 and new.status == BatchStatus.ACTIVE.value
        ev = s.execute(select(FundingEvent).where(FundingEvent.batch_id == new.id)).scalar_one()
        assert ev.kind == "TOP_UP" and ev.amount == pytest.approx(4.72) and ev.prev_equity == pytest.approx(995.28)
        summ = funding_summary(s)
    assert summ["total_deposited"] == 1004.72 and summ["total_withdrawn"] == 0.0
    assert summ["net_invested"] == 1004.72
    assert summ["overall_pnl"] == -4.72 and summ["current_capital"] == 1000.0
    b1, b2 = summ["batches"]
    assert b1["pnl"] == -4.72 and b1["status"] == BatchStatus.CLOSED.value and b1["top_up"] == 1000.0
    assert b2["top_up"] == 4.72 and b2["carried_in"] == 995.28 and b2["pnl"] == 0.0
    assert summ["active_batch_id"] == b2["batch_id"]


def test_new_batch_after_a_profit_withdraws_the_excess(svc) -> None:  # type: ignore[no-untyped-def]
    _set_equity(svc, 1100.0)
    r = _new_batch(svc, starting_capital=1000.0, name="Fresh start")
    assert r["ok"] and r["data"]["top_up"] == -100.0 and "withdrew ₹100.00" in r["message"]
    with svc.session() as s:
        summ = funding_summary(s)
    assert summ["total_withdrawn"] == 100.0 and summ["net_invested"] == 900.0
    assert summ["overall_pnl"] == 100.0 and summ["current_capital"] == 1000.0
    assert summ["batches"][1]["funding_kind"] == "WITHDRAW"


def test_custom_capital_from_the_dashboard(svc) -> None:  # type: ignore[no-untyped-def]
    _set_equity(svc, 980.0)
    r = _new_batch(svc, starting_capital=2000.0)
    assert r["ok"] and r["data"]["top_up"] == 1020.0
    with svc.session() as s:
        summ = funding_summary(s)
    assert summ["net_invested"] == 2020.0 and summ["current_capital"] == 2000.0 and summ["overall_pnl"] == -20.0


def test_batches_without_records_are_derived(svc) -> None:  # type: ignore[no-untyped-def]
    _set_equity(svc, 990.0)
    assert _new_batch(svc)["ok"]
    with svc.session() as s:
        s.execute(delete(FundingEvent))                        # a database from before this feature
        s.commit()
        summ = funding_summary(s)
    kinds = [(e["kind"], e["amount"], e["derived"]) for e in summ["events"]]
    assert kinds == [("INITIAL", 1000.0, True), ("TOP_UP", 10.0, True)]
    assert summ["current_capital"] == 1000.0 and summ["overall_pnl"] == -10.0


def test_preview_when_nothing_to_add(svc) -> None:  # type: ignore[no-untyped-def]
    with svc.session() as s:
        pv = topup_preview(s, 1000.0)
    assert pv["action"] == "NONE" and pv["top_up"] == 0.0
