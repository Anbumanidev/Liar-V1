"""CapitalManager tests (section 6 + 6.4 guarantees)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from capital.accounts import CapitalManager
from capital.errors import AccountNotFoundError, CapitalError, InsufficientCashError, UnconfirmedTransferError
from config.schema import Settings
from core.clock import IST, SimulatedClock
from core.enums import Direction, DrawdownTier, LedgerEntryType as L, Mode, Product
from core.errors import LedgerImbalanceError, ReserveAccessError
from core.logging import read_events
from core.types import OpenPositionView
from storage.models import AuditLog, CapitalLedgerEntry, DailySummary

BATCH = 1
DAY0 = date(2026, 11, 24)


@pytest.fixture
def mgr(settings: Settings, sim_clock: SimulatedClock) -> CapitalManager:
    return CapitalManager(settings, sim_clock)


@pytest.fixture
def acct_session(session: Session, mgr: CapitalManager) -> Session:
    mgr.initialize_batch(session, BATCH, 1000.0)
    return session


def _rows(session: Session, entry_type: L | None = None) -> list[CapitalLedgerEntry]:
    stmt = select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == BATCH).order_by(CapitalLedgerEntry.id)
    rows = list(session.execute(stmt).scalars())
    return [r for r in rows if entry_type is None or r.entry_type == entry_type]


def _trade(mgr: CapitalManager, session: Session, pnl: float, margin: float = 500.0) -> None:
    mgr.block(session, BATCH, margin, position_id=1, mode=Mode.INTRADAY)
    mgr.release(session, BATCH, margin, pnl, position_id=1, trade_id=1, mode=Mode.INTRADAY)


def _balanced(mgr: CapitalManager, session: Session) -> None:
    ok, details = mgr.check_ledger_balance(session, BATCH)
    assert ok, details["problems"]


def _days(n: int) -> list[date]:
    out, d = [], DAY0
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ----------------------------------------------------------------------------- init / basics


def test_initialize_batch(acct_session: Session, mgr: CapitalManager) -> None:
    a = mgr.account(acct_session, BATCH)
    assert (a.starting_capital, a.base_capital, a.trading_cash, a.blocked, a.reserve) == (1000, 1000, 1000, 0, 0)
    assert a.hwm == 1000 and a.sizing_capital == 1000 and a.tier == "NORMAL"
    rows = _rows(acct_session)
    assert len(rows) == 1 and rows[0].entry_type == L.INITIAL_CAPITAL and rows[0].amount == 1000
    assert rows[0].trading_cash_after == 1000 and rows[0].to_account == "TRADING"
    with pytest.raises(CapitalError):
        mgr.initialize_batch(acct_session, BATCH, 1000.0)
    with pytest.raises(CapitalError):
        mgr.initialize_batch(acct_session, 2, 0.0)
    with pytest.raises(AccountNotFoundError):
        mgr.account(acct_session, 99)
    _balanced(mgr, acct_session)


def test_block_and_release_running_balances(acct_session: Session, mgr: CapitalManager) -> None:
    mgr.block(acct_session, BATCH, 600.0, position_id=7, mode=Mode.SWING)
    a = mgr.account(acct_session, BATCH)
    assert a.trading_cash == 400 and a.blocked == 600
    assert mgr.available_cash(acct_session, BATCH) == 400
    rows = mgr.release(acct_session, BATCH, 300.0, 12.5, position_id=7, trade_id=3, mode=Mode.SWING)
    assert [r.entry_type for r in rows] == [L.RELEASE, L.REALIZED_PNL]
    assert rows[0].trading_cash_after == 700 and rows[0].blocked_after == 300
    assert rows[1].trading_cash_after == 712.5 and rows[1].amount == 12.5 and rows[1].trade_id == 3
    assert a.cum_realized_net == 12.5
    mgr.release(acct_session, BATCH, 300.0, -40.0, position_id=7, trade_id=4, mode=Mode.SWING)
    assert a.blocked == 0 and a.trading_cash == pytest.approx(972.5)
    _balanced(mgr, acct_session)


def test_release_validation(acct_session: Session, mgr: CapitalManager) -> None:
    mgr.block(acct_session, BATCH, 100.0)
    with pytest.raises(LedgerImbalanceError):
        mgr.release(acct_session, BATCH, 150.0, 0.0)
    with pytest.raises(CapitalError):
        mgr.release(acct_session, BATCH, -1.0, 0.0)
    # sub-paisa residue from proportional releases is clamped to what is blocked
    mgr.release(acct_session, BATCH, 100.004, 0.0)
    assert mgr.account(acct_session, BATCH).blocked == 0.0
    _balanced(mgr, acct_session)


def test_reserve_is_never_available_for_blocking(acct_session: Session, mgr: CapitalManager) -> None:
    _trade(mgr, acct_session, 200.0)
    mgr.run_eod(acct_session, BATCH, DAY0, 200.0, 1200.0, DrawdownTier.NORMAL)
    a = mgr.account(acct_session, BATCH)
    assert a.reserve == pytest.approx(60.0) and a.trading_cash == pytest.approx(1140.0)
    assert mgr.available_cash(acct_session, BATCH) == pytest.approx(1140.0)  # excludes RESERVE
    with pytest.raises(InsufficientCashError):
        mgr.block(acct_session, BATCH, 1150.0)  # cash + reserve would cover it - still refused
    with pytest.raises(ReserveAccessError):
        mgr.block(acct_session, BATCH, 1140.01)
    with pytest.raises(CapitalError):
        mgr.block(acct_session, BATCH, 0.0)
    mgr.block(acct_session, BATCH, 1140.0)  # exactly all cash is fine
    assert mgr.available_cash(acct_session, BATCH) == 0.0
    assert a.reserve == pytest.approx(60.0)
    _balanced(mgr, acct_session)


# ----------------------------------------------------------------------------- EOD 6.2


def test_profit_lock_math(acct_session: Session, mgr: CapitalManager) -> None:
    _trade(mgr, acct_session, 100.0)
    res = mgr.run_eod(acct_session, BATCH, DAY0, 100.0, 1100.0, DrawdownTier.NORMAL)
    a = mgr.account(acct_session, BATCH)
    assert res.profit_locked == pytest.approx(30.0)
    assert a.reserve == pytest.approx(30.0) and a.reserve_peak == pytest.approx(30.0)
    assert mgr.trading_balance(a) == pytest.approx(1070.0)  # 70 % compounds
    assert a.hwm == pytest.approx(1100.0)
    lock = _rows(acct_session, L.PROFIT_LOCK)
    assert len(lock) == 1 and lock[0].from_account == "TRADING" and lock[0].to_account == "RESERVE"
    assert lock[0].reserve_after == pytest.approx(30.0)
    _balanced(mgr, acct_session)


def test_profit_lock_capped_by_free_cash(acct_session: Session, mgr: CapitalManager) -> None:
    _trade(mgr, acct_session, 100.0)
    mgr.block(acct_session, BATCH, 1090.0)  # only 10 free cash left
    res = mgr.run_eod(acct_session, BATCH, DAY0, 100.0, 1100.0, DrawdownTier.NORMAL)
    assert res.profit_locked == pytest.approx(10.0)
    assert res.profit_lock_shortfall == pytest.approx(20.0)
    assert mgr.account(acct_session, BATCH).trading_cash == pytest.approx(0.0)
    _balanced(mgr, acct_session)


def test_loss_absorb_without_reserve(acct_session: Session, mgr: CapitalManager) -> None:
    _trade(mgr, acct_session, -50.0)
    res = mgr.run_eod(acct_session, BATCH, DAY0, -50.0, 950.0, DrawdownTier.CAUTION)
    a = mgr.account(acct_session, BATCH)
    assert res.loss_absorbed == pytest.approx(50.0)
    assert res.refilled == 0.0 and res.refill_needed == pytest.approx(50.0)
    assert mgr.trading_balance(a) == pytest.approx(950.0) and a.reserve == 0.0
    marker = _rows(acct_session, L.LOSS_ABSORB)
    assert len(marker) == 1 and marker[0].amount == 0.0 and "-50.00" in (marker[0].note or "")
    assert not _rows(acct_session, L.REFILL)
    _balanced(mgr, acct_session)


def test_capped_refill_protects_reserve(acct_session: Session, mgr: CapitalManager) -> None:
    d1, d2, d3 = _days(3)
    _trade(mgr, acct_session, 200.0)
    mgr.run_eod(acct_session, BATCH, d1, 200.0, 1200.0, DrawdownTier.NORMAL)
    a = mgr.account(acct_session, BATCH)
    assert a.reserve == pytest.approx(60.0) and a.reserve_peak == pytest.approx(60.0)
    # big loss: TRADING 1140 - 300 = 840 < BASE 1000, need 160, but RESERVE may only go to 50 % of 60 = 30
    _trade(mgr, acct_session, -300.0)
    res = mgr.run_eod(acct_session, BATCH, d2, -300.0, 840.0 + 60.0, None)
    assert res.refill_needed == pytest.approx(160.0)
    assert res.refilled == pytest.approx(30.0)
    assert a.reserve == pytest.approx(30.0) and a.reserve_peak == pytest.approx(60.0)
    assert mgr.trading_balance(a) == pytest.approx(870.0)
    # another loss: RESERVE already at its protected level -> no refill
    _trade(mgr, acct_session, -10.0)
    res = mgr.run_eod(acct_session, BATCH, d3, -10.0, 890.0, None)
    assert res.refilled == 0.0
    assert a.reserve == pytest.approx(30.0)
    assert a.reserve >= a.reserve_peak * 0.5 - 1e-9
    _balanced(mgr, acct_session)


def test_refill_never_exceeds_base(acct_session: Session, mgr: CapitalManager) -> None:
    d1, d2 = _days(2)
    _trade(mgr, acct_session, 200.0)
    mgr.run_eod(acct_session, BATCH, d1, 200.0, 1200.0, DrawdownTier.NORMAL)
    _trade(mgr, acct_session, -145.0)  # TRADING 1140 -> 995
    res = mgr.run_eod(acct_session, BATCH, d2, -145.0, 1055.0, None)
    a = mgr.account(acct_session, BATCH)
    assert res.refilled == pytest.approx(5.0)
    assert mgr.trading_balance(a) == pytest.approx(1000.0)
    assert a.reserve == pytest.approx(55.0)
    _balanced(mgr, acct_session)


def test_no_refill_when_trading_still_above_base(acct_session: Session, mgr: CapitalManager) -> None:
    d1, d2 = _days(2)
    _trade(mgr, acct_session, 200.0)
    mgr.run_eod(acct_session, BATCH, d1, 200.0, 1200.0, DrawdownTier.NORMAL)
    _trade(mgr, acct_session, -40.0)
    res = mgr.run_eod(acct_session, BATCH, d2, -40.0, 1160.0, None)
    assert res.refilled == 0.0 and res.refill_needed == 0.0
    assert mgr.account(acct_session, BATCH).reserve == pytest.approx(60.0)


def test_eod_idempotent_per_day(acct_session: Session, mgr: CapitalManager) -> None:
    _trade(mgr, acct_session, 100.0)
    first = mgr.run_eod(acct_session, BATCH, DAY0, 100.0, 1100.0, DrawdownTier.NORMAL)
    n_rows = len(_rows(acct_session))
    again = mgr.run_eod(acct_session, BATCH, DAY0, 100.0, 1100.0, DrawdownTier.NORMAL)
    assert not first.skipped and again.skipped
    assert len(_rows(acct_session)) == n_rows
    assert mgr.account(acct_session, BATCH).reserve == pytest.approx(30.0)
    earlier = mgr.run_eod(acct_session, BATCH, DAY0 - timedelta(days=1), 50.0, 1100.0, DrawdownTier.NORMAL)
    assert earlier.skipped


def test_eod_realized_from_ledger_when_not_given(acct_session: Session, mgr: CapitalManager,
                                                  sim_clock: SimulatedClock) -> None:
    _trade(mgr, acct_session, 80.0)
    mgr.credit_dividend(acct_session, BATCH, 20.0, symbol="ITC")
    assert mgr.realized_net_for_day(acct_session, BATCH, DAY0) == pytest.approx(100.0)
    assert mgr.realized_net_for_day(acct_session, BATCH, DAY0 + timedelta(days=1)) == 0.0
    res = mgr.run_eod(acct_session, BATCH, DAY0)
    assert res.realized_net == pytest.approx(100.0) and res.profit_locked == pytest.approx(30.0)
    assert res.equity == pytest.approx(1100.0)
    _balanced(mgr, acct_session)


def test_daily_summary_total_upsert(acct_session: Session, mgr: CapitalManager) -> None:
    acct_session.add(DailySummary(batch_id=BATCH, day=DAY0, mode="TOTAL", trades=4, wins=3, net_pnl=100.0))
    acct_session.flush()
    _trade(mgr, acct_session, 100.0)
    mgr.run_eod(acct_session, BATCH, DAY0, 100.0, 1100.0, DrawdownTier.NORMAL)
    rows = list(acct_session.execute(select(DailySummary).where(DailySummary.batch_id == BATCH)).scalars())
    assert len(rows) == 1
    r = rows[0]
    assert r.trades == 4 and r.wins == 3 and r.net_pnl == 100.0  # other owners' fields untouched
    assert r.trading_balance == pytest.approx(1070.0) and r.trading_cash == pytest.approx(1070.0)
    assert r.reserve == pytest.approx(30.0) and r.equity == pytest.approx(1100.0)
    assert r.hwm == pytest.approx(1100.0) and r.drawdown_pct == 0.0 and r.tier == "NORMAL"
    assert r.sizing_capital == pytest.approx(1000.0) and r.profit_locked == pytest.approx(30.0)
    assert r.refilled == 0.0


# ----------------------------------------------------------------------------- sizing 6.3


def test_sizing_steps_up_after_exactly_three_days(acct_session: Session, mgr: CapitalManager) -> None:
    d1, d2, d3 = _days(3)
    a = mgr.account(acct_session, BATCH)
    _trade(mgr, acct_session, 200.0)  # TRADING after lock = 1140 >= 1000 * 1.1
    r1 = mgr.run_eod(acct_session, BATCH, d1, 200.0, 1200.0, DrawdownTier.NORMAL)
    assert r1.sizing_after == 1000 and a.sizing_up_streak == 1
    r2 = mgr.run_eod(acct_session, BATCH, d2, 0.0, 1200.0, DrawdownTier.NORMAL)
    assert r2.sizing_after == 1000 and a.sizing_up_streak == 2
    r3 = mgr.run_eod(acct_session, BATCH, d3, 0.0, 1200.0, DrawdownTier.NORMAL)
    assert r3.sizing_before == 1000 and r3.sizing_after == pytest.approx(1140.0)  # TO_TRADING
    assert a.sizing_up_streak == 0
    changes = _rows(acct_session, L.SIZING_CHANGE)
    assert len(changes) == 1 and changes[0].amount == pytest.approx(140.0)
    assert changes[0].sizing_capital_after == pytest.approx(1140.0)
    _balanced(mgr, acct_session)


def test_sizing_step_pct_mode(acct_session: Session, settings: Settings, sim_clock: SimulatedClock) -> None:
    s = settings.model_copy(deep=True)
    s.capital.sizing_step_mode = "STEP_PCT"
    s.capital.sizing_step_pct = 5.0
    mgr = CapitalManager(s, sim_clock)
    _trade(mgr, acct_session, 300.0)  # TRADING 1210 after lock
    for d in _days(3):
        res = mgr.run_eod(acct_session, BATCH, d, 300.0 if d == DAY0 else 0.0, 1300.0, DrawdownTier.NORMAL)
    assert res.sizing_after == pytest.approx(1050.0)  # +5 %, below TRADING
    # STEP_PCT never exceeds TRADING
    s.capital.sizing_step_pct = 50.0
    mgr2 = CapitalManager(s, sim_clock)
    for d in _days(6)[3:]:
        res = mgr2.run_eod(acct_session, BATCH, d, 0.0, 1300.0, DrawdownTier.NORMAL)
    assert res.sizing_after == pytest.approx(1210.0)


def test_sizing_streak_resets_on_drawdown_tier(acct_session: Session, mgr: CapitalManager) -> None:
    d = _days(5)
    a = mgr.account(acct_session, BATCH)
    _trade(mgr, acct_session, 200.0)
    mgr.run_eod(acct_session, BATCH, d[0], 200.0, 1200.0, DrawdownTier.NORMAL)
    mgr.run_eod(acct_session, BATCH, d[1], 0.0, 1200.0, DrawdownTier.NORMAL)
    assert a.sizing_up_streak == 2
    mgr.run_eod(acct_session, BATCH, d[2], 0.0, 1160.0, DrawdownTier.CAUTION)
    assert a.sizing_up_streak == 0 and a.sizing_capital == 1000
    mgr.run_eod(acct_session, BATCH, d[3], 0.0, 1200.0, DrawdownTier.NORMAL)
    mgr.run_eod(acct_session, BATCH, d[4], 0.0, 1200.0, DrawdownTier.NORMAL)
    assert a.sizing_up_streak == 2 and a.sizing_capital == 1000


def test_sizing_streak_resets_when_below_threshold(acct_session: Session, mgr: CapitalManager) -> None:
    d = _days(3)
    a = mgr.account(acct_session, BATCH)
    _trade(mgr, acct_session, 200.0)
    mgr.run_eod(acct_session, BATCH, d[0], 200.0, 1200.0, DrawdownTier.NORMAL)
    _trade(mgr, acct_session, -50.0)  # 1090 < 1100 threshold (still >= sizing)
    mgr.run_eod(acct_session, BATCH, d[1], -50.0, 1150.0, DrawdownTier.NORMAL)
    assert a.sizing_up_streak == 0 and a.sizing_capital == 1000


def test_sizing_decreases_immediately(acct_session: Session, mgr: CapitalManager) -> None:
    a = mgr.account(acct_session, BATCH)
    _trade(mgr, acct_session, -30.0)  # no EOD: release itself shrinks sizing
    assert a.sizing_capital == pytest.approx(970.0)
    rows = _rows(acct_session, L.SIZING_CHANGE)
    assert len(rows) == 1 and rows[0].amount == pytest.approx(-30.0)
    # a later profit does NOT raise it back immediately
    _trade(mgr, acct_session, 50.0)
    assert a.sizing_capital == pytest.approx(970.0)
    _balanced(mgr, acct_session)


def test_mode_sizing_capital(acct_session: Session, mgr: CapitalManager, settings: Settings) -> None:
    a = mgr.account(acct_session, BATCH)
    ms = mgr.mode_sizing_capital(a)
    assert ms[Mode.INTRADAY] == pytest.approx(500.0) and ms[Mode.SWING] == pytest.approx(500.0)
    assert ms[Mode.BTST] == 0.0 and ms[Mode.POSITIONAL] == 0.0
    s = settings.model_copy(deep=True)
    s.modes.modes[Mode.INTRADAY].allocation_pct = 30
    assert mgr.mode_sizing_capital(a, s)[Mode.INTRADAY] == pytest.approx(300.0)


# ----------------------------------------------------------------------------- owner actions


def test_unconfirmed_reserve_transfer_rejected(acct_session: Session, mgr: CapitalManager) -> None:
    _trade(mgr, acct_session, 200.0)
    mgr.run_eod(acct_session, BATCH, DAY0, 200.0, 1200.0, DrawdownTier.NORMAL)
    with pytest.raises(UnconfirmedTransferError):
        mgr.manual_reserve_transfer(acct_session, BATCH, 10.0, "owner", confirmed=False)
    a = mgr.account(acct_session, BATCH)
    assert a.reserve == pytest.approx(60.0)
    assert not _rows(acct_session, L.MANUAL_TRANSFER)
    actions = [r.action for r in acct_session.execute(select(AuditLog)).scalars()]
    assert "RESERVE_TRANSFER_REJECTED" in actions


def test_confirmed_reserve_transfer(acct_session: Session, mgr: CapitalManager) -> None:
    _trade(mgr, acct_session, 200.0)
    mgr.run_eod(acct_session, BATCH, DAY0, 200.0, 1200.0, DrawdownTier.NORMAL)
    a = mgr.account(acct_session, BATCH)
    total_before = a.trading_cash + a.blocked + a.reserve
    row = mgr.manual_reserve_transfer(acct_session, BATCH, 50.0, "owner", confirmed=True, note="topup")
    assert row.entry_type == L.MANUAL_TRANSFER and row.actor == "owner" and row.confirmed
    assert a.reserve == pytest.approx(10.0) and a.trading_cash == pytest.approx(1190.0)
    assert a.trading_cash + a.blocked + a.reserve == pytest.approx(total_before)  # sum unchanged
    assert a.reserve_peak == pytest.approx(60.0)
    audits = list(acct_session.execute(select(AuditLog).where(AuditLog.action == "RESERVE_TRANSFER")).scalars())
    assert len(audits) == 1 and audits[0].actor == "owner" and audits[0].batch_id == BATCH
    with pytest.raises(ReserveAccessError):
        mgr.manual_reserve_transfer(acct_session, BATCH, 10.01, "owner", confirmed=True)
    with pytest.raises(CapitalError):
        mgr.manual_reserve_transfer(acct_session, BATCH, 0.0, "owner", confirmed=True)
    _balanced(mgr, acct_session)


def test_manual_adjustment(acct_session: Session, mgr: CapitalManager) -> None:
    with pytest.raises(UnconfirmedTransferError):
        mgr.manual_adjustment(acct_session, BATCH, 25.0, "owner", "fix")
    mgr.manual_adjustment(acct_session, BATCH, 25.0, "owner", "broker credit", confirmed=True)
    a = mgr.account(acct_session, BATCH)
    assert a.cum_manual_adjustments == 25.0 and a.trading_cash == 1025.0
    mgr.manual_adjustment(acct_session, BATCH, -100.0, "owner", "correction", confirmed=True)
    assert a.cum_manual_adjustments == -75.0 and a.sizing_capital == pytest.approx(925.0)
    with pytest.raises(CapitalError):
        mgr.manual_adjustment(acct_session, BATCH, -5000.0, "owner", "too much", confirmed=True)
    with pytest.raises(CapitalError):
        mgr.manual_adjustment(acct_session, BATCH, 0.0, "owner", "nothing", confirmed=True)
    _balanced(mgr, acct_session)


def test_dividend_counts_as_realized(acct_session: Session, mgr: CapitalManager) -> None:
    mgr.credit_dividend(acct_session, BATCH, 4.5, position_id=3, symbol="ITC")
    a = mgr.account(acct_session, BATCH)
    assert a.cum_realized_net == 4.5 and a.trading_cash == 1004.5
    assert _rows(acct_session, L.DIVIDEND)[0].note == "dividend ITC"
    with pytest.raises(CapitalError):
        mgr.credit_dividend(acct_session, BATCH, -1.0)
    _balanced(mgr, acct_session)


def test_halt_and_resume_are_audited(acct_session: Session, mgr: CapitalManager) -> None:
    mgr.set_halted(acct_session, BATCH, "floor hit")
    a = mgr.account(acct_session, BATCH)
    assert a.halted and a.halt_reason == "floor hit" and a.tier == "HALT"
    mgr.clear_halt(acct_session, BATCH, "owner")
    assert not a.halted and a.halt_reason is None
    actions = [r.action for r in acct_session.execute(select(AuditLog)).scalars()]
    assert actions == ["CAPITAL_HALT", "CAPITAL_RESUME_HALT"]


# ----------------------------------------------------------------------------- snapshot


def _view(pid: int, mode: Mode, qty: int, entry: float, sl: float, ltp: float | None, charges: float,
          margin: float) -> OpenPositionView:
    return OpenPositionView(
        position_id=pid, symbol=f"S{pid}", mode=mode,
        product=Product.MIS if mode == Mode.INTRADAY else Product.CNC, direction=Direction.LONG,
        qty=qty, avg_entry=entry, stop_loss=sl, ltp=ltp, margin_blocked=margin, entry_charges=charges,
        opened_at=datetime(2026, 11, 24, 10, 0, tzinfo=IST),
    )


def test_snapshot(acct_session: Session, mgr: CapitalManager, settings: Settings) -> None:
    mgr.block(acct_session, BATCH, 400.0, position_id=1, mode=Mode.INTRADAY)
    mgr.block(acct_session, BATCH, 300.0, position_id=2, mode=Mode.SWING)
    positions = [
        _view(1, Mode.INTRADAY, 4, 100.0, 98.0, 105.0, 1.0, 400.0),   # +20 gross, -1 charges
        _view(2, Mode.SWING, 3, 100.0, 95.0, None, 1.5, 300.0),       # no quote: -1.5 charges
    ]
    snap = mgr.snapshot(acct_session, BATCH, positions, realized_today=12.0)
    assert snap.trading_cash == 300 and snap.blocked == 700 and snap.reserve == 0
    assert snap.unrealized_pnl == pytest.approx(20.0 - 1.0 - 1.5)
    assert snap.total_equity == pytest.approx(1017.5)
    assert snap.hwm == pytest.approx(1017.5)
    assert mgr.account(acct_session, BATCH).hwm == pytest.approx(1017.5)  # persisted
    assert snap.absolute_floor == pytest.approx(910.0)
    assert snap.trailing_floor == pytest.approx(1017.5 * 0.91)
    assert snap.effective_floor == pytest.approx(1017.5 * 0.91)
    assert snap.tier == DrawdownTier.NORMAL and not snap.halted
    assert snap.mode_blocked == {"INTRADAY": 400.0, "SWING": 300.0}
    assert snap.mode_sizing_capital["INTRADAY"] == pytest.approx(500.0)
    assert snap.realized_today == 12.0
    expected_risk = mgr.protection.open_risk(positions)
    assert snap.open_risk == pytest.approx(expected_risk)
    # SWING gap risk max(15, 3*100*3% = 9) = 15; INTRADAY MTM risk 4*(105-98) = 28 (+ charges)
    assert expected_risk > 28 + 1 + 15 + 1.5
    assert snap.loss_budget == pytest.approx(max(0.0, 1017.5 - expected_risk - snap.effective_floor))
    assert snap.trading_balance == 1000


def test_snapshot_without_hwm_update_and_halt(acct_session: Session, mgr: CapitalManager) -> None:
    mgr.block(acct_session, BATCH, 1000.0, position_id=1, mode=Mode.INTRADAY)
    up = [_view(1, Mode.INTRADAY, 10, 100.0, 95.0, 101.0, 0.0, 1000.0)]
    snap = mgr.snapshot(acct_session, BATCH, up, update_hwm=False)
    assert snap.hwm == pytest.approx(1010.0)
    assert mgr.account(acct_session, BATCH).hwm == 1000.0
    crash = [_view(1, Mode.INTRADAY, 10, 100.0, 95.0, 90.0, 0.0, 1000.0)]
    snap = mgr.snapshot(acct_session, BATCH, crash)
    assert snap.total_equity == pytest.approx(900.0)
    assert snap.tier == DrawdownTier.HALT and snap.halted and snap.loss_budget == 0.0


def test_snapshot_applies_immediate_sizing_decrease(acct_session: Session, mgr: CapitalManager) -> None:
    a = mgr.account(acct_session, BATCH)
    a.sizing_capital = 1200.0  # e.g. after settings edits; TRADING is 1000
    snap = mgr.snapshot(acct_session, BATCH, [])
    assert snap.sizing_capital == pytest.approx(1000.0)


# ----------------------------------------------------------------------------- ledger checks


def test_check_ledger_detects_tampering(acct_session: Session, mgr: CapitalManager) -> None:
    _trade(mgr, acct_session, 25.0)
    _balanced(mgr, acct_session)
    a = mgr.account(acct_session, BATCH)
    a.trading_cash += 5.0
    ok, details = mgr.check_ledger_balance(acct_session, BATCH)
    assert not ok and any("invariant" in p for p in details["problems"])
    with pytest.raises(LedgerImbalanceError):
        mgr.assert_ledger_balanced(acct_session, BATCH)
    a.trading_cash -= 5.0
    _balanced(mgr, acct_session)
    row = _rows(acct_session, L.REALIZED_PNL)[0]
    row.amount = 30.0
    ok, details = mgr.check_ledger_balance(acct_session, BATCH)
    assert not ok and any("replayed" in p or "ledger" in p for p in details["problems"])


def test_capital_events_logged(acct_session: Session, mgr: CapitalManager) -> None:
    import core.logging as cl

    _trade(mgr, acct_session, 10.0)
    events = read_events(DAY0, "capital", cl._log_dir)
    types = [e.get("entry_type") for e in events if e.get("event") == "ledger"]
    assert types == ["INITIAL_CAPITAL", "BLOCK", "RELEASE", "REALIZED_PNL"]


# ----------------------------------------------------------------------------- randomized 6.4


_RESERVE_UP = {L.PROFIT_LOCK.value}
_RESERVE_DOWN = {L.REFILL.value, L.MANUAL_TRANSFER.value}


@pytest.mark.parametrize("seed", list(range(12)))
def test_randomized_sequences_keep_ledger_balanced(session: Session, settings: Settings, seed: int) -> None:
    rng = np.random.default_rng(seed)
    clock = SimulatedClock(datetime(2026, 11, 24, 9, 15, tzinfo=IST))
    mgr = CapitalManager(settings, clock)
    mgr.initialize_batch(session, BATCH, float(rng.choice([1000.0, 5000.0, 25000.0])))
    open_margins: dict[int, float] = {}
    next_id = 1
    day = DAY0
    for _ in range(150):
        a = mgr.account(session, BATCH)
        op = rng.choice(["block", "release", "release", "eod", "transfer", "dividend", "adjust", "bad"])
        if op == "block" and a.trading_cash > 1:
            amt = float(rng.uniform(0.01, 1.0) * a.trading_cash)
            mgr.block(session, BATCH, amt, position_id=next_id)
            open_margins[next_id] = amt
            next_id += 1
        elif op == "release" and open_margins:
            pid = int(rng.choice(list(open_margins)))
            frac = float(rng.choice([1.0, 0.5, 1 / 3]))
            margin = open_margins[pid] * frac
            pnl = float(rng.normal(0, 0.03) * margin)
            mgr.release(session, BATCH, margin, pnl, position_id=pid)
            open_margins[pid] -= margin
            if open_margins[pid] < 1e-6:
                del open_margins[pid]
        elif op == "eod":
            realized = mgr.realized_net_for_day(session, BATCH, day)
            tier = DrawdownTier(rng.choice(["NORMAL", "NORMAL", "CAUTION"]))
            mgr.run_eod(session, BATCH, day, realized, None, tier)
            day += timedelta(days=1)
            clock.set(datetime.combine(day, datetime.min.time(), tzinfo=IST).replace(hour=9, minute=15))
        elif op == "transfer" and a.reserve > 0.01:
            confirmed = bool(rng.integers(0, 2))
            amount = float(rng.uniform(0.0, 1.0) * a.reserve) + 0.001
            if confirmed:
                mgr.manual_reserve_transfer(session, BATCH, min(amount, a.reserve), "owner", True)
            else:
                with pytest.raises(UnconfirmedTransferError):
                    mgr.manual_reserve_transfer(session, BATCH, amount, "owner", False)
        elif op == "dividend":
            mgr.credit_dividend(session, BATCH, float(rng.uniform(0.1, 5.0)))
        elif op == "adjust":
            mgr.manual_adjustment(session, BATCH, float(rng.uniform(-1, 5)), "owner", "rand", confirmed=True)
        elif op == "bad":
            # trying to block more than cash must always fail and change nothing
            before = (a.trading_cash, a.blocked, a.reserve)
            with pytest.raises(ReserveAccessError):
                mgr.block(session, BATCH, a.trading_cash + float(rng.uniform(0.01, 10.0)))
            assert (a.trading_cash, a.blocked, a.reserve) == before
        a = mgr.account(session, BATCH)
        lhs = a.starting_capital + a.cum_realized_net + a.cum_manual_adjustments
        assert lhs == pytest.approx(a.trading_cash + a.blocked + a.reserve, abs=0.01)
        assert a.reserve >= -1e-9 and a.blocked >= -1e-6
    _balanced(mgr, session)

    # RESERVE only grows by profit locks and only shrinks by refill / confirmed manual transfer
    prev = 0.0
    for r in _rows(session):
        if r.reserve_after > prev + 1e-9:
            assert r.entry_type in _RESERVE_UP, r.entry_type
        elif r.reserve_after < prev - 1e-9:
            assert r.entry_type in _RESERVE_DOWN, r.entry_type
            if r.entry_type == L.MANUAL_TRANSFER:
                assert r.confirmed and r.actor == "owner"
        prev = r.reserve_after
    # every refill respected the protected share of the peak at that time
    peak = 0.0
    for r in _rows(session):
        if r.entry_type == L.PROFIT_LOCK:
            peak = max(peak, r.reserve_after)
        if r.entry_type == L.REFILL:
            assert r.reserve_after >= peak * settings.capital.reserve_protected_pct / 100.0 - 1e-6
