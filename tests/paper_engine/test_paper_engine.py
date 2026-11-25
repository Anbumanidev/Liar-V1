"""Paper engine: fills, exits, trade management, gaps, HALT, owner edits, corporate actions."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from capital.accounts import CapitalManager
from config.loader import update_settings
from config.schema import HardCaps, Settings
from core.clock import IST, SimulatedClock
from core.enums import (
    BatchStatus,
    Direction,
    EntryType,
    EventType,
    ExitReason,
    Grade,
    Mode,
    OrderStatus,
    PositionStatus,
    Timeframe,
    TrailingMethod,
)
from core.errors import OvernightShortError, StopLossLooseningError
from core.types import MarketEvent, RiskVerdict, TradeProposal
from data.calendar import load_calendar
from paper_engine.broker import OrderRejected, PaperBroker
from paper_engine.corporate_actions import apply_corporate_actions
from paper_engine.fills import Bar, limit_fill_price, round_tick, slipped_price, stop_fill, target_fill
from paper_engine.monitor import PositionMonitor
from storage.models import Batch, CapitalLedgerEntry, ModeState, Order, Position, Trade
from tests.paper_engine.paper_fakes import FakeData, bars

DAY = date(2026, 11, 25)    # Wednesday (24th is Guru Nanak Jayanti - NSE holiday)
NEXT = date(2026, 11, 26)


def at(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=IST)


@pytest.fixture
def env(settings: Settings, caps: HardCaps, session):  # type: ignore[no-untyped-def]
    s = update_settings(settings, {"engine.slippage_pct": 0.0})
    clock = SimulatedClock(at(DAY, 9, 59))
    cal = load_calendar()
    data = FakeData(clock)
    capital = CapitalManager(s, clock)
    batch = Batch(name="T", starting_capital=1000.0, status=BatchStatus.ACTIVE.value, created_at=clock.now())
    session.add(batch)
    session.flush()
    capital.initialize_batch(session, batch.id, 1000.0)
    broker = PaperBroker(s, caps, clock, capital, data, cal)
    monitor = PositionMonitor(s, clock, cal, data, broker, capital)
    return SimpleNamespace(s=s, clock=clock, cal=cal, data=data, capital=capital, broker=broker, monitor=monitor,
                           session=session, batch_id=batch.id)


def open_pos(env, *, symbol="INFY", mode=Mode.INTRADAY, direction=Direction.LONG, qty=4, entry=100.0,  # type: ignore[no-untyped-def]
             sl=99.0, t1=101.5, t2=103.0, entry_type=EntryType.MARKET, first_bar=(100.0, 100.2, 99.8, 100.1),
             trailing=TrailingMethod.NONE, max_days=None, day=DAY, hhmm=(10, 0)) -> Position:
    env.clock.set(at(day, hhmm[0], hhmm[1], 30))
    prop = TradeProposal(batch_id=env.batch_id, symbol=symbol, mode=mode, direction=direction, entry_price=entry,
                         entry_type=entry_type, stop_loss=sl, target_1=t1, target_2=t2, strategy="t", grade=Grade.A,
                         confidence=80, sector="IT", trailing_rule=trailing, max_holding_days=max_days,
                         created_at=env.clock.now())
    verdict = RiskVerdict(approved=True, qty=qty, max_qty=qty)
    order = env.broker.submit_entry(env.session, prop, verdict)
    bar_start = f"{hhmm[0]:02d}:{hhmm[1] + 1:02d}"
    env.data.add(symbol, Timeframe.M1, bars(day, bar_start, [first_bar]))
    env.clock.set(at(day, hhmm[0], hhmm[1] + 2))
    rep = env.monitor.check(env.session)
    assert order.status == OrderStatus.FILLED.value, (order.status, order.reason)
    assert rep.opened
    return env.session.get(Position, rep.opened[0])


def feed(env, symbol: str, day: date, start: str, rows, until: tuple[int, int]):  # type: ignore[no-untyped-def]
    env.data.add(symbol, Timeframe.M1, bars(day, start, rows))
    env.clock.set(at(day, *until))
    return env.monitor.check(env.session)


def ledger_ok(env) -> None:  # type: ignore[no-untyped-def]
    ok, details = env.capital.check_ledger_balance(env.session, env.batch_id)
    assert ok, details


# ---------------------------------------------------------------- fill primitives


def test_fill_primitives() -> None:
    assert slipped_price(__import__("core.enums").enums.OrderSide.BUY, 100.0, 0.05) == 100.05
    assert slipped_price(__import__("core.enums").enums.OrderSide.SELL, 100.0, 0.05) == 99.95
    assert round_tick(100.03) == 100.05 and round_tick(100.02) == 100.0
    from core.enums import OrderSide
    assert limit_fill_price(OrderSide.BUY, 99.5, Bar(100, 100.5, 99.5, 100)) is None      # touch is not enough
    assert limit_fill_price(OrderSide.BUY, 99.5, Bar(100, 100.5, 99.45, 100)) == 99.5
    assert limit_fill_price(OrderSide.BUY, 99.5, Bar(99.0, 99.6, 98.9, 99.2)) == 99.0     # gapped below: open
    assert stop_fill(Direction.LONG, 97.0, Bar(95, 96, 94, 95.5), 0.05) == (95.0, True)
    assert stop_fill(Direction.LONG, 97.0, Bar(98, 98.5, 96.9, 97.5), 0.0) == (97.0, False)
    assert stop_fill(Direction.SHORT, 103.0, Bar(104, 105, 103.5, 104), 0.0) == (104.0, True)
    assert target_fill(Direction.LONG, 103.0, Bar(102, 103.0, 101.5, 102.8)) == 103.0
    assert target_fill(Direction.SHORT, 97.0, Bar(96.5, 97.5, 96, 96.2)) == 96.5


# ---------------------------------------------------------------- entries


def test_market_entry_fills_next_bar_with_slippage(settings: Settings, caps: HardCaps, session) -> None:  # type: ignore[no-untyped-def]
    clock = SimulatedClock(at(DAY, 10, 0, 30))
    data = FakeData(clock)
    capital = CapitalManager(settings, clock)
    batch = Batch(name="S", starting_capital=1000.0, status="ACTIVE", created_at=clock.now())
    session.add(batch)
    session.flush()
    capital.initialize_batch(session, batch.id, 1000.0)
    broker = PaperBroker(settings, caps, clock, capital, data, load_calendar())
    prop = TradeProposal(batch_id=batch.id, symbol="INFY", mode=Mode.INTRADAY, direction=Direction.LONG,
                         entry_price=100, entry_type=EntryType.MARKET, stop_loss=99, target_1=101.5,
                         created_at=clock.now())
    order = broker.submit_entry(session, prop, RiskVerdict(approved=True, qty=3, max_qty=3))
    assert broker.process_pending_orders(session) == []           # no bar yet
    data.add("INFY", Timeframe.M1, bars(DAY, "10:01", [(100.0, 100.4, 99.9, 100.2)]))
    clock.set(at(DAY, 10, 2))
    opened = broker.process_pending_orders(session)
    assert len(opened) == 1 and order.status == OrderStatus.FILLED.value
    assert opened[0].avg_entry == 100.05                          # open + 0.05 % slippage, tick-rounded
    acct = capital.account(session, batch.id)
    assert acct.blocked == pytest.approx(3 * 100.05)
    assert opened[0].entry_charges > 0


def test_unapproved_verdict_cannot_be_submitted(env) -> None:  # type: ignore[no-untyped-def]
    prop = TradeProposal(batch_id=env.batch_id, symbol="X", mode=Mode.INTRADAY, direction=Direction.LONG,
                         entry_price=100, stop_loss=99, target_1=102, created_at=env.clock.now())
    with pytest.raises(OrderRejected):
        env.broker.submit_entry(env.session, prop, RiskVerdict(approved=False, qty=0))


def test_cnc_short_is_impossible(env) -> None:  # type: ignore[no-untyped-def]
    prop = TradeProposal(batch_id=env.batch_id, symbol="X", mode=Mode.SWING, direction=Direction.SHORT,
                         entry_price=100, stop_loss=101, target_1=97, created_at=env.clock.now())
    with pytest.raises(OvernightShortError):
        env.broker.submit_entry(env.session, prop, RiskVerdict(approved=True, qty=1, max_qty=1))


def test_limit_order_needs_trade_through_and_expires(env) -> None:  # type: ignore[no-untyped-def]
    env.clock.set(at(DAY, 10, 0, 30))
    prop = TradeProposal(batch_id=env.batch_id, symbol="TCS", mode=Mode.INTRADAY, direction=Direction.LONG,
                         entry_price=99.5, entry_type=EntryType.LIMIT, stop_loss=98.5, target_1=101.5,
                         created_at=env.clock.now())
    order = env.broker.submit_entry(env.session, prop, RiskVerdict(approved=True, qty=2, max_qty=2),
                                    time_validity_minutes=5)
    feed(env, "TCS", DAY, "10:01", [(100, 100.3, 99.5, 100.1), (100.1, 100.2, 99.6, 99.9)], (10, 3))
    assert order.status == OrderStatus.PENDING.value
    feed(env, "TCS", DAY, "10:03", [(99.9, 100.0, 99.6, 99.8)] * 3, (10, 7))
    assert order.status == OrderStatus.EXPIRED.value


def test_entry_through_stop_is_skipped(env) -> None:  # type: ignore[no-untyped-def]
    env.clock.set(at(DAY, 10, 0, 30))
    prop = TradeProposal(batch_id=env.batch_id, symbol="SBIN", mode=Mode.INTRADAY, direction=Direction.LONG,
                         entry_price=100, entry_type=EntryType.MARKET, stop_loss=99, target_1=101.5,
                         created_at=env.clock.now())
    order = env.broker.submit_entry(env.session, prop, RiskVerdict(approved=True, qty=1, max_qty=1))
    feed(env, "SBIN", DAY, "10:01", [(98.5, 98.8, 98.2, 98.6)], (10, 2))
    assert order.status == OrderStatus.REJECTED.value and "through the stop" in (order.reason or "")


def test_evening_swing_order_waits_for_next_session_window(env) -> None:  # type: ignore[no-untyped-def]
    env.clock.set(at(DAY, 16, 30))
    prop = TradeProposal(batch_id=env.batch_id, symbol="LT", mode=Mode.SWING, direction=Direction.LONG,
                         entry_price=100, entry_type=EntryType.LIMIT, stop_loss=96, target_1=108,
                         created_at=env.clock.now())
    order = env.broker.submit_entry(env.session, prop, RiskVerdict(approved=True, qty=2, max_qty=2))
    nb = order.meta_json["not_before"]
    assert nb.startswith("2026-11-26T09:20") and order.valid_until == at(NEXT, 10, 30)
    feed(env, "LT", NEXT, "09:15", [(100.5, 100.8, 99.8, 100.2)] * 4, (9, 19))   # before 09:20: ignored
    assert order.status == OrderStatus.PENDING.value
    feed(env, "LT", NEXT, "09:20", [(100.2, 100.4, 99.9, 100.0)], (9, 21))
    assert order.status == OrderStatus.FILLED.value and order.fill_price == 100.0


# ---------------------------------------------------------------- exits


def test_stop_loss_exit_books_loss_and_balances_ledger(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env)
    feed(env, "INFY", DAY, "10:02", [(100.0, 100.1, 98.9, 99.0)], (10, 3))
    assert pos.status == PositionStatus.CLOSED.value
    trade = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert trade.exit_reason == ExitReason.SL.value and trade.exit_price == 99.0
    assert trade.gross_pnl == pytest.approx(-4.0) and trade.net_pnl < trade.gross_pnl
    assert trade.r_multiple == pytest.approx(-1.0)
    acct = env.capital.account(env.session, env.batch_id)
    assert acct.blocked == pytest.approx(0.0)
    assert acct.trading_cash == pytest.approx(1000.0 + trade.net_pnl)
    ledger_ok(env)


def test_sl_and_target_in_same_bar_assumes_sl(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=1)
    feed(env, "INFY", DAY, "10:02", [(100.0, 103.5, 98.5, 101.0)], (10, 3))
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == ExitReason.SL.value


def test_partial_at_1r_breakeven_then_trail_exit(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=4)
    feed(env, "INFY", DAY, "10:02", [(100.2, 101.05, 100.1, 100.9)], (10, 3))
    assert pos.qty == 2 and pos.partial_done and pos.breakeven_moved and pos.stop_loss == 100.0
    part = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert part.is_partial and part.exit_reason == ExitReason.TARGET.value and part.exit_price == 101.0
    feed(env, "INFY", DAY, "10:03", [(100.5, 100.6, 99.9, 100.0)], (10, 4))
    assert pos.status == PositionStatus.CLOSED.value
    last = env.session.execute(select(Trade).where(Trade.position_id == pos.id).order_by(Trade.id.desc())).scalars().first()
    assert last.exit_reason == ExitReason.TRAIL.value and last.exit_price == 100.0
    ledger_ok(env)


def test_qty_one_only_moves_to_breakeven(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=1)
    feed(env, "INFY", DAY, "10:02", [(100.2, 101.1, 100.1, 100.9)], (10, 3))
    assert pos.qty == 1 and pos.status == PositionStatus.OPEN.value and pos.stop_loss == 100.0
    assert env.session.execute(select(Trade).where(Trade.position_id == pos.id)).first() is None


def test_final_target_t2(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=4)
    feed(env, "INFY", DAY, "10:02", [(100.2, 101.2, 100.1, 101.1), (101.1, 103.2, 101.0, 103.0)], (10, 4))
    assert pos.status == PositionStatus.CLOSED.value
    trades = env.session.execute(select(Trade).where(Trade.position_id == pos.id).order_by(Trade.id)).scalars().all()
    assert [t.exit_reason for t in trades] == ["TARGET", "TARGET"] and trades[-1].exit_price == 103.0
    ledger_ok(env)


def test_short_position_stop(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, direction=Direction.SHORT, qty=2, entry=100, sl=101, t1=98.5, t2=97,
                   first_bar=(100.0, 100.2, 99.8, 99.9))
    feed(env, "INFY", DAY, "10:02", [(100.0, 101.2, 99.9, 101.0)], (10, 3))
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == "SL" and t.exit_price == 101.0 and t.gross_pnl == pytest.approx(-2.0)


def test_intraday_squareoff(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=2, hhmm=(15, 10))
    feed(env, "INFY", DAY, "15:12", [(100.1, 100.3, 99.9, 100.2)] * 3, (15, 15))
    assert pos.status == PositionStatus.CLOSED.value
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == ExitReason.SQUAREOFF.value


def test_intraday_time_stop(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=2)
    rows = [(100.1, 100.3, 99.9, 100.2)] * 60
    feed(env, "INFY", DAY, "10:02", rows, (11, 2))
    assert pos.status == PositionStatus.CLOSED.value
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == ExitReason.TIME.value


def test_overnight_gap_down_fills_at_open(env) -> None:  # type: ignore[no-untyped-def]
    env.s.risk.max_charges_pct_of_value  # noqa: B018 - settings present
    pos = open_pos(env, symbol="LT", mode=Mode.SWING, qty=2, entry=100, sl=97, t1=106, t2=110, hhmm=(10, 0))
    feed(env, "LT", DAY, "10:02", [(100.1, 100.5, 99.5, 100.0)], (15, 31))
    feed(env, "LT", NEXT, "09:15", [(95.0, 95.5, 94.5, 95.2)], (9, 16))
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == ExitReason.GAP.value and t.exit_price == 95.0
    ledger_ok(env)


def test_swing_time_stop_on_max_holding_day(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, symbol="LT", mode=Mode.SWING, qty=1, entry=100, sl=96, t1=108, t2=112, max_days=1)
    feed(env, "LT", NEXT, "15:18", [(100.1, 100.5, 99.5, 100.0)] * 2, (15, 20))
    assert pos.holding_days == 1
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == ExitReason.TIME.value


def test_btst_exits_next_morning(env, caps: HardCaps) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, symbol="ITC", mode=Mode.BTST, qty=1, entry=100, sl=98, t1=104, t2=106, hhmm=(15, 0))
    feed(env, "ITC", NEXT, "10:28", [(100.5, 100.8, 100.2, 100.6)] * 2, (10, 30))
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == ExitReason.TIME.value and t.holding_days == 1


def test_pre_event_exit(env) -> None:  # type: ignore[no-untyped-def]
    ev = MarketEvent(symbol="LT", event_type=EventType.RESULTS, event_date=NEXT)
    env.monitor.events_fn = lambda sym, a, b: [ev] if sym == "LT" and a <= NEXT <= b else []
    pos = open_pos(env, symbol="LT", mode=Mode.SWING, qty=1, entry=100, sl=96, t1=108, t2=112)
    feed(env, "LT", DAY, "15:19", [(100.1, 100.5, 99.5, 100.0)], (15, 20))
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == ExitReason.EVENT.value


def test_event_pin_prevents_pre_event_exit(env) -> None:  # type: ignore[no-untyped-def]
    ev = MarketEvent(symbol="LT", event_type=EventType.RESULTS, event_date=NEXT)
    env.monitor.events_fn = lambda sym, a, b: [ev]
    pos = open_pos(env, symbol="LT", mode=Mode.SWING, qty=1, entry=100, sl=96, t1=108, t2=112)
    env.broker.set_event_pin(env.session, pos, True, actor="owner")
    feed(env, "LT", DAY, "15:19", [(100.1, 100.5, 99.5, 100.0)], (15, 20))
    assert pos.status == PositionStatus.OPEN.value


def test_consecutive_losses_pause_intraday(env) -> None:  # type: ignore[no-untyped-def]
    for i, sym in enumerate(["A1", "A2", "A3"]):
        pos = open_pos(env, symbol=sym, qty=1, hhmm=(10, 10 * i))
        feed(env, sym, DAY, f"10:{10 * i + 2:02d}", [(100.0, 100.1, 98.9, 99.0)], (10, 10 * i + 3))
        assert pos.status == PositionStatus.CLOSED.value
    ms = env.session.execute(select(ModeState).where(ModeState.mode == "INTRADAY")).scalar_one()
    assert ms.paused_until is not None and ms.paused_until > env.clock.now()


def test_trailing_atr_moves_stop_up_only(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=1, trailing=TrailingMethod.ATR, t1=110, t2=115)
    # 5m history for ATR: a steady up-trend, ATR ~0.4
    hist = [(98 + i * 0.05, 98.3 + i * 0.05, 97.9 + i * 0.05, 98.1 + i * 0.05) for i in range(40)]
    env.data.add("INFY", Timeframe.M5, bars(DAY, "09:15", hist[:9], freq_min=5))
    feed(env, "INFY", DAY, "10:02", [(100.5, 101.6, 100.4, 101.5)], (10, 3))
    assert pos.breakeven_moved
    sl_after = pos.stop_loss
    assert sl_after >= 100.0
    feed(env, "INFY", DAY, "10:03", [(101.5, 101.6, 101.0, 101.2)], (10, 4))
    assert pos.stop_loss >= sl_after          # never loosens


# ---------------------------------------------------------------- owner edits


def test_owner_cannot_loosen_stop(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=1)
    with pytest.raises(StopLossLooseningError):
        env.broker.tighten_sl(env.session, pos, 98.5, actor="owner", ltp=100.1)
    env.broker.tighten_sl(env.session, pos, 99.5, actor="owner", ltp=100.1)
    assert pos.stop_loss == 99.5 and pos.sl_history_json[-1]["by"] == "OWNER"


def test_owner_targets_trailing_holding(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=1)
    with pytest.raises(OrderRejected):
        env.broker.update_targets(env.session, pos, 99.0, None, actor="owner")
    env.broker.update_targets(env.session, pos, 102.0, 104.0, actor="owner")
    assert (pos.target_1, pos.target_2) == (102.0, 104.0)
    with pytest.raises(OrderRejected):
        env.broker.update_trailing(env.session, pos, TrailingMethod.EMA20_DAILY, None, actor="owner")
    env.broker.update_trailing(env.session, pos, TrailingMethod.EMA9_5M, None, actor="owner")
    with pytest.raises(OrderRejected):
        env.broker.update_holding(env.session, pos, 5, actor="owner")    # MIS has no holding period


def test_partial_manual_close(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=4)
    t = env.broker.exit_position(env.session, pos, 1, ExitReason.MANUAL, price=100.5, actor="owner")
    assert t.is_partial and pos.qty == 3
    acct = env.capital.account(env.session, env.batch_id)
    assert acct.blocked == pytest.approx(3 * pos.avg_entry, rel=1e-6)
    ledger_ok(env)


def test_convert_intraday_to_swing(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, qty=2)
    mis_charges = pos.entry_charges
    env.broker.convert_position(env.session, pos, Mode.SWING, RiskVerdict(approved=True, qty=2, max_qty=2),
                                actor="owner")
    assert pos.mode == "SWING" and pos.product == "CNC" and pos.overnight and pos.converted_from_mode == "INTRADAY"
    from core.enums import OrderSide, Product
    from paper_engine.charges import order_charges
    expected = order_charges(Product.CNC, OrderSide.BUY, 2, pos.avg_entry, env.s.charges).total
    assert pos.entry_charges == pytest.approx(expected) and pos.entry_charges != mis_charges   # CNC model now
    with pytest.raises(OrderRejected):
        env.broker.convert_position(env.session, pos, Mode.SWING, RiskVerdict(approved=False, qty=0), actor="owner")
    ledger_ok(env)


def test_dp_charged_once_per_scrip_per_day(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, symbol="LT", mode=Mode.SWING, qty=4, entry=100, sl=96, t1=108, t2=112)
    t1 = env.broker.exit_position(env.session, pos, 2, ExitReason.MANUAL, price=101.0, actor="owner")
    t2 = env.broker.exit_position(env.session, pos, 2, ExitReason.MANUAL, price=101.0, actor="owner")
    assert t1.charges_json["dp"] > 0 and t2.charges_json["dp"] == 0
    ledger_ok(env)


# ---------------------------------------------------------------- HALT / pending exits


def test_halt_exits_everything_and_halts_batch(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, symbol="LT", mode=Mode.SWING, qty=5, entry=100, sl=50, t1=200, t2=250)
    # a huge adverse move (beyond any sane stop) drags equity under the Rs 910 floor
    feed(env, "LT", DAY, "10:02", [(100, 100, 80.5, 81.0)], (10, 3))
    batch = env.session.get(Batch, env.batch_id)
    assert batch.status == BatchStatus.HALTED.value
    assert pos.status == PositionStatus.CLOSED.value
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == ExitReason.HALT.value
    assert env.capital.account(env.session, env.batch_id).halted


def test_pending_exit_when_market_closed(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, symbol="LT", mode=Mode.SWING, qty=1, entry=100, sl=96, t1=108, t2=112)
    env.clock.set(at(DAY, 18, 0))
    trades = env.broker.close_positions(env.session, reason=ExitReason.KILL_SWITCH, market_open=False)
    assert trades == [] and pos.notes_json["pending_exit"]["reason"] == "KILL_SWITCH"
    feed(env, "LT", NEXT, "09:15", [(101.0, 101.2, 100.8, 101.1)], (9, 16))
    t = env.session.execute(select(Trade).where(Trade.position_id == pos.id)).scalar_one()
    assert t.exit_reason == ExitReason.KILL_SWITCH.value and t.exit_price == 101.0


# ---------------------------------------------------------------- corporate actions


def test_split_and_dividend(env) -> None:  # type: ignore[no-untyped-def]
    pos = open_pos(env, symbol="LT", mode=Mode.SWING, qty=3, entry=100, sl=96, t1=108, t2=112)
    margin = pos.margin_blocked
    split = MarketEvent(symbol="LT", event_type=EventType.SPLIT, event_date=NEXT, ratio=2.0)
    rows = apply_corporate_actions(env.session, [split], NEXT, env.capital, env.clock)
    assert len(rows) == 1 and pos.qty == 6 and pos.avg_entry == pytest.approx(50.0) and pos.stop_loss == 48.0
    assert pos.margin_blocked == margin
    assert apply_corporate_actions(env.session, [split], NEXT, env.capital, env.clock) == []   # idempotent
    div = MarketEvent(symbol="LT", event_type=EventType.EX_DIVIDEND, event_date=NEXT, amount=1.5)
    rows = apply_corporate_actions(env.session, [div], NEXT, env.capital, env.clock)
    assert rows[0].cash_credit == pytest.approx(9.0)
    kinds = [r.entry_type for r in env.session.execute(select(CapitalLedgerEntry)).scalars()]
    assert "DIVIDEND" in kinds
    ledger_ok(env)


def test_entry_skipped_when_fill_ruins_the_reward_risk(env) -> None:  # type: ignore[no-untyped-def]
    """ITC 2026-11-25: planned R:R 1.52, filled 0.60 higher -> 0.59. The fill check now skips it."""
    env.clock.set(at(DAY, 10, 0, 30))
    prop = TradeProposal(batch_id=env.batch_id, symbol="ITC", mode=Mode.INTRADAY, direction=Direction.LONG,
                         entry_price=310.5, entry_type=EntryType.MARKET, stop_loss=309.45, target_1=312.08,
                         atr=0.84, created_at=env.clock.now())
    order = env.broker.submit_entry(env.session, prop, RiskVerdict(approved=True, qty=1, max_qty=1))
    feed(env, "ITC", DAY, "10:01", [(311.1, 311.5, 310.9, 311.3)], (10, 2))
    assert order.status == OrderStatus.REJECTED.value
    assert "R:R to target 1 at the fill is 0.59" in (order.reason or "")


def test_entry_skipped_when_fill_lands_on_top_of_the_stop(env) -> None:  # type: ignore[no-untyped-def]
    """BEL 2026-11-25: filled 0.50 (0.59 ATR) above its stop and was stopped in 3 minutes."""
    env.clock.set(at(DAY, 10, 0, 30))
    prop = TradeProposal(batch_id=env.batch_id, symbol="BEL", mode=Mode.INTRADAY, direction=Direction.LONG,
                         entry_price=404.3, entry_type=EntryType.MARKET, stop_loss=401.95, target_1=409.9,
                         atr=0.8462, created_at=env.clock.now())
    order = env.broker.submit_entry(env.session, prop, RiskVerdict(approved=True, qty=1, max_qty=1))
    feed(env, "BEL", DAY, "10:01", [(402.45, 402.6, 402.2, 402.4)], (10, 2))
    assert order.status == OrderStatus.REJECTED.value and "noise would stop it out" in (order.reason or "")


def test_manual_fill_is_not_second_guessed(env) -> None:  # type: ignore[no-untyped-def]
    from core.enums import TradeSource

    env.clock.set(at(DAY, 10, 0, 30))
    prop = TradeProposal(batch_id=env.batch_id, symbol="ITC", mode=Mode.INTRADAY, direction=Direction.LONG,
                         entry_price=310.5, entry_type=EntryType.MARKET, stop_loss=309.45, target_1=312.08,
                         atr=0.84, source=TradeSource.MANUAL, created_at=env.clock.now())
    order = env.broker.submit_entry(env.session, prop, RiskVerdict(approved=True, qty=1, max_qty=1))
    feed(env, "ITC", DAY, "10:01", [(311.1, 311.5, 310.9, 311.3)], (10, 2))
    assert order.status == OrderStatus.FILLED.value
