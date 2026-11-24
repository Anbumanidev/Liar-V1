"""build_eod_report on an in-memory DB + equity_chart_png."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest
from sqlalchemy.orm import Session

from config.schema import Settings
from core.clock import IST, SimulatedClock
from notifications.reports import build_eod_report, equity_chart_png, unrealized_pnl
from storage.models import (
    AgentReportRow,
    ApprovalItem,
    Batch,
    CapitalAccount,
    CapitalLedgerEntry,
    DailySummary,
    DecisionRow,
    Position,
    ShadowTrade,
    StockAnalyzed,
    Trade,
)

DAY = date(2026, 11, 24)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def ts(d: int, hh: int, mm: int = 0, month: int = 11) -> datetime:
    return datetime(2026, month, d, hh, mm, tzinfo=IST)


# --------------------------------------------------------------------------- row factories


def add_trade(s: Session, *, id: int, position_id: int, symbol: str, mode: str, gross: float, charges: float,
              exit_time: datetime, batch_id: int = 1, partial: bool = False, product: str = "MIS",
              reason: str = "TARGET") -> None:
    s.add(Trade(id=id, batch_id=batch_id, position_id=position_id, symbol=symbol, mode=mode, product=product,
                direction="LONG", qty=1, entry_time=exit_time.replace(hour=9, minute=30), entry_price=100.0,
                exit_time=exit_time, exit_price=100.0 + gross, stop_loss=95.0, target_1=110.0, exit_reason=reason,
                is_partial=partial, gross_pnl=gross, charges=charges, net_pnl=round(gross - charges, 6),
                r_multiple=(gross - charges) / 5.0, source="SYSTEM", strategy="orb"))


def add_position(s: Session, *, id: int, symbol: str, mode: str, qty: int, avg: float, last: float | None,
                 entry_charges: float, opened_at: datetime, status: str = "OPEN", holding_days: int = 0,
                 product: str = "CNC", batch_id: int = 1) -> None:
    s.add(Position(id=id, batch_id=batch_id, symbol=symbol, mode=mode, product=product, direction="LONG",
                   status=status, qty=qty, initial_qty=qty, avg_entry=avg, stop_loss=avg * 0.95,
                   initial_stop_loss=avg * 0.95, target_1=avg * 1.1, opened_at=opened_at, margin_blocked=avg * qty,
                   entry_charges=entry_charges, last_price=last, holding_days=holding_days))


def add_decision(s: Session, *, id: int, symbol: str, mode: str, action: str, created: datetime,
                 rule: str | None = None, meta: dict[str, Any] | None = None, confidence: int = 0) -> None:
    s.add(DecisionRow(id=id, batch_id=1, symbol=symbol, mode=mode, decision="NO_TRADE" if action == "NO_TRADE"
                      else "BUY", confidence=confidence, final_action=action, risk_rule=rule, meta_json=meta,
                      created_at=created, grade="B"))


def add_approval(s: Session, *, id: int, symbol: str, status: str, created: datetime,
                 decided: datetime | None = None, decision_id: int | None = None, reason: str | None = None) -> None:
    s.add(ApprovalItem(id=id, batch_id=1, decision_id=decision_id, symbol=symbol, mode="SWING", direction="LONG",
                       entry_type="LIMIT", entry_price=100.0, stop_loss=95.0, target_1=110.0, qty=1, max_qty=2,
                       confidence=70, proposal_json={}, status=status, created_at=created,
                       expires_at=created.replace(hour=23), decided_at=decided, user_reason=reason))


def add_shadow(s: Session, *, id: int, symbol: str, source: str, outcome: str, created: datetime,
               exit_time: datetime | None = None, net: float | None = None) -> None:
    s.add(ShadowTrade(id=id, batch_id=1, symbol=symbol, mode="SWING", direction="LONG", entry_price=100.0,
                      stop_loss=95.0, target_1=110.0, rejection_source=source, created_at=created,
                      status="OPEN" if outcome == "OPEN" else "CLOSED", outcome=outcome, exit_time=exit_time,
                      hypothetical_net_pnl=net))


def add_agent(s: Session, *, id: int, agent: str, symbol: str, status: str, created: datetime,
              error: str | None = None) -> None:
    s.add(AgentReportRow(id=id, batch_id=1, symbol=symbol, mode="INTRADAY", agent=agent, signal="NEUTRAL",
                         confidence=0, status=status, report_json={}, error=error, created_at=created))


def add_ledger(s: Session, *, entry_type: str, amount: float, at: datetime, confirmed: bool = True) -> None:
    s.add(CapitalLedgerEntry(batch_id=1, ts=at, entry_type=entry_type, amount=amount, trading_cash_after=700.0,
                             blocked_after=300.0, reserve_after=15.0, sizing_capital_after=1000.0,
                             confirmed=confirmed))


@pytest.fixture
def populated(session: Session) -> Session:
    s = session
    s.add(Batch(id=1, name="B1", starting_capital=1000.0, status="ACTIVE", created_at=ts(1, 9)))
    s.add(Batch(id=2, name="B2", starting_capital=1000.0, status="CLOSED", created_at=ts(1, 9)))
    s.add(CapitalAccount(batch_id=1, starting_capital=1000.0, base_capital=1000.0, trading_cash=700.0,
                         blocked=300.0, reserve=15.0, reserve_peak=15.0, hwm=1030.0, sizing_capital=1000.0,
                         tier="NORMAL", updated_at=ts(24, 15, 30)))
    # trades
    add_trade(s, id=1, position_id=10, symbol="RELIANCE", mode="INTRADAY", gross=55.0, charges=5.0, exit_time=ts(24, 11))
    add_trade(s, id=2, position_id=11, symbol="TCS", mode="INTRADAY", gross=-17.0, charges=3.0, exit_time=ts(24, 14),
              reason="SL")
    add_trade(s, id=3, position_id=12, symbol="INFY", mode="SWING", gross=32.0, charges=2.0, exit_time=ts(24, 10, 30),
              partial=True, product="CNC")
    add_trade(s, id=4, position_id=12, symbol="INFY", mode="SWING", gross=-8.0, charges=2.0, exit_time=ts(24, 15),
              product="CNC", reason="TRAIL")
    add_trade(s, id=5, position_id=13, symbol="OLD", mode="INTRADAY", gross=99.0, charges=1.0, exit_time=ts(23, 11))
    add_trade(s, id=6, position_id=14, symbol="OTHER", mode="INTRADAY", gross=99.0, charges=1.0, exit_time=ts(24, 11),
              batch_id=2)
    # positions
    add_position(s, id=10, symbol="RELIANCE", mode="INTRADAY", qty=1, avg=100.0, last=155.0, entry_charges=1.0,
                 opened_at=ts(24, 9, 30), status="CLOSED", product="MIS")
    add_position(s, id=20, symbol="HDFCBANK", mode="SWING", qty=2, avg=150.0, last=160.0, entry_charges=0.5,
                 opened_at=ts(19, 10), holding_days=3)
    add_position(s, id=21, symbol="ITC", mode="POSITIONAL", qty=1, avg=400.0, last=None, entry_charges=0.4,
                 opened_at=ts(24, 10), holding_days=0)
    add_position(s, id=22, symbol="FUTURE", mode="SWING", qty=1, avg=10.0, last=11.0, entry_charges=0.1,
                 opened_at=ts(25, 10))  # opened after the report day
    # capital ledger
    add_ledger(s, entry_type="PROFIT_LOCK", amount=15.0, at=ts(24, 15, 30))
    add_ledger(s, entry_type="SIZING_CHANGE", amount=0.0, at=ts(24, 15, 30))
    add_ledger(s, entry_type="REFILL", amount=10.0, at=ts(23, 15, 30))
    add_ledger(s, entry_type="MANUAL_TRANSFER", amount=5.0, at=ts(24, 12), confirmed=True)
    # daily summary history (TOTAL rows), none for today
    s.add(DailySummary(batch_id=1, day=date(2026, 11, 20), mode="TOTAL", equity=1000.0))
    s.add(DailySummary(batch_id=1, day=date(2026, 11, 23), mode="TOTAL", equity=1010.0))
    s.add(DailySummary(batch_id=2, day=date(2026, 11, 23), mode="TOTAL", equity=5.0))
    # decisions
    add_decision(s, id=1, symbol="SBIN", mode="INTRADAY", action="NO_TRADE", created=ts(24, 9, 40),
                 meta={"reasoning": "Weak volume"}, confidence=40)
    add_decision(s, id=2, symbol="SBIN", mode="INTRADAY", action="RISK_REJECTED", created=ts(24, 10),
                 rule="RR_BELOW_MIN")
    add_decision(s, id=3, symbol="RELIANCE", mode="INTRADAY", action="EXECUTED", created=ts(24, 9, 29))
    add_decision(s, id=4, symbol="RELIANCE", mode="INTRADAY", action="NO_TRADE", created=ts(24, 12))
    add_decision(s, id=5, symbol="WIPRO", mode="SWING", action="QUEUED", created=ts(24, 15, 5))
    add_decision(s, id=6, symbol="LT", mode="SWING", action="QUEUED", created=ts(24, 15, 6))
    add_decision(s, id=7, symbol="ACC", mode="SWING", action="NO_TRADE", created=ts(23, 15, 6))  # yesterday
    # stocks analysed
    s.add(StockAnalyzed(batch_id=1, symbol="YESBANK", mode="INTRADAY", as_of=ts(24, 9, 0), stage="UNIVERSE",
                        included=False, reason="price below minimum"))
    s.add(StockAnalyzed(batch_id=1, symbol="SBIN", mode="INTRADAY", as_of=ts(24, 9, 0), stage="WATCHLIST",
                        included=False, reason="dup"))
    s.add(StockAnalyzed(batch_id=1, symbol="TCS", mode="INTRADAY", as_of=ts(24, 9, 0), stage="WATCHLIST",
                        included=True))
    # approvals
    add_approval(s, id=1, symbol="RELIANCE", status="APPROVED", created=ts(24, 9, 20), decided=ts(24, 9, 25))
    add_approval(s, id=2, symbol="WIPRO", status="REJECTED", created=ts(24, 15, 5), decided=ts(24, 15, 10),
                 decision_id=5, reason="Too extended")
    add_approval(s, id=3, symbol="LT", status="TIMEOUT", created=ts(24, 15, 6), decided=ts(24, 15, 36), decision_id=6)
    add_approval(s, id=4, symbol="HDFCBANK", status="MODIFIED", created=ts(23, 15, 6), decided=ts(24, 9, 16))
    add_approval(s, id=5, symbol="ACC", status="PENDING", created=ts(23, 15, 6))
    add_approval(s, id=6, symbol="OLD", status="REJECTED", created=ts(20, 15, 6), decided=ts(20, 15, 9))
    # shadow trades
    add_shadow(s, id=1, symbol="WIPRO", source="USER", outcome="TARGET_1", created=ts(24, 15, 10),
               exit_time=ts(24, 15, 25), net=25.0)
    add_shadow(s, id=2, symbol="LT", source="TIMEOUT", outcome="OPEN", created=ts(24, 15, 36))
    add_shadow(s, id=3, symbol="SBIN", source="RISK", outcome="SL", created=ts(24, 10), exit_time=ts(24, 13), net=-8.0)
    add_shadow(s, id=4, symbol="ACC", source="META", outcome="SL", created=ts(23, 15), exit_time=ts(24, 9, 20),
               net=-5.0)
    add_shadow(s, id=5, symbol="BEL", source="USER", outcome="TARGET_2", created=ts(20, 15), exit_time=ts(21, 11),
               net=40.0)
    # agent reports
    add_agent(s, id=1, agent="news_sentiment", symbol="SBIN", status="FAILED", created=ts(24, 9, 40), error="timeout")
    add_agent(s, id=2, agent="news_sentiment", symbol="TCS", status="FAILED", created=ts(24, 10),
              error="invalid JSON after 2 retries")
    add_agent(s, id=3, agent="fundamental", symbol="SBIN", status="FAILED", created=ts(23, 10), error="old")
    add_agent(s, id=4, agent="technical", symbol="SBIN", status="OK", created=ts(24, 10))
    s.commit()
    return s


@pytest.fixture
def eod_clock() -> SimulatedClock:
    return SimulatedClock(datetime(2026, 11, 24, 15, 45, tzinfo=IST))


# --------------------------------------------------------------------------- tests


def test_trades_by_mode_and_totals(populated: Session, eod_clock: SimulatedClock) -> None:
    r = build_eod_report(populated, 1, DAY, eod_clock)
    assert r["batch_name"] == "B1" and r["day"] == "2026-11-24" and r["empty"] is False
    modes = {m["mode"]: m for m in r["modes"]}
    assert list(modes) == ["INTRADAY", "SWING"]  # Mode enum order
    intra, swing = modes["INTRADAY"], modes["SWING"]
    assert (intra["trades"], intra["wins"], intra["losses"]) == (2, 1, 1)
    assert intra["gross"] == pytest.approx(38.0) and intra["charges"] == pytest.approx(8.0)
    assert intra["net"] == pytest.approx(30.0) and intra["win_rate"] == pytest.approx(50.0)
    # partial + final leg of the same position are one trade (+30 - 10 = +20 -> win)
    assert (swing["trades"], swing["wins"], swing["losses"]) == (1, 1, 0)
    assert swing["net"] == pytest.approx(20.0) and swing["win_rate"] == pytest.approx(100.0)
    t = r["totals"]
    assert (t["trades"], t["legs"], t["wins"], t["losses"]) == (3, 4, 2, 1)
    assert t["net"] == pytest.approx(50.0) and t["charges"] == pytest.approx(12.0) and t["gross"] == pytest.approx(62.0)
    assert t["win_rate"] == pytest.approx(66.67)
    assert [x["id"] for x in r["trades"]] == [3, 1, 2, 4]  # ordered by exit time, other days/batches excluded


def test_open_positions_unrealized_and_days_held(populated: Session, eod_clock: SimulatedClock) -> None:
    r = build_eod_report(populated, 1, DAY, eod_clock)
    pos = {p["symbol"]: p for p in r["open_positions"]}
    assert set(pos) == {"HDFCBANK", "ITC"}  # closed and future positions excluded
    h = pos["HDFCBANK"]
    assert h["unrealized_gross"] == pytest.approx(20.0)
    assert h["unrealized_net"] == pytest.approx(19.5)  # net of accrued entry charges
    assert h["unrealized_pct"] == pytest.approx(6.5)
    assert h["days_held"] == 3 and h["calendar_days"] == 5 and h["price_missing"] is False
    itc = pos["ITC"]
    assert itc["price_missing"] is True and itc["last_price"] is None
    assert itc["unrealized_net"] == pytest.approx(-0.4)
    assert r["open_unrealized_total"] == pytest.approx(19.1)


def test_capital_section_computed_from_accounts_and_ledger(populated: Session, eod_clock: SimulatedClock) -> None:
    r = build_eod_report(populated, 1, DAY, eod_clock)
    c = r["capital"]
    assert c["available"] is True and c["source"] == "computed"
    assert c["realized_today"] == pytest.approx(50.0)
    assert c["profit_locked"] == pytest.approx(15.0)
    assert c["refilled"] == 0.0  # yesterday's refill is not today's
    assert c["loss_absorbed"] == 0.0
    assert c["manual_transfers"] == pytest.approx(5.0)
    assert c["trading_balance"] == pytest.approx(1000.0) and c["blocked"] == pytest.approx(300.0)
    assert c["reserve"] == pytest.approx(15.0)
    assert c["unrealized_pnl"] == pytest.approx(19.1)
    assert c["total_equity"] == pytest.approx(1034.1)
    assert c["hwm"] == pytest.approx(1034.1)  # equity above the stored HWM
    assert c["drawdown_pct"] == pytest.approx(0.0)
    assert c["tier"] == "NORMAL" and c["sizing_capital"] == pytest.approx(1000.0)
    assert c["eod_processed"] is True  # PROFIT_LOCK entry today
    series = r["equity_series"]
    assert [p["day"] for p in series] == ["2026-11-20", "2026-11-23", "2026-11-24"]
    assert series[-1]["equity"] == pytest.approx(1034.1)


def test_capital_prefers_daily_summary_and_mode_rows(populated: Session, eod_clock: SimulatedClock) -> None:
    populated.add(DailySummary(batch_id=1, day=DAY, mode="TOTAL", equity=1020.0, hwm=1040.0, drawdown_pct=1.92,
                               tier="NORMAL", sizing_capital=990.0, unrealized_pnl=19.1, profit_locked=15.0))
    populated.add(DailySummary(batch_id=1, day=DAY, mode="SWING", sizing_capital=297.0))
    populated.add(DailySummary(batch_id=1, day=DAY, mode="INTRADAY", sizing_capital=396.0))
    populated.commit()
    r = build_eod_report(populated, 1, DAY, eod_clock)
    c = r["capital"]
    assert c["source"] == "daily_summary"
    assert c["total_equity"] == pytest.approx(1020.0) and c["hwm"] == pytest.approx(1040.0)
    assert c["drawdown_pct"] == pytest.approx(1.92) and c["sizing_capital"] == pytest.approx(990.0)
    assert list(c["mode_sizing"].items()) == [("INTRADAY", 396.0), ("SWING", 297.0)]
    assert r["equity_series"][-1] == {"day": "2026-11-24", "equity": 1020.0}
    assert len(r["equity_series"]) == 3


def test_mode_sizing_from_settings_when_no_mode_rows(populated: Session, eod_clock: SimulatedClock,
                                                     settings: Settings) -> None:
    r = build_eod_report(populated, 1, DAY, eod_clock, settings=settings)
    expected = {m.value: round(1000.0 * cfg.allocation_pct / 100.0, 2)
                for m, cfg in settings.modes.modes.items() if cfg.enabled and cfg.allocation_pct > 0}
    assert r["capital"]["mode_sizing"] == expected


def test_loss_day_shows_loss_absorbed(session: Session, eod_clock: SimulatedClock) -> None:
    session.add(CapitalAccount(batch_id=7, starting_capital=1000.0, base_capital=1000.0, trading_cash=960.0,
                               blocked=0.0, reserve=20.0, reserve_peak=30.0, hwm=1050.0, sizing_capital=950.0,
                               tier="CAUTION", updated_at=ts(24, 15, 30)))
    add_trade(session, id=50, position_id=60, symbol="TCS", mode="INTRADAY", gross=-35.0, charges=5.0,
              exit_time=ts(24, 12), batch_id=7, reason="SL")
    session.add(CapitalLedgerEntry(batch_id=7, ts=ts(24, 15, 30), entry_type="REFILL", amount=12.5,
                                   trading_cash_after=972.5, blocked_after=0.0, reserve_after=7.5,
                                   sizing_capital_after=950.0))
    session.commit()
    r = build_eod_report(session, 7, DAY, eod_clock)
    c = r["capital"]
    assert c["realized_today"] == pytest.approx(-40.0)
    assert c["loss_absorbed"] == pytest.approx(40.0)
    assert c["refilled"] == pytest.approx(12.5)
    assert c["total_equity"] == pytest.approx(980.0)
    assert c["drawdown_pct"] == pytest.approx((1050 - 980) / 1050 * 100, abs=0.01)
    assert c["tier"] == "CAUTION"
    assert r["totals"]["losses"] == 1 and r["totals"]["win_rate"] == 0.0


def test_not_traded_with_reasons(populated: Session, eod_clock: SimulatedClock) -> None:
    r = build_eod_report(populated, 1, DAY, eod_clock)
    items = r["not_traded"]
    keys = [(i["symbol"], i["mode"]) for i in items]
    assert keys == [("SBIN", "INTRADAY"), ("LT", "SWING"), ("WIPRO", "SWING"), ("YESBANK", "INTRADAY")]
    by = {i["symbol"]: i for i in items}
    assert by["SBIN"]["reason"] == "Risk veto: RR_BELOW_MIN"  # latest decision wins
    assert by["WIPRO"]["reason"] == "Rejected by owner: Too extended"
    assert by["LT"]["reason"] == "Approval timed out"
    assert by["YESBANK"]["source"] == "FILTER" and "price below minimum" in by["YESBANK"]["reason"]
    assert "RELIANCE" not in by  # traded today
    assert "ACC" not in by  # yesterday's decision
    assert r["not_traded_total"] == 4


def test_no_trade_reason_uses_meta_reasoning(session: Session, eod_clock: SimulatedClock) -> None:
    add_decision(session, id=90, symbol="BEL", mode="SWING", action="NO_TRADE", created=ts(24, 16),
                 meta={"reasoning": "  Debate   split; bear stronger "}, confidence=55)
    session.commit()
    r = build_eod_report(session, 1, DAY, eod_clock)
    assert r["not_traded"][0]["reason"] == "Meta NO_TRADE (confidence 55): Debate split; bear stronger"


def test_approvals_and_shadow(populated: Session, eod_clock: SimulatedClock) -> None:
    r = build_eod_report(populated, 1, DAY, eod_clock)
    a = r["approvals"]
    assert (a["total"], a["approved"], a["modified"], a["rejected"], a["timeout"], a["pending"]) == (4, 1, 1, 1, 1, 0)
    sh = r["shadow"]
    assert (sh["total"], sh["open"], sh["closed"], sh["target_hits"], sh["sl_hits"]) == (4, 1, 3, 1, 2)
    assert sh["hypothetical_net"] == pytest.approx(12.0)
    by = {row["source"]: row for row in sh["by_source"]}
    assert set(by) == {"META", "RISK", "TIMEOUT", "USER"}
    assert by["USER"]["target_hits"] == 1 and by["USER"]["hypothetical_net"] == pytest.approx(25.0)
    assert by["TIMEOUT"]["closed"] == 0
    cum = {row["source"]: row for row in sh["cumulative_by_source"]}
    assert cum["USER"]["closed"] == 2 and cum["USER"]["hypothetical_net"] == pytest.approx(65.0)
    assert "TIMEOUT" not in cum  # only closed shadows count cumulatively


def test_agent_failures(populated: Session, eod_clock: SimulatedClock) -> None:
    r = build_eod_report(populated, 1, DAY, eod_clock)
    f = r["agent_failures"]
    assert f["total"] == 2
    assert f["by_agent"] == [{"agent": "news_sentiment", "count": 2, "symbols": ["SBIN", "TCS"],
                              "last_error": "invalid JSON after 2 retries"}]


def test_empty_day_is_graceful(session: Session, eod_clock: SimulatedClock) -> None:
    r = build_eod_report(session, 99, DAY, eod_clock)
    assert r["empty"] is True and r["batch_name"] is None
    assert r["modes"] == [] and r["totals"]["trades"] == 0 and r["totals"]["win_rate"] is None
    assert r["open_positions"] == [] and r["not_traded"] == [] and r["equity_series"] == []
    assert r["capital"]["available"] is False and r["capital"]["total_equity"] is None
    assert r["approvals"]["total"] == 0 and r["shadow"]["by_source"] == []
    assert r["agent_failures"] == {"total": 0, "by_agent": []}


def test_empty_day_with_account_has_single_equity_point(session: Session, eod_clock: SimulatedClock) -> None:
    session.add(CapitalAccount(batch_id=3, starting_capital=1000.0, base_capital=1000.0, trading_cash=1000.0,
                               blocked=0.0, reserve=0.0, reserve_peak=0.0, hwm=1000.0, sizing_capital=1000.0,
                               updated_at=ts(24, 9)))
    session.commit()
    r = build_eod_report(session, 3, DAY, eod_clock)
    assert r["empty"] is True
    assert r["capital"]["total_equity"] == pytest.approx(1000.0) and r["capital"]["eod_processed"] is False
    assert r["equity_series"] == [{"day": "2026-11-24", "equity": 1000.0}]


def test_report_is_json_serialisable(populated: Session, eod_clock: SimulatedClock) -> None:
    import json

    r = build_eod_report(populated, 1, DAY, eod_clock)
    json.dumps(r)  # no datetimes / sets / enums left


def test_unrealized_pnl_short() -> None:
    assert unrealized_pnl("SHORT", 3, 100.0, 90.0) == pytest.approx(30.0)
    assert unrealized_pnl("LONG", 3, 100.0, 90.0) == pytest.approx(-30.0)


@pytest.mark.parametrize(
    "points",
    [
        [{"day": "2026-11-20", "equity": 1000.0}, {"day": "2026-11-23", "equity": 1010.0},
         {"day": "2026-11-24", "equity": 1034.1}],
        [(date(2026, 11, 23), 1000.0), (date(2026, 11, 24), 995.5)],
        [{"day": "2026-11-24", "equity": 1000.0}],
        [],
        [{"day": "bad", "equity": 1.0}, {"day": "2026-11-24", "equity": None}],
        [(date(2026, 1, 1) .replace(day=d), 100000.0 + d * 1000) for d in range(1, 29)],
    ],
)
def test_equity_chart_png_signature(points: list[Any]) -> None:
    png = equity_chart_png(points, baseline=1000.0)
    assert png.startswith(PNG_SIGNATURE) and len(png) > 1000
