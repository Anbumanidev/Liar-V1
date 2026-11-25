"""Realistic seed data for the analytics tests (imported by the other analytics test modules).

One CLOSED batch ("ana-old", ledger-only history) and one ACTIVE batch ("ana-cur") whose rows
are internally consistent: the capital ledger balances (start + realized + manual = TRADING +
RESERVE), the daily summaries match the ledger, trades match positions/approvals/decisions,
and every kind of "not traded" outcome is present with its shadow trade.

Key numbers of "ana-cur" (starting capital 1,000; hard stop 9 %):

* closed positions: TATASTEEL +34 (approved), ITC -65 (modified), TATAPOWER +8.8 (two legs,
  AUTO), NHPC +18.5 (manual short) -> net -3.7, gross 2.0, charges 5.7;
* EOD equity: D3 1,034 (HWM), D2 969 (DEFENSIVE), D1 996.3 (CAUTION);
* now: TRADING 988.01 (cash 588.01 + blocked 400), RESERVE 8.29, open PNB 4 @ 100 with LTP 103
  (unrealized 11.6) -> TOTAL EQUITY 1,007.9.
"""

from __future__ import annotations

import copy
from datetime import date, datetime, time, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config.loader import load_hard_caps
from config.store import ensure_seeded
from core.clock import IST
from storage.models import (
    AgentReportRow,
    ApprovalItem,
    AuditLog,
    Batch,
    CapitalAccount,
    CapitalLedgerEntry,
    ControlCommand,
    DailySummary,
    DebateRow,
    DecisionRow,
    EngineStatus,
    Order,
    Position,
    Run,
    ShadowTrade,
    StockAnalyzed,
    Trade,
)

#: Fixed clock for the unit tests (a Tuesday).
NOW = datetime(2026, 11, 24, 12, 0, tzinfo=IST)


def business_days_back(day: date, n: int) -> date:
    """The n-th weekday before ``day``."""
    out = day
    while n > 0:
        out -= timedelta(days=1)
        if out.weekday() < 5:
            n -= 1
    return out


def at(day: date, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.combine(day, time(h, m), tzinfo=IST)


def seed_days(now: datetime) -> dict[str, date]:
    today = now.astimezone(IST).date()
    return {"D0": today, "D1": business_days_back(today, 1), "D2": business_days_back(today, 2),
            "D3": business_days_back(today, 3)}


def seed_analytics_db(s: Session, now: datetime = NOW) -> dict[str, int]:
    """Insert the analytics fixture rows and commit. Returns the interesting ids."""
    days = seed_days(now)
    d0, d1, d2, d3 = days["D0"], days["D1"], days["D2"], days["D3"]
    ids: dict[str, int] = {}

    v1 = ensure_seeded(s, load_hard_caps())
    snapshot = copy.deepcopy(v1.settings_json)
    snapshot["capital"]["profit_lock_pct"] = min(100.0, float(snapshot["capital"]["profit_lock_pct"]) + 10.0)

    old = Batch(name="ana-old", starting_capital=1000.0, status="CLOSED", created_at=now - timedelta(days=30),
                started_at=now - timedelta(days=30), ended_at=now - timedelta(days=10), settings_version_id=v1.id)
    cur = Batch(name="ana-cur", starting_capital=1000.0, status="ACTIVE", created_at=at(d3, "09:00"),
                started_at=at(d3, "09:00"), settings_snapshot=snapshot)
    s.add_all([old, cur])
    s.flush()
    b = cur.id
    ids.update(batch=b, old_batch=old.id, settings_v1=v1.id)

    # ------------------------------------------------------------------ old batch (ledger-only history)
    o0 = now - timedelta(days=30)
    s.add(CapitalAccount(batch_id=old.id, starting_capital=1000.0, base_capital=1000.0, trading_cash=1049.0,
                         blocked=0.0, reserve=21.0, reserve_peak=21.0, hwm=1070.0, sizing_capital=1000.0,
                         updated_at=now - timedelta(days=10)))
    for ts, kind, amount, cash, reserve in (
        (o0, "INITIAL_CAPITAL", 1000.0, 1000.0, 0.0),
        (o0 + timedelta(days=5, hours=2), "REALIZED_PNL", 50.0, 1050.0, 0.0),
        (o0 + timedelta(days=5, hours=6), "PROFIT_LOCK", 15.0, 1035.0, 15.0),
        (o0 + timedelta(days=10, hours=2), "REALIZED_PNL", 20.0, 1055.0, 15.0),
        (o0 + timedelta(days=10, hours=6), "PROFIT_LOCK", 6.0, 1049.0, 21.0),
    ):
        s.add(CapitalLedgerEntry(batch_id=old.id, ts=ts, entry_type=kind, amount=amount, trading_cash_after=cash,
                                 blocked_after=0.0, reserve_after=reserve, sizing_capital_after=1000.0))
    for k, (pnl, offset) in enumerate(((50.0, 5), (20.0, 10))):
        t_in = o0 + timedelta(days=offset, hours=1)
        s.add(Trade(batch_id=old.id, position_id=900 + k, symbol="SAIL", mode="INTRADAY", product="MIS",
                    direction="LONG", qty=10, entry_time=t_in, entry_price=120.0,
                    exit_time=t_in + timedelta(hours=1), exit_price=125.0 + k, stop_loss=118.0, exit_reason="TARGET",
                    gross_pnl=pnl + 1.0, charges=1.0, net_pnl=pnl, r_multiple=pnl / 20.0, source="SYSTEM",
                    strategy="orb", regime="TRENDING_UP"))

    # ------------------------------------------------------------------ decisions
    def decision(sym: str, mode: str, ts: datetime, final: str, **kw: object) -> DecisionRow:
        row = DecisionRow(batch_id=b, symbol=sym, mode=mode, created_at=ts, final_action=final,
                          decision=str(kw.pop("decision", "BUY")), confidence=int(kw.pop("confidence", 75)), **kw)
        s.add(row)
        s.flush()
        return row

    d_t1 = decision("TATASTEEL", "INTRADAY", at(d3, "09:34"), "QUEUED", strategy="orb", regime="TRENDING_UP",
                    direction="LONG", confidence=80, grade="A", confluence_score=78.0, entry_price=150.0,
                    stop_loss=147.0, target_1=156.0, target_2=159.0, latency_ms=2500,
                    meta_json={"reasoning": "Opening range breakout with 2x volume"},
                    context_json={"confluence": {"score": 78.0, "grade": "A"}},
                    risk_approved=True, risk_json={"approved": True, "checks": []}, qty=6)
    d_t2 = decision("ITC", "SWING", at(d3, "14:58"), "QUEUED", strategy="swing_breakout", regime="RANGING",
                    direction="LONG", confidence=74, grade="B", entry_price=450.0, stop_loss=435.0, target_1=480.0,
                    latency_ms=3000, meta_json={"reasoning": "Base breakout"}, risk_approved=True, qty=1)
    d_t3 = decision("TATAPOWER", "INTRADAY", at(d1, "10:04"), "EXECUTED", strategy="vwap_pullback",
                    regime="TRENDING_UP", direction="LONG", confidence=82, grade="A+", entry_price=200.0,
                    stop_loss=197.0, target_1=203.0, meta_json={"reasoning": "VWAP pullback in uptrend"}, qty=4)
    d_a3 = decision("WIPRO", "INTRADAY", at(d0, "09:59"), "QUEUED", direction="LONG", confidence=71, grade="B",
                    entry_price=250.0, stop_loss=247.0, target_1=256.0, strategy="ema_pullback")
    d_a4 = decision("LT", "SWING", at(d1, "14:59"), "QUEUED", direction="LONG", confidence=72, grade="B",
                    entry_price=360.0, stop_loss=350.0, target_1=380.0, strategy="swing_breakout")
    d_a5 = decision("BEL", "SWING", now - timedelta(minutes=11), "QUEUED", direction="LONG", confidence=76,
                    grade="A", entry_price=300.0, stop_loss=290.0, target_1=320.0, strategy="swing_breakout")
    d_a7 = decision("GAIL", "INTRADAY", at(d1, "11:25"), "QUEUED", direction="LONG", confidence=73, grade="B",
                    entry_price=210.0, stop_loss=207.0, target_1=216.0, strategy="orb")
    d_nt = decision("AXISBANK", "INTRADAY", at(d0, "10:30"), "NO_TRADE", decision="NO_TRADE", confidence=40,
                    meta_json={"reasoning": "Conflicting signals: trend up but sentiment negative"})
    d_risk = decision("KOTAKBANK", "INTRADAY", at(d0, "10:45"), "RISK_REJECTED", direction="LONG", confidence=78,
                      grade="A", risk_approved=False, risk_rule="MAX_OPEN_POSITIONS",
                      risk_json={"approved": False, "rule": "MAX_OPEN_POSITIONS", "message": "5 positions open"})
    d_sig = decision("SUNPHARMA", "POSITIONAL", at(d1, "09:30"), "SIGNAL_ONLY", direction="LONG", confidence=77,
                     grade="A", entry_price=700.0, stop_loss=670.0, target_1=760.0)
    d_filt = decision("TITAN", "SWING", at(d1, "09:31"), "FILTERED", decision="NO_TRADE", confidence=0,
                      context_json={"reason": "confluence 42 < 60"})
    ids.update(d_t1=d_t1.id, d_t2=d_t2.id, d_t3=d_t3.id, d_nt=d_nt.id, d_risk=d_risk.id, d_sig=d_sig.id,
               d_filt=d_filt.id)

    # ------------------------------------------------------------------ approvals
    def approval(dec: DecisionRow | None, sym: str, mode: str, status: str, created: datetime, **kw: object) -> ApprovalItem:
        row = ApprovalItem(batch_id=b, decision_id=dec.id if dec else None, symbol=sym, mode=mode,
                           direction="LONG", entry_type="MARKET", entry_price=float(kw.pop("entry", 100.0)),
                           stop_loss=float(kw.pop("sl", 98.0)), target_1=float(kw.pop("t1", 104.0)),
                           qty=int(kw.pop("qty", 1)), max_qty=int(kw.pop("max_qty", 5)), confidence=70,
                           proposal_json={}, status=status, created_at=created,
                           expires_at=kw.pop("expires", created + timedelta(minutes=5)), **kw)
        s.add(row)
        s.flush()
        return row

    a1 = approval(d_t1, "TATASTEEL", "INTRADAY", "APPROVED", at(d3, "09:35"), entry=150.0, sl=147.0, t1=156.0,
                  qty=6, max_qty=6, decided_at=at(d3, "09:38"), decided_by="admin", strategy="orb", grade="A")
    a2 = approval(d_t2, "ITC", "SWING", "MODIFIED", at(d3, "15:00"), entry=450.0, sl=435.0, t1=480.0, qty=1,
                  max_qty=2, decided_at=at(d3, "15:05"), decided_by="admin", user_reason="wider stop below base",
                  user_edits_json={"stop_loss": {"old": 435.0, "new": 420.0}, "qty": {"old": 1, "new": 2}},
                  strategy="swing_breakout", expires=at(d2, "09:45"))
    a3 = approval(d_a3, "WIPRO", "INTRADAY", "REJECTED", at(d0, "10:00"), entry=250.0, sl=247.0, t1=256.0, qty=3,
                  decided_at=at(d0, "10:02"), decided_by="admin", user_reason="too extended")
    a4 = approval(d_a4, "LT", "SWING", "TIMEOUT", at(d1, "15:00"), entry=360.0, sl=350.0, t1=380.0,
                  expires=at(d2, "09:45"))
    a5 = approval(d_a5, "BEL", "SWING", "PENDING", now - timedelta(minutes=10), entry=300.0, sl=290.0, t1=320.0,
                  expires=now + timedelta(hours=2))
    a6 = approval(None, "IRFC", "INTRADAY", "REJECTED", at(d0, "11:00"), entry=140.0, sl=139.0, t1=141.0,
                  decided_at=at(d0, "11:01"), decided_by="admin", block_rule="RR_BELOW_MIN")
    a7 = approval(d_a7, "GAIL", "INTRADAY", "APPROVED", at(d1, "11:26"), entry=210.0, sl=207.0, t1=216.0,
                  decided_at=at(d1, "11:30"), decided_by="admin")
    ids.update(a1=a1.id, a2=a2.id, a3=a3.id, a4=a4.id, a5=a5.id, a6=a6.id, a7=a7.id)

    # ------------------------------------------------------------------ positions & trades
    def position(sym: str, mode: str, product: str, direction: str, qty: int, entry: float, sl: float,
                 opened: datetime, status: str = "CLOSED", **kw: object) -> Position:
        row = Position(batch_id=b, symbol=sym, mode=mode, product=product, direction=direction, status=status,
                       qty=qty if status == "OPEN" else 0, initial_qty=qty, avg_entry=entry, stop_loss=sl,
                       initial_stop_loss=sl, opened_at=opened, margin_blocked=entry * qty if status == "OPEN" else 0.0,
                       source=str(kw.pop("source", "SYSTEM")), **kw)
        s.add(row)
        s.flush()
        return row

    p1 = position("TATASTEEL", "INTRADAY", "MIS", "LONG", 6, 150.0, 147.0, at(d3, "09:40"), target_1=156.0,
                  strategy="orb", decision_id=d_t1.id, approval_id=a1.id, closed_at=at(d3, "11:15"))
    p2 = position("ITC", "SWING", "CNC", "LONG", 2, 450.0, 420.0, at(d2, "09:20"), target_1=480.0,
                  strategy="swing_breakout", decision_id=d_t2.id, approval_id=a2.id, closed_at=at(d2, "13:10"),
                  sl_history_json=[{"ts": at(d2, "11:00").isoformat(), "old": 420.0, "new": 422.0, "by": "admin",
                                    "reason": "tighten"}])
    p3 = position("TATAPOWER", "INTRADAY", "MIS", "LONG", 4, 200.0, 197.0, at(d1, "10:05"), target_1=203.0,
                  strategy="vwap_pullback", decision_id=d_t3.id, closed_at=at(d1, "12:30"))
    p4 = position("NHPC", "INTRADAY", "MIS", "SHORT", 10, 85.0, 86.0, at(d1, "13:00"), target_1=83.0,
                  source="MANUAL", closed_at=at(d1, "14:00"))
    p5 = position("PNB", "INTRADAY", "MIS", "LONG", 4, 100.0, 98.0, at(d0, "09:45"), status="OPEN", target_1=104.0,
                  target_2=106.0, strategy="orb", entry_charges=0.4, last_price=103.0,
                  last_price_at=now - timedelta(seconds=20))
    ids.update(p1=p1.id, p2=p2.id, p3=p3.id, p4=p4.id, p5=p5.id)

    def trade(pos: Position, qty: int, entry_t: datetime, exit_t: datetime, exit_px: float, reason: str,
              gross: float, charges: float, r: float, **kw: object) -> Trade:
        row = Trade(batch_id=b, position_id=pos.id, symbol=pos.symbol, mode=pos.mode, product=pos.product,
                    direction=pos.direction, qty=qty, entry_time=entry_t, entry_price=pos.avg_entry,
                    exit_time=exit_t, exit_price=exit_px, stop_loss=pos.initial_stop_loss,
                    target_1=pos.target_1, exit_reason=reason, gross_pnl=gross, charges=charges,
                    net_pnl=round(gross - charges, 2), r_multiple=r, source=pos.source, strategy=pos.strategy,
                    decision_id=pos.decision_id, approval_id=pos.approval_id, **kw)
        s.add(row)
        s.flush()
        return row

    t1 = trade(p1, 6, at(d3, "09:40"), at(d3, "11:15"), 156.0, "TARGET", 36.0, 2.0, 1.89, regime="TRENDING_UP",
               grade="A", holding_minutes=95)
    t2 = trade(p2, 2, at(d2, "09:20"), at(d2, "13:10"), 418.0, "SL", -64.0, 1.0, -1.08, grade="B", holding_days=0)
    t3 = trade(p3, 2, at(d1, "10:05"), at(d1, "11:00"), 205.0, "TARGET", 10.0, 0.6, 1.57, regime="TRENDING_UP",
               grade="A+", is_partial=True, holding_minutes=55)
    t4 = trade(p3, 2, at(d1, "10:05"), at(d1, "12:30"), 200.0, "SL", 0.0, 0.6, -0.1, regime="TRENDING_UP",
               grade="A+", holding_minutes=145)
    t5 = trade(p4, 10, at(d1, "13:00"), at(d1, "14:00"), 83.0, "TARGET", 20.0, 1.5, 1.85, holding_minutes=60)
    ids.update(t1=t1.id, t2=t2.id, t3=t3.id, t4=t4.id, t5=t5.id)

    s.add_all([
        Order(batch_id=b, position_id=p1.id, approval_id=a1.id, decision_id=d_t1.id, symbol="TATASTEEL",
              mode="INTRADAY", product="MIS", side="BUY", order_type="MARKET", purpose="ENTRY", qty=6, status="FILLED",
              created_at=at(d3, "09:39"), filled_at=at(d3, "09:40"), fill_price=150.0, slippage=0.05),
        Order(batch_id=b, position_id=p1.id, symbol="TATASTEEL", mode="INTRADAY", product="MIS", side="SELL",
              order_type="LIMIT", purpose="EXIT", qty=6, limit_price=156.0, status="FILLED",
              created_at=at(d3, "09:40"), filled_at=at(d3, "11:15"), fill_price=156.0, reason="TARGET"),
        Order(batch_id=b, position_id=p5.id, symbol="PNB", mode="INTRADAY", product="MIS", side="BUY",
              order_type="MARKET", purpose="ENTRY", qty=4, status="FILLED", created_at=at(d0, "09:44"),
              filled_at=at(d0, "09:45"), fill_price=100.0),
    ])

    # ------------------------------------------------------------------ shadow trades
    def shadow(src: str, sym: str, created: datetime, *, outcome: str = "OPEN", pnl: float | None = None,
               r: float | None = None, dec: DecisionRow | None = None, appr: ApprovalItem | None = None,
               **kw: object) -> ShadowTrade:
        row = ShadowTrade(batch_id=b, decision_id=dec.id if dec else None, approval_id=appr.id if appr else None,
                          symbol=sym, mode=str(kw.pop("mode", "INTRADAY")), direction="LONG", entry_price=100.0,
                          stop_loss=98.0, target_1=104.0, qty=1, rejection_source=src, created_at=created,
                          status="OPEN" if outcome == "OPEN" else "CLOSED", outcome=outcome,
                          hypothetical_net_pnl=pnl, r_multiple=r,
                          exit_time=None if outcome == "OPEN" else created + timedelta(hours=2), **kw)
        s.add(row)
        s.flush()
        return row

    sh_a2 = shadow("USER", "ITC", at(d3, "15:05"), outcome="SL", pnl=-40.0, r=-1.0, dec=d_t2, appr=a2, mode="SWING",
                   rejection_reason="original proposal (modified by owner)")
    sh_a3 = shadow("USER", "WIPRO", at(d0, "10:02"), outcome="TARGET_1", pnl=24.0, r=2.0, dec=d_a3, appr=a3)
    sh_a4 = shadow("TIMEOUT", "LT", at(d2, "09:45"), outcome="SL", pnl=-30.0, r=-1.0, dec=d_a4, appr=a4, mode="SWING")
    sh_nt = shadow("META", "AXISBANK", at(d0, "10:30"), dec=d_nt)
    sh_sig = shadow("SIGNALS_ONLY", "SUNPHARMA", at(d1, "09:30"), outcome="NOT_TRIGGERED", dec=d_sig,
                    mode="POSITIONAL")
    sh_orphan = shadow("FILTER", "HAL", at(d1, "09:32"), outcome="TARGET_1", pnl=12.0, r=1.2,
                       rejection_reason="below minimum confluence")
    ids.update(sh_a2=sh_a2.id, sh_a3=sh_a3.id, sh_a4=sh_a4.id, sh_nt=sh_nt.id, sh_sig=sh_sig.id,
               sh_orphan=sh_orphan.id)

    # ------------------------------------------------------------------ agents & debate
    reports = (
        (d_t1, "technical", "BULLISH", 75, "OK", 1200), (d_t1, "sentiment", "BEARISH", 60, "OK", 800),
        (d_t1, "news", "NEUTRAL", 0, "FAILED", 30000), (d_t2, "technical", "BULLISH", 70, "OK", 1500),
        (d_t2, "sentiment", "BEARISH", 65, "OK", 900), (d_t3, "technical", "BULLISH", 80, "OK", 1000),
        (d_t3, "sentiment", "BULLISH", 55, "SKIPPED", None),
    )
    for dec, agent, signal, conf, status, latency in reports:
        s.add(AgentReportRow(batch_id=b, decision_id=dec.id, symbol=dec.symbol, mode=dec.mode, agent=agent,
                             signal=signal, confidence=conf, status=status, model="qwen3:4b", prompt_version=1,
                             latency_ms=latency, tokens_in=500 if latency else None, tokens_out=120 if latency else None,
                             report_json={"signal": signal, "reasons": [f"{agent} reason"]},
                             error="timeout" if status == "FAILED" else None, created_at=dec.created_at))
    s.add(DebateRow(batch_id=b, decision_id=d_t1.id, symbol="TATASTEEL", mode="INTRADAY", rounds=2, status="OK",
                    transcript_json=[{"side": "bull", "round": 1, "text": "Volume confirms the breakout"},
                                     {"side": "bear", "round": 1, "text": "Index is weak"}],
                    bull_json={"points": ["volume"]}, bear_json={"points": ["index"]}, model="qwen3:8b",
                    latency_ms=4000, created_at=at(d3, "09:33")))

    # ------------------------------------------------------------------ runs & scans
    run = Run(batch_id=b, run_type="INTRADAY_CYCLE", mode="INTRADAY", started_at=at(d0, "09:30"),
              finished_at=at(d0, "09:31") + timedelta(seconds=10), status="DONE", n_symbols=12)
    eod = Run(batch_id=b, run_type="EOD", started_at=at(d1, "15:30"), finished_at=at(d1, "15:31"), status="DONE",
              n_symbols=0, summary_json={"net": 27.3})
    s.add_all([run, eod])
    s.flush()
    ids["run"] = run.id
    for sym, stage, included, reason, conf, grade in (
        ("PNB", "ANALYZED", True, None, 72.0, "A"), ("WIPRO", "ANALYZED", True, None, 66.0, "B"),
        ("YESBANK", "UNIVERSE", False, "illiquid: spread 0.8%", None, None),
    ):
        s.add(StockAnalyzed(run_id=run.id, batch_id=b, symbol=sym, mode="INTRADAY", as_of=at(d0, "09:30"),
                            stage=stage, included=included, reason=reason, regime="TRENDING_UP", confluence=conf,
                            grade=grade))

    # ------------------------------------------------------------------ capital
    ledger = (
        (at(d3, "09:00"), "INITIAL_CAPITAL", 1000.0, 1000.0, 0.0, 0.0, 1000.0, None, None),
        (at(d3, "09:40"), "BLOCK", 900.0, 100.0, 900.0, 0.0, 1000.0, "INTRADAY", None),
        (at(d3, "11:15"), "RELEASE", 900.0, 1000.0, 0.0, 0.0, 1000.0, "INTRADAY", None),
        (at(d3, "11:15"), "REALIZED_PNL", 34.0, 1034.0, 0.0, 0.0, 1000.0, "INTRADAY", None),
        (at(d3, "15:30"), "PROFIT_LOCK", 10.2, 1023.8, 0.0, 10.2, 1000.0, None, "30% of +34.00"),
        (at(d2, "09:20"), "BLOCK", 900.0, 123.8, 900.0, 10.2, 1000.0, "SWING", None),
        (at(d2, "13:10"), "RELEASE", 900.0, 1023.8, 0.0, 10.2, 1000.0, "SWING", None),
        (at(d2, "13:10"), "REALIZED_PNL", -65.0, 958.8, 0.0, 10.2, 1000.0, "SWING", None),
        (at(d2, "15:30"), "LOSS_ABSORB", 0.0, 958.8, 0.0, 10.2, 1000.0, None, "loss of -65.00 absorbed"),
        (at(d2, "15:30"), "REFILL", 5.1, 963.9, 0.0, 5.1, 1000.0, None, "refill capped at 50% of reserve peak"),
        (at(d2, "15:31"), "SIZING_CHANGE", -36.1, 963.9, 0.0, 5.1, 963.9, None, "shrink fast"),
        (at(d1, "10:05"), "BLOCK", 800.0, 163.9, 800.0, 5.1, 963.9, "INTRADAY", None),
        (at(d1, "11:00"), "RELEASE", 400.0, 563.9, 400.0, 5.1, 963.9, "INTRADAY", None),
        (at(d1, "11:00"), "REALIZED_PNL", 9.4, 573.3, 400.0, 5.1, 963.9, "INTRADAY", None),
        (at(d1, "12:30"), "RELEASE", 400.0, 973.3, 0.0, 5.1, 963.9, "INTRADAY", None),
        (at(d1, "12:30"), "REALIZED_PNL", -0.6, 972.7, 0.0, 5.1, 963.9, "INTRADAY", None),
        (at(d1, "13:00"), "BLOCK", 850.0, 122.7, 850.0, 5.1, 963.9, "INTRADAY", None),
        (at(d1, "14:00"), "RELEASE", 850.0, 972.7, 0.0, 5.1, 963.9, "INTRADAY", None),
        (at(d1, "14:00"), "REALIZED_PNL", 18.5, 991.2, 0.0, 5.1, 963.9, "INTRADAY", None),
        (at(d1, "15:30"), "PROFIT_LOCK", 8.19, 983.01, 0.0, 13.29, 963.9, None, "30% of +27.30"),
        (at(d1, "16:00"), "MANUAL_TRANSFER", 5.0, 988.01, 0.0, 8.29, 963.9, None, "top up trading"),
        (at(d0, "09:45"), "BLOCK", 400.0, 588.01, 400.0, 8.29, 963.9, "INTRADAY", None),
    )
    for ts, kind, amount, cash, blocked, reserve, sizing, mode, note in ledger:
        s.add(CapitalLedgerEntry(batch_id=b, ts=ts, entry_type=kind, amount=amount, trading_cash_after=cash,
                                 blocked_after=blocked, reserve_after=reserve, sizing_capital_after=sizing, mode=mode,
                                 note=note, actor="admin" if kind == "MANUAL_TRANSFER" else "SYSTEM",
                                 from_account="RESERVE" if kind in ("REFILL", "MANUAL_TRANSFER") else None,
                                 to_account="TRADING" if kind in ("REFILL", "MANUAL_TRANSFER") else None))
    s.add(CapitalAccount(batch_id=b, starting_capital=1000.0, base_capital=1000.0, trading_cash=588.01, blocked=400.0,
                         reserve=8.29, reserve_peak=13.29, hwm=1034.0, sizing_capital=963.9, tier="NORMAL",
                         cum_realized_net=-3.7, cum_manual_adjustments=0.0, last_eod_date=d1, updated_at=now))
    for day, trading, reserve, equity, dd, tier, trades, net, locked, refilled in (
        (d3, 1023.8, 10.2, 1034.0, 0.0, "NORMAL", 1, 34.0, 10.2, 0.0),
        (d2, 963.9, 5.1, 969.0, 6.286, "DEFENSIVE", 1, -65.0, 0.0, 5.1),
        (d1, 983.01, 13.29, 996.3, 3.646, "CAUTION", 2, 27.3, 8.19, 0.0),
    ):
        s.add(DailySummary(batch_id=b, day=day, mode="TOTAL", trades=trades, wins=1 if net > 0 else 0,
                           losses=0 if net > 0 else 1, net_pnl=net, trading_balance=trading, reserve=reserve,
                           equity=equity, hwm=1034.0, drawdown_pct=dd, tier=tier, profit_locked=locked,
                           refilled=refilled))
    s.add(DailySummary(batch_id=b, day=d1, mode="INTRADAY", trades=2, wins=2, net_pnl=27.3, equity=5.0))

    # ------------------------------------------------------------------ commands, audit, engine
    s.add_all([
        ControlCommand(created_at=at(d3, "09:38"), command_type="APPROVE", payload_json={"approval_id": a1.id},
                       status="DONE", requested_by="admin", batch_id=b, result_json={"ok": True, "message": "filled"}),
        ControlCommand(created_at=at(d3, "15:05"), command_type="MODIFY_APPROVE",
                       payload_json={"approval_id": a2.id, "stop_loss": 420.0, "qty": 2}, status="DONE",
                       requested_by="admin", batch_id=b, result_json={"ok": True, "message": "approved as modified"}),
        ControlCommand(created_at=at(d2, "11:00"), command_type="TIGHTEN_SL",
                       payload_json={"position_id": p2.id, "stop_loss": 422.0}, status="DONE", requested_by="admin",
                       batch_id=b, result_json={"ok": True, "message": "SL 420.00 -> 422.00"}),
        ControlCommand(created_at=at(d0, "11:00"), command_type="APPROVE", payload_json={"approval_id": a6.id},
                       status="REJECTED", requested_by="admin", batch_id=b,
                       result_json={"ok": False, "rule": "RR_BELOW_MIN", "message": "R:R 1.0 < 1.5"}),
        AuditLog(ts=at(d3, "15:05"), actor="admin", action="APPROVAL_MODIFY", target=f"approval:{a2.id}",
                 batch_id=b, details_json={"stop_loss": [435.0, 420.0]}),
    ])
    if s.get(EngineStatus, 1) is None:
        s.add(EngineStatus(id=1, state="RUNNING", heartbeat_at=now, started_at=now - timedelta(hours=3), pid=4321,
                     activity="intraday cycle", next_cycle_at=now + timedelta(minutes=3), batch_id=b,
                     data_feed_ok=True, llm_ok=True, market_open=True,
                     regime_json={"INTRADAY": {"regime": "TRENDING_UP", "adx": 28.5}, "SWING": {"regime": "RANGING"},
                                  "vix": 13.2}))
    s.commit()
    return ids


@pytest.fixture
def ana_session(session: Session) -> Session:
    seed_analytics_db(session)
    return session


def test_dashboard_analytics_seed_ledger_balances(ana_session: Session) -> None:
    """Section 6.4: start + realized + manual adjustments == TRADING + RESERVE (seed sanity)."""
    batch = ana_session.execute(select(Batch).where(Batch.name == "ana-cur")).scalar_one()
    acct = ana_session.get(CapitalAccount, batch.id)
    realized = ana_session.execute(select(func.sum(Trade.net_pnl)).where(Trade.batch_id == batch.id)).scalar_one()
    assert realized == pytest.approx(-3.7)
    assert acct.starting_capital + realized == pytest.approx(acct.trading_cash + acct.blocked + acct.reserve)
    last = ana_session.execute(select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == batch.id)
                               .order_by(CapitalLedgerEntry.ts.desc(), CapitalLedgerEntry.id.desc())).scalars().first()
    assert (last.trading_cash_after, last.blocked_after, last.reserve_after) == pytest.approx(
        (acct.trading_cash, acct.blocked, acct.reserve))
