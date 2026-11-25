"""Shared helpers for the web dashboard tests: seed data, fake market data, context builder, login."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from config.loader import load_hard_caps
from config.schema import HardCaps
from config.store import ensure_seeded
from core.clock import IST, SimulatedClock
from core.types import Quote
from data.calendar import NSECalendar
from storage.models import (
    AgentReportRow,
    ApprovalItem,
    Batch,
    CapitalAccount,
    CapitalLedgerEntry,
    DebateRow,
    DecisionRow,
    EngineStatus,
    ErrorLog,
    Order,
    Position,
    StockAnalyzed,
    Trade,
)
from tests.helpers import make_daily_candles, make_intraday_candles, trading_days
from webui.context import WebContext

PASSWORD = "correct horse battery"
DAY = date(2026, 11, 25)                     # Wednesday (24 Nov is an NSE holiday)
NOW = datetime(2026, 11, 25, 14, 0, tzinfo=IST)
OPTIONAL_MODULES = ("capital.funding", "analytics.postmortem", "analytics.diagnostics")


def at(hh: int, mm: int, ss: int = 0, day: date = DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=IST)


# =========================================================================== fake market data


class FakeData:
    """Deterministic MarketDataService stand-in (5 trading days ending on DAY)."""

    FREQ = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}

    def __init__(self) -> None:
        self.days = trading_days(date(2026, 11, 19), 5)  # 19, 20, 23, 24, 25 Nov (weekdays)
        self.calls: list[tuple[str, str, int, Any]] = []
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}

    def frame(self, symbol: str, timeframe: str) -> pd.DataFrame:
        key = (symbol, timeframe)
        if key not in self._frames:
            seed = sum(ord(c) for c in symbol)
            if timeframe in self.FREQ:
                df = make_intraday_candles(self.days, freq_minutes=self.FREQ[timeframe], start_price=800.0, seed=seed)
            elif timeframe == "1d":
                df = make_daily_candles(DAY, n=120, start_price=800.0, seed=seed)
            else:  # 1wk
                d = make_daily_candles(DAY, n=400, start_price=800.0, seed=seed)
                df = d.iloc[::5]
            self._frames[key] = df
        return self._frames[key]

    def candles(self, symbol: str, timeframe: str, lookback_bars: int = 300, start: datetime | None = None
                ) -> pd.DataFrame:
        self.calls.append((symbol, str(timeframe), lookback_bars, start))
        df = self.frame(symbol, str(timeframe))
        if start is not None:
            return df.loc[df.index >= pd.Timestamp(start)].copy()
        return df.tail(lookback_bars).copy()

    def quote(self, symbol: str) -> Quote | None:
        return self.quotes([symbol]).get(symbol)

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        prices = {"SBIN": (812.0, 800.0), "ITC": (309.0, 310.0), "BEL": (401.0, 400.0), "TCS": (3000.0, 2990.0)}
        out = {}
        for s in symbols:
            if s in prices:
                ltp, prev = prices[s]
                out[s] = Quote(symbol=s, ltp=ltp, prev_close=prev, timestamp=NOW - timedelta(seconds=2), source="fake")
        return out

    def healthy(self) -> bool:
        return True


# =========================================================================== seed


def seed(factory: sessionmaker[Session]) -> dict[str, int]:
    caps = load_hard_caps()
    with factory() as s:
        version = ensure_seeded(s, caps)
        old = Batch(name="Batch 1", starting_capital=1000.0, status="CLOSED", created_at=at(9, 0, day=date(2026, 11, 2)),
                    started_at=at(9, 0, day=date(2026, 11, 2)), ended_at=at(15, 40, day=date(2026, 11, 20)))
        cur = Batch(name="Batch 2", starting_capital=1000.0, status="ACTIVE", created_at=at(9, 0, day=date(2026, 11, 23)),
                    started_at=at(9, 0, day=date(2026, 11, 23)), settings_version_id=version.id)
        s.add_all([old, cur])
        s.flush()
        s.add(CapitalAccount(batch_id=old.id, starting_capital=1000.0, base_capital=1000.0, trading_cash=995.28,
                             blocked=0.0, reserve=0.0, reserve_peak=0.0, hwm=1003.0, sizing_capital=1000.0,
                             updated_at=at(15, 40, day=date(2026, 11, 20))))
        s.add(CapitalAccount(batch_id=cur.id, starting_capital=1000.0, base_capital=1000.0, trading_cash=600.0,
                             blocked=400.0, reserve=10.0, reserve_peak=10.0, hwm=1015.0, sizing_capital=1000.0,
                             tier="NORMAL", updated_at=at(13, 30)))
        s.add_all([
            CapitalLedgerEntry(batch_id=cur.id, ts=at(9, 0, day=date(2026, 11, 23)), entry_type="INITIAL_CAPITAL",
                               amount=1000.0, trading_cash_after=1000.0, blocked_after=0.0, reserve_after=0.0,
                               sizing_capital_after=1000.0),
            CapitalLedgerEntry(batch_id=cur.id, ts=at(11, 0), entry_type="REALIZED_PNL", amount=12.0,
                               trading_cash_after=1012.0, blocked_after=0.0, reserve_after=0.0,
                               sizing_capital_after=1000.0),
            CapitalLedgerEntry(batch_id=cur.id, ts=at(12, 45), entry_type="REALIZED_PNL", amount=-2.0,
                               trading_cash_after=1010.0, blocked_after=0.0, reserve_after=0.0,
                               sizing_capital_after=1000.0),
        ])
        s.add(EngineStatus(id=1, state="RUNNING", heartbeat_at=NOW - timedelta(seconds=5), started_at=at(9, 0), pid=1,
                           activity="intraday cycle", next_cycle_at=NOW + timedelta(minutes=5), batch_id=cur.id,
                           data_feed_ok=True, llm_ok=True, market_open=True, status_json={"state": "RUNNING"}))

        # decisions -------------------------------------------------------------------
        ctx_exec = {"filtered_reason": None,
                    "signals": [{"strategy": "orb", "direction": "LONG", "entry_price": 300.0, "stop_loss": 298.0,
                                 "target_1": 304.0, "target_2": 306.0, "strength": 80, "notes": ["opening range"]}],
                    "confluence": {"score": 74.0, "grade": "A", "components": {"trend": 20.0, "volume": 12.5},
                                   "notes": ["direction LONG"]},
                    "sl_candidates": [{"method": "ATR", "price": 298.0, "distance": 2.0, "distance_atr": 1.0,
                                       "note": "1x ATR"}],
                    "vote": {"score": 30.0, "bullish": 4, "bearish": 1, "neutral": 3, "failed": 1, "skipped": 0,
                             "tradeable": True, "contributions": {"technical": 1.2, "price_action": 0.8},
                             "failed_agents": ["news_sentiment"]},
                    "market": {"nifty_trend": "BULLISH", "india_vix": 14.2, "sector_strength": {"IT": 1.0}}}
        risk_ok = {"approved": True, "rule": None, "message": "approved", "qty": 2, "max_qty": 2,
                   "charges_estimate": {"brokerage": 0.1, "gst": 0.02},
                   "checks": [{"rule": "PAPER_ONLY", "passed": True, "detail": "paper trading only"},
                              {"rule": "RR_BELOW_MIN", "passed": True, "detail": "R:R 2.0 >= 1.5"}]}
        d_exec = DecisionRow(batch_id=cur.id, symbol="ITC", mode="INTRADAY", strategy="orb", regime="TRENDING_UP",
                             confluence_score=74.0, grade="A", decision="BUY", direction="LONG", confidence=82,
                             entry_price=300.0, stop_loss=298.0, target_1=304.0, target_2=306.0,
                             meta_json={"decision": "BUY", "confidence": 82, "reasons_for": ["trend up"],
                                        "reasons_against_considered": ["VIX rising"], "why_not": None},
                             context_json=ctx_exec, risk_approved=True, risk_json=risk_ok, qty=2,
                             final_action="EXECUTED", latency_ms=42000, created_at=at(10, 0))
        d_rej = DecisionRow(batch_id=cur.id, symbol="BEL", mode="INTRADAY", strategy="vwap", decision="BUY",
                            direction="LONG", confidence=75, confluence_score=65.0, grade="B", entry_price=402.0,
                            stop_loss=401.0, target_1=404.0, risk_approved=False, risk_rule="MAX_TRADES_PER_DAY",
                            risk_json={"approved": False, "rule": "MAX_TRADES_PER_DAY",
                                       "message": "MAX_TRADES_PER_DAY: 3/3 INTRADAY trades today",
                                       "checks": [{"rule": "MAX_TRADES_PER_DAY", "passed": False,
                                                   "detail": "3/3 INTRADAY trades today"}]},
                            context_json={"filtered_reason": None}, final_action="RISK_REJECTED",
                            created_at=at(12, 30))
        d_filt = DecisionRow(batch_id=cur.id, symbol="TCS", mode="INTRADAY", decision="NO_TRADE", confidence=0,
                             confluence_score=41.0, grade="C",
                             context_json={"filtered_reason": "confluence 41.0 < 60.0 (grade C)"},
                             final_action="FILTERED", created_at=at(12, 40))
        d_no = DecisionRow(batch_id=cur.id, symbol="INFY", mode="INTRADAY", decision="NO_TRADE", confidence=60,
                           confluence_score=66.0, grade="B", meta_json={"why_not": "charges eat the edge"},
                           context_json={"filtered_reason": None}, final_action="NO_TRADE", created_at=at(12, 50))
        d_q = DecisionRow(batch_id=cur.id, symbol="TCS", mode="SWING", decision="BUY", direction="LONG", confidence=80,
                          confluence_score=72.0, grade="A", entry_price=3000.0, stop_loss=2940.0, target_1=3120.0,
                          context_json={"filtered_reason": None}, risk_json=risk_ok, final_action="QUEUED",
                          created_at=at(13, 0))
        d_bel = DecisionRow(batch_id=cur.id, symbol="BEL", mode="INTRADAY", strategy="ema", decision="BUY",
                            direction="LONG", confidence=78, confluence_score=70.0, grade="B", entry_price=402.5,
                            stop_loss=401.5, target_1=405.0, final_action="EXECUTED", created_at=at(11, 55))
        s.add_all([d_exec, d_bel, d_rej, d_filt, d_no, d_q])
        s.flush()
        s.add(AgentReportRow(batch_id=cur.id, decision_id=d_exec.id, symbol="ITC", mode="INTRADAY", agent="technical",
                             signal="BULLISH", confidence=75, status="OK", model="qwen3:4b", latency_ms=12000,
                             report_json={"key_findings": ["EMA stack up", "RSI 61"], "data_used": {"big": "x" * 50}},
                             created_at=at(10, 0)))
        s.add(AgentReportRow(batch_id=cur.id, decision_id=d_exec.id, symbol="ITC", mode="INTRADAY",
                             agent="news_sentiment", signal="NEUTRAL", confidence=0, status="FAILED",
                             error="timeout", report_json={}, created_at=at(10, 0)))
        s.add(DebateRow(batch_id=cur.id, decision_id=d_exec.id, symbol="ITC", mode="INTRADAY", rounds=2,
                        bull_json={"stance": "Go long", "top_arguments": ["trend"], "conviction": 80},
                        bear_json={"stance": "Wait", "top_arguments": ["VIX"], "conviction": 55},
                        transcript_json=[{"round": 0, "speaker": "bull", "phase": "opening", "stance": "Go long",
                                          "arguments": ["trend"], "conviction": 80}],
                        status="OK", created_at=at(10, 0)))
        s.add(StockAnalyzed(batch_id=cur.id, symbol="TCS", mode="INTRADAY", as_of=at(12, 40), stage="ANALYZED",
                            included=False, reason="confluence"))

        # positions / orders / trades --------------------------------------------------
        itc = Position(batch_id=cur.id, symbol="ITC", mode="INTRADAY", product="MIS", direction="LONG",
                       status="CLOSED", qty=0, initial_qty=2, avg_entry=300.0, stop_loss=300.0, initial_stop_loss=298.0,
                       target_1=304.0, target_2=306.0, opened_at=at(10, 2), closed_at=at(11, 0), strategy="orb",
                       grade="A", margin_blocked=0.0, decision_id=d_exec.id, partial_done=True, breakeven_moved=True,
                       sl_history_json=[{"ts": at(10, 2).isoformat(), "old": None, "new": 298.0, "by": "ENTRY"},
                                        {"ts": at(10, 40).isoformat(), "old": 298.0, "new": 300.0,
                                         "by": "BREAKEVEN"}])
        bel = Position(batch_id=cur.id, symbol="BEL", mode="INTRADAY", product="MIS", direction="LONG",
                       status="CLOSED", qty=0, initial_qty=1, avg_entry=402.5, stop_loss=401.5,
                       initial_stop_loss=401.5, target_1=405.0, opened_at=at(12, 0), closed_at=at(12, 45),
                       strategy="ema", grade="B", margin_blocked=0.0, decision_id=d_bel.id)
        sbin = Position(batch_id=cur.id, symbol="SBIN", mode="INTRADAY", product="MIS", direction="LONG",
                        status="OPEN", qty=4, initial_qty=4, avg_entry=800.0, stop_loss=790.0, initial_stop_loss=790.0,
                        target_1=820.0, target_2=830.0, opened_at=at(13, 10), margin_blocked=400.0,
                        entry_charges=1.2, last_price=805.0, last_price_at=at(13, 55), strategy="orb", grade="A")
        s.add_all([itc, bel, sbin])
        s.flush()
        s.add_all([
            Order(batch_id=cur.id, position_id=itc.id, decision_id=d_exec.id, symbol="ITC", mode="INTRADAY",
                  product="MIS", side="BUY", order_type="MARKET", purpose="ENTRY", qty=2, status="FILLED",
                  created_at=at(10, 1), filled_at=at(10, 2), fill_price=300.0, slippage=0.1,
                  charges_json={"total": 0.12}),
            Order(batch_id=cur.id, position_id=itc.id, symbol="ITC", mode="INTRADAY", product="MIS", side="SELL",
                  order_type="MARKET", purpose="PARTIAL_EXIT", qty=1, status="FILLED", created_at=at(10, 30),
                  filled_at=at(10, 30), fill_price=304.0, reason="TARGET"),
            Order(batch_id=cur.id, position_id=itc.id, symbol="ITC", mode="INTRADAY", product="MIS", side="SELL",
                  order_type="MARKET", purpose="EXIT", qty=1, status="FILLED", created_at=at(11, 0),
                  filled_at=at(11, 0), fill_price=309.0, reason="TRAIL"),
            Order(batch_id=cur.id, position_id=bel.id, decision_id=d_bel.id, symbol="BEL", mode="INTRADAY",
                  product="MIS", side="BUY", order_type="MARKET", purpose="ENTRY", qty=1, status="FILLED",
                  created_at=at(11, 58), filled_at=at(12, 0), fill_price=402.5),
            Order(batch_id=cur.id, position_id=bel.id, symbol="BEL", mode="INTRADAY", product="MIS", side="SELL",
                  order_type="MARKET", purpose="EXIT", qty=1, status="FILLED", created_at=at(12, 45),
                  filled_at=at(12, 45), fill_price=401.0, reason="SL"),
            Order(batch_id=cur.id, position_id=sbin.id, symbol="SBIN", mode="INTRADAY", product="MIS", side="BUY",
                  order_type="MARKET", purpose="ENTRY", qty=4, status="FILLED", created_at=at(13, 9),
                  filled_at=at(13, 10), fill_price=800.0),
            Order(batch_id=cur.id, symbol="COALINDIA", mode="INTRADAY", product="MIS", side="BUY", order_type="MARKET",
                  purpose="ENTRY", qty=1, status="REJECTED", created_at=at(13, 20),
                  reason="fill is through the stop-loss; entry skipped"),
        ])
        s.add_all([
            Trade(batch_id=cur.id, position_id=itc.id, symbol="ITC", mode="INTRADAY", product="MIS", direction="LONG",
                  qty=1, entry_time=at(10, 2), entry_price=300.0, exit_time=at(10, 30), exit_price=304.0,
                  stop_loss=298.0, target_1=304.0, target_2=306.0, exit_reason="TARGET", is_partial=True,
                  gross_pnl=4.0, charges=0.2, net_pnl=3.8, r_multiple=2.0, source="SYSTEM", strategy="orb", grade="A",
                  holding_minutes=28, holding_days=0, decision_id=d_exec.id),
            Trade(batch_id=cur.id, position_id=itc.id, symbol="ITC", mode="INTRADAY", product="MIS", direction="LONG",
                  qty=1, entry_time=at(10, 2), entry_price=300.0, exit_time=at(11, 0), exit_price=309.0,
                  stop_loss=298.0, target_1=304.0, target_2=306.0, exit_reason="TRAIL", is_partial=False,
                  gross_pnl=9.0, charges=0.8, net_pnl=8.2, r_multiple=4.5, source="SYSTEM", strategy="orb", grade="A",
                  holding_minutes=58, holding_days=0, decision_id=d_exec.id),
            Trade(batch_id=cur.id, position_id=bel.id, symbol="BEL", mode="INTRADAY", product="MIS", direction="LONG",
                  qty=1, entry_time=at(12, 0), entry_price=402.5, exit_time=at(12, 45), exit_price=401.0,
                  stop_loss=401.5, target_1=405.0, exit_reason="SL", gross_pnl=-1.5, charges=0.3, net_pnl=-1.8,
                  r_multiple=-1.5, source="SYSTEM", strategy="ema", grade="B", holding_minutes=45, holding_days=0,
                  decision_id=d_bel.id),
            # an older trade on another day (journal / date filters)
            Trade(batch_id=old.id, position_id=900, symbol="HDFCBANK", mode="SWING", product="CNC", direction="LONG",
                  qty=1, entry_time=at(10, 0, day=date(2026, 11, 10)), entry_price=1600.0,
                  exit_time=at(14, 0, day=date(2026, 11, 12)), exit_price=1590.0, stop_loss=1580.0, target_1=1640.0,
                  exit_reason="SL", gross_pnl=-10.0, charges=2.0, net_pnl=-12.0, r_multiple=-0.5, source="SYSTEM",
                  strategy="breakout", holding_minutes=3120, holding_days=2),
        ])
        appr = ApprovalItem(batch_id=cur.id, decision_id=d_q.id, symbol="TCS", mode="SWING", direction="LONG",
                            entry_type="LIMIT", entry_price=3000.0, stop_loss=2940.0, target_1=3120.0, qty=1,
                            max_qty=1, confidence=80, grade="A", strategy="breakout", proposal_json={},
                            summary_json={"bull_points": ["breakout on volume"], "bear_points": ["near resistance"],
                                          "tax": {"head": "STCG", "rate_pct": 20.8, "breakeven_move_pct": 0.3,
                                                  "scenarios": [{"label": "Target 1", "exit_price": 3120.0,
                                                                 "gross": 120.0, "charges": 3.1, "tax": 24.3,
                                                                 "net_after_tax": 92.6}]}},
                            status="PENDING", created_at=at(13, 0), expires_at=NOW + timedelta(hours=3))
        done = ApprovalItem(batch_id=cur.id, symbol="INFY", mode="INTRADAY", direction="LONG", entry_type="MARKET",
                            entry_price=1500.0, stop_loss=1490.0, target_1=1520.0, qty=1, max_qty=1, confidence=70,
                            proposal_json={}, status="REJECTED", created_at=at(11, 0), expires_at=at(11, 3),
                            decided_at=at(11, 1), decided_by="admin")
        s.add_all([appr, done])
        s.add(ErrorLog(ts=at(13, 5), component="data", severity="WARNING", message="candles BEL 1m: empty"))
        s.add(ErrorLog(ts=at(13, 6), component="data", severity="WARNING", message="candles BEL 1m: empty"))
        s.commit()
        return {"batch": cur.id, "old_batch": old.id, "itc": itc.id, "bel": bel.id, "sbin": sbin.id,
                "d_exec": d_exec.id, "d_rej": d_rej.id, "approval": appr.id, "approval_done": done.id,
                "settings_version": version.id}


def make_ctx(env: dict[str, Any], *, caps: HardCaps | None = None, clock: SimulatedClock | None = None,
             data: Any = None, **kw: Any) -> WebContext:
    opts: dict[str, Any] = {"allowed_hosts": ["testserver", "127.0.0.1", "localhost"], "stream_interval": 0.01,
                            "stream_max_iterations": 2, "quote_wait": 2.0}
    opts.update(kw)
    return WebContext(
        session_factory=env["factory"], caps=caps or load_hard_caps(),
        clock=clock or SimulatedClock(NOW), log_dir=env["log_dir"], auth_file=env["auth_file"],
        secret_key_path=env["tmp"] / "runtime" / "web_secret.key", runtime_dir=env["tmp"] / "runtime",
        data=data if data is not None else FakeData(), calendar=NSECalendar(), **opts)


def login(client: TestClient, password: str = PASSWORD) -> str:
    r = client.post("/api/login", json={"username": "admin", "password": password})
    assert r.status_code == 200, r.text
    csrf = r.json()["csrf"]
    client.headers["X-CSRF-Token"] = csrf
    return csrf


