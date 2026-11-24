"""Fixtures for dashboard tests: a temp-file SQLite DB (DATABASE_URL) seeded with realistic rows."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from config.loader import load_hard_caps
from config.store import ensure_seeded
from core.clock import IST
from storage.db import create_all_for_url
from storage.models import (
    AgentReportRow,
    ApprovalItem,
    Batch,
    CapitalAccount,
    CapitalLedgerEntry,
    ControlCommand,
    DebateRow,
    DecisionRow,
    EngineStatus,
    ModeState,
    NotificationLog,
    Position,
    PromptTemplate,
    StrategyState,
    Trade,
)


def _now() -> datetime:
    """Wall-clock IST (the dashboard uses SystemClock; test data must be relative to it)."""
    return datetime.now(IST)


@pytest.fixture
def dash_db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Temp-file SQLite URL exported as DATABASE_URL (+ runtime dir and auth file in tmp)."""
    db = tmp_path / "trading.db"
    url = f"sqlite:///{db.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("DASHBOARD_AUTH_FILE", str(tmp_path / "secrets" / "dashboard_auth.yaml"))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    return url


@pytest.fixture
def dash_factory(dash_db_url: str) -> sessionmaker[Session]:
    return create_all_for_url(dash_db_url)


def seed_dashboard_db(factory: sessionmaker[Session]) -> dict[str, int]:
    """Insert one of everything the control pages display. Returns ids."""
    now = _now()
    caps = load_hard_caps()
    with factory() as s:
        version = ensure_seeded(s, caps)
        old = Batch(name="old-batch", starting_capital=1000.0, status="CLOSED", created_at=now - timedelta(days=30),
                    ended_at=now - timedelta(days=2))
        cur = Batch(name="batch-1", starting_capital=1000.0, status="ACTIVE", created_at=now - timedelta(days=1),
                    started_at=now - timedelta(days=1), settings_version_id=version.id)
        s.add_all([old, cur])
        s.flush()
        s.add(CapitalAccount(batch_id=cur.id, starting_capital=1000.0, base_capital=1000.0, trading_cash=700.0,
                             blocked=300.0, reserve=50.0, reserve_peak=50.0, hwm=1050.0, sizing_capital=1000.0,
                             updated_at=now))
        s.add(CapitalLedgerEntry(batch_id=cur.id, ts=now, entry_type="INITIAL_CAPITAL", amount=1000.0,
                                 trading_cash_after=1000.0, blocked_after=0.0, reserve_after=0.0,
                                 sizing_capital_after=1000.0))
        s.add(EngineStatus(id=1, state="RUNNING", heartbeat_at=now, started_at=now - timedelta(hours=1), pid=1234,
                           activity="intraday cycle", batch_id=cur.id, data_feed_ok=True, llm_ok=False,
                           market_open=True))
        dec = DecisionRow(batch_id=cur.id, symbol="RELIANCE", mode="INTRADAY", strategy="orb", regime="TRENDING_UP",
                          confluence_score=78.0, grade="A", decision="BUY", direction="LONG", confidence=80,
                          entry_price=100.0, stop_loss=98.0, target_1=104.0, target_2=106.0,
                          meta_json={"reasoning": "Strong trend with volume"},
                          context_json={"confluence": {"score": 78.0, "grade": "A"}},
                          risk_approved=True, risk_json={"approved": True, "checks": []}, qty=5,
                          final_action="QUEUED", created_at=now)
        s.add(dec)
        s.flush()
        s.add(AgentReportRow(batch_id=cur.id, decision_id=dec.id, symbol="RELIANCE", mode="INTRADAY",
                             agent="technical", signal="BULLISH", confidence=75, status="OK", model="qwen3:4b",
                             report_json={"reasons": ["EMA stack"]}, created_at=now))
        s.add(DebateRow(batch_id=cur.id, decision_id=dec.id, symbol="RELIANCE", mode="INTRADAY", rounds=2,
                        transcript_json=[{"side": "bull", "text": "trend up"}, {"side": "bear", "text": "overbought"}],
                        status="OK", created_at=now))
        appr = ApprovalItem(batch_id=cur.id, decision_id=dec.id, symbol="RELIANCE", mode="INTRADAY",
                            direction="LONG", entry_type="LIMIT", entry_price=100.0, stop_loss=98.0, target_1=104.0,
                            target_2=106.0, qty=5, max_qty=8, confidence=80, grade="A", strategy="orb",
                            proposal_json={}, summary_json={"bull_points": ["trend"], "bear_points": ["rsi high"]},
                            status="PENDING", created_at=now, expires_at=now + timedelta(minutes=3))
        appr2 = ApprovalItem(batch_id=cur.id, decision_id=None, symbol="TCS", mode="SWING", direction="LONG",
                             entry_type="MARKET", entry_price=3000.0, stop_loss=2940.0, target_1=3120.0, qty=1,
                             max_qty=1, confidence=72, grade="B", proposal_json={}, status="PENDING",
                             created_at=now, expires_at=now + timedelta(hours=14))
        done = ApprovalItem(batch_id=cur.id, symbol="INFY", mode="INTRADAY", direction="SHORT", entry_type="MARKET",
                            entry_price=1500.0, stop_loss=1510.0, target_1=1480.0, qty=1, max_qty=1, confidence=70,
                            proposal_json={}, status="REJECTED", created_at=now - timedelta(hours=1),
                            expires_at=now - timedelta(minutes=50), decided_at=now - timedelta(minutes=55),
                            decided_by="admin", block_rule="RR_BELOW_MIN")
        s.add_all([appr, appr2, done])
        pos_i = Position(batch_id=cur.id, symbol="SBIN", mode="INTRADAY", product="MIS", direction="LONG",
                         status="OPEN", qty=4, initial_qty=4, avg_entry=800.0, stop_loss=790.0,
                         initial_stop_loss=790.0, target_1=820.0, target_2=830.0, opened_at=now - timedelta(hours=2),
                         margin_blocked=3200.0, entry_charges=1.2, last_price=805.0)
        pos_s = Position(batch_id=cur.id, symbol="ITC", mode="SWING", product="CNC", direction="LONG", status="OPEN",
                         qty=2, initial_qty=2, avg_entry=450.0, stop_loss=440.0, initial_stop_loss=440.0,
                         target_1=470.0, opened_at=now - timedelta(days=3), margin_blocked=900.0, max_holding_days=10,
                         holding_days=3, overnight=True)
        s.add_all([pos_i, pos_s])
        s.add(Trade(batch_id=cur.id, position_id=99, symbol="HDFCBANK", mode="INTRADAY", product="MIS",
                    direction="LONG", qty=1, entry_time=now - timedelta(hours=5), entry_price=1600.0,
                    exit_time=now - timedelta(hours=4), exit_price=1610.0, stop_loss=1590.0, exit_reason="TARGET",
                    gross_pnl=10.0, charges=1.0, net_pnl=9.0, r_multiple=1.0, source="SYSTEM", strategy="orb"))
        s.add(ModeState(batch_id=cur.id, mode="INTRADAY", paused=False, consecutive_losses=1, updated_at=now))
        s.add(ModeState(batch_id=cur.id, mode="SWING", paused=True, pause_reason="owner", updated_at=now))
        s.add(StrategyState(batch_id=cur.id, strategy="vwap_reversion", mode="INTRADAY", auto_disabled=True,
                            disabled_reason="negative expectancy", trades_count=31, expectancy_r=-0.2,
                            updated_at=now))
        s.add(ControlCommand(created_at=now - timedelta(minutes=5), command_type="MANUAL_TRADE",
                             payload_json={"symbol": "SBIN"}, status="REJECTED", requested_by="admin",
                             batch_id=cur.id, completed_at=now - timedelta(minutes=5),
                             result_json={"ok": False, "message": "R:R 1.2 < 1.5", "rule": "RR_BELOW_MIN"}))
        s.add(PromptTemplate(agent="technical", version=1, system_prompt="You are a technical analyst.",
                             user_template="Facts: {{ facts }}", created_at=now, author="system"))
        s.add(NotificationLog(ts=now, email_type="TEST", recipients="a@b.c", subject="[PAPER] test", status="SENT",
                              attempts=1))
        s.commit()
        return {"batch": cur.id, "old_batch": old.id, "approval": appr.id, "approval2": appr2.id,
                "decision": dec.id, "pos_intraday": pos_i.id, "pos_swing": pos_s.id}


@pytest.fixture
def seeded(dash_factory: sessionmaker[Session]) -> Iterator[dict[str, int]]:
    yield seed_dashboard_db(dash_factory)
