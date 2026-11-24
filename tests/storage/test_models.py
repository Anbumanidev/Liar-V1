"""Storage layer smoke tests (Phase 1)."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from core.clock import IST
from storage.audit import audit, record_error
from storage.models import AuditLog, Base, ErrorLog, Order

EXPECTED_TABLES = {
    "batches", "settings_versions", "prompt_templates", "runs", "stocks_analyzed", "agent_reports",
    "debates", "decisions", "approval_queue", "orders", "positions", "trades", "shadow_trades",
    "corporate_action_adjustments", "capital_accounts", "capital_ledger", "daily_summary",
    "mode_states", "strategy_states", "control_commands", "engine_status", "market_events",
    "audit_log", "login_attempts", "errors", "notifications_log",
}


def test_all_tables_created(session_factory) -> None:  # type: ignore[no-untyped-def]
    engine = session_factory.kw["bind"]
    assert EXPECTED_TABLES <= set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES <= set(Base.metadata.tables)


def test_datetimes_round_trip_as_ist(session) -> None:  # type: ignore[no-untyped-def]
    ts = datetime(2026, 11, 24, 10, 0, tzinfo=IST)
    audit(session, "owner", "TEST", "x", {"a": 1}, clock=None)
    row = session.query(AuditLog).one()
    row.ts = ts
    session.commit()
    session.expire_all()
    again = session.query(AuditLog).one()
    assert again.ts.tzinfo is not None and again.ts == ts


def test_orders_must_be_paper(session) -> None:  # type: ignore[no-untyped-def]
    now = datetime(2026, 11, 24, 10, 0, tzinfo=IST)
    session.add(Order(batch_id=1, symbol="X", mode="INTRADAY", product="MIS", side="BUY", order_type="MARKET",
                      purpose="ENTRY", qty=1, status="PENDING", created_at=now, is_paper=False))
    with pytest.raises(IntegrityError):
        session.flush()


def test_record_error_never_raises(session) -> None:  # type: ignore[no-untyped-def]
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        record_error(session, "test", exc, {"k": 1})
    assert session.query(ErrorLog).count() == 1
    record_error(None, "test", "no session")
