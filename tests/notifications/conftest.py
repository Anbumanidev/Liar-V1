"""Fixtures for the notifications tests: fake sender + realistic sample rows."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy.orm import Session, sessionmaker

from config.schema import Settings
from config.secrets import Secrets
from core.clock import IST, Clock
from notifications.email import STATUS_FAILED, STATUS_SENT, Attachment, SendResult
from notifications.service import Notifier
from storage.models import ApprovalItem, Order, Position, Trade


class FakeSender:
    """Records every send; returns a configurable status; can block or raise."""

    def __init__(self, status: str = STATUS_SENT, raise_exc: BaseException | None = None,
                 block: threading.Event | None = None) -> None:
        self.status = status
        self.raise_exc = raise_exc
        self.block = block
        self.started = threading.Event()
        self.calls: list[dict[str, Any]] = []
        self.thread_names: list[str] = []

    def send(self, subject: str, html: str, recipients: Sequence[str] | str | None = None,
             attachments: Sequence[Attachment] | None = None, text: str | None = None) -> SendResult:
        self.started.set()
        self.thread_names.append(threading.current_thread().name)
        if self.block is not None:
            self.block.wait(10)
        self.calls.append({"subject": subject, "html": html, "attachments": list(attachments or [])})
        if self.raise_exc is not None:
            raise self.raise_exc
        ok = self.status == STATUS_SENT
        return SendResult(ok=ok, attempts=1 if ok else 4, error=None if ok else "SMTPServerDisconnected: down",
                          status=self.status, recipients=["owner@example.com"], subject=f"[PAPER] {subject}")

    @property
    def last_html(self) -> str:
        return self.calls[-1]["html"]


@pytest.fixture
def sender_cls() -> type[FakeSender]:
    """The FakeSender class (for tests that need a custom instance)."""
    return FakeSender


@pytest.fixture
def fake_sender() -> FakeSender:
    return FakeSender()


@pytest.fixture
def failing_sender() -> FakeSender:
    return FakeSender(status=STATUS_FAILED)


@pytest.fixture
def secrets_ok() -> Secrets:
    return Secrets.model_construct(smtp_user="bot@gmail.com", smtp_app_password=SecretStr("abcdefghijklmnop"),
                                   alert_email_to="owner@example.com")


@pytest.fixture
def make_notifier(session_factory: sessionmaker[Session], settings: Settings, secrets_ok: Secrets,
                  sim_clock: Clock) -> Iterator[Callable[..., Notifier]]:
    created: list[Notifier] = []

    def _make(sender: Any = None, *, clock: Clock | None = None, async_send: bool = False,
              settings_: Settings | None = None, **kw: Any) -> Notifier:
        n = Notifier(session_factory, settings_ or settings, secrets_ok, clock or sim_clock, sender=sender,
                     async_send=async_send, **kw)
        created.append(n)
        return n

    yield _make
    for n in created:
        n.close(timeout=5)


@pytest.fixture
def approval_item() -> ApprovalItem:
    return ApprovalItem(
        id=42, batch_id=1, decision_id=7, symbol="RELIANCE", mode="SWING", direction="LONG", entry_type="LIMIT",
        entry_price=2450.5, stop_loss=2401.0, target_1=2549.5, target_2=2620.0, qty=3, max_qty=4, confidence=78,
        grade="A", strategy="pullback_ema21",
        proposal_json={"regime": "TRENDING_UP"},
        risk_json={"charges_estimate": {"brokerage": 0.0, "stt": 14.85, "exchange_txn": 0.48, "sebi": 0.01,
                                        "stamp_duty": 1.1, "gst": 0.09, "dp": 15.93}},
        summary_json={
            "bull_points": ["Higher highs on daily with rising volume", "Above EMA21 and VWAP",
                            "Sector (energy) leading NIFTY", "fourth point is dropped"],
            "debate": {"bear": {"top_arguments": ["RSI 71 near overbought", "<script>alert(1)</script> FII selling",
                                                  "Results in 9 days"]}},
            "reasoning": "Confluence 78 (A); bull case stronger after two debate rounds.",
        },
        status="PENDING", created_at=datetime(2026, 11, 24, 9, 15, tzinfo=IST),
        expires_at=datetime(2026, 11, 24, 9, 45, tzinfo=IST),
    )


@pytest.fixture
def position_row() -> Position:
    return Position(
        id=11, batch_id=1, symbol="TCS", mode="SWING", product="CNC", direction="LONG", status="OPEN", qty=2,
        initial_qty=2, avg_entry=3512.4, stop_loss=3440.0, initial_stop_loss=3440.0, target_1=3657.2,
        target_2=3730.0, trailing_rule="EMA20_DAILY", max_holding_days=10, opened_at=datetime(2026, 11, 24, 9, 20,
                                                                                               tzinfo=IST),
        strategy="breakout_52w", regime="TRENDING_UP", grade="A+", source="SYSTEM", margin_blocked=7024.8,
        entry_charges=8.62,
    )


@pytest.fixture
def order_row() -> Order:
    return Order(
        id=101, batch_id=1, position_id=11, symbol="TCS", mode="SWING", product="CNC", side="BUY",
        order_type="LIMIT", purpose="ENTRY", qty=2, limit_price=3515.0, status="FILLED",
        created_at=datetime(2026, 11, 24, 9, 16, tzinfo=IST), filled_at=datetime(2026, 11, 24, 9, 20, tzinfo=IST),
        fill_price=3512.4, slippage=0.0, is_paper=True,
    )


@pytest.fixture
def trade_row() -> Trade:
    return Trade(
        id=501, batch_id=1, position_id=11, symbol="TCS", mode="SWING", product="CNC", direction="LONG", qty=2,
        entry_time=datetime(2026, 11, 20, 9, 20, tzinfo=IST), entry_price=3512.4,
        exit_time=datetime(2026, 11, 24, 11, 35, tzinfo=IST), exit_price=3657.2, stop_loss=3440.0,
        target_1=3657.2, exit_reason="TARGET", is_partial=False, gross_pnl=289.6, charges=24.31, net_pnl=265.29,
        r_multiple=1.83, holding_days=3, source="SYSTEM", strategy="breakout_52w", grade="A+",
        regime="TRENDING_UP",
        charges_json={"entry": {"brokerage": 0.0, "stt": 7.02, "stamp_duty": 1.05, "gst": 0.05},
                      "exit": {"brokerage": 0.0, "stt": 7.31, "dp": 15.93, "gst": 2.87}},
    )


@pytest.fixture
def eod_report() -> dict[str, Any]:
    """A realistic report dict as produced by build_eod_report."""
    return {
        "batch_id": 1, "batch_name": "Batch Nov-26", "day": "2026-11-24", "generated_at": "2026-11-24T15:45:00+05:30",
        "empty": False,
        "modes": [
            {"mode": "INTRADAY", "trades": 2, "wins": 1, "losses": 1, "gross": 38.0, "charges": 8.0, "net": 30.0,
             "win_rate": 50.0},
            {"mode": "SWING", "trades": 1, "wins": 1, "losses": 0, "gross": 24.0, "charges": 4.0, "net": 20.0,
             "win_rate": 100.0},
        ],
        "totals": {"trades": 3, "legs": 4, "wins": 2, "losses": 1, "gross": 62.0, "charges": 12.0, "net": 50.0,
                   "win_rate": 66.67},
        "trades": [{"id": 1, "symbol": "RELIANCE", "mode": "INTRADAY", "exit_reason": "TARGET", "is_partial": False,
                    "net": 50.0},
                   {"id": 2, "symbol": "TCS", "mode": "INTRADAY", "exit_reason": "SL", "is_partial": False,
                    "net": -20.0}],
        "open_positions": [
            {"id": 20, "symbol": "HDFCBANK", "mode": "SWING", "product": "CNC", "direction": "LONG", "qty": 2,
             "avg_entry": 1500.0, "last_price": 1560.0, "price_missing": False, "unrealized_net": 119.5,
             "unrealized_pct": 3.98, "days_held": 3},
            {"id": 21, "symbol": "ITC", "mode": "POSITIONAL", "product": "CNC", "direction": "LONG", "qty": 1,
             "avg_entry": 400.0, "last_price": None, "price_missing": True, "unrealized_net": -0.4,
             "unrealized_pct": -0.1, "days_held": 0},
        ],
        "open_unrealized_total": 119.1,
        "capital": {"available": True, "starting_capital": 1000.0, "trading_cash": 700.0, "blocked": 300.0,
                    "trading_balance": 1000.0, "reserve": 15.0, "unrealized_pnl": 119.1, "total_equity": 1134.1,
                    "hwm": 1134.1, "drawdown_pct": 0.0, "tier": "NORMAL", "sizing_capital": 1000.0,
                    "mode_sizing": {"INTRADAY": 400.0, "SWING": 300.0}, "realized_today": 50.0,
                    "profit_locked": 15.0, "loss_absorbed": 0.0, "refilled": 0.0, "halted": False,
                    "eod_processed": True},
        "not_traded": [{"symbol": "SBIN", "mode": "INTRADAY", "reason": "Risk veto: RR_BELOW_MIN"},
                       {"symbol": "WIPRO", "mode": "SWING", "reason": "Rejected by owner: Too extended"}],
        "not_traded_total": 2,
        "approvals": {"total": 4, "approved": 1, "modified": 1, "rejected": 1, "timeout": 1, "pending": 0,
                      "items": []},
        "shadow": {"total": 2, "open": 1, "closed": 1, "target_hits": 1, "sl_hits": 0, "hypothetical_net": 25.0,
                   "by_source": [{"source": "USER", "count": 1, "closed": 1, "target_hits": 1, "sl_hits": 0,
                                  "hypothetical_net": 25.0},
                                 {"source": "TIMEOUT", "count": 1, "closed": 0, "target_hits": 0, "sl_hits": 0,
                                  "hypothetical_net": 0.0}],
                   "cumulative_by_source": [{"source": "USER", "closed": 2, "target_hits": 2, "sl_hits": 0,
                                             "hypothetical_net": 65.0}],
                   "items": []},
        "agent_failures": {"total": 2, "by_agent": [{"agent": "news_sentiment", "count": 2,
                                                      "symbols": ["SBIN", "TCS"], "last_error": "timeout"}]},
        "equity_series": [{"day": "2026-11-20", "equity": 1000.0}, {"day": "2026-11-23", "equity": 1010.0},
                          {"day": "2026-11-24", "equity": 1134.1}],
    }
