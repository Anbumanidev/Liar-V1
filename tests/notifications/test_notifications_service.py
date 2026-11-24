"""Notifier: toggles, quiet hours (overnight + URGENT bypass), logging, async, never-raise."""

from __future__ import annotations

import threading
import time as time_mod
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from config.schema import Settings
from core.clock import IST, SimulatedClock
from core.enums import EmailType
from core.logging import read_events
from notifications.email import STATUS_FAILED, STATUS_SENT, STATUS_SKIPPED, EmailSender
from notifications.service import (
    STATUS_QUEUED,
    Notifier,
    charges_breakdown,
    extract_points,
    in_quiet_hours,
    reward_risk,
)
from notifications.templating import build_environment
from storage.models import NotificationLog

DAY = date(2026, 11, 24)


def logs(session_factory: sessionmaker[Session]) -> list[NotificationLog]:
    with session_factory() as s:
        return list(s.scalars(select(NotificationLog).order_by(NotificationLog.id)).all())


def clock_at(hh: int, mm: int = 0, day: int = 24) -> SimulatedClock:
    return SimulatedClock(datetime(2026, 11, day, hh, mm, tzinfo=IST))


def with_quiet(settings: Settings, start: str | None, end: str | None, urgent_bypass: bool = True) -> Settings:
    s = settings.model_copy(deep=True)
    s.notifications.quiet_hours_start = start
    s.notifications.quiet_hours_end = end
    s.notifications.urgent_ignores_quiet_hours = urgent_bypass
    return s


def with_disabled(settings: Settings, *types: EmailType) -> Settings:
    s = settings.model_copy(deep=True)
    enabled = dict(s.notifications.enabled_types)
    for t in types:
        enabled[t] = False
    s.notifications.enabled_types = enabled
    return s


# --------------------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(
    ("start", "end", "now", "expected"),
    [
        ("22:00", "07:00", time(23, 0), True),
        ("22:00", "07:00", time(22, 0), True),
        ("22:00", "07:00", time(0, 30), True),
        ("22:00", "07:00", time(6, 59), True),
        ("22:00", "07:00", time(7, 0), False),
        ("22:00", "07:00", time(12, 0), False),
        ("22:00", "07:00", time(21, 59), False),
        ("12:00", "13:00", time(12, 30), True),
        ("12:00", "13:00", time(13, 0), False),
        ("12:00", "13:00", time(11, 59), False),
        (None, "07:00", time(3, 0), False),
        ("22:00", None, time(23, 0), False),
        ("09:00", "09:00", time(9, 0), False),
        ("25:99", "07:00", time(3, 0), False),
    ],
)
def test_in_quiet_hours(start: str | None, end: str | None, now: time, expected: bool) -> None:
    assert in_quiet_hours(now, start, end) is expected


def test_in_quiet_hours_converts_to_ist() -> None:
    utc_now = datetime(2026, 11, 24, 17, 0, tzinfo=__import__("datetime").timezone.utc)  # 22:30 IST
    assert in_quiet_hours(utc_now, "22:00", "07:00") is True


def test_reward_risk_and_points_and_charges() -> None:
    assert reward_risk(100.0, 95.0, 110.0) == 2.0
    assert reward_risk(100.0, 100.0, 110.0) is None
    assert reward_risk(None, 95.0, 110.0) is None
    assert extract_points({"top_bull": "single point"}, "bull") == ["single point"]
    assert extract_points({"bear": [{"text": "a"}, {"point": "b"}, "", "c", "d"]}, "bear") == ["a", "b", "c"]
    assert extract_points({"bull": {"top_arguments": ["x"]}}, "bull") == ["x"]
    assert extract_points(None, "bull") == []
    rows = dict(charges_breakdown({"entry": {"stt": 1.0, "gst": 0.1}, "exit": {"stt": 2.0, "total": 9.9}}))
    assert rows == {"STT": 3.0, "GST": 0.1}
    assert charges_breakdown(None) == []


# --------------------------------------------------------------------------- gating


def test_sent_is_logged_in_db_and_jsonl(make_notifier: Callable[..., Notifier], fake_sender: Any,
                                        session_factory: sessionmaker[Session], approval_item: Any) -> None:
    n = make_notifier(fake_sender)
    out = n.pending_approval(approval_item)
    assert out.status == STATUS_SENT and out.ok and out.attempts == 1
    assert len(fake_sender.calls) == 1
    rows = logs(session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "SENT" and row.email_type == "PENDING_APPROVAL" and row.attempts == 1
    assert row.recipients == "owner@example.com" and row.batch_id == 1
    assert row.subject.startswith("[PAPER] APPROVAL NEEDED: BUY RELIANCE (SWING)")
    events = read_events(DAY, "notifications")
    assert events[-1]["status"] == "SENT" and events[-1]["email_type"] == "PENDING_APPROVAL"


def test_disabled_type_is_skipped_and_logged(make_notifier: Callable[..., Notifier], fake_sender: Any,
                                             settings: Settings, session_factory: sessionmaker[Session],
                                             position_row: Any, order_row: Any, trade_row: Any) -> None:
    n = make_notifier(fake_sender, settings_=with_disabled(settings, EmailType.ENTRY))
    out = n.entry(position_row, order_row)
    assert out.status == STATUS_SKIPPED and "disabled" in (out.reason or "")
    assert fake_sender.calls == []
    assert n.exit(trade_row).status == STATUS_SENT  # other types unaffected
    rows = logs(session_factory)
    assert [(r.email_type, r.status) for r in rows] == [("ENTRY", "SKIPPED"), ("EXIT", "SENT")]
    assert "disabled" in (rows[0].error or "")


def test_quiet_hours_overnight_skip_and_urgent_bypass(make_notifier: Callable[..., Notifier], fake_sender: Any,
                                                      settings: Settings, trade_row: Any,
                                                      eod_report: dict[str, Any]) -> None:
    quiet = with_quiet(settings, "22:00", "07:00")
    n = make_notifier(fake_sender, clock=clock_at(23, 30), settings_=quiet)
    assert n.is_quiet_now()
    assert n.exit(trade_row).status == STATUS_SKIPPED
    assert n.eod(eod_report).status == STATUS_SKIPPED
    assert n.swing_scan([]).status == STATUS_SKIPPED
    out = n.urgent("HALT", "Trailing floor hit")
    assert out.status == STATUS_SENT
    assert n.test().status == STATUS_SENT  # explicit owner action
    assert [c["subject"] for c in fake_sender.calls] == ["URGENT: HALT", "Test e-mail"]


def test_quiet_hours_early_morning_side(make_notifier: Callable[..., Notifier], fake_sender: Any,
                                        settings: Settings, trade_row: Any) -> None:
    quiet = with_quiet(settings, "22:00", "07:00")
    assert make_notifier(fake_sender, clock=clock_at(6, 45), settings_=quiet).exit(trade_row).status == STATUS_SKIPPED
    assert make_notifier(fake_sender, clock=clock_at(7, 0), settings_=quiet).exit(trade_row).status == STATUS_SENT


def test_urgent_respects_quiet_hours_when_bypass_disabled(make_notifier: Callable[..., Notifier], fake_sender: Any,
                                                          settings: Settings) -> None:
    quiet = with_quiet(settings, "22:00", "07:00", urgent_bypass=False)
    n = make_notifier(fake_sender, clock=clock_at(23, 0), settings_=quiet)
    out = n.urgent("Data feed down", "No quotes for 5 minutes")
    assert out.status == STATUS_SKIPPED and "quiet hours" in (out.reason or "")


def test_disabled_urgent_is_skipped_even_outside_quiet_hours(make_notifier: Callable[..., Notifier],
                                                            fake_sender: Any, settings: Settings) -> None:
    n = make_notifier(fake_sender, settings_=with_disabled(settings, EmailType.URGENT))
    assert n.urgent("HALT", "x").status == STATUS_SKIPPED


def test_urgent_dedupe_window(make_notifier: Callable[..., Notifier], fake_sender: Any) -> None:
    clock = clock_at(10, 0)
    n = make_notifier(fake_sender, clock=clock, urgent_dedupe_minutes=15)
    assert n.urgent("Data feed down", "angelone stale 120s").status == STATUS_SENT
    dup = n.urgent("Data feed down", "angelone stale 120s")
    assert dup.status == STATUS_SKIPPED and "duplicate" in (dup.reason or "")
    assert n.urgent("Data feed down", "upstox stale 60s").status == STATUS_SENT
    clock.advance(timedelta(minutes=16))
    assert n.urgent("Data feed down", "angelone stale 120s").status == STATUS_SENT
    assert len(fake_sender.calls) == 3


def test_failed_send_is_logged(make_notifier: Callable[..., Notifier], failing_sender: Any,
                               session_factory: sessionmaker[Session], trade_row: Any) -> None:
    n = make_notifier(failing_sender)
    out = n.exit(trade_row)
    assert out.status == STATUS_FAILED and not out.ok and out.attempts == 4
    row = logs(session_factory)[0]
    assert row.status == "FAILED" and row.attempts == 4 and "SMTPServerDisconnected" in (row.error or "")


def test_sender_exception_is_contained(sender_cls: Any, make_notifier: Callable[..., Notifier], session_factory: sessionmaker[Session],
                                       trade_row: Any) -> None:
    n = make_notifier(sender_cls(raise_exc=RuntimeError("socket exploded")))
    out = n.exit(trade_row)
    assert out.status == STATUS_FAILED and "socket exploded" in (out.reason or "")
    assert logs(session_factory)[0].status == "FAILED"


def test_bad_input_never_raises(make_notifier: Callable[..., Notifier], fake_sender: Any, approval_item: Any,
                                session_factory: sessionmaker[Session]) -> None:
    approval_item.qty = "not-a-number"
    n = make_notifier(fake_sender)
    out = n.pending_approval(approval_item)
    assert out.status == STATUS_FAILED and "internal error" in (out.reason or "")
    assert fake_sender.calls == []
    assert logs(session_factory)[0].status == "FAILED"


def test_db_failure_does_not_break_sending(settings: Settings, secrets_ok: Any, sim_clock: SimulatedClock,
                                           fake_sender: Any, trade_row: Any) -> None:
    def broken_factory() -> Session:
        raise RuntimeError("database is locked")

    n = Notifier(broken_factory, settings, secrets_ok, sim_clock, sender=fake_sender, async_send=False)  # type: ignore[arg-type]
    assert n.exit(trade_row).status == STATUS_SENT


def test_template_error_falls_back_to_raw_body(make_notifier: Callable[..., Notifier], fake_sender: Any,
                                               tmp_path: Path) -> None:
    (tmp_path / "urgent.html.j2").write_text("{{ this_is_undefined.attr }}", encoding="utf-8")
    n = make_notifier(fake_sender, env=build_environment(tmp_path))
    out = n.urgent("HALT", "Equity <below> floor", {"equity": 905.25})
    assert out.status == STATUS_SENT
    html = fake_sender.last_html
    assert "PAPER TRADING" in html and "template failed" in html
    assert "Equity &lt;below&gt; floor" in html  # escaped


def test_chart_failure_still_sends_eod(make_notifier: Callable[..., Notifier], fake_sender: Any,
                                       eod_report: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import notifications.service as svc

    def boom(*a: Any, **k: Any) -> bytes:
        raise RuntimeError("matplotlib broke")

    monkeypatch.setattr(svc, "equity_chart_png", boom)
    out = make_notifier(fake_sender).eod(eod_report)
    assert out.status == STATUS_SENT
    assert fake_sender.calls[0]["attachments"] == []
    assert "Equity chart unavailable" in fake_sender.last_html


# --------------------------------------------------------------------------- async


def test_async_send_does_not_block_caller(sender_cls: Any, make_notifier: Callable[..., Notifier],
                                          session_factory: sessionmaker[Session], trade_row: Any) -> None:
    gate = threading.Event()
    sender = sender_cls(block=gate)
    n = make_notifier(sender, async_send=True)
    t0 = time_mod.monotonic()
    out = n.exit(trade_row)
    elapsed = time_mod.monotonic() - t0
    assert out.status == STATUS_QUEUED and out.ok
    assert elapsed < 1.0  # sender is blocked, caller is not
    assert sender.started.wait(5)
    assert n.pending == 1
    assert n.flush(timeout=0.05) is False  # still blocked
    gate.set()
    assert n.flush(timeout=5) is True
    assert sender.thread_names == ["notifier-email"]
    rows = logs(session_factory)
    assert [(r.email_type, r.status) for r in rows] == [("EXIT", "SENT")]


def test_async_skip_is_logged_by_worker(make_notifier: Callable[..., Notifier], fake_sender: Any,
                                        settings: Settings, session_factory: sessionmaker[Session],
                                        trade_row: Any) -> None:
    n = make_notifier(fake_sender, async_send=True, settings_=with_disabled(settings, EmailType.EXIT))
    assert n.exit(trade_row).status == STATUS_SKIPPED
    assert n.flush(5)
    assert [r.status for r in logs(session_factory)] == ["SKIPPED"]


def test_async_queue_full_drops_and_logs(sender_cls: Any, make_notifier: Callable[..., Notifier],
                                         session_factory: sessionmaker[Session], trade_row: Any) -> None:
    gate = threading.Event()
    sender = sender_cls(block=gate)
    n = make_notifier(sender, async_send=True, queue_size=1)
    assert n.exit(trade_row).status == STATUS_QUEUED
    assert sender.started.wait(5)  # worker busy with #1
    assert n.exit(trade_row).status == STATUS_QUEUED  # fills the queue
    dropped = n.exit(trade_row)
    assert dropped.status == STATUS_FAILED and "queue full" in (dropped.reason or "")
    gate.set()
    assert n.flush(5)
    statuses = sorted(r.status for r in logs(session_factory))
    assert statuses == ["FAILED", "SENT", "SENT"]


def test_close_is_idempotent_and_worker_restarts(make_notifier: Callable[..., Notifier], fake_sender: Any,
                                                 trade_row: Any) -> None:
    n = make_notifier(fake_sender, async_send=True)
    n.exit(trade_row)
    n.close(timeout=5)
    n.close(timeout=5)
    assert len(fake_sender.calls) == 1
    n.exit(trade_row)  # a new worker is started on demand
    assert n.flush(5) and len(fake_sender.calls) == 2


# --------------------------------------------------------------------------- settings / default sender


def test_default_sender_follows_settings_reload(session_factory: sessionmaker[Session], settings: Settings,
                                                secrets_ok: Any, sim_clock: SimulatedClock) -> None:
    n = Notifier(session_factory, settings, secrets_ok, sim_clock, async_send=False)
    assert isinstance(n.sender, EmailSender)
    new = settings.model_copy(deep=True)
    new.notifications.subject_prefix = "[PAPER-TEST]"
    n.update_settings(new)
    assert n.sender.cfg.subject_prefix == "[PAPER-TEST]"  # type: ignore[attr-defined]
    assert n.cfg is new.notifications


def test_missing_smtp_credentials_logged_as_skipped(session_factory: sessionmaker[Session], settings: Settings,
                                                    sim_clock: SimulatedClock, monkeypatch: pytest.MonkeyPatch) -> None:
    import smtplib

    from config.secrets import Secrets

    def no_network(*a: Any, **k: Any) -> None:
        raise AssertionError("SMTP must not be contacted without credentials")

    monkeypatch.setattr(smtplib, "SMTP", no_network)
    n = Notifier(session_factory, settings, Secrets.model_construct(smtp_user=None, smtp_app_password=None,
                                                                     alert_email_to=None), sim_clock,
                 async_send=False)
    out = n.test()
    assert out.status == STATUS_SKIPPED and "credentials" in (out.reason or "")
    assert logs(session_factory)[0].status == "SKIPPED"
