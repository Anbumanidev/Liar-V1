"""EOD e-mail: "Tomorrow's intraday watchlist" section renders with a plan and is absent without one."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from pydantic import SecretStr

from config.secrets import Secrets
from notifications.email import STATUS_SENT, SendResult
from notifications.reports import build_eod_report, next_day_plan_section
from notifications.service import Notifier, normalize_eod_report

DAY = date(2026, 11, 25)
HEADING = "Tomorrow&#39;s intraday watchlist (paper plan, analysed again at 09:20)"


class _Sender:
    def __init__(self) -> None:
        self.html: list[str] = []

    def send(self, subject: str, html: str, recipients: Sequence[str] | str | None = None,
             attachments: Sequence[Any] | None = None, text: str | None = None) -> SendResult:
        self.html.append(html)
        return SendResult(ok=True, attempts=1, error=None, status=STATUS_SENT, recipients=["owner@example.com"],
                          subject=subject)


def _notifier(session_factory, settings, clock, sender: _Sender) -> Notifier:  # type: ignore[no-untyped-def]
    secrets = Secrets.model_construct(smtp_user="bot@gmail.com", smtp_app_password=SecretStr("abcdefghijklmnop"),
                                      alert_email_to="owner@example.com")
    return Notifier(session_factory, settings, secrets, clock, sender=sender, async_send=False)


PLAN = [
    {"symbol": "TCS", "setup": "BREAKOUT", "direction": "LONG", "trigger": 3400.0, "stop": 3350.0,
     "target": 3500.0, "rr": 2.0, "score": 72.5, "reasons": ["NR7: narrowest range", "<b>bold</b> claim"],
     "affordable": False, "status": "ACTIVE"},
    {"symbol": "ITC", "setup": "PULLBACK", "direction": "LONG", "trigger": 312.4, "stop": 308.0,
     "target": 321.2, "rr": 2.0, "score": 61.0, "reasons": ["uptrend"], "affordable": True, "status": "ACTIVE"},
]


def test_eod_email_without_a_plan_has_no_section(session_factory, session, settings, sim_clock) -> None:  # type: ignore[no-untyped-def]
    sender = _Sender()
    n = _notifier(session_factory, settings, sim_clock, sender)
    try:
        n.eod(build_eod_report(session, 1, DAY, sim_clock))
        n.eod({**build_eod_report(session, 1, DAY, sim_clock), "next_day_plan": []})
    finally:
        n.close(timeout=5)
    assert len(sender.html) == 2
    assert all("intraday watchlist" not in h for h in sender.html)
    assert all("The formatted template failed" not in h for h in sender.html)
    assert next_day_plan_section(session, date(2026, 11, 26)) == {}       # no plan in the DB -> no keys
    assert "next_day_plan" not in normalize_eod_report({})


def test_eod_email_with_a_plan_renders_the_section(session_factory, session, settings, sim_clock) -> None:  # type: ignore[no-untyped-def]
    sender = _Sender()
    n = _notifier(session_factory, settings, sim_clock, sender)
    report = {**build_eod_report(session, 1, DAY, sim_clock), "next_day_plan": PLAN,
              "next_day_plan_for": "2026-11-26", "next_day_plan_check": "09:20"}
    try:
        n.eod(report)
        n.eod({**build_eod_report(session, 1, DAY, sim_clock),                # minimal rows still render
               "next_day_plan": [{"symbol": "SBIN", "trigger": 800.0}]})
    finally:
        n.close(timeout=5)
    html = sender.html[0]
    assert "The formatted template failed" not in html
    assert HEADING in html and "For 26 Nov 2026: 2 stocks" in html
    assert html.index("TCS") < html.index("ITC")
    assert "3,400.00" in html and "312.40" in html and "321.20" in html
    assert "NR7: narrowest range" in html and "&lt;b&gt;bold&lt;/b&gt; claim" in html   # escaped
    assert "TCS *" in html and "not affordable at the current capital" in html
    assert "not advice" in html and "risk manager" in html
    assert HEADING in sender.html[1] and "SBIN" in sender.html[1] and "800.00" in sender.html[1]
    assert "The formatted template failed" not in sender.html[1]
