"""EOD email renders the FY tax estimate + tips when the report carries them."""

from __future__ import annotations

from datetime import date

from notifications.reports import build_eod_report
from tax.engine import TaxSummary, TaxTip
from tests.notifications.conftest import FakeSender


def test_eod_email_includes_tax_section(make_notifier, session, sim_clock) -> None:  # type: ignore[no-untyped-def]
    report = dict(build_eod_report(session, 1, date(2026, 11, 25), sim_clock))
    s = TaxSummary(fy="FY2026-27", start=date(2026, 4, 1), end=date(2027, 3, 31), total_tax=123.45,
                   net_profit_before_tax=600.0, net_profit_after_tax=476.55, stt_paid=12.0)
    report["tax"] = s.as_dict()
    report["tax_tips"] = [TaxTip("saving", "Tax-loss harvesting: LT", "book the loss", 20.8).as_dict()]
    sender = FakeSender()
    make_notifier(sender).eod(report)
    html = sender.last_html
    assert "Tax estimate (FY2026-27" in html and "Tax-loss harvesting: LT" in html and "123.45" in html
