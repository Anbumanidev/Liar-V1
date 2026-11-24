"""Control 12 - Notifications: e-mail types on/off, recipients, quiet hours, test e-mail."""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.enums import CommandType, EmailType
from dashboard import components as ui
from dashboard import services as svc

TITLE = "Notifications"
ICON = ":material/mail:"

_DESCRIPTIONS = {
    EmailType.PENDING_APPROVAL: "Pending approval (approval itself only in the dashboard)",
    EmailType.ENTRY: "Entry",
    EmailType.EXIT: "Exit",
    EmailType.EOD: "End-of-day report (15:45)",
    EmailType.SWING_SCAN: "Evening SWING/POSITIONAL scan",
    EmailType.URGENT: "URGENT: HALT, gap through SL, data feed down",
    EmailType.TEST: "Test e-mails",
}


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _vid, settings = loaded
    n = settings.notifications
    if not ctx.secrets.smtp_user or ctx.secrets.smtp_app_password is None:
        st.warning("SMTP_USER / SMTP_APP_PASSWORD are not set in .env - e-mails cannot be sent.")

    with st.form("notif_form"):
        st.markdown("#### E-mail types")
        enabled = {t.value: st.toggle(_DESCRIPTIONS.get(t, t.value), value=n.enabled_types.get(t, True),
                                      key=f"nt_{t.value}") for t in EmailType}
        st.markdown("#### Recipients")
        recips = st.text_area("Recipients (comma / newline separated; empty = ALERT_EMAIL_TO from .env)",
                              value=", ".join(n.recipients), key="nt_recips", height=70)
        st.markdown("#### Quiet hours (IST)")
        qs = st.text_input("Start (HH:MM, empty = none)", value=n.quiet_hours_start or "", key="nt_qs")
        qe = st.text_input("End (HH:MM, empty = none)", value=n.quiet_hours_end or "", key="nt_qe")
        urgent = st.toggle("URGENT e-mails ignore quiet hours", value=n.urgent_ignores_quiet_hours, key="nt_urg")
        prefix = st.text_input("Subject prefix", value=n.subject_prefix, key="nt_prefix")
        note = ui.note_input("nt_note")
        if st.form_submit_button("Save notifications", type="primary"):
            valid, invalid = svc.parse_emails(recips)
            try:
                start = svc.validate_hhmm(qs, allow_empty=True)
                end = svc.validate_hhmm(qe, allow_empty=True)
                if (start is None) != (end is None):
                    raise svc.DashboardError("Set both quiet-hours start and end, or neither.")
                if invalid:
                    raise svc.DashboardError(f"Invalid e-mail address(es): {', '.join(invalid)}")
            except svc.DashboardError as exc:
                st.error(str(exc))
            else:
                proposed: dict[str, Any] = {
                    "notifications.enabled_types": enabled, "notifications.recipients": valid,
                    "notifications.quiet_hours_start": start, "notifications.quiet_hours_end": end,
                    "notifications.urgent_ignores_quiet_hours": urgent,
                    "notifications.subject_prefix": prefix.strip() or "[PAPER]",
                }
                ui.save_settings(ctx, proposed, ui.flat_values(settings, proposed.keys()), note)

    st.markdown("#### Test")
    if st.button("Send test e-mail", icon=":material/send:", key="nt_test"):
        ui.submit_command(ctx, CommandType.SEND_TEST_EMAIL, {}, track_key="nt_last_cmd")
    ui.tracked_command(ctx, "nt_last_cmd")

    st.markdown("#### Recent e-mails")
    with ctx.session() as session:
        logs = svc.recent_notifications(session, limit=20)
    ui.table([{"ts": ui.fmt_dt(r.ts), "type": r.email_type, "status": r.status, "subject": r.subject,
               "attempts": r.attempts, "error": (r.error or "")[:120]} for r in logs])
