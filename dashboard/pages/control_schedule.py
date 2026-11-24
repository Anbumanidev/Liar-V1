"""Control 11 - Schedule: cycle interval, scan times, square-off (within NSE hours), holiday overrides."""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import services as svc

TITLE = "Schedule"
ICON = ":material/schedule:"

TIME_FIELDS: tuple[tuple[str, str], ...] = (
    ("premarket", "Pre-market job"),
    ("watchlist_build", "Intraday watchlist build"),
    ("preopen_check", "Pre-open gap check"),
    ("opening_range_start", "Opening range start"),
    ("opening_range_end", "Opening range end"),
    ("intraday_cycle_start", "INTRADAY cycle start"),
    ("intraday_cycle_end", "INTRADAY cycle end (<= square-off)"),
    ("intraday_squareoff", "INTRADAY square-off (09:15-15:25)"),
    ("cnc_time_stop", "CNC time-stop exits"),
    ("eod_processing", "EOD capital processing"),
    ("eod_report", "EOD report + email"),
    ("swing_scan", "SWING/POSITIONAL scan start"),
    ("swing_scan_deadline", "SWING/POSITIONAL scan deadline"),
)


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _vid, settings = loaded
    sch = settings.schedule
    st.caption("All times IST, NSE trading days only. Invalid combinations (e.g. square-off outside market hours) "
               "are rejected on save.")

    with st.form("schedule_form"):
        st.markdown("#### Cycles")
        proposed: dict[str, Any] = {
            "schedule.intraday_cycle_minutes": int(st.number_input(
                "INTRADAY cycle every N minutes", 1, 60, int(sch.intraday_cycle_minutes), key="sch_cyc")),
            "schedule.btst_scan_minutes": int(st.number_input(
                "BTST scan every N minutes", 1, 60, int(sch.btst_scan_minutes), key="sch_btst")),
            "schedule.approval_timeout_check_seconds": int(st.number_input(
                "Approval timeout check (s)", 5, 300, int(sch.approval_timeout_check_seconds), key="sch_appr")),
        }
        st.markdown("#### Times")
        times: dict[str, str] = {}
        for name, label in TIME_FIELDS:
            times[name] = st.text_input(label, value=getattr(sch, name), key=f"sch_{name}")
        cnc_win = st.text_input("CNC entry window", value=svc.format_windows([sch.cnc_entry_window]), key="sch_cncw")
        btst_win = st.text_input("BTST scan window", value=svc.format_windows([sch.btst_scan_window]), key="sch_btw")
        st.markdown("#### Holidays")
        hol = st.text_area("Holiday overrides - extra NON-trading dates (YYYY-MM-DD)",
                           value="\n".join(d.isoformat() for d in sch.holiday_overrides), key="sch_hol", height=80)
        spec = st.text_area("Special sessions - trading on otherwise-holiday dates (YYYY-MM-DD)",
                            value="\n".join(d.isoformat() for d in sch.special_sessions), key="sch_spec", height=80)
        note = ui.note_input("sch_note")
        if st.form_submit_button("Save schedule", type="primary"):
            try:
                for name, value in times.items():
                    proposed[f"schedule.{name}"] = svc.validate_hhmm(value)
                cw = svc.parse_windows(cnc_win)
                bw = svc.parse_windows(btst_win)
                if len(cw) != 1 or len(bw) != 1:
                    raise svc.DashboardError("CNC entry window and BTST scan window take exactly one HH:MM-HH:MM range.")
                proposed["schedule.cnc_entry_window"] = cw[0]
                proposed["schedule.btst_scan_window"] = bw[0]
                holidays = svc.parse_dates(hol)
                specials = svc.parse_dates(spec)
                overlap = set(holidays) & set(specials)
                if overlap:
                    raise svc.DashboardError(f"Dates cannot be both holiday and special session: "
                                             f"{sorted(d.isoformat() for d in overlap)}")
                proposed["schedule.holiday_overrides"] = [d.isoformat() for d in holidays]
                proposed["schedule.special_sessions"] = [d.isoformat() for d in specials]
            except svc.DashboardError as exc:
                st.error(str(exc))
            else:
                ui.save_settings(ctx, proposed, ui.flat_values(settings, proposed.keys()), note)
