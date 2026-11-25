"""Analytics 4 - Day-wise log: the full timeline of one date (runs / scans, stocks analysed,
decisions, approvals, orders, entries, exits, capital movements, shadow tracking, EOD)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard import analytics_queries as aq
from dashboard import components as ui
from storage.audit import record_error

TITLE = "Day-wise log"
ICON = ":material/event_note:"


def _body(ctx: Any, batch_id: int) -> None:
    today = ctx.clock.now().date()
    with ctx.session() as session:
        days = aq.activity_days(session, batch_id)
    default = days[0] if days else today
    day = st.date_input("Day", value=default, max_value=max(today, default), key=f"dw_day_{batch_id}")
    if days:
        recent = ", ".join(d.strftime("%d %b") for d in days[:8])
        st.caption(f"{len(days)} day(s) with activity; latest: {recent}")
    include_scans = st.toggle("Include per-stock scan rows", value=True, key="dw_scans")
    with ctx.session() as session:
        timeline = aq.day_timeline(session, batch_id, day, include_scans=include_scans)
    if timeline.empty:
        st.info(f"Nothing recorded for this batch on {day:%d %b %Y}.")
        return
    counts = timeline["kind"].value_counts().to_dict()
    kinds = [k for k in aq.TIMELINE_KINDS if k in counts]
    chosen = st.pills("Show", kinds, selection_mode="multi", default=kinds, key=f"dw_kinds_{batch_id}_{day}",
                      format_func=lambda k: f"{k} {counts.get(k, 0)}")
    view = timeline[timeline["kind"].isin(chosen or [])].copy()
    st.markdown(f"#### {day:%A %d %b %Y} - {len(view)} event(s)")
    if view.empty:
        st.caption("Select at least one kind of event.")
        return
    view["time"] = view["ts"].map(lambda t: t.strftime("%H:%M:%S") if pd.notna(t) else "")
    ui.table(view[["time", "kind", "symbol", "mode", "detail", "ref"]], key="dw_table")
    eod = timeline[timeline["kind"] == "EOD"]
    if not eod.empty:
        with st.expander("End of day"):
            for row in eod.itertuples():
                st.markdown(f"- {row.detail}")


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    batch_id = ui.require_batch(ctx)
    if batch_id is None:
        return
    try:
        _body(ctx, batch_id)
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.analytics_daywise", exc, {"batch_id": batch_id})
        st.error(f"Could not build the day log: {exc}")
