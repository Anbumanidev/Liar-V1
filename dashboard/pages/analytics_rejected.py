"""Analytics 6 - Rejected / No-trade: everything not traded with its reason (meta NO_TRADE,
risk rule, owner rejection, timeout, signals-only, filters) and the shadow-tracked outcome."""

from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from dashboard import analytics_queries as aq
from dashboard import charts as ch
from dashboard import components as ui
from storage.audit import record_error

TITLE = "Rejected / No-trade"
ICON = ":material/block:"

_DETAIL = {
    "ts": "time", "symbol": "symbol", "mode": "mode", "direction": "dir", "category": "why", "reason": "reason",
    "rule": "rule", "confidence": "conf", "grade": "grade", "entry": "entry", "stop_loss": "SL", "target_1": "T1",
    "shadow_outcome": "shadow outcome", "shadow_r": "shadow R", "shadow_pnl": "shadow net", "strategy": "strategy",
    "decision_id": "decision", "approval_id": "approval",
}


def _start_default(batch: dict[str, Any] | None, today: date) -> date:
    if batch:
        started = batch.get("started_at") or batch.get("created_at")
        if started is not None:
            return min(started.date(), today)
    return today


def _body(ctx: Any, batch_id: int) -> None:
    today = ctx.clock.now().date()
    with ctx.session() as session:
        batch = aq.batch_info(session, batch_id)
    rng = st.date_input("Date range", value=(_start_default(batch, today), today), key=f"rj_dates_{batch_id}")
    start, end = aq.normalize_date_range(rng)
    with ctx.session() as session:
        df = aq.rejected_frame(session, batch_id, start=start, end=end)
        filtered = aq.filtered_stocks_frame(session, batch_id, start=start, end=end)
    if df.empty and filtered.empty:
        st.info("Nothing was rejected, filtered or left untraded in this range.")
        return
    summary = aq.rejected_summary(df)
    tracked = int(summary["shadow_tracked"].sum()) if not summary.empty else 0
    items = [
        ("Not traded", str(len(df)), "decisions / approvals that did not become trades"),
        ("Shadow-tracked", str(tracked), "followed to their would-be SL / target without capital"),
        ("Profit missed", ch.fmt_rs(summary["missed_profit"].sum() if not summary.empty else 0.0),
         "closed shadows that would have made money"),
        ("Loss avoided", ch.fmt_rs(summary["avoided_loss"].sum() if not summary.empty else 0.0),
         "closed shadows that would have lost money"),
    ]
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for col, (label, value, help_) in zip(cols, items[i:i + 2], strict=False):
            col.metric(label, value, help=help_, border=True)

    if not df.empty:
        st.markdown("#### By reason")
        st.plotly_chart(ch.rejected_figure(summary), key="rj_chart", config={"displayModeBar": False})
        st.caption("Net hypothetical P&L of the closed shadow trades: right / blue = profit missed, "
                   "left / red = loss avoided.")
        ui.table(summary.drop(columns=["category"]).rename(columns={"label": "reason"}))
        st.markdown("#### Every setup not traded")
        cats = list(summary["category"])
        chosen = st.multiselect("Reason", cats, format_func=lambda c: aq.CATEGORY_LABELS.get(c, c),
                                key=f"rj_cats_{batch_id}", placeholder="all reasons")
        view = df[df["category"].isin(chosen)] if chosen else df
        view = view.assign(category=view["category"].map(lambda c: aq.CATEGORY_LABELS.get(c, c)))
        ui.table(aq.round_frame(view[list(_DETAIL)]).rename(columns=_DETAIL), key="rj_table")
    with st.expander(f"Stocks excluded by universe / watchlist / scan filters ({len(filtered)})"):
        if filtered.empty:
            st.caption("No filtered stocks recorded in this range.")
        else:
            ui.table(filtered)


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    batch_id = ui.require_batch(ctx)
    if batch_id is None:
        return
    try:
        _body(ctx, batch_id)
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.analytics_rejected", exc, {"batch_id": batch_id})
        st.error(f"Could not load rejected setups: {exc}")
