"""Analytics 7 - Me vs System: P&L of the trades I approved, rejected or modified (and my manual
trades) vs what the system alone would have done, using trades + approval_queue + shadow_trades."""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import analytics_queries as aq
from dashboard import charts as ch
from dashboard import components as ui
from storage.audit import record_error

TITLE = "Me vs System"
ICON = ":material/compare_arrows:"

_DETAIL = {
    "ts": "time", "symbol": "symbol", "mode": "mode", "category": "my decision", "trade_status": "status",
    "my_pnl": "me", "system_pnl": "system alone", "edge": "my edge", "note": "note", "approval_id": "approval",
}
_METHOD = """
* **Approved as proposed** - the system's trade itself: me = system.
* **Modified** - me = the trade I executed; system = shadow outcome of the original proposal.
* **Rejected / timed out / blocked** - me = 0 (no trade); system = shadow outcome of the proposal
  (followed to its SL / target without capital).
* **Manual trades** - me = their net P&L; system = 0 (it never proposed them).

Rows whose outcome is still open (position or shadow) or that were never shadow-tracked are listed
but left out of the totals. All P&L is net of charges.
"""


def _body(ctx: Any, batch_id: int) -> None:
    with ctx.session() as session:
        detail = aq.me_vs_system(session, batch_id)
    if detail.empty:
        st.info("No decided approvals or manual trades in this batch yet - this page fills up once you approve, "
                "reject or modify proposals (MANUAL_APPROVE modes) or place manual trades.")
        return
    totals = aq.me_vs_system_totals(detail)
    summary = aq.me_vs_system_summary(detail)
    edge = totals["edge"]
    items = [
        ("My P&L", ch.fmt_rs(totals["my_pnl"], signed=True), "what my decisions produced (comparable rows)", None),
        ("System alone", ch.fmt_rs(totals["system_pnl"], signed=True), "what the system would have produced", None),
        ("My edge", ch.fmt_rs(edge, signed=True), "me - system alone", None),
        ("Compared", f"{totals['compared']} / {totals['items']}",
         f"{totals['pending']} still open or not shadow-tracked", None),
    ]
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for col, (label, value, help_, delta) in zip(cols, items[i:i + 2], strict=False):
            col.metric(label, value, delta=delta, help=help_, border=True)
    if totals["compared"]:
        verdict = "added" if edge > 0 else ("cost" if edge < 0 else "changed")
        st.markdown(f"Over {totals['compared']} comparable decision(s) my choices **{verdict} "
                    f"{ch.fmt_rs(abs(edge))}** versus letting the system trade alone.")
    st.plotly_chart(ch.me_vs_system_figure(summary), key="mvs_chart", config={"displayModeBar": False})
    ui.table(summary.drop(columns=["category"]).rename(columns={"label": "my decision", "my_pnl": "me",
                                                                "system_pnl": "system alone", "edge": "my edge"}))
    with st.expander("How this is computed"):
        st.markdown(_METHOD)
    st.markdown("#### Every decision")
    cats = [c for c in aq.MVS_CATEGORIES if c in set(detail["category"])]
    chosen = st.multiselect("Decision", cats, format_func=lambda c: aq.MVS_LABELS.get(c, c), key=f"mvs_cats_{batch_id}",
                            placeholder="all decisions")
    view = detail[detail["category"].isin(chosen)] if chosen else detail
    view = view.assign(category=view["category"].map(lambda c: aq.MVS_LABELS.get(c, c)))
    ui.table(aq.round_frame(view.iloc[::-1][list(_DETAIL)]).rename(columns=_DETAIL), key="mvs_table")


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    batch_id = ui.require_batch(ctx)
    if batch_id is None:
        return
    try:
        _body(ctx, batch_id)
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.analytics_me_vs_system", exc, {"batch_id": batch_id})
        st.error(f"Could not compare decisions: {exc}")
