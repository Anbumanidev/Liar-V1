"""Analytics 3 - Capital ledger: every lock, refill, sizing change, manual transfer (and every
other movement) with running balances, filters and CSV download."""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import analytics_queries as aq
from dashboard import charts as ch
from dashboard import components as ui
from storage.audit import record_error

TITLE = "Capital ledger"
ICON = ":material/account_balance:"

_COLUMNS = {
    "ts": "time", "entry_type": "type", "amount": "amount", "trading_cash": "cash", "blocked": "blocked",
    "trading": "TRADING", "reserve": "RESERVE", "total": "TRADING+RESERVE", "sizing_capital": "sizing",
    "cum_locked": "locked (cum)", "cum_refilled": "refilled (cum)", "cum_manual_transfer": "manual (cum)",
    "cum_realized": "realized (cum)", "mode": "mode", "from_account": "from", "to_account": "to",
    "actor": "by", "confirmed": "confirmed", "position_id": "position", "trade_id": "trade", "note": "note",
}


def _body(ctx: Any, batch_id: int) -> None:
    with ctx.session() as session:
        full = aq.ledger_frame(session, batch_id)
    if full.empty:
        st.info("No capital ledger entries for this batch yet.")
        return
    first, last = full["ts"].min().date(), full["ts"].max().date()
    with st.expander("Filters"):
        types = sorted(full["entry_type"].dropna().unique())
        chosen_types = st.multiselect("Entry types", types, key=f"lg_types_{batch_id}", placeholder="all types")
        rng = st.date_input("Date range", value=(first, last), min_value=first, max_value=last,
                            key=f"lg_dates_{batch_id}")
        modes = sorted(full["mode"].dropna().unique())
        chosen_modes = st.multiselect("Mode", modes, key=f"lg_modes_{batch_id}", placeholder="all modes")
        search = st.text_input("Search note / actor", key=f"lg_search_{batch_id}")
    start, end = aq.normalize_date_range(rng)
    df = aq.filter_ledger(full, entry_types=chosen_types, start=start, end=end, modes=chosen_modes, search=search)
    summary = aq.ledger_summary(df)
    now = aq.ledger_summary(full)["last"] or {}

    items = [
        ("TRADING now", ch.fmt_rs(now.get("trading")), "cash + blocked after the latest movement"),
        ("RESERVE now", ch.fmt_rs(now.get("reserve")), "locked profits"),
        ("Profit locked", ch.fmt_rs(summary["locked"]), "TRADING -> RESERVE at EOD (in the filter)"),
        ("Refilled", ch.fmt_rs(summary["refilled"]), "RESERVE -> TRADING, capped at EOD (in the filter)"),
        ("Manual transfers", ch.fmt_rs(summary["manual_transfers"]), "owner-confirmed RESERVE -> TRADING"),
        ("Sizing changes", str(summary["sizing_changes"]), "grow slowly, shrink fast"),
    ]
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for col, (label, value, help_) in zip(cols, items[i:i + 2], strict=False):
            col.metric(label, value, help=help_, border=True)

    st.markdown(f"#### Running balances ({summary['rows']} of {len(full)} entries)")
    if df.empty:
        st.info("No entries match the filters.")
        return
    st.plotly_chart(ch.ledger_balances_figure(df), key="lg_chart", config={"displayModeBar": False})
    table = aq.round_frame(df.iloc[::-1][list(_COLUMNS)]).rename(columns=_COLUMNS)
    ui.table(table, key="lg_table")
    with st.expander("Totals by entry type"):
        ui.table(summary["by_type"])
    st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"ledger_batch_{batch_id}.csv", mime="text/csv", key="lg_csv",
                       icon=":material/download:")


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    batch_id = ui.require_batch(ctx)
    if batch_id is None:
        return
    try:
        _body(ctx, batch_id)
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.analytics_ledger", exc, {"batch_id": batch_id})
        st.error(f"Could not load the ledger: {exc}")
