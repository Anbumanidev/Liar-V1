"""Analytics 9 - Batch comparison: side-by-side metrics, overlaid equity curves and the settings
diff between two batches (batch snapshot, else its settings version)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard import analytics_queries as aq
from dashboard import charts as ch
from dashboard import components as ui
from dashboard import services as svc
from storage.audit import record_error

TITLE = "Batch comparison"
ICON = ":material/compare:"

_ROWS: dict[str, tuple[str, str]] = {
    "status": ("Status", "text"), "starting_capital": ("Starting capital", "rs"), "started": ("Started", "dt"),
    "ended": ("Ended", "dt"), "days": ("Days", "num1"), "trades": ("Trades", "int"),
    "win_rate_pct": ("Win rate", "pct"), "profit_factor": ("Profit factor", "pf"),
    "expectancy_r": ("Expectancy (R)", "num"), "expectancy_rs": ("Expectancy (Rs/trade)", "rs"),
    "net_pnl": ("Net P&L", "rs"), "charges": ("Charges", "rs"), "equity": ("TOTAL EQUITY (last)", "rs"),
    "return_pct": ("Return", "pct"), "max_drawdown_pct": ("Max drawdown", "pct"), "reserve": ("RESERVE", "rs"),
    "hwm": ("HWM", "rs"),
}


def _fmt(value: Any, kind: str) -> str:
    if kind == "rs":
        return ch.fmt_rs(value)
    if kind == "pct":
        return ch.fmt_pct(value)
    if kind == "pf":
        return ch.fmt_pf(value)
    if kind == "num":
        return ch.fmt_num(value, 3)
    if kind == "num1":
        return ch.fmt_num(value, 1)
    if kind == "int":
        return "-" if value is None or pd.isna(value) else f"{int(value):,}"
    if kind == "dt":
        return "-" if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
    return "-" if value is None else str(value)


def _side_by_side(comp: pd.DataFrame) -> pd.DataFrame:
    """Metrics as rows, batches as columns, every cell pre-formatted text (phone-readable)."""
    data: dict[str, list[str]] = {"metric": [label for label, _ in _ROWS.values()]}
    for row in comp.to_dict("records"):
        data[f"#{row['batch_id']} {row['batch']}"] = [_fmt(row.get(col), kind) for col, (_, kind) in _ROWS.items()]
    return pd.DataFrame(data)


def _body(ctx: Any) -> None:
    with ctx.session() as session:
        options = svc.batch_options(session)
    if not options:
        st.info("No batches yet. Start one on Control -> Batches.")
        return
    ids = [o.id for o in options]
    labels = {o.id: o.label for o in options}
    default = [ctx.batch_id] if ctx.batch_id in ids else ids[:1]
    others = [i for i in ids if i not in default]
    if others:
        default.append(others[0])
    chosen = st.multiselect("Batches to compare", ids, default=default, format_func=lambda i: labels.get(i, str(i)),
                            max_selections=8, key="bt_compare")
    if not chosen:
        st.info("Pick at least one batch.")
        return
    now = ctx.clock.now()
    with ctx.session() as session:
        comp = aq.batch_comparison(session, chosen, now=now)
        curves = aq.batch_equity_curves(session, chosen, now=now)

    st.markdown("#### Side by side")
    ui.table(_side_by_side(comp), key="bt_table")

    st.markdown("#### Equity curves")
    if curves.empty:
        st.info("No equity history for the selected batches yet.")
    else:
        y = st.radio("Show", ["return_pct", "equity"], horizontal=True, key="bt_y",
                     format_func={"return_pct": "Return % (comparable)", "equity": "Equity (Rs)"}.get)
        x = st.radio("X axis", ["day_n", "ts"], horizontal=True, key="bt_x",
                     format_func={"day_n": "Days since start", "ts": "Calendar date"}.get)
        st.plotly_chart(ch.batch_equity_figure(curves, y=y, x=x), key="bt_chart", config={"displayModeBar": False})

    st.markdown("#### Settings diff")
    a = st.selectbox("Batch A", ids, index=ids.index(chosen[0]), format_func=lambda i: labels.get(i, str(i)),
                     key="bt_diff_a")
    b_default = chosen[1] if len(chosen) > 1 else (ids[1] if len(ids) > 1 else ids[0])
    b = st.selectbox("Batch B", ids, index=ids.index(b_default), format_func=lambda i: labels.get(i, str(i)),
                     key="bt_diff_b")
    with ctx.session() as session:
        sa, src_a = aq.batch_settings(session, int(a))
        sb, src_b = aq.batch_settings(session, int(b))
    st.caption(f"A: {src_a} | B: {src_b}")
    if a == b:
        st.caption("Pick two different batches.")
    elif sa is None or sb is None:
        st.info("No settings snapshot stored for one of these batches.")
    else:
        diff = aq.settings_diff_frame(sa, sb)
        if diff.empty:
            st.success("Identical settings.")
        else:
            st.markdown(f"{len(diff)} setting(s) differ.")
            ui.table(diff.rename(columns={"a": f"A #{a}", "b": f"B #{b}"}), key="bt_diff")


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    try:
        _body(ctx)
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.analytics_batches", exc, {"batch_id": ctx.batch_id})
        st.error(f"Could not compare batches: {exc}")
