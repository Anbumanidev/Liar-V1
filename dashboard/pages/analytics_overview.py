"""Analytics 1 - Overview: capital (TRADING / RESERVE / TOTAL EQUITY / HWM), drawdown & tier,
P&L (total and per mode), charges, win rate, profit factor, expectancy; equity curve with
floors and tier bands, daily P&L bars and the mode split."""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import analytics_queries as aq
from dashboard import charts as ch
from dashboard import components as ui
from storage.audit import record_error

TITLE = "Overview"
ICON = ":material/dashboard:"

_CHART_CONFIG = {"displayModeBar": False}


def _metric_grid(items: list[tuple[str, str, str | None, str | None]]) -> None:
    """Two metrics per row (stacks to one column on a phone). Items: label, value, help, delta."""
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for col, (label, value, help_, delta) in zip(cols, items[i:i + 2], strict=False):
            col.metric(label, value, delta=delta, help=help_, border=True)


def _capital_section(snap: dict[str, Any] | None, params: aq.RiskParams) -> None:
    st.markdown("#### Capital")
    if snap is None:
        st.info("No capital account for this batch yet - balances appear once the engine initialises the batch.")
        return
    tier = snap["tier"]
    _metric_grid([
        ("TRADING", ch.fmt_rs(snap["trading"]),
         f"cash {ch.fmt_rs(snap['trading_cash'])} + blocked in positions {ch.fmt_rs(snap['blocked'])}", None),
        ("RESERVE", ch.fmt_rs(snap["reserve"]),
         f"locked profit, never traded (peak {ch.fmt_rs(snap['reserve_peak'])})", None),
        ("TOTAL EQUITY", ch.fmt_rs(snap["equity"]),
         f"TRADING + unrealized {ch.fmt_rs(snap['unrealized'], signed=True)} + RESERVE",
         f"{ch.fmt_rs(snap['net_since_start'], signed=True)} since start"),
        ("High-water mark", ch.fmt_rs(snap["hwm"]), "highest TOTAL EQUITY seen", None),
        ("Drawdown from HWM", ch.fmt_pct(snap["drawdown_pct"]),
         f"tiers: CAUTION {params.caution_pct:g}% / DEFENSIVE {params.defensive_pct:g}% / "
         f"HALT {params.hard_stop_pct:g}%", None),
        ("Tier", tier, f"engine's stored tier: {snap['engine_tier']}", None),
    ])
    st.caption(
        f"Floors (stricter wins): absolute {ch.fmt_rs(snap['abs_floor'])} = start "
        f"{ch.fmt_rs(snap['starting_capital'], nd=0)} x (1 - {params.hard_stop_pct:g}%) | trailing "
        f"{ch.fmt_rs(snap['trailing_floor'])} = HWM x (1 - {params.hard_stop_pct:g}%) | sizing capital "
        f"{ch.fmt_rs(snap['sizing_capital'])}"
    )
    if snap["halted"] or tier == "HALT":
        st.error(f"HALT: {snap.get('halt_reason') or 'floor / hard stop reached'} - resume only from Control -> Risk.")
    elif tier in ("CAUTION", "DEFENSIVE"):
        st.warning(f"Drawdown tier {tier}: position size reduced, only top grades allowed.")


def _performance_section(m: dict[str, Any], modes: Any) -> None:
    st.markdown("#### Performance (closed trades)")
    _metric_grid([
        ("Net P&L", ch.fmt_rs(m["net_pnl"], signed=True),
         f"gross {ch.fmt_rs(m['gross_pnl'], signed=True)} - charges {ch.fmt_rs(m['charges'])}", None),
        ("Charges", ch.fmt_rs(m["charges"]), "brokerage, STT, exchange, SEBI, stamp, GST, DP", None),
        ("Win rate", ch.fmt_pct(m["win_rate_pct"], nd=1),
         f"{m['wins']} won / {m['losses']} lost of {m['trades']} trade(s)", None),
        ("Profit factor", ch.fmt_pf(m["profit_factor"]), "gross profit / gross loss", None),
        ("Expectancy", f"{ch.fmt_num(m['expectancy_r'])} R",
         f"{ch.fmt_rs(m['expectancy_rs'], signed=True)} per trade; average R {ch.fmt_num(m['avg_r'])}", None),
        ("Trades", str(m["trades"]), "positions (partial exits merged into one trade)", None),
    ])
    if not modes.empty:
        st.caption("Net P&L per mode: " + " | ".join(
            f"{row.mode} {ch.fmt_rs(row.net_pnl, signed=True)} ({int(row.trades)})" for row in modes.itertuples()))


def _body(ctx: Any, batch_id: int) -> None:
    loaded = ui.active_settings(ctx)
    params = aq.RiskParams.from_settings(loaded[1] if loaded else None)
    now = ctx.clock.now()
    with ctx.session() as session:
        snap = aq.capital_snapshot(session, batch_id, params)
        trades = aq.trades_frame(session, batch_id)
        series = aq.equity_series(session, batch_id, live=aq.live_point(snap, now))
        batch = aq.batch_info(session, batch_id) or {}
    starting = snap["starting_capital"] if snap else batch.get("starting_capital")
    levels = aq.with_protection_levels(series, starting, params)
    metrics = aq.overview_metrics(trades)
    modes = aq.performance_table(trades, "mode")
    if batch:
        st.caption(f"Batch **{batch.get('name')}** ({batch.get('status')}), starting capital "
                   f"{ch.fmt_rs(batch.get('starting_capital'), nd=0)}")

    _capital_section(snap, params)
    _performance_section(metrics, modes)

    tab_eq, tab_daily, tab_modes = st.tabs(["Equity curve", "Daily P&L", "Mode split"])
    with tab_eq:
        if levels.empty:
            st.info("No equity history yet.")
        else:
            zoom = st.toggle("Zoom to the floor / tier region", value=True, key="ov_zoom")
            st.plotly_chart(ch.equity_curve_figure(levels, zoom=zoom), key="ov_equity", config=_CHART_CONFIG)
            sources = ", ".join(sorted(set(levels["source"])))
            st.caption(f"TRADING and RESERVE stacked; TOTAL EQUITY includes unrealized P&L. Bands = drawdown tiers "
                       f"from the HWM. Points: {sources} (EOD = daily summary, LIVE = now).")
            with st.expander("Equity data"):
                cols = ["ts", "source", "trading", "reserve", "unrealized", "equity", "hwm", "drawdown_pct", "tier",
                        "abs_floor", "trailing_floor"]
                ui.table(aq.round_frame(levels[cols].iloc[::-1]))
    with tab_daily:
        daily = aq.daily_pnl(trades)
        if daily.empty:
            st.info("No closed trades yet.")
        else:
            st.plotly_chart(ch.daily_pnl_figure(daily), key="ov_daily", config=_CHART_CONFIG)
            with st.expander("Daily P&L table"):
                ui.table(daily.iloc[::-1])
    with tab_modes:
        if modes.empty:
            st.info("No closed trades yet.")
        else:
            st.plotly_chart(ch.mode_split_figure(modes), key="ov_modes", config=_CHART_CONFIG)
            ui.table(modes[["mode", "trades", "win_rate_pct", "net_pnl", "charges", "profit_factor", "expectancy_r",
                            "avg_r", "max_drawdown"]])


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    batch_id = ui.require_batch(ctx)
    if batch_id is None:
        return
    try:
        _body(ctx, batch_id)
    except Exception as exc:  # noqa: BLE001 - analytics must degrade, never take the dashboard down
        record_error(None, "dashboard.analytics_overview", exc, {"batch_id": batch_id})
        st.error(f"Could not build the overview: {exc}")
