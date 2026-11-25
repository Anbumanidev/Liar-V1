"""Analytics 8 - Strategy, mode & agent analytics: win rate, average R, expectancy, profit
factor and max drawdown per mode / strategy / regime; agent accuracy; agent latency and
failure rates (section 7.6)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard import analytics_queries as aq
from dashboard import charts as ch
from dashboard import components as ui
from storage.audit import record_error

TITLE = "Strategy, mode & agents"
ICON = ":material/insights:"

_CHART_CONFIG = {"displayModeBar": False}
_METRICS = {"net_pnl": "Net P&L", "expectancy_r": "Expectancy (R)", "win_rate_pct": "Win rate %",
            "profit_factor": "Profit factor", "max_drawdown": "Max drawdown (Rs)"}
_TABLE = ["trades", "wins", "losses", "win_rate_pct", "avg_r", "expectancy_r", "expectancy_rs", "profit_factor",
          "net_pnl", "charges", "max_drawdown"]


def _group_tab(trades: pd.DataFrame, by: str) -> None:
    table = aq.performance_table(trades, by)
    if table.empty:
        st.info("No closed trades yet.")
        return
    metric = st.radio("Chart", list(_METRICS), format_func=_METRICS.get, horizontal=True, key=f"pf_metric_{by}")
    st.plotly_chart(
        ch.group_bar_figure(table, by, metric, money=metric in ("net_pnl", "max_drawdown"),
                            signed=metric in ("net_pnl", "expectancy_r"), hover_extra="trades"),
        key=f"pf_chart_{by}", config=_CHART_CONFIG)
    ui.table(table[[by] + _TABLE])
    if by == "strategy":
        st.caption("A strategy is auto-disabled in the batch when its expectancy is negative after 30+ trades "
                   "(re-enable on Control -> Strategies).")


def _agents_tab(accuracy: pd.DataFrame) -> None:
    if accuracy.empty:
        st.info("No traded decisions with agent reports yet.")
        return
    st.plotly_chart(ch.group_bar_figure(accuracy, "agent", "hit_rate_pct", money=False, signed=False,
                                        hover_extra="votes"), key="pf_agents_chart", config=_CHART_CONFIG)
    st.caption("Hit rate = votes that agreed with a winning trade or opposed a losing one, over all non-neutral "
               "votes on traded decisions.")
    ui.table(accuracy)


def _health_tab(health: pd.DataFrame) -> None:
    if health.empty:
        st.info("No agent calls recorded for this batch yet.")
        return
    calls = int(health["calls"].sum())
    failed = int(health["failed"].sum())
    st.markdown(f"{calls:,} call(s), {failed:,} failed ({(failed / calls * 100.0) if calls else 0.0:.1f}%).")
    st.plotly_chart(ch.failure_rate_figure(health), key="pf_fail_chart", config=_CHART_CONFIG)
    st.plotly_chart(ch.agent_latency_figure(health), key="pf_latency_chart", config=_CHART_CONFIG)
    ui.table(health)


def _body(ctx: Any, batch_id: int) -> None:
    with ctx.session() as session:
        trades = aq.trades_frame(session, batch_id)
        accuracy = aq.agent_accuracy(session, batch_id, trades)
        health = aq.agent_health(session, batch_id)
    tabs = st.tabs(["Mode", "Strategy", "Regime", "Agents", "Latency & failures"])
    for tab, by in zip(tabs[:3], ("mode", "strategy", "regime"), strict=True):
        with tab:
            _group_tab(trades, by)
    with tabs[3]:
        _agents_tab(accuracy)
    with tabs[4]:
        _health_tab(health)


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    batch_id = ui.require_batch(ctx)
    if batch_id is None:
        return
    try:
        _body(ctx, batch_id)
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.analytics_performance", exc, {"batch_id": batch_id})
        st.error(f"Could not compute analytics: {exc}")
