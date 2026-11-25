"""Analytics 5 - Trades: filter by mode / strategy / source / date; open a trade for its chart
(candles when stored, otherwise its levels on its own timeline), agent reports, debate, meta
reasoning, risk verdict and the approval / edit history."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from config.schema import Settings
from core.enums import Mode
from dashboard import analytics_queries as aq
from dashboard import charts as ch
from dashboard import components as ui
from dashboard import services as svc
from storage.audit import record_error

TITLE = "Trades"
ICON = ":material/candlestick_chart:"

_TABLE = {
    "id": "id", "exit_time": "exit", "symbol": "symbol", "mode": "mode", "direction": "dir", "qty": "qty",
    "entry_price": "entry", "exit_price": "exit px", "exit_reason": "reason", "net_pnl": "net", "r_multiple": "R",
    "charges": "charges", "strategy": "strategy", "source": "source", "is_partial": "partial",
}


def _timeframe(settings: Settings | None, mode: str | None) -> str | None:
    if settings is None or not mode:
        return None
    try:
        return settings.mode(Mode(mode)).primary_timeframe.value
    except (ValueError, KeyError):
        return None


def _render_agents(details: svc.DecisionDetails) -> None:
    if not details.agent_reports:
        st.caption("No agent reports stored for this trade (manual trade or reports not linked).")
        return
    ui.table([{"agent": r["agent"], "signal": r["signal"], "conf": r["confidence"], "status": r["status"],
               "model": r.get("model"), "prompt v": r.get("prompt_version"), "ms": r.get("latency_ms"),
               "error": (r.get("error") or "")[:120]} for r in details.agent_reports])
    for r in details.agent_reports:
        ui.json_expander(f"{r['agent']} - full report", r.get("report_json") or {})


def _render_debate(details: svc.DecisionDetails) -> None:
    if not details.debates:
        st.caption("No debate stored for this trade.")
        return
    for d in details.debates:
        st.markdown(f"**Debate #{d['id']}** - {d['rounds']} round(s), {d['status']}"
                    f"{(', ' + str(d['model'])) if d.get('model') else ''}")
        for turn in d.get("transcript_json") or []:
            if isinstance(turn, dict):
                who = turn.get("side") or turn.get("agent") or turn.get("role") or "?"
                rnd = f" (round {turn['round']})" if turn.get("round") is not None else ""
                text = turn.get("text") or turn.get("argument") or turn.get("content") or aq.jsonish(turn)
                st.markdown(f"**{who}{rnd}:** {text}")
            else:
                st.markdown(str(turn))
        for side in ("bull_json", "bear_json"):
            if d.get(side):
                ui.json_expander(f"{side.split('_')[0].title()} case (structured)", d[side])


def _render_meta(details: svc.DecisionDetails) -> None:
    dec = details.decision
    if dec is None:
        st.caption("No decision linked to this trade (manual trade).")
        return
    st.markdown(f"**Meta decision:** {dec.get('decision')} {dec.get('direction') or ''} - confidence "
                f"{dec.get('confidence')}, grade {dec.get('grade') or '-'}, confluence "
                f"{ch.fmt_num(dec.get('confluence_score'), 1)}, regime {dec.get('regime') or '-'}, "
                f"strategy {dec.get('strategy') or '-'} -> {dec.get('final_action')}")
    meta = details.meta
    reasoning = meta.get("reasoning") or meta.get("rationale") or meta.get("summary")
    if reasoning:
        st.markdown("**Reasoning**")
        st.write(reasoning)
    ui.json_expander("Meta output (full)", meta)
    ui.json_expander("Confluence / strategies / SL candidates", details.context)
    risk = details.risk
    if risk:
        approved = risk.get("approved")
        verdict = "APPROVED" if approved else ("REJECTED" if approved is False else "-")
        rule = f" - rule {risk.get('rule')}" if risk.get("rule") else ""
        st.markdown(f"**Risk verdict:** {verdict}{rule}")
        ui.json_expander("Risk checks (all)", risk)


def _render_approval(detail: dict[str, Any]) -> None:
    appr = detail["approval"]
    if appr is None:
        st.caption("This trade did not go through the approval queue (AUTO mode or manual trade).")
    else:
        edits = appr.get("user_edits_json")
        st.markdown(f"**Approval #{appr['id']}**: {appr['status']}"
                    f"{(' by ' + appr['decided_by']) if appr.get('decided_by') else ''}"
                    f"{(' at ' + ui.fmt_dt(appr.get('decided_at'))) if appr.get('decided_at') else ''}")
        st.caption(f"Proposed: {appr['direction']} {appr['qty']} (max {appr['max_qty']}) @ "
                   f"{ch.fmt_num(appr['entry_price'])} SL {ch.fmt_num(appr['stop_loss'])} T1 "
                   f"{ch.fmt_num(appr['target_1'])} T2 {ch.fmt_num(appr.get('target_2'))} as {appr['mode']}"
                   f"{(' -> approved as ' + appr['final_mode']) if appr.get('final_mode') else ''}")
        if appr.get("user_reason"):
            st.markdown(f"**My reason:** {appr['user_reason']}")
        if edits:
            ui.json_expander("My edits", edits, expanded=True)
        if appr.get("block_rule"):
            st.warning(f"My approval was blocked by risk rule {appr['block_rule']}.")
    history = detail["history"]
    st.markdown("**History**")
    if history.empty:
        st.caption("No approval / edit history recorded.")
    else:
        ui.table(history)


def _render_detail(ctx: Any, settings: Settings | None, trade_id: int) -> None:
    with ctx.session() as session:
        detail = aq.trade_detail(session, trade_id)
        if detail is None:
            st.warning("Trade not found.")
            return
        trade = detail["trade"]
        timeframe = _timeframe(settings, trade.get("mode"))
        candles, source = aq.trade_candles(session, trade, timeframe=timeframe, cache_dir=ctx.runtime_dir / "cache")
    legs = detail["legs"]
    net = float(legs["net_pnl"].sum()) if not legs.empty else float(trade.get("net_pnl") or 0.0)
    st.markdown(f"**#{trade['id']} {trade['symbol']}** {trade['mode']} {trade['direction']} - "
                f"{trade.get('strategy') or 'no strategy'} [{trade.get('source')}] - position net "
                f"{ch.fmt_rs(net, signed=True)}")
    tabs = st.tabs(["Chart", "Agents", "Debate", "Meta & risk", "Approval & edits", "Legs & orders"])
    with tabs[0]:
        st.plotly_chart(ch.trade_figure(trade, candles, legs=legs if len(legs) > 1 else None),
                        key=f"tr_chart_{trade_id}", config={"displayModeBar": False})
        if source == "none":
            st.caption("No stored candles for this trade (the dashboard never fetches market data), so its "
                       "levels are drawn on its own entry/exit timeline.")
        else:
            st.caption(f"{timeframe or ''} candles from the {source}.")
        held = (f"{trade['holding_minutes']} min" if trade.get("holding_minutes") is not None
                else f"{trade.get('holding_days') or 0} day(s)")
        st.caption(f"Entry {ch.fmt_num(trade['entry_price'])} at {ui.fmt_dt(trade['entry_time'])} | exit "
                   f"{ch.fmt_num(trade['exit_price'])} at {ui.fmt_dt(trade['exit_time'])} ({trade['exit_reason']}) | "
                   f"R {ch.fmt_num(trade.get('r_multiple'))} | gross {ch.fmt_rs(trade.get('gross_pnl'), signed=True)} "
                   f"| charges {ch.fmt_rs(trade.get('charges'))} | held {held} | regime {trade.get('regime') or '-'}")
    with tabs[1]:
        _render_agents(detail["decision"])
    with tabs[2]:
        _render_debate(detail["decision"])
    with tabs[3]:
        _render_meta(detail["decision"])
    with tabs[4]:
        _render_approval(detail)
    with tabs[5]:
        st.markdown("**Exit legs**")
        ui.table(aq.round_frame(legs))
        st.markdown("**Orders**")
        ui.table(aq.round_frame(detail["orders"]))
        if detail["position"]:
            ui.json_expander("Position (incl. frozen settings and SL history)",
                             {k: (str(v) if not isinstance(v, (int, float, str, bool, dict, list, type(None))) else v)
                              for k, v in detail["position"].items()})


def _body(ctx: Any, batch_id: int) -> None:
    loaded = ui.active_settings(ctx)
    settings = loaded[1] if loaded else None
    with ctx.session() as session:
        trades = aq.trades_frame(session, batch_id)
    if trades.empty:
        st.info("No closed trades in this batch yet.")
        return
    opts = aq.trade_filter_options(trades)
    first = pd.Timestamp(trades["exit_time"].min()).date()
    last = pd.Timestamp(trades["exit_time"].max()).date()
    with st.expander("Filters"):
        modes = st.multiselect("Mode", opts["modes"], key=f"tr_modes_{batch_id}", placeholder="all modes")
        strategies = st.multiselect("Strategy", opts["strategies"], key=f"tr_strat_{batch_id}",
                                    placeholder="all strategies")
        sources = st.multiselect("Source", opts["sources"], key=f"tr_src_{batch_id}", placeholder="SYSTEM and MANUAL")
        rng = st.date_input("Exit date range", value=(first, last), min_value=first, max_value=last,
                            key=f"tr_dates_{batch_id}")
    start, end = aq.normalize_date_range(rng)
    view = aq.filter_trades(trades, modes=modes, strategies=strategies, sources=sources, start=start, end=end)
    m = aq.overview_metrics(view)
    st.caption(f"{len(view)} exit leg(s) = {m['trades']} trade(s) | net {ch.fmt_rs(m['net_pnl'], signed=True)} | "
               f"win rate {ch.fmt_pct(m['win_rate_pct'], nd=1)} | PF {ch.fmt_pf(m['profit_factor'])} | "
               f"expectancy {ch.fmt_num(m['expectancy_r'])} R")
    if view.empty:
        st.info("No trades match the filters.")
        return
    ui.table(aq.round_frame(view.iloc[::-1][list(_TABLE)]).rename(columns=_TABLE), key="tr_table", height=300)
    choices = aq.trade_choices(view)
    labels = dict(choices)
    trade_id = st.selectbox("Open a trade", [c[0] for c in choices], format_func=lambda i: labels.get(i, str(i)),
                            key=f"tr_pick_{batch_id}")
    if trade_id is not None:
        _render_detail(ctx, settings, int(trade_id))


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    batch_id = ui.require_batch(ctx)
    if batch_id is None:
        return
    try:
        _body(ctx, batch_id)
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.analytics_trades", exc, {"batch_id": batch_id})
        st.error(f"Could not load trades: {exc}")
