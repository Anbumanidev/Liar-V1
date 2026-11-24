"""Control 3 - Approval queue (live, auto-refresh, countdown, full reasoning, approve/reject/modify)."""

from __future__ import annotations

from typing import Any

import streamlit as st
from loguru import logger

from core.enums import CommandType, EntryType
from dashboard import components as ui
from dashboard import services as svc
from dashboard.auth import AUTOREFRESH_KEY

TITLE = "Approvals"
ICON = ":material/fact_check:"

_REFRESH_MS = 5000
_APPROVAL_COMMANDS = (CommandType.APPROVE, CommandType.REJECT, CommandType.MODIFY_APPROVE, CommandType.BULK_REJECT)


def _auto_refresh() -> None:
    paused = st.toggle("Pause auto-refresh (while editing)", key="appr_pause_refresh", value=False)
    if paused:
        st.button("Refresh now", icon=":material/refresh:", key="appr_manual_refresh")
        return
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=_REFRESH_MS, key=AUTOREFRESH_KEY)
        st.caption(f"Auto-refreshing every {_REFRESH_MS // 1000} s.")
    except ImportError as exc:
        logger.warning(f"streamlit-autorefresh unavailable: {exc}")
        st.button("Refresh", icon=":material/refresh:", key="appr_manual_refresh")


def _render_reasoning(details: svc.DecisionDetails, item: Any) -> None:
    summary = item.summary_json or {}
    if summary:
        bull = summary.get("bull_points") or summary.get("top_bull") or []
        bear = summary.get("bear_points") or summary.get("top_bear") or []
        if bull:
            st.markdown("**Bull:** " + " | ".join(str(b) for b in bull[:3]))
        if bear:
            st.markdown("**Bear:** " + " | ".join(str(b) for b in bear[:3]))
    meta = details.meta
    if meta:
        st.markdown("**Meta reasoning**")
        st.write(meta.get("reasoning") or meta.get("rationale") or "")
        ui.json_expander("Meta decision (full)", meta)
    ctxj = details.context
    if ctxj:
        conf = ctxj.get("confluence") or {}
        if conf:
            st.markdown(f"**Confluence:** {conf.get('score', '-')} ({conf.get('grade', '-')})")
        ui.json_expander("Confluence / strategies / SL candidates", ctxj)
    if details.agent_reports:
        st.markdown("**Agent reports**")
        ui.table([{"agent": r["agent"], "signal": r["signal"], "conf": r["confidence"], "status": r["status"],
                   "model": r.get("model"), "ms": r.get("latency_ms")} for r in details.agent_reports])
        for r in details.agent_reports:
            ui.json_expander(f"{r['agent']} report", r.get("report_json") or {})
    for d in details.debates:
        with st.expander(f"Debate transcript ({d['rounds']} rounds, {d['status']})"):
            for turn in d.get("transcript_json") or []:
                if isinstance(turn, dict):
                    who = turn.get("side") or turn.get("agent") or turn.get("role") or "?"
                    text = turn.get("text") or turn.get("argument") or turn.get("content") or turn
                    st.markdown(f"**{who}:** {text}")
                else:
                    st.markdown(str(turn))
    risk = details.risk
    if risk:
        approved = risk.get("approved")
        st.markdown(f"**Risk verdict:** {'APPROVED' if approved else ('REJECTED' if approved is False else '-')}"
                    f"{(' - rule ' + str(risk.get('rule'))) if risk.get('rule') else ''}")
        charges = risk.get("charges") or risk.get("charges_breakdown")
        if charges:
            ui.json_expander("Charges", charges)
        ui.json_expander("Risk checks (all)", risk)
    if not (meta or details.agent_reports or details.debates or risk or summary):
        st.caption("No reasoning stored for this item.")


def _render_item(ctx: Any, view: svc.ApprovalView) -> None:
    item = view.item
    rr = view.rr
    head = (f"{item.symbol} {item.mode} {item.direction} {item.qty}@{item.entry_price:.2f} "
            f"SL {item.stop_loss:.2f} T1 {item.target_1:.2f} | {item.grade or '-'} {item.confidence}% | "
            f"{'EXPIRED' if view.expired else 'expires in ' + view.countdown}")
    with st.container(border=True):
        st.markdown(f"**#{item.id}** {head}")
        st.caption(f"strategy {item.strategy or '-'} | R:R {rr:.2f} | risk Rs {view.risk_amount:,.2f} | "
                   f"max qty {item.max_qty} | {item.entry_type}" if rr is not None else f"max qty {item.max_qty}")
        with st.expander("Full reasoning"):
            with ctx.session() as session:
                _, details = svc.approval_details(session, item.id)
            _render_reasoning(details, item)
        b1, b2 = st.columns(2)
        if b1.button("Approve", key=f"ap_{item.id}", type="primary", width="stretch", icon=":material/check:"):
            ui.submit_command(ctx, CommandType.APPROVE, {"approval_id": item.id}, track_key="appr_last_cmd")
        with b2.popover("Reject", width="stretch", icon=":material/close:"):
            reason = st.text_input("Reason (optional)", key=f"rj_reason_{item.id}")
            if st.button("Confirm reject", key=f"rj_{item.id}"):
                ui.submit_command(ctx, CommandType.REJECT, {"approval_id": item.id, "reason": reason or None},
                                  track_key="appr_last_cmd")
        with st.expander("Modify / approve as different mode"):
            allowed = svc.allowed_modes_for_direction(item.direction)
            with st.form(f"modify_{item.id}"):
                mode = st.selectbox("Mode", allowed, index=allowed.index(item.mode) if item.mode in allowed else 0,
                                    key=f"md_mode_{item.id}")
                etypes = [e.value for e in EntryType]
                etype = st.selectbox("Entry type", etypes,
                                     index=etypes.index(item.entry_type) if item.entry_type in etypes else 0,
                                     key=f"md_et_{item.id}")
                entry = st.number_input("Entry price", min_value=0.01, value=float(item.entry_price), step=0.05,
                                        key=f"md_entry_{item.id}")
                sl = st.number_input("Stop-loss", min_value=0.01, value=float(item.stop_loss), step=0.05,
                                     key=f"md_sl_{item.id}",
                                     help="Before entry any valid SL is allowed; after entry SL may only be tightened.")
                t1 = st.number_input("Target 1", min_value=0.01, value=float(item.target_1), step=0.05,
                                     key=f"md_t1_{item.id}")
                t2 = st.number_input("Target 2 (0 = none)", min_value=0.0, value=float(item.target_2 or 0.0),
                                     step=0.05, key=f"md_t2_{item.id}")
                qty = st.number_input(f"Qty (max {item.max_qty} - risk-manager maximum)", min_value=1,
                                      max_value=max(1, int(item.max_qty)), value=min(max(1, int(item.qty)),
                                                                                    max(1, int(item.max_qty))),
                                      key=f"md_qty_{item.id}")
                reason = st.text_input("Reason for the change (optional)", key=f"md_reason_{item.id}")
                if st.form_submit_button("Modify & approve", type="primary"):
                    errors = svc.validate_trade_levels(item.direction, float(entry), float(sl), float(t1),
                                                       float(t2) if t2 > 0 else None)
                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        payload: dict[str, Any] = {"approval_id": item.id, "reason": reason or None}
                        if mode != item.mode:
                            payload["mode"] = mode
                        if etype != item.entry_type:
                            payload["entry_type"] = etype
                        for key, new, old in (("entry_price", entry, item.entry_price), ("stop_loss", sl, item.stop_loss),
                                              ("target_1", t1, item.target_1)):
                            if abs(float(new) - float(old)) > 1e-9:
                                payload[key] = float(new)
                        if t2 > 0 and abs(float(t2) - float(item.target_2 or 0.0)) > 1e-9:
                            payload["target_2"] = float(t2)
                        if int(qty) != item.qty:
                            payload["qty"] = int(qty)
                        ui.submit_command(ctx, CommandType.MODIFY_APPROVE, payload, track_key="appr_last_cmd")


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    _auto_refresh()
    ui.tracked_command(ctx, "appr_last_cmd", "Last approval action")

    with ctx.session() as session:
        views = svc.pending_approvals(session, ctx.batch_id, ctx.clock.now())
    st.markdown(f"#### Pending ({len(views)})")
    if not views:
        st.caption("No pending approvals.")
    for view in views:
        _render_item(ctx, view)

    if views:
        with st.expander("Bulk reject"):
            ids = [v.item.id for v in views]
            labels = {v.item.id: f"#{v.item.id} {v.item.symbol} {v.item.mode}" for v in views}
            chosen = st.multiselect("Items", ids, format_func=lambda i: labels[i], key="bulk_ids")
            reason = st.text_input("Reason (optional)", key="bulk_reason")
            ok = ui.confirm_box("Bulk reject", "bulk_confirm")
            if st.button("Reject selected", disabled=not (ok and chosen), key="bulk_btn"):
                ui.submit_command(ctx, CommandType.BULK_REJECT, {"approval_ids": list(chosen), "reason": reason or None},
                                  track_key="appr_last_cmd")

    st.markdown("#### Recently decided")
    with ctx.session() as session:
        history = svc.recent_approval_history(session, ctx.batch_id, limit=20)
    ui.table([{"id": h.id, "symbol": h.symbol, "mode": h.final_mode or h.mode, "status": h.status,
               "decided": ui.fmt_dt(h.decided_at), "by": h.decided_by, "blocked rule": h.block_rule or "",
               "reason": h.user_reason or ""} for h in history])
    st.markdown("#### Approval commands")
    ui.commands_table(ctx, _APPROVAL_COMMANDS, limit=10)
