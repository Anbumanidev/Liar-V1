"""Control 4 - Manual trade (tagged MANUAL; same risk-manager checks; optional 'Analyze first')."""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.enums import CommandType, Direction, EntryType, Mode
from dashboard import components as ui
from dashboard import services as svc
from modes.base import direction_allowed

TITLE = "Manual trade"
ICON = ":material/add_shopping_cart:"

_ANALYZE_KEY = "manual_analyze_cmd"
_TRADE_KEY = "manual_trade_cmd"


def _show_analysis(ctx: Any) -> None:
    view = ui.tracked_command(ctx, _ANALYZE_KEY, "Analysis")
    if view is None or not view.finished or view.result is None:
        return
    decision_id = view.result.data.get("decision_id")
    if decision_id is None:
        return
    with ctx.session() as session:
        details = svc.decision_details(session, int(decision_id))
    d = details.decision or {}
    st.markdown(f"**Decision:** {d.get('decision', '-')} {d.get('direction') or ''} | confidence "
                f"{d.get('confidence', '-')} | grade {d.get('grade') or '-'} | entry {d.get('entry_price')} "
                f"SL {d.get('stop_loss')} T1 {d.get('target_1')} T2 {d.get('target_2')} | risk "
                f"{'OK' if d.get('risk_approved') else d.get('risk_rule') or '-'}")
    ui.json_expander("Meta reasoning", details.meta)
    ui.json_expander("Agent reports", [{"agent": r["agent"], "signal": r["signal"], "confidence": r["confidence"],
                                        "report": r.get("report_json")} for r in details.agent_reports])


def _preview(ctx: Any, mode: str, direction: str, entry: float, qty: int, sl: float, t1: float, t2: float) -> None:
    """Charges + tax + net at SL / T1 / T2 before submitting (estimates, not tax advice)."""
    if entry <= 0 or sl <= 0 or t1 <= 0:
        st.info("Enter entry, stop-loss and target 1 to preview charges and tax.")
        return
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    from tax.engine import pre_trade_estimate

    settings = loaded[1]
    est = pre_trade_estimate(Mode(mode), Direction(direction), qty, entry, sl, t1, t2 or None, settings.charges,
                             settings.tax)
    st.markdown(f"**{est.head.value}** at ~{est.rate_pct:.2f}% incl. cess · charges {est.charges_pct_of_value:.2f}% "
                f"of value · breakeven move {est.breakeven_move_pct:.2f}%"
                + (f" · after-tax R:R {est.post_tax_rr}" if est.post_tax_rr is not None else ""))
    ui.table([{"exit": x.label, "price": x.exit_price, "gross": x.gross, "charges": x.charges, "tax": x.tax,
               "you keep": x.net_after_tax} for x in est.scenarios], key="mt_tax_preview")
    if est.charges_pct_of_value > settings.risk.max_charges_pct_of_value:
        st.warning(f"Charges exceed {settings.risk.max_charges_pct_of_value}% of value: the risk manager will reject "
                   "this (CHARGES_EXCEED_EDGE).")


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    if ui.require_batch(ctx) is None:
        return
    st.caption("Your trade goes through the same deterministic risk manager. If blocked, the exact rule is shown.")

    symbol = st.text_input("Symbol (NSE)", key="mt_symbol", placeholder="e.g. RELIANCE").strip().upper()
    mode = st.selectbox("Mode", [m.value for m in Mode], key="mt_mode")
    directions = [d.value for d in Direction if direction_allowed(Mode(mode), d)]
    direction = st.radio("Side", directions, horizontal=True, key="mt_dir",
                         help="BTST / SWING / POSITIONAL are LONG only (no overnight shorts).")
    if direction not in directions:  # stale widget value after switching mode
        direction = directions[0]

    if st.button("Analyze first", icon=":material/psychology:", key="mt_analyze", disabled=not symbol):
        valid, invalid = svc.parse_symbols(symbol)
        if invalid or not valid:
            st.error(f"Invalid symbol: {symbol}")
        else:
            ui.submit_command(ctx, CommandType.ANALYZE_NOW, {"symbol": valid[0], "mode": mode, "route": False},
                              track_key=_ANALYZE_KEY)
    _show_analysis(ctx)

    with st.form("manual_trade_form"):
        etype = st.selectbox("Entry type", [e.value for e in EntryType], key="mt_etype")
        entry = st.number_input("Entry / reference price (required for LIMIT)", min_value=0.0, value=0.0,
                                step=0.05, key="mt_entry")
        qty = st.number_input("Quantity", min_value=1, value=1, step=1, key="mt_qty")
        sl = st.number_input("Stop-loss (mandatory)", min_value=0.0, value=0.0, step=0.05, key="mt_sl")
        t1 = st.number_input("Target 1", min_value=0.0, value=0.0, step=0.05, key="mt_t1")
        t2 = st.number_input("Target 2 (0 = none)", min_value=0.0, value=0.0, step=0.05, key="mt_t2")
        analyze_first = st.checkbox("Run the full agent pipeline before executing (analyze_first)", key="mt_af")
        note = st.text_input("Note", key="mt_note")
        c_prev, c_sub = st.columns(2)
        preview = c_prev.form_submit_button("Preview charges & tax")
        submitted = c_sub.form_submit_button("Submit manual trade", type="primary")
    if preview:
        _preview(ctx, mode, direction, float(entry), int(qty), float(sl), float(t1), float(t2))
    if submitted:
        errors: list[str] = []
        valid, invalid = svc.parse_symbols(symbol)
        if invalid or len(valid) != 1:
            errors.append("Enter exactly one valid NSE symbol.")
        if not direction_allowed(Mode(mode), Direction(direction)):
            errors.append(f"{direction} is not allowed in {mode} (long only).")
        if etype == EntryType.LIMIT.value and entry <= 0:
            errors.append("LIMIT orders need an entry price.")
        errors += svc.validate_trade_levels(direction, float(entry) if entry > 0 else None, float(sl), float(t1),
                                            float(t2) if t2 > 0 else None)
        if errors:
            for e in errors:
                st.error(e)
        else:
            payload: dict[str, Any] = {
                "symbol": valid[0], "mode": mode, "direction": direction, "entry_type": etype,
                "entry_price": float(entry) if entry > 0 else None, "qty": int(qty), "stop_loss": float(sl),
                "target_1": float(t1), "target_2": float(t2) if t2 > 0 else None,
                "analyze_first": bool(analyze_first), "note": note or None,
            }
            ui.submit_command(ctx, CommandType.MANUAL_TRADE, payload, track_key=_TRADE_KEY)
    ui.tracked_command(ctx, _TRADE_KEY, "Manual trade result")

    st.markdown("#### Recent manual trades / analyses")
    ui.commands_table(ctx, (CommandType.MANUAL_TRADE, CommandType.ANALYZE_NOW), limit=10)
