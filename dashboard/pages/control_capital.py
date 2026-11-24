"""Control 10 - Capital: profit lock, reserve protection, sizing step rules, RESERVE -> TRADING transfer."""

from __future__ import annotations

import streamlit as st

from core.enums import CommandType
from dashboard import components as ui
from dashboard import services as svc

TITLE = "Capital"
ICON = ":material/savings:"


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _vid, settings = loaded
    cap = settings.capital

    st.markdown("#### Balances (selected batch)")
    with ctx.session() as session:
        acct = svc.capital_account(session, ctx.batch_id)
        ledger = svc.recent_ledger(session, ctx.batch_id, limit=15)
    if acct is None:
        st.caption("No capital account for this batch yet.")
    else:
        trading = acct.trading_cash + acct.blocked
        st.write({
            "TRADING (cash + blocked)": ui.fmt_money(trading), "cash": ui.fmt_money(acct.trading_cash),
            "blocked in positions": ui.fmt_money(acct.blocked), "RESERVE": ui.fmt_money(acct.reserve),
            "reserve peak": ui.fmt_money(acct.reserve_peak), "HWM": ui.fmt_money(acct.hwm),
            "sizing capital": ui.fmt_money(acct.sizing_capital), "starting capital": ui.fmt_money(acct.starting_capital),
            "tier": acct.tier, "halted": acct.halted,
        })
    with st.expander("Recent ledger entries"):
        ui.table([{"ts": ui.fmt_dt(e.ts), "type": e.entry_type, "amount": e.amount, "cash": e.trading_cash_after,
                   "blocked": e.blocked_after, "reserve": e.reserve_after, "by": e.actor, "note": e.note or ""}
                  for e in ledger])

    with st.form("capital_rules_form"):
        st.markdown("#### Rules")
        proposed = {
            "capital.profit_lock_pct": float(st.slider(
                "PROFIT_LOCK_PCT - % of positive realized daily net P&L moved to RESERVE", 0.0, 100.0,
                float(cap.profit_lock_pct), step=5.0, key="cap_lock")),
            "capital.reserve_protected_pct": float(st.slider(
                "RESERVE_PROTECTED_PCT - reserve never refilled below this % of its peak", 0.0, 100.0,
                float(cap.reserve_protected_pct), step=5.0, key="cap_prot")),
            "capital.sizing_increase_threshold_pct": float(st.number_input(
                "Sizing increase: TRADING must be >= sizing x (1 + N%)", min_value=0.0,
                value=float(cap.sizing_increase_threshold_pct), step=1.0, key="cap_thr")),
            "capital.sizing_increase_days": int(st.number_input(
                "... for N consecutive trading days (no drawdown tier)", min_value=1,
                value=int(cap.sizing_increase_days), step=1, key="cap_days")),
            "capital.sizing_step_mode": st.selectbox(
                "Sizing step", ["TO_TRADING", "STEP_PCT"], index=["TO_TRADING", "STEP_PCT"].index(cap.sizing_step_mode),
                key="cap_mode", help="TO_TRADING: sizing jumps to the TRADING balance. STEP_PCT: grows by a fixed %."),
            "capital.sizing_step_pct": float(st.number_input(
                "Step % (STEP_PCT)", min_value=0.1, value=float(cap.sizing_step_pct), step=1.0, key="cap_step")),
        }
        st.caption("Sizing capital decreases immediately when TRADING falls below it (shrink fast).")
        note = ui.note_input("cap_note")
        if st.form_submit_button("Save capital rules", type="primary"):
            ui.save_settings(ctx, proposed, ui.flat_values(settings, proposed.keys()), note)

    st.markdown("#### Manual RESERVE -> TRADING transfer")
    if acct is None:
        st.caption("Needs a batch with a capital account.")
        return
    st.caption("Moves locked profit back into trading capital. Confirmed and audit-logged.")
    with st.form("reserve_transfer_form"):
        amount = st.number_input(f"Amount (available reserve {ui.fmt_money(acct.reserve)})", min_value=0.0,
                                 max_value=max(0.0, float(acct.reserve)), value=0.0, step=10.0, key="cap_tr_amt")
        tnote = st.text_input("Note", key="cap_tr_note")
        typed = st.text_input("Type CONFIRM", key="cap_tr_confirm")
        if st.form_submit_button("Transfer"):
            if amount <= 0:
                st.error("Enter an amount above 0.")
            elif amount > acct.reserve + 1e-9:
                st.error("Amount exceeds the reserve balance.")
            elif not svc.confirmation_ok(typed):
                st.error("Type CONFIRM to proceed.")
            else:
                ui.submit_command(ctx, CommandType.RESERVE_TRANSFER,
                                  {"amount": float(amount), "confirm_text": typed.strip(), "note": tnote or None},
                                  track_key="cap_last_cmd")
    ui.tracked_command(ctx, "cap_last_cmd")
