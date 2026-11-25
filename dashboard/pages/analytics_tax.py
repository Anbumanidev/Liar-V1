"""Tax & charges - FY tax estimate (with set-off rules), tips to reduce tax/charges, and a
pre-trade calculator (charges + tax + net for SL / T1 / T2). Estimates only, not tax advice."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
from sqlalchemy import select

from core.enums import Direction, Mode, PositionStatus
from dashboard import components as ui
from storage.audit import record_error
from storage.models import Position, Trade
from tax.engine import DISCLAIMER, fy_bounds, pre_trade_estimate, summarize, tax_tips, trade_tax_lines

TITLE = "Tax & charges"
ICON = ":material/receipt_long:"


def _fy_options(trades: list[Trade], today: date) -> list[date]:
    days = {fy_bounds(today)[0]} | {fy_bounds(t.exit_time.date())[0] for t in trades}
    return sorted(days, reverse=True)


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _, settings = loaded
    today = ctx.clock.now().date()
    st.caption(DISCLAIMER)
    tab_fy, tab_calc, tab_set = st.tabs(["FY estimate & tips", "Before-trade calculator", "Tax settings"])

    with tab_fy:
        if ctx.batch_id is None:
            st.info("No batch selected.")
        else:
            try:
                with ctx.session() as session:
                    trades = list(session.execute(select(Trade).where(Trade.batch_id == ctx.batch_id)).scalars())
                    open_pos = list(session.execute(select(Position).where(
                        Position.batch_id == ctx.batch_id, Position.status == PositionStatus.OPEN.value)).scalars())
            except Exception as exc:  # noqa: BLE001
                record_error(None, "dashboard.tax", exc)
                st.error(f"Could not load trades: {exc}")
                trades, open_pos = [], []
            fys = _fy_options(trades, today)
            fy_start = st.selectbox("Financial year", fys, format_func=lambda d: fy_bounds(d)[2], key="tax_fy")
            lines = trade_tax_lines(trades, settings.charges, settings.tax)
            s = summarize(lines, settings.tax, fy_start)
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated tax", f"Rs {s.total_tax:,.2f}",
                      f"{s.effective_tax_pct}% of net" if s.effective_tax_pct is not None else None, delta_color="off")
            c2.metric("Net P&L before tax", f"Rs {s.net_profit_before_tax:,.2f}")
            c3.metric("Net P&L after tax", f"Rs {s.net_profit_after_tax:,.2f}")
            c4, c5, c6 = st.columns(3)
            c4.metric("STT paid", f"Rs {s.stt_paid:,.2f}")
            c5.metric("Other charges", f"Rs {s.other_charges:,.2f}")
            c6.metric("Trades in FY", str(s.n_trades))
            heads = pd.DataFrame([
                {"head": "Intraday (speculative, slab)", "net / taxable": s.speculative_net, "tax": s.tax_speculative},
                {"head": "Business (delivery, if chosen)", "net / taxable": s.business_net, "tax": s.tax_business},
                {"head": f"STCG ({settings.tax.stcg_rate_pct:g}%)", "net / taxable": s.stcg_taxable, "tax": s.tax_stcg},
                {"head": f"LTCG ({settings.tax.ltcg_rate_pct:g}% above exemption)", "net / taxable": s.ltcg_taxable,
                 "tax": s.tax_ltcg},
                {"head": "Surcharge + cess", "net / taxable": None, "tax": round(s.surcharge + s.cess, 2)},
            ])
            ui.table(heads, key="tax_heads")
            cf = {k: v for k, v in (("Speculative loss (4 yrs)", s.carry_forward_speculative_loss),
                                    ("Business loss (8 yrs)", s.carry_forward_business_loss),
                                    ("Short-term capital loss (8 yrs)", s.carry_forward_st_capital_loss),
                                    ("Long-term capital loss (8 yrs)", s.carry_forward_lt_capital_loss)) if v}
            if cf:
                st.markdown("**Losses to carry forward** (file your return on time):")
                ui.table([{"loss": k, "amount": v} for k, v in cf.items()], key="tax_cf")
            for n in s.notes:
                st.caption(n)
            st.markdown("#### Tips to reduce tax and charges")
            for tip in tax_tips(s, open_pos, today, settings.tax, settings.charges):
                text = f"**{tip.title}** - {tip.detail}" + (
                    f"  \n*Estimated saving: Rs {tip.est_saving:,.2f}*" if tip.est_saving else "")
                {"saving": st.success, "warning": st.warning}.get(tip.severity, st.info)(text)
            with st.expander("Per-trade tax lines"):
                start, end, _ = fy_bounds(fy_start)
                rows = [ln.as_dict() for ln in lines if start <= ln.exit_date <= end]
                ui.table(rows or [{"info": "no trades in this FY"}], key="tax_lines")

    with tab_calc:
        st.markdown("Check charges and tax **before** you trade (or while approving).")
        with st.form("tax_calc"):
            c1, c2 = st.columns(2)
            mode = c1.selectbox("Mode", [m.value for m in Mode], key="tc_mode")
            direction = c2.selectbox("Side", ["LONG", "SHORT"], key="tc_dir",
                                     help="SHORT is only possible intraday (no overnight shorts).")
            c3, c4 = st.columns(2)
            qty = c3.number_input("Quantity", 1, 100_000, 2, key="tc_qty")
            entry = c4.number_input("Entry price", 0.05, 1_000_000.0, 500.0, step=0.05, key="tc_entry")
            c5, c6, c7 = st.columns(3)
            sl = c5.number_input("Stop-loss", 0.05, 1_000_000.0, 490.0, step=0.05, key="tc_sl")
            t1 = c6.number_input("Target 1", 0.05, 1_000_000.0, 520.0, step=0.05, key="tc_t1")
            t2 = c7.number_input("Target 2 (0 = none)", 0.0, 1_000_000.0, 0.0, step=0.05, key="tc_t2")
            go = st.form_submit_button("Calculate")
        if go:
            if direction == "SHORT" and mode != Mode.INTRADAY.value:
                st.error("Short selling is only allowed intraday (no overnight shorts in the cash market).")
            else:
                est = pre_trade_estimate(Mode(mode), Direction(direction), int(qty), float(entry), float(sl),
                                         float(t1), float(t2) or None, settings.charges, settings.tax)
                st.markdown(f"Tax head: **{est.head.value}** at ~{est.rate_pct:.2f}% (incl. cess) · charges "
                            f"{est.charges_pct_of_value:.2f}% of position value · price must move "
                            f"{est.breakeven_move_pct:.2f}% just to pay the charges"
                            + (f" · after-tax R:R {est.post_tax_rr}" if est.post_tax_rr is not None else ""))
                ui.table([sc.__dict__ for sc in est.scenarios], key="tc_table")
                for n in est.notes:
                    st.caption(n)
                if est.charges_pct_of_value > settings.risk.max_charges_pct_of_value:
                    st.warning(f"Charges exceed {settings.risk.max_charges_pct_of_value}% of the position value - the "
                               "risk manager would reject this (CHARGES_EXCEED_EDGE). Increase size or use intraday.")

    with tab_set:
        cur = {"tax.slab_rate_pct": settings.tax.slab_rate_pct,
               "tax.delivery_as_business": settings.tax.delivery_as_business,
               "tax.basic_exemption_unused": settings.tax.basic_exemption_unused,
               "tax.surcharge_pct": settings.tax.surcharge_pct,
               "tax.show_in_approvals": settings.tax.show_in_approvals}
        with st.form("tax_settings"):
            slab = st.number_input("Your slab rate for intraday / business income (%)", 0.0, 50.0,
                                   float(cur["tax.slab_rate_pct"]), step=1.0)
            biz = st.checkbox("Treat delivery trades as business income (ask your CA)",
                              bool(cur["tax.delivery_as_business"]))
            basic = st.number_input("Unused basic exemption usable against capital gains (Rs)", 0.0, 1e7,
                                    float(cur["tax.basic_exemption_unused"]), step=1000.0)
            sur = st.number_input("Surcharge (%)", 0.0, 37.0, float(cur["tax.surcharge_pct"]), step=1.0)
            show = st.checkbox("Show tax estimate on approval cards", bool(cur["tax.show_in_approvals"]))
            note = ui.note_input("tax_note")
            if st.form_submit_button("Save tax settings"):
                ui.save_settings(ctx, {"tax.slab_rate_pct": slab, "tax.delivery_as_business": biz,
                                       "tax.basic_exemption_unused": basic, "tax.surcharge_pct": sur,
                                       "tax.show_in_approvals": show}, cur, note)
        st.caption(f"Rates: STCG {settings.tax.stcg_rate_pct:g}% · LTCG {settings.tax.ltcg_rate_pct:g}% above "
                   f"Rs {settings.tax.ltcg_exemption:,.0f} · cess {settings.tax.cess_pct:g}% · last verified "
                   f"{settings.tax.last_verified} (config/tax.yaml).")
