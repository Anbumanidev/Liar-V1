"""Control 5 - Positions: close (full/partial), tighten SL, targets, trailing, convert, holding, event pin."""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.enums import CommandType, Direction, Mode, TrailingMethod
from dashboard import components as ui
from dashboard import services as svc
from modes.base import conversion_allowed

TITLE = "Positions"
ICON = ":material/account_balance_wallet:"

_POSITION_COMMANDS = (
    CommandType.CLOSE_POSITION, CommandType.TIGHTEN_SL, CommandType.UPDATE_TARGETS, CommandType.UPDATE_TRAILING,
    CommandType.CONVERT_POSITION, CommandType.UPDATE_HOLDING, CommandType.SET_EVENT_PIN,
)
_TRACK = "pos_last_cmd"


def _position_controls(ctx: Any, p: Any) -> None:
    pid = p.id
    ltp = p.last_price
    st.caption(f"opened {ui.fmt_dt(p.opened_at)} | strategy {p.strategy or '-'} | source {p.source} | "
               f"initial SL {p.initial_stop_loss} | margin {ui.fmt_money(p.margin_blocked)} | "
               f"LTP {ltp if ltp is not None else '-'} | uPnL {svc.unrealized_pnl(p)}")

    with st.form(f"close_{pid}"):
        st.markdown("**Close**")
        partial = st.number_input("Qty to close (0 = close fully)", min_value=0, max_value=int(p.qty), value=0,
                                  key=f"close_qty_{pid}")
        reason = st.text_input("Reason (optional)", key=f"close_reason_{pid}")
        if st.form_submit_button("Close position", type="primary"):
            ui.submit_command(ctx, CommandType.CLOSE_POSITION,
                              {"position_id": pid, "qty": int(partial) if partial > 0 else None,
                               "reason": reason or None}, track_key=_TRACK)

    with st.form(f"sl_{pid}"):
        st.markdown(f"**Tighten stop-loss** (current {p.stop_loss:.2f}; tighten only)")
        new_sl = st.number_input("New SL", min_value=0.01, value=float(p.stop_loss), step=0.05, key=f"sl_new_{pid}")
        if st.form_submit_button("Tighten SL"):
            err = svc.validate_sl_tighten(p.direction, float(p.stop_loss), float(new_sl), ltp)
            if err:
                st.error(err)
            else:
                ui.submit_command(ctx, CommandType.TIGHTEN_SL, {"position_id": pid, "new_stop_loss": float(new_sl)},
                                  track_key=_TRACK)

    with st.form(f"tg_{pid}"):
        st.markdown("**Targets**")
        t1 = st.number_input("Target 1 (0 = keep)", min_value=0.0, value=float(p.target_1 or 0.0), step=0.05,
                             key=f"t1_{pid}")
        t2 = st.number_input("Target 2 (0 = keep)", min_value=0.0, value=float(p.target_2 or 0.0), step=0.05,
                             key=f"t2_{pid}")
        if st.form_submit_button("Update targets"):
            errors = []
            for t in (t1, t2):
                if t > 0 and ((p.direction == Direction.LONG.value and t <= p.avg_entry)
                              or (p.direction == Direction.SHORT.value and t >= p.avg_entry)):
                    errors.append(f"Target {t} is on the wrong side of entry {p.avg_entry}.")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                ui.submit_command(ctx, CommandType.UPDATE_TARGETS,
                                  {"position_id": pid, "target_1": t1 if t1 > 0 else None,
                                   "target_2": t2 if t2 > 0 else None}, track_key=_TRACK)

    with st.form(f"tr_{pid}"):
        st.markdown("**Trailing rule**")
        rules = [t.value for t in TrailingMethod]
        rule = st.selectbox("Rule", rules, index=rules.index(p.trailing_rule) if p.trailing_rule in rules else 0,
                            key=f"trail_{pid}")
        mult = st.number_input("ATR multiple (0 = mode default)", min_value=0.0, value=0.0, step=0.1,
                               key=f"trail_mult_{pid}")
        if st.form_submit_button("Update trailing"):
            ui.submit_command(ctx, CommandType.UPDATE_TRAILING,
                              {"position_id": pid, "trailing_rule": rule,
                               "trail_atr_mult": float(mult) if mult > 0 else None}, track_key=_TRACK)

    targets = [m.value for m in (Mode.SWING, Mode.POSITIONAL)
               if conversion_allowed(Mode(p.mode), m, Direction(p.direction))]
    if targets:
        with st.form(f"cv_{pid}"):
            st.markdown("**Convert INTRADAY -> CNC** (long only, before the conversion cutoff; risk re-checked)")
            to_mode = st.selectbox("Convert to", targets, key=f"cv_mode_{pid}")
            if st.form_submit_button("Convert"):
                ui.submit_command(ctx, CommandType.CONVERT_POSITION, {"position_id": pid, "to_mode": to_mode},
                                  track_key=_TRACK)

    if p.mode != Mode.INTRADAY.value:
        with st.form(f"hd_{pid}"):
            st.markdown(f"**Holding period** (held {p.holding_days} day(s))")
            days = st.number_input("Max holding days", min_value=0, max_value=180,
                                   value=int(p.max_holding_days or 0), key=f"hd_days_{pid}")
            if st.form_submit_button("Update holding period"):
                ui.submit_command(ctx, CommandType.UPDATE_HOLDING, {"position_id": pid, "max_holding_days": int(days)},
                                  track_key=_TRACK)

    pinned = st.toggle("Pin: do not auto-exit before event (still subject to SL and HALT)",
                       value=bool(p.pin_no_event_exit), key=f"pin_{pid}")
    if pinned != bool(p.pin_no_event_exit):
        if st.button("Apply pin change", key=f"pin_apply_{pid}"):
            ui.submit_command(ctx, CommandType.SET_EVENT_PIN, {"position_id": pid, "pinned": bool(pinned)},
                              track_key=_TRACK)


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    ui.tracked_command(ctx, _TRACK, "Last position action")
    with ctx.session() as session:
        positions = svc.open_positions(session, ctx.batch_id)
    st.markdown(f"#### Open positions ({len(positions)})")
    ui.table(svc.position_rows(positions))
    for p in positions:
        with st.expander(f"#{p.id} {p.symbol} {p.mode} {p.direction} {p.qty} @ {p.avg_entry:.2f} | SL {p.stop_loss:.2f}"):
            _position_controls(ctx, p)
    st.markdown("#### Recent position commands")
    ui.commands_table(ctx, _POSITION_COMMANDS, limit=15)
