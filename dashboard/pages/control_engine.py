"""Control 1 - Engine: start / pause / resume / stop, per-mode pause, EMERGENCY KILL SWITCH."""

from __future__ import annotations

import streamlit as st

from core.enums import CommandType, Mode
from dashboard import components as ui
from dashboard import services as svc

TITLE = "Engine"
ICON = ":material/power_settings_new:"

_ENGINE_COMMANDS = (
    CommandType.ENGINE_START, CommandType.ENGINE_PAUSE, CommandType.ENGINE_RESUME, CommandType.ENGINE_STOP,
    CommandType.MODE_PAUSE, CommandType.MODE_RESUME, CommandType.KILL_SWITCH, CommandType.RESUME_HALT,
)


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    settings = loaded[1] if loaded else None
    view = ui.engine_view(ctx, settings)

    ui.heartbeat_badge(view)
    if view.state == "UNKNOWN":
        st.warning("No engine heartbeat found. Start the engine process: `python main.py engine` "
                   "(the dashboard cannot start the process itself; ENGINE_START resumes a running engine).")
    with st.expander("Engine details", expanded=False):
        st.write({
            "state": view.state, "heartbeat": ui.fmt_dt(view.heartbeat_at), "started": ui.fmt_dt(view.started_at),
            "pid": view.pid, "activity": view.activity, "next cycle": ui.fmt_dt(view.next_cycle_at),
            "batch": view.batch_id, "settings version": view.settings_version_id,
            "data feed ok": view.data_feed_ok, "LLM ok": view.llm_ok, "market open": view.market_open,
            "last error": view.last_error,
        })

    st.markdown("#### Engine")
    c1, c2 = st.columns(2)
    if c1.button("Start", icon=":material/play_arrow:", width="stretch", key="eng_start"):
        ui.submit_command(ctx, CommandType.ENGINE_START, {}, track_key="eng_last_cmd")
    if c2.button("Pause", icon=":material/pause:", width="stretch", key="eng_pause"):
        ui.submit_command(ctx, CommandType.ENGINE_PAUSE, {}, track_key="eng_last_cmd")
    c3, c4 = st.columns(2)
    if c3.button("Resume", icon=":material/resume:", width="stretch", key="eng_resume"):
        ui.submit_command(ctx, CommandType.ENGINE_RESUME, {}, track_key="eng_last_cmd")
    with c4.popover("Stop", icon=":material/stop:", width="stretch"):
        st.caption("Stops the engine loop. Open positions stay open (monitoring stops).")
        ok = ui.confirm_box("Stop engine", "eng_stop_confirm")
        if st.button("Confirm stop", key="eng_stop", disabled=not ok, type="primary"):
            ui.submit_command(ctx, CommandType.ENGINE_STOP, {}, track_key="eng_last_cmd")
    ui.tracked_command(ctx, "eng_last_cmd")

    st.markdown("#### Modes (pause / resume)")
    with ctx.session() as session:
        states = svc.mode_states(session, ctx.batch_id)
    for mode in Mode:
        state = states.get(mode.value)
        enabled = settings.mode(mode).enabled if settings else None
        status = "disabled in settings" if enabled is False else (
            "PAUSED" if state is not None and state.paused else "active")
        if state is not None and state.paused_until is not None and state.paused_until > ctx.clock.now():
            status += f" (auto-pause until {ui.fmt_dt(state.paused_until)})"
        with st.expander(f"{mode.value}: {status}"):
            if state is not None:
                st.caption(f"consecutive losses: {state.consecutive_losses} | reason: {state.pause_reason or '-'}")
            reason = st.text_input("Reason (optional)", key=f"mode_reason_{mode.value}")
            m1, m2 = st.columns(2)
            if m1.button("Pause mode", key=f"mode_pause_{mode.value}", width="stretch"):
                ui.submit_command(ctx, CommandType.MODE_PAUSE, {"mode": mode.value, "reason": reason or None},
                                  track_key="eng_last_cmd")
            if m2.button("Resume mode", key=f"mode_resume_{mode.value}", width="stretch"):
                ui.submit_command(ctx, CommandType.MODE_RESUME, {"mode": mode.value, "reason": reason or None},
                                  track_key="eng_last_cmd")

    st.markdown("#### :red[EMERGENCY KILL SWITCH]")
    with st.container(border=True):
        st.error("Pauses EVERYTHING, squares off all INTRADAY positions now, and optionally exits all CNC "
                 "positions at the next available price.")
        exit_cnc = st.checkbox("Also exit ALL CNC (BTST/SWING/POSITIONAL) positions", key="kill_exit_cnc")
        reason = st.text_input("Reason", value="Owner kill switch", key="kill_reason")
        ok = ui.confirm_box("Kill switch", "kill_confirm")
        if st.button("ACTIVATE KILL SWITCH", type="primary", disabled=not ok, key="kill_btn", width="stretch",
                     icon=":material/emergency:"):
            ui.submit_command(ctx, CommandType.KILL_SWITCH, {"exit_cnc": exit_cnc, "reason": reason or "Owner kill switch"},
                              track_key="eng_last_cmd")

    st.markdown("#### Recent engine commands")
    ui.commands_table(ctx, _ENGINE_COMMANDS, limit=15)
