"""Reusable Streamlit widgets for the control pages (mobile friendly: few columns, expanders)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from loguru import logger

from config.loader import get_path, settings_to_dict
from config.schema import Settings
from core.enums import CommandStatus, CommandType
from dashboard import services as svc
from dashboard.auth import ensure_authenticated
from dashboard.state import DashboardContext, get_ctx
from storage.audit import record_error
from storage.db import session_scope

# --------------------------------------------------------------------------- formatting


def fmt_dt(value: datetime | None, with_date: bool = True) -> str:
    """IST timestamp for display."""
    if value is None:
        return "-"
    return value.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


def fmt_money(value: float | None) -> str:
    return "-" if value is None else f"Rs {value:,.2f}"


def table(rows: list[dict[str, Any]] | pd.DataFrame, *, key: str | None = None, height: int | None = None) -> None:
    """Compact full-width table (empty -> caption)."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        st.caption("Nothing to show.")
        return
    kwargs: dict[str, Any] = {"hide_index": True, "width": "stretch"}
    if height:
        kwargs["height"] = height
    if key:
        kwargs["key"] = key
    st.dataframe(df, **kwargs)


# --------------------------------------------------------------------------- page scaffolding


def page_header(title: str, icon: str = "") -> DashboardContext:
    """Auth guard + header + batch caption. Every page's ``render()`` starts with this."""
    ctx = get_ctx()
    ensure_authenticated(ctx)
    st.subheader(f"{icon} {title}".strip())
    st.caption(
        f"PAPER TRADING ONLY | batch filter: {('#' + str(ctx.batch_id)) if ctx.batch_id else 'none'} | "
        f"user: {ctx.user}"
    )
    return ctx


def active_settings(ctx: DashboardContext) -> tuple[int, Settings] | None:
    """Active settings version (seeds v1 on an empty DB). Shows an error and returns None on failure."""
    try:
        with session_scope(ctx.session_factory) as session:
            return svc.load_settings(session, ctx.caps, ctx.clock)
    except Exception as exc:  # noqa: BLE001 - page must degrade, not crash
        record_error(None, "dashboard.settings", exc)
        st.error(f"Could not load settings: {exc}")
        return None


def engine_view(ctx: DashboardContext, settings: Settings | None) -> svc.EngineStatusView:
    with ctx.session() as session:
        return svc.engine_status(session, ctx.clock.now(), svc.heartbeat_stale_seconds(settings))


def heartbeat_badge(view: svc.EngineStatusView, where: Any = None) -> None:
    """Coloured engine state + heartbeat age."""
    target = where or st
    color = "green" if view.alive and view.state == "RUNNING" else ("orange" if view.alive else "red")
    age = "never" if view.age_seconds is None else f"{view.age_seconds:,.0f}s ago"
    target.markdown(f"Engine: :{color}[**{view.label}**]  \nheartbeat {age}")


def flat_values(settings: Settings, paths: Iterable[str]) -> dict[str, Any]:
    """``{dotted path: current value}`` for the given settings paths."""
    data = settings_to_dict(settings)
    return {p: get_path(data, p) for p in paths}


# --------------------------------------------------------------------------- settings widgets


def number_setting(
    ctx: DashboardContext,
    label: str,
    path: str,
    value: float | int,
    *,
    key: str,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    step: float | int | None = None,
    integer: bool = False,
    help: str | None = None,
) -> float | int:
    """Numeric input that shows the current value and the hard cap (and is bounded by it)."""
    cap_min, cap_max = svc.cap_bounds(path, ctx.caps)
    lo = min_value if cap_min is None else (cap_min if min_value is None else max(min_value, cap_min))
    hi = max_value if cap_max is None else (cap_max if max_value is None else min(max_value, cap_max))
    hint = svc.cap_hint(path, ctx.caps)
    full_help = " | ".join(x for x in (help, hint, f"current: {value}") if x)
    if integer:
        conv: Any = int
        default_step: Any = 1
    else:
        conv = float
        default_step = 0.1
    cur = conv(value)
    if lo is not None:
        cur = max(cur, conv(lo))
    if hi is not None:
        cur = min(cur, conv(hi))
    result = st.number_input(
        label + (f"  [{hint.split(' (')[0]}]" if hint else ""),
        value=cur,
        min_value=None if lo is None else conv(lo),
        max_value=None if hi is None else conv(hi),
        step=conv(step if step is not None else default_step),
        key=key,
        help=full_help,
    )
    return conv(result)


def save_settings(
    ctx: DashboardContext,
    proposed: dict[str, Any],
    current: dict[str, Any],
    note: str,
    *,
    extra_validation: str | None = None,
) -> bool:
    """Save only changed values as a new settings version + RELOAD_SETTINGS. Shows the outcome."""
    if extra_validation:
        st.error(extra_validation)
        return False
    changes = svc.changed_only(current, proposed)
    if not changes:
        st.info("No changes to save.")
        return False
    try:
        with session_scope(ctx.session_factory) as session:
            res = svc.apply_settings_changes(
                session, changes, ctx.caps, actor=ctx.actor, note=note or "dashboard edit",
                batch_id=ctx.batch_id, clock=ctx.clock,
            )
    except svc.DashboardError as exc:
        st.error(str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.settings", exc, {"changes": list(changes)})
        st.error(f"Saving settings failed: {exc}")
        return False
    for adj in res.adjustments:
        st.warning(f"Clamped to hard cap: {adj}")
    if res.changed:
        st.success(f"Saved settings version {res.version_id} ({len(changes)} change(s)); "
                   f"RELOAD_SETTINGS queued as command #{res.command_id}. Applies to NEW trades.")
    else:
        st.info("Nothing changed after clamping.")
    return res.changed


def note_input(key: str) -> str:
    return st.text_input("Change note (saved with the version)", key=key, placeholder="why this change?")


# --------------------------------------------------------------------------- commands


def confirm_box(label: str, key: str, *, batch_name: str | None = None) -> bool:
    """Typed confirmation for destructive actions (CONFIRM, or the batch name if given)."""
    extra = f" or the batch name `{batch_name}`" if batch_name else ""
    typed = st.text_input(f"{label} - type CONFIRM{extra}", key=key)
    return svc.confirmation_ok(typed, batch_name=batch_name)


def submit_command(
    ctx: DashboardContext,
    command_type: CommandType,
    payload: dict[str, Any] | None = None,
    *,
    track_key: str | None = None,
    success: str | None = None,
) -> int | None:
    """Enqueue a command for the engine and report it. Optionally remember its id for follow-up."""
    try:
        with session_scope(ctx.session_factory) as session:
            cmd_id = svc.send_command(session, command_type, payload, actor=ctx.actor,
                                      batch_id=ctx.batch_id, clock=ctx.clock)
    except svc.DashboardError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.command", exc, {"type": str(command_type)})
        st.error(f"Could not queue {command_type}: {exc}")
        return None
    if track_key:
        st.session_state[track_key] = cmd_id
    st.success(success or f"{CommandType(command_type).value} queued as command #{cmd_id}. The engine picks it up "
                          f"within a few seconds.")
    logger.info(f"Dashboard queued {command_type} #{cmd_id} by {ctx.actor}")
    return cmd_id


def show_command_result(view: svc.CommandView | None) -> None:
    """Render a command's status; when blocked, show the exact risk rule."""
    if view is None:
        st.caption("Command not found.")
        return
    head = f"#{view.id} {view.command_type}"
    if not view.finished:
        st.info(f"{head}: {view.status} - waiting for the engine...")
        return
    if view.status == CommandStatus.DONE.value and (view.result is None or view.result.ok):
        st.success(f"{head}: DONE. {view.message}")
    elif view.rule:
        st.error(f"{head}: BLOCKED by rule **{view.rule}** - {view.message}")
    else:
        st.error(f"{head}: {view.status}. {view.message}")
    if view.result is not None and view.result.data:
        with st.expander("Result data"):
            st.json(view.result.data, expanded=False)


def tracked_command(ctx: DashboardContext, track_key: str, label: str = "Last request") -> svc.CommandView | None:
    """Show the result of the command id remembered under ``track_key``."""
    cmd_id = st.session_state.get(track_key)
    if not cmd_id:
        return None
    with ctx.session() as session:
        view = svc.command_status(session, int(cmd_id))
    st.markdown(f"**{label}**")
    show_command_result(view)
    if view is not None and not view.finished:
        st.button("Refresh status", key=f"{track_key}_refresh", icon=":material/refresh:")
    return view


def commands_table(ctx: DashboardContext, types: Iterable[CommandType] | None = None, limit: int = 15) -> None:
    """Recent commands with status and result/rule."""
    with ctx.session() as session:
        views = svc.recent_command_views(session, limit=limit, types=list(types) if types else None)
    table([
        {"id": v.id, "type": v.command_type, "status": v.status, "at": fmt_dt(v.created_at),
         "by": v.requested_by, "rule": v.rule or "", "message": v.message[:160]}
        for v in views
    ])


def json_expander(label: str, data: Any, expanded: bool = False) -> None:
    with st.expander(label, expanded=expanded):
        if data:
            st.json(data, expanded=False)
        else:
            st.caption("No data.")


def require_batch(ctx: DashboardContext) -> int | None:
    """Return the selected batch id or show a hint."""
    if ctx.batch_id is None:
        st.info("No batch selected / no batch exists yet. Start one on the Batches page.")
    return ctx.batch_id


# ----------------------------------------------------------------------------- autonomy

AUTONOMY_HELP: dict[str, tuple[str, str]] = {
    "PER_MODE": ("Per mode", "Each mode uses its own execution policy (Settings > Modes)."),
    "HUMAN_IN_THE_LOOP": ("Human in the loop", "Every trade waits for your Approve / Reject. Nothing executes "
                                               "without you."),
    "HYBRID": ("Hybrid", "Top setups (grade and confidence at or above your thresholds) execute automatically; "
                         "everything else waits for your approval."),
    "FULL_AUTO": ("Fully automatic", "Every trade the risk manager approves executes immediately - no human. "
                                     "The risk manager and capital protection still veto everything."),
    "SIGNALS_ONLY": ("Signals only", "Trades are logged and emailed, never executed."),
}


def autonomy_control(ctx: DashboardContext, settings: Settings, *, key: str = "autonomy") -> None:
    """Global 'fully automatic vs human in the loop' switch (PAPER trades only)."""
    from core.enums import Autonomy, Grade, Mode

    current = settings.modes.autonomy.value
    options = [a.value for a in Autonomy]
    locked = not ctx.caps.allow_full_auto
    choice = st.radio(
        "Who decides?", options, index=options.index(current), key=f"{key}_choice", horizontal=True,
        format_func=lambda v: AUTONOMY_HELP[v][0] + (" (locked)" if locked and v in ("FULL_AUTO", "HYBRID") else ""),
        help="Applies to PAPER trades only. The risk manager vetoes every trade in every setting.",
    )
    st.caption(AUTONOMY_HELP[choice][1])
    proposed: dict[str, Any] = {"modes.autonomy": choice}
    current_vals: dict[str, Any] = {"modes.autonomy": current}
    if choice == "HYBRID":
        grades = [g.value for g in (Grade.A_PLUS, Grade.A, Grade.B)]
        c1, c2 = st.columns(2)
        g = c1.selectbox("Auto-execute grade at least", grades,
                         index=grades.index(settings.modes.hybrid_auto_min_grade.value)
                         if settings.modes.hybrid_auto_min_grade.value in grades else 0, key=f"{key}_grade")
        conf = c2.number_input("and confidence at least", 0, 100, int(settings.modes.hybrid_auto_min_confidence),
                               key=f"{key}_conf")
        proposed |= {"modes.hybrid_auto_min_grade": g, "modes.hybrid_auto_min_confidence": int(conf)}
        current_vals |= {"modes.hybrid_auto_min_grade": settings.modes.hybrid_auto_min_grade.value,
                         "modes.hybrid_auto_min_confidence": settings.modes.hybrid_auto_min_confidence}
    if locked and choice in ("FULL_AUTO", "HYBRID"):
        st.error("Locked by config/hard_caps.yaml (allow_full_auto: false). Change it on disk to enable.")
        return
    needs_confirm = choice in ("FULL_AUTO", "HYBRID") and choice != current
    ok = True
    if needs_confirm:
        st.warning("Automatic execution: trades will be placed (on PAPER) without asking you.")
        ok = confirm_box("Type CONFIRM to switch to automatic execution", f"{key}_confirm")
    if choice != current or proposed != current_vals:
        if st.button("Apply", key=f"{key}_apply", type="primary", disabled=not ok):
            save_settings(ctx, proposed, current_vals, f"autonomy -> {choice}")
    rows = []
    for m in Mode:
        cfg = settings.mode(m)
        if not cfg.enabled:
            continue
        rows.append({"mode": m.value, "mode policy": cfg.execution_policy.value,
                     "effective now": settings.modes.effective_policy(m).value
                     if choice == current else "(after Apply)"})
    if rows:
        table(rows, key=f"{key}_eff")
