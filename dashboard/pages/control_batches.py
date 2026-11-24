"""Control 13 - Batches: start new, resume HALTED, close, export CSV/Excel, soft delete (double confirm)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.enums import BatchStatus, CommandType
from dashboard import components as ui
from dashboard import services as svc
from dashboard.state import SELECTED_BATCH_KEY
from storage.audit import audit, record_error
from storage.batches import get_current_batch
from storage.db import session_scope
from storage.models import Batch

TITLE = "Batches"
ICON = ":material/inventory_2:"

_TRACK = "batch_last_cmd"


def _new_batch(ctx: Any, current: Any, settings: Any) -> None:
    st.markdown("#### Start a new batch")
    if current is not None:
        st.warning(f"Starting a new batch archives the current batch '{current.name}' ({current.status}). "
                   "History is never deleted.")
    with ctx.session() as session:
        versions = svc.settings_history(session, limit=50)
    version_labels = {v.id: f"v{v.id} - {ui.fmt_dt(v.created_at)} - {v.author}: {(v.note or '')[:40]}"
                      for v in versions}
    with st.form("new_batch_form"):
        name = st.text_input("Name", key="nb_name", max_chars=120)
        capital = st.number_input("Starting capital (Rs)", min_value=100.0, value=float(settings.capital.starting_capital
                                                                                          if settings else 1000.0),
                                  step=100.0, key="nb_cap")
        source = st.radio("Settings", ["Copy current settings", "Copy a previous version"], key="nb_src")
        version_id = st.selectbox("Previous settings version", list(version_labels),
                                  format_func=lambda i: version_labels[i], key="nb_ver") if version_labels else None
        cnc = st.radio("Open CNC positions of the current batch", ["EXIT", "RUN_TO_COMPLETION"], key="nb_cnc",
                       help="EXIT: exit at the next available price. RUN_TO_COMPLETION: they run under the old batch "
                            "in CLOSING state.")
        expected = current.name if current is not None else None
        typed = st.text_input("Type CONFIRM" + (f" or the current batch name '{expected}'" if expected else ""),
                              key="nb_confirm")
        if st.form_submit_button("Start new batch", type="primary"):
            if not name.strip():
                st.error("Name is required.")
            elif not svc.confirmation_ok(typed, batch_name=expected):
                st.error("Confirmation text does not match.")
            else:
                payload = {
                    "name": name.strip(), "starting_capital": float(capital),
                    "copy_settings_from_version_id": (int(version_id) if source.startswith("Copy a previous")
                                                      and version_id is not None else None),
                    "open_cnc_handling": cnc, "confirm_text": typed.strip(),
                }
                ui.submit_command(ctx, CommandType.NEW_BATCH, payload, track_key=_TRACK)


def _batch_actions(ctx: Any, batch: Any) -> None:
    st.markdown(f"#### Selected batch: #{batch.id} {batch.name} [{batch.status}]")
    st.caption(f"starting capital {ui.fmt_money(batch.starting_capital)} | created {ui.fmt_dt(batch.created_at)} | "
               f"ended {ui.fmt_dt(batch.ended_at)}" + (f" | HALT: {batch.halt_reason}" if batch.halt_reason else ""))

    if batch.status == BatchStatus.HALTED.value:
        with st.container(border=True):
            st.error("This batch is HALTED by capital protection (section 8). Resuming re-enables new trades with the "
                     "current floors - make sure you understand why it halted.")
            with st.form("resume_halt_form"):
                typed = st.text_input("Type CONFIRM to resume", key="rh_confirm")
                if st.form_submit_button("Resume HALTED batch"):
                    if not svc.confirmation_ok(typed):
                        st.error("Type CONFIRM.")
                    else:
                        ui.submit_command(ctx, CommandType.RESUME_HALT, {"confirm_text": typed.strip()},
                                          track_key=_TRACK)

    if batch.status in (BatchStatus.ACTIVE.value, BatchStatus.PAUSED.value, BatchStatus.HALTED.value,
                        BatchStatus.CLOSING.value):
        with st.expander("Close batch"):
            with st.form("close_batch_form"):
                st.caption("Closes the batch (open positions are exited by the engine).")
                typed = st.text_input(f"Type CONFIRM or the batch name '{batch.name}'", key="cb_confirm")
                if st.form_submit_button("Close batch"):
                    if not svc.confirmation_ok(typed, batch_name=batch.name):
                        st.error("Confirmation text does not match.")
                    else:
                        ui.submit_command(ctx, CommandType.CLOSE_BATCH,
                                          {"batch_id": batch.id, "confirm_text": typed.strip()}, track_key=_TRACK)

    with st.expander("Export"):
        fmt = st.radio("Format", ["xlsx", "csv"], horizontal=True, key="exp_fmt",
                       format_func=lambda f: "Excel (one sheet per table)" if f == "xlsx" else "CSV files (zip)")
        if st.button("Prepare export", key="exp_prepare"):
            try:
                with session_scope(ctx.session_factory) as session:
                    data, fname, mime = svc.export_batch(session, batch.id, fmt)
                    audit(session, ctx.actor, "BATCH_EXPORT", f"batch:{batch.id}", {"format": fmt, "file": fname},
                          batch_id=batch.id, clock=ctx.clock)
                st.session_state["exp_payload"] = (data, fname, mime)
            except Exception as exc:  # noqa: BLE001
                record_error(None, "dashboard.export", exc, {"batch_id": batch.id})
                st.error(f"Export failed: {exc}")
        payload = st.session_state.get("exp_payload")
        if payload:
            st.download_button("Download " + payload[1], data=payload[0], file_name=payload[1], mime=payload[2],
                               key="exp_download", icon=":material/download:")

    with st.expander("Delete batch (soft delete)"):
        st.caption("Hides the batch from lists. All rows and the audit trail are retained. Only non-current batches "
                   "without open positions can be deleted.")
        with st.form("delete_batch_form"):
            c1 = st.text_input("1) Type CONFIRM", key="del_c1")
            c2 = st.text_input(f"2) Type the batch name '{batch.name}'", key="del_c2")
            if st.form_submit_button("Delete batch"):
                try:
                    with session_scope(ctx.session_factory) as session:
                        svc.soft_delete_batch(session, batch.id, confirm_text=c1, confirm_name=c2, actor=ctx.actor,
                                              clock=ctx.clock)
                except svc.DashboardError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Batch {batch.name} deleted (soft).")
                    st.session_state.pop(SELECTED_BATCH_KEY, None)


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    settings = loaded[1] if loaded else None
    with ctx.session() as session:
        options = svc.batch_options(session)
        current = get_current_batch(session)
        selected = session.get(Batch, ctx.batch_id) if ctx.batch_id else None
    ui.table([{"id": o.id, "name": o.name, "status": o.status, "capital": o.starting_capital,
               "created": ui.fmt_dt(o.created_at)} for o in options])
    ui.tracked_command(ctx, _TRACK, "Last batch action")
    if selected is not None and not selected.deleted:
        _batch_actions(ctx, selected)
    _new_batch(ctx, current, settings)
    st.markdown("#### Recent batch commands")
    ui.commands_table(ctx, (CommandType.NEW_BATCH, CommandType.CLOSE_BATCH, CommandType.RESUME_HALT), limit=10)
