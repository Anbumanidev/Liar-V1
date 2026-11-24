"""Control 14 - Data & system: provider, Analyze now, logs, settings history, DB backup/restore, health."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

from core.enums import CommandType, Mode
from dashboard import components as ui
from dashboard import services as svc
from storage.audit import audit, record_error
from storage.db import session_scope

TITLE = "Data & system"
ICON = ":material/settings:"

PROVIDERS = ("yfinance", "angelone", "upstox")
_ANALYZE_KEY = "sys_analyze_cmd"


def _provider_section(ctx: Any, settings: Any) -> None:
    eng = settings.engine
    with st.form("sys_provider_form"):
        st.markdown("#### Data provider (market data only - never orders)")
        opts = list(PROVIDERS) + ([eng.data_provider] if eng.data_provider not in PROVIDERS else [])
        provider = st.selectbox("Primary provider", opts, index=opts.index(eng.data_provider), key="sys_prov")
        fallbacks = st.multiselect("Fallback providers (in order)", list(PROVIDERS),
                                   default=[f for f in eng.fallback_providers if f in PROVIDERS], key="sys_fb")
        backend = st.radio("LLM backend", ["ollama", "rules"], horizontal=True,
                           index=["ollama", "rules"].index(eng.llm_backend), key="sys_llm",
                           help="rules = deterministic interpretation without LLM calls")
        slippage = ui.number_setting(ctx, "Paper slippage %", "engine.slippage_pct", eng.slippage_pct, key="sys_slip",
                                     min_value=0.0, step=0.01)
        note = ui.note_input("sys_prov_note")
        if st.form_submit_button("Save"):
            proposed = {"engine.data_provider": provider, "engine.fallback_providers": fallbacks,
                        "engine.llm_backend": backend, "engine.slippage_pct": float(slippage)}
            ui.save_settings(ctx, proposed, ui.flat_values(settings, proposed.keys()), note)


def _analyze_section(ctx: Any) -> None:
    st.markdown("#### Analyze now")
    with st.form("sys_analyze_form"):
        symbol = st.text_input("Symbol", key="sys_an_sym").strip().upper()
        mode = st.selectbox("Mode", [m.value for m in Mode], key="sys_an_mode")
        route = st.checkbox("Route the result by the mode's execution policy (otherwise analysis only)",
                            key="sys_an_route")
        if st.form_submit_button("Analyze"):
            valid, invalid = svc.parse_symbols(symbol)
            if invalid or len(valid) != 1:
                st.error("Enter exactly one valid symbol.")
            else:
                ui.submit_command(ctx, CommandType.ANALYZE_NOW, {"symbol": valid[0], "mode": mode, "route": route},
                                  track_key=_ANALYZE_KEY)
    view = ui.tracked_command(ctx, _ANALYZE_KEY, "Analysis result")
    if view is not None and view.finished and view.result is not None:
        decision_id = view.result.data.get("decision_id")
        if decision_id is not None:
            with ctx.session() as session:
                details = svc.decision_details(session, int(decision_id))
            ui.json_expander("Decision", details.decision, expanded=True)
            ui.json_expander("Meta", details.meta)
            ui.json_expander("Risk verdict", details.risk)


def _logs_section(ctx: Any) -> None:
    st.markdown("#### Log viewer (JSONL)")
    day = st.date_input("Date", value=ctx.clock.now().date(), key="sys_log_day")
    cats = st.multiselect("Categories", list(svc.LOG_CATEGORIES), default=["errors", "control", "audit"],
                          key="sys_log_cats")
    text = st.text_input("Contains text", key="sys_log_text")
    limit = st.select_slider("Max rows", [50, 100, 200, 500, 1000], value=200, key="sys_log_limit")
    if isinstance(day, date) and cats:
        events = svc.query_logs(day, cats, text=text, limit=int(limit), log_dir=ctx.log_dir)
        st.caption(f"{len(events)} event(s)")
        ui.table([{"ts": e.get("ts", ""), "category": e.get("category", ""),
                   "summary": str({k: v for k, v in e.items() if k not in ("ts", "category")})[:300]}
                  for e in events], key="sys_log_table")


def _settings_history_section(ctx: Any) -> None:
    st.markdown("#### Settings versions (diff & revert)")
    with ctx.session() as session:
        versions = svc.settings_history(session, limit=50)
    if not versions:
        st.caption("No versions yet.")
        return
    ui.table([{"version": v.id, "at": ui.fmt_dt(v.created_at), "author": v.author, "changes": len(v.diff_json or []),
               "reverted from": v.reverted_from_id or "", "note": (v.note or "")[:80]} for v in versions])
    ids = [v.id for v in versions]
    chosen = st.selectbox("Version", ids, key="sys_ver_pick")
    row = next(v for v in versions if v.id == chosen)
    with st.expander(f"Diff of v{chosen} vs previous"):
        ui.table([{"path": d.get("path"), "old": str(d.get("old")), "new": str(d.get("new"))}
                  for d in (row.diff_json or [])])
    if chosen != ids[0]:
        typed = st.text_input(f"Type CONFIRM to revert to v{chosen}", key="sys_ver_confirm")
        if st.button(f"Revert to v{chosen}", key="sys_ver_revert", disabled=not svc.confirmation_ok(typed)):
            try:
                with session_scope(ctx.session_factory) as session:
                    res = svc.revert_settings(session, int(chosen), ctx.caps, actor=ctx.actor,
                                              batch_id=ctx.batch_id, clock=ctx.clock)
            except svc.DashboardError as exc:
                st.error(str(exc))
            else:
                st.success(f"Reverted: new version v{res.version_id}; RELOAD_SETTINGS command #{res.command_id}.")


def _backup_section(ctx: Any, view: svc.EngineStatusView) -> None:
    st.markdown("#### Database backup / restore")
    if svc.sqlite_path_from_url(ctx.db_url) is None:
        st.info("Backup/restore from the dashboard supports file-based SQLite only.")
        return
    if st.button("Back up now", icon=":material/backup:", key="sys_backup"):
        try:
            path = svc.backup_database(ctx.db_url, ctx.backup_dir, ctx.clock.now())
            with session_scope(ctx.session_factory) as session:
                audit(session, ctx.actor, "DB_BACKUP", str(path.name), {"path": str(path)}, clock=ctx.clock)
            st.success(f"Backup written: {path}")
        except svc.DashboardError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            record_error(None, "dashboard.backup", exc)
            st.error(f"Backup failed: {exc}")
    backups = svc.list_backups(ctx.backup_dir)
    if not backups:
        st.caption("No backups yet.")
        return
    ui.table([{"file": b.name, "MB": round(b.stat().st_size / 1_048_576, 2)} for b in backups[:20]])
    with st.expander("Restore a backup"):
        if not view.safe_for_restore:
            st.warning(f"Engine is {view.state} with a fresh heartbeat. Stop the engine first (Engine page).")
        names = [b.name for b in backups]
        choice = st.selectbox("Backup", names, key="sys_restore_pick")
        typed = st.text_input("Type CONFIRM to overwrite the live database", key="sys_restore_confirm")
        if st.button("Restore", key="sys_restore_btn", disabled=not view.safe_for_restore, type="primary"):
            try:
                safety = svc.restore_database(
                    ctx.db_url, Path(ctx.backup_dir) / choice, engine_view=view, confirm_text=typed,
                    backup_dir=ctx.backup_dir, now=ctx.clock.now(), sa_engine=ctx.engine,
                )
                with session_scope(ctx.session_factory) as session:
                    audit(session, ctx.actor, "DB_RESTORE", choice, {"safety_backup": str(safety)}, clock=ctx.clock)
                st.success(f"Restored {choice}. Previous database saved as {safety.name}.")
            except svc.DashboardError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                record_error(None, "dashboard.restore", exc)
                st.error(f"Restore failed: {exc}")


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _vid, settings = loaded
    view = ui.engine_view(ctx, settings)

    st.markdown("#### Health")
    check = st.toggle("Check Ollama reachability", value=False, key="sys_check_ollama")
    with ctx.session() as session:
        health = svc.health_status(session, db_url=ctx.db_url,
                                   ollama_host=svc.ollama_host(settings, ctx.secrets.ollama_host),
                                   now=ctx.clock.now(), stale_seconds=svc.heartbeat_stale_seconds(settings),
                                   check_ollama=check)
    st.write(health)

    _provider_section(ctx, settings)
    _analyze_section(ctx)
    _logs_section(ctx)
    _settings_history_section(ctx)
    _backup_section(ctx, view)
