"""Control 2 - Modes: enable/disable, allocation, selection & execution policy, timeouts, windows, thresholds."""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.enums import ExecutionPolicy, Grade, Mode, SelectionPolicy
from dashboard import components as ui
from dashboard import services as svc
from modes.base import get_mode_spec

TITLE = "Modes"
ICON = ":material/tune:"


def _mode_paths(mode: Mode) -> dict[str, str]:
    base = f"modes.modes.{mode.value}"
    return {k: f"{base}.{k}" for k in (
        "enabled", "allocation_pct", "execution_policy", "approval_timeout_minutes",
        "approval_until_next_session", "max_trades_per_day", "min_holding_days", "max_holding_days",
        "entry_windows", "confidence_threshold", "min_confluence", "min_rr", "min_grade",
    )}


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    version_id, settings = loaded
    st.caption(f"Settings version {version_id}. Changes apply to NEW trades; open positions keep their settings.")

    # ---- allocations + selection policy (one form so the sum can be validated together)
    with st.form("modes_alloc_form"):
        st.markdown("#### Enable & allocate")
        sel = st.radio("Mode selection policy", [p.value for p in SelectionPolicy], horizontal=True,
                       index=[p.value for p in SelectionPolicy].index(settings.modes.selection_policy.value),
                       help="FIXED: analyse only for the configured enabled modes. AUTO: the meta agent "
                            "recommends the best enabled mode for each setup.")
        proposed: dict[str, Any] = {"modes.selection_policy": sel}
        allocs: dict[str, float] = {}
        for mode in Mode:
            cfg = settings.mode(mode)
            spec = get_mode_spec(mode)
            st.markdown(f"**{mode.value}** - {spec.product.value}, {spec.description}")
            proposed[f"modes.modes.{mode.value}.enabled"] = st.toggle(
                "Enabled", value=cfg.enabled, key=f"en_{mode.value}")
            allocs[mode.value] = st.slider("Allocation % of trading capital", 0.0, 100.0, float(cfg.allocation_pct),
                                           step=5.0, key=f"alloc_{mode.value}")
            proposed[f"modes.modes.{mode.value}.allocation_pct"] = allocs[mode.value]
        total = svc.allocation_total(allocs)
        st.caption(f"Allocation total: {total:.1f}% (must be <= 100%)")
        note = ui.note_input("modes_alloc_note")
        if st.form_submit_button("Save modes & allocations", type="primary"):
            current = ui.flat_values(settings, proposed.keys())
            ui.save_settings(ctx, proposed, current, note, extra_validation=svc.validate_allocations(allocs))

    # ---- per-mode policy details
    st.markdown("#### Per-mode policy")
    for mode in Mode:
        cfg = settings.mode(mode)
        paths = _mode_paths(mode)
        with st.expander(f"{mode.value} - {cfg.execution_policy.value}"
                         f"{'' if cfg.enabled else ' (disabled)'}"):
            with st.form(f"mode_form_{mode.value}"):
                p: dict[str, Any] = {}
                policies = [x.value for x in ExecutionPolicy]
                p[paths["execution_policy"]] = st.selectbox(
                    "Execution policy", policies, index=policies.index(cfg.execution_policy.value),
                    key=f"pol_{mode.value}",
                    help="AUTO: execute after risk approval. MANUAL_APPROVE: approval queue. "
                         "SIGNALS_ONLY: log + email, never execute.")
                timeout_kind = st.radio(
                    "Approval timeout", ["minutes", "until next session"], horizontal=True,
                    index=1 if cfg.approval_until_next_session else 0, key=f"tok_{mode.value}")
                minutes = st.number_input("Timeout minutes", 1, 24 * 60, int(cfg.approval_timeout_minutes or 3),
                                          key=f"tom_{mode.value}")
                until = st.text_input("Until next session at (HH:MM IST)",
                                      value=cfg.approval_until_next_session or "09:45", key=f"tou_{mode.value}")
                p[paths["max_trades_per_day"]] = st.number_input(
                    "Max trades per day", 0, 50, int(cfg.max_trades_per_day), key=f"mtd_{mode.value}")
                if mode != Mode.INTRADAY:
                    p[paths["min_holding_days"]] = st.number_input(
                        "Min holding days", 0, 180, int(cfg.min_holding_days), key=f"minh_{mode.value}")
                    p[paths["max_holding_days"]] = st.number_input(
                        "Max holding days", 0, 180, int(cfg.max_holding_days), key=f"maxh_{mode.value}")
                windows_text = st.text_input("Entry windows (HH:MM-HH:MM, comma separated)",
                                             value=svc.format_windows(cfg.entry_windows), key=f"win_{mode.value}")
                p[paths["confidence_threshold"]] = st.slider(
                    "Meta confidence threshold", 0, 100, int(cfg.confidence_threshold), key=f"conf_{mode.value}")
                p[paths["min_confluence"]] = st.slider(
                    "Min confluence score", 0.0, 100.0, float(cfg.min_confluence), step=1.0, key=f"mcf_{mode.value}")
                p[paths["min_rr"]] = ui.number_setting(
                    ctx, "Min R:R", paths["min_rr"], cfg.min_rr, key=f"rr_{mode.value}", max_value=10.0, step=0.1,
                    help="Cannot go below the hard floor min_rr_floor.")
                grades = [g.value for g in Grade]
                p[paths["min_grade"]] = st.selectbox("Min grade", grades, index=grades.index(cfg.min_grade.value),
                                                     key=f"grade_{mode.value}")
                note = ui.note_input(f"mode_note_{mode.value}")
                if st.form_submit_button(f"Save {mode.value}"):
                    try:
                        p[paths["entry_windows"]] = svc.parse_windows(windows_text)
                        if timeout_kind == "minutes":
                            p[paths["approval_timeout_minutes"]] = int(minutes)
                            p[paths["approval_until_next_session"]] = None
                        else:
                            p[paths["approval_until_next_session"]] = svc.validate_hhmm(until)
                            p[paths["approval_timeout_minutes"]] = None
                    except svc.DashboardError as exc:
                        st.error(str(exc))
                    else:
                        current = ui.flat_values(settings, p.keys())
                        ui.save_settings(ctx, p, current, note)
