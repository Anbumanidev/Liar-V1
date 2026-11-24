"""Control 9 - Risk: every section 5 / 8 limit, within the hard caps (caps shown, inputs clamped)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

from core.enums import Grade
from dashboard import components as ui
from dashboard import services as svc

TITLE = "Risk"
ICON = ":material/shield:"


@dataclass(frozen=True)
class Field:
    path: str
    label: str
    kind: str = "float"   # float | int | bool | time
    min: float | None = None
    max: float | None = None
    step: float | None = None
    help: str | None = None


GROUPS: dict[str, list[Field]] = {
    "Per trade & positions (section 5)": [
        Field("risk.risk_per_trade_pct", "Risk per trade % of mode sizing capital", min=0.05, step=0.05),
        Field("risk.max_open_positions", "Max open positions (normal capital)", "int", 1, 50),
        Field("risk.max_position_pct", "Max % of trading capital per position", min=1.0, max=100.0, step=1.0),
        Field("risk.max_positions_per_sector", "Max positions per sector", "int", 1, 20),
        Field("risk.daily_loss_limit_pct", "Daily loss limit % (no new entries that day)", min=0.1, step=0.1),
    ],
    "Small-capital rules (while trading capital < threshold)": [
        Field("risk.small_capital_threshold", "Threshold (Rs)", min=0.0, step=500.0),
        Field("risk.small_max_open_positions", "Max open positions", "int", 1, 20),
        Field("risk.small_max_position_pct", "Max % of trading capital per position", min=1.0, max=100.0, step=1.0),
    ],
    "Intraday timing": [
        Field("risk.intraday_no_entry_before", "No entries before", "time"),
        Field("risk.intraday_no_entry_after", "No entries after", "time"),
        Field("risk.intraday_squareoff", "Square-off time", "time"),
        Field("risk.conversion_cutoff", "INTRADAY -> CNC conversion cutoff", "time"),
        Field("risk.intraday_daily_profit_lock_pct", "Daily profit lock % (then A+ only, half size)", min=0.1, step=0.1),
        Field("risk.intraday_profit_giveback_pct", "Stop intraday after giving back % of day peak", min=1.0,
              max=100.0, step=5.0),
    ],
    "Overnight & events": [
        Field("risk.max_overnight_exposure_pct", "Max overnight exposure % of trading capital", min=0.0, max=100.0,
              step=5.0),
        Field("risk.gap_buffer_pct", "Gap buffer % (overnight risk floor)", min=0.0, max=20.0, step=0.5),
        Field("risk.avoid_events_in_window", "Avoid entries with an event inside the holding window", "bool"),
    ],
    "Trade validity": [
        Field("risk.circuit_buffer_pct", "Circuit buffer % (reject near circuit limits)", min=0.0, max=10.0, step=0.5),
        Field("risk.sl_min_atr", "Min SL distance (x ATR)", min=0.05, step=0.05),
        Field("risk.sl_max_atr", "Max SL distance (x ATR)", min=0.1, step=0.1),
        Field("risk.max_charges_pct_of_value", "Max round-trip charges % of position value", min=0.05, max=10.0,
              step=0.05),
        Field("risk.max_correlation", "Max correlation with open positions", min=0.05, max=1.0, step=0.05),
        Field("risk.correlation_lookback_days", "Correlation lookback (days)", "int", 5, 250),
        Field("risk.stale_price_seconds", "Stale price limit (s)", "int", 10, 3600),
    ],
    "Capital protection (section 8)": [
        Field("risk.max_loss_pct", "MAX_LOSS_PCT", min=0.5, step=0.5, help="Absolute ceiling 10 % (hard-coded)."),
        Field("risk.hard_stop_pct", "HARD_STOP_PCT (floor; must be < MAX_LOSS_PCT)", min=0.5, step=0.5),
        Field("risk.loss_budget_divisor", "Loss budget divisor (new trade max risk = budget / N)", min=1.0, step=0.5),
        Field("risk.drawdown.caution_pct", "CAUTION tier from HWM drawdown %", min=0.5, step=0.5),
        Field("risk.drawdown.defensive_pct", "DEFENSIVE tier from HWM drawdown %", min=0.5, step=0.5),
        Field("risk.drawdown.caution_size_mult", "CAUTION size multiplier", min=0.05, max=1.0, step=0.05),
        Field("risk.drawdown.defensive_size_mult", "DEFENSIVE size multiplier", min=0.05, max=1.0, step=0.05),
        Field("risk.drawdown.defensive_max_open_positions", "DEFENSIVE max open positions", "int", 0, 10),
        Field("risk.drawdown.defensive_allow_new_overnight", "DEFENSIVE allows new overnight positions", "bool"),
    ],
}


def _field_input(ctx: Any, f: Field, value: Any) -> Any:
    key = f"risk_{f.path}"
    if f.kind == "bool":
        return st.toggle(f.label, value=bool(value), key=key, help=f.help)
    if f.kind == "time":
        return st.text_input(f"{f.label} (HH:MM IST)", value=str(value), key=key, help=f.help)
    return ui.number_setting(ctx, f.label, f.path, value, key=key, min_value=f.min, max_value=f.max, step=f.step,
                             integer=f.kind == "int", help=f.help)


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _vid, settings = loaded
    caps = ctx.caps
    st.caption(f"Hard caps (config/hard_caps.yaml, NOT editable here): max loss {caps.max_loss_pct}% | "
               f"max risk/trade {caps.max_risk_per_trade_pct}% | max daily loss {caps.max_daily_loss_limit_pct}% | "
               f"max MIS leverage {caps.max_mis_leverage}x | min R:R floor {caps.min_rr_floor} | "
               f"paper only | no overnight shorts | SL tighten-only | no averaging down.")

    for group, fields in GROUPS.items():
        with st.expander(group, expanded=False):
            with st.form(f"risk_form_{group}"):
                current = ui.flat_values(settings, [f.path for f in fields])
                proposed = {f.path: _field_input(ctx, f, current[f.path]) for f in fields}
                if group.startswith("Capital protection"):
                    grades = [g.value for g in Grade if g != Grade.C]
                    cg = st.multiselect("CAUTION allowed grades", grades,
                                        default=[g.value for g in settings.risk.drawdown.caution_grades],
                                        key="risk_caution_grades")
                    dg = st.multiselect("DEFENSIVE allowed grades", grades,
                                        default=[g.value for g in settings.risk.drawdown.defensive_grades],
                                        key="risk_def_grades")
                    proposed["risk.drawdown.caution_grades"] = cg
                    proposed["risk.drawdown.defensive_grades"] = dg
                    current.update(ui.flat_values(settings, ["risk.drawdown.caution_grades",
                                                             "risk.drawdown.defensive_grades"]))
                if group.startswith("Overnight"):
                    types_text = st.text_input("Event types to avoid (comma separated)",
                                               value=", ".join(settings.risk.avoid_event_types), key="risk_evt")
                    proposed["risk.avoid_event_types"] = [t.strip().upper() for t in types_text.split(",") if t.strip()]
                    current["risk.avoid_event_types"] = list(settings.risk.avoid_event_types)
                note = ui.note_input(f"risk_note_{group}")
                if st.form_submit_button("Save"):
                    error = None
                    for f in fields:
                        if f.kind == "time":
                            try:
                                svc.validate_hhmm(proposed[f.path])
                            except svc.DashboardError as exc:
                                error = str(exc)
                    ui.save_settings(ctx, proposed, current, note, extra_validation=error)

    st.markdown("#### Leverage (INTRADAY / MIS only)")
    lev = settings.risk.mis_leverage
    if lev > 1:
        st.error(f"LEVERAGE IS ON: {lev}x for INTRADAY. Losses are multiplied. CNC modes never use leverage.")
    with st.form("risk_leverage_form"):
        new_lev = ui.number_setting(ctx, "MIS leverage (1 = off)", "risk.mis_leverage", lev, key="risk_lev",
                                    min_value=1.0, step=0.5)
        if new_lev > 1:
            st.warning("You are enabling leverage for INTRADAY. This amplifies losses and the loss budget is "
                       "consumed faster.")
        ack = st.checkbox("I understand the risk of leverage", key="risk_lev_ack")
        note = ui.note_input("risk_lev_note")
        if st.form_submit_button("Save leverage"):
            if new_lev > 1 and not ack:
                st.error("Tick the acknowledgement to enable leverage.")
            else:
                ui.save_settings(ctx, {"risk.mis_leverage": float(new_lev)}, {"risk.mis_leverage": lev}, note)
