"""Control 7 - Strategies: enable/disable per mode, parameters, live performance, re-enable auto-disabled."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import streamlit as st

from config.loader import settings_to_dict
from core.enums import CommandType, Mode
from dashboard import components as ui
from dashboard import services as svc

TITLE = "Strategies"
ICON = ":material/strategy:"

_NAME_RE = re.compile(r"^[a-z0-9_]{2,64}$")


def _param_input(strategy: str, name: str, value: Any) -> Any:
    key = f"sp_{strategy}_{name}"
    if isinstance(value, bool):
        return st.checkbox(name, value=value, key=key)
    if isinstance(value, int):
        return int(st.number_input(name, value=value, step=1, key=key))
    if isinstance(value, float):
        return float(st.number_input(name, value=value, step=0.1, format="%.4f", key=key))
    text = st.text_input(f"{name} (JSON)", value=json.dumps(value), key=key)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _vid, settings = loaded
    all_cfg: dict[str, Any] = settings_to_dict(settings)["strategies"]["strategies"]
    st.caption("Parameter ranges are enforced by the engine (values outside a strategy's range are clamped). "
               "Strategies without an entry here run with their coded defaults.")

    with ctx.session() as session:
        states = svc.strategy_states(session, ctx.batch_id)
        perf = svc.strategy_performance(session, ctx.batch_id)

    st.markdown("#### Live performance (selected batch)")
    ui.table(perf)

    st.markdown("#### Strategy states")
    ui.table([{"strategy": s.strategy, "mode": s.mode, "auto-disabled": s.auto_disabled, "trades": s.trades_count,
               "expectancy R": s.expectancy_r, "kelly": s.kelly_fraction, "reason": s.disabled_reason or ""}
              for s in states])
    for s in [s for s in states if s.auto_disabled]:
        if st.button(f"Re-enable {s.strategy} for {s.mode}", key=f"reen_{s.strategy}_{s.mode}"):
            ui.submit_command(ctx, CommandType.REENABLE_STRATEGY, {"strategy": s.strategy, "mode": s.mode},
                              track_key="strat_last_cmd")
    ui.tracked_command(ctx, "strat_last_cmd")

    st.markdown("#### Configuration")
    names = sorted(set(all_cfg) | {s.strategy for s in states})
    for name in names:
        cfg = all_cfg.get(name) or {"enabled": True, "modes": [], "params": {}}
        with st.expander(f"{name} - {'enabled' if cfg.get('enabled', True) else 'DISABLED'}"):
            with st.form(f"strat_form_{name}"):
                enabled = st.toggle("Enabled", value=bool(cfg.get("enabled", True)), key=f"se_{name}")
                modes = st.multiselect("Modes (empty = strategy's coded modes)", [m.value for m in Mode],
                                       default=[m for m in cfg.get("modes", []) if m in {x.value for x in Mode}],
                                       key=f"sm_{name}")
                params = {p: _param_input(name, p, v) for p, v in (cfg.get("params") or {}).items()}
                new_param = st.text_input("Add parameter (name=value)", key=f"snp_{name}",
                                          placeholder="e.g. atr_mult=1.5")
                note = ui.note_input(f"strat_note_{name}")
                if st.form_submit_button(f"Save {name}"):
                    if new_param.strip():
                        pname, _, pval = new_param.partition("=")
                        pname = pname.strip()
                        if not _NAME_RE.match(pname) or not pval.strip():
                            st.error("Use name=value with a snake_case name.")
                            st.stop()
                        try:
                            params[pname] = json.loads(pval.strip())
                        except json.JSONDecodeError:
                            params[pname] = pval.strip()
                    updated = copy.deepcopy(all_cfg)
                    updated[name] = {"enabled": enabled, "modes": modes, "params": params}
                    ui.save_settings(ctx, {"strategies.strategies": updated}, {"strategies.strategies": all_cfg}, note)

    with st.form("strat_add_form"):
        st.markdown("#### Add a strategy override")
        new_name = st.text_input("Strategy name (as registered by the engine, snake_case)", key="strat_new")
        note = ui.note_input("strat_add_note")
        if st.form_submit_button("Add"):
            new_name = new_name.strip()
            if not _NAME_RE.match(new_name):
                st.error("Invalid strategy name.")
            elif new_name in all_cfg:
                st.info("Already configured.")
            else:
                updated = copy.deepcopy(all_cfg)
                updated[new_name] = {"enabled": True, "modes": [], "params": {}}
                ui.save_settings(ctx, {"strategies.strategies": updated}, {"strategies.strategies": all_cfg}, note)
