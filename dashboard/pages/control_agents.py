"""Control 8 - Agents & models: enable, model per agent, profiles, weights, thresholds, debate, prompts."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.loader import settings_to_dict
from config.schema import AGENT_NAMES, ALL_LLM_AGENTS
from core.enums import Mode
from dashboard import components as ui
from dashboard import services as svc
from storage.db import session_scope

TITLE = "Agents & models"
ICON = ":material/smart_toy:"

_DEFAULT = "(profile default)"


@st.cache_data(ttl=30, show_spinner=False)
def _models(host: str) -> list[str]:
    return svc.installed_ollama_models(host)


def _prompt_editor(ctx: Any, settings: Any) -> None:
    st.markdown("#### Prompt templates (versioned)")
    agent = st.selectbox("Agent", list(ALL_LLM_AGENTS), key="pr_agent")
    pinned = settings.agents.prompt_versions.get(agent)
    if pinned:
        st.warning(f"{agent} is pinned to prompt version {pinned} in settings (agents.prompt_versions); "
                   "new versions are not used until you unpin it.")
    with ctx.session() as session:
        versions = svc.prompt_versions(session, agent)
    if not versions:
        st.info("No prompt stored yet for this agent (the engine seeds version 1 from agents/prompts on start). "
                "You can create one below.")
    latest = versions[0] if versions else None
    with st.form(f"prompt_form_{agent}"):
        st.caption(f"Editing on top of v{latest.version}" if latest else "New prompt")
        system = st.text_area("System prompt", value=latest.system_prompt if latest else "", height=200,
                              key=f"pr_sys_{agent}")
        user_t = st.text_area("User template (Jinja2; facts are injected by code)",
                              value=latest.user_template if latest else "", height=200, key=f"pr_user_{agent}")
        note = st.text_input("Version note", key=f"pr_note_{agent}")
        if st.form_submit_button("Save as new version", type="primary"):
            if latest and system == latest.system_prompt and user_t == latest.user_template:
                st.info("No changes.")
            else:
                try:
                    with session_scope(ctx.session_factory) as session:
                        row = svc.save_prompt_version(session, agent, system, user_t, author=ctx.actor,
                                                      note=note or None, clock=ctx.clock)
                        version = row.version
                except svc.DashboardError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Saved {agent} prompt v{version}.")
    if versions:
        ui.table([{"version": v.version, "active": v.active, "created": ui.fmt_dt(v.created_at), "author": v.author,
                   "note": v.note or ""} for v in versions])
        older = [v.version for v in versions[1:]]
        if older:
            target = st.selectbox("Revert to version", older, key=f"pr_rev_{agent}")
            with st.expander(f"Preview v{target}"):
                chosen = next(v for v in versions if v.version == target)
                st.code(chosen.system_prompt, language=None)
                st.code(chosen.user_template, language=None)
            if st.button(f"Revert {agent} to v{target}", key=f"pr_rev_btn_{agent}"):
                try:
                    with session_scope(ctx.session_factory) as session:
                        row = svc.revert_prompt(session, agent, int(target), author=ctx.actor, clock=ctx.clock)
                        version = row.version
                except svc.DashboardError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Reverted: v{target} copied to new version v{version}.")


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _vid, settings = loaded
    host = svc.ollama_host(settings, ctx.secrets.ollama_host)
    installed = _models(host)
    if installed:
        st.caption(f"Ollama at {host}: {len(installed)} model(s) installed.")
    else:
        st.warning(f"Ollama not reachable at {host} (or no models). Type model names manually.")

    with st.form("agents_enable_form"):
        st.markdown("#### Agents & models")
        models_cfg = settings.models
        proposed: dict[str, Any] = {}
        profiles = list(models_cfg.profiles)
        proposed["models.active_profile"] = st.selectbox(
            "Model profile", profiles, index=profiles.index(models_cfg.active_profile), key="ag_profile")
        enabled: dict[str, bool] = {}
        assignments: dict[str, str] = {}
        for agent in ALL_LLM_AGENTS:
            st.markdown(f"**{agent}**  \n:gray[effective model: {models_cfg.model_for(agent)}]")
            enabled[agent] = st.toggle("Enabled", value=settings.agents.enabled.get(agent, True), key=f"ag_en_{agent}")
            current = models_cfg.assignments.get(agent, "")
            if installed:
                opts = [_DEFAULT] + installed + ([current] if current and current not in installed else [])
                choice = st.selectbox("Model", opts, index=opts.index(current) if current else 0, key=f"ag_m_{agent}")
                if choice != _DEFAULT:
                    assignments[agent] = choice
            else:
                typed = st.text_input("Model (empty = profile default)", value=current, key=f"ag_mt_{agent}").strip()
                if typed:
                    assignments[agent] = typed
        proposed["agents.enabled"] = enabled
        proposed["models.assignments"] = assignments
        proposed["agents.debate_enabled"] = st.toggle("Bull/Bear debate enabled", value=settings.agents.debate_enabled,
                                                      key="ag_deb_en")
        proposed["agents.debate_rounds"] = st.number_input("Debate rebuttal rounds", 0, 5,
                                                           int(settings.agents.debate_rounds), key="ag_deb_r")
        note = ui.note_input("ag_note")
        if st.form_submit_button("Save agents & models", type="primary"):
            ui.save_settings(ctx, proposed, ui.flat_values(settings, proposed.keys()), note)

    with st.expander("Model profiles"):
        with st.form("profiles_form"):
            profiles_data = settings_to_dict(settings)["models"]["profiles"]
            new_profiles: dict[str, Any] = {}
            for name, prof in profiles_data.items():
                st.markdown(f"**{name}**")
                new_profiles[name] = {
                    "reasoning": st.text_input("Reasoning model (bull/bear/meta)", prof["reasoning"], key=f"pf_r_{name}"),
                    "sub_agent": st.text_input("Sub-agent model", prof["sub_agent"], key=f"pf_s_{name}"),
                }
            note = ui.note_input("pf_note")
            if st.form_submit_button("Save profiles"):
                ui.save_settings(ctx, {"models.profiles": new_profiles}, {"models.profiles": profiles_data}, note)

    st.markdown("#### Per-mode weights & thresholds")
    for mode in Mode:
        cfg = settings.mode(mode)
        with st.expander(f"{mode.value}"):
            with st.form(f"weights_{mode.value}"):
                base = f"modes.modes.{mode.value}"
                p: dict[str, Any] = {}
                p[f"{base}.confidence_threshold"] = st.slider("Meta confidence threshold", 0, 100,
                                                              int(cfg.confidence_threshold), key=f"w_ct_{mode.value}")
                weights: dict[str, float] = {}
                agents_on: dict[str, bool] = {}
                for agent in AGENT_NAMES:
                    weights[agent] = float(st.number_input(f"{agent} weight", min_value=0.0, max_value=5.0,
                                                           value=float(cfg.agent_weights.get(agent, 1.0)), step=0.1,
                                                           key=f"w_{mode.value}_{agent}"))
                    agents_on[agent] = st.checkbox(f"{agent} runs in {mode.value}",
                                                   value=cfg.agents_enabled.get(agent, True),
                                                   key=f"wa_{mode.value}_{agent}")
                p[f"{base}.agent_weights"] = weights
                p[f"{base}.agents_enabled"] = {a: v for a, v in agents_on.items() if not v or a in cfg.agents_enabled}
                note = ui.note_input(f"w_note_{mode.value}")
                if st.form_submit_button(f"Save {mode.value} weights"):
                    ui.save_settings(ctx, p, ui.flat_values(settings, p.keys()), note)

    _prompt_editor(ctx, settings)
