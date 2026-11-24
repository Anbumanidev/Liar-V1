"""Control 6 - Watchlist & universe: per-mode universe, custom symbols, whitelist/blacklist, filters."""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.enums import Mode
from dashboard import components as ui
from dashboard import services as svc

TITLE = "Universe"
ICON = ":material/travel_explore:"

UNIVERSE_SOURCES = ("NIFTY50", "NIFTY100", "NIFTY200", "FNO", "CUSTOM")


def _symbols_editor(label: str, current: list[str], key: str) -> tuple[list[str], list[str]]:
    text = st.text_area(f"{label} ({len(current)})", value=", ".join(current), key=key, height=80,
                        help="Comma / space / newline separated NSE tickers (e.g. RELIANCE, TCS).")
    return svc.parse_symbols(text)


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _vid, settings = loaded
    uni = settings.universe

    with st.form("universe_modes_form"):
        st.markdown("#### Universe per mode")
        proposed: dict[str, Any] = {}
        for mode in Mode:
            cur = settings.mode(mode).universe
            opts = list(UNIVERSE_SOURCES) + ([cur] if cur not in UNIVERSE_SOURCES else [])
            proposed[f"modes.modes.{mode.value}.universe"] = st.selectbox(
                mode.value, opts, index=opts.index(cur), key=f"uni_{mode.value}")
        note = ui.note_input("uni_modes_note")
        if st.form_submit_button("Save universes"):
            ui.save_settings(ctx, proposed, ui.flat_values(settings, proposed.keys()), note)

    with st.form("universe_lists_form"):
        st.markdown("#### Symbols")
        custom, bad1 = _symbols_editor("Custom symbols (added to every universe / CUSTOM)", uni.custom_symbols, "u_custom")
        whitelist_only = st.toggle("Whitelist-only mode (trade ONLY whitelisted symbols)", value=uni.whitelist_only,
                                   key="u_wlo")
        whitelist, bad2 = _symbols_editor("Whitelist", uni.whitelist, "u_wl")
        blacklist, bad3 = _symbols_editor("Blacklist (never traded)", uni.blacklist, "u_bl")
        note = ui.note_input("uni_lists_note")
        if st.form_submit_button("Save symbol lists"):
            bad = bad1 + bad2 + bad3
            if bad:
                st.error(f"Invalid symbols: {', '.join(bad)}")
            elif whitelist_only and not whitelist:
                st.error("Whitelist-only mode needs at least one whitelisted symbol.")
            else:
                proposed = {"universe.custom_symbols": custom, "universe.whitelist": whitelist,
                            "universe.blacklist": blacklist, "universe.whitelist_only": whitelist_only}
                ui.save_settings(ctx, proposed, ui.flat_values(settings, proposed.keys()), note)

    with st.form("universe_filters_form"):
        st.markdown("#### Filters")
        min_price = st.number_input("Min price (Rs)", min_value=0.0, value=float(uni.min_price), step=1.0, key="u_minp")
        max_price = st.number_input("Max price (Rs, 0 = no limit)", min_value=0.0, value=float(uni.max_price or 0.0),
                                    step=10.0, key="u_maxp")
        turnover = st.number_input("Min 20-day avg daily turnover (Rs crore)", min_value=0.0,
                                   value=float(uni.min_avg_daily_turnover_cr), step=0.5, key="u_turn")
        spread = st.number_input("Max bid-ask spread %", min_value=0.0, value=float(uni.max_spread_pct), step=0.05,
                                 key="u_spread")
        sectors_text = st.text_input("Excluded sectors (comma separated)", value=", ".join(uni.excluded_sectors),
                                     key="u_sectors")
        note = ui.note_input("uni_filters_note")
        if st.form_submit_button("Save filters"):
            if max_price and max_price < min_price:
                st.error("Max price must be above min price.")
            else:
                sectors = sorted({s.strip() for s in sectors_text.split(",") if s.strip()})
                proposed = {"universe.min_price": float(min_price),
                            "universe.max_price": float(max_price) if max_price > 0 else None,
                            "universe.min_avg_daily_turnover_cr": float(turnover),
                            "universe.max_spread_pct": float(spread), "universe.excluded_sectors": sectors}
                ui.save_settings(ctx, proposed, ui.flat_values(settings, proposed.keys()), note)
