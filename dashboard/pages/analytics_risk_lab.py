"""Risk lab - advanced methods: Monte Carlo risk-of-ruin, agent calibration with suggested
weights, and the latest 'stocks in play' ranking."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from analytics.calibration import agent_calibration
from analytics.montecarlo import monte_carlo, outcomes_from_r
from core.enums import Mode
from core.logging import read_events
from dashboard import components as ui
from storage.audit import record_error
from storage.models import CapitalAccount, Trade

TITLE = "Risk lab"
ICON = ":material/science:"


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    loaded = ui.active_settings(ctx)
    if loaded is None or ctx.batch_id is None:
        st.info("Select a batch.")
        return
    _, settings = loaded
    tab_mc, tab_cal, tab_rank = st.tabs(["Monte Carlo risk", "Agent calibration", "Stocks in play"])

    with ctx.session() as session:
        trades = list(session.execute(select(Trade).where(Trade.batch_id == ctx.batch_id)).scalars())
        acct = session.get(CapitalAccount, ctx.batch_id)

    with tab_mc:
        st.markdown("Replays your **own trade results** thousands of times in random order to show the range of "
                    "outcomes and the chance of hitting the capital-protection HALT.")
        per_pos: dict[int, float] = defaultdict(float)
        rs: dict[int, list[float]] = defaultdict(list)
        for t in trades:
            per_pos[t.position_id] += t.net_pnl
            if t.r_multiple is not None:
                rs[t.position_id].append(t.r_multiple)
        source = st.radio("Outcomes", ["Net P&L per trade (Rs)", "R-multiples x today's risk"], horizontal=True,
                          key="mc_src")
        c1, c2 = st.columns(2)
        n_trades = c1.slider("Future trades", 20, 300, 100, 10, key="mc_n")
        n_paths = c2.select_slider("Simulations", [500, 1000, 2000, 5000], 2000, key="mc_p")
        if acct is None:
            st.info("No capital account for this batch.")
        elif not per_pos:
            st.info("No closed trades yet - run the engine or a replay first.")
        else:
            equity = acct.trading_cash + acct.blocked + acct.reserve
            if source.startswith("Net"):
                outcomes = list(per_pos.values())
            else:
                risk = settings.risk.risk_per_trade_pct / 100.0 * acct.sizing_capital
                outcomes = outcomes_from_r([sum(v) / len(v) for v in rs.values() if v], risk)
            res = monte_carlo(outcomes, equity, n_trades=n_trades, n_paths=int(n_paths),
                              hard_stop_pct=settings.risk.hard_stop_pct, starting_capital=acct.starting_capital)
            if res is None:
                st.info("Not enough data.")
            else:
                if not res.reliable:
                    st.warning(f"Only {res.n_samples} trades in the sample - treat the result as a rough sketch "
                               "(30+ recommended).")
                m1, m2, m3 = st.columns(3)
                m1.metric("Chance of profit", f"{res.prob_profit * 100:.1f}%")
                m2.metric("Chance of HALT", f"{res.prob_halt * 100:.1f}%")
                m3.metric("Median final equity", f"Rs {res.median_final:,.2f}")
                m4, m5, m6 = st.columns(3)
                m4.metric("Bad case (5%)", f"Rs {res.p5_final:,.2f}")
                m5.metric("Good case (95%)", f"Rs {res.p95_final:,.2f}")
                m6.metric("Max drawdown (95%)", f"{res.p95_max_dd_pct:.2f}%")
                fig = go.Figure()
                x = list(range(len(res.bands["p50"])))
                fig.add_scatter(x=x, y=res.bands["p95"], name="95th pct", line={"width": 1})
                fig.add_scatter(x=x, y=res.bands["p50"], name="median", line={"width": 3})
                fig.add_scatter(x=x, y=res.bands["p5"], name="5th pct", line={"width": 1})
                floor = acct.starting_capital * (1 - settings.risk.hard_stop_pct / 100.0)
                fig.add_hline(y=floor, line_dash="dot", annotation_text="absolute floor")
                fig.update_layout(height=360, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                                  xaxis_title="future trades", yaxis_title="equity (Rs)", legend={"orientation": "h"})
                st.plotly_chart(fig, width="stretch")

    with tab_cal:
        st.markdown("How often each AI agent's vote matched what the price did next (traded and shadow setups). "
                    "**Suggested weights are not applied automatically** - validate them with a walk-forward replay.")
        try:
            weights = {m.value: settings.mode(m).agent_weights for m in Mode}
            with ctx.session() as session:
                df = agent_calibration(session, ctx.batch_id, weights)
        except Exception as exc:  # noqa: BLE001
            record_error(None, "dashboard.risk_lab", exc)
            st.error(f"Calibration failed: {exc}")
            df = None
        if df is None or df.empty:
            st.info("No resolved setups with agent votes yet.")
        else:
            ui.table(df, key="cal_table")
            modes = sorted(df["mode"].unique())
            mode = st.selectbox("Apply suggested weights for mode", modes, key="cal_mode")
            sub = df[(df["mode"] == mode) & (df["suggested_weight"] != df["weight"])]
            if sub.empty:
                st.caption("No changes suggested for this mode (not enough samples or already matching).")
            else:
                cur = {f"modes.modes.{mode}.agent_weights": dict(settings.mode(Mode(mode)).agent_weights)}
                new_w = dict(cur[f"modes.modes.{mode}.agent_weights"])
                for _, r in sub.iterrows():
                    new_w[r["agent"]] = float(r["suggested_weight"])
                st.write({r["agent"]: f"{r['weight']} -> {r['suggested_weight']}" for _, r in sub.iterrows()})
                if st.button("Apply suggested weights", key="cal_apply"):
                    ui.save_settings(ctx, {f"modes.modes.{mode}.agent_weights": new_w}, cur,
                                     "agent weights from calibration")

    with tab_rank:
        st.markdown("The intraday cycle analyses the **stocks most in play** first (relative volume, gap/move, "
                    "range vs ATR, strength vs NIFTY). Latest ranking from today's scan log:")
        events = [e for e in read_events(ctx.clock.now().date(), "scans", ctx.log_dir)
                  if e.get("event") == "in_play_ranking"]
        if not events:
            st.caption("No ranking yet today (it runs during the 09:30-14:45 intraday cycle).")
        else:
            last = events[-1]
            st.caption(f"Ranked at {last.get('ts', '')[:19]} for {last.get('mode')}")
            ui.table(last.get("top") or [], key="rank_table")
