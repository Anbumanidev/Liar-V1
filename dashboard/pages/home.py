"""Home - one-screen cockpit: status, who decides (auto vs human), what needs you, setup checklist."""

from __future__ import annotations

import streamlit as st

from core.enums import CommandType
from dashboard import components as ui
from dashboard import services as svc
from dashboard.cockpit import cockpit, setup_checklist
from storage.audit import record_error

TITLE = "Home"
ICON = ":material/home:"


def _money(v: float | None) -> str:
    return "-" if v is None else f"Rs {v:,.2f}"


def render() -> None:
    ctx = ui.page_header("Home", "")
    loaded = ui.active_settings(ctx)
    if loaded is None:
        return
    _, settings = loaded
    now = ctx.clock.now()
    view = ui.engine_view(ctx, settings)
    if st.toggle("Auto-refresh every 15 s", key="home_autorefresh", value=False):
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=15_000, key="home_refresh")
        except ImportError:
            st.caption("streamlit-autorefresh not installed")
    try:
        with ctx.session() as session:
            c = cockpit(session, ctx.batch_id, now)
            approvals = svc.pending_approvals(session, ctx.batch_id, now)
            positions = svc.open_positions(session, ctx.batch_id)
            pos_rows = svc.position_rows(positions)
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.home", exc)
        st.error(f"Could not load the cockpit: {exc}")
        return

    # ------------------------------------------------------------------ alerts
    if c.halted:
        st.error(f"HALTED - capital protection stopped trading. {c.halt_reason or ''} "
                 "Resume only from System > Engine after reviewing.", icon=":material/gpp_bad:")
    if not view.alive:
        st.warning("The engine is not running (no fresh heartbeat). Start it:  .\\tasks.ps1 run",
                   icon=":material/power_off:")
    if view.data_feed_ok is False:
        st.warning("Market data feed reported DOWN by the engine.", icon=":material/signal_disconnected:")
    if approvals:
        st.info(f"{len(approvals)} trade(s) waiting for your decision - see below.", icon=":material/notifications:")

    # ------------------------------------------------------------------ numbers
    st.markdown(f"**{c.batch_name or 'No batch'}** · {c.batch_status or '-'} · engine **{view.label}**"
                f" · decides: **{ui.AUTONOMY_HELP[settings.modes.autonomy.value][0]}**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total equity", _money(c.equity),
              f"{c.total_return_pct:+.2f}% since start" if c.total_return_pct is not None else None)
    m2.metric("Today's net P&L", _money(c.today_net), f"{c.today_trades} closed trade(s)")
    m3.metric("Drawdown", f"{c.drawdown_pct:.2f}%" if c.drawdown_pct is not None else "-", c.tier or None,
              delta_color="off")
    m4, m5, m6 = st.columns(3)
    m4.metric("TRADING", _money(c.trading))
    m5.metric("RESERVE (locked profit)", _money(c.reserve))
    m6.metric("Open positions", str(c.open_positions), _money(c.unrealized) + " unrealized", delta_color="off")

    # ------------------------------------------------------------------ engine quick controls
    with st.container(border=True):
        st.markdown("**Engine**")
        b1, b2, b3 = st.columns(3)
        if b1.button("Start / Resume", key="home_start", icon=":material/play_arrow:", width="stretch"):
            ui.submit_command(ctx, CommandType.ENGINE_RESUME, {}, track_key="home_cmd")
        if b2.button("Pause new entries", key="home_pause", icon=":material/pause:", width="stretch"):
            ui.submit_command(ctx, CommandType.ENGINE_PAUSE, {}, track_key="home_cmd")
        b3.caption("Emergency kill switch: System > Engine")
        ui.tracked_command(ctx, "home_cmd")

    # ------------------------------------------------------------------ autonomy
    with st.container(border=True):
        st.markdown("**Who decides: fully automatic or you (human in the loop)?**")
        ui.autonomy_control(ctx, settings, key="home_autonomy")

    # ------------------------------------------------------------------ needs you
    with st.container(border=True):
        st.markdown(f"**Waiting for you** ({len(approvals)})")
        if not approvals:
            st.caption("Nothing to approve right now.")
        for v in approvals[:5]:
            it = v.item
            tax = ((it.summary_json or {}).get("tax") or {})
            t1 = next((s for s in tax.get("scenarios", []) if s.get("label") == "Target 1"), None)
            extra = f" · after-tax at T1 {_money(t1['net_after_tax'])}" if t1 else ""
            st.markdown(f"**{it.symbol}** {it.mode} {it.direction} {it.qty} @ {it.entry_price:.2f} · SL "
                        f"{it.stop_loss:.2f} · T1 {it.target_1:.2f} · {it.grade or '-'} {it.confidence}%{extra} · "
                        f"{'EXPIRED' if v.expired else 'expires in ' + v.countdown}")
            c1, c2 = st.columns(2)
            if c1.button("Approve", key=f"home_ap_{it.id}", type="primary", width="stretch", icon=":material/check:"):
                ui.submit_command(ctx, CommandType.APPROVE, {"approval_id": it.id}, track_key="home_cmd")
            if c2.button("Reject", key=f"home_rj_{it.id}", width="stretch", icon=":material/close:"):
                ui.submit_command(ctx, CommandType.REJECT, {"approval_id": it.id, "reason": "rejected from Home"},
                                  track_key="home_cmd")
        if len(approvals) > 5:
            st.caption(f"... {len(approvals) - 5} more in Trading > Approvals (full reasoning, modify, bulk reject).")

    # ------------------------------------------------------------------ positions
    with st.container(border=True):
        st.markdown(f"**Open positions** ({len(pos_rows)})")
        if pos_rows:
            ui.table(pos_rows, key="home_positions")
        else:
            st.caption("No open positions.")

    # ------------------------------------------------------------------ setup checklist
    models: list[str] | None = None
    if settings.engine.llm_backend != "rules":
        host = svc.ollama_host(settings, ctx.secrets.ollama_host)
        models = svc.installed_ollama_models(host) if svc.ollama_reachable(host) else None
    items = setup_checklist(settings, ctx.caps, ctx.secrets, engine_alive=view.alive,
                            engine_data_ok=view.data_feed_ok, ollama_models=models, today=now.date())
    failing = [i for i in items if i.ok is False]
    with st.expander(f"Setup checklist ({len(items) - len(failing)}/{len(items)} OK)", expanded=bool(failing)):
        for i in items:
            mark = ":green[OK]" if i.ok else (":red[FIX]" if i.ok is False else ":blue[INFO]")
            st.markdown(f"{mark} **{i.name}** - {i.detail}" + (f"  \n&nbsp;&nbsp;&nbsp;&nbsp;*{i.fix}*"
                                                                if i.fix and i.ok is not True else ""))

    with st.expander("How this works (plain language)"):
        st.markdown(
            "1. Every few minutes the engine scans stocks, computes indicators and runs 17 rule-based strategies.\n"
            "2. Promising setups go to 10 AI specialist agents, a Bull vs Bear debate and a final Meta agent.\n"
            "3. A strict **risk manager** (plain Python, not AI) sizes the trade and can veto anything - "
            "including your own trades.\n"
            "4. Depending on **Who decides** above, the trade executes automatically or waits for you.\n"
            "5. The paper engine manages stops, targets, partial profits and square-off; profits are partly "
            "locked into RESERVE every evening.\n"
            "6. Capital protection halts everything if equity falls 9% from its high or below Rs 910 on Rs 1,000.\n\n"
            "**Everything is paper trading - no real orders are ever placed.**"
        )
