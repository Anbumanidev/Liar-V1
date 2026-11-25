"""Analytics 2 - Live: engine heartbeat and next cycle, loss budget remaining, regime, open
positions with live / unrealized P&L, pending approvals with countdown. Auto-refreshes."""

from __future__ import annotations

from typing import Any

import streamlit as st
from loguru import logger

from dashboard import analytics_queries as aq
from dashboard import charts as ch
from dashboard import components as ui
from dashboard import services as svc
from dashboard.auth import AUTOREFRESH_KEY
from storage.audit import record_error

TITLE = "Live"
ICON = ":material/monitor_heart:"

_REFRESH_MS = 10_000


def _auto_refresh() -> None:
    """Periodic rerun. Uses the auth module's auto-refresh key, so these reruns never count as
    owner activity for the 30-minute idle timeout."""
    if st.toggle("Pause auto-refresh", key="live_pause_refresh", value=False):
        st.button("Refresh now", icon=":material/refresh:", key="live_refresh_now")
        return
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=_REFRESH_MS, key=AUTOREFRESH_KEY)
        st.caption(f"Auto-refreshing every {_REFRESH_MS // 1000} s.")
    except ImportError as exc:
        logger.warning(f"streamlit-autorefresh unavailable: {exc}")
        st.button("Refresh", icon=":material/refresh:", key="live_refresh_now")


def _flag(value: bool | None, ok: str, bad: str) -> str:
    if value is None:
        return f"{ok}: ?"
    return f":green[{ok}]" if value else f":red[{bad}]"


def _engine_section(view: svc.EngineStatusView, now: Any) -> None:
    st.markdown("#### Engine")
    ui.heartbeat_badge(view)
    if view.next_cycle_at is not None:
        left = (view.next_cycle_at - now).total_seconds()
        st.markdown(f"Next cycle: **{ui.fmt_dt(view.next_cycle_at, with_date=False)}** "
                    f"({'due now' if left <= 0 else 'in ' + svc.format_countdown(left)})")
    else:
        st.caption("Next cycle: not scheduled / unknown.")
    st.markdown(" | ".join((_flag(view.market_open, "market open", "market closed"),
                            _flag(view.data_feed_ok, "data feed OK", "data feed DOWN"),
                            _flag(view.llm_ok, "LLM OK", "LLM DOWN"))))
    if view.activity:
        st.caption(f"Activity: {view.activity[:200]}")
    if view.last_error:
        st.warning(f"Last engine error: {view.last_error[:300]}")


def _budget_section(snap: dict[str, Any] | None) -> None:
    st.markdown("#### Loss budget")
    if snap is None:
        st.info("No capital account for this batch yet.")
        return
    items = [
        ("Loss budget remaining", ch.fmt_rs(snap["loss_budget"]),
         "TOTAL EQUITY - open-position risk - stricter floor"),
        ("Max risk on next trade", ch.fmt_rs(snap["max_new_trade_risk"]), "loss budget / divisor (risk manager caps it)"),
        ("Open-position risk", ch.fmt_rs(snap["open_risk"]),
         "worst stop loss incl. gap buffer for overnight positions + accrued charges"),
        ("Stricter floor", ch.fmt_rs(snap["effective_floor"]),
         f"absolute {ch.fmt_rs(snap['abs_floor'])} / trailing {ch.fmt_rs(snap['trailing_floor'])}"),
        ("TOTAL EQUITY", ch.fmt_rs(snap["equity"]), f"unrealized {ch.fmt_rs(snap['unrealized'], signed=True)}"),
        ("Tier / drawdown", f"{snap['tier']} / {ch.fmt_pct(snap['drawdown_pct'])}", f"HWM {ch.fmt_rs(snap['hwm'])}"),
    ]
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for col, (label, value, help_) in zip(cols, items[i:i + 2], strict=False):
            col.metric(label, value, help=help_, border=True)
    pct = float(snap["budget_left_pct"] or 0.0)
    st.progress(min(1.0, max(0.0, pct / 100.0)),
                text=f"{pct:.0f}% of the cushion above the floor is not at risk in open positions")
    if snap["unpriced_positions"]:
        st.caption(f"{snap['unpriced_positions']} open position(s) have no live price yet - their unrealized P&L "
                   f"counts as 0.")


def _positions_section(positions: Any) -> None:
    st.markdown(f"#### Open positions ({len(positions)})")
    if positions.empty:
        st.caption("No open positions.")
        return
    total = float(positions["unrealized"].fillna(0).sum())
    st.markdown(f"Unrealized (net of entry charges): **{ch.fmt_rs(total, signed=True)}** | risk to stops "
                f"{ch.fmt_rs(float(positions['risk_to_sl'].sum()))}")
    view = positions[["symbol", "mode", "direction", "qty", "avg_entry", "last_price", "unrealized", "unrealized_pct",
                      "r_now", "stop_loss", "target_1", "risk_to_sl", "price_age_s", "holding_days", "opened_at"]]
    ui.table(view.rename(columns={"direction": "dir", "avg_entry": "entry", "last_price": "ltp", "unrealized": "uPnL",
                                  "unrealized_pct": "uPnL %", "r_now": "R now", "stop_loss": "SL",
                                  "target_1": "T1", "risk_to_sl": "risk", "price_age_s": "price age s",
                                  "holding_days": "days"}))


def _approvals_section(pending: list[svc.ApprovalView]) -> None:
    st.markdown(f"#### Pending approvals ({len(pending)})")
    if not pending:
        st.caption("No pending approvals.")
        return
    ui.table([{
        "id": v.item.id, "symbol": v.item.symbol, "mode": v.item.mode, "dir": v.item.direction, "qty": v.item.qty,
        "entry": v.item.entry_price, "SL": v.item.stop_loss, "T1": v.item.target_1, "conf": v.item.confidence,
        "grade": v.item.grade, "R:R": None if v.rr is None else round(v.rr, 2),
        "expires in": "EXPIRED" if v.expired else v.countdown,
    } for v in pending])
    st.caption("Approve, reject or modify on Control -> Approvals.")


def _body(ctx: Any, batch_id: int) -> None:
    loaded = ui.active_settings(ctx)
    settings = loaded[1] if loaded else None
    params = aq.RiskParams.from_settings(settings)
    now = ctx.clock.now()
    with ctx.session() as session:
        view = svc.engine_status(session, now, svc.heartbeat_stale_seconds(settings))
        snap = aq.capital_snapshot(session, batch_id, params)
        positions = aq.open_positions_frame(session, batch_id, now=now, params=params)
        pending = svc.pending_approvals(session, batch_id, now)
    _engine_section(view, now)
    _budget_section(snap)
    st.markdown("#### Market regime")
    regime = aq.regime_rows(view.regime)
    if regime.empty:
        st.caption("No regime published by the engine yet.")
    else:
        ui.table(regime)
    _positions_section(positions)
    _approvals_section(pending)
    st.caption(f"As of {ui.fmt_dt(now)} IST.")


def render() -> None:
    ctx = ui.page_header(TITLE, "")
    _auto_refresh()
    batch_id = ui.require_batch(ctx)
    if batch_id is None:
        return
    try:
        _body(ctx, batch_id)
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.analytics_live", exc, {"batch_id": batch_id})
        st.error(f"Could not load live data: {exc}")
