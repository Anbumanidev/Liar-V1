"""Server-Sent Events: one stream per browser tab, polling the shared DB every ~2 s.

Each iteration opens a SHORT DB session (never held between polls), emits typed events for
rows newer than the client's cursors, and closes it:

* ``decision``  - new analysis rows (same shape as ``/api/decisions``)
* ``order``     - new orders and status changes of pending ones
* ``trade``     - new closed legs (exits)
* ``approvals`` - pending approvals count changed
* ``status``    - engine heartbeat/status + capital (only when something changed)
* ``log``       - new lines appended to today's ``engine.log`` (file offset tracking)
* ``activity``  - new events appended to today's ``activity.jsonl`` (what the engine / AI is doing)

A comment heartbeat (``: ping``) is sent every 15 s. The stream ends with
``session_expired`` when the cookie it was opened with passes the idle deadline (the browser
reconnects with its fresh cookie, or gets 401 and shows the login screen).
"""

from __future__ import annotations

import asyncio
import json
import time as _time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from storage.models import ApprovalItem, Order, Trade
from webui import queries as q
from webui.context import WebContext
from webui.logs import LogFollower
from webui.util import iso


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, separators=(',', ':'))}\n\n"


@dataclass
class StreamState:
    decision: int = 0
    order: int = 0
    trade: int = 0
    pending_orders: dict[int, str] = field(default_factory=dict)
    approvals_sig: tuple[Any, ...] | None = None
    status_sig: tuple[Any, ...] | None = None
    follower: LogFollower | None = None
    activity: LogFollower | None = None


def init_state(ctx: WebContext, cursors: dict[str, int | None]) -> StreamState:
    st = StreamState()
    with ctx.session() as s:
        latest = q.latest_ids(s)
        st.decision = latest["decision"] if cursors.get("decision") is None else int(cursors["decision"] or 0)
        st.order = latest["order"] if cursors.get("order") is None else int(cursors["order"] or 0)
        st.trade = latest["trade"] if cursors.get("trade") is None else int(cursors["trade"] or 0)
        for oid, status in s.execute(select(Order.id, Order.status).where(Order.status == "PENDING")).all():
            st.pending_orders[oid] = status
    today = ctx.clock.now().date()
    st.follower = LogFollower(ctx.log_dir, "engine")
    st.follower.start(today)
    st.activity = LogFollower(ctx.log_dir, "activity")
    st.activity.start(today)
    return st


def _trade_event(t: Trade) -> dict[str, Any]:
    row = q.leg_row(t)
    row.update({"symbol": t.symbol, "mode": t.mode, "direction": t.direction, "batch_id": t.batch_id,
                "exit_label": q.exit_label(t.exit_reason, direction=t.direction, exit_price=t.exit_price,
                                           target_2=t.target_2, partial=bool(t.is_partial))})
    return row


def poll_once(ctx: WebContext, st: StreamState) -> list[tuple[str, Any]]:
    """One DB poll -> list of (event, data). Opens and closes its own short session."""
    out: list[tuple[str, Any]] = []
    now = ctx.clock.now()
    try:
        stale = ctx.settings().engine.heartbeat_stale_seconds
    except Exception:  # noqa: BLE001
        stale = 30
    with ctx.session() as s:
        rows = q.decision_feed(s, since_id=st.decision, limit=50)
        for r in reversed(rows):
            out.append(("decision", r))
            st.decision = max(st.decision, r["id"])

        new_orders = list(s.execute(select(Order).where(Order.id > st.order).order_by(Order.id).limit(100)).scalars())
        for o in new_orders:
            out.append(("order", q.order_row(o)))
            st.order = max(st.order, o.id)
            if o.status == "PENDING":
                st.pending_orders[o.id] = o.status
        watched = [oid for oid in st.pending_orders if oid not in {o.id for o in new_orders}]
        if watched:
            for o in s.execute(select(Order).where(Order.id.in_(watched))).scalars():
                if o.status != st.pending_orders.get(o.id):
                    out.append(("order", q.order_row(o)))
                if o.status != "PENDING":
                    st.pending_orders.pop(o.id, None)
                else:
                    st.pending_orders[o.id] = o.status

        for t in s.execute(select(Trade).where(Trade.id > st.trade).order_by(Trade.id).limit(100)).scalars():
            out.append(("trade", _trade_event(t)))
            st.trade = max(st.trade, t.id)

        pending = list(s.execute(select(ApprovalItem.id).where(ApprovalItem.status == "PENDING")).scalars())
        sig_a = (len(pending), max(pending) if pending else 0)
        if sig_a != st.approvals_sig:
            out.append(("approvals", {"pending": len(pending), "latest_id": sig_a[1],
                                      "changed": st.approvals_sig is not None}))
            st.approvals_sig = sig_a

        engine = q.engine_view(s, now, stale)
        batch = q.current_batch(s)
        capital = q.capital_view(s, batch.id, 0.0) if batch is not None else None
        sig_s = (engine["state"], engine["heartbeat_at"], engine["activity"], engine["market_open"],
                 engine["data_feed_ok"], engine["llm_ok"], engine["stale"], engine["next_cycle_at"],
                 capital and capital["updated_at"], capital and capital["realized_equity"],
                 capital and capital["tier"], capital and capital["halted"], batch and batch.id, batch and batch.status)
        if sig_s != st.status_sig:
            out.append(("status", {"engine": engine, "capital": capital, "batch": q.batch_brief(batch),
                                   "server_time": iso(now)}))
            st.status_sig = sig_s

    if st.follower is not None:
        try:
            lines = st.follower.poll(now.date())
        except Exception as exc:  # noqa: BLE001 - log tailing is best effort
            logger.debug(f"[webui] log follow failed: {exc}")
            lines = []
        if lines:
            out.append(("log", {"file": "engine", "entries": lines[-200:]}))
    if st.activity is not None:
        try:
            acts = st.activity.poll(now.date())
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[webui] activity follow failed: {exc}")
            acts = []
        if acts:
            out.append(("activity", {"entries": acts[-300:]}))
    return out


async def event_stream(request: Request, ctx: WebContext, session_deadline: float | None,
                       cursors: dict[str, int | None]):  # noqa: ANN201 - async generator
    st = await run_in_threadpool(init_state, ctx, cursors)
    yield "retry: 3000\n\n"
    yield sse("hello", {"server_time": iso(ctx.clock.now()), "cursors": {"decision": st.decision,
                                                                       "order": st.order, "trade": st.trade},
                        "interval_s": ctx.stream_interval})
    iterations = 0
    last_beat = _time.monotonic()
    while True:
        try:
            events = await run_in_threadpool(poll_once, ctx, st)
        except Exception as exc:  # noqa: BLE001 - a failed poll must not kill the stream
            logger.warning(f"[webui] stream poll failed: {exc}")
            events = []
        for name, data in events:
            yield sse(name, data)
        iterations += 1
        if ctx.stream_max_iterations is not None and iterations >= ctx.stream_max_iterations:
            break
        if session_deadline is not None and ctx.clock.now().timestamp() >= session_deadline:
            yield sse("session_expired", {"detail": "session idle deadline reached; reconnect"})
            break
        if _time.monotonic() - last_beat >= ctx.stream_heartbeat:
            yield ": ping\n\n"
            last_beat = _time.monotonic()
        await asyncio.sleep(ctx.stream_interval)
        if await request.is_disconnected():
            break
