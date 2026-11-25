"""Owner funding across batches: money in, money out, overall P&L (paper money only).

Every new batch starts from a chosen capital (default ``capital.starting_capital`` = 1000).
The old batch's equity is carried over, and the difference is recorded as a funding event:

    top_up = new starting capital - carried equity      (+ = owner adds money, - = withdraws)

Identities (hold by construction, also for CLOSING batches that keep running after a switch):

    net_invested    = sum(INITIAL + TOP_UP) - sum(WITHDRAW)
    overall_pnl     = sum over batches of (equity_now_or_final - starting_capital)
    current_capital = net_invested + overall_pnl
                    = equity of the active batch when every older batch is CLOSED

Batches created before this module existed have no funding rows; their events are derived
(first batch = INITIAL starting capital, later ones = starting capital - previous equity)
and flagged ``derived``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.enums import BatchStatus, PositionStatus
from storage.models import Batch, CapitalAccount, FundingEvent, Position, Trade

_EPS = 0.005  # half a paisa: below this a top-up is "NONE"


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 2)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def batch_equity(session: Session, batch_id: int) -> dict[str, float]:
    """Equity of a batch now: cash + blocked + reserve + open unrealized (net of entry charges).

    Same definition as ``CapitalManager.snapshot`` (marked at each position's last price).
    """
    acct = session.get(CapitalAccount, batch_id)
    if acct is None:
        return {"cash": 0.0, "blocked": 0.0, "reserve": 0.0, "unrealized": 0.0, "equity": 0.0, "open_positions": 0}
    unrealized = 0.0
    n_open = 0
    rows = session.execute(select(Position).where(Position.batch_id == batch_id,
                                                  Position.status == PositionStatus.OPEN.value)).scalars()
    for p in rows:
        n_open += 1
        px = p.last_price if p.last_price else p.avg_entry
        sign = 1.0 if p.direction == "LONG" else -1.0
        unrealized += sign * (px - p.avg_entry) * p.qty - (p.entry_charges or 0.0)
    equity = acct.trading_cash + acct.blocked + acct.reserve + unrealized
    return {"cash": acct.trading_cash, "blocked": acct.blocked, "reserve": acct.reserve,
            "unrealized": unrealized, "equity": equity, "open_positions": n_open}


def _batch_trade_stats(session: Session, batch_id: int) -> dict[str, Any]:
    """Closed-position stats (a position with partial exits counts once, by its total net)."""
    rows = session.execute(
        select(Trade.position_id, func.sum(Trade.net_pnl)).where(Trade.batch_id == batch_id)
        .group_by(Trade.position_id)
    ).all()
    closed_ids = set(session.execute(select(Position.id).where(
        Position.batch_id == batch_id, Position.status == PositionStatus.CLOSED.value)).scalars())
    nets = [float(n or 0.0) for pid, n in rows if pid in closed_ids]
    realized = float(session.scalar(select(func.coalesce(func.sum(Trade.net_pnl), 0.0))
                                    .where(Trade.batch_id == batch_id)) or 0.0)
    wins = sum(1 for n in nets if n > 0)
    losses = sum(1 for n in nets if n < 0)
    return {"trades": len(nets), "wins": wins, "losses": losses, "realized_net": realized,
            "win_rate": round(100.0 * wins / len(nets), 1) if nets else None}


def record_funding(session: Session, *, batch: Batch, prev_batch: Batch | None, prev_equity: float | None,
                   now: datetime, actor: str, note: str | None = None) -> FundingEvent:
    """Record the owner money movement for a newly created batch. Caller flushes / commits."""
    start = float(batch.starting_capital)
    if prev_batch is None or prev_equity is None:
        kind, amount = "INITIAL", start
    else:
        amount = start - float(prev_equity)
        kind = "NONE" if abs(amount) < _EPS else ("TOP_UP" if amount > 0 else "WITHDRAW")
        if kind == "NONE":
            amount = 0.0
    ev = FundingEvent(ts=now, batch_id=batch.id, kind=kind, amount=round(amount, 4), starting_capital=start,
                      prev_batch_id=prev_batch.id if prev_batch is not None else None,
                      prev_equity=round(float(prev_equity), 4) if prev_equity is not None else None,
                      actor=actor, note=note)
    session.add(ev)
    session.flush()
    return ev


def _events(session: Session, batches: list[Batch]) -> list[dict[str, Any]]:
    """Recorded funding events, plus derived ones for batches that have none."""
    recorded = {e.batch_id: e for e in session.execute(select(FundingEvent).order_by(FundingEvent.id)).scalars()}
    out: list[dict[str, Any]] = []
    prev: Batch | None = None
    for b in batches:
        e = recorded.get(b.id)
        if e is not None:
            out.append({"ts": _iso(e.ts), "batch_id": b.id, "kind": e.kind, "amount": _r(e.amount),
                        "prev_equity": _r(e.prev_equity), "note": e.note, "derived": False})
        elif prev is None:
            out.append({"ts": _iso(b.started_at or b.created_at), "batch_id": b.id, "kind": "INITIAL",
                        "amount": _r(b.starting_capital), "prev_equity": None,
                        "note": "first batch (derived)", "derived": True})
        else:
            prev_eq = batch_equity(session, prev.id)["equity"]
            amount = float(b.starting_capital) - prev_eq
            kind = "NONE" if abs(amount) < _EPS else ("TOP_UP" if amount > 0 else "WITHDRAW")
            out.append({"ts": _iso(b.started_at or b.created_at), "batch_id": b.id, "kind": kind,
                        "amount": _r(0.0 if kind == "NONE" else amount), "prev_equity": _r(prev_eq),
                        "note": "derived from the previous batch's equity", "derived": True})
        prev = b
    return out


def funding_summary(session: Session) -> dict[str, Any]:
    """Overall money picture across every (non-deleted) batch. Read-only."""
    batches = list(session.execute(select(Batch).where(Batch.deleted.is_(False)).order_by(Batch.id)).scalars())
    events = _events(session, batches)
    by_batch = {e["batch_id"]: e for e in events}
    deposited = sum(e["amount"] for e in events if e["kind"] in ("INITIAL", "TOP_UP"))
    withdrawn = -sum(e["amount"] for e in events if e["kind"] == "WITHDRAW")
    rows: list[dict[str, Any]] = []
    overall_pnl = 0.0
    for b in batches:
        eq = batch_equity(session, b.id)
        st = _batch_trade_stats(session, b.id)
        pnl = eq["equity"] - float(b.starting_capital)
        overall_pnl += pnl
        ev = by_batch.get(b.id, {})
        rows.append({
            "batch_id": b.id, "name": b.name, "status": b.status, "starting_capital": _r(b.starting_capital),
            "top_up": ev.get("amount"), "funding_kind": ev.get("kind"), "carried_in": ev.get("prev_equity"),
            "equity": _r(eq["equity"]), "cash": _r(eq["cash"]), "blocked": _r(eq["blocked"]),
            "reserve": _r(eq["reserve"]), "unrealized": _r(eq["unrealized"]), "open_positions": eq["open_positions"],
            "realized_net": _r(st["realized_net"]), "pnl": _r(pnl),
            "pnl_pct": _r(100.0 * pnl / float(b.starting_capital)) if b.starting_capital else None,
            "trades": st["trades"], "wins": st["wins"], "losses": st["losses"], "win_rate": st["win_rate"],
            "started_at": _iso(b.started_at or b.created_at), "ended_at": _iso(b.ended_at),
        })
    net_invested = deposited - withdrawn
    current = net_invested + overall_pnl
    return {
        "total_deposited": _r(deposited), "total_withdrawn": _r(withdrawn), "net_invested": _r(net_invested),
        "current_capital": _r(current), "overall_pnl": _r(overall_pnl),
        "overall_pnl_pct": _r(100.0 * overall_pnl / deposited) if deposited else None,
        "active_batch_id": next((b.id for b in reversed(batches)
                                 if b.status in (BatchStatus.ACTIVE.value, BatchStatus.PAUSED.value,
                                                 BatchStatus.HALTED.value)), None),
        "batches": rows, "events": events,
    }


def topup_preview(session: Session, starting_capital: float) -> dict[str, Any]:
    """What starting a new batch with ``starting_capital`` would need from the owner (read-only).

    The current batch's open INTRADAY positions are closed at market by NEW_BATCH, so the
    preview uses the equity marked at the last known prices.
    """
    from storage.batches import get_current_batch

    cur = get_current_batch(session)
    start = float(starting_capital)
    if cur is None:
        return {"current_batch_id": None, "current_equity": None, "open_positions": 0, "starting_capital": _r(start),
                "top_up": _r(start), "action": "ADD",
                "message": f"First batch: deposit ₹{start:,.2f} of paper money."}
    eq = batch_equity(session, cur.id)
    amount = start - eq["equity"]
    if abs(amount) < _EPS:
        action, msg = "NONE", f"Current equity ₹{eq['equity']:,.2f} already equals ₹{start:,.2f}: nothing to add."
    elif amount > 0:
        action, msg = "ADD", (f"Add ₹{amount:,.2f} to start the new batch with ₹{start:,.2f} "
                              f"(current batch equity ₹{eq['equity']:,.2f}).")
    else:
        action, msg = "WITHDRAW", (f"₹{-amount:,.2f} of profit will be taken out to start with ₹{start:,.2f} "
                                   f"(current batch equity ₹{eq['equity']:,.2f}).")
    if eq["open_positions"]:
        msg += f" {eq['open_positions']} open position(s) are valued at their last price; the final figure is fixed when they close."
    return {"current_batch_id": cur.id, "current_equity": _r(eq["equity"]), "open_positions": eq["open_positions"],
            "starting_capital": _r(start), "top_up": _r(0.0 if action == "NONE" else amount), "action": action,
            "message": msg}


__all__ = ["batch_equity", "funding_summary", "record_funding", "topup_preview"]
