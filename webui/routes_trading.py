"""Today, trade journal, trade detail / post-mortem, batches and the analysis (decision) feed."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from core.enums import CommandType
from storage.models import Batch, ControlCommand, DecisionRow, Order, Position, Trade
from webui import queries as q
from webui.deps import enqueue, get_ctx, get_user
from webui.scope import funding_or_fallback, resolve_scope
from webui.util import (
    NOT_AVAILABLE,
    NotAvailable,
    clean_symbol,
    day_bounds,
    iso,
    not_available,
    optional_attr,
    parse_day,
    r2,
    text,
)

router = APIRouter(prefix="/api")

RESULTS = ("WIN", "LOSS", "FLAT", "OPEN")


def _batches_list(session: Any) -> list[dict[str, Any]]:
    return [{"id": b.id, "name": b.name, "status": b.status}
            for b in session.execute(select(Batch).where(Batch.deleted.is_(False)).order_by(Batch.id.desc())).scalars()]


# =========================================================================== journal


@router.get("/trades")
def trades(request: Request, batch_id: int | None = Query(default=None), date_from: str | None = Query(None, alias="from"),
           date_to: str | None = Query(None, alias="to"), mode: str | None = Query(default=None),
           result: str | None = Query(default=None), symbol: str | None = Query(default=None),
           limit: int = Query(default=300, ge=1, le=5000), scope: str | None = Query(default=None, max_length=10),
           date: str | None = Query(default=None)) -> dict[str, Any]:
    """Completed-trade journal. ``scope=today|batch|all`` (+ ``date`` / ``batch_id``) sets the base range;
    explicit ``from`` / ``to`` still narrow it (legacy params keep working without ``scope``)."""
    ctx = get_ctx(request)
    now = ctx.clock.now()
    lo = day_bounds(parse_day(date_from, now.date()))[0] if date_from else None
    hi = day_bounds(parse_day(date_to, now.date()))[1] if date_to else None
    sym = clean_symbol(symbol) if symbol else None
    mode_v = mode.strip().upper() if mode else None
    res = result.strip().upper() if result else None
    if res and res not in RESULTS:
        raise HTTPException(status_code=422, detail=f"result must be one of {', '.join(RESULTS)}")
    legs_only = False
    scope_info = None
    with ctx.session() as s:
        if scope:
            sc = resolve_scope(s, now, scope, date, batch_id)
            scope_info = sc.as_dict()
            batch_id = sc.batch_id
            lo = max(x for x in (lo, sc.lo) if x is not None) if (lo or sc.lo) else None
            hi = min(x for x in (hi, sc.hi) if x is not None) if (hi or sc.hi) else None
            legs_only = sc.is_today
        rows = q.trades_between(s, lo, hi, batch_id=batch_id, mode=mode_v, symbol=sym, legs_in_range_only=legs_only)
        batches = _batches_list(s)
    if res:
        rows = [r for r in rows if r["result"] == res]
    totals = q.trade_totals(rows)
    return {"rows": rows[:limit], "count": len(rows), "truncated": len(rows) > limit, "totals": totals,
            "batches": batches, "modes": ["INTRADAY", "BTST", "SWING", "POSITIONAL"], "scope": scope_info}


def _chart_window(entry: datetime | None, exit_: datetime | None, now: datetime) -> dict[str, Any]:
    start = entry or now
    end = exit_ or now
    span_days = (end - start).total_seconds() / 86_400
    tf = "5m" if span_days <= 2 else ("15m" if span_days <= 6 else "1h")
    return {"timeframe": tf, "from": iso(start - timedelta(minutes=90)), "to": iso(end + timedelta(minutes=60))}


@router.get("/trades/{position_id}")
def trade_detail(position_id: int, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    now = ctx.clock.now()
    with ctx.session() as s:
        pos = s.get(Position, position_id)
        if pos is None:
            raise HTTPException(status_code=404, detail=f"position {position_id} not found")
        legs = list(s.execute(select(Trade).where(Trade.position_id == position_id)
                              .order_by(Trade.exit_time, Trade.id)).scalars())
        orders = list(s.execute(select(Order).where(Order.position_id == position_id).order_by(Order.id)).scalars())
        if pos.decision_id:
            orders_by_decision = list(s.execute(select(Order).where(Order.decision_id == pos.decision_id,
                                                                    Order.position_id.is_(None))).scalars())
            orders += orders_by_decision
        row = q.position_trade_row(legs, pos) if legs else None
        open_row = q.position_row(pos) if pos.status == "OPEN" else None
        decision = None
        if pos.decision_id:
            d = s.get(DecisionRow, pos.decision_id)
            if d is not None:
                decision = {"id": d.id, "time": iso(d.created_at), "confidence": d.confidence,
                            "confluence": r2(d.confluence_score), "grade": d.grade, "strategy": d.strategy,
                            "final_action": d.final_action, "latency_ms": d.latency_ms,
                            "risk_rule": d.risk_rule, "behaviour": q.behaviour_view(d.context_json, d.entry_price)}
        position = {
            "id": pos.id, "batch_id": pos.batch_id, "symbol": pos.symbol, "mode": pos.mode, "product": pos.product,
            "direction": pos.direction, "status": pos.status, "qty": pos.qty, "initial_qty": pos.initial_qty,
            "avg_entry": r2(pos.avg_entry), "stop_loss": r2(pos.stop_loss),
            "initial_stop_loss": r2(pos.initial_stop_loss), "target_1": r2(pos.target_1),
            "target_2": r2(pos.target_2), "trailing_rule": pos.trailing_rule, "opened_at": iso(pos.opened_at),
            "closed_at": iso(pos.closed_at), "strategy": pos.strategy, "grade": pos.grade, "regime": pos.regime,
            "source": pos.source, "sector": pos.sector, "highest_price": r2(pos.highest_price),
            "lowest_price": r2(pos.lowest_price), "last_price": r2(pos.last_price),
            "partial_done": bool(pos.partial_done), "breakeven_moved": bool(pos.breakeven_moved),
            "entry_charges": r2(pos.entry_charges), "decision_id": pos.decision_id, "approval_id": pos.approval_id,
            "sl_history": q.sl_moves(pos),
        }
        chart = _chart_window(pos.opened_at, pos.closed_at or (legs[-1].exit_time if legs else None), now)
        postmortem = None
        pm_error: dict[str, Any] | None = None
        try:
            fn = optional_attr("analytics.postmortem", "analyze_position")
            postmortem = fn(s, position_id, data=ctx.get_data())
        except NotAvailable:
            pm_error = {"status": 501, "detail": NOT_AVAILABLE}
        except Exception as exc:  # noqa: BLE001 - the page must still work without the post-mortem
            logger.warning(f"[webui] analyze_position({position_id}) failed: {exc}")
            s.rollback()
            pm_error = {"status": 500, "detail": f"post-mortem failed: {text(str(exc), 200)}"}
    if isinstance(postmortem, dict) and isinstance(postmortem.get("chart"), dict):
        chart = {**chart, **{k: v for k, v in postmortem["chart"].items() if v}}
    return {"position": position, "row": row, "open": open_row, "legs": [q.leg_row(t) for t in legs],
            "orders": [q.order_row(o) for o in orders], "decision": decision, "chart": chart,
            "postmortem": postmortem, "postmortem_available": postmortem is not None, "postmortem_error": pm_error}


@router.get("/postmortem/day")
def postmortem_day(request: Request, date: str | None = Query(default=None),
                   batch_id: int | None = Query(default=None)) -> Any:
    ctx = get_ctx(request)
    day = parse_day(date, ctx.clock.now().date())
    try:
        fn = optional_attr("analytics.postmortem", "analyze_day")
    except NotAvailable:
        raise not_available() from None
    with ctx.session() as s:
        try:
            return fn(s, day, batch_id=batch_id, data=ctx.get_data())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[webui] analyze_day({day}) failed: {exc}")
            raise HTTPException(status_code=500, detail=f"post-mortem failed: {text(str(exc), 200)}") from exc


# =========================================================================== batches


def _unrealized_by_batch(ctx: Any, wait: float = 0.0) -> dict[int, float]:
    with ctx.session() as s:
        positions = q.open_positions(s)
    quotes = ctx.quotes.get_many([p.symbol for p in positions], wait=wait) if positions else {}
    out: dict[int, float] = {}
    for p in positions:
        row = q.position_row(p, quotes.get(p.symbol))
        out[p.batch_id] = out.get(p.batch_id, 0.0) + (row["unrealized"] or 0.0)
    return out


def _last_new_batch_command(session: Any) -> dict[str, Any] | None:
    row = session.execute(select(ControlCommand).where(ControlCommand.command_type == "NEW_BATCH")
                          .order_by(ControlCommand.id.desc()).limit(1)).scalar_one_or_none()
    return q.command_view(row) if row is not None else None


@router.get("/batches")
def batches(request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    unreal = _unrealized_by_batch(ctx)
    settings = ctx.settings()
    with ctx.session() as s:
        summary, source = funding_or_fallback(s, unreal)
        cur = q.current_batch(s)
        last_cmd = _last_new_batch_command(s)
        n_batches = int(s.execute(select(func.count(Batch.id))).scalar_one() or 0)
    return {**summary, "source": source, "current_batch_id": cur.id if cur else None,
            "current_batch_name": cur.name if cur else None,
            "default_starting_capital": r2(settings.capital.starting_capital),
            "suggested_name": f"Batch {n_batches + 1}", "last_new_batch_command": last_cmd}


@router.get("/batches/topup-preview")
def topup_preview(request: Request, capital: float = Query(..., gt=0, le=1e9)) -> Any:
    ctx = get_ctx(request)
    try:
        fn = optional_attr("capital.funding", "topup_preview")
    except NotAvailable:
        return JSONResponse(status_code=501, content={"detail": NOT_AVAILABLE, "estimate": _topup_estimate(ctx, capital)})
    with ctx.session() as s:
        return fn(s, float(capital))


def _topup_estimate(ctx: Any, capital: float) -> dict[str, Any]:
    """Server-side estimate shown while ``capital.funding.topup_preview`` is not available."""
    unreal = _unrealized_by_batch(ctx)
    with ctx.session() as s:
        cur = q.current_batch(s)
        cap = q.capital_view(s, cur.id, unreal.get(cur.id, 0.0)) if cur else None
        n_open = len(q.open_positions(s, cur.id)) if cur else 0
    equity = cap["equity"] if cap else 0.0
    diff = round(capital - (equity or 0.0), 2)
    action = "ADD" if diff > 0.004 else ("WITHDRAW" if diff < -0.004 else "NONE")
    if action == "ADD":
        msg = f"You need to add ₹{diff:,.2f} to start with ₹{capital:,.2f}"
    elif action == "WITHDRAW":
        msg = f"₹{-diff:,.2f} will be withdrawn (current equity ₹{equity:,.2f})"
    else:
        msg = "No top-up needed: current equity equals the new starting capital"
    return {"current_batch_id": cur.id if cur else None, "current_equity": r2(equity), "open_positions": n_open,
            "starting_capital": r2(capital), "top_up": r2(diff), "action": action, "message": msg,
            "source": "estimate"}


@router.get("/batches/{batch_id}")
def batch_detail(batch_id: int, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    now = ctx.clock.now()
    unreal = _unrealized_by_batch(ctx, wait=1.0)
    with ctx.session() as s:
        b = s.get(Batch, batch_id)
        if b is None:
            raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
        capital = q.capital_view(s, batch_id, unreal.get(batch_id, 0.0))
        rows = q.batch_detail_rows(s, batch_id)
        curve = q.batch_equity_curve(s, batch_id, capital["equity"] if capital else None, now)
        funding_row = None
        try:
            summary = optional_attr("capital.funding", "funding_summary")(s)
            funding_row = next((r for r in summary.get("batches", []) if r.get("batch_id") == batch_id), None)
        except NotAvailable:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[webui] funding_summary failed: {exc}")
    return {"batch": q.batch_brief(b), "capital": capital, "totals": q.trade_totals(rows),
            "by_mode": q.breakdown(rows, "mode"), "by_strategy": q.breakdown(rows, "strategy"),
            "equity_curve": curve, "recent_trades": rows[:50], "funding": funding_row}


class NewBatchBody(BaseModel):
    starting_capital: float | None = Field(default=None, gt=0, le=1e9)
    name: str | None = Field(default=None, max_length=120)
    open_cnc_handling: str = Field(default="EXIT", pattern="^(EXIT|RUN_TO_COMPLETION)$")
    confirm_text: str = Field(default="", max_length=120)


@router.post("/batches/new")
def new_batch(body: NewBatchBody, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    settings = ctx.settings()
    capital = float(body.starting_capital) if body.starting_capital else float(settings.capital.starting_capital)
    with ctx.session() as s:
        cur = q.current_batch(s)
        typed = (body.confirm_text or "").strip()
        if typed != "CONFIRM" and not (cur is not None and typed == cur.name):
            raise HTTPException(status_code=400, detail="Type CONFIRM (or the current batch name) to start a new batch.")
        name = (body.name or "").strip()
        if name:
            if s.execute(select(Batch.id).where(Batch.name == name)).first() is not None:
                raise HTTPException(status_code=409, detail=f"A batch named {name!r} already exists.")
        else:
            n = int(s.execute(select(func.count(Batch.id))).scalar_one() or 0) + 1
            name = f"Batch {n}"
            while s.execute(select(Batch.id).where(Batch.name == name)).first() is not None:
                n += 1
                name = f"Batch {n}"
    result = enqueue(ctx, user, CommandType.NEW_BATCH, {"name": name, "starting_capital": round(capital, 2),
                                                        "open_cnc_handling": body.open_cnc_handling,
                                                        "confirm_text": typed})
    return result | {"name": name, "starting_capital": r2(capital)}


# =========================================================================== decisions


@router.get("/decisions")
def decisions(request: Request, since_id: int | None = Query(default=None, ge=0),
              before_id: int | None = Query(default=None, ge=1), limit: int = Query(default=100, ge=1, le=500),
              action: str | None = Query(default=None, max_length=120), symbol: str | None = Query(default=None),
              hide_filtered: bool = Query(default=False), date: str | None = Query(default=None),
              scope: str | None = Query(default=None, max_length=10), batch_id: int | None = Query(default=None)
              ) -> dict[str, Any]:
    """Analysis feed. ``scope=today|batch|all`` (+ ``date`` / ``batch_id``); legacy ``date`` alone = that day."""
    ctx = get_ctx(request)
    now = ctx.clock.now()
    lo = hi = None
    scope_info = None
    sym = clean_symbol(symbol) if symbol else None
    with ctx.session() as s:
        if scope:
            sc = resolve_scope(s, now, scope, date, batch_id)
            lo, hi, batch_id, scope_info = sc.lo, sc.hi, sc.batch_id, sc.as_dict()
        elif date:
            lo, hi = day_bounds(parse_day(date, now.date()))
        rows = q.decision_feed(s, since_id=since_id, before_id=before_id, limit=limit, action=action, symbol=sym,
                               hide_filtered=hide_filtered, lo=lo, hi=hi, batch_id=batch_id)
        latest = int(s.execute(select(func.max(DecisionRow.id))).scalar_one() or 0)
    return {"rows": rows, "latest_id": latest, "scope": scope_info}


@router.get("/decisions/{decision_id}")
def decision(decision_id: int, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    with ctx.session() as s:
        detail = q.decision_detail(s, decision_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"decision {decision_id} not found")
    return detail
