"""Scope-aware results: ``/api/scope/options``, ``/api/summary``, ``/api/results`` (+ legacy ``/api/today``)
and ``/api/postmortem`` for Today | Batch | Overall."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from storage.models import Batch, Position
from webui import queries as q
from webui.deps import get_ctx
from webui.scope import Scope, funding_or_fallback, resolve_scope
from webui.util import NotAvailable, iso, not_available, optional_attr, r2, text

router = APIRouter(prefix="/api")

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _scope_positions(session: Session, sc: Scope, now: datetime) -> list[Position]:
    """Open positions that belong to the scope (unrealized P&L is only meaningful for 'now')."""
    if sc.kind == "today":
        if sc.day == now.date():
            return q.open_positions(session, sc.batch_id)
        stmt = select(Position).where(Position.status == "OPEN", Position.opened_at >= sc.lo, Position.opened_at < sc.hi)
        if sc.batch_id is not None:
            stmt = stmt.where(Position.batch_id == sc.batch_id)
        return list(session.execute(stmt).scalars())
    return q.open_positions(session, sc.batch_id if sc.kind == "batch" else None)


def _open_rows(ctx: Any, positions: list[Position], live: bool, wait: float) -> list[dict[str, Any]]:
    quotes = ctx.quotes.get_many([p.symbol for p in positions], wait=wait) if (positions and live) else {}
    return [q.position_row(p, quotes.get(p.symbol)) for p in positions]


def _still_open(session: Session, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Positions loaded before the quote wait may have closed meanwhile; their exit leg is in the trade
    rows read now, so keep only the ones still OPEN (else they would count twice)."""
    if not rows:
        return rows
    ids = [r["id"] for r in rows]
    still = set(session.execute(select(Position.id).where(Position.id.in_(ids), Position.status == "OPEN")).scalars())
    return [r for r in rows if r["id"] in still]


def _unreal_by_batch(rows: list[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in rows:
        out[r["batch_id"]] = out.get(r["batch_id"], 0.0) + (r["unrealized"] or 0.0)
    return out


def _scope_rows(session: Session, sc: Scope) -> list[dict[str, Any]]:
    return q.trades_between(session, sc.lo, sc.hi, batch_id=sc.batch_id, legs_in_range_only=sc.is_today)


@router.get("/scope/options")
def scope_options(request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    now = ctx.clock.now()
    with ctx.session() as s:
        cur = q.current_batch(s)
        return {"today": now.date().isoformat(), "batches": q.batches_brief(s), "active_batch_id": cur.id if cur else None,
                "server_time": iso(now)}


# =========================================================================== summary


@router.get("/summary")
def summary(request: Request, scope: str | None = Query(default="today", max_length=10),
            date: str | None = Query(default=None), batch_id: int | None = Query(default=None)) -> dict[str, Any]:
    ctx = get_ctx(request)
    now = ctx.clock.now()
    with ctx.session() as s:
        sc = resolve_scope(s, now, scope, date, batch_id)
        positions = _scope_positions(s, sc, now)
    live = not sc.is_today or sc.day == now.date()
    open_rows = _open_rows(ctx, positions, live, ctx.quote_wait)
    with ctx.session() as s:
        open_rows = _still_open(s, open_rows)
        unreal_by_batch = _unreal_by_batch(open_rows)
        unreal = round(sum(unreal_by_batch.values()), 2)
        rows = _scope_rows(s, sc)
        t = q.trade_totals(rows)
        money = _money(s, sc, rows, open_rows, unreal_by_batch, t)
    kpis = {
        "realized": t["net"], "gross": t["gross"], "charges": t["charges"], "unrealized": r2(unreal),
        "total": r2((t["net"] or 0.0) + unreal), "trades": t["closed"], "wins": t["wins"], "losses": t["losses"],
        "flat": t["flat"], "win_rate": t["win_rate"], "avg_r": t["avg_r"], "avg_win": t["avg_win"],
        "avg_loss": t["avg_loss"], "profit_factor": t["profit_factor"], "best": t["best"], "worst": t["worst"],
        "open_positions": len(open_rows),
    }
    return {"scope": sc.as_dict(), "server_time": iso(now), "kpis": kpis, "money": money}


def _money(s: Session, sc: Scope, rows: list[dict[str, Any]], open_rows: list[dict[str, Any]],
           unreal_by_batch: dict[int, float], t: dict[str, Any]) -> dict[str, Any]:
    if sc.kind == "today":
        nets: dict[int, float] = {}
        for r in rows:
            nets[r["batch_id"]] = nets.get(r["batch_id"], 0.0) + (r["net"] or 0.0)
        ids = sorted(set(nets) | {r["batch_id"] for r in open_rows})
        names = {b.id: b for b in s.execute(select(Batch).where(Batch.id.in_(ids))).scalars()} if ids else {}
        unreal = round(sum(unreal_by_batch.values()), 2)
        return {"kind": "today", "day_pnl": r2((t["net"] or 0.0) + unreal), "realized": t["net"], "unrealized": r2(unreal),
                "batches": [{"id": i, "name": names[i].name if i in names else f"#{i}",
                             "status": names[i].status if i in names else None, "net": r2(nets.get(i, 0.0)),
                             "trades": sum(1 for r in rows if r["batch_id"] == i and r["result"] != "OPEN")}
                            for i in ids]}
    funding, source = funding_or_fallback(s, unreal_by_batch)
    if sc.kind == "batch":
        b = s.get(Batch, sc.batch_id)
        row = next((r for r in funding.get("batches", []) if r.get("batch_id") == sc.batch_id), {}) or {}
        cap = q.capital_view(s, sc.batch_id, unreal_by_batch.get(sc.batch_id, 0.0))
        starting = float(b.starting_capital) if b else (row.get("starting_capital") or 0.0)
        equity = cap["equity"] if cap else row.get("equity")
        pnl = (equity - starting) if equity is not None else row.get("pnl")
        return {"kind": "batch", "source": source, "batch": q.batch_brief(b), "status": b.status if b else None,
                "starting_capital": r2(starting), "top_up": r2(row.get("top_up")), "funding_kind": row.get("funding_kind"),
                "carried_in": r2(row.get("carried_in")), "equity": r2(equity), "pnl": r2(pnl),
                "pnl_pct": r2(pnl / starting * 100.0) if pnl is not None and starting else None,
                "reserve": cap["reserve"] if cap else row.get("reserve"), "drawdown_pct": cap["drawdown_pct"] if cap else None,
                "tier": cap["tier"] if cap else None, "hwm": cap["hwm"] if cap else None,
                "halted": cap["halted"] if cap else False, "started_at": iso(b.started_at or b.created_at) if b else None,
                "ended_at": iso(b.ended_at) if b else None, "open_positions": len(open_rows)}
    return {"kind": "all", "source": source,
            **{k: r2(funding.get(k)) for k in ("net_invested", "total_deposited", "total_withdrawn", "current_capital",
                                               "overall_pnl", "overall_pnl_pct")},
            "batches": len(funding.get("batches", [])), "active_batch_id": funding.get("active_batch_id"),
            "realized_all": t["net"]}


# =========================================================================== results (+ legacy /today)


def _results(ctx: Any, sc: Scope, now: datetime) -> dict[str, Any]:
    with ctx.session() as s:
        positions = _scope_positions(s, sc, now)
    live = not sc.is_today or sc.day == now.date()
    open_rows = _open_rows(ctx, positions, live, 1.0)
    with ctx.session() as s:
        open_rows = _still_open(s, open_rows)
        unreal_by_batch = _unreal_by_batch(open_rows)
        rows = _scope_rows(s, sc)
        funnel = q.decision_funnel(s, sc.lo, sc.hi, sc.batch_id)
        timeline = q.day_timeline(s, sc.lo, sc.hi, sc.batch_id) if sc.is_today else []
        if sc.is_today:
            eq_batch = sc.batch_id
            if eq_batch is None:
                cur = q.current_batch(s)
                eq_batch = cur.id if cur else None
            equity = q.day_equity(s, sc.lo, sc.hi, eq_batch)
        else:
            ids = [sc.batch_id] if sc.kind == "batch" else [b["id"] for b in reversed(q.batches_brief(s))]
            curves: dict[int, list[dict[str, Any]]] = {}
            for bid in ids:
                cap = q.capital_view(s, bid, unreal_by_batch.get(bid, 0.0))
                curves[bid] = q.batch_equity_curve(s, bid, cap["equity"] if cap else None, now)
            if sc.kind == "batch":
                equity = q.concat_curves(list(curves.values()))
            else:   # one continuous line of total capital (invested so far + every batch's P&L)
                funding, _src = funding_or_fallback(s, unreal_by_batch)
                equity = q.overall_capital_curve(curves, funding.get("batches", []))
    summary = q.trade_totals(rows)
    unreal = round(sum(r["unrealized"] or 0.0 for r in open_rows), 2)
    summary |= {"unrealized": r2(unreal), "total": r2((summary["net"] or 0.0) + unreal), "open": len(open_rows)}
    return {"scope": sc.as_dict(), "day": sc.day.isoformat() if sc.day else None,
            "is_today": bool(sc.is_today and sc.day == now.date()), "batch_id": sc.batch_id,
            "server_time": iso(now), "summary": summary, "trades": rows, "open_positions": open_rows,
            "funnel": funnel, "timeline": timeline, "equity": equity}


@router.get("/results")
def results(request: Request, scope: str | None = Query(default="today", max_length=10),
            date: str | None = Query(default=None), batch_id: int | None = Query(default=None)) -> dict[str, Any]:
    ctx = get_ctx(request)
    now = ctx.clock.now()
    with ctx.session() as s:
        sc = resolve_scope(s, now, scope, date, batch_id)
    return _results(ctx, sc, now)


@router.get("/today")
def today(request: Request, date: str | None = Query(default=None), batch_id: int | None = Query(default=None),
          scope: str | None = Query(default=None, max_length=10)) -> dict[str, Any]:
    """Legacy alias of ``/api/results`` (defaults to the Today scope)."""
    return results(request, scope=scope or "today", date=date, batch_id=batch_id)


# =========================================================================== post-mortem


@router.get("/postmortem")
def postmortem(request: Request, scope: str | None = Query(default="today", max_length=10),
               date: str | None = Query(default=None), batch_id: int | None = Query(default=None),
               limit: int = Query(default=50, ge=1, le=100)) -> Any:
    """Today: ``analyze_day``. Batch / Overall: ``analyze_position`` over the latest ``limit`` closed
    positions (``data=None`` - no candle fetches - to stay fast), aggregated into top issues."""
    ctx = get_ctx(request)
    now = ctx.clock.now()
    with ctx.session() as s:
        sc = resolve_scope(s, now, scope, date, batch_id)
        if sc.is_today:
            try:
                fn = optional_attr("analytics.postmortem", "analyze_day")
            except NotAvailable:
                raise not_available() from None
            try:
                out = fn(s, sc.day, batch_id=sc.batch_id, data=ctx.get_data())
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[webui] analyze_day({sc.day}) failed: {exc}")
                raise HTTPException(status_code=500, detail=f"post-mortem failed: {text(str(exc), 200)}") from exc
            return {**out, "scope": sc.as_dict()} if isinstance(out, dict) else out
        try:
            fn = optional_attr("analytics.postmortem", "analyze_position")
        except NotAvailable:
            raise not_available() from None
        ids = q.closed_position_ids(s, batch_id=sc.batch_id, limit=limit)
        analyses: list[dict[str, Any]] = []
        failed = 0
        for pid in ids:
            try:
                a = fn(s, pid, data=None)
            except Exception as exc:  # noqa: BLE001 - one bad position must not hide the rest
                logger.warning(f"[webui] analyze_position({pid}) failed: {exc}")
                s.rollback()
                failed += 1
                continue
            if isinstance(a, dict):
                analyses.append(a)
    return _aggregate(analyses, sc, limited=len(ids) >= limit, failed=failed)


def _aggregate(analyses: list[dict[str, Any]], sc: Scope, *, limited: bool, failed: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    info: dict[str, dict[str, Any]] = {}
    for a in analyses:
        seen: set[str] = set()
        for d in a.get("diagnosis") or []:
            if not isinstance(d, dict):
                continue
            code = str(d.get("code") or d.get("title") or "?")
            if code in seen:
                continue
            seen.add(code)
            counts[code] += 1
            cur = info.setdefault(code, {"code": code, "title": d.get("title"), "fix": d.get("fix"),
                                         "fixed_by": d.get("fixed_by"), "severity": d.get("severity")})
            sev = d.get("severity")
            if SEVERITY_ORDER.get(sev, 3) < SEVERITY_ORDER.get(cur.get("severity"), 3):
                cur["severity"] = sev
    top = [info[c] | {"count": n} for c, n in counts.items()]
    top.sort(key=lambda i: (-i["count"], SEVERITY_ORDER.get(i.get("severity"), 3)))
    wins = sum(1 for a in analyses if a.get("result") == "WIN")
    losses = sum(1 for a in analyses if a.get("result") == "LOSS")
    trades = [{k: a.get(k) for k in ("position_id", "symbol", "mode", "direction", "strategy", "result", "net_pnl",
                                     "r_multiple", "exit_time", "exit_reasons")}
              | {"issues": [d.get("code") for d in (a.get("diagnosis") or []) if isinstance(d, dict)]}
              for a in analyses]
    return {"scope": sc.as_dict(), "day": None, "trades": trades,
            "summary": {"trades": len(analyses), "wins": wins, "losses": losses,
                        "net_pnl": r2(sum(float(a.get("net_pnl") or 0.0) for a in analyses)),
                        "charges": r2(sum(float(a.get("charges") or 0.0) for a in analyses))},
            "top_issues": top[:12], "limited": limited, "failed": failed}
