"""Follow-up lists (owner watchlists) and the next-day plan.

Watchlists are owner data, not trading state: the web process writes them directly through
``data.watchlists`` (which audits every change), not through control commands. Every write still
passes the security middleware's CSRF + same-origin checks. ``data.watchlists`` is imported lazily:
until it exists every endpoint answers HTTP 501. Scanning a list goes through the engine as a
``SCAN_NOW`` control command (analysis only, never trades).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from core.clock import IST, to_ist
from core.enums import CommandType, Mode
from webui import queries as q
from webui.deps import enqueue, get_ctx, get_user
from webui.util import clean_symbol, iso, not_available, parse_day, r2, str_list, text

router = APIRouter(prefix="/api")

MODULE = "data.watchlists"
PLAN_STATUSES = ("ACTIVE", "TRIGGERED", "INVALIDATED")
PRE_OPEN_CHECK = time(9, 20)


def _fn(name: str, /) -> Callable[..., Any]:
    """``data.watchlists.<name>`` or HTTP 501 while the module is not deployed (or fails to import)."""
    try:
        mod = importlib.import_module(MODULE)
        fn = getattr(mod, name)
    except Exception as exc:  # noqa: BLE001 - ImportError, AttributeError or a half-written module
        logger.debug(f"[webui] {MODULE}.{name} unavailable: {exc}")
        raise not_available() from None
    if not callable(fn):
        raise not_available()
    return fn


def _bad_input(exc: Exception) -> HTTPException:
    msg = text(str(exc), 300) or "invalid request"
    low = msg.lower()
    if "not found" in low or "no such" in low or "unknown" in low:
        return HTTPException(status_code=404, detail=msg)
    if "exist" in low or "duplicate" in low or "already" in low:
        return HTTPException(status_code=409, detail=msg)
    return HTTPException(status_code=422, detail=msg)


def _call(ctx: Any, fn_name: str, /, *args: Any, write: bool = False, **kwargs: Any) -> Any:
    """Run one ``data.watchlists`` function in its own session (commit on writes; ValueError -> 4xx)."""
    fn = _fn(fn_name)
    with ctx.session() as s:
        try:
            out = fn(s, *args, **kwargs)
            if write:
                s.commit()
            return out
        except ValueError as exc:
            s.rollback()
            raise _bad_input(exc) from exc
        except IntegrityError as exc:
            s.rollback()
            raise HTTPException(status_code=409, detail="A list with that name (or that stock) already exists.") from exc


def _mode(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    m = str(value).strip().upper()
    if m not in {x.value for x in Mode}:
        raise HTTPException(status_code=422, detail=f"invalid mode {value!r}")
    return m


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def plan_view(plan: Any) -> dict[str, Any] | None:
    """Normalised next-day plan levels of one item (tolerates the engine's key variants)."""
    if not isinstance(plan, dict) or not plan:
        return None

    def first(*keys: str) -> Any:
        for k in keys:
            if plan.get(k) is not None:
                return plan.get(k)
        return None

    trigger = _num(first("trigger", "entry", "entry_price", "trigger_price"))
    stop = _num(first("stop", "stop_loss", "sl"))
    target = _num(first("target", "target_1", "t1"))
    rr = _num(first("rr", "risk_reward", "r_r"))
    if rr is None and trigger is not None and stop is not None and target is not None and trigger != stop:
        rr = abs(target - trigger) / abs(trigger - stop)
    affordable = first("affordable")
    if affordable is None and plan.get("not_affordable") is not None:
        affordable = not bool(plan.get("not_affordable"))
    direction = str(first("direction", "side") or "").upper()
    if direction in ("BUY", "LONG"):
        direction = "LONG"
    elif direction in ("SELL", "SHORT"):
        direction = "SHORT"
    else:
        direction = None
    reasons = plan.get("reasons") if plan.get("reasons") is not None else plan.get("notes")
    return {"setup": text(first("setup", "strategy", "pattern"), 80), "direction": direction,
            "trigger": r2(trigger), "stop": r2(stop), "target": r2(target), "rr": r2(rr),
            "score": r2(_num(first("score", "confluence"))), "qty": first("qty"),
            "affordable": None if affordable is None else bool(affordable),
            "reasons": str_list(reasons, 8, 300), "mode": text(first("mode"), 16)}


def _item_view(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    for k in ("note", "last_summary", "status_note"):
        out[k] = text(item.get(k), 800)
    for k in ("added_at", "last_analysis_at"):
        v = item.get(k)
        out[k] = iso(v) if isinstance(v, datetime) else v
    out["added_price"] = r2(item.get("added_price"))
    status = str(item.get("status") or "ACTIVE").upper()
    out["status"] = status
    out["plan_view"] = plan_view(item.get("plan"))
    return out


def _list_view(wl: dict[str, Any]) -> dict[str, Any]:
    out = dict(wl)
    for k in ("created_at", "updated_at"):
        v = wl.get(k)
        out[k] = iso(v) if isinstance(v, datetime) else v
    fd = wl.get("for_date")
    out["for_date"] = fd.isoformat() if isinstance(fd, date) else fd
    out["description"] = text(wl.get("description"), 400)
    if isinstance(wl.get("items"), list):
        out["items"] = [_item_view(i) for i in wl["items"] if isinstance(i, dict)]
    return out


def _existing(ctx: Any, watchlist_id: int) -> dict[str, Any]:
    wl = _call(ctx, "get_watchlist", watchlist_id)
    if wl is None:
        raise HTTPException(status_code=404, detail=f"list {watchlist_id} not found")
    return wl


def _added_price(ctx: Any, symbol: str) -> float | None:
    try:
        quote = ctx.quotes.get_many([symbol], wait=min(ctx.quote_wait, 1.5)).get(symbol)
    except Exception as exc:  # noqa: BLE001 - adding a stock never fails because of quotes
        logger.warning(f"[webui] quote for {symbol} failed: {exc}")
        return None
    d = q.quote_dict(symbol, quote)
    return d["ltp"] if d else None


# =========================================================================== lists


class ListBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=400)
    use_for_trading: bool = False


class ListPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=400)
    use_for_trading: bool | None = None
    archived: bool | None = None


class ItemBody(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    note: str | None = Field(default=None, max_length=500)
    mode: str | None = Field(default=None, max_length=16)
    decision_id: int | None = Field(default=None, ge=1)


class ScanBody(BaseModel):
    mode: str | None = Field(default=None, max_length=16)
    use_llm: bool = True


@router.get("/watchlists")
def watchlists(request: Request, include_archived: bool = Query(default=False)) -> dict[str, Any]:
    ctx = get_ctx(request)
    rows = _call(ctx, "list_watchlists", include_archived=include_archived) or []
    return {"lists": [_list_view(r) for r in rows if isinstance(r, dict)], "server_time": iso(ctx.clock.now())}


@router.post("/watchlists")
def create_list(body: ListBody, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Give the list a name.")
    wl = _call(ctx, "create_watchlist", name, write=True, now=ctx.clock.now(), kind="MANUAL",
               description=(body.description or "").strip() or None, use_for_trading=bool(body.use_for_trading),
               actor=user)
    logger.info(f"[webui] {user} created follow-up list {name!r}")
    return _list_view(wl)


# NOTE: declared before the ``{watchlist_id}`` routes so "default" is not parsed as an id.
@router.post("/watchlists/default/items")
def add_to_default(body: ItemBody, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    sym = clean_symbol(body.symbol)
    mode = _mode(body.mode)
    fn_default, fn_add = _fn("get_or_create_default"), _fn("add_item")
    price = _added_price(ctx, sym)
    now = ctx.clock.now()
    with ctx.session() as s:
        try:
            wl = fn_default(s, now=now, actor=user)
            item = fn_add(s, int(wl["id"]), sym, now=now, note=(body.note or "").strip() or None,
                          source="ANALYSIS" if body.decision_id else "MANUAL", added_price=price, mode=mode,
                          decision_id=body.decision_id, actor=user)
            s.commit()
        except ValueError as exc:
            s.rollback()
            raise _bad_input(exc) from exc
        except IntegrityError as exc:
            s.rollback()
            raise HTTPException(status_code=409, detail=f"{sym} is already on this list.") from exc
    return {"watchlist": _list_view({k: v for k, v in wl.items() if k != "items"}), "item": _item_view(item)}


@router.get("/watchlists/{watchlist_id}")
def get_list(watchlist_id: int, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    return _list_view(_existing(ctx, watchlist_id)) | {"server_time": iso(ctx.clock.now())}


@router.patch("/watchlists/{watchlist_id}")
def patch_list(watchlist_id: int, body: ListPatch, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    _existing(ctx, watchlist_id)
    name = body.name.strip() if body.name is not None else None
    if name is not None and not name:
        raise HTTPException(status_code=422, detail="The name cannot be empty.")
    wl = _call(ctx, "update_watchlist", watchlist_id, write=True, now=ctx.clock.now(), name=name,
               description=body.description.strip() if body.description is not None else None,
               use_for_trading=body.use_for_trading, archived=body.archived, actor=user)
    return _list_view(wl)


@router.delete("/watchlists/{watchlist_id}")
def delete_list(watchlist_id: int, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    wl = _existing(ctx, watchlist_id)
    _call(ctx, "delete_watchlist", watchlist_id, write=True, now=ctx.clock.now(), actor=user)
    logger.info(f"[webui] {user} deleted follow-up list #{watchlist_id} {wl.get('name')!r}")
    return {"ok": True, "id": watchlist_id}


@router.post("/watchlists/{watchlist_id}/items")
def add_item(watchlist_id: int, body: ItemBody, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    sym = clean_symbol(body.symbol)
    mode = _mode(body.mode)
    _existing(ctx, watchlist_id)
    price = _added_price(ctx, sym)
    item = _call(ctx, "add_item", watchlist_id, sym, write=True, now=ctx.clock.now(),
                 note=(body.note or "").strip() or None, source="ANALYSIS" if body.decision_id else "MANUAL",
                 added_price=price, mode=mode, decision_id=body.decision_id, actor=user)
    return _item_view(item)


@router.delete("/watchlists/{watchlist_id}/items/{item_id}")
def remove_item(watchlist_id: int, item_id: int, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    wl = _existing(ctx, watchlist_id)
    if not any(int(i.get("id", -1)) == item_id for i in wl.get("items") or [] if isinstance(i, dict)):
        raise HTTPException(status_code=404, detail=f"item {item_id} is not on list {watchlist_id}")
    _call(ctx, "remove_item", watchlist_id, item_id, write=True, now=ctx.clock.now(), actor=user)
    return {"ok": True, "id": item_id}


@router.post("/watchlists/{watchlist_id}/scan")
def scan_list(watchlist_id: int, request: Request, body: ScanBody | None = None) -> dict[str, Any]:
    """Queue a SCAN_NOW of this list's stocks (analysis only - never trades)."""
    ctx = get_ctx(request)
    user = get_user(request)
    body = body or ScanBody()
    wl = _existing(ctx, watchlist_id)
    if not wl.get("items"):
        raise HTTPException(status_code=400, detail="This list is empty - add a stock first.")
    mode = _mode(body.mode) or "INTRADAY"
    out = enqueue(ctx, user, CommandType.SCAN_NOW, {"mode": mode, "use_llm": bool(body.use_llm),
                                                    "watchlist_id": watchlist_id})
    return out | {"watchlist_id": watchlist_id, "name": wl.get("name"), "symbols": len(wl.get("items") or [])}


# =========================================================================== next-day plan


def plan_label(for_date: date | None, now: datetime) -> str:
    """"Plan for today (checked again at 09:20)" before the opening check, "Plan for tomorrow" after the
    close, else "Plan for <day>"."""
    if for_date is None:
        return "Plan"
    local = to_ist(now)
    today = local.date()
    if for_date == today:
        return "Plan for today (checked again at 09:20)" if local.time() < PRE_OPEN_CHECK else "Plan for today"
    if (for_date - today).days == 1:
        return "Plan for tomorrow"
    return f"Plan for {for_date:%a %d %b}"


@router.get("/plan")
def plan(request: Request, day_q: str | None = Query(default=None, alias="date")) -> dict[str, Any]:
    """Newest next-day plan (NEXT_DAY list) for ``date`` (default today, IST) or later."""
    ctx = get_ctx(request)
    now = ctx.clock.now()
    day = parse_day(day_q, to_ist(now).date())
    wl = _call(ctx, "latest_plan", on_or_after=day)
    if not wl:
        return {"plan": None, "date": day.isoformat(), "server_time": iso(now)}
    view = _list_view(wl)
    fd = wl.get("for_date")
    if isinstance(fd, str):
        try:
            fd = datetime.fromisoformat(fd[:10]).date()
        except ValueError:
            fd = None
    view["label"] = plan_label(fd if isinstance(fd, date) else None, now)
    statuses = [i.get("status") for i in view.get("items") or []]
    view["counts"] = {s: statuses.count(s) for s in PLAN_STATUSES}
    return {"plan": view, "date": day.isoformat(), "server_time": iso(now),
            "checked_at_open": datetime.combine(fd, PRE_OPEN_CHECK, tzinfo=IST).isoformat()
            if isinstance(fd, date) else None}


__all__ = ["plan_label", "plan_view", "router"]
