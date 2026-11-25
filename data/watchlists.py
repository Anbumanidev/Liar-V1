"""Follow-up lists: stocks the owner tracks, plus the system's end-of-day plan for the next session.

Tables: ``watchlists`` / ``watchlist_items`` (``storage.models.Watchlist`` / ``WatchlistItem``).

* MANUAL lists are the owner's ("Follow-up" is the default one). ``use_for_trading=True`` makes
  INTRADAY cycles analyse the list's ACTIVE stocks first - every trade still goes through the
  normal pipeline and the deterministic risk manager; nothing here places orders.
* NEXT_DAY lists are written by ``scheduler.jobs.job_next_day_plan`` at 15:40 for ``for_date``
  (items carry the plan levels in ``plan_json``) and re-checked by ``job_opening_check`` at 09:20.

Every function takes an open session; CALLERS COMMIT. Bad input raises ``ValueError`` with a clear
message, worded for the web layer's 4xx mapping: "... not found" for unknown ids / items (404),
"... already exists" for duplicate names (409), "invalid ..." / "... must ..." / "... too long" for
everything else (422). Each create/update/delete/add/remove writes an ``audit_log`` row and one
``activity("watchlist")`` line. ``trading_symbols`` treats ACTIVE and TRIGGERED items as live (a plan
stock that triggered at the opening check is exactly what should be analysed); INVALIDATED ones are
skipped. Datetimes in the returned dicts are ISO strings (IST).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.activity import activity
from core.clock import SimulatedClock, to_ist
from core.enums import Mode
from storage.audit import audit
from storage.models import Watchlist, WatchlistItem

DEFAULT_LIST_NAME = "Follow-up"
KINDS = ("MANUAL", "NEXT_DAY", "SYSTEM")
SOURCES = ("MANUAL", "ANALYSIS", "SCAN", "EOD_PLAN")
STATUSES = ("ACTIVE", "TRIGGERED", "INVALIDATED")
#: Item statuses that are still "live" for trading (INVALIDATED items are only shown, never traded).
LIVE_STATUSES = ("ACTIVE", "TRIGGERED")
MAX_NAME = 80
MAX_NOTE = 2000
MAX_SUMMARY = 1000
SYMBOL_RE = re.compile(r"^[A-Z0-9&.\-]{1,32}$")


# =========================================================================== helpers


def normalize_symbol(symbol: Any) -> str:
    """Upper-cased, trimmed NSE symbol; ``ValueError`` if it is not ``^[A-Z0-9&.-]{1,32}$``."""
    sym = str(symbol or "").strip().upper()
    if not SYMBOL_RE.match(sym):
        raise ValueError(f"invalid symbol {symbol!r}: use 1-32 characters A-Z, 0-9, '&', '.' or '-'")
    return sym


def normalize_symbols(symbols: Iterable[Any], *, limit: int | None = None) -> tuple[list[str], list[str]]:
    """``(valid upper-cased de-duplicated symbols, rejected raw values)``; at most ``limit`` kept."""
    ok: list[str] = []
    bad: list[str] = []
    for raw in symbols or []:
        try:
            sym = normalize_symbol(raw)
        except ValueError:
            bad.append(str(raw))
            continue
        if sym not in ok:
            ok.append(sym)
    return (ok[:limit] if limit is not None else ok), bad


def _clean_name(name: Any) -> str:
    n = " ".join(str(name or "").split())
    if not n:
        raise ValueError("list name must not be empty")
    if len(n) > MAX_NAME:
        raise ValueError(f"list name is too long ({len(n)} > {MAX_NAME} characters)")
    return n


def _clean_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s[:limit] if s else None


def _iso(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_ist(value).isoformat()
    return value.isoformat()


def _clock(now: datetime) -> SimulatedClock:
    return SimulatedClock(now)


def _norm_mode(mode: Any) -> str | None:
    if mode is None or mode == "":
        return None
    try:
        return Mode(str(mode).upper()).value
    except ValueError as exc:
        raise ValueError(f"invalid mode {mode!r} (use one of {', '.join(m.value for m in Mode)})") from exc


def _get_list(session: Session, watchlist_id: Any) -> Watchlist:
    try:
        wid = int(watchlist_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid watchlist id {watchlist_id!r}") from exc
    wl = session.get(Watchlist, wid)
    if wl is None:
        raise ValueError(f"watchlist #{wid} not found")
    return wl


def _name_taken(session: Session, name: str, exclude_id: int | None = None) -> bool:
    q = select(Watchlist.id).where(func.lower(Watchlist.name) == name.lower())
    if exclude_id is not None:
        q = q.where(Watchlist.id != exclude_id)
    return session.execute(q).first() is not None


def _count_items(session: Session, watchlist_id: int) -> int:
    return int(session.scalar(select(func.count(WatchlistItem.id)).where(WatchlistItem.watchlist_id == watchlist_id))
               or 0)


def _list_dict(wl: Watchlist, items: int) -> dict[str, Any]:
    return {"id": wl.id, "name": wl.name, "kind": wl.kind, "description": wl.description,
            "use_for_trading": bool(wl.use_for_trading), "for_date": _iso(wl.for_date),
            "archived": bool(wl.archived), "created_at": _iso(wl.created_at), "updated_at": _iso(wl.updated_at),
            "items": items}


def _log(session: Session, actor: str, action: str, wl_id: int | None, details: dict[str, Any], now: datetime,
         message: str, **fields: Any) -> None:
    audit(session, actor, action, f"watchlist:{wl_id}" if wl_id is not None else "watchlist", details,
          clock=_clock(now))
    activity("watchlist", message, clock=_clock(now), watchlist_id=wl_id, actor=actor, **fields)


def item_dict(item: WatchlistItem) -> dict[str, Any]:
    """JSON-friendly view of one ``WatchlistItem``."""
    return {"id": item.id, "watchlist_id": item.watchlist_id, "symbol": item.symbol, "mode": item.mode,
            "source": item.source, "note": item.note, "added_at": _iso(item.added_at), "added_by": item.added_by,
            "added_price": item.added_price, "plan": item.plan_json, "status": item.status,
            "status_note": item.status_note, "decision_id": item.decision_id,
            "last_analysis_id": item.last_analysis_id, "last_analysis_at": _iso(item.last_analysis_at),
            "last_summary": item.last_summary}


def _items(session: Session, watchlist_id: int) -> list[WatchlistItem]:
    return list(session.execute(select(WatchlistItem).where(WatchlistItem.watchlist_id == watchlist_id)
                                .order_by(WatchlistItem.added_at.desc(), WatchlistItem.id.desc())).scalars())


def _score(item: WatchlistItem) -> float:
    try:
        return float((item.plan_json or {}).get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


# =========================================================================== lists


def list_watchlists(session: Session, *, include_archived: bool = False) -> list[dict[str, Any]]:
    """Every list (MANUAL first - "Follow-up" leading - then NEXT_DAY plans newest first, then SYSTEM)."""
    q = select(Watchlist)
    if not include_archived:
        q = q.where(Watchlist.archived.is_(False))
    rows = list(session.execute(q).scalars())
    counts = dict(session.execute(select(WatchlistItem.watchlist_id, func.count(WatchlistItem.id))
                                  .group_by(WatchlistItem.watchlist_id)).all())
    order = {"MANUAL": 0, "NEXT_DAY": 1, "SYSTEM": 2}

    def key(w: Watchlist) -> tuple[Any, ...]:
        fd = -(w.for_date.toordinal()) if w.for_date else 0
        return (order.get(w.kind, 3), w.name != DEFAULT_LIST_NAME, fd, w.name.lower())

    return [_list_dict(w, int(counts.get(w.id, 0))) for w in sorted(rows, key=key)]


def get_watchlist(session: Session, watchlist_id: int) -> dict[str, Any] | None:
    """One list with its items (newest first), or None if the id is unknown."""
    try:
        wl = session.get(Watchlist, int(watchlist_id))
    except (TypeError, ValueError):
        return None
    if wl is None:
        return None
    items = _items(session, wl.id)
    out = _list_dict(wl, len(items))
    out["items"] = [item_dict(i) for i in items]
    return out


def create_watchlist(session: Session, name: str, *, now: datetime, kind: str = "MANUAL",
                     description: str | None = None, use_for_trading: bool = False, for_date: date | None = None,
                     actor: str = "owner") -> dict[str, Any]:
    """Create a list. Names are trimmed, 1-80 chars and unique (case-insensitive)."""
    clean = _clean_name(name)
    kind = str(kind or "MANUAL").upper()
    if kind not in KINDS:
        raise ValueError(f"invalid list kind {kind!r} (use {', '.join(KINDS)})")
    if _name_taken(session, clean):
        raise ValueError(f"a list named {clean!r} already exists")
    wl = Watchlist(name=clean, kind=kind, description=_clean_text(description, MAX_NOTE),
                   use_for_trading=bool(use_for_trading), for_date=for_date, archived=False,
                   created_at=now, updated_at=now)
    session.add(wl)
    session.flush()
    _log(session, actor, "WATCHLIST_CREATE", wl.id,
         {"name": clean, "kind": kind, "use_for_trading": bool(use_for_trading), "for_date": _iso(for_date)}, now,
         f"list '{clean}' created ({kind.lower()}" + (", used for trading" if use_for_trading else "") + ")")
    return _list_dict(wl, 0)


def get_or_create_default(session: Session, *, now: datetime, actor: str = "owner") -> dict[str, Any]:
    """The owner's default MANUAL list "Follow-up" (created on first use)."""
    wl = session.execute(select(Watchlist).where(func.lower(Watchlist.name) == DEFAULT_LIST_NAME.lower())) \
        .scalars().first()
    if wl is not None:
        return _list_dict(wl, _count_items(session, wl.id))
    return create_watchlist(session, DEFAULT_LIST_NAME, now=now, kind="MANUAL",
                            description="Stocks I asked the system to analyse and want to follow up", actor=actor)


def update_watchlist(session: Session, watchlist_id: int, *, now: datetime, name: str | None = None,
                     description: str | None = None, use_for_trading: bool | None = None,
                     archived: bool | None = None, actor: str = "owner") -> dict[str, Any]:
    """Rename / describe / toggle trading use / archive a list (None = unchanged; description "" clears it)."""
    wl = _get_list(session, watchlist_id)
    changes: dict[str, Any] = {}
    if name is not None:
        clean = _clean_name(name)
        if clean != wl.name:
            if _name_taken(session, clean, exclude_id=wl.id):
                raise ValueError(f"a list named {clean!r} already exists")
            changes["name"] = [wl.name, clean]
            wl.name = clean
    if description is not None:
        new = _clean_text(description, MAX_NOTE)
        if new != wl.description:
            changes["description"] = [wl.description, new]
            wl.description = new
    if use_for_trading is not None and bool(use_for_trading) != bool(wl.use_for_trading):
        changes["use_for_trading"] = [bool(wl.use_for_trading), bool(use_for_trading)]
        wl.use_for_trading = bool(use_for_trading)
    if archived is not None and bool(archived) != bool(wl.archived):
        changes["archived"] = [bool(wl.archived), bool(archived)]
        wl.archived = bool(archived)
    if changes:
        wl.updated_at = now
        session.flush()
        _log(session, actor, "WATCHLIST_UPDATE", wl.id, {"name": wl.name, "changes": changes}, now,
             f"list '{wl.name}' updated: " + ", ".join(f"{k} -> {v[1]}" for k, v in changes.items()))
    return _list_dict(wl, _count_items(session, wl.id))


def delete_watchlist(session: Session, watchlist_id: int, *, now: datetime, actor: str = "owner") -> None:
    """Delete a list and all of its items."""
    wl = _get_list(session, watchlist_id)
    items = _items(session, wl.id)
    symbols = [i.symbol for i in items]
    for i in items:
        session.delete(i)
    name, wid = wl.name, wl.id
    session.delete(wl)
    session.flush()
    _log(session, actor, "WATCHLIST_DELETE", wid, {"name": name, "symbols": symbols}, now,
         f"list '{name}' deleted ({len(symbols)} stock(s))")


# =========================================================================== items


def add_item(session: Session, watchlist_id: int, symbol: str, *, now: datetime, note: str | None = None,
             source: str = "MANUAL", added_price: float | None = None, mode: str | None = None,
             plan: dict[str, Any] | None = None, decision_id: int | None = None,
             actor: str = "owner") -> dict[str, Any]:
    """Add ``symbol`` to a list, or update it if already there (note / plan / mode / decision change;
    ``added_at``, ``added_by`` and the first ``added_price`` are kept). A new ``plan`` resets the
    status to ACTIVE."""
    wl = _get_list(session, watchlist_id)
    sym = normalize_symbol(symbol)
    src = str(source or "MANUAL").upper()
    if src not in SOURCES:
        raise ValueError(f"invalid source {source!r} (use {', '.join(SOURCES)})")
    mode_v = _norm_mode(mode)
    if plan is not None and not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    price = None
    if added_price is not None:
        try:
            price = float(added_price)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid added_price {added_price!r}") from exc
        if price <= 0:
            raise ValueError("added_price must be > 0")
    row = session.execute(select(WatchlistItem).where(WatchlistItem.watchlist_id == wl.id,
                                                      WatchlistItem.symbol == sym)).scalars().first()
    if row is None:
        row = WatchlistItem(watchlist_id=wl.id, symbol=sym, mode=mode_v, source=src,
                            note=_clean_text(note, MAX_NOTE), added_at=now, added_by=str(actor or "owner")[:64],
                            added_price=price, plan_json=plan, status="ACTIVE", decision_id=decision_id)
        session.add(row)
        action, verb = "WATCHLIST_ADD", "added to"
    else:
        if note is not None:
            row.note = _clean_text(note, MAX_NOTE)
        if mode_v is not None:
            row.mode = mode_v
        if plan is not None:
            row.plan_json = plan
            row.status, row.status_note = "ACTIVE", None
        if decision_id is not None:
            row.decision_id = decision_id
        if row.added_price is None and price is not None:
            row.added_price = price
        action, verb = "WATCHLIST_ADD", "updated in"
    wl.updated_at = now
    session.flush()
    _log(session, actor, action, wl.id, {"symbol": sym, "source": src, "note": row.note, "mode": row.mode,
                                         "decision_id": row.decision_id, "updated": verb == "updated in"}, now,
         f"{sym} {verb} '{wl.name}'" + (f" - {row.note}" if row.note else ""), symbol=sym)
    return item_dict(row)


def remove_item(session: Session, watchlist_id: int, item_id_or_symbol: int | str, *, now: datetime,
                actor: str = "owner") -> None:
    """Remove one stock from a list, by item id (int) or by symbol (str)."""
    wl = _get_list(session, watchlist_id)
    row: WatchlistItem | None = None
    key = item_id_or_symbol
    if isinstance(key, str) and not key.strip().isdigit():
        sym = normalize_symbol(key)
        row = session.execute(select(WatchlistItem).where(WatchlistItem.watchlist_id == wl.id,
                                                          WatchlistItem.symbol == sym)).scalars().first()
    else:
        if isinstance(key, str):   # digits: a symbol first, then an item id
            row = session.execute(select(WatchlistItem).where(WatchlistItem.watchlist_id == wl.id,
                                                              WatchlistItem.symbol == key.strip())).scalars().first()
        if row is None:
            try:
                cand = session.get(WatchlistItem, int(key))
            except (TypeError, ValueError):
                cand = None
            row = cand if cand is not None and cand.watchlist_id == wl.id else None
    if row is None:
        raise ValueError(f"stock {item_id_or_symbol!r} not found in list '{wl.name}'")
    sym, item_id = row.symbol, row.id
    session.delete(row)
    wl.updated_at = now
    session.flush()
    _log(session, actor, "WATCHLIST_REMOVE", wl.id, {"symbol": sym, "item_id": item_id}, now,
         f"{sym} removed from '{wl.name}'", symbol=sym)


def set_item_status(session: Session, item_id: int, status: str, note: str | None, *, now: datetime,
                    actor: str = "system") -> dict[str, Any]:
    """Set an item's status (ACTIVE | TRIGGERED | INVALIDATED) with a plain-language note."""
    st = str(status or "").upper()
    if st not in STATUSES:
        raise ValueError(f"invalid status {status!r} (use {', '.join(STATUSES)})")
    row = session.get(WatchlistItem, int(item_id))
    if row is None:
        raise ValueError(f"watchlist item #{item_id} not found")
    old = row.status
    row.status, row.status_note = st, _clean_text(note, MAX_NOTE)
    session.flush()
    audit(session, actor, "WATCHLIST_STATUS", f"watchlist:{row.watchlist_id}",
          {"symbol": row.symbol, "item_id": row.id, "from": old, "to": st, "note": row.status_note},
          clock=_clock(now))
    return item_dict(row)


# =========================================================================== engine-facing


def _plan_lists(session: Session, for_date: date) -> list[Watchlist]:
    return list(session.execute(select(Watchlist).where(Watchlist.kind == "NEXT_DAY", Watchlist.for_date == for_date,
                                                        Watchlist.archived.is_(False))
                                .order_by(Watchlist.id.desc())).scalars())


def plan_items(session: Session, *, for_date: date, statuses: Iterable[str] | None = None) -> list[WatchlistItem]:
    """Items of the NEXT_DAY plan(s) for ``for_date`` (highest plan score first)."""
    ids = [w.id for w in _plan_lists(session, for_date)]
    if not ids:
        return []
    q = select(WatchlistItem).where(WatchlistItem.watchlist_id.in_(ids))
    if statuses is not None:
        q = q.where(WatchlistItem.status.in_(list(statuses)))
    rows = list(session.execute(q).scalars())
    rows.sort(key=lambda i: (-_score(i), i.id))
    return rows


def trading_symbols(session: Session, *, today: date) -> list[str]:
    """Symbols INTRADAY cycles analyse first: today's NEXT_DAY plan (best score first), then the
    stocks of every non-archived list with ``use_for_trading=True`` (oldest first). Only live items
    (ACTIVE, or TRIGGERED at the opening check - a triggered plan is exactly what should be analysed);
    INVALIDATED ones are skipped. De-duplicated, plan first. NEXT_DAY plans of other dates never count.
    """
    out: list[str] = [i.symbol for i in plan_items(session, for_date=today, statuses=LIVE_STATUSES)]
    q = (select(WatchlistItem.symbol)
         .join(Watchlist, Watchlist.id == WatchlistItem.watchlist_id)
         .where(Watchlist.use_for_trading.is_(True), Watchlist.archived.is_(False), Watchlist.kind != "NEXT_DAY",
                WatchlistItem.status.in_(LIVE_STATUSES))
         .order_by(WatchlistItem.added_at, WatchlistItem.id))
    out += list(session.execute(q).scalars())
    return list(dict.fromkeys(out))


def followed_symbols(session: Session) -> set[str]:
    """Every symbol in any non-archived list (used to decide which analyses to record)."""
    q = (select(WatchlistItem.symbol).join(Watchlist, Watchlist.id == WatchlistItem.watchlist_id)
         .where(Watchlist.archived.is_(False)))
    return set(session.execute(q).scalars())


def record_analysis(session: Session, symbol: str, *, decision_id: int | None, summary: str | None,
                    now: datetime) -> int:
    """Store the latest analysis on every item of ``symbol`` (all lists). Returns rows updated."""
    try:
        sym = normalize_symbol(symbol)
    except ValueError:
        return 0
    rows = list(session.execute(select(WatchlistItem).where(WatchlistItem.symbol == sym)).scalars())
    for r in rows:
        if decision_id is not None:
            r.last_analysis_id = int(decision_id)
        r.last_analysis_at = now
        r.last_summary = _clean_text(summary, MAX_SUMMARY)
    if rows:
        session.flush()
    return len(rows)


def latest_plan(session: Session, *, on_or_after: date) -> dict[str, Any] | None:
    """Newest non-archived NEXT_DAY list with ``for_date >= on_or_after``; items sorted by plan score."""
    wl = session.execute(select(Watchlist).where(Watchlist.kind == "NEXT_DAY", Watchlist.for_date >= on_or_after,
                                                 Watchlist.archived.is_(False))
                         .order_by(Watchlist.for_date.desc(), Watchlist.id.desc())).scalars().first()
    if wl is None:
        return None
    items = _items(session, wl.id)
    items.sort(key=lambda i: (-_score(i), i.id))
    out = _list_dict(wl, len(items))
    out["items"] = [item_dict(i) for i in items]
    return out


def save_plan(session: Session, for_date: date, candidates: list[dict[str, Any]], *, now: datetime,
              actor: str = "system", description: str | None = None) -> dict[str, Any]:
    """Create or REPLACE the NEXT_DAY list for ``for_date`` ("Plan 26 Nov 2026", used for trading)
    with one EOD_PLAN item per candidate (``plan_json`` = the candidate dict). Older plans
    (``for_date`` before today's date) are archived. Returns ``get_watchlist`` of the plan."""
    name = f"Plan {for_date:%d %b %Y}"
    existing = _plan_lists(session, for_date)
    wl = existing[0] if existing else session.execute(
        select(Watchlist).where(Watchlist.kind == "NEXT_DAY", func.lower(Watchlist.name) == name.lower())
    ).scalars().first()
    desc = description or (f"End-of-day paper plan for {for_date:%a %d %b %Y}: intraday candidates ranked after "
                           "the close and re-checked at the opening. Not advice; every trade still needs the "
                           "risk manager.")
    replaced = 0
    if wl is None:
        if _name_taken(session, name):
            name = f"{name} (system)"[:MAX_NAME]
        wl = Watchlist(name=name, kind="NEXT_DAY", description=desc, use_for_trading=True, for_date=for_date,
                       archived=False, created_at=now, updated_at=now, meta_json={"generated_at": _iso(now)})
        session.add(wl)
        session.flush()
    else:
        old = _items(session, wl.id)
        replaced = len(old)
        for i in old:
            session.delete(i)
        wl.for_date, wl.use_for_trading, wl.archived, wl.updated_at = for_date, True, False, now
        wl.description = desc
        wl.meta_json = {**(wl.meta_json or {}), "generated_at": _iso(now)}
        session.flush()
    for c in candidates:
        sym = normalize_symbol(c.get("symbol"))
        session.add(WatchlistItem(watchlist_id=wl.id, symbol=sym, mode=Mode.INTRADAY.value, source="EOD_PLAN",
                                  note=f"{str(c.get('setup') or '').lower()} {str(c.get('direction') or '').lower()}"
                                       f" - trigger {c.get('trigger')}, stop {c.get('stop')}, "
                                       f"target {c.get('target')}".strip(),
                                  added_at=now, added_by=actor, added_price=c.get("close"), plan_json=dict(c),
                                  status="ACTIVE"))
    today = to_ist(now).date()
    for old_plan in session.execute(select(Watchlist).where(Watchlist.kind == "NEXT_DAY", Watchlist.for_date < today,
                                                            Watchlist.archived.is_(False))).scalars():
        old_plan.archived, old_plan.updated_at = True, now
    session.flush()
    syms = [str(c.get("symbol")) for c in candidates]
    _log(session, actor, "WATCHLIST_PLAN", wl.id, {"name": wl.name, "for_date": _iso(for_date), "symbols": syms,
                                                   "replaced_items": replaced}, now,
         f"next-day plan '{wl.name}' saved: {len(syms)} stock(s)" + (f" ({', '.join(syms)})" if syms else "")
         + (f", replacing {replaced}" if replaced else ""))
    out = get_watchlist(session, wl.id)
    assert out is not None
    return out


__all__ = [
    "DEFAULT_LIST_NAME", "KINDS", "LIVE_STATUSES", "SOURCES", "STATUSES", "SYMBOL_RE",
    "add_item", "create_watchlist", "delete_watchlist", "followed_symbols", "get_or_create_default",
    "get_watchlist", "item_dict", "latest_plan", "list_watchlists", "normalize_symbol", "normalize_symbols",
    "plan_items", "record_analysis", "remove_item", "save_plan", "set_item_status", "trading_symbols",
    "update_watchlist",
]
