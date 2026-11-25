"""Market data (candles, quotes, watch list), health & logs, and the SSE stream."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import func, select

from core.clock import IST
from storage.models import ErrorLog
from webui import queries as q
from webui.candles import (
    HISTORY_DAYS,
    HISTORY_NOTE,
    MAX_BARS,
    RANGES,
    earliest_available,
    frame_payload,
    load_window,
    tf_spec,
    trade_overlays,
)
from webui.deps import get_ctx
from webui.logs import LOG_FILES, read_log
from webui.stream import event_stream
from webui.util import (
    IST_OFFSET_S,
    NOT_AVAILABLE,
    NotAvailable,
    clean_symbol,
    day_bounds,
    is_symbol,
    iso,
    optional_attr,
    parse_day,
    parse_dt,
    text,
)

router = APIRouter(prefix="/api")


# =========================================================================== candles / quotes


@router.get("/candles")
def candles(request: Request, symbol: str = Query(...), tf: str = Query(default="5m"),
            bars: int = Query(default=300, ge=10, le=MAX_BARS), date_from: str | None = Query(None, alias="from"),
            date_to: str | None = Query(None, alias="to"), position_id: int | None = Query(default=None),
            range_: str | None = Query(default=None, alias="range", max_length=4),
            before: int | None = Query(default=None, ge=946_684_800, le=4_102_444_800)) -> dict[str, Any]:
    """Chart window. ``before`` (chart time) = the ``bars`` bars ending before it (scroll-back);
    ``range`` = 1D|5D|1M|3M|6M|1Y|5Y|MAX (clamped to the timeframe's history); else ``from``/``to`` or the
    latest ``bars``. ``has_more`` / ``earliest_available`` tell the chart when to stop loading older bars."""
    ctx = get_ctx(request)
    sym = clean_symbol(symbol)
    try:
        spec = tf_spec(tf)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rng = range_.strip().upper() if range_ else None
    if rng is not None and rng not in RANGES:
        raise HTTPException(status_code=422, detail=f"range must be one of {', '.join(RANGES)}")
    start, end = parse_dt(date_from), parse_dt(date_to)
    before_dt = datetime.fromtimestamp(before - IST_OFFSET_S, tz=IST) if before is not None else None
    data = ctx.get_data()
    if data is None:
        raise HTTPException(status_code=503, detail="market data service unavailable")
    now = ctx.clock.now()
    cache = request.app.state.candle_cache
    key = (sym, spec.key, bars, iso(start), iso(end), rng, before)
    frame = cache.get(key)
    if frame is None:
        try:
            calendar = ctx.get_calendar()
        except Exception as exc:  # noqa: BLE001 - weekday fallback inside load_window
            logger.warning(f"[webui] calendar unavailable for candles: {exc}")
            calendar = None
        try:
            df, info = load_window(data, sym, spec, now, calendar, bars=bars, start=start, end=end, range_key=rng,
                                   before=before_dt)
        except Exception as exc:  # noqa: BLE001 - data errors must not 500 the chart
            logger.warning(f"[webui] candles {sym} {spec.key} failed: {exc}")
            df, info = None, {"has_more": False, "earliest_available": iso(earliest_available(spec, now)),
                              "history_days": HISTORY_DAYS[spec.key], "history_note": HISTORY_NOTE[spec.key],
                              "range": rng, "clamped": False, "start": None, "before": None, "error": True}
        frame = {**frame_payload(df, spec), **info}
        if frame["candles"]:
            cache.put(key, frame)
    times = [c["time"] for c in frame["candles"]]
    with ctx.session() as s:
        markers, lines = trade_overlays(s, sym, times, spec, position_id=position_id, hi_time=before)
    quote = ctx.quotes.get_many([sym], wait=1.0).get(sym)
    return {"symbol": sym, "tf": spec.key, "source_tf": spec.source, "intraday": spec.intraday,
            "time_basis": "ist_epoch", **frame, "bars": len(frame["candles"]), "markers": markers, "lines": lines,
            "quote": q.quote_dict(sym, quote), "as_of": iso(now)}


@router.get("/quotes")
def quotes(request: Request, symbols: str = Query(..., max_length=800)) -> dict[str, Any]:
    ctx = get_ctx(request)
    syms = []
    for raw in symbols.split(","):
        raw = raw.strip().upper()
        if not raw:
            continue
        if not is_symbol(raw):
            raise HTTPException(status_code=422, detail=f"invalid symbol {raw!r}")
        if raw not in syms:
            syms.append(raw)
    if len(syms) > 40:
        raise HTTPException(status_code=422, detail="at most 40 symbols")
    got = ctx.quotes.get_many(syms, wait=ctx.quote_wait)
    return {"quotes": {s: q.quote_dict(s, got.get(s)) for s in syms}, "as_of": iso(ctx.clock.now())}


@router.get("/watchlist")
def watchlist(request: Request, limit: int = Query(default=20, ge=1, le=40)) -> dict[str, Any]:
    ctx = get_ctx(request)
    with ctx.session() as s:
        recent = q.recent_symbols(s, limit)
        open_syms = sorted({p.symbol for p in q.open_positions(s)})
    syms = list(dict.fromkeys(open_syms + [r["symbol"] for r in recent]))[:40]
    got = ctx.quotes.get_many(syms, wait=1.5)
    by_sym = {r["symbol"]: r for r in recent}
    rows = []
    for sym in syms:
        row = by_sym.get(sym, {"symbol": sym})
        rows.append({**row, "open_position": sym in open_syms, "quote": q.quote_dict(sym, got.get(sym))})
    return {"rows": rows, "as_of": iso(ctx.clock.now())}


# =========================================================================== health & logs


@router.get("/health")
def health(request: Request, date: str | None = Query(default=None)) -> dict[str, Any]:
    ctx = get_ctx(request)
    now = ctx.clock.now()
    day = parse_day(date, now.date())
    lo, hi = day_bounds(day)
    settings = ctx.settings()
    diagnostics = None
    diag_error = None
    with ctx.session() as s:
        engine = q.engine_view(s, now, settings.engine.heartbeat_stale_seconds)
        errors = q.errors_grouped(s, hi, limit=50)
        errors_day = int(s.execute(select(func.count(ErrorLog.id)).where(ErrorLog.ts >= lo, ErrorLog.ts < hi))
                         .scalar_one() or 0)
        try:
            diagnostics = optional_attr("analytics.diagnostics", "diagnose")(s, day, ctx.log_dir)
        except NotAvailable:
            diag_error = {"status": 501, "detail": NOT_AVAILABLE}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[webui] diagnose({day}) failed: {exc}")
            s.rollback()
            diag_error = {"status": 500, "detail": f"diagnostics failed: {text(str(exc), 200)}"}
    data = ctx.data
    data_status = None
    if data is not None:
        try:
            data_status = {"healthy": bool(data.healthy()) if hasattr(data, "healthy") else None,
                           "last_error": text(getattr(data, "last_error", None), 300)}
        except Exception:  # noqa: BLE001
            data_status = None
    return {"day": day.isoformat(), "server_time": iso(now), "engine": engine, "diagnostics": diagnostics,
            "diagnostics_available": diagnostics is not None, "diagnostics_error": diag_error,
            "errors": errors, "errors_today": errors_day, "web_data": data_status}


@router.get("/logs")
def logs(request: Request, file: str = Query(default="engine"), date: str | None = Query(default=None),
         lines: int = Query(default=200, ge=1, le=3000), q_: str | None = Query(default=None, alias="q", max_length=120),
         level: str | None = Query(default=None, max_length=12), stage: str | None = Query(default=None, max_length=120),
         symbol: str | None = Query(default=None, max_length=32)) -> dict[str, Any]:
    ctx = get_ctx(request)
    if file not in LOG_FILES:
        raise HTTPException(status_code=422, detail=f"file must be one of {', '.join(LOG_FILES)}")
    day = parse_day(date, ctx.clock.now().date())
    out = read_log(ctx.log_dir, day, file, lines, query=q_, level=level, stage=stage, symbol=symbol)
    return out | {"files": list(LOG_FILES)}


@router.get("/stream")
async def stream(request: Request, decision: int | None = Query(default=None, ge=0),
                 order: int | None = Query(default=None, ge=0), trade: int | None = Query(default=None, ge=0)
                 ) -> StreamingResponse:
    ctx = get_ctx(request)
    payload = getattr(request.state, "session", None)
    deadline = request.app.state.sessions.idle_deadline(payload) if payload else None
    gen = event_stream(request, ctx, deadline, {"decision": decision, "order": order, "trade": trade})
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Connection": "keep-alive"})
