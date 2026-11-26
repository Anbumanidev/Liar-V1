"""Auth, overview, engine controls, autonomy, approvals and command status."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from core.enums import Autonomy, CommandType, Mode
from storage.models import ApprovalItem, ControlCommand, Position
from webui import queries as q
from webui.deps import enqueue, get_ctx, get_user, validation_message
from webui.security import COOKIE_NAME, cookie_header
from webui.util import clean_symbol, day_bounds, iso, r2

router = APIRouter(prefix="/api")

AUTONOMY_INFO: dict[str, tuple[str, str]] = {
    "PER_MODE": ("Per mode", "Each mode uses its own execution policy (set in the Streamlit console > Modes)."),
    "HUMAN_IN_THE_LOOP": ("Human in the loop", "Every trade waits for your Approve / Reject. Nothing executes "
                                               "without you."),
    "HYBRID": ("Hybrid", "Top setups (grade and confidence at or above your thresholds) execute automatically; "
                         "everything else waits for your approval."),
    "FULL_AUTO": ("Fully automatic", "Every trade the risk manager approves executes immediately - no human. "
                                     "The risk manager and capital protection still veto everything."),
    "SIGNALS_ONLY": ("Signals only", "Trades are logged and emailed, never executed."),
}
AUTO_LEVELS = ("FULL_AUTO", "HYBRID")


# =========================================================================== auth


class LoginBody(BaseModel):
    username: str = Field(default="", max_length=64)
    password: str = Field(default="", max_length=256)
    keep_live: bool = False   # trading-monitor screen: no 30-minute idle logout (max 16 h)


@router.post("/login")
def login(body: LoginBody, request: Request) -> JSONResponse:
    from dashboard.auth import REASON_EMPTY, REASON_LOCKED, authenticate, load_auth_config

    ctx = get_ctx(request)
    config = load_auth_config(ctx.auth_file)
    if config is None:
        return JSONResponse(status_code=503, content={
            "detail": "Dashboard credentials are not set up. Run: .venv\\Scripts\\python.exe scripts/set_password.py",
            "setup_required": True})
    now = ctx.clock.now()
    client = " | ".join(x for x in ((request.client.host if request.client else None),
                                    (request.headers.get("user-agent") or "")[:120], "webui") if x)
    try:
        with ctx.session() as s:
            result = authenticate(s, config, body.username, body.password, now, client=client, clock=ctx.clock)
            s.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[webui] login failed with a DB error: {exc}")
        return JSONResponse(status_code=500, content={"detail": "Login failed because of a database error."})
    if result.ok:
        sm = request.app.state.sessions
        token, payload = sm.issue(result.username, now.timestamp(), keep_live=body.keep_live)
        if body.keep_live:
            logger.info(f"[webui] {result.username} signed in with 'Keep this screen live' "
                        f"(idle limit {sm.idle_limit(payload)})")
        resp = JSONResponse(content={"username": result.username, "csrf": payload["csrf"],
                                     "idle_timeout_s": int(sm.idle_limit(payload).total_seconds()),
                                     "keep_live": bool(body.keep_live)})
        resp.headers.append("set-cookie", cookie_header(token, secure=ctx.secure_cookies).decode("latin-1"))
        return resp
    if result.reason == REASON_EMPTY:
        return JSONResponse(status_code=400, content={"detail": "Enter username and password."})
    if result.reason == REASON_LOCKED or result.locked_until is not None:
        until = result.locked_until.strftime("%H:%M:%S IST") if result.locked_until else "later"
        return JSONResponse(status_code=429, content={
            "detail": f"Too many failed logins. Locked for 15 minutes (until {until}).",
            "locked_until": iso(result.locked_until)})
    return JSONResponse(status_code=401, content={
        "detail": f"Invalid credentials. {result.remaining_attempts} attempt(s) left before a 15-minute lockout.",
        "remaining_attempts": result.remaining_attempts, "login_failed": True})


@router.post("/logout")
def logout(request: Request) -> JSONResponse:
    from core.logging import log_event

    ctx = get_ctx(request)
    user = get_user(request)
    payload = getattr(request.state, "session", None) or {}
    if payload.get("sid"):
        request.app.state.sessions.revoke(payload["sid"])
    request.state.logout = True
    # JSONL audit only: the web process writes nothing to the DB except login attempts, commands, settings.
    log_event("audit", {"actor": user, "action": "LOGOUT", "target": f"user:{user}", "details": {"via": "webui"}},
              clock=ctx.clock)
    resp = JSONResponse(content={"ok": True})
    resp.headers.append("set-cookie", cookie_header("", secure=ctx.secure_cookies, max_age=0).decode("latin-1"))
    return resp


@router.get("/me")
def me(request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    payload = request.state.session
    sm = request.app.state.sessions
    return {"username": payload["u"], "csrf": payload["csrf"],
            "idle_timeout_s": int(sm.idle_limit(payload).total_seconds()), "keep_live": bool(payload.get("kl")),
            "server_time": iso(ctx.clock.now()), "streamlit_url": ctx.streamlit_url, "paper_only": True,
            "allow_full_auto": bool(ctx.caps.allow_full_auto), "cookie": COOKIE_NAME}


# =========================================================================== overview


def market_info(ctx: Any, now: Any) -> dict[str, Any]:
    try:
        cal = ctx.get_calendar()
        today = now.date()
        is_open = bool(cal.is_market_open(now))
        if cal.is_trading_day(today) and now < cal.market_open_dt(today):
            next_open = cal.market_open_dt(today)
        else:
            next_open = cal.market_open_dt(cal.next_trading_day(today))
        close = cal.market_close_dt(today) if cal.is_trading_day(today) else None
        return {"open": is_open, "trading_day": bool(cal.is_trading_day(today)),
                "holiday": cal.holiday_name(today), "next_open": iso(next_open) if not is_open else None,
                "closes_at": iso(close) if is_open and close else None}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[webui] calendar unavailable: {exc}")
        return {"open": None, "trading_day": None, "holiday": None, "next_open": None, "closes_at": None}


def autonomy_view(ctx: Any, settings: Any) -> dict[str, Any]:
    modes = settings.modes
    value = modes.autonomy.value
    per_mode = []
    for m in Mode:
        cfg = settings.mode(m)
        eff = "HYBRID" if value == "HYBRID" else modes.effective_policy(m).value
        per_mode.append({"mode": m.value, "enabled": bool(cfg.enabled), "policy": cfg.execution_policy.value,
                         "effective": eff, "allocation_pct": r2(cfg.allocation_pct)})
    return {"value": value, "label": AUTONOMY_INFO[value][0], "description": AUTONOMY_INFO[value][1],
            "allow_full_auto": bool(ctx.caps.allow_full_auto),
            "hybrid_min_grade": modes.hybrid_auto_min_grade.value,
            "hybrid_min_confidence": modes.hybrid_auto_min_confidence,
            "options": [{"value": k, "label": v[0], "description": v[1],
                         "locked": (k in AUTO_LEVELS and not ctx.caps.allow_full_auto)}
                        for k, v in AUTONOMY_INFO.items()],
            "per_mode": per_mode}


def overall_view(ctx: Any, unrealized_by_batch: dict[int, float]) -> dict[str, Any]:
    from webui.scope import funding_or_fallback

    with ctx.session() as s:
        summary, source = funding_or_fallback(s, unrealized_by_batch)
    keys = ("total_deposited", "total_withdrawn", "net_invested", "current_capital", "overall_pnl", "overall_pnl_pct")
    return {"source": source} | {k: r2(summary.get(k)) for k in keys}


@router.get("/overview")
def overview(request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    now = ctx.clock.now()
    settings = ctx.settings()
    with ctx.session() as s:
        engine = q.engine_view(s, now, settings.engine.heartbeat_stale_seconds)
        positions = q.open_positions(s)
        pending = q.pending_count(s)
    quotes = ctx.quotes.get_many([p.symbol for p in positions], wait=ctx.quote_wait) if positions else {}
    rows = [q.position_row(p, quotes.get(p.symbol)) for p in positions]
    unreal_by_batch: dict[int, float] = {}
    for r in rows:
        unreal_by_batch[r["batch_id"]] = unreal_by_batch.get(r["batch_id"], 0.0) + (r["unrealized"] or 0.0)
    lo, hi = day_bounds(now.date())
    with ctx.session() as s:
        batch = q.current_batch(s)
        capital = q.capital_view(s, batch.id, unreal_by_batch.get(batch.id, 0.0)) if batch else None
        today = q.today_realized(s, lo, hi, None)
        stats = q.batch_trade_stats(s, batch.id) if batch else None
    unreal_all = round(sum(unreal_by_batch.values()), 2)
    today |= {"unrealized": r2(unreal_all), "total": r2((today["realized"] or 0.0) + unreal_all),
              "open_positions": len(rows)}
    return {
        "server_time": iso(now), "paper_only": True,
        "engine": engine,
        "market": market_info(ctx, now),
        "batch": q.batch_brief(batch),
        "capital": capital,
        "today": today,
        "batch_stats": stats,
        "overall": overall_view(ctx, unreal_by_batch),
        "autonomy": autonomy_view(ctx, settings),
        "approvals_pending": pending,
        "positions": rows,
        "links": {"streamlit": ctx.streamlit_url},
    }


# =========================================================================== engine controls


class ControlBody(BaseModel):
    action: Literal["pause", "resume", "kill_switch", "analyze_now", "scan_now", "close_position"]
    confirm_text: str | None = Field(default=None, max_length=120)
    symbol: str | None = Field(default=None, max_length=32)
    mode: str | None = Field(default=None, max_length=16)
    route: bool = False
    position_id: int | None = None
    qty: int | None = Field(default=None, ge=1)
    reason: str | None = Field(default=None, max_length=300)
    exit_cnc: bool = False
    use_llm: bool = True


@router.post("/control")
def control(body: ControlBody, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    if body.action == "pause":
        return enqueue(ctx, user, CommandType.ENGINE_PAUSE, {})
    if body.action == "resume":
        return enqueue(ctx, user, CommandType.ENGINE_RESUME, {})
    if body.action == "kill_switch":
        if (body.confirm_text or "").strip() != "CONFIRM":
            raise HTTPException(status_code=400, detail="Type CONFIRM to use the kill switch.")
        return enqueue(ctx, user, CommandType.KILL_SWITCH,
                       {"exit_cnc": bool(body.exit_cnc), "reason": body.reason or f"Owner kill switch ({user}, web)"})
    if body.action == "analyze_now":
        symbol = clean_symbol(body.symbol)
        mode = (body.mode or "INTRADAY").upper()
        if mode not in {m.value for m in Mode}:
            raise HTTPException(status_code=422, detail=f"invalid mode {body.mode!r}")
        return enqueue(ctx, user, CommandType.ANALYZE_NOW, {"symbol": symbol, "mode": mode, "route": bool(body.route)})
    if body.action == "scan_now":
        scan = getattr(CommandType, "SCAN_NOW", None)
        if scan is None:  # engine side not deployed yet
            raise HTTPException(status_code=501, detail="Scan now is not available yet (engine update pending).")
        mode = (body.mode or "INTRADAY").upper()
        if mode not in {m.value for m in Mode}:
            raise HTTPException(status_code=422, detail=f"invalid mode {body.mode!r}")
        return enqueue(ctx, user, scan, {"mode": mode, "use_llm": bool(body.use_llm)})
    # close_position
    if body.position_id is None:
        raise HTTPException(status_code=422, detail="position_id is required")
    with ctx.session() as s:
        pos = s.get(Position, body.position_id)
        if pos is None:
            raise HTTPException(status_code=404, detail=f"position {body.position_id} not found")
        if pos.status != "OPEN":
            raise HTTPException(status_code=409, detail=f"position {body.position_id} is already {pos.status}")
    payload: dict[str, Any] = {"position_id": body.position_id, "reason": body.reason or "closed from web dashboard"}
    if body.qty:
        payload["qty"] = body.qty
    return enqueue(ctx, user, CommandType.CLOSE_POSITION, payload)


# =========================================================================== autonomy


class AutonomyBody(BaseModel):
    autonomy: str
    confirm_text: str | None = Field(default=None, max_length=120)
    hybrid_min_grade: str | None = None
    hybrid_min_confidence: int | None = Field(default=None, ge=0, le=100)
    note: str | None = Field(default=None, max_length=300)


@router.post("/settings/autonomy")
def set_autonomy(body: AutonomyBody, request: Request) -> dict[str, Any]:
    """Global autonomy switch: new settings version + RELOAD_SETTINGS (dashboard/components pattern)."""
    from config.store import apply_changes
    from control.commands import enqueue_command
    from core.errors import HardCapViolationError

    ctx = get_ctx(request)
    user = get_user(request)
    value = (body.autonomy or "").strip().upper()
    if value not in {a.value for a in Autonomy}:
        raise HTTPException(status_code=422, detail=f"invalid autonomy {body.autonomy!r}")
    if value in AUTO_LEVELS and not ctx.caps.allow_full_auto:
        raise HTTPException(status_code=403, detail="Locked by config/hard_caps.yaml (allow_full_auto: false). "
                                                    "Change it on disk to enable automatic execution.")
    version_id, settings = ctx.settings_with_version()
    current = settings.modes.autonomy.value
    if value in AUTO_LEVELS and value != current and (body.confirm_text or "").strip() != "CONFIRM":
        raise HTTPException(status_code=400, detail="Type CONFIRM to switch to automatic execution.")
    proposed: dict[str, Any] = {"modes.autonomy": value}
    if value == "HYBRID":
        if body.hybrid_min_grade:
            if body.hybrid_min_grade not in ("A+", "A", "B"):
                raise HTTPException(status_code=422, detail="hybrid_min_grade must be A+, A or B")
            proposed["modes.hybrid_auto_min_grade"] = body.hybrid_min_grade
        if body.hybrid_min_confidence is not None:
            proposed["modes.hybrid_auto_min_confidence"] = int(body.hybrid_min_confidence)
    current_vals = {"modes.autonomy": current,
                    "modes.hybrid_auto_min_grade": settings.modes.hybrid_auto_min_grade.value,
                    "modes.hybrid_auto_min_confidence": settings.modes.hybrid_auto_min_confidence}
    changes = {k: v for k, v in proposed.items() if current_vals.get(k) != v}
    if not changes:
        return {"changed": False, "version_id": version_id, "command_id": None, "autonomy": current}
    with ctx.session() as s:
        batch = q.current_batch(s)
        bid = batch.id if batch else None
        try:
            row = apply_changes(s, changes, ctx.caps, author=user, note=body.note or f"autonomy -> {value} (web)",
                                batch_id=bid, clock=ctx.clock)
        except HardCapViolationError as exc:
            s.rollback()
            raise HTTPException(status_code=400, detail=f"Hard cap violation: {exc}") from exc
        except ValidationError as exc:
            s.rollback()
            raise HTTPException(status_code=400, detail=f"Invalid settings: {validation_message(exc)}") from exc
        except (KeyError, ValueError) as exc:
            s.rollback()
            raise HTTPException(status_code=400, detail=f"Invalid settings: {exc}") from exc
        if row.id == version_id:
            s.rollback()
            return {"changed": False, "version_id": version_id, "command_id": None, "autonomy": current}
        cmd = enqueue_command(s, CommandType.RELOAD_SETTINGS, {}, actor=user, batch_id=bid, clock=ctx.clock)
        s.commit()
        new_id, cmd_id = row.id, cmd.id
    ctx.invalidate_settings()
    logger.info(f"[webui] {user} set autonomy {current} -> {value} (settings v{new_id}, command #{cmd_id})")
    return {"changed": True, "version_id": new_id, "command_id": cmd_id, "autonomy": value}


# =========================================================================== approvals


class RejectBody(BaseModel):
    reason: str | None = Field(default=None, max_length=300)


@router.get("/approvals")
def approvals(request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    now = ctx.clock.now()
    with ctx.session() as s:
        items = q.pending_approvals(s, now)
    return {"items": items, "count": len(items), "server_time": iso(now)}


def _pending_item(ctx: Any, approval_id: int) -> None:
    with ctx.session() as s:
        item = s.get(ApprovalItem, approval_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"approval {approval_id} not found")
        if item.status != "PENDING":
            raise HTTPException(status_code=409, detail=f"approval {approval_id} is already {item.status}")


@router.post("/approvals/{approval_id}/approve")
def approve(approval_id: int, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    _pending_item(ctx, approval_id)
    return enqueue(ctx, user, CommandType.APPROVE, {"approval_id": approval_id})


@router.post("/approvals/{approval_id}/reject")
def reject(approval_id: int, request: Request, body: RejectBody | None = None) -> dict[str, Any]:
    ctx = get_ctx(request)
    user = get_user(request)
    _pending_item(ctx, approval_id)
    reason = (body.reason if body else None) or "rejected from web dashboard"
    return enqueue(ctx, user, CommandType.REJECT, {"approval_id": approval_id, "reason": reason})


# =========================================================================== commands


@router.get("/commands/{command_id}")
def command(command_id: int, request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    with ctx.session() as s:
        row = s.get(ControlCommand, command_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"command {command_id} not found")
        return q.command_view(row)


@router.get("/commands")
def commands(request: Request, limit: int = Query(default=15, ge=1, le=100)) -> dict[str, Any]:
    ctx = get_ctx(request)
    with ctx.session() as s:
        rows = list(s.execute(select(ControlCommand).order_by(ControlCommand.id.desc()).limit(limit)).scalars())
        return {"items": [q.command_view(r) for r in rows]}

