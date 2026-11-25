"""FastAPI app factory + runner for the owner-only web dashboard. PAPER TRADING ONLY.

Run::

    .venv\\Scripts\\python.exe -m webui                 # http://127.0.0.1:8600
    .venv\\Scripts\\python.exe -m webui --port 8700

or from code: ``webui.server.run(host="127.0.0.1", port=8600)``; tests use ``create_app(ctx)``.

The web process never places orders. Writes are limited to: login attempts/audit (via
``dashboard.auth``), control commands (``control.commands.enqueue_command``), settings
versions (``config.store.apply_changes`` + ``RELOAD_SETTINGS``) and the owner's follow-up lists
(``data.watchlists``, audited - owner data, not trading state).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from starlette.middleware.trustedhost import TrustedHostMiddleware

from webui.candles import CandleCache
from webui.context import WebContext, build_context
from webui.deps import SafeJSONResponse
from webui.security import SecurityMiddleware, SessionManager, load_or_create_secret

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_PORT = 8600


def _absolute_lifetime(ctx: WebContext) -> timedelta:
    """Absolute session lifetime = auth file ``cookie.expiry_days`` (default 1 day)."""
    try:
        from dashboard.auth import load_auth_config

        cfg = load_auth_config(ctx.auth_file)
        days = float(cfg.cookie_expiry_days) if cfg is not None else 1.0
    except Exception:  # noqa: BLE001
        days = 1.0
    return timedelta(days=max(0.04, min(days, 30.0)))


def create_app(ctx: WebContext | None = None) -> FastAPI:
    """Build the FastAPI app. ``ctx`` is injectable (tests); default = :func:`build_context`."""
    from dashboard.auth import IDLE_TIMEOUT
    from webui import routes_core, routes_scope, routes_system, routes_trading, routes_watchlists

    ctx = ctx or build_context()
    secret = load_or_create_secret(ctx.secret_key_path)
    sessions = SessionManager(secret, idle=IDLE_TIMEOUT, absolute=_absolute_lifetime(ctx))

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        ctx.close()

    app = FastAPI(title="Liar-V1 (PAPER) web dashboard", docs_url=None, redoc_url=None, openapi_url=None,
                  default_response_class=SafeJSONResponse, lifespan=lifespan)
    app.state.ctx = ctx
    app.state.sessions = sessions
    app.state.candle_cache = CandleCache(ttl=ctx.candle_ttl)

    app.include_router(routes_core.router)
    app.include_router(routes_scope.router)
    app.include_router(routes_trading.router)
    app.include_router(routes_system.router)
    app.include_router(routes_watchlists.router)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"[webui] unhandled error on {request.method} {request.url.path}: {exc}")
        return JSONResponse(status_code=500, content={"detail": "internal error (see logs)"})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        path = STATIC_DIR / "favicon.svg"
        if path.exists():
            return FileResponse(path, media_type="image/svg+xml")
        return Response(status_code=204)

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
    def api_not_found(rest: str) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"unknown API route /api/{rest}"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    app.add_middleware(SecurityMiddleware, sessions=sessions, clock=ctx.clock, secure_cookies=ctx.secure_cookies,
                       setup_required=lambda: not Path(ctx.auth_file).exists())
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ctx.allowed_hosts), www_redirect=False)
    return app


def run(host: str = "127.0.0.1", port: int = DEFAULT_PORT, *, allow_remote: bool = False,
        log_level: str = "info") -> None:
    """Serve the dashboard with uvicorn (blocking). Binds to localhost unless ``allow_remote``."""
    import uvicorn

    from config.secrets import load_secrets
    from core.logging import set_log_dir, setup_logging

    if host not in LOOPBACK and not allow_remote:
        raise SystemExit(f"Refusing to bind {host!r}: the dashboard is owner-only. Use 127.0.0.1 (default) or pass "
                         f"allow_remote=True / --allow-remote if you really mean it.")
    secrets = load_secrets()
    log_dir = Path(secrets.log_dir)
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parent.parent / log_dir
    setup_logging(process_name="web", log_dir=log_dir, console=True)
    set_log_dir(log_dir)
    ctx = build_context(host=host, port=port)
    app = create_app(ctx)
    logger.info(f"[webui] PAPER TRADING ONLY - web dashboard on http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level=log_level, access_log=False, server_header=False,
                proxy_headers=False, timeout_graceful_shutdown=3)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m webui", description="Liar-V1 owner web dashboard (PAPER only)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})")
    parser.add_argument("--allow-remote", action="store_true", help="allow binding a non-loopback address")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    run(args.host, args.port, allow_remote=args.allow_remote, log_level=args.log_level)


def _app_from_env() -> Any:  # pragma: no cover - for `uvicorn webui.server:_app_from_env --factory`
    return create_app()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
