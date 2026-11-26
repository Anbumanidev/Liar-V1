"""Session cookies, CSRF and security headers for the owner-only web dashboard.

* Login reuses ``dashboard.auth`` (credential file, bcrypt, lockout, ``login_attempts`` + audit).
* Session = signed (itsdangerous) HttpOnly cookie, ``SameSite=Strict``. Payload:
  ``{sid, u, csrf, iat, at}`` where ``at`` is the last *user* activity (epoch, from the
  context clock). Idle timeout = ``dashboard.auth.IDLE_TIMEOUT`` (30 min); absolute lifetime =
  the auth file's ``cookie.expiry_days``. Requests flagged ``X-Background: 1`` (timers,
  auto-refresh) and the SSE stream do not count as activity - exactly like the Streamlit
  dashboard, where auto-refresh reruns never extend the session.
* CSRF: every state-changing request needs ``X-CSRF-Token`` equal to the token issued at login
  (bound into the signed cookie) AND a same-origin ``Origin``/``Referer``/``Sec-Fetch-Site``.
* The signing key is generated once into ``runtime/web_secret.key`` (git-ignored).
"""

from __future__ import annotations

import hmac
import json
import os
import secrets as pysecrets
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from itsdangerous import BadSignature, URLSafeSerializer
from loguru import logger

COOKIE_NAME = "liar_web_session"
CSRF_HEADER = "x-csrf-token"
BACKGROUND_HEADER = "x-background"

#: API paths reachable without a session.
PUBLIC_API_PATHS = frozenset({"/api/login"})
#: API paths that never count as user activity (no sliding of the idle timeout).
BACKGROUND_PATHS = frozenset({"/api/stream"})
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
       "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")

SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"content-security-policy", CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"same-origin"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
)


def load_or_create_secret(path: Path) -> bytes:
    """Read the cookie signing key, creating a random 64-byte key on first use."""
    path = Path(path)
    try:
        if path.exists():
            data = path.read_text(encoding="ascii").strip()
            if len(data) >= 32:
                return data.encode("ascii")
            logger.warning(f"[webui] secret key file {path} too short; regenerating")
    except OSError as exc:
        logger.warning(f"[webui] cannot read {path}: {exc}; regenerating")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = pysecrets.token_urlsafe(64)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(key, encoding="ascii")
    try:
        os.chmod(tmp, 0o600)
    except OSError:  # Windows ignores most mode bits
        pass
    os.replace(tmp, path)
    logger.info(f"[webui] generated cookie signing key at {path}")
    return key.encode("ascii")


class SessionManager:
    """Sign / verify session cookies (stateless, plus an in-memory revocation list)."""

    #: "Keep this screen live" sessions (a trading monitor): no 30-min idle logout, at most this long.
    LIVE_IDLE = timedelta(hours=16)

    def __init__(self, secret: bytes, *, idle: timedelta, absolute: timedelta) -> None:
        self._serializer = URLSafeSerializer(secret, salt="liar-v1-webui-session")
        self.idle = idle
        self.absolute = absolute
        self.live_idle = min(self.LIVE_IDLE, absolute)
        self._revoked: set[str] = set()
        self._lock = threading.Lock()

    def issue(self, username: str, now_ts: float, *, sid: str | None = None, csrf: str | None = None,
              iat: float | None = None, keep_live: bool = False) -> tuple[str, dict[str, Any]]:
        payload = {"sid": sid or pysecrets.token_urlsafe(16), "u": username,
                   "csrf": csrf or pysecrets.token_urlsafe(32), "iat": iat if iat is not None else now_ts,
                   "at": now_ts, "kl": bool(keep_live)}
        return self._serializer.dumps(payload), payload

    def idle_limit(self, payload: dict[str, Any]) -> timedelta:
        """30 min normally; up to 16 h for a session opened with 'Keep this screen live'."""
        return self.live_idle if payload.get("kl") else self.idle

    def refresh(self, payload: dict[str, Any], now_ts: float) -> str:
        """Re-sign with ``at = now`` (sliding idle timeout)."""
        return self.issue(payload["u"], now_ts, sid=payload["sid"], csrf=payload["csrf"], iat=payload["iat"],
                          keep_live=bool(payload.get("kl")))[0]

    def verify(self, token: str | None, now_ts: float) -> tuple[dict[str, Any] | None, str]:
        """Return ``(payload, "")`` or ``(None, reason)``."""
        if not token:
            return None, "not authenticated"
        try:
            payload = self._serializer.loads(token)
        except BadSignature:
            return None, "invalid session"
        if not isinstance(payload, dict) or not all(k in payload for k in ("sid", "u", "csrf", "iat", "at")):
            return None, "invalid session"
        with self._lock:
            if payload["sid"] in self._revoked:
                return None, "logged out"
        try:
            at, iat = float(payload["at"]), float(payload["iat"])
        except (TypeError, ValueError):
            return None, "invalid session"
        if now_ts - at >= self.idle_limit(payload).total_seconds():
            return None, ("Session expired after 30 minutes of inactivity. Please log in again (tick 'Keep this "
                          "screen live' to keep a monitor screen signed in)." if not payload.get("kl")
                          else "Live-screen session ended after 16 hours. Please log in again.")
        if now_ts - iat >= self.absolute.total_seconds():
            return None, "Session expired. Please log in again."
        return payload, ""

    def revoke(self, sid: str) -> None:
        with self._lock:
            self._revoked.add(sid)

    def idle_deadline(self, payload: dict[str, Any]) -> float:
        return float(payload["at"]) + self.idle_limit(payload).total_seconds()


def cookie_header(value: str, *, secure: bool, max_age: int | None = None) -> bytes:
    parts = [f"{COOKIE_NAME}={value}", "Path=/", "HttpOnly", "SameSite=Strict"]
    if max_age is not None:
        parts.append(f"Max-Age={max_age}")
    if secure:
        parts.append("Secure")
    return "; ".join(parts).encode("latin-1")


def _headers(scope: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in scope.get("headers") or []:
        out[k.decode("latin-1").lower()] = v.decode("latin-1")
    return out


def _cookie(headers: dict[str, str], name: str) -> str | None:
    raw = headers.get("cookie")
    if not raw:
        return None
    for part in raw.split(";"):
        k, _, v = part.strip().partition("=")
        if k == name:
            return v.strip().strip('"')
    return None


def same_origin(headers: dict[str, str], scheme: str) -> bool:
    """Reject cross-site browser requests (Origin / Referer / Sec-Fetch-Site vs Host)."""
    host = headers.get("host", "")
    if headers.get("sec-fetch-site") in ("cross-site", "same-site"):
        return False
    expected = {f"{scheme}://{host}".lower()}
    origin = headers.get("origin")
    if origin is not None:
        return origin.lower() in expected
    referer = headers.get("referer")
    if referer:
        parts = urlsplit(referer)
        return f"{parts.scheme}://{parts.netloc}".lower() in expected
    return True  # non-browser client (no Origin/Referer): the CSRF token still applies


class SecurityMiddleware:
    """Pure ASGI middleware: security headers, cache policy, session + CSRF enforcement.

    Authenticated requests get ``scope["state"]["user"]`` / ``["session"]``. Endpoints can set
    ``scope["state"]["logout"] = True`` to suppress the sliding cookie refresh.
    """

    def __init__(self, app: Any, *, sessions: SessionManager, clock: Any, secure_cookies: bool = False,
                 setup_required: Any = None) -> None:
        self.app = app
        self.sessions = sessions
        self.clock = clock
        self.secure = secure_cookies
        self.setup_required = setup_required or (lambda: False)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET").upper()
        headers = _headers(scope)
        state = scope.setdefault("state", {})
        refresh_cookie: bytes | None = None

        if path.startswith("/api/"):
            if method in UNSAFE_METHODS and not same_origin(headers, scope.get("scheme", "http")):
                await self._json(send, 403, {"detail": "cross-origin request blocked"})
                return
            if path not in PUBLIC_API_PATHS:
                now_ts = self.clock.now().timestamp()
                payload, reason = self.sessions.verify(_cookie(headers, COOKIE_NAME), now_ts)
                if payload is None:
                    await self._json(send, 401, {"detail": reason or "not authenticated",
                                                 "setup_required": bool(self.setup_required())},
                                     clear_cookie=reason != "not authenticated")
                    return
                if method in UNSAFE_METHODS:
                    token = headers.get(CSRF_HEADER, "")
                    if not token or not hmac.compare_digest(token, str(payload["csrf"])):
                        await self._json(send, 403, {"detail": "CSRF token missing or invalid"})
                        return
                state["user"] = payload["u"]
                state["session"] = payload
                background = path in BACKGROUND_PATHS or headers.get(BACKGROUND_HEADER) == "1"
                if not background:
                    refresh_cookie = cookie_header(self.sessions.refresh(payload, now_ts), secure=self.secure)

        cache = self._cache_control(path)

        async def _send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                hdrs = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"
                        or not cache]
                existing = {k.lower() for k, _ in hdrs}
                for k, v in SECURITY_HEADERS:
                    if k not in existing:
                        hdrs.append((k, v))
                if cache:
                    hdrs.append((b"cache-control", cache))
                if refresh_cookie is not None and not state.get("logout") and not state.get("cookie_set"):
                    hdrs.append((b"set-cookie", refresh_cookie))
                message = {**message, "headers": hdrs}
            await send(message)

        await self.app(scope, receive, _send)

    @staticmethod
    def _cache_control(path: str) -> bytes | None:
        if path.startswith("/api/"):
            return b"no-store"
        if path.startswith("/static/vendor/"):
            return b"public, max-age=86400"
        if path.startswith("/static/") or path in ("/", "/index.html"):
            return b"no-cache"
        return None

    async def _json(self, send: Any, status: int, body: dict[str, Any], *, clear_cookie: bool = False) -> None:
        raw = json.dumps(body).encode("utf-8")
        hdrs = [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode()),
                (b"cache-control", b"no-store"), *SECURITY_HEADERS]
        if clear_cookie:
            hdrs.append((b"set-cookie", cookie_header("", secure=self.secure, max_age=0)))
        await send({"type": "http.response.start", "status": status, "headers": hdrs})
        await send({"type": "http.response.body", "body": raw})
