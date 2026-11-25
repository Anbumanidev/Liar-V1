"""Web dashboard: authentication, session idle timeout, CSRF/Origin checks, security headers."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select

from core.clock import SimulatedClock
from storage.models import LoginAttempt
from tests.webui.webui_kit import NOW, PASSWORD, login, make_ctx
from webui.server import create_app


def test_webui_api_requires_session(client: TestClient) -> None:
    for path in ("/api/me", "/api/overview", "/api/today", "/api/trades", "/api/decisions", "/api/stream",
                 "/api/candles?symbol=SBIN", "/api/health", "/api/logs", "/api/batches", "/api/nope"):
        r = client.get(path)
        assert r.status_code == 401, path
        assert r.json()["setup_required"] is False
    assert client.post("/api/control", json={"action": "pause"}).status_code == 401


def test_webui_login_success_sets_hardened_cookie(client: TestClient, web_env: dict[str, Any]) -> None:
    r = client.post("/api/login", json={"username": " Admin ", "password": PASSWORD})
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "admin" and len(body["csrf"]) >= 32
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie
    assert PASSWORD not in r.text
    me = client.get("/api/me")
    assert me.status_code == 200 and me.json()["username"] == "admin" and me.json()["csrf"] == body["csrf"]
    assert me.json()["paper_only"] is True
    with web_env["factory"]() as s:
        rows = s.execute(select(LoginAttempt)).scalars().all()
    assert len(rows) == 1 and rows[0].success and "webui" in (rows[0].client or "")


def test_webui_login_failure_and_lockout(client: TestClient, web_env: dict[str, Any]) -> None:
    for i in range(4):
        r = client.post("/api/login", json={"username": "admin", "password": "wrong"})
        assert r.status_code == 401
        assert r.json()["remaining_attempts"] == 4 - i
    fifth = client.post("/api/login", json={"username": "admin", "password": "wrong"})
    assert fifth.status_code == 429 and "Locked" in fifth.json()["detail"]
    # even the right password is refused while locked (lockout checked before bcrypt)
    locked = client.post("/api/login", json={"username": "admin", "password": PASSWORD})
    assert locked.status_code == 429
    assert client.get("/api/me").status_code == 401
    with web_env["factory"]() as s:
        reasons = [r.reason for r in s.execute(select(LoginAttempt).order_by(LoginAttempt.id)).scalars()]
    assert reasons[-1] == "LOCKED_OUT" and reasons.count("BAD_PASSWORD") == 5
    empty = client.post("/api/login", json={"username": "", "password": ""})
    assert empty.status_code == 400


def test_webui_missing_auth_file_shows_setup(web_env: dict[str, Any]) -> None:
    web_env["auth_file"].unlink()
    ctx = make_ctx(web_env)
    with TestClient(create_app(ctx)) as c:
        r = c.post("/api/login", json={"username": "admin", "password": PASSWORD})
        assert r.status_code == 503 and r.json()["setup_required"] is True
        assert "set_password.py" in r.json()["detail"]
        assert c.get("/api/me").json()["setup_required"] is True
    ctx.close()


def test_webui_csrf_and_origin_checks(authed: TestClient) -> None:
    token = authed.headers.pop("X-CSRF-Token")
    r = authed.post("/api/control", json={"action": "pause"})
    assert r.status_code == 403 and "CSRF" in r.json()["detail"]
    r = authed.post("/api/control", json={"action": "pause"}, headers={"X-CSRF-Token": "forged"})
    assert r.status_code == 403
    r = authed.post("/api/control", json={"action": "pause"},
                    headers={"X-CSRF-Token": token, "Origin": "http://evil.example"})
    assert r.status_code == 403 and "cross-origin" in r.json()["detail"]
    r = authed.post("/api/control", json={"action": "pause"},
                    headers={"X-CSRF-Token": token, "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    r = authed.post("/api/control", json={"action": "pause"},
                    headers={"X-CSRF-Token": token, "Origin": "http://testserver"})
    assert r.status_code == 200
    # login CSRF: a cross-site login POST is blocked too
    r = authed.post("/api/login", json={"username": "admin", "password": PASSWORD},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_webui_untrusted_host_rejected(client: TestClient) -> None:
    r = client.get("/", headers={"Host": "attacker.example"})
    assert r.status_code == 400


def test_webui_idle_timeout_and_background_requests(web_env: dict[str, Any]) -> None:
    clock = SimulatedClock(NOW)
    ctx = make_ctx(web_env, clock=clock)
    with TestClient(create_app(ctx)) as c:
        login(c)
        clock.advance(timedelta(minutes=20))
        # background (auto-refresh) request: allowed but does NOT extend the session
        assert c.get("/api/me", headers={"X-Background": "1"}).status_code == 200
        clock.advance(timedelta(minutes=11))
        r = c.get("/api/me")
        assert r.status_code == 401 and "inactivity" in r.json()["detail"]
    with TestClient(create_app(ctx)) as c:
        login(c)
        clock.advance(timedelta(minutes=20))
        assert c.get("/api/me").status_code == 200       # user activity slides the idle timeout
        clock.advance(timedelta(minutes=20))
        assert c.get("/api/me").status_code == 200
    ctx.close()


def test_webui_logout_revokes_session(authed: TestClient) -> None:
    cookie = authed.cookies.get("liar_web_session")
    r = authed.post("/api/logout")
    assert r.status_code == 200 and "Max-Age=0" in r.headers["set-cookie"]
    authed.cookies.set("liar_web_session", cookie)  # replaying the old cookie must fail
    assert authed.get("/api/me").status_code == 401


def test_webui_static_index_and_security_headers(client: TestClient, ctx: Any) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert 'id="app"' in r.text and "/static/js/app.js" in r.text
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"
    assert "cdn" not in r.text.lower() and "googleapis" not in r.text
    js = client.get("/static/js/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    vendor = client.get("/static/vendor/lightweight-charts.standalone.production.js")
    assert vendor.status_code == 200 and "max-age" in vendor.headers["cache-control"]
    assert client.get("/favicon.ico").status_code == 200
    assert ctx.secret_key_path.exists() and len(ctx.secret_key_path.read_text()) >= 32


def test_webui_run_refuses_non_loopback_bind() -> None:
    import pytest

    from webui.server import run

    with pytest.raises(SystemExit, match="owner-only"):
        run("0.0.0.0", 8600)
