"""'Keep this screen live' login: a monitoring screen is not signed out after 30 idle minutes."""

from __future__ import annotations


def test_login_with_keep_live(client) -> None:  # type: ignore[no-untyped-def]
    from tests.webui.webui_kit import PASSWORD

    r = client.post("/api/login", json={"username": "admin", "password": PASSWORD, "keep_live": True})
    assert r.status_code == 200 and r.json()["keep_live"] is True and r.json()["idle_timeout_s"] == 16 * 3600
    client.headers["X-CSRF-Token"] = r.json()["csrf"]
    me = client.get("/api/me").json()
    assert me["keep_live"] is True and me["idle_timeout_s"] == 16 * 3600
