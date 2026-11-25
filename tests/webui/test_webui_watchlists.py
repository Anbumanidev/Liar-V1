"""Follow-up lists API (/api/watchlists*, /api/plan): CRUD, validation, CSRF, 501 while ``data.watchlists`` is
missing, SCAN_NOW enqueue payload and the next-day plan. Uses an in-memory fake with the exact contract of
``data.watchlists`` (keyword-only signatures, so a wrong call fails the test); when the real module exists the
same flows also run against it on the temp test DB."""

from __future__ import annotations

import importlib
import sys
import types
from datetime import date, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from core.clock import IST
from storage.models import ControlCommand
from tests.webui.webui_kit import NOW

# =========================================================================== fake data.watchlists


class FakeWatchlists:
    """Contract-faithful in-memory stand-in for ``data.watchlists`` (callers commit; ValueError = bad input)."""

    def __init__(self) -> None:
        self.lists: dict[int, dict[str, Any]] = {}
        self.items: dict[int, dict[str, Any]] = {}
        self._id = 0
        self.audit: list[tuple[str, str]] = []

    def _next(self) -> int:
        self._id += 1
        return self._id

    def _items_of(self, wid: int) -> list[dict[str, Any]]:
        return [dict(i) for i in self.items.values() if i["watchlist_id"] == wid]

    def _brief(self, wl: dict[str, Any]) -> dict[str, Any]:
        return dict(wl) | {"items": len(self._items_of(wl["id"]))}

    # ---- contract
    def list_watchlists(self, session: Any, *, include_archived: bool = False) -> list[dict[str, Any]]:
        return [self._brief(w) for w in self.lists.values() if include_archived or not w["archived"]]

    def get_watchlist(self, session: Any, watchlist_id: int) -> dict[str, Any] | None:
        wl = self.lists.get(watchlist_id)
        return None if wl is None else dict(wl) | {"items": self._items_of(watchlist_id)}

    def get_or_create_default(self, session: Any, *, now: datetime, actor: str = "owner") -> dict[str, Any]:
        for w in self.lists.values():
            if w["name"] == "Follow-up":
                return self._brief(w)
        return self.create_watchlist(session, "Follow-up", now=now, actor=actor)

    def create_watchlist(self, session: Any, name: str, *, now: datetime, kind: str = "MANUAL",
                         description: str | None = None, use_for_trading: bool = False, for_date: date | None = None,
                         actor: str = "owner") -> dict[str, Any]:
        if not name or not name.strip():
            raise ValueError("name is required")
        if any(w["name"] == name for w in self.lists.values()):
            raise ValueError(f"a list named {name!r} already exists")
        wid = self._next()
        self.lists[wid] = {"id": wid, "name": name, "kind": kind, "description": description,
                           "use_for_trading": use_for_trading, "for_date": for_date, "archived": False,
                           "created_at": now, "updated_at": now}
        self.audit.append((actor, f"CREATE {name}"))
        return self._brief(self.lists[wid])

    def update_watchlist(self, session: Any, watchlist_id: int, *, now: datetime, name: str | None = None,
                         description: str | None = None, use_for_trading: bool | None = None,
                         archived: bool | None = None, actor: str = "owner") -> dict[str, Any]:
        wl = self.lists.get(watchlist_id)
        if wl is None:
            raise ValueError(f"list {watchlist_id} not found")
        if name is not None:
            if any(w["name"] == name and w["id"] != watchlist_id for w in self.lists.values()):
                raise ValueError(f"a list named {name!r} already exists")
            wl["name"] = name
        if description is not None:
            wl["description"] = description
        if use_for_trading is not None:
            wl["use_for_trading"] = use_for_trading
        if archived is not None:
            wl["archived"] = archived
        wl["updated_at"] = now
        self.audit.append((actor, f"UPDATE {watchlist_id}"))
        return self._brief(wl)

    def delete_watchlist(self, session: Any, watchlist_id: int, *, now: datetime, actor: str = "owner") -> None:
        if self.lists.pop(watchlist_id, None) is None:
            raise ValueError(f"list {watchlist_id} not found")
        for iid in [i for i, it in self.items.items() if it["watchlist_id"] == watchlist_id]:
            del self.items[iid]
        self.audit.append((actor, f"DELETE {watchlist_id}"))

    def add_item(self, session: Any, watchlist_id: int, symbol: str, *, now: datetime, note: str | None = None,
                 source: str = "MANUAL", added_price: float | None = None, mode: str | None = None,
                 plan: dict[str, Any] | None = None, decision_id: int | None = None,
                 actor: str = "owner") -> dict[str, Any]:
        if watchlist_id not in self.lists:
            raise ValueError(f"list {watchlist_id} not found")
        for it in self.items.values():
            if it["watchlist_id"] == watchlist_id and it["symbol"] == symbol:   # upsert
                it.update({k: v for k, v in {"note": note, "mode": mode, "decision_id": decision_id}.items() if v})
                return dict(it)
        iid = self._next()
        self.items[iid] = {"id": iid, "watchlist_id": watchlist_id, "symbol": symbol, "mode": mode, "source": source,
                           "note": note, "added_at": now, "added_by": actor, "added_price": added_price, "plan": plan,
                           "status": "ACTIVE", "status_note": None, "decision_id": decision_id,
                           "last_analysis_id": None, "last_analysis_at": None, "last_summary": None}
        self.audit.append((actor, f"ADD {symbol}"))
        return dict(self.items[iid])

    def remove_item(self, session: Any, watchlist_id: int, item_id_or_symbol: int | str, *, now: datetime,
                    actor: str = "owner") -> None:
        for iid, it in list(self.items.items()):
            if it["watchlist_id"] == watchlist_id and item_id_or_symbol in (iid, it["symbol"]):
                del self.items[iid]
                return
        raise ValueError(f"item {item_id_or_symbol} not found")

    def latest_plan(self, session: Any, *, on_or_after: date) -> dict[str, Any] | None:
        plans = [w for w in self.lists.values() if w["kind"] == "NEXT_DAY" and w["for_date"] and w["for_date"] >= on_or_after]
        if not plans:
            return None
        wl = max(plans, key=lambda w: (w["for_date"], w["id"]))
        items = sorted(self._items_of(wl["id"]), key=lambda i: -float((i.get("plan") or {}).get("score") or 0))
        return dict(wl) | {"items": items}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
        mod = types.ModuleType("data.watchlists")
        for name in ("list_watchlists", "get_watchlist", "get_or_create_default", "create_watchlist",
                     "update_watchlist", "delete_watchlist", "add_item", "remove_item", "latest_plan"):
            setattr(mod, name, getattr(self, name))
        monkeypatch.setitem(sys.modules, "data.watchlists", mod)
        return mod


@pytest.fixture
def fake_wl(monkeypatch: pytest.MonkeyPatch) -> FakeWatchlists:
    fake = FakeWatchlists()
    fake.install(monkeypatch)
    return fake


def _commands(env: dict[str, Any]) -> list[ControlCommand]:
    with env["factory"]() as s:
        return list(s.execute(select(ControlCommand).order_by(ControlCommand.id)).scalars())


# =========================================================================== 501 / security


def test_webui_watchlists_501_when_module_missing(authed: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "data.watchlists", None)   # import -> ImportError
    assert authed.get("/api/watchlists").status_code == 501
    assert authed.post("/api/watchlists", json={"name": "Breakouts"}).status_code == 501
    assert authed.get("/api/watchlists/1").status_code == 501
    assert authed.patch("/api/watchlists/1", json={"name": "x"}).status_code == 501
    assert authed.delete("/api/watchlists/1").status_code == 501
    assert authed.post("/api/watchlists/1/items", json={"symbol": "SBIN"}).status_code == 501
    assert authed.delete("/api/watchlists/1/items/2").status_code == 501
    assert authed.post("/api/watchlists/default/items", json={"symbol": "SBIN"}).status_code == 501
    assert authed.post("/api/watchlists/1/scan", json={}).status_code == 501
    r = authed.get("/api/plan")
    assert r.status_code == 501 and r.json()["detail"] == "not available yet"


def test_webui_watchlists_require_login_and_csrf(client: TestClient, fake_wl: FakeWatchlists) -> None:
    assert client.get("/api/watchlists").status_code == 401
    from tests.webui.webui_kit import login

    login(client)
    wid = client.post("/api/watchlists", json={"name": "Breakouts"}).json()["id"]
    token = client.headers.pop("X-CSRF-Token")
    for method, path, body in (("post", "/api/watchlists", {"name": "Other"}),
                               ("patch", f"/api/watchlists/{wid}", {"name": "Renamed"}),
                               ("delete", f"/api/watchlists/{wid}", None),
                               ("post", f"/api/watchlists/{wid}/items", {"symbol": "SBIN"}),
                               ("post", "/api/watchlists/default/items", {"symbol": "SBIN"}),
                               ("post", f"/api/watchlists/{wid}/scan", {})):
        kw = {"json": body} if body is not None else {}
        r = getattr(client, method)(path, **kw)
        assert r.status_code == 403 and "CSRF" in r.json()["detail"], (method, path, r.text)
    client.headers["X-CSRF-Token"] = token
    r = client.post("/api/watchlists", json={"name": "Evil"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and "cross-origin" in r.json()["detail"]
    r = client.delete(f"/api/watchlists/{wid}", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert [w["name"] for w in fake_wl.lists.values()] == ["Breakouts"]   # nothing changed by the rejected calls
    assert not fake_wl.items


# =========================================================================== CRUD


def test_webui_watchlists_crud_and_validation(authed: TestClient, fake_wl: FakeWatchlists) -> None:
    r = authed.post("/api/watchlists", json={"name": "  Breakouts ", "description": "watch for ORB", "use_for_trading": True})
    assert r.status_code == 200, r.text
    wl = r.json()
    assert wl["name"] == "Breakouts" and wl["kind"] == "MANUAL" and wl["use_for_trading"] is True and wl["items"] == 0
    assert wl["created_at"].endswith("+05:30")
    assert authed.post("/api/watchlists", json={"name": "Breakouts"}).status_code == 409      # duplicate (ValueError)
    assert authed.post("/api/watchlists", json={"name": ""}).status_code == 422
    assert authed.post("/api/watchlists", json={"name": "   "}).status_code == 422
    assert authed.post("/api/watchlists", json={"name": "x" * 81}).status_code == 422
    wid = wl["id"]

    # add stocks: symbol validated + upper-cased, added_price = current quote (FakeData SBIN ltp 812)
    r = authed.post(f"/api/watchlists/{wid}/items", json={"symbol": "sbin.ns", "note": "wait for 820", "mode": "swing"})
    assert r.status_code == 200, r.text
    item = r.json()
    assert item["symbol"] == "SBIN" and item["added_price"] == 812.0 and item["mode"] == "SWING"
    assert item["source"] == "MANUAL" and item["status"] == "ACTIVE" and item["plan_view"] is None
    r = authed.post(f"/api/watchlists/{wid}/items", json={"symbol": "ITC", "decision_id": 3})
    assert r.json()["source"] == "ANALYSIS" and r.json()["decision_id"] == 3 and r.json()["added_price"] == 309.0
    r = authed.post(f"/api/watchlists/{wid}/items", json={"symbol": "NOQUOTE"})
    assert r.status_code == 200 and r.json()["added_price"] is None          # no quote -> no price, still added
    assert authed.post(f"/api/watchlists/{wid}/items", json={"symbol": "BAD SYMBOL!"}).status_code == 422
    assert authed.post(f"/api/watchlists/{wid}/items", json={"symbol": "../etc"}).status_code == 422
    assert authed.post(f"/api/watchlists/{wid}/items", json={"symbol": "TCS", "mode": "WEEKLY"}).status_code == 422
    assert authed.post("/api/watchlists/999/items", json={"symbol": "TCS"}).status_code == 404

    got = authed.get(f"/api/watchlists/{wid}").json()
    assert [i["symbol"] for i in got["items"]] == ["SBIN", "ITC", "NOQUOTE"]
    assert got["items"][0]["note"] == "wait for 820" and got["items"][0]["added_at"].endswith("+05:30")
    lists = authed.get("/api/watchlists").json()["lists"]
    assert [(x["name"], x["items"]) for x in lists] == [("Breakouts", 3)]

    # rename / toggle
    r = authed.patch(f"/api/watchlists/{wid}", json={"name": "Breakout watch", "use_for_trading": False})
    assert r.status_code == 200 and r.json()["name"] == "Breakout watch" and r.json()["use_for_trading"] is False
    assert authed.patch(f"/api/watchlists/{wid}", json={"name": ""}).status_code == 422
    assert authed.patch("/api/watchlists/999", json={"name": "x"}).status_code == 404
    authed.post("/api/watchlists", json={"name": "Second"})
    assert authed.patch(f"/api/watchlists/{wid}", json={"name": "Second"}).status_code == 409

    # remove a stock
    sbin_id = got["items"][0]["id"]
    assert authed.delete(f"/api/watchlists/{wid}/items/{sbin_id}").json() == {"ok": True, "id": sbin_id}
    assert authed.delete(f"/api/watchlists/{wid}/items/{sbin_id}").status_code == 404
    assert [i["symbol"] for i in authed.get(f"/api/watchlists/{wid}").json()["items"]] == ["ITC", "NOQUOTE"]

    # delete the list
    assert authed.delete(f"/api/watchlists/{wid}").json()["ok"] is True
    assert authed.get(f"/api/watchlists/{wid}").status_code == 404
    assert authed.delete(f"/api/watchlists/{wid}").status_code == 404
    assert all(actor == "admin" for actor, _ in fake_wl.audit)   # every write attributed to the logged-in owner


def test_webui_watchlists_default_list(authed: TestClient, fake_wl: FakeWatchlists) -> None:
    r = authed.post("/api/watchlists/default/items", json={"symbol": "bel", "decision_id": 7, "mode": "INTRADAY"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["watchlist"]["name"] == "Follow-up" and body["item"]["symbol"] == "BEL"
    assert body["item"]["source"] == "ANALYSIS" and body["item"]["added_price"] == 401.0
    r2 = authed.post("/api/watchlists/default/items", json={"symbol": "TCS"})
    assert r2.json()["watchlist"]["id"] == body["watchlist"]["id"]          # same default list
    assert authed.post("/api/watchlists/default/items", json={"symbol": "no way"}).status_code == 422
    assert len(fake_wl.lists) == 1 and sorted(i["symbol"] for i in fake_wl.items.values()) == ["BEL", "TCS"]


# =========================================================================== scan


def test_webui_watchlist_scan_enqueues_scan_now(authed: TestClient, fake_wl: FakeWatchlists, web_env: dict[str, Any]) -> None:
    wid = authed.post("/api/watchlists", json={"name": "Scan me"}).json()["id"]
    assert authed.post(f"/api/watchlists/{wid}/scan", json={}).status_code == 400       # empty list
    authed.post(f"/api/watchlists/{wid}/items", json={"symbol": "SBIN"})
    authed.post(f"/api/watchlists/{wid}/items", json={"symbol": "ITC"})
    r = authed.post(f"/api/watchlists/{wid}/scan", json={"mode": "swing", "use_llm": False})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["type"] == "SCAN_NOW" and out["watchlist_id"] == wid and out["symbols"] == 2 and out["name"] == "Scan me"
    r = authed.post(f"/api/watchlists/{wid}/scan")                                    # no body: INTRADAY + AI
    assert r.status_code == 200
    assert authed.post(f"/api/watchlists/{wid}/scan", json={"mode": "WEEKLY"}).status_code == 422
    assert authed.post("/api/watchlists/999/scan", json={}).status_code == 404
    cmds = _commands(web_env)
    assert [c.command_type for c in cmds] == ["SCAN_NOW", "SCAN_NOW"]
    assert all(c.status == "PENDING" and c.requested_by == "admin" for c in cmds)
    p0 = {k: v for k, v in cmds[0].payload_json.items() if v is not None}
    p1 = {k: v for k, v in cmds[1].payload_json.items() if v is not None}
    assert p0 == {"mode": "SWING", "use_llm": False, "watchlist_id": wid}
    assert p1 == {"mode": "INTRADAY", "use_llm": True, "watchlist_id": wid}


# =========================================================================== plan


def _seed_plan(fake: FakeWatchlists, for_date: date) -> int:
    wl = fake.create_watchlist(None, f"Plan {for_date}", now=NOW, kind="NEXT_DAY", for_date=for_date, actor="system")
    fake.add_item(None, wl["id"], "SBIN", now=NOW, source="EOD_PLAN", actor="system",
                  plan={"setup": "ORB breakout", "direction": "LONG", "trigger": 815.0, "stop": 805.0, "target": 835.0,
                        "score": 71, "reasons": ["closed near the high", "volume 2x"], "affordable": True})
    fake.add_item(None, wl["id"], "ITC", now=NOW, source="EOD_PLAN", actor="system",
                  plan={"strategy": "vwap_reclaim", "side": "SELL", "entry_price": 310.0, "stop_loss": 313.0,
                        "target_1": 304.0, "rr": 2.0, "score": 80, "notes": ["lower highs"], "not_affordable": True})
    for it in fake.items.values():
        if it["symbol"] == "ITC":
            it["status"], it["status_note"] = "INVALIDATED", "gapped below the stop at the open"
    return int(wl["id"])


def test_webui_plan_endpoint(authed: TestClient, fake_wl: FakeWatchlists) -> None:
    assert authed.get("/api/plan").json()["plan"] is None
    wid = _seed_plan(fake_wl, date(2026, 11, 26))
    r = authed.get("/api/plan")
    assert r.status_code == 200, r.text
    plan = r.json()["plan"]
    assert plan["id"] == wid and plan["for_date"] == "2026-11-26" and plan["label"] == "Plan for tomorrow"
    assert plan["counts"] == {"ACTIVE": 1, "TRIGGERED": 0, "INVALIDATED": 1}
    itc, sbin = plan["items"]                                   # sorted by score (module order kept)
    assert itc["symbol"] == "ITC" and itc["status"] == "INVALIDATED" and itc["status_note"].startswith("gapped")
    assert itc["plan_view"] == {"setup": "vwap_reclaim", "direction": "SHORT", "trigger": 310.0, "stop": 313.0,
                                "target": 304.0, "rr": 2.0, "score": 80.0, "qty": None, "affordable": False,
                                "reasons": ["lower highs"], "mode": None}
    assert sbin["plan_view"]["rr"] == 2.0 and sbin["plan_view"]["direction"] == "LONG"   # R:R computed 20 / 10
    assert sbin["plan_view"]["affordable"] is True and sbin["plan_view"]["reasons"] == ["closed near the high", "volume 2x"]
    assert authed.get("/api/plan?date=2026-11-27").json()["plan"] is None
    assert authed.get("/api/plan?date=2026-11-26").json()["plan"]["id"] == wid
    assert authed.get("/api/plan?date=garbage").status_code == 422
    # NEXT_DAY lists are listed too (the Follow-up page shows them under "Plans")
    kinds = {x["kind"] for x in authed.get("/api/watchlists").json()["lists"]}
    assert kinds == {"NEXT_DAY"}


def test_webui_plan_label() -> None:
    from webui.routes_watchlists import plan_label

    today = date(2026, 11, 26)
    assert plan_label(today, datetime(2026, 11, 26, 8, 45, tzinfo=IST)) == "Plan for today (checked again at 09:20)"
    assert plan_label(today, datetime(2026, 11, 26, 9, 25, tzinfo=IST)) == "Plan for today"
    assert plan_label(today + timedelta(days=1), datetime(2026, 11, 26, 15, 45, tzinfo=IST)) == "Plan for tomorrow"
    assert plan_label(date(2026, 11, 30), datetime(2026, 11, 27, 16, 0, tzinfo=IST)) == "Plan for Mon 30 Nov"
    assert plan_label(None, datetime(2026, 11, 27, 16, 0, tzinfo=IST)) == "Plan"


def test_webui_plan_view_tolerates_garbage() -> None:
    from webui.routes_watchlists import plan_view

    assert plan_view(None) is None and plan_view({}) is None and plan_view("x") is None
    pv = plan_view({"trigger": "abc", "stop": float("nan"), "direction": "sideways", "reasons": "one reason"})
    assert pv["trigger"] is None and pv["stop"] is None and pv["direction"] is None and pv["reasons"] == ["one reason"]


# =========================================================================== against the real module (when deployed)


def _real_module_or_skip() -> Any:
    try:
        return importlib.import_module("data.watchlists")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"data.watchlists not importable yet: {exc}")


def test_webui_watchlists_real_module(authed: TestClient, web_env: dict[str, Any]) -> None:
    _real_module_or_skip()
    r = authed.post("/api/watchlists", json={"name": "Real list", "description": "via the real module"})
    assert r.status_code == 200, r.text
    wid = r.json()["id"]
    assert authed.post("/api/watchlists", json={"name": "Real list"}).status_code in (409, 422)
    r = authed.post(f"/api/watchlists/{wid}/items", json={"symbol": "sbin", "note": "n1"})
    assert r.status_code == 200, r.text
    assert r.json()["symbol"] == "SBIN" and r.json()["added_price"] == 812.0
    got = authed.get(f"/api/watchlists/{wid}").json()
    assert [i["symbol"] for i in got["items"]] == ["SBIN"]
    assert any(x["id"] == wid and x["items"] == 1 for x in authed.get("/api/watchlists").json()["lists"])
    r = authed.patch(f"/api/watchlists/{wid}", json={"name": "Real list 2", "use_for_trading": True})
    assert r.status_code == 200, r.text
    assert r.json()["use_for_trading"] is True and r.json()["name"] == "Real list 2"
    d = authed.post("/api/watchlists/default/items", json={"symbol": "ITC"})
    assert d.status_code == 200, d.text
    assert d.json()["item"]["symbol"] == "ITC"
    r = authed.post(f"/api/watchlists/{wid}/scan", json={"mode": "INTRADAY", "use_llm": False})
    assert r.status_code == 200, r.text
    cmd = _commands(web_env)[-1]
    assert cmd.command_type == "SCAN_NOW" and cmd.payload_json["watchlist_id"] == wid
    item_id = got["items"][0]["id"]
    assert authed.delete(f"/api/watchlists/{wid}/items/{item_id}").status_code == 200
    assert authed.get(f"/api/watchlists/{wid}").json()["items"] == []
    assert authed.delete(f"/api/watchlists/{wid}").status_code == 200
    assert authed.get(f"/api/watchlists/{wid}").status_code == 404
    r = authed.get("/api/plan")
    assert r.status_code == 200 and r.json()["plan"] is None
