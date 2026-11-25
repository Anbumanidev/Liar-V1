"""Web dashboard writes: every action goes through the control-command queue / settings versions."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from config.loader import load_hard_caps
from storage.models import ControlCommand, SettingsVersion
from tests.webui.webui_kit import login, make_ctx
from webui.server import create_app


def _commands(env: dict[str, Any]) -> list[ControlCommand]:
    with env["factory"]() as s:
        return list(s.execute(select(ControlCommand).order_by(ControlCommand.id)).scalars())


def _versions(env: dict[str, Any]) -> int:
    with env["factory"]() as s:
        return int(s.execute(select(func.count(SettingsVersion.id))).scalar_one())


def test_webui_engine_controls_enqueue_commands(authed: TestClient, web_env: dict[str, Any]) -> None:
    assert authed.post("/api/control", json={"action": "pause"}).json()["type"] == "ENGINE_PAUSE"
    assert authed.post("/api/control", json={"action": "resume"}).json()["type"] == "ENGINE_RESUME"
    r = authed.post("/api/control", json={"action": "kill_switch", "confirm_text": "yes"})
    assert r.status_code == 400
    r = authed.post("/api/control", json={"action": "kill_switch", "confirm_text": "CONFIRM"})
    assert r.status_code == 200 and r.json()["type"] == "KILL_SWITCH"
    r = authed.post("/api/control", json={"action": "analyze_now", "symbol": "reliance", "mode": "SWING"})
    assert r.status_code == 200
    r = authed.post("/api/control", json={"action": "analyze_now", "symbol": "BAD SYMBOL!"})
    assert r.status_code == 422
    sbin = web_env["ids"]["sbin"]
    assert authed.post("/api/control", json={"action": "close_position", "position_id": sbin}).status_code == 200
    r = authed.post("/api/control", json={"action": "close_position", "position_id": web_env["ids"]["itc"]})
    assert r.status_code == 409
    assert authed.post("/api/control", json={"action": "trade_live"}).status_code == 422
    cmds = _commands(web_env)
    assert [c.command_type for c in cmds] == ["ENGINE_PAUSE", "ENGINE_RESUME", "KILL_SWITCH", "ANALYZE_NOW",
                                              "CLOSE_POSITION"]
    assert all(c.status == "PENDING" and c.requested_by == "admin" for c in cmds)
    assert cmds[3].payload_json == {"symbol": "RELIANCE", "mode": "SWING", "route": False}
    assert cmds[4].payload_json["position_id"] == sbin and cmds[2].payload_json["exit_cnc"] is False
    st = authed.get(f"/api/commands/{cmds[0].id}").json()
    assert st["status"] == "PENDING" and st["finished"] is False and st["type"] == "ENGINE_PAUSE"
    assert authed.get("/api/commands/99999").status_code == 404
    assert len(authed.get("/api/commands?limit=3").json()["items"]) == 3


def test_webui_approve_and_reject_enqueue(authed: TestClient, web_env: dict[str, Any]) -> None:
    aid = web_env["ids"]["approval"]
    r = authed.post(f"/api/approvals/{aid}/approve")
    assert r.status_code == 200 and r.json()["type"] == "APPROVE"
    r = authed.post(f"/api/approvals/{aid}/reject", json={"reason": "too extended"})
    assert r.status_code == 200 and r.json()["type"] == "REJECT"
    assert authed.post(f"/api/approvals/{web_env['ids']['approval_done']}/approve").status_code == 409
    assert authed.post("/api/approvals/9999/reject").status_code == 404
    cmds = _commands(web_env)
    assert cmds[0].payload_json == {"approval_id": aid}
    assert cmds[1].payload_json == {"approval_id": aid, "reason": "too extended"}


def test_webui_autonomy_creates_version_and_reload(authed: TestClient, web_env: dict[str, Any], ctx: Any) -> None:
    before = _versions(web_env)
    r = authed.post("/api/settings/autonomy", json={"autonomy": "HUMAN_IN_THE_LOOP"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is True and body["autonomy"] == "HUMAN_IN_THE_LOOP"
    assert _versions(web_env) == before + 1
    [cmd] = _commands(web_env)
    assert cmd.command_type == "RELOAD_SETTINGS" and cmd.id == body["command_id"]
    assert authed.get("/api/overview").json()["autonomy"]["value"] == "HUMAN_IN_THE_LOOP"
    same = authed.post("/api/settings/autonomy", json={"autonomy": "HUMAN_IN_THE_LOOP"}).json()
    assert same["changed"] is False and _versions(web_env) == before + 1
    # automatic execution needs a typed confirmation
    assert authed.post("/api/settings/autonomy", json={"autonomy": "FULL_AUTO"}).status_code == 400
    r = authed.post("/api/settings/autonomy", json={"autonomy": "HYBRID", "confirm_text": "CONFIRM",
                                                    "hybrid_min_grade": "A", "hybrid_min_confidence": 80})
    assert r.status_code == 200 and r.json()["changed"] is True
    s = ctx.settings()
    assert s.modes.autonomy.value == "HYBRID" and s.modes.hybrid_auto_min_grade.value == "A"
    assert s.modes.hybrid_auto_min_confidence == 80
    assert authed.post("/api/settings/autonomy", json={"autonomy": "YOLO"}).status_code == 422


def test_webui_autonomy_respects_allow_full_auto_false(web_env: dict[str, Any]) -> None:
    caps = load_hard_caps().model_copy(update={"allow_full_auto": False})
    ctx = make_ctx(web_env, caps=caps)
    with TestClient(create_app(ctx)) as c:
        login(c)
        before = _versions(web_env)
        for level in ("FULL_AUTO", "HYBRID"):
            r = c.post("/api/settings/autonomy", json={"autonomy": level, "confirm_text": "CONFIRM"})
            assert r.status_code == 403 and "allow_full_auto" in r.json()["detail"]
        assert _versions(web_env) == before and _commands(web_env) == []
        opts = {o["value"]: o for o in c.get("/api/overview").json()["autonomy"]["options"]}
        assert opts["FULL_AUTO"]["locked"] and opts["HYBRID"]["locked"] and not opts["SIGNALS_ONLY"]["locked"]
        assert c.post("/api/settings/autonomy", json={"autonomy": "SIGNALS_ONLY"}).json()["changed"] is True
    ctx.close()


def test_webui_new_batch_enqueues_command(authed: TestClient, web_env: dict[str, Any]) -> None:
    r = authed.post("/api/batches/new", json={"starting_capital": 1500, "open_cnc_handling": "RUN_TO_COMPLETION",
                                              "confirm_text": "nope"})
    assert r.status_code == 400
    r = authed.post("/api/batches/new", json={"name": "Batch 1", "confirm_text": "CONFIRM"})
    assert r.status_code == 409
    r = authed.post("/api/batches/new", json={"starting_capital": 1500, "open_cnc_handling": "RUN_TO_COMPLETION",
                                              "confirm_text": "CONFIRM"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Batch 3" and r.json()["type"] == "NEW_BATCH"
    r2 = authed.post("/api/batches/new", json={"name": "Fresh start", "confirm_text": "Batch 2"})
    assert r2.status_code == 200 and r2.json()["starting_capital"] == 1000.0  # settings default
    c1, c2 = _commands(web_env)
    assert c1.command_type == "NEW_BATCH"
    assert {k: c1.payload_json[k] for k in ("name", "starting_capital", "open_cnc_handling", "confirm_text")} == {
        "name": "Batch 3", "starting_capital": 1500.0, "open_cnc_handling": "RUN_TO_COMPLETION",
        "confirm_text": "CONFIRM"}
    assert c2.payload_json["name"] == "Fresh start" and c2.payload_json["confirm_text"] == "Batch 2"
    last = authed.get("/api/batches").json()["last_new_batch_command"]
    assert last["id"] == c2.id and last["status"] == "PENDING"
