"""Web dashboard: Today | Batch | Overall scopes, the activity live log (+ SSE), Scan now."""

from __future__ import annotations

import importlib.util
import json
from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from core.enums import CommandType
from storage.models import ControlCommand
from tests.webui.webui_kit import DAY, NOW


# =========================================================================== summary


def test_webui_summary_today(authed: TestClient, web_env: dict[str, Any]) -> None:
    r = authed.get(f"/api/summary?scope=today&date={DAY.isoformat()}")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["scope"]["kind"] == "today" and d["scope"]["label"] == "Today"
    k = d["kpis"]
    assert k["realized"] == 10.2 and k["gross"] == 11.5 and k["charges"] == 1.3
    assert k["trades"] == 2 and k["wins"] == 1 and k["losses"] == 1 and k["win_rate"] == 50.0
    assert k["unrealized"] == 46.8 and k["total"] == 57.0 and k["open_positions"] == 1
    assert k["best"]["symbol"] == "ITC" and k["worst"]["symbol"] == "BEL" and k["profit_factor"] == round(12.0 / 1.8, 2)
    m = d["money"]
    assert m["kind"] == "today" and m["day_pnl"] == 57.0
    assert m["batches"] == [{"id": web_env["ids"]["batch"], "name": "Batch 2", "status": "ACTIVE", "net": 10.2,
                             "trades": 2}]
    other = authed.get("/api/summary?scope=today&date=2026-11-12").json()
    assert other["kpis"]["trades"] == 1 and other["kpis"]["unrealized"] == 0.0 and other["scope"]["label"] == "12 Nov 2026"


def test_webui_summary_batch_fallback(authed: TestClient, web_env: dict[str, Any], no_optional_modules: None) -> None:
    old = web_env["ids"]["old_batch"]
    d = authed.get(f"/api/summary?scope=batch&batch_id={old}").json()
    assert d["scope"]["kind"] == "batch" and d["scope"]["batch_name"] == "Batch 1"
    assert d["kpis"]["trades"] == 1 and d["kpis"]["realized"] == -12.0 and d["kpis"]["unrealized"] == 0.0
    m = d["money"]
    assert m["source"] == "fallback" and m["status"] == "CLOSED" and m["starting_capital"] == 1000.0
    assert m["top_up"] == 1000.0 and m["carried_in"] is None and m["equity"] == 995.28
    assert m["pnl"] == -4.72 and m["pnl_pct"] == -0.47 and m["ended_at"]
    cur = authed.get("/api/summary?scope=batch").json()  # default = active batch
    assert cur["scope"]["batch_id"] == web_env["ids"]["batch"] and cur["kpis"]["trades"] == 2
    assert cur["money"]["top_up"] == 4.72 and cur["money"]["carried_in"] == 995.28
    assert cur["money"]["equity"] == round(1010 + 46.8, 2)
    assert authed.get("/api/summary?scope=batch&batch_id=999").status_code == 404
    assert authed.get("/api/summary?scope=weekly").status_code == 422


def test_webui_summary_overall_fake_funding(authed: TestClient, fake_module: Any) -> None:
    fake_module("capital.funding", funding_summary=lambda s: {
        "total_deposited": 1500.0, "total_withdrawn": 100.0, "net_invested": 1400.0, "current_capital": 1450.0,
        "overall_pnl": 50.0, "overall_pnl_pct": 3.33, "active_batch_id": 2, "batches": [{"batch_id": 1}, {"batch_id": 2}],
        "events": []})
    d = authed.get("/api/summary?scope=all").json()
    assert d["scope"]["kind"] == "all" and d["kpis"]["trades"] == 3 and d["kpis"]["realized"] == -1.8
    m = d["money"]
    assert m == {"kind": "all", "source": "funding", "net_invested": 1400.0, "total_deposited": 1500.0,
                 "total_withdrawn": 100.0, "current_capital": 1450.0, "overall_pnl": 50.0, "overall_pnl_pct": 3.33,
                 "batches": 2, "active_batch_id": 2, "realized_all": -1.8}
    assert d["kpis"]["unrealized"] == 46.8


@pytest.mark.skipif(importlib.util.find_spec("capital.funding") is None, reason="capital.funding not deployed yet")
def test_webui_summary_with_real_funding(authed: TestClient, web_env: dict[str, Any]) -> None:
    for scope in ("all", f"batch&batch_id={web_env['ids']['batch']}"):
        d = authed.get(f"/api/summary?scope={scope}").json()
        assert d["money"]["source"] == "funding", d
    m = authed.get("/api/summary?scope=all").json()["money"]
    assert all(isinstance(m[k], (int, float)) for k in ("net_invested", "total_deposited", "current_capital"))
    b = authed.get(f"/api/summary?scope=batch&batch_id={web_env['ids']['batch']}").json()["money"]
    assert b["starting_capital"] == 1000.0 and b["status"] == "ACTIVE"


# =========================================================================== scoped lists


def test_webui_results_and_lists_follow_scope(authed: TestClient, web_env: dict[str, Any],
                                              no_optional_modules: None) -> None:
    old, cur = web_env["ids"]["old_batch"], web_env["ids"]["batch"]
    t = authed.get("/api/trades?scope=today").json()
    assert {r["symbol"] for r in t["rows"]} == {"ITC", "BEL"} and t["scope"]["kind"] == "today"
    assert [r["symbol"] for r in authed.get(f"/api/trades?scope=batch&batch_id={old}").json()["rows"]] == ["HDFCBANK"]
    assert authed.get("/api/trades?scope=all").json()["count"] == 3
    assert authed.get("/api/trades?scope=all&result=LOSS").json()["count"] == 2
    assert authed.get(f"/api/trades?batch_id={old}").json()["count"] == 1  # legacy params still work
    dec = authed.get("/api/decisions?scope=today").json()
    assert len(dec["rows"]) == 6 and dec["scope"]["kind"] == "today"
    assert authed.get("/api/decisions?scope=today&date=2026-11-24").json()["rows"] == []
    assert authed.get(f"/api/decisions?scope=batch&batch_id={old}").json()["rows"] == []
    assert len(authed.get(f"/api/decisions?scope=batch&batch_id={cur}").json()["rows"]) == 6
    assert len(authed.get("/api/decisions?scope=all").json()["rows"]) == 6
    res = authed.get(f"/api/results?scope=batch&batch_id={cur}").json()
    assert res["scope"]["kind"] == "batch" and res["timeline"] == [] and res["summary"]["net"] == 10.2
    assert res["funnel"]["counts"]["TOTAL"] == 6 and res["equity"][-1]["source"] == "LIVE"
    times = [p["time"] for p in res["equity"]]
    assert times == sorted(times) and len(set(times)) == len(times)
    al = authed.get("/api/results?scope=all").json()
    assert al["summary"]["trades"] == 3 and al["day"] is None and al["is_today"] is False
    # Overall = ONE continuous line of total capital (invested so far + every batch's P&L), ending at the
    # current capital - never interleaved per-batch equity points (that zigzagged across batch switches).
    assert al["equity"] and {p["source"] for p in al["equity"]} == {"OVERALL"}
    eq_times = [p["time"] for p in al["equity"]]
    assert eq_times == sorted(eq_times) and len(set(eq_times)) == len(eq_times)
    # (it ends at current capital on real data; this seed's quote differs from last_price, so only sanity here)
    assert all(p["equity"] > 0 for p in al["equity"]) and al["equity"][0]["equity"] == 1000.0
    opts = authed.get("/api/scope/options").json()
    assert opts["active_batch_id"] == cur and [b["name"] for b in opts["batches"]] == ["Batch 2", "Batch 1"]
    assert opts["today"] == DAY.isoformat()


def test_webui_postmortem_scoped(authed: TestClient, web_env: dict[str, Any], fake_module: Any) -> None:
    calls: list[tuple[int, Any]] = []

    def analyze_position(session: Any, position_id: int, data: Any = None) -> dict[str, Any]:
        calls.append((position_id, data))
        loss = position_id == web_env["ids"]["bel"]
        return {"position_id": position_id, "symbol": "X", "result": "LOSS" if loss else "WIN",
                "net_pnl": -1.8 if loss else 12.0, "charges": 0.3,
                "diagnosis": [{"code": "SL_TIGHT", "severity": "medium", "title": "Stop too tight", "fix": "widen",
                               "fixed_by": "risk.sl_min_atr"}] if loss else
                [{"code": "LATE", "severity": "low", "title": "Late entry", "fix": "enter earlier", "fixed_by": None},
                 {"code": "SL_TIGHT", "severity": "high", "title": "Stop too tight", "fix": "widen", "fixed_by": None}]}

    fake_module("analytics.postmortem", analyze_position=analyze_position,
                analyze_day=lambda s, day, batch_id=None, data=None: {"day": day.isoformat(), "trades": [],
                                                                       "summary": {"trades": 0}, "top_issues": []})
    d = authed.get("/api/postmortem?scope=all").json()
    assert {pid for pid, _ in calls} == {web_env["ids"]["itc"], web_env["ids"]["bel"]}
    assert all(data is None for _, data in calls)  # aggregate list never fetches candles
    assert d["summary"] == {"trades": 2, "wins": 1, "losses": 1, "net_pnl": 10.2, "charges": 0.6}
    top = d["top_issues"]
    assert top[0]["code"] == "SL_TIGHT" and top[0]["count"] == 2 and top[0]["severity"] == "high"
    assert top[0]["fixed_by"] == "risk.sl_min_atr" and top[1]["code"] == "LATE"
    assert d["scope"]["kind"] == "all" and d["trades"][0]["issues"]
    calls.clear()
    authed.get(f"/api/postmortem?scope=batch&batch_id={web_env['ids']['old_batch']}")
    assert calls == []  # the old batch has no closed positions rows
    today = authed.get("/api/postmortem?scope=today").json()
    assert today["scope"]["kind"] == "today" and today["day"] == DAY.isoformat()


def test_webui_postmortem_scoped_not_available(authed: TestClient, no_optional_modules: None) -> None:
    assert authed.get("/api/postmortem?scope=all").status_code == 501
    assert authed.get("/api/postmortem?scope=today").status_code == 501


# =========================================================================== live log (activity)


def _activity(**kw: Any) -> str:
    base = {"ts": NOW.isoformat(), "category": "activity", "level": "info"}
    return json.dumps(base | kw)


def test_webui_activity_log_tail(authed: TestClient, web_env: dict[str, Any]) -> None:
    path = web_env["log_dir"] / DAY.isoformat() / "activity.jsonl"
    path.write_text("\n".join([
        _activity(stage="engine", message="idle: next swing_scan at 16:00", state="RUNNING", next_job="swing_scan"),
        _activity(stage="llm", message="ITC technical start", symbol="ITC", agent="technical", event="start",
                  model="qwen3:4b"),
        _activity(stage="llm", message="ITC technical done", symbol="ITC", agent="technical", event="done",
                  status="OK", latency_ms=12000),
        _activity(stage="risk", message="BEL rejected", symbol="BEL", level="warning", approved=False,
                  rule="MAX_TRADES_PER_DAY"),
    ]) + "\n", encoding="utf-8")
    d = authed.get("/api/logs?file=activity&lines=300").json()
    assert d["kind"] == "jsonl" and d["count"] == 4 and d["entries"][1]["event"] == "start"
    assert authed.get("/api/logs?file=activity&stage=llm").json()["count"] == 2
    assert authed.get("/api/logs?file=activity&symbol=bel").json()["entries"][0]["rule"] == "MAX_TRADES_PER_DAY"
    assert authed.get("/api/logs?file=activity&level=WARNING").json()["count"] == 1
    assert "activity" in d["files"]


def _events(body: str) -> list[tuple[str, str]]:
    out = []
    for block in body.split("\n\n"):
        lines = block.strip().splitlines()
        ev = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
        if ev:
            out.append((ev, next((ln[6:] for ln in lines if ln.startswith("data: ")), "")))
    return out


def test_webui_stream_emits_activity_events(authed: TestClient, web_env: dict[str, Any], monkeypatch: Any) -> None:
    from webui import stream as stream_mod

    folder = web_env["log_dir"] / DAY.isoformat()
    path = folder / "activity.jsonl"
    assert not path.exists()  # file appears only after the stream started (missing file handled)
    real_poll = stream_mod.poll_once
    state = {"n": 0}

    def poll_and_write(c: Any, st: Any) -> Any:
        state["n"] += 1
        if state["n"] == 1:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(_activity(stage="cycle", message="intraday_cycle start", job="intraday_cycle") + "\n")
                fh.write(_activity(stage="llm", message="ITC meta start", symbol="ITC", agent="meta",
                                   event="start", model="qwen3:8b"))  # no newline yet: partial line is buffered
        elif state["n"] == 2:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n")
            with open(folder / "engine.log", "a", encoding="utf-8") as fh:
                fh.write("2026-11-25 14:00:01.000 +05:30 | INFO     | control.engine:_fire:89 - [engine] tick\n")
        return real_poll(c, st)

    monkeypatch.setattr(stream_mod, "poll_once", poll_and_write)
    authed.app.state.ctx.stream_max_iterations = 3
    with authed.stream("GET", "/api/stream") as r:
        body = r.read().decode()
    events = _events(body)
    acts = [json.loads(d)["entries"] for e, d in events if e == "activity"]
    assert [a["message"] for batch in acts for a in batch] == ["intraday_cycle start", "ITC meta start"]
    assert acts[1][0]["model"] == "qwen3:8b"
    logs = [json.loads(d)["entries"] for e, d in events if e == "log"]
    assert logs and logs[0][0]["message"] == "[engine] tick"


# =========================================================================== scan now


@pytest.mark.skipif(not hasattr(CommandType, "SCAN_NOW"), reason="CommandType.SCAN_NOW not deployed yet (engine side)")
def test_webui_scan_now_enqueues_command(authed: TestClient, web_env: dict[str, Any]) -> None:
    r = authed.post("/api/control", json={"action": "scan_now", "mode": "swing", "use_llm": False})
    assert r.status_code == 200, r.text
    assert r.json()["type"] == "SCAN_NOW"
    r2 = authed.post("/api/control", json={"action": "scan_now"})
    assert r2.status_code == 200
    assert authed.post("/api/control", json={"action": "scan_now", "mode": "WEEKLY"}).status_code == 422
    with web_env["factory"]() as s:
        cmds = list(s.execute(select(ControlCommand).order_by(ControlCommand.id)).scalars())
    assert [c.command_type for c in cmds] == ["SCAN_NOW", "SCAN_NOW"]
    set_fields = [{k: v for k, v in c.payload_json.items() if v is not None} for c in cmds]  # optional extras = None
    assert set_fields[0] == {"mode": "SWING", "use_llm": False}
    assert set_fields[1] == {"mode": "INTRADAY", "use_llm": True}
    st = authed.get(f"/api/commands/{cmds[0].id}").json()
    assert st["status"] == "PENDING" and st["finished"] is False


def test_webui_command_running_status(authed: TestClient, web_env: dict[str, Any]) -> None:
    with web_env["factory"]() as s:
        row = ControlCommand(created_at=NOW - timedelta(seconds=30), command_type="ANALYZE_NOW", payload_json={},
                             status="RUNNING", requested_by="admin", picked_at=NOW - timedelta(seconds=28))
        done = ControlCommand(created_at=NOW, command_type="ANALYZE_NOW", payload_json={}, status="DONE",
                              requested_by="admin", completed_at=NOW,
                              result_json={"ok": True, "message": "ITC INTRADAY: no trade - no strategy setup right "
                                                                  "now (analysis only)"})
        s.add_all([row, done])
        s.commit()
        ids = (row.id, done.id)
    running = authed.get(f"/api/commands/{ids[0]}").json()
    assert running["status"] == "RUNNING" and running["finished"] is False and running["ok"] is None
    finished = authed.get(f"/api/commands/{ids[1]}").json()
    assert finished["ok"] is True and finished["message"].startswith("ITC INTRADAY: no trade")
