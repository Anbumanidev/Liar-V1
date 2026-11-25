"""Web dashboard batches: funding module (fake) vs fallback, batch detail, top-up preview."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def test_webui_batches_fallback_without_funding_module(authed: TestClient, no_optional_modules: None) -> None:
    r = authed.get("/api/batches")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["source"] == "fallback" and b["current_batch_name"] == "Batch 2"
    assert b["default_starting_capital"] == 1000.0 and b["suggested_name"] == "Batch 3"
    first, second = b["batches"]
    assert first["name"] == "Batch 1" and first["top_up"] == 1000.0 and first["carried_in"] is None
    assert first["equity"] == 995.28 and first["trades"] == 1 and first["losses"] == 1
    # second batch starts with 1000 after the first ended with 995.28 -> 4.72 added
    assert second["carried_in"] == 995.28 and second["top_up"] == 4.72
    assert second["trades"] == 2 and second["wins"] == 1 and second["win_rate"] == 50.0
    assert b["total_deposited"] == 1004.72 and b["net_invested"] == 1004.72
    assert b["current_capital"] == second["equity"]
    assert b["overall_pnl"] == round(second["equity"] - 1004.72, 2)
    assert [e["kind"] for e in b["events"]] == ["INITIAL", "TOP_UP"]


def test_webui_batches_with_fake_funding(authed: TestClient, fake_module: Any) -> None:
    summary = {"total_deposited": 1004.72, "total_withdrawn": 0.0, "net_invested": 1004.72, "current_capital": 1010.0,
               "overall_pnl": 5.28, "overall_pnl_pct": 0.53,
               "batches": [{"batch_id": 1, "name": "Batch 1", "status": "CLOSED", "starting_capital": 1000.0,
                            "top_up": 1000.0, "carried_in": None, "equity": 995.28, "realized_net": -4.72,
                            "unrealized": 0.0, "pnl": -4.72, "pnl_pct": -0.47, "trades": 1, "wins": 0, "losses": 1,
                            "win_rate": 0.0, "started_at": None, "ended_at": None}],
               "events": [{"ts": None, "batch_id": 1, "kind": "INITIAL", "amount": 1000.0, "note": ""}]}
    fake_module("capital.funding", funding_summary=lambda s: summary,
                topup_preview=lambda s, capital: {"current_batch_id": 2, "current_equity": 995.28,
                                                  "open_positions": 1, "starting_capital": capital,
                                                  "top_up": round(capital - 995.28, 2), "action": "ADD",
                                                  "message": "You need to add ₹4.72"})
    b = authed.get("/api/batches").json()
    assert b["source"] == "funding" and b["batches"] == summary["batches"] and b["overall_pnl"] == 5.28
    p = authed.get("/api/batches/topup-preview?capital=1000").json()
    assert p["top_up"] == 4.72 and p["action"] == "ADD"
    d = authed.get("/api/batches/1").json()
    assert d["funding"]["name"] == "Batch 1"


def test_webui_topup_preview_not_available(authed: TestClient, no_optional_modules: None) -> None:
    r = authed.get("/api/batches/topup-preview?capital=1000")
    assert r.status_code == 501
    est = r.json()["estimate"]
    assert r.json()["detail"] == "not available yet" and est["source"] == "estimate"
    assert est["action"] in ("ADD", "WITHDRAW") and est["open_positions"] == 1
    assert authed.get("/api/batches/topup-preview?capital=-5").status_code == 422


def test_webui_batch_detail(authed: TestClient, web_env: dict[str, Any], no_optional_modules: None) -> None:
    d = authed.get(f"/api/batches/{web_env['ids']['batch']}").json()
    assert d["batch"]["name"] == "Batch 2" and d["capital"]["starting_capital"] == 1000.0
    assert d["totals"]["trades"] == 2 and d["totals"]["net"] == 10.2
    assert {r["strategy"] for r in d["by_strategy"]} == {"orb", "ema"}
    assert d["by_mode"][0]["mode"] == "INTRADAY" and d["by_mode"][0]["trades"] == 2
    curve = d["equity_curve"]
    times = [p["time"] for p in curve]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert curve[0]["equity"] == 1000.0 and curve[-1]["source"] == "LIVE"
    assert d["funding"] is None
    assert authed.get("/api/batches/999").status_code == 404
