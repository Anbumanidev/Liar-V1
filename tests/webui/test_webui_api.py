"""Web dashboard read API: overview, today, journal, trade detail, decisions, approvals, health, logs."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.webui.webui_kit import DAY


def test_webui_overview(authed: TestClient, web_env: dict[str, Any], no_optional_modules: None) -> None:
    r = authed.get("/api/overview")
    assert r.status_code == 200, r.text
    o = r.json()
    assert o["paper_only"] is True
    assert o["engine"]["state"] == "RUNNING" and o["engine"]["alive"] is True
    assert o["engine"]["heartbeat_age_s"] == 5.0
    assert o["batch"]["name"] == "Batch 2" and o["approvals_pending"] == 1
    [pos] = o["positions"]
    assert pos["symbol"] == "SBIN" and pos["ltp"] == 812.0 and pos["ltp_source"] == "live"
    assert pos["unrealized"] == round((812.0 - 800.0) * 4 - 1.2, 2)
    assert 0 < pos["progress"]["fraction"] < 1 and pos["progress"]["entry_pct"] == 25.0
    cap = o["capital"]
    assert cap["equity"] == round(600 + 400 + 10 + pos["unrealized"], 2) and cap["tier"] == "NORMAL"
    today = o["today"]
    assert today["realized"] == round(3.8 + 8.2 - 1.8, 2) and today["trades"] == 2
    assert today["wins"] == 1 and today["losses"] == 1 and today["total"] == round(today["realized"] + 46.8, 2)
    assert o["overall"]["source"] == "fallback"
    assert o["autonomy"]["value"] == "PER_MODE" and len(o["autonomy"]["per_mode"]) == 4
    assert o["market"]["open"] is True


def test_webui_overview_uses_funding_module(authed: TestClient, fake_module: Any) -> None:
    fake_module("capital.funding", funding_summary=lambda s: {
        "total_deposited": 1004.72, "total_withdrawn": 0.0, "net_invested": 1004.72, "current_capital": 1056.0,
        "overall_pnl": 51.28, "overall_pnl_pct": 5.1, "batches": [], "events": []})
    o = authed.get("/api/overview").json()["overall"]
    assert o == {"source": "funding", "total_deposited": 1004.72, "total_withdrawn": 0.0, "net_invested": 1004.72,
                 "current_capital": 1056.0, "overall_pnl": 51.28, "overall_pnl_pct": 5.1}


def test_webui_today_groups_legs_per_position(authed: TestClient, web_env: dict[str, Any]) -> None:
    r = authed.get(f"/api/today?date={DAY.isoformat()}")
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["is_today"] is True
    rows = {row["symbol"]: row for row in t["trades"]}
    assert set(rows) == {"ITC", "BEL"}
    itc = rows["ITC"]
    assert itc["legs"] == 2 and itc["qty"] == 2 and itc["net"] == 12.0 and itc["charges"] == 1.0
    assert itc["exit_price"] == 306.5 and itc["result"] == "WIN" and itc["r_multiple"] == 3.25
    assert itc["exit_reason_labels"] == ["Partial profit (T1)", "Trailing stop"]
    assert itc["initial_sl"] == 298.0 and itc["target_1"] == 304.0 and itc["holding_minutes"] == 58
    assert rows["BEL"]["exit_reason"] == "Stop-loss hit" and rows["BEL"]["result"] == "LOSS"
    s = t["summary"]
    assert s["net"] == 10.2 and s["wins"] == 1 and s["losses"] == 1 and s["best"]["symbol"] == "ITC"
    assert s["worst"]["symbol"] == "BEL" and s["open"] == 1
    f = t["funnel"]
    assert f["counts"]["TOTAL"] == 6 and f["counts"]["FILTERED"] == 1 and f["counts"]["EXECUTED"] == 2
    stages = {st["key"]: st["count"] for st in f["stages"]}
    assert stages == {"analysed": 6, "passed_filter": 5, "trade_signal": 4, "risk_rejected": 1, "queued": 1,
                      "executed": 2}
    assert f["risk_rules_top"] == [{"rule": "MAX_TRADES_PER_DAY", "count": 1}]
    assert f["filter_reasons_top"][0]["reason"] == "confluence below minimum"
    kinds = {e["kind"] for e in t["timeline"]}
    assert {"ENTRY", "EXIT", "SL_MOVE", "RISK", "ORDER", "APPROVAL"} <= kinds
    assert [e["ts"] for e in t["timeline"]] == sorted(e["ts"] for e in t["timeline"])
    assert t["equity"][0]["equity"] == 1000.0 and t["equity"][-1]["equity"] == 1010.0
    assert len(t["open_positions"]) == 1
    other = authed.get("/api/today?date=2026-11-12").json()
    assert [row["symbol"] for row in other["trades"]] == ["HDFCBANK"] and other["is_today"] is False
    assert authed.get("/api/today?date=garbage").status_code == 422


def test_webui_trade_journal_filters(authed: TestClient) -> None:
    j = authed.get("/api/trades").json()
    assert [r["symbol"] for r in j["rows"]] == ["BEL", "ITC", "HDFCBANK"]
    assert j["totals"]["trades"] == 3 and j["totals"]["net"] == round(12.0 - 1.8 - 12.0, 2)
    assert {b["name"] for b in j["batches"]} == {"Batch 1", "Batch 2"}
    losses = authed.get("/api/trades?result=LOSS").json()
    assert {r["symbol"] for r in losses["rows"]} == {"BEL", "HDFCBANK"}
    ranged = authed.get("/api/trades?from=2026-11-25&to=2026-11-25&mode=INTRADAY").json()
    assert {r["symbol"] for r in ranged["rows"]} == {"BEL", "ITC"}
    assert authed.get("/api/trades?result=MAYBE").status_code == 422


def test_webui_trade_detail_without_postmortem(authed: TestClient, web_env: dict[str, Any],
                                               no_optional_modules: None) -> None:
    pid = web_env["ids"]["itc"]
    r = authed.get(f"/api/trades/{pid}")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["postmortem"] is None and d["postmortem_available"] is False
    assert d["postmortem_error"]["status"] == 501
    assert len(d["legs"]) == 2 and [o["purpose"] for o in d["orders"]] == ["ENTRY", "PARTIAL_EXIT", "EXIT"]
    assert d["row"]["net"] == 12.0 and d["position"]["sl_history"][1]["new"] == 300.0
    assert d["chart"]["timeframe"] == "5m" and d["decision"]["id"] == web_env["ids"]["d_exec"]
    assert authed.get("/api/trades/9999").status_code == 404


def test_webui_trade_detail_with_postmortem(authed: TestClient, web_env: dict[str, Any], fake_module: Any) -> None:
    seen: dict[str, Any] = {}

    def analyze_position(session: Any, position_id: int, data: Any = None) -> dict[str, Any]:
        seen["data"] = data
        return {"position_id": position_id, "result": "WIN", "story": ["entered", "trailed"], "mfe": float("nan"),
                "diagnosis": [{"code": "LATE", "severity": "low", "title": "t", "detail": "d", "fix": "f",
                               "fixed_by": None}], "chart": {"timeframe": "1m", "from": "2026-11-25T09:50:00+05:30",
                                                             "to": "2026-11-25T11:30:00+05:30"}}

    fake_module("analytics.postmortem", analyze_position=analyze_position)
    d = authed.get(f"/api/trades/{web_env['ids']['itc']}").json()
    assert d["postmortem_available"] is True and d["postmortem"]["story"] == ["entered", "trailed"]
    assert d["postmortem"]["mfe"] is None  # NaN sanitised to null
    assert d["chart"]["timeframe"] == "1m" and seen["data"] is not None


def test_webui_postmortem_day(authed: TestClient, fake_module: Any, monkeypatch: Any) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "analytics.postmortem", None)
    r = authed.get("/api/postmortem/day")
    assert r.status_code == 501 and r.json()["detail"] == "not available yet"
    fake_module("analytics.postmortem", analyze_day=lambda s, day, batch_id=None, data=None: {
        "day": day.isoformat(), "trades": [], "summary": {"trades": 0}, "top_issues": [
            {"code": "SL_TIGHT", "count": 2, "title": "Stops too tight", "fix": "widen"}]})
    r = authed.get("/api/postmortem/day?date=2026-11-25")
    assert r.status_code == 200 and r.json()["top_issues"][0]["count"] == 2


def test_webui_decision_feed_and_detail(authed: TestClient, web_env: dict[str, Any]) -> None:
    feed = authed.get("/api/decisions?limit=50").json()
    rows = feed["rows"]
    assert len(rows) == 6 and rows[0]["id"] > rows[-1]["id"]
    by_action = {r["final_action"]: r for r in rows}
    assert by_action["FILTERED"]["reason"] == "confluence 41.0 < 60.0 (grade C)"
    assert by_action["RISK_REJECTED"]["reason"].startswith("MAX_TRADES_PER_DAY")
    assert by_action["NO_TRADE"]["reason"] == "charges eat the edge"
    assert all(r["final_action"] != "FILTERED" for r in
               authed.get("/api/decisions?hide_filtered=true").json()["rows"])
    newer = authed.get(f"/api/decisions?since_id={rows[1]['id']}").json()["rows"]
    assert [r["id"] for r in newer] == [rows[0]["id"]]
    assert {r["symbol"] for r in authed.get("/api/decisions?action=EXECUTED").json()["rows"]} == {"ITC", "BEL"}
    d = authed.get(f"/api/decisions/{web_env['ids']['d_exec']}").json()
    assert d["symbol"] == "ITC" and d["confluence_detail"]["components"] == {"trend": 20.0, "volume": 12.5}
    assert d["vote"]["failed_agents"] == ["news_sentiment"] and d["vote"]["contributions"]["technical"] == 1.2
    agents = {a["agent"]: a for a in d["agents"]}
    assert agents["technical"]["key_findings"] == ["EMA stack up", "RSI 61"] and agents["news_sentiment"]["error"]
    assert "data_used" not in str(agents["technical"])
    assert d["debate"]["bull"]["conviction"] == 80 and d["debate"]["bear"]["stance"] == "Wait"
    assert d["meta"]["reasons_for"] == ["trend up"] and d["meta"]["reasons_against"] == ["VIX rising"]
    assert d["risk"]["approved"] is True and len(d["risk"]["checks"]) == 2
    assert d["risk"]["charges_estimate_total"] == 0.12
    assert d["orders"][0]["purpose"] == "ENTRY" and d["positions"][0]["status"] == "CLOSED"
    rej = authed.get(f"/api/decisions/{web_env['ids']['d_rej']}").json()
    assert rej["risk"]["failed"][0]["rule"] == "MAX_TRADES_PER_DAY"
    assert authed.get("/api/decisions/99999").status_code == 404


def test_webui_approvals_list(authed: TestClient, web_env: dict[str, Any]) -> None:
    a = authed.get("/api/approvals").json()
    assert a["count"] == 1
    [item] = a["items"]
    assert item["symbol"] == "TCS" and item["bull_points"] == ["breakout on volume"]
    assert item["bear_points"] == ["near resistance"] and item["tax"]["scenarios"][0]["net_after_tax"] == 92.6
    assert item["risk_amount"] == 60.0 and item["rr"] == 2.0 and item["charges_estimate"] == 0.12
    assert item["seconds_left"] == 3 * 3600 and item["expired"] is False


def test_webui_health_and_logs(authed: TestClient, no_optional_modules: None) -> None:
    h = authed.get("/api/health").json()
    assert h["diagnostics"] is None and h["diagnostics_error"]["status"] == 501
    assert h["engine"]["state"] == "RUNNING" and h["errors_today"] == 2
    assert h["errors"][0]["count"] == 2 and h["errors"][0]["component"] == "data"
    lg = authed.get("/api/logs?file=engine&lines=50").json()
    assert lg["exists"] and lg["count"] == 3
    assert lg["entries"][1]["level"] == "ERROR" and "Traceback" in lg["entries"][1]["message"]
    only_err = authed.get("/api/logs?file=engine&level=ERROR").json()
    assert only_err["count"] == 1
    found = authed.get("/api/logs?file=engine&q=done").json()
    assert found["count"] == 1
    js = authed.get("/api/logs?file=decisions").json()
    assert js["kind"] == "jsonl" and js["entries"][0]["symbol"] == "TCS" and js["entries"][1]["_raw"] == "not json"
    assert authed.get("/api/logs?file=../../secrets").status_code == 422
    missing = authed.get("/api/logs?file=risk&date=2026-01-01").json()
    assert missing["exists"] is False and missing["entries"] == []


def test_webui_health_with_diagnostics(authed: TestClient, fake_module: Any) -> None:
    fake_module("analytics.diagnostics", diagnose=lambda s, day, log_dir: {
        "day": day.isoformat(), "llm": {"calls": 10, "failed": 1}, "findings": [
            {"severity": "high", "title": "LLM slow", "detail": "p90 60s", "action": "use smaller model"}]})
    h = authed.get("/api/health?date=2026-11-25").json()
    assert h["diagnostics_available"] is True and h["diagnostics"]["findings"][0]["title"] == "LLM slow"


def test_webui_quotes_and_watchlist(authed: TestClient) -> None:
    q = authed.get("/api/quotes?symbols=SBIN,ITC,NOPE").json()["quotes"]
    assert q["SBIN"]["ltp"] == 812.0 and q["SBIN"]["change_pct"] == 1.5
    assert q["ITC"]["change"] == -1.0 and q["NOPE"] is None
    assert authed.get("/api/quotes?symbols=bad sym!").status_code == 422
    w = authed.get("/api/watchlist").json()["rows"]
    assert w[0]["symbol"] == "SBIN" and w[0]["open_position"] is True
    assert {r["symbol"] for r in w} >= {"TCS", "ITC", "BEL", "INFY"}


def test_webui_analyzed_action_reason(authed: TestClient, web_env: dict[str, Any]) -> None:
    from storage.models import DecisionRow
    from tests.webui.webui_kit import NOW

    with web_env["factory"]() as s:
        s.add(DecisionRow(batch_id=web_env["ids"]["batch"], symbol="LT", mode="SWING", decision="NO_TRADE",
                          confidence=0, meta_json={"why_not": "no strategy signal"}, final_action="ANALYZED",
                          context_json={}, created_at=NOW))
        s.add(DecisionRow(batch_id=web_env["ids"]["batch"], symbol="LT", mode="SWING", decision="BUY",
                          confidence=70, final_action="ANALYZED", context_json={}, created_at=NOW))
        s.commit()
    rows = authed.get("/api/decisions?action=ANALYZED").json()["rows"]
    assert [r["reason"] for r in rows] == ["analysis only (Analyze now) - not routed for execution",
                                           "no strategy signal"]
    assert authed.get("/api/today").json()["funnel"]["counts"]["ANALYZED"] == 2
