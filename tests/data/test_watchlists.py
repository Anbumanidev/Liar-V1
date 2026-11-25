"""Follow-up lists (data.watchlists): CRUD, validation, upsert, audit, trading symbols, plans."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from core.clock import IST
from core.logging import set_log_dir
from data import watchlists as wl
from storage.models import AuditLog, Watchlist, WatchlistItem

NOW = datetime(2026, 11, 25, 15, 40, tzinfo=IST)
TODAY = date(2026, 11, 25)


def _audit_actions(session) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    return [(a.actor, a.action) for a in session.execute(select(AuditLog).order_by(AuditLog.id)).scalars()]


# --------------------------------------------------------------------------- lists


def test_create_list_validates_name_and_kind(session) -> None:  # type: ignore[no-untyped-def]
    out = wl.create_watchlist(session, "  Banks   to   watch ", now=NOW, description="PSU banks")
    assert out["name"] == "Banks to watch" and out["kind"] == "MANUAL" and out["items"] == 0
    assert out["use_for_trading"] is False and out["archived"] is False
    assert out["created_at"] == NOW.isoformat()
    with pytest.raises(ValueError, match="already exists"):
        wl.create_watchlist(session, "banks TO watch", now=NOW)          # case-insensitive unique
    with pytest.raises(ValueError, match="empty"):
        wl.create_watchlist(session, "   ", now=NOW)
    with pytest.raises(ValueError, match="too long"):
        wl.create_watchlist(session, "x" * 81, now=NOW)
    with pytest.raises(ValueError, match="kind"):
        wl.create_watchlist(session, "Other", now=NOW, kind="WEIRD")
    assert len(wl.create_watchlist(session, "y" * 80, now=NOW)["name"]) == 80


def test_default_list_is_created_once(session) -> None:  # type: ignore[no-untyped-def]
    a = wl.get_or_create_default(session, now=NOW)
    b = wl.get_or_create_default(session, now=NOW)
    assert a["id"] == b["id"] and a["name"] == "Follow-up" and a["kind"] == "MANUAL"
    assert session.query(Watchlist).count() == 1


def test_update_archive_and_delete(session) -> None:  # type: ignore[no-untyped-def]
    a = wl.create_watchlist(session, "A", now=NOW)
    wl.create_watchlist(session, "B", now=NOW)
    wl.add_item(session, a["id"], "ITC", now=NOW)
    with pytest.raises(ValueError, match="already exists"):
        wl.update_watchlist(session, a["id"], now=NOW, name="b")
    out = wl.update_watchlist(session, a["id"], now=NOW, name="Alpha", description="d", use_for_trading=True)
    assert out["name"] == "Alpha" and out["use_for_trading"] is True and out["items"] == 1
    wl.update_watchlist(session, a["id"], now=NOW, archived=True)
    assert [x["name"] for x in wl.list_watchlists(session)] == ["B"]
    assert {x["name"] for x in wl.list_watchlists(session, include_archived=True)} == {"Alpha", "B"}
    wl.delete_watchlist(session, a["id"], now=NOW)
    assert session.get(Watchlist, a["id"]) is None
    assert session.query(WatchlistItem).count() == 0                     # items go with the list
    assert wl.get_watchlist(session, a["id"]) is None
    with pytest.raises(ValueError, match="not found"):
        wl.update_watchlist(session, 999, now=NOW, name="x")
    with pytest.raises(ValueError, match="not found"):
        wl.delete_watchlist(session, 999, now=NOW)


# --------------------------------------------------------------------------- items


def test_add_item_validates_and_upserts(session) -> None:  # type: ignore[no-untyped-def]
    lst = wl.get_or_create_default(session, now=NOW)
    first = wl.add_item(session, lst["id"], " itc ", now=NOW, note="results next week", added_price=310.5,
                        mode="intraday", source="ANALYSIS", decision_id=7)
    assert first["symbol"] == "ITC" and first["mode"] == "INTRADAY" and first["source"] == "ANALYSIS"
    assert first["added_price"] == 310.5 and first["status"] == "ACTIVE" and first["decision_id"] == 7
    later = datetime(2026, 11, 26, 10, 0, tzinfo=IST)
    again = wl.add_item(session, lst["id"], "ITC", now=later, note="breakout watch", added_price=320.0,
                        plan={"trigger": 312.4, "score": 70})
    assert again["id"] == first["id"]                                     # upsert, not a duplicate
    assert again["note"] == "breakout watch" and again["plan"] == {"trigger": 312.4, "score": 70}
    assert again["added_at"] == NOW.isoformat() and again["added_price"] == 310.5   # first add kept
    assert session.query(WatchlistItem).count() == 1
    assert wl.add_item(session, lst["id"], "M&M", now=NOW)["symbol"] == "M&M"
    assert wl.add_item(session, lst["id"], "BAJAJ-AUTO", now=NOW)["symbol"] == "BAJAJ-AUTO"
    for bad in ("", "BAD SYMBOL", "X" * 33, "ITC;DROP", None):
        with pytest.raises(ValueError, match="invalid symbol"):
            wl.add_item(session, lst["id"], bad, now=NOW)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source"):
        wl.add_item(session, lst["id"], "TCS", now=NOW, source="RUMOUR")
    with pytest.raises(ValueError, match="mode"):
        wl.add_item(session, lst["id"], "TCS", now=NOW, mode="SCALP")
    with pytest.raises(ValueError, match="not found"):
        wl.add_item(session, 12345, "TCS", now=NOW)
    got = wl.get_watchlist(session, lst["id"])
    assert got is not None and got["items"][0]["symbol"] in {"M&M", "BAJAJ-AUTO"}   # newest first
    assert [i["symbol"] for i in got["items"]][-1] == "ITC"


def test_remove_item_by_id_or_symbol(session) -> None:  # type: ignore[no-untyped-def]
    lst = wl.create_watchlist(session, "L", now=NOW)
    a = wl.add_item(session, lst["id"], "ITC", now=NOW)
    wl.add_item(session, lst["id"], "TCS", now=NOW)
    wl.remove_item(session, lst["id"], a["id"], now=NOW)
    wl.remove_item(session, lst["id"], "tcs", now=NOW)
    assert wl.get_watchlist(session, lst["id"])["items"] == []  # type: ignore[index]
    with pytest.raises(ValueError, match="not found in list"):
        wl.remove_item(session, lst["id"], "INFY", now=NOW)
    other = wl.create_watchlist(session, "Other", now=NOW)
    b = wl.add_item(session, other["id"], "SBIN", now=NOW)
    with pytest.raises(ValueError, match="not found in list"):
        wl.remove_item(session, lst["id"], b["id"], now=NOW)               # id of another list


def test_every_change_is_audited_and_logged(session, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    set_log_dir(tmp_path)
    lst = wl.create_watchlist(session, "Audit me", now=NOW, actor="owner")
    wl.add_item(session, lst["id"], "ITC", now=NOW, actor="owner")
    wl.update_watchlist(session, lst["id"], now=NOW, use_for_trading=True, actor="owner")
    wl.remove_item(session, lst["id"], "ITC", now=NOW, actor="owner")
    wl.delete_watchlist(session, lst["id"], now=NOW, actor="dashboard")
    assert _audit_actions(session) == [("owner", "WATCHLIST_CREATE"), ("owner", "WATCHLIST_ADD"),
                                       ("owner", "WATCHLIST_UPDATE"), ("owner", "WATCHLIST_REMOVE"),
                                       ("dashboard", "WATCHLIST_DELETE")]
    path = tmp_path / "2026-11-25" / "activity.jsonl"
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    msgs = [e["message"] for e in lines if e["stage"] == "watchlist"]
    assert len(msgs) == 5 and "ITC added to 'Audit me'" in msgs[1] and "deleted" in msgs[-1]
    wl.update_watchlist(session, wl.create_watchlist(session, "x", now=NOW)["id"], now=NOW)   # no change: no row
    assert _audit_actions(session)[-1] == ("owner", "WATCHLIST_CREATE")


# --------------------------------------------------------------------------- engine-facing


def _plan(session, for_date: date, rows: list[tuple[str, float]]) -> dict:  # type: ignore[no-untyped-def,type-arg]
    cands = [{"symbol": s, "setup": "BREAKOUT", "direction": "LONG", "trigger": 100.0, "stop": 98.0,
              "target": 104.0, "score": sc, "close": 99.5, "affordable": True} for s, sc in rows]
    return wl.save_plan(session, for_date, cands, now=NOW)


def test_trading_symbols_plan_first_then_trading_lists(session) -> None:  # type: ignore[no-untyped-def]
    plan = _plan(session, TODAY, [("SBIN", 60.0), ("ITC", 80.0), ("TCS", 70.0)])
    _plan(session, date(2026, 11, 26), [("WIPRO", 90.0)])                  # tomorrow's plan: not today
    trade = wl.create_watchlist(session, "Trade these", now=NOW, use_for_trading=True)
    wl.add_item(session, trade["id"], "INFY", now=NOW)
    wl.add_item(session, trade["id"], "ITC", now=NOW)                       # duplicate of the plan
    watch = wl.create_watchlist(session, "Just watching", now=NOW)
    wl.add_item(session, watch["id"], "HDFCBANK", now=NOW)                  # use_for_trading=False
    gone = wl.create_watchlist(session, "Old", now=NOW, use_for_trading=True)
    wl.add_item(session, gone["id"], "LT", now=NOW)
    wl.update_watchlist(session, gone["id"], now=NOW, archived=True)       # archived
    items = {i["symbol"]: i for i in plan["items"]}
    wl.set_item_status(session, items["TCS"]["id"], "INVALIDATED", "gapped past", now=NOW)
    wl.set_item_status(session, items["SBIN"]["id"], "TRIGGERED", "above trigger", now=NOW)
    assert wl.trading_symbols(session, today=TODAY) == ["ITC", "SBIN", "INFY"]
    assert wl.trading_symbols(session, today=date(2026, 11, 26)) == ["WIPRO", "INFY", "ITC"]
    assert ("system", "WATCHLIST_STATUS") in _audit_actions(session)
    with pytest.raises(ValueError, match="status"):
        wl.set_item_status(session, items["TCS"]["id"], "DONE", None, now=NOW)


def test_record_analysis_updates_every_list(session) -> None:  # type: ignore[no-untyped-def]
    a = wl.create_watchlist(session, "A", now=NOW)
    b = wl.create_watchlist(session, "B", now=NOW)
    wl.add_item(session, a["id"], "ITC", now=NOW)
    wl.add_item(session, b["id"], "ITC", now=NOW)
    wl.add_item(session, b["id"], "TCS", now=NOW)
    at = datetime(2026, 11, 26, 10, 5, tzinfo=IST)
    assert wl.record_analysis(session, "itc", decision_id=42, summary="ITC INTRADAY: no trade", now=at) == 2
    assert wl.record_analysis(session, "INFY", decision_id=1, summary="x", now=at) == 0
    assert wl.record_analysis(session, "not a symbol", decision_id=1, summary="x", now=at) == 0
    rows = {(i["watchlist_id"], i["symbol"]): i for lst in (a, b) for i in wl.get_watchlist(session, lst["id"])["items"]}
    assert rows[(a["id"], "ITC")]["last_analysis_id"] == 42
    assert rows[(b["id"], "ITC")]["last_summary"] == "ITC INTRADAY: no trade"
    assert rows[(b["id"], "ITC")]["last_analysis_at"] == at.isoformat()
    assert rows[(b["id"], "TCS")]["last_analysis_at"] is None
    assert wl.followed_symbols(session) == {"ITC", "TCS"}


def test_latest_plan_and_replacing_a_plan(session) -> None:  # type: ignore[no-untyped-def]
    assert wl.latest_plan(session, on_or_after=TODAY) is None
    old = _plan(session, date(2026, 11, 20), [("LT", 50.0)])
    first = _plan(session, date(2026, 11, 26), [("ITC", 60.0), ("TCS", 75.0)])
    assert first["name"] == "Plan 26 Nov 2026" and first["kind"] == "NEXT_DAY" and first["use_for_trading"]
    assert first["for_date"] == "2026-11-26"
    assert {i["source"] for i in first["items"]} == {"EOD_PLAN"} and {i["mode"] for i in first["items"]} == {"INTRADAY"}
    assert session.get(Watchlist, old["id"]).archived is True               # past plans archived
    latest = wl.latest_plan(session, on_or_after=date(2026, 11, 26))
    assert latest is not None and [i["symbol"] for i in latest["items"]] == ["TCS", "ITC"]   # by score
    assert latest["items"][0]["plan"]["score"] == 75.0
    again = _plan(session, date(2026, 11, 26), [("SBIN", 66.0)])           # re-run: replace, same list
    assert again["id"] == first["id"] and [i["symbol"] for i in again["items"]] == ["SBIN"]
    assert session.query(Watchlist).filter(Watchlist.kind == "NEXT_DAY").count() == 2
    assert wl.latest_plan(session, on_or_after=date(2026, 11, 27)) is None
