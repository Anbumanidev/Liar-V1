"""Next-day plan (15:40), opening check (09:20), follow-up priority in INTRADAY cycles,
Scan now on chosen stocks / a follow-up list, and recording analyses on follow-up items.

Uses the real bootstrap + ReplayDataProvider (synthetic, consistent multi-timeframe data from
tests/engine/test_engine_layer.py) with a temp SQLite DB and temp logs. Nothing here may create
an order: the plan only changes which stocks are analysed first.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import func, select

from config.loader import update_settings
from control.bootstrap import bootstrap
from control.commands import enqueue_command, get_command
from control.processor import CommandProcessor
from core.clock import IST, SimulatedClock
from core.enums import CommandType, EngineState, Mode, Timeframe
from data import watchlists as wl
from data.calendar import load_calendar
from data.providers.replay_provider import ReplayDataProvider
from scheduler import jobs
from scheduler.plan import JOBS, DailyPlan
from storage.models import DecisionRow, Order, Watchlist, WatchlistItem

SYMS = ["INFY", "TCS", "ITC", "SBIN"]


def _synthetic(symbols: list[str]) -> dict[tuple[str, Timeframe], pd.DataFrame]:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "engine" / "test_engine_layer.py"
    spec = importlib.util.spec_from_file_location("_engine_layer_helpers_nd", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = mod  # type: ignore[union-attr]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.synthetic_frames(symbols)


def breakout_daily(end: date, scale: float = 1.0) -> pd.DataFrame:
    """Uptrend + tight 10-day base under the high; last day NR7 inside day (a clean BREAKOUT)."""
    days: list[date] = []
    d = end
    while len(days) < 100:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    c = [100 + 0.5 * i for i in range(90)] + [144.5, 145, 144.8, 145.2, 144.9, 145.1, 145.0, 145.3, 145.2, 145.25]
    h = [x + 1.0 for x in c]
    lo = [x - 1.0 for x in c]
    h[-1], lo[-1] = c[-1] + 0.2, c[-1] - 0.3
    v = [1e6] * 99 + [1.2e6]
    idx = pd.DatetimeIndex([datetime.combine(x, time(9, 15), tzinfo=IST) for x in days])
    return pd.DataFrame({"open": [c[0]] + c[:-1], "high": h, "low": lo, "close": c, "volume": v}, index=idx) * \
        pd.Series({"open": scale, "high": scale, "low": scale, "close": scale, "volume": 1.0})


def _svc(tmp_path: Path, at: datetime, frames: dict[tuple[str, Timeframe], pd.DataFrame] | None = None,
         changes: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
    clock = SimulatedClock(at)
    provider = ReplayDataProvider(frames or _synthetic(SYMS), clock, load_calendar())
    svc = bootstrap(process="test", clock=clock, db_url=f"sqlite:///{(tmp_path / 'nd.db').as_posix()}",
                    primary_provider=provider, llm_backend="rules", with_notifier=False, offline=True,
                    runtime_dir=tmp_path / "rt", log_dir=tmp_path / "logs", console_log=False)
    base = {"engine.next_day_universe": "CUSTOM", "universe.custom_symbols": list(SYMS)}
    svc.apply_settings(update_settings(svc.settings, {**base, **(changes or {})}))
    return svc


def _activity(tmp: Path, day: str) -> list[dict]:  # type: ignore[type-arg]
    path = tmp / "logs" / day / "activity.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def _orders(svc) -> int:  # type: ignore[no-untyped-def]
    with svc.session() as s:
        return int(s.scalar(select(func.count(Order.id))) or 0)


def _cmd(svc, ctype: CommandType, payload: dict) -> dict:  # type: ignore[no-untyped-def,type-arg]
    with svc.session() as s:
        row = enqueue_command(s, ctype, payload, actor="owner", batch_id=svc.ctx.batch_id, clock=svc.clock)
        s.commit()
        cid = row.id
    CommandProcessor(svc).process_pending()
    with svc.session() as s:
        row = get_command(s, cid)
        return {"status": row.status, **(row.result_json or {})}


def _save_plan(svc, for_date: date, rows: list[dict]) -> dict:  # type: ignore[no-untyped-def,type-arg]
    with svc.session() as s:
        out = wl.save_plan(s, for_date, rows, now=svc.clock.now())
        s.commit()
    return out


# --------------------------------------------------------------------------- schedule


def test_jobs_are_registered_and_scheduled(settings) -> None:  # type: ignore[no-untyped-def]
    assert {"next_day_plan", "opening_check"} <= set(JOBS)
    assert {"next_day_plan", "opening_check"} <= set(jobs.JOB_FUNCS)
    slots = {(s.job, s.at.strftime("%H:%M")) for s in DailyPlan(settings).slots()}
    assert ("next_day_plan", "15:40") in slots and ("opening_check", "09:20") in slots
    due = DailyPlan(settings).jobs_due(datetime(2026, 11, 25, 15, 39, tzinfo=IST),
                                       datetime(2026, 11, 25, 15, 46, tzinfo=IST))
    assert due == ["next_day_plan", "eod_report"]                            # plan before the e-mail
    off = update_settings(settings, {"engine.next_day_plan_enabled": False})
    assert not {"next_day_plan", "opening_check"} & {s.job for s in DailyPlan(off).slots()}


# --------------------------------------------------------------------------- 15:40 plan


def test_next_day_plan_creates_and_replaces_the_list(tmp_path: Path) -> None:
    frames = _synthetic(SYMS)
    frames[("INFY", Timeframe.D1)] = breakout_daily(date(2026, 11, 25))
    svc = _svc(tmp_path, datetime(2026, 11, 25, 15, 40, tzinfo=IST), frames)
    out = jobs.run_job(svc, "next_day_plan")
    assert "error" not in out and out["for_date"] == "2026-11-26" and out["universe"] == 4
    assert out["max_affordable_price"] == pytest.approx(500.0)
    picked = {c["symbol"]: c for c in out["candidates"]}
    assert picked["INFY"]["setup"] == "BREAKOUT" and picked["INFY"]["affordable"] is True
    with svc.session() as s:
        lists = list(s.execute(select(Watchlist).where(Watchlist.kind == "NEXT_DAY")).scalars())
        assert len(lists) == 1
        plan = lists[0]
        assert plan.name == "Plan 26 Nov 2026" and plan.for_date == date(2026, 11, 26) and plan.use_for_trading
        items = list(s.execute(select(WatchlistItem).where(WatchlistItem.watchlist_id == plan.id)).scalars())
    assert len(items) == out["selected"] >= 1 and {i.source for i in items} == {"EOD_PLAN"}
    infy = next(i for i in items if i.symbol == "INFY")
    assert infy.plan_json["trigger"] == picked["INFY"]["trigger"] and infy.plan_json["for_date"] == "2026-11-26"
    assert infy.mode == "INTRADAY" and infy.status == "ACTIVE"
    assert all(i.plan_json["score"] >= svc.settings.engine.next_day_min_score for i in items)
    msgs = [e["message"] for e in _activity(tmp_path, "2026-11-25") if e["stage"] == "plan"]
    assert any(m.startswith("next-day plan: ranking 4 CUSTOM stock(s) for Thu 26 Nov") for m in msgs)
    assert any(m.startswith("next-day plan: ") and "INFY breakout trigger" in m for m in msgs)
    assert any(m.startswith("plan INFY: BREAKOUT LONG trigger") for m in msgs)
    again = jobs.run_job(svc, "next_day_plan")                               # re-run replaces, never duplicates
    assert again["watchlist_id"] == out["watchlist_id"]
    with svc.session() as s:
        assert s.scalar(select(func.count(Watchlist.id)).where(Watchlist.kind == "NEXT_DAY")) == 1
        assert s.scalar(select(func.count(WatchlistItem.id))) == again["selected"]
    assert _orders(svc) == 0


def test_next_day_plan_respects_the_switch_and_min_score(tmp_path: Path) -> None:
    frames = _synthetic(SYMS)
    frames[("INFY", Timeframe.D1)] = breakout_daily(date(2026, 11, 25))
    svc = _svc(tmp_path, datetime(2026, 11, 25, 15, 40, tzinfo=IST), frames,
               {"engine.next_day_plan_enabled": False})
    assert "skipped" in jobs.run_job(svc, "next_day_plan")
    svc.apply_settings(update_settings(svc.settings, {"engine.next_day_plan_enabled": True,
                                                      "engine.next_day_min_score": 100.0}))
    out = jobs.run_job(svc, "next_day_plan")
    assert out["selected"] == 0 and out["candidates"] == []                  # an empty plan is still saved
    assert any("no stock qualified" in e["message"] for e in _activity(tmp_path, "2026-11-25"))


def test_unaffordable_candidates_only_fill_left_over_places(tmp_path: Path) -> None:
    frames = _synthetic(SYMS)
    frames[("INFY", Timeframe.D1)] = breakout_daily(date(2026, 11, 25))
    frames[("TCS", Timeframe.D1)] = breakout_daily(date(2026, 11, 25), scale=5.0)   # trigger ~730 > Rs 500
    svc = _svc(tmp_path, datetime(2026, 11, 25, 15, 40, tzinfo=IST), frames,
               {"engine.next_day_max_candidates": 1, "engine.next_day_min_score": 0.0})
    out = jobs.run_job(svc, "next_day_plan")
    assert [c["symbol"] for c in out["candidates"]] == ["INFY"]
    svc.apply_settings(update_settings(svc.settings, {"engine.next_day_max_candidates": 10}))
    flags = {c["symbol"]: c["affordable"] for c in jobs.run_job(svc, "next_day_plan")["candidates"]}
    assert flags["INFY"] is True and flags["TCS"] is False


# --------------------------------------------------------------------------- 09:20 opening check


def test_opening_verdict_long_and_short() -> None:
    long_ = {"direction": "LONG", "trigger": 100.0, "stop": 98.0, "atr": 2.0}
    assert jobs.opening_verdict(long_, 99.0)[0] == "ACTIVE"
    assert jobs.opening_verdict(long_, 101.0)[0] == "TRIGGERED"
    st, note = jobs.opening_verdict(long_, 102.5, 102.0)
    assert st == "INVALIDATED" and "don't chase" in note and "open 102.00" in note
    assert jobs.opening_verdict(long_, 97.9)[0] == "INVALIDATED"
    short = {"direction": "SHORT", "trigger": 100.0, "stop": 102.0, "atr": 2.0}
    assert jobs.opening_verdict(short, 101.0)[0] == "ACTIVE"
    assert jobs.opening_verdict(short, 99.0)[0] == "TRIGGERED"
    assert jobs.opening_verdict(short, 97.5)[0] == "INVALIDATED"
    assert "above the stop" in jobs.opening_verdict(short, 102.5)[1]


def test_opening_check_sets_statuses_and_trading_symbols(tmp_path: Path) -> None:
    svc = _svc(tmp_path, datetime(2026, 11, 26, 9, 20, tzinfo=IST))
    assert "skipped" in jobs.run_job(svc, "opening_check")                   # no plan for today yet
    px = {s: svc.data.quote(s).ltp for s in SYMS}

    def row(sym: str, score: float, direction: str, trigger: float, stop: float) -> dict:  # type: ignore[type-arg]
        return {"symbol": sym, "setup": "BREAKOUT", "direction": direction, "trigger": round(trigger, 2),
                "stop": round(stop, 2), "target": 0.0, "score": score, "atr": 2.0, "affordable": True,
                "close": px[sym]}

    _save_plan(svc, date(2026, 11, 26), [
        row("INFY", 80, "LONG", px["INFY"] - 0.5, px["INFY"] - 3),          # just past the trigger
        row("TCS", 70, "LONG", px["TCS"] + 2, px["TCS"] - 2),               # waiting
        row("ITC", 90, "LONG", px["ITC"] - 5, px["ITC"] - 8),               # 2.5 ATR past: don't chase
        row("SBIN", 60, "SHORT", px["SBIN"] - 3, px["SBIN"] - 1),           # short, price above its stop
    ])
    out = jobs.run_job(svc, "opening_check")
    assert out == {"checked": 4, "triggered": 1, "waiting": 1, "invalidated": 2, "no_price": 0}
    with svc.session() as s:
        st = {i.symbol: (i.status, i.status_note) for i in s.execute(select(WatchlistItem)).scalars()}
        assert wl.trading_symbols(s, today=date(2026, 11, 26)) == ["INFY", "TCS"]
    assert st["INFY"][0] == "TRIGGERED" and "past the trigger" in st["INFY"][1]
    assert st["TCS"][0] == "ACTIVE" and st["TCS"][1].startswith("waiting for trigger")
    assert st["ITC"][0] == "INVALIDATED" and "gapped past" in st["ITC"][1]
    assert st["SBIN"][0] == "INVALIDATED" and "above the stop" in st["SBIN"][1]
    msgs = [e["message"] for e in _activity(tmp_path, "2026-11-26")]
    assert any(m.startswith("opening check ITC BREAKOUT: INVALIDATED") for m in msgs)
    assert any(m.startswith("opening check: 4 plan stock(s) - 1 triggered, 1 waiting, 2 invalidated") for m in msgs)
    assert _orders(svc) == 0


# --------------------------------------------------------------------------- INTRADAY priority


class _Recorder:
    """Pipeline double with only analyze(): records the order the cycle analyses symbols in."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def analyze(self, symbol: str, mode: Mode, **kw: Any) -> Any:
        self.calls.append(symbol)
        return SimpleNamespace(symbol=symbol, mode=mode, decision_id=None, final_action="FILTERED", meta=None,
                               verdict=None, confluence=None, filtered_reason="test double", error=None,
                               latency_ms=0)


def _running(svc) -> None:  # type: ignore[no-untyped-def]
    from control.status import set_engine_state

    with svc.session() as s:
        set_engine_state(s, EngineState.RUNNING, now=svc.clock.now())
        s.commit()


def test_intraday_cycle_analyses_follow_up_stocks_first(tmp_path: Path) -> None:
    svc = _svc(tmp_path, datetime(2026, 11, 26, 10, 0, tzinfo=IST), changes={
        "engine.watchlist_ranking": False, "modes.modes.INTRADAY.max_symbols_per_cycle": 5})
    _running(svc)

    def row(sym: str, score: float, affordable: bool, close: float = 300.0) -> dict:  # type: ignore[type-arg]
        return {"symbol": sym, "setup": "BREAKOUT", "direction": "LONG", "trigger": close + 1, "stop": close - 2,
                "target": close + 7, "score": score, "atr": 2.0, "affordable": affordable, "close": close}

    _save_plan(svc, date(2026, 11, 26), [row("SBIN", 80, True), row("HDFCBANK", 75, False),
                                         row("ITC", 70, True), row("LT", 65, True)])
    with svc.session() as s:
        lst = wl.create_watchlist(s, "Trade these", now=svc.clock.now(), use_for_trading=True)
        wl.add_item(s, lst["id"], "TCS", now=svc.clock.now())
        wl.add_item(s, lst["id"], "ZOMATO", now=svc.clock.now())
        s.commit()
    rec = _Recorder()
    svc.pipeline = rec
    out = jobs.run_job(svc, "intraday_cycle")
    assert out["symbols"] == 5
    # plan by score (HDFCBANK: unaffordable + outside the universe -> skipped; LT: outside the universe but
    # in the plan and affordable -> kept), then the "use for trading" list (ZOMATO outside the universe ->
    # skipped), then the normal rotation fills the last place.
    assert rec.calls == ["SBIN", "ITC", "LT", "TCS", "INFY"]
    with svc.session() as s:
        tcs = s.execute(select(WatchlistItem).where(WatchlistItem.symbol == "TCS")).scalars().one()
    assert tcs.last_summary.startswith("TCS INTRADAY: no trade - test double") and tcs.last_analysis_at is not None
    assert any("follow-up stocks first: SBIN, ITC, LT, TCS" in e["message"] and "HDFCBANK" in e["message"]
               for e in _activity(tmp_path, "2026-11-26"))
    svc.apply_settings(update_settings(svc.settings, {"engine.followup_priority": False}))
    rec.calls.clear()
    jobs.run_job(svc, "intraday_cycle")
    assert "LT" not in rec.calls and sorted(rec.calls) == ["INFY", "ITC", "SBIN", "TCS"]


def test_priority_with_ranking_keeps_the_cap(tmp_path: Path) -> None:
    svc = _svc(tmp_path, datetime(2026, 11, 26, 10, 0, tzinfo=IST),
               changes={"modes.modes.INTRADAY.max_symbols_per_cycle": 2})
    _save_plan(svc, date(2026, 11, 26), [{"symbol": "TCS", "trigger": 1.0, "stop": 0.5, "score": 70.0,
                                          "affordable": True}])
    symbols = jobs._watchlist(svc, Mode.INTRADAY, svc.ctx.batch_id)
    prio = jobs._followup_priority(svc, Mode.INTRADAY, symbols, svc.ctx.batch_id)
    chunk = jobs._chunk_for(svc, Mode.INTRADAY, symbols, rotate=True, priority=prio)
    assert prio == ["TCS"] and chunk[0] == "TCS" and len(chunk) == 2 and len(set(chunk)) == 2
    assert jobs._followup_priority(svc, Mode.SWING, symbols, svc.ctx.batch_id) == []   # INTRADAY only


# --------------------------------------------------------------------------- Scan now / Analyze now


def test_scan_now_on_chosen_symbols_never_trades(tmp_path: Path) -> None:
    svc = _svc(tmp_path, datetime(2026, 11, 26, 10, 0, tzinfo=IST))
    svc.extras["rotation:INTRADAY"] = 3
    r = _cmd(svc, CommandType.SCAN_NOW, {"mode": "INTRADAY", "symbols": ["infy", "TCS", "bad sym", "INFY"]})
    assert r["status"] == "DONE", r
    assert r["data"]["scanned"] == ["INFY", "TCS"] and r["data"]["rejected"] == ["bad sym"]
    assert r["message"].startswith("INTRADAY scan of 2 stock(s)") and "ignored invalid symbol(s): bad sym" in r["message"]
    assert r["message"].endswith("Analysis only - nothing was traded.")
    assert svc.extras["rotation:INTRADAY"] == 3 and "ranking:INTRADAY" not in svc.extras
    with svc.session() as s:
        rows = list(s.execute(select(DecisionRow)).scalars())
    assert rows and {d.symbol for d in rows} <= {"INFY", "TCS"}
    assert {d.final_action for d in rows} <= {"ANALYZED", "FILTERED", "NO_TRADE"}
    assert _orders(svc) == 0


def test_scan_now_on_a_follow_up_list_records_the_analysis(tmp_path: Path) -> None:
    svc = _svc(tmp_path, datetime(2026, 11, 26, 10, 0, tzinfo=IST))
    with svc.session() as s:
        lst = wl.get_or_create_default(s, now=svc.clock.now())
        wl.add_item(s, lst["id"], "ITC", now=svc.clock.now())
        wl.add_item(s, lst["id"], "SBIN", now=svc.clock.now() + timedelta(seconds=1))
        s.commit()
    r = _cmd(svc, CommandType.SCAN_NOW, {"mode": "INTRADAY", "watchlist_id": lst["id"], "use_llm": False})
    assert r["status"] == "DONE", r
    assert r["data"]["scanned"] == ["ITC", "SBIN"]
    with svc.session() as s:
        items = {i["symbol"]: i for i in wl.get_watchlist(s, lst["id"])["items"]}  # type: ignore[index]
    for sym in ("ITC", "SBIN"):
        assert items[sym]["last_analysis_at"] is not None
        assert items[sym]["last_summary"].startswith(f"{sym} INTRADAY: ")
        assert "analysis only - not traded" in items[sym]["last_summary"]
    missing = _cmd(svc, CommandType.SCAN_NOW, {"mode": "INTRADAY", "watchlist_id": 999})
    assert missing["status"] == "REJECTED" and "watchlist #999 not found" in missing["message"]
    assert _orders(svc) == 0


def test_analyze_now_updates_the_follow_up_item(tmp_path: Path) -> None:
    svc = _svc(tmp_path, datetime(2026, 11, 26, 10, 0, tzinfo=IST))
    with svc.session() as s:
        lst = wl.get_or_create_default(s, now=svc.clock.now())
        wl.add_item(s, lst["id"], "INFY", now=svc.clock.now(), source="ANALYSIS")
        s.commit()
    r = _cmd(svc, CommandType.ANALYZE_NOW, {"symbol": "INFY", "mode": "INTRADAY"})
    assert r["status"] == "DONE", r
    with svc.session() as s:
        item = wl.get_watchlist(s, lst["id"])["items"][0]  # type: ignore[index]
    assert item["last_summary"] == r["message"] and item["last_analysis_id"] == r["data"]["decision_id"]
    assert _orders(svc) == 0


# --------------------------------------------------------------------------- EOD e-mail


def test_eod_report_carries_tomorrows_plan(tmp_path: Path) -> None:
    svc = _svc(tmp_path, datetime(2026, 11, 25, 15, 45, tzinfo=IST))
    sent: list[dict] = []  # type: ignore[type-arg]
    svc.notifier = SimpleNamespace(eod=lambda report: sent.append(dict(report)))
    assert jobs.run_job(svc, "eod_report") == {"sent": True}
    assert "next_day_plan" not in sent[-1]                                    # no plan: no section
    _save_plan(svc, date(2026, 11, 26), [
        {"symbol": "ITC", "setup": "PULLBACK", "direction": "LONG", "trigger": 312.4, "stop": 308.0,
         "target": 321.2, "rr": 2.0, "score": 61.0, "reasons": ["uptrend"], "affordable": True},
        {"symbol": "TCS", "setup": "BREAKOUT", "direction": "LONG", "trigger": 3400.0, "stop": 3350.0,
         "target": 3500.0, "rr": 2.0, "score": 72.5, "reasons": ["NR7"], "affordable": False}])
    jobs.run_job(svc, "eod_report")
    r = sent[-1]
    assert [p["symbol"] for p in r["next_day_plan"]] == ["TCS", "ITC"]
    assert r["next_day_plan"][1] == {"symbol": "ITC", "status": "ACTIVE", "setup": "PULLBACK", "direction": "LONG",
                                     "trigger": 312.4, "stop": 308.0, "target": 321.2, "rr": 2.0, "score": 61.0,
                                     "reasons": ["uptrend"], "affordable": True}
    assert r["next_day_plan_for"] == "2026-11-26" and r["next_day_plan_check"] == "09:20"
