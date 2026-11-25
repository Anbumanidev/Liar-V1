"""Engine transparency: activity feed, idle status, Scan now, background long commands.

2026-11-25: the owner started the engine at 14:53 (after the 09:30-14:45 intraday window), saw
nothing in the terminal for 20 minutes and assumed it was broken. These tests pin down that the
engine now says what it is doing and why it is idle, and that analysis on demand works without
blocking position monitoring.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from control.bootstrap import bootstrap
from control.commands import enqueue_command, get_command
from control.engine import TradingEngine
from control.processor import CommandProcessor, describe_result
from core.clock import IST, SimulatedClock
from core.enums import CommandStatus, CommandType, Mode
from data.calendar import load_calendar
from data.providers.replay_provider import ReplayDataProvider
from storage.models import ControlCommand, DecisionRow, Order


def _frames():  # type: ignore[no-untyped-def]
    import importlib.util
    import sys

    path = Path(__file__).with_name("test_engine_layer.py")
    spec = importlib.util.spec_from_file_location("_engine_layer_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.synthetic_frames(["INFY", "TCS"])


@pytest.fixture
def svc(tmp_path: Path):  # type: ignore[no-untyped-def]
    clock = SimulatedClock(datetime(2026, 11, 25, 14, 53, tzinfo=IST))
    provider = ReplayDataProvider(_frames(), clock, load_calendar())
    return bootstrap(process="test", clock=clock, db_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}",
                     primary_provider=provider, llm_backend="rules", with_notifier=False, offline=True,
                     runtime_dir=tmp_path / "rt", log_dir=tmp_path / "logs", console_log=False)


def _activity(tmp: Path) -> list[dict]:  # type: ignore[type-arg]
    path = tmp / "logs" / "2026-11-25" / "activity.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def _enqueue(svc, ctype: CommandType, payload: dict) -> int:  # type: ignore[no-untyped-def,type-arg]
    with svc.session() as s:
        row = enqueue_command(s, ctype, payload, actor="owner", batch_id=svc.ctx.batch_id, clock=svc.clock)
        s.commit()
        return row.id


# --------------------------------------------------------------------------- idle status


def test_status_after_the_intraday_window_says_why_and_what_is_next(svc) -> None:  # type: ignore[no-untyped-def]
    eng = TradingEngine(svc, use_scheduler=False)
    text = eng.status_text(datetime(2026, 11, 25, 14, 53, tzinfo=IST))
    assert text.startswith("idle - intraday scans run 09:30-14:45 (over for today)")
    assert "next: squareoff at 15:15 (in 22 min); next scan: swing_scan at 16:00 (in 67 min)" in text
    assert "Scan now" in text
    morning = eng.status_text(datetime(2026, 11, 25, 9, 10, tzinfo=IST))
    assert "intraday scans start at 09:30" in morning
    holiday = eng.status_text(datetime(2026, 11, 24, 11, 0, tzinfo=IST))   # Guru Nanak Jayanti
    assert "not an NSE trading day" in holiday and "next session Wed 25 Nov" in holiday
    eng._running.add("intraday_cycle")
    assert eng.status_text(datetime(2026, 11, 25, 10, 0, tzinfo=IST)) == "running intraday_cycle"


def test_tick_writes_status_to_heartbeat_and_activity_log(svc, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from storage.models import EngineStatus

    eng = TradingEngine(svc, use_scheduler=False)
    eng.tick()
    with svc.session() as s:
        row = s.get(EngineStatus, 1)
        assert row.activity.startswith("idle - intraday scans run 09:30-14:45")
        assert row.status_json["text"] == row.activity
    events = [e for e in _activity(tmp_path) if e["stage"] == "engine"]
    assert events and events[-1]["next_job"] == "squareoff" and events[-1]["level"] == "info"
    assert "next scan: swing_scan at 16:00" in events[-1]["message"]
    eng.tick()   # unchanged status within 10 min -> not repeated
    assert len([e for e in _activity(tmp_path) if e["stage"] == "engine"]) == len(events)


def test_orphaned_running_commands_are_failed_on_start(svc) -> None:  # type: ignore[no-untyped-def]
    cid = _enqueue(svc, CommandType.ANALYZE_NOW, {"symbol": "INFY", "mode": "INTRADAY"})
    with svc.session() as s:
        s.get(ControlCommand, cid).status = CommandStatus.RUNNING.value
        s.commit()
    TradingEngine(svc, use_scheduler=False)._fail_orphaned_commands()
    with svc.session() as s:
        row = get_command(s, cid)
        assert row.status == "FAILED" and "engine restarted" in row.error


# --------------------------------------------------------------------------- long commands


def test_long_commands_run_in_the_background_with_an_engine(svc) -> None:  # type: ignore[no-untyped-def]
    eng = TradingEngine(svc, use_scheduler=False)
    cid = _enqueue(svc, CommandType.ANALYZE_NOW, {"symbol": "INFY", "mode": "INTRADAY"})
    eng.processor.process_pending()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with svc.session() as s:
            status = get_command(s, cid).status
        if status not in ("PENDING", "RUNNING"):
            break
        time.sleep(0.2)
    eng.processor._worker.shutdown(wait=True)
    with svc.session() as s:
        row = get_command(s, cid)
        assert row.status == "DONE", row.result_json
        msg = row.result_json["message"]
    assert msg.startswith("INFY INTRADAY: ") and "analysis only - not traded" in msg


def test_without_an_engine_commands_stay_synchronous(svc) -> None:  # type: ignore[no-untyped-def]
    cid = _enqueue(svc, CommandType.ANALYZE_NOW, {"symbol": "TCS", "mode": "INTRADAY"})
    CommandProcessor(svc).process_pending()
    with svc.session() as s:
        assert get_command(s, cid).status == "DONE"


# --------------------------------------------------------------------------- scan now


def test_scan_now_analyses_and_never_trades(svc, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cid = _enqueue(svc, CommandType.SCAN_NOW, {"mode": "INTRADAY"})
    CommandProcessor(svc).process_pending()
    with svc.session() as s:
        row = get_command(s, cid)
        assert row.status == "DONE", row.result_json
        msg = row.result_json["message"]
        assert s.scalar(select(func.count(Order.id))) == 0                     # analysis only
        actions = {d.final_action for d in s.execute(select(DecisionRow)).scalars()}
    assert msg.startswith("INTRADAY scan of ") and msg.endswith("Analysis only - nothing was traded.")
    assert actions <= {"ANALYZED", "FILTERED", "NO_TRADE"}
    stages = {e["stage"] for e in _activity(tmp_path)}
    assert {"cycle", "command"} <= stages
    assert any(e["stage"] == "cycle" and "screening" in e["message"] for e in _activity(tmp_path))


def test_scan_now_without_llm(svc) -> None:  # type: ignore[no-untyped-def]
    from scheduler.jobs import scan_now

    out = scan_now(svc, Mode.INTRADAY, use_llm=False)
    assert out["symbols"] >= 1 and "0 analysed by the AI" in out["message"]


# --------------------------------------------------------------------------- messages


def _res(**kw):  # type: ignore[no-untyped-def]
    base = dict(symbol="ITC", mode=Mode.INTRADAY, error=None, filtered_reason=None, meta=None, verdict=None,
                confluence=None, final_action="ANALYZED", latency_ms=43000)
    base.update(kw)
    return SimpleNamespace(**base)


def test_describe_result_explains_no_trade() -> None:
    from core.enums import TradeDecision

    meta = SimpleNamespace(decision=TradeDecision.NO_TRADE, why_not="no strategy signal")
    msg = describe_result(_res(meta=meta), routed=False)
    assert msg == ("ITC INTRADAY: no trade - no strategy setup right now (none of the mode's strategies matched "
                   "the chart) (analysis only - not traded) [43s]")
    assert "no trade - confluence 41.0 < 60" in describe_result(_res(filtered_reason="confluence 41.0 < 60"),
                                                                 routed=True)


def test_describe_result_shows_the_risk_verdict() -> None:
    from core.enums import TradeDecision

    meta = SimpleNamespace(decision=TradeDecision.BUY, strategy="vwap_pullback", entry_price=310.5, stop_loss=309.4,
                           target_1=312.1, confidence=80)
    verdict = SimpleNamespace(approved=False, message="NO_ROOM: nearest opposing level only 0.13R away", qty=0)
    msg = describe_result(_res(meta=meta, verdict=verdict, confluence=SimpleNamespace(score=72.0)), routed=False)
    assert "BUY vwap_pullback entry 310.5, SL 309.4, T1 312.1 (confidence 80, confluence 72)" in msg
    assert "risk manager REJECTED: NO_ROOM" in msg


# --------------------------------------------------------------------------- review fixes


def test_stop_fails_queued_long_commands_instead_of_leaving_them_running(svc, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import threading

    gate = threading.Event()
    eng = TradingEngine(svc, use_scheduler=False)
    real = eng.processor.handlers[CommandType.ANALYZE_NOW]

    def slow(s, p, a):  # type: ignore[no-untyped-def]
        gate.wait(10)
        return real(s, p, a)

    eng.processor.handlers[CommandType.ANALYZE_NOW] = slow
    first = _enqueue(svc, CommandType.ANALYZE_NOW, {"symbol": "INFY", "mode": "INTRADAY"})
    second = _enqueue(svc, CommandType.ANALYZE_NOW, {"symbol": "TCS", "mode": "INTRADAY"})
    eng.processor.process_pending()
    time.sleep(0.3)
    assert eng.status_text(datetime(2026, 11, 25, 14, 53, tzinfo=IST)).startswith("running ANALYZE_NOW")
    assert eng.processor.shutdown() == 1                 # the queued one is cancelled ...
    gate.set()
    eng.processor._worker.shutdown(wait=True)
    with svc.session() as s:
        assert get_command(s, first).status == "DONE"    # ... the one in flight finishes normally
        row = get_command(s, second)
        assert row.status == "FAILED" and "engine stopped before this command ran" in row.error


def test_countdown_alone_does_not_relog_the_status(svc, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from datetime import timedelta

    eng = TradingEngine(svc, use_scheduler=False)
    for _ in range(4):
        eng.tick()
        svc.clock.advance(timedelta(minutes=1))
    assert len([e for e in _activity(tmp_path) if e["stage"] == "engine"]) == 1


def test_scan_now_keeps_screened_out_stocks_filtered_and_rotation_untouched(svc) -> None:  # type: ignore[no-untyped-def]
    from config.loader import update_settings
    from scheduler.jobs import scan_now

    svc.extras["rotation:INTRADAY"] = 0
    svc.apply_settings(update_settings(svc.settings, {"engine.watchlist_ranking": False,
                                                       "modes.modes.INTRADAY.max_symbols_per_cycle": 1}))
    scan_now(svc, Mode.INTRADAY, use_llm=False)
    assert svc.extras["rotation:INTRADAY"] == 0          # analysis-only scan never moves the live rotation
    with svc.session() as s:
        rows = list(s.execute(select(DecisionRow)).scalars())
    assert rows and all(r.final_action == "FILTERED" for r in rows if (r.context_json or {}).get("filtered_reason"))
