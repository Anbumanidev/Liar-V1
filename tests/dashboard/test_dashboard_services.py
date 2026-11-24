"""Unit tests for dashboard.services (no Streamlit, no network)."""

from __future__ import annotations

import io
import sqlite3
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import openpyxl
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from config.loader import load_hard_caps
from config.store import get_active_settings, list_versions
from core.clock import IST, SimulatedClock
from core.enums import CommandType
from core.logging import log_event
from dashboard import services as svc
from storage.models import AuditLog, Batch, ControlCommand, EngineStatus, PromptTemplate

NOW = datetime(2026, 11, 24, 10, 0, tzinfo=IST)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


def _commands(session: Session, ctype: CommandType) -> list[ControlCommand]:
    return list(session.execute(select(ControlCommand).where(ControlCommand.command_type == ctype.value)).scalars())


# --------------------------------------------------------------------------- pure helpers


def test_dashboard_format_countdown() -> None:
    assert svc.format_countdown(0) == "EXPIRED"
    assert svc.format_countdown(-5) == "EXPIRED"
    assert svc.format_countdown(155) == "02:35"
    assert svc.format_countdown(3600 + 5 * 60) == "1h 05m"
    assert svc.format_countdown(26 * 3600) == "1d 2h 00m"
    assert svc.format_countdown(None) == "-"


def test_dashboard_confirmation() -> None:
    assert svc.confirmation_ok("CONFIRM")
    assert svc.confirmation_ok("  CONFIRM ")
    assert not svc.confirmation_ok("confirm")
    assert svc.confirmation_ok("batch-1", batch_name="batch-1")
    assert not svc.confirmation_ok("batch-2", batch_name="batch-1")
    assert not svc.confirmation_ok("")


def test_dashboard_parse_helpers() -> None:
    valid, invalid = svc.parse_symbols("reliance, TCS.NS\nm&m  tcs bad$sym")
    assert valid == ["M&M", "RELIANCE", "TCS"]
    assert invalid == ["bad$sym"]
    assert svc.parse_windows("09:30-14:45, 15:00 - 15:10") == [
        {"start": "09:30", "end": "14:45"}, {"start": "15:00", "end": "15:10"}]
    assert svc.parse_windows("") == []
    with pytest.raises(svc.DashboardError):
        svc.parse_windows("14:45-09:30")
    with pytest.raises(svc.DashboardError):
        svc.parse_windows("9:30-25:00")
    assert svc.parse_dates("2026-12-25\n2026-12-25, 2027-01-26") == [date(2026, 12, 25), date(2027, 1, 26)]
    with pytest.raises(svc.DashboardError):
        svc.parse_dates("25/12/2026")
    ok, bad = svc.parse_emails("a@b.com, nope")
    assert ok == ["a@b.com"] and bad == ["nope"]
    assert svc.validate_hhmm("09:15") == "09:15"
    assert svc.validate_hhmm("", allow_empty=True) is None
    with pytest.raises(svc.DashboardError):
        svc.validate_hhmm("9:15")


def test_dashboard_sl_tighten_and_levels() -> None:
    assert svc.validate_sl_tighten("LONG", 98.0, 99.0, 101.0) is None
    assert "ABOVE" in svc.validate_sl_tighten("LONG", 98.0, 97.0, 101.0)
    assert "below the last price" in svc.validate_sl_tighten("LONG", 98.0, 101.5, 101.0)
    assert svc.validate_sl_tighten("SHORT", 105.0, 104.0, 100.0) is None
    assert "BELOW" in svc.validate_sl_tighten("SHORT", 105.0, 106.0, 100.0)
    assert svc.validate_trade_levels("LONG", 100.0, 98.0, 104.0, 106.0) == []
    assert svc.validate_trade_levels("LONG", 100.0, 101.0, 104.0, None)
    assert svc.validate_trade_levels("SHORT", 100.0, 102.0, 96.0, 94.0) == []
    assert svc.validate_trade_levels("SHORT", 100.0, 98.0, 96.0, None)
    assert svc.allowed_modes_for_direction("SHORT") == ["INTRADAY"]
    assert set(svc.allowed_modes_for_direction("LONG")) == {"INTRADAY", "BTST", "SWING", "POSITIONAL"}


def test_dashboard_allocations() -> None:
    assert svc.validate_allocations({"A": 50, "B": 50}) is None
    assert "100" in svc.validate_allocations({"A": 60, "B": 50})
    assert svc.validate_allocations({"A": -1}) is not None


# --------------------------------------------------------------------------- approvals / decisions


def test_dashboard_pending_approvals_countdown(dash_factory: sessionmaker[Session], seeded: dict[str, int]) -> None:
    with dash_factory() as s:
        now = datetime.now(IST)
        views = svc.pending_approvals(s, seeded["batch"], now)
        assert [v.item.id for v in views] == [seeded["approval"], seeded["approval2"]]  # soonest expiry first
        first = views[0]
        assert 170 <= first.seconds_left <= 180 and not first.expired
        assert first.countdown.startswith("02:") or first.countdown == "03:00"
        assert first.risk_amount == pytest.approx(10.0)
        assert first.rr == pytest.approx(2.0)
        later = svc.pending_approvals(s, seeded["batch"], now + timedelta(minutes=5))
        assert later[0].expired and later[0].seconds_left == 0 and later[0].countdown == "EXPIRED"
        assert svc.pending_approvals(s, 999, now) == []
        history = svc.recent_approval_history(s, seeded["batch"])
        assert [h.status for h in history] == ["REJECTED"]


def test_dashboard_decision_details(dash_factory: sessionmaker[Session], seeded: dict[str, int]) -> None:
    with dash_factory() as s:
        item, details = svc.approval_details(s, seeded["approval"])
        assert item is not None and item.symbol == "RELIANCE"
        assert details.decision["decision"] == "BUY"
        assert [r["agent"] for r in details.agent_reports] == ["technical"]
        assert details.debates[0]["rounds"] == 2
        assert details.meta["reasoning"].startswith("Strong")
        assert details.risk["approved"] is True
        missing, empty = svc.approval_details(s, 12345)
        assert missing is None and empty.decision is None


# --------------------------------------------------------------------------- settings


def test_dashboard_settings_change_clamped_creates_version_and_reload(session: Session, clock: SimulatedClock) -> None:
    caps = load_hard_caps()
    vid0, _ = get_active_settings(session, caps, clock=clock)
    res = svc.apply_settings_changes(
        session, {"risk.risk_per_trade_pct": 5.0, "modes.modes.INTRADAY.min_rr": 0.6}, caps,
        actor="admin", note="test", batch_id=None, clock=clock)
    assert res.changed and res.version_id != vid0
    assert len(res.adjustments) == 2
    _, active = get_active_settings(session, caps, clock=clock)
    assert active.risk.risk_per_trade_pct == caps.max_risk_per_trade_pct  # clamped to 2.0
    assert active.mode("INTRADAY").min_rr == caps.min_rr_floor            # clamped up to the floor
    cmds = _commands(session, CommandType.RELOAD_SETTINGS)
    assert len(cmds) == 1 and cmds[0].id == res.command_id and cmds[0].requested_by == "admin"
    actions = {a.action for a in session.execute(select(AuditLog)).scalars()}
    assert {"SETTINGS_CHANGE", "COMMAND_RELOAD_SETTINGS"} <= actions

    # same values again -> no new version, no command
    again = svc.apply_settings_changes(session, {"risk.risk_per_trade_pct": 2.0}, caps, actor="admin", note=None,
                                       batch_id=None, clock=clock)
    assert not again.changed and again.command_id is None
    assert len(_commands(session, CommandType.RELOAD_SETTINGS)) == 1


def test_dashboard_settings_invalid_and_revert(session: Session, clock: SimulatedClock) -> None:
    caps = load_hard_caps()
    v1, _ = get_active_settings(session, caps, clock=clock)
    with pytest.raises(svc.DashboardError):
        svc.apply_settings_changes(session, {"modes.modes.INTRADAY.allocation_pct": 90.0}, caps, actor="a",
                                   note=None, batch_id=None, clock=clock)  # sum > 100
    with pytest.raises(svc.DashboardError):
        svc.apply_settings_changes(session, {"risk.no_such_field": 1}, caps, actor="a", note=None, batch_id=None,
                                   clock=clock)
    with pytest.raises(svc.DashboardError):
        svc.apply_settings_changes(session, {"schedule.intraday_squareoff": "16:00"}, caps, actor="a", note=None,
                                   batch_id=None, clock=clock)
    res = svc.apply_settings_changes(session, {"capital.profit_lock_pct": 40.0}, caps, actor="a", note=None,
                                     batch_id=None, clock=clock)
    rev = svc.revert_settings(session, v1, caps, actor="a", batch_id=None, clock=clock)
    assert rev.version_id > res.version_id
    _, active = get_active_settings(session, caps, clock=clock)
    assert active.capital.profit_lock_pct == 30.0
    assert len(_commands(session, CommandType.RELOAD_SETTINGS)) == 2
    assert list_versions(session)[0].reverted_from_id == v1
    with pytest.raises(svc.DashboardError):
        svc.revert_settings(session, 9999, caps, actor="a", batch_id=None, clock=clock)


def test_dashboard_changed_only_and_caps() -> None:
    caps = load_hard_caps()
    assert svc.changed_only({"a": 1.0, "b": "x"}, {"a": 1.0, "b": "y"}) == {"b": "y"}
    assert svc.cap_bounds("risk.mis_leverage", caps) == (None, caps.max_mis_leverage)
    assert svc.cap_bounds("modes.modes.SWING.min_rr", caps) == (caps.min_rr_floor, None)
    assert svc.cap_bounds("risk.gap_buffer_pct", caps) == (None, None)
    assert "max" in svc.cap_hint("risk.daily_loss_limit_pct", caps)
    clamped, adj = svc.clamp_changes({"risk.mis_leverage": 9, "risk.avoid_events_in_window": True}, caps)
    assert clamped["risk.mis_leverage"] == 5 and isinstance(clamped["risk.mis_leverage"], int)
    assert clamped["risk.avoid_events_in_window"] is True and len(adj) == 1


# --------------------------------------------------------------------------- prompts


def test_dashboard_prompt_versioning(session: Session, clock: SimulatedClock) -> None:
    v1 = svc.save_prompt_version(session, "meta", "sys v1", "user v1", author="admin", note="first", clock=clock)
    v2 = svc.save_prompt_version(session, "meta", "sys v2", "user v2", author="admin", note=None, clock=clock)
    other = svc.save_prompt_version(session, "bull", "b", "u", author="admin", note=None, clock=clock)
    assert (v1.version, v2.version, other.version) == (1, 2, 1)
    versions = svc.prompt_versions(session, "meta")
    assert [v.version for v in versions] == [2, 1]
    assert versions[0].active and not versions[1].active
    rev = svc.revert_prompt(session, "meta", 1, author="admin", clock=clock)
    assert rev.version == 3 and rev.system_prompt == "sys v1" and rev.note == "Revert to v1"
    assert svc.latest_prompt(session, "meta").version == 3
    assert len(session.execute(select(PromptTemplate)).scalars().all()) == 4  # history append-only
    with pytest.raises(svc.DashboardError):
        svc.revert_prompt(session, "meta", 42, author="admin", clock=clock)
    with pytest.raises(svc.DashboardError):
        svc.save_prompt_version(session, "meta", " ", "x", author="admin", note=None, clock=clock)
    actions = [a.action for a in session.execute(select(AuditLog)).scalars()]
    assert actions.count("PROMPT_SAVE") == 4 and "PROMPT_REVERT" in actions


# --------------------------------------------------------------------------- batches


def test_dashboard_soft_delete_rules(dash_factory: sessionmaker[Session], seeded: dict[str, int],
                                     clock: SimulatedClock) -> None:
    with dash_factory() as s:
        with pytest.raises(svc.DashboardError, match="current batch"):
            svc.soft_delete_batch(s, seeded["batch"], confirm_text="CONFIRM", confirm_name="batch-1",
                                  actor="admin", clock=clock)
        with pytest.raises(svc.DashboardError, match="CONFIRM"):
            svc.soft_delete_batch(s, seeded["old_batch"], confirm_text="yes", confirm_name="old-batch",
                                  actor="admin", clock=clock)
        with pytest.raises(svc.DashboardError, match="batch name"):
            svc.soft_delete_batch(s, seeded["old_batch"], confirm_text="CONFIRM", confirm_name="old",
                                  actor="admin", clock=clock)
        with pytest.raises(svc.DashboardError, match="not found"):
            svc.soft_delete_batch(s, 999, confirm_text="CONFIRM", confirm_name="x", actor="admin", clock=clock)
        closing = Batch(name="closing", starting_capital=1000, status="CLOSING", created_at=NOW)
        s.add(closing)
        s.flush()
        with pytest.raises(svc.DashboardError, match="CLOSING"):
            svc.soft_delete_batch(s, closing.id, confirm_text="CONFIRM", confirm_name="closing", actor="admin",
                                  clock=clock)
        batch = svc.soft_delete_batch(s, seeded["old_batch"], confirm_text="CONFIRM", confirm_name="old-batch",
                                      actor="admin", clock=clock)
        s.commit()
        assert batch.deleted
        with pytest.raises(svc.DashboardError, match="already deleted"):
            svc.soft_delete_batch(s, seeded["old_batch"], confirm_text="CONFIRM", confirm_name="old-batch",
                                  actor="admin", clock=clock)
        assert s.get(Batch, seeded["old_batch"]) is not None  # row retained
        assert [o.id for o in svc.batch_options(s)] == [closing.id, seeded["batch"]]
        assert len(svc.batch_options(s, include_deleted=True)) == 3
        audit_rows = s.execute(select(AuditLog).where(AuditLog.action == "BATCH_SOFT_DELETE")).scalars().all()
        assert len(audit_rows) == 1


def test_dashboard_export(dash_factory: sessionmaker[Session], seeded: dict[str, int]) -> None:
    with dash_factory() as s:
        data, name, mime = svc.export_batch(s, seeded["batch"], "xlsx")
        assert name.endswith(".xlsx") and "spreadsheet" in mime
        wb = openpyxl.load_workbook(io.BytesIO(data))
        assert set(wb.sheetnames) == {"trades", "positions", "ledger", "decisions", "approvals", "orders"}
        assert wb["positions"].max_row == 3  # header + 2 positions
        data, name, mime = svc.export_batch(s, seeded["batch"], "csv")
        assert mime == "application/zip"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "trades.csv" in zf.namelist()
            assert "HDFCBANK" in zf.read("trades.csv").decode()
        with pytest.raises(svc.DashboardError):
            svc.export_batch(s, seeded["batch"], "pdf")
        with pytest.raises(svc.DashboardError):
            svc.export_batch(s, 999, "csv")


# --------------------------------------------------------------------------- engine / commands


def test_dashboard_engine_status(session: Session) -> None:
    missing = svc.engine_status(session, NOW, 30)
    assert missing.state == "UNKNOWN" and missing.stale and missing.safe_for_restore and not missing.alive
    session.add(EngineStatus(id=1, state="RUNNING", heartbeat_at=NOW - timedelta(seconds=10), data_feed_ok=True))
    session.flush()
    fresh = svc.engine_status(session, NOW, 30)
    assert not fresh.stale and fresh.alive and not fresh.safe_for_restore and fresh.label == "RUNNING"
    stale = svc.engine_status(session, NOW + timedelta(seconds=60), 30)
    assert stale.stale and stale.safe_for_restore and stale.label == "RUNNING (stale)"
    session.get(EngineStatus, 1).state = "STOPPED"
    session.flush()
    assert svc.engine_status(session, NOW, 30).safe_for_restore


def test_dashboard_command_status(dash_factory: sessionmaker[Session], seeded: dict[str, int],
                                  clock: SimulatedClock) -> None:
    with dash_factory() as s:
        views = svc.recent_command_views(s, types=[CommandType.MANUAL_TRADE])
        assert views[0].rule == "RR_BELOW_MIN" and views[0].finished and "1.2" in views[0].message
        cid = svc.send_command(s, CommandType.ENGINE_PAUSE, {}, actor="admin", batch_id=seeded["batch"], clock=clock)
        s.commit()
        view = svc.command_status(s, cid)
        assert view.status == "PENDING" and not view.finished and view.result is None and view.rule is None
        with pytest.raises(svc.DashboardError, match="Invalid TIGHTEN_SL"):
            svc.send_command(s, CommandType.TIGHTEN_SL, {"position_id": 1, "new_stop_loss": -1}, actor="admin",
                             batch_id=None, clock=clock)
        assert svc.command_status(s, 99999) is None


def test_dashboard_positions_and_states(dash_factory: sessionmaker[Session], seeded: dict[str, int]) -> None:
    with dash_factory() as s:
        positions = svc.open_positions(s, seeded["batch"])
        assert [p.symbol for p in positions] == ["ITC", "SBIN"]
        rows = {r["symbol"]: r for r in svc.position_rows(positions)}
        assert rows["SBIN"]["uPnL"] == pytest.approx((805 - 800) * 4 - 1.2)
        assert rows["ITC"]["uPnL"] is None
        assert svc.mode_states(s, seeded["batch"])["SWING"].paused
        assert svc.strategy_states(s, seeded["batch"])[0].auto_disabled
        perf = svc.strategy_performance(s, seeded["batch"])
        assert perf.iloc[0]["strategy"] == "orb" and perf.iloc[0]["trades"] == 1
        assert svc.strategy_performance(s, None).empty
        assert svc.capital_account(s, seeded["batch"]).reserve == 50.0
        assert len(svc.recent_ledger(s, seeded["batch"])) == 1


# --------------------------------------------------------------------------- ollama / logs


def test_dashboard_ollama_models() -> None:
    def ok_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}, {"name": "qwen3:4b"}, {"model": "x"}]})

    with httpx.Client(transport=httpx.MockTransport(ok_handler)) as client:
        assert svc.installed_ollama_models("http://ollama:11434", client=client) == ["qwen3:4b", "qwen3:8b", "x"]
        assert svc.ollama_reachable("http://ollama:11434/", client=client)

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(fail_handler)) as client:
        assert svc.installed_ollama_models("http://ollama:11434", client=client) == []
        assert not svc.ollama_reachable("http://ollama:11434", client=client)

    def bad_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    with httpx.Client(transport=httpx.MockTransport(bad_json)) as client:
        assert svc.installed_ollama_models("http://ollama:11434", client=client) == []


def test_dashboard_query_logs(tmp_path: Path, clock: SimulatedClock) -> None:
    from core.logging import set_log_dir

    set_log_dir(tmp_path)
    log_event("errors", {"component": "data", "message": "feed down"}, clock=clock)
    clock.advance(timedelta(minutes=1))
    log_event("control", {"event": "enqueued", "type": "ENGINE_PAUSE"}, clock=clock)
    events = svc.query_logs(NOW.date(), ["errors", "control"], log_dir=tmp_path)
    assert [e["category"] for e in events] == ["control", "errors"]  # newest first
    assert len(svc.query_logs(NOW.date(), ["errors", "control"], text="FEED", log_dir=tmp_path)) == 1
    assert svc.query_logs(NOW.date(), ["control"], limit=0, log_dir=tmp_path) == []
    assert svc.query_logs(date(2020, 1, 1), ["errors"], log_dir=tmp_path) == []


# --------------------------------------------------------------------------- backup / restore


def test_dashboard_backup_and_restore(dash_db_url: str, dash_factory: sessionmaker[Session],
                                      seeded: dict[str, int], tmp_path: Path) -> None:
    backup_dir = tmp_path / "backups"
    path = svc.backup_database(dash_db_url, backup_dir, NOW)
    assert path.name == "trading_20261124_100000.db" and path.exists()
    second = svc.backup_database(dash_db_url, backup_dir, NOW)
    assert second != path and second.exists()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM batches").fetchone()[0] == 2
    assert svc.list_backups(backup_dir)
    assert svc.database_size_bytes(dash_db_url) > 0

    # change the live DB after the backup
    with dash_factory() as s:
        s.add(Batch(name="after-backup", starting_capital=1000, status="CLOSED", created_at=NOW))
        s.commit()

    running = svc.EngineStatusView(state="RUNNING", heartbeat_at=NOW, age_seconds=1, stale=False)
    stopped = svc.EngineStatusView(state="STOPPED", heartbeat_at=NOW, age_seconds=1, stale=False)
    with pytest.raises(svc.DashboardError, match="CONFIRM"):
        svc.restore_database(dash_db_url, path, engine_view=stopped, confirm_text="yes", backup_dir=backup_dir,
                             now=NOW)
    with pytest.raises(svc.DashboardError, match="Stop the engine"):
        svc.restore_database(dash_db_url, path, engine_view=running, confirm_text="CONFIRM", backup_dir=backup_dir,
                             now=NOW)
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database at all" * 100)
    with pytest.raises(svc.DashboardError):
        svc.restore_database(dash_db_url, junk, engine_view=stopped, confirm_text="CONFIRM", backup_dir=backup_dir,
                             now=NOW)
    safety = svc.restore_database(dash_db_url, path, engine_view=stopped, confirm_text="CONFIRM",
                                  backup_dir=backup_dir, now=NOW, sa_engine=dash_factory.kw["bind"])
    assert safety.name.startswith("pre_restore_") and safety.exists()
    with dash_factory() as s:
        names = {b.name for b in s.execute(select(Batch)).scalars()}
    assert names == {"old-batch", "batch-1"}  # 'after-backup' gone
    with sqlite3.connect(safety) as conn:
        assert conn.execute("SELECT count(*) FROM batches").fetchone()[0] == 3


def test_dashboard_backup_requires_sqlite_file(tmp_path: Path) -> None:
    with pytest.raises(svc.DashboardError):
        svc.backup_database("sqlite://", tmp_path, NOW)
    with pytest.raises(svc.DashboardError):
        svc.backup_database("postgresql+psycopg://u:p@h/db", tmp_path, NOW)
    with pytest.raises(svc.DashboardError, match="not found"):
        svc.backup_database(f"sqlite:///{(tmp_path / 'missing.db').as_posix()}", tmp_path, NOW)
    assert svc.sqlite_path_from_url("sqlite:///:memory:") is None
    assert svc.database_size_bytes("sqlite://") is None


def test_dashboard_health(dash_db_url: str, dash_factory: sessionmaker[Session], seeded: dict[str, int]) -> None:
    with dash_factory() as s:
        health = svc.health_status(s, db_url=dash_db_url, ollama_host="http://127.0.0.1:9", now=datetime.now(IST),
                                   stale_seconds=30, check_ollama=False)
    assert health["engine_state"] == "RUNNING" and health["engine_alive"]
    assert health["db_size_mb"] is not None and health["data_feed_ok (engine)"] is True
    assert health["ollama_reachable"] is None
