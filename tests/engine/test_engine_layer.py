"""Engine layer: bootstrap, command processor (dashboard -> engine), batches, replay."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select

from control.bootstrap import bootstrap
from control.commands import enqueue_command, get_command
from control.processor import CommandProcessor
from core.clock import IST, SimulatedClock
from core.enums import (
    BatchStatus,
    CommandStatus,
    CommandType,
    Direction,
    EngineState,
    EntryType,
    Mode,
    PositionStatus,
    RiskRule,
    Timeframe,
)
from data.calendar import load_calendar
from data.providers.replay_provider import ReplayDataProvider
from scheduler.replay import ReplayConfig, ReplayRunner
from storage.models import Batch, CapitalAccount, DailySummary, DecisionRow, EngineStatus, Order, Position, Run
from tests.helpers import make_daily_candles, make_intraday_candles

HIST = [date(2026, 11, 18), date(2026, 11, 19), date(2026, 11, 20), date(2026, 11, 23)]
REPLAY_DAYS = [date(2026, 11, 25), date(2026, 11, 26)]


def synthetic_frames(symbols: list[str], drift: float = 0.00005) -> dict[tuple[str, Timeframe], pd.DataFrame]:
    """Consistent multi-timeframe history: every timeframe is resampled from one 1m series."""
    from data.bars import resample_ohlcv
    from features.market import DEFAULT_GLOBAL_CUES, DEFAULT_SECTOR_INDICES

    frames: dict[tuple[str, Timeframe], pd.DataFrame] = {}
    all_days = HIST + REPLAY_DAYS
    main = symbols + ["NIFTY50", "BANKNIFTY", "INDIAVIX"]
    for i, sym in enumerate(main):
        px = {"NIFTY50": 24000.0, "BANKNIFTY": 52000.0, "INDIAVIX": 13.0}.get(sym, 200.0 + 50 * i)
        m1 = make_intraday_candles(all_days, 1, px, seed=i, drift=drift, vol=0.0008)
        frames[(sym, Timeframe.M1)] = m1
        for tf in (Timeframe.M5, Timeframe.M15, Timeframe.H1):
            frames[(sym, tf)] = resample_ohlcv(m1, tf)
        recent = resample_ohlcv(m1, Timeframe.D1)
        hist = make_daily_candles(date(2026, 11, 17), 296, px * 0.8, seed=i + 40)
        hist = hist * (float(m1["open"].iloc[0]) / float(hist["close"].iloc[-1]))
        hist["volume"] = recent["volume"].mean()
        frames[(sym, Timeframe.D1)] = pd.concat([hist, recent]).sort_index()
    for j, sym in enumerate(list(DEFAULT_SECTOR_INDICES.values()) + list(DEFAULT_GLOBAL_CUES)):
        frames[(sym, Timeframe.D1)] = make_daily_candles(date(2026, 11, 26), 60, 1000.0 + j, seed=100 + j)
    return frames


@pytest.fixture
def svc(tmp_path: Path):  # type: ignore[no-untyped-def]
    clock = SimulatedClock(datetime(2026, 11, 25, 10, 0, tzinfo=IST))
    frames = synthetic_frames(["INFY", "TCS"])
    provider = ReplayDataProvider(frames, clock, load_calendar())
    services = bootstrap(process="test", clock=clock, db_url=f"sqlite:///{(tmp_path / 'e.db').as_posix()}",
                         primary_provider=provider, llm_backend="rules", with_notifier=False, offline=True,
                         runtime_dir=tmp_path / "rt", log_dir=tmp_path / "logs", console_log=False)
    return services


def _cmd(svc, ctype: CommandType, payload: dict) -> dict:  # type: ignore[no-untyped-def,type-arg]
    with svc.session() as s:
        row = enqueue_command(s, ctype, payload, actor="owner", batch_id=svc.ctx.batch_id, clock=svc.clock)
        s.commit()
        cid = row.id
    CommandProcessor(svc).process_pending()
    with svc.session() as s:
        row = get_command(s, cid)
        return {"status": row.status, **(row.result_json or {})}


def _open_position(svc, symbol: str = "INFY", mode: Mode = Mode.INTRADAY) -> int:  # type: ignore[no-untyped-def]
    r = _cmd(svc, CommandType.MANUAL_TRADE, {"symbol": symbol, "mode": mode.value, "direction": "LONG",
                                             "entry_type": "MARKET", "qty": 1,
                                             "stop_loss": 1.0, "target_1": 10_000.0})
    return r  # type: ignore[return-value]


# ---------------------------------------------------------------- bootstrap


def test_bootstrap_creates_batch_and_capital(svc) -> None:  # type: ignore[no-untyped-def]
    with svc.session() as s:
        b = s.get(Batch, svc.ctx.batch_id)
        assert b.status == BatchStatus.ACTIVE.value and b.starting_capital == 1000.0
        acct = s.get(CapitalAccount, b.id)
        assert acct.trading_cash == 1000.0 and acct.reserve == 0.0
        assert b.settings_snapshot is not None


# ---------------------------------------------------------------- command processor


def test_engine_pause_resume_and_invalid_payload(svc) -> None:  # type: ignore[no-untyped-def]
    r = _cmd(svc, CommandType.ENGINE_PAUSE, {})
    assert r["status"] == CommandStatus.DONE.value
    with svc.session() as s:
        assert s.get(EngineStatus, 1).state == EngineState.PAUSED.value
    assert _cmd(svc, CommandType.ENGINE_RESUME, {})["ok"]
    with svc.session() as s:
        row = enqueue_command(s, CommandType.ENGINE_START, {}, actor="owner")
        row.payload_json = {"bogus": 1}       # tampered payload is rejected by the engine
        s.commit()
        cid = row.id
    CommandProcessor(svc).execute(cid)
    with svc.session() as s:
        assert get_command(s, cid).status == CommandStatus.REJECTED.value


def test_manual_trade_blocked_rule_reported(svc) -> None:  # type: ignore[no-untyped-def]
    r = _cmd(svc, CommandType.MANUAL_TRADE, {"symbol": "INFY", "mode": "SWING", "direction": "SHORT",
                                             "entry_type": "LIMIT", "entry_price": 200, "qty": 1,
                                             "stop_loss": 205, "target_1": 190})
    assert r["status"] == CommandStatus.REJECTED.value and r["rule"] == RiskRule.NO_OVERNIGHT_SHORT.value


def _make_position(svc, sl_gap: float = 2.0) -> Position:  # type: ignore[no-untyped-def]
    """Open an intraday position directly through the broker at the current price."""
    from core.types import RiskVerdict, TradeProposal

    q = svc.data.quote("INFY")
    with svc.session() as s:
        prop = TradeProposal(batch_id=svc.ctx.batch_id, symbol="INFY", mode=Mode.INTRADAY, direction=Direction.LONG,
                             entry_price=q.ltp, entry_type=EntryType.MARKET, stop_loss=round(q.ltp - sl_gap, 2),
                             target_1=round(q.ltp + 3 * sl_gap, 2), created_at=svc.clock.now())
        svc.broker.submit_entry(s, prop, RiskVerdict(approved=True, qty=1, max_qty=1))
        s.commit()
    svc.clock.set(datetime(2026, 11, 25, 10, 3, tzinfo=IST))
    with svc.session() as s:
        svc.monitor.check(s)
        s.commit()
        pos = s.execute(select(Position)).scalar_one()
        return pos


def test_position_commands_tighten_close_and_loosen_blocked(svc) -> None:  # type: ignore[no-untyped-def]
    pos = _make_position(svc)
    r = _cmd(svc, CommandType.TIGHTEN_SL, {"position_id": pos.id, "new_stop_loss": round(pos.stop_loss - 1, 2)})
    assert r["status"] == CommandStatus.REJECTED.value and r["rule"] == RiskRule.SL_LOOSENING_FORBIDDEN.value
    r = _cmd(svc, CommandType.TIGHTEN_SL, {"position_id": pos.id, "new_stop_loss": round(pos.stop_loss + 0.5, 2)})
    assert r["ok"], r
    r = _cmd(svc, CommandType.CLOSE_POSITION, {"position_id": pos.id})
    assert r["ok"], r
    with svc.session() as s:
        assert s.get(Position, pos.id).status == PositionStatus.CLOSED.value
        ok, details = svc.capital.check_ledger_balance(s, svc.ctx.batch_id)
        assert ok, details


def test_kill_switch_squares_off_and_pauses(svc) -> None:  # type: ignore[no-untyped-def]
    pos = _make_position(svc)
    r = _cmd(svc, CommandType.KILL_SWITCH, {"exit_cnc": True, "reason": "test"})
    assert r["ok"] and r["data"]["exits"] == 1
    with svc.session() as s:
        assert s.get(Position, pos.id).status == PositionStatus.CLOSED.value
        assert s.get(EngineStatus, 1).state == EngineState.PAUSED.value


def test_reserve_transfer_needs_confirm(svc) -> None:  # type: ignore[no-untyped-def]
    r = _cmd(svc, CommandType.RESERVE_TRANSFER, {"amount": 10.0, "confirm_text": "yes"})
    assert r["status"] == CommandStatus.REJECTED.value
    r = _cmd(svc, CommandType.RESERVE_TRANSFER, {"amount": 10.0, "confirm_text": "CONFIRM"})
    assert r["status"] == CommandStatus.REJECTED.value and "reserve" in r["message"].lower()   # reserve is 0


def test_new_batch_exits_and_starts_fresh(svc) -> None:  # type: ignore[no-untyped-def]
    _make_position(svc)
    old = svc.ctx.batch_id
    r = _cmd(svc, CommandType.NEW_BATCH, {"name": "Batch B", "starting_capital": 2000.0, "confirm_text": "CONFIRM"})
    assert r["ok"], r
    with svc.session() as s:
        assert s.get(Batch, old).status == BatchStatus.CLOSED.value
        new = s.get(Batch, r["data"]["batch_id"])
        assert new.status == BatchStatus.ACTIVE.value and s.get(CapitalAccount, new.id).trading_cash == 2000.0
        assert s.scalar(select(func.count(Position.id)).where(Position.status == "OPEN")) == 0
        ok, details = svc.capital.check_ledger_balance(s, old)
        assert ok, details


def test_analyze_now_and_run_job(svc) -> None:  # type: ignore[no-untyped-def]
    r = _cmd(svc, CommandType.ANALYZE_NOW, {"symbol": "INFY", "mode": "INTRADAY"})
    assert r["status"] in (CommandStatus.DONE.value, CommandStatus.REJECTED.value)
    assert r["data"]["decision_id"] is not None
    with svc.session() as s:
        assert s.get(DecisionRow, r["data"]["decision_id"]).final_action == "ANALYZED"
    r = _cmd(svc, CommandType.RUN_JOB, {"job": "monitor"})
    assert r["ok"], r


def test_reload_settings(svc) -> None:  # type: ignore[no-untyped-def]
    from config.store import apply_changes

    with svc.session() as s:
        apply_changes(s, {"capital.profit_lock_pct": 40.0}, svc.caps, author="owner", clock=svc.clock)
        s.commit()
    r = _cmd(svc, CommandType.RELOAD_SETTINGS, {})
    assert r["ok"] and svc.settings.capital.profit_lock_pct == 40.0
    assert svc.capital.settings.capital.profit_lock_pct == 40.0


# ---------------------------------------------------------------- replay


@pytest.mark.slow
def test_replay_two_days(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "rt"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    frames = synthetic_frames(["INFY", "TCS"])
    cfg = ReplayConfig(start=REPLAY_DAYS[0], end=REPLAY_DAYS[-1], modes=[Mode.INTRADAY, Mode.SWING],
                       symbols=["INFY", "TCS"], name="t", llm_backend="rules",
                       db_url=f"sqlite:///{(tmp_path / 'replay.db').as_posix()}", step_minutes=5)
    report = ReplayRunner(cfg, frames=frames).run()
    assert report.days == 2 and report.ledger_ok
    from storage.db import create_all_for_url

    sf = create_all_for_url(cfg.db_url)
    with sf() as s:
        assert s.scalar(select(func.count(Run.id)).where(Run.run_type == "INTRADAY_CYCLE")) > 0
        totals = s.execute(select(DailySummary).where(DailySummary.mode == "TOTAL")).scalars().all()
        assert {t.day for t in totals} == set(REPLAY_DAYS)
        # no intraday position survives the session
        assert s.scalar(select(func.count(Position.id)).where(Position.mode == "INTRADAY",
                                                              Position.status == "OPEN")) == 0
        assert s.scalar(select(func.count(Order.id)).where(Order.is_paper.is_(False))) == 0


@pytest.mark.slow
def test_replay_full_trade_lifecycle(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Force a setup + BUY decision; everything downstream is real: risk sizing, AUTO order,
    fill, monitor exits / square-off, EOD capital processing, ledger."""
    from core.enums import AgentStatus, TradeDecision, TrailingMethod
    from core.types import ConfluenceResult, MetaDecision, StrategySignal
    from core.enums import Grade

    def fake_strategies(features, market, mode, settings, now, disabled=frozenset()):  # type: ignore[no-untyped-def]
        if mode != Mode.INTRADAY:
            return []
        atr = features.atr or features.ltp * 0.004
        e = round(features.ltp, 2)
        return [StrategySignal(strategy="vwap_pullback", mode=mode, direction=Direction.LONG, entry_price=e,
                               entry_type=EntryType.MARKET, stop_loss=round(e - atr, 2),
                               target_1=round(e + 2 * atr, 2), target_2=round(e + 3 * atr, 2), strength=90,
                               trailing_rule=TrailingMethod.ATR)]

    def fake_meta(ctx, reports, vote, debate, settings, llm, prompts, *, enabled_modes):  # type: ignore[no-untyped-def]
        sig = ctx.strategy_signals[0]
        return MetaDecision(decision=TradeDecision.BUY, symbol=ctx.symbol, mode=ctx.mode, strategy=sig.strategy,
                            entry_price=sig.entry_price, entry_type=EntryType.MARKET, stop_loss=sig.stop_loss,
                            target_1=sig.target_1, target_2=sig.target_2, trailing_rule=sig.trailing_rule,
                            confidence=85, status=AgentStatus.OK)

    monkeypatch.setattr("strategies.registry.run_strategies", fake_strategies)
    monkeypatch.setattr("strategies.confluence.compute_confluence",
                        lambda *a, **k: ConfluenceResult(score=90.0, grade=Grade.A_PLUS))
    monkeypatch.setattr("agents.meta.meta.run_meta", fake_meta)
    frames = synthetic_frames(["INFY"])
    db_url = f"sqlite:///{(tmp_path / 'life.db').as_posix()}"
    cfg = ReplayConfig(start=REPLAY_DAYS[0], end=REPLAY_DAYS[0], modes=[Mode.INTRADAY], symbols=["INFY"],
                       name="life", llm_backend="rules", db_url=db_url, step_minutes=5)
    report = ReplayRunner(cfg, frames=frames).run()
    assert report.ledger_ok
    assert report.trades >= 1, report
    from storage.db import create_all_for_url
    from storage.models import Trade

    sf = create_all_for_url(db_url)
    with sf() as s:
        assert s.scalar(select(func.count(Position.id)).where(Position.status == "OPEN")) == 0
        reasons = {t.exit_reason for t in s.execute(select(Trade)).scalars()}
        assert reasons <= {"SL", "TARGET", "TRAIL", "TIME", "SQUAREOFF"}
        acct = s.execute(select(CapitalAccount)).scalar_one()
        assert acct.last_eod_date == REPLAY_DAYS[0] and acct.blocked == pytest.approx(0.0, abs=1e-6)
        # risk manager limits respected: at most max_trades_per_day INTRADAY entries
        assert s.scalar(select(func.count(Position.id))) <= 3


def test_cli_set_autonomy_and_router_policy(svc) -> None:  # type: ignore[no-untyped-def]
    from main import set_autonomy
    from core.enums import ExecutionPolicy, Grade

    set_autonomy(svc, "full_auto")
    assert svc.settings.modes.autonomy.value == "FULL_AUTO"
    assert svc.settings.modes.effective_policy(Mode.SWING) == ExecutionPolicy.AUTO
    set_autonomy(svc, "HYBRID")
    m = svc.settings.modes
    assert m.effective_policy(Mode.INTRADAY, grade=Grade.A_PLUS, confidence=90) == ExecutionPolicy.AUTO
    assert m.effective_policy(Mode.INTRADAY, grade=Grade.A, confidence=95) == ExecutionPolicy.MANUAL_APPROVE
    set_autonomy(svc, "HUMAN_IN_THE_LOOP")
    assert svc.queue.settings.modes.autonomy.value == "HUMAN_IN_THE_LOOP"   # pushed into services


def test_full_auto_locked_by_hard_cap(settings) -> None:  # type: ignore[no-untyped-def]
    from config.loader import clamp_to_caps, update_settings, validate_against_caps
    from config.schema import HardCaps

    caps = HardCaps(allow_full_auto=False)
    s = update_settings(settings, {"modes.autonomy": "FULL_AUTO"})
    assert any("allow_full_auto" in v for v in validate_against_caps(s, caps))
    fixed, adj = clamp_to_caps(s, caps)
    assert fixed.modes.autonomy.value == "HUMAN_IN_THE_LOOP" and adj
