"""Execution-quality gates added after the 2026-11-25 post-mortem.

The three live INTRADAY trades of 2026-11-25 all lost. Each is replayed here with its REAL
numbers (from runtime/trading.db) and must now be refused, while a fresh, well-placed setup
still passes. Rules: SIGNAL_EXPIRED, PRICE_MOVED, AGENT_QUORUM, DEBATE_AGAINST, NO_ROOM,
REENTRY_COOLDOWN (+ the fill-time check in the paper broker).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from capital.protection import CapitalProtection
from config.loader import load_default_settings, load_hard_caps, update_settings
from config.schema import Settings
from core.clock import IST
from core.enums import Direction, DrawdownTier, EntryType, Grade, Mode, RiskRule, TradeSource
from core.types import CapitalSnapshot, TradeProposal
from risk.manager import RiskManager
from risk.state import RiskState

DAY = datetime(2026, 11, 25, tzinfo=IST)


def at(hh: int, mm: int, ss: int = 0) -> datetime:
    return DAY.replace(hour=hh, minute=mm, second=ss)


@pytest.fixture
def settings() -> Settings:
    return load_default_settings()


@pytest.fixture
def rm(settings: Settings) -> RiskManager:
    return RiskManager(settings, load_hard_caps())


def snap(settings: Settings, now: datetime) -> CapitalSnapshot:
    prot = CapitalProtection(settings.risk)
    abs_f, trail_f = prot.floors(1000.0, 1000.0)
    return CapitalSnapshot(
        batch_id=1, as_of=now, starting_capital=1000.0, base_capital=1000.0, trading_cash=1000.0, blocked=0.0,
        reserve=0.0, reserve_peak=0.0, unrealized_pnl=0.0, total_equity=1000.0, hwm=1000.0, drawdown_pct=0.0,
        tier=DrawdownTier.NORMAL, sizing_capital=1000.0,
        mode_sizing_capital={m.value: 1000.0 * settings.mode(m).allocation_pct / 100 for m in Mode},
        mode_blocked={}, absolute_floor=abs_f, trailing_floor=trail_f, open_risk=0.0,
        loss_budget=prot.loss_budget(1000.0, 0.0, (abs_f, trail_f)), halted=False,
    )


def state(settings: Settings, now: datetime, **kw) -> RiskState:  # type: ignore[no-untyped-def]
    return RiskState(now=now, batch_id=1, capital=snap(settings, now), **kw)


def failed_rules(verdict) -> set[RiskRule]:  # type: ignore[no-untyped-def]
    return {c.rule for c in verdict.checks if not c.passed}


# --------------------------------------------------------------------------- the real trades


def itc_proposal() -> TradeProposal:
    """Decision 9: ITC relative_strength_momentum, data read 11:15:14, decided 11:23:49 (515 s)."""
    return TradeProposal(
        batch_id=1, symbol="ITC", mode=Mode.INTRADAY, direction=Direction.LONG, entry_price=310.5,
        entry_type=EntryType.MARKET, stop_loss=309.45, target_1=312.08, target_2=313.65,
        strategy="relative_strength_momentum", confidence=85, confluence_score=81.3, grade=Grade.A, atr=0.8425,
        tradeable=True, sector="FMCG", ltp=311.05, created_at=at(11, 15, 14), valid_until=at(11, 30, 14),
        failed_agents_pct=0.0, debate_bull=88, debate_bear=85, room_r=94.29,
    )


def bel1_proposal() -> TradeProposal:
    """Decision 148: BEL ema_9_21_pullback; 9/9 agents timed out; data 12:15, decided 12:40."""
    return TradeProposal(
        batch_id=1, symbol="BEL", mode=Mode.INTRADAY, direction=Direction.LONG, entry_price=404.3,
        entry_type=EntryType.MARKET, stop_loss=401.95, target_1=409.9, target_2=412.25,
        strategy="ema_9_21_pullback", confidence=70, confluence_score=73.43, grade=Grade.B, atr=0.8462,
        tradeable=True, sector="DEFENCE", ltp=402.4, created_at=at(12, 15), valid_until=at(12, 30),
        failed_agents_pct=100.0, debate_bull=70, debate_bear=85, room_r=0.04,
    )


def bel2_proposal() -> TradeProposal:
    """Decision 211: BEL supertrend_adx_15m; data 12:35, decided 12:56, 13 min after BEL's stop."""
    return TradeProposal(
        batch_id=1, symbol="BEL", mode=Mode.INTRADAY, direction=Direction.LONG, entry_price=402.95,
        entry_type=EntryType.MARKET, stop_loss=401.55, target_1=405.07, target_2=406.55,
        strategy="supertrend_adx_15m", confidence=70, confluence_score=70.49, grade=Grade.B, atr=0.8236,
        tradeable=True, sector="DEFENCE", ltp=402.5, created_at=at(12, 35), valid_until=at(12, 50),
        failed_agents_pct=100.0, debate_bull=70, debate_bear=95, room_r=0.13,
    )


def test_itc_trade_is_now_refused(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(itc_proposal(), state(settings, at(11, 23, 49)))
    assert not v.approved
    bad = failed_rules(v)
    assert RiskRule.SIGNAL_EXPIRED in bad          # 8.6 min old > 6 min
    assert RiskRule.PRICE_MOVED in bad             # R:R at 311.05 = 0.66 < 1.2
    assert RiskRule.AGENT_QUORUM not in bad and RiskRule.DEBATE_AGAINST not in bad and RiskRule.NO_ROOM not in bad


def test_bel_first_trade_is_now_refused(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(bel1_proposal(), state(settings, at(12, 40)))
    assert not v.approved
    bad = failed_rules(v)
    assert {RiskRule.AGENT_QUORUM, RiskRule.NO_ROOM, RiskRule.SIGNAL_EXPIRED, RiskRule.PRICE_MOVED} <= bad
    assert RiskRule.DEBATE_AGAINST in bad          # 85 - 70 = 15 >= 10


def test_bel_reentry_is_now_refused(rm: RiskManager, settings: Settings) -> None:
    st = state(settings, at(12, 56, 28), last_sl_exit={"BEL:INTRADAY": at(12, 43, 47)})
    v = rm.evaluate(bel2_proposal(), st)
    assert not v.approved
    bad = failed_rules(v)
    assert {RiskRule.REENTRY_COOLDOWN, RiskRule.AGENT_QUORUM, RiskRule.DEBATE_AGAINST, RiskRule.NO_ROOM,
            RiskRule.SIGNAL_EXPIRED} <= bad
    msg = next(c.detail for c in v.checks if c.rule == RiskRule.REENTRY_COOLDOWN)
    assert "13 min ago" in msg


def test_entry_windows_skip_the_midday_lull(rm: RiskManager, settings: Settings) -> None:
    p = itc_proposal().model_copy(update={"created_at": at(12, 30), "valid_until": None, "ltp": None})
    v = rm.evaluate(p, state(settings, at(12, 30)))
    assert v.rule == RiskRule.ENTRY_WINDOW


# --------------------------------------------------------------------------- a good setup still passes


def good(now: datetime, **kw) -> TradeProposal:  # type: ignore[no-untyped-def]
    base = dict(batch_id=1, symbol="INFY", mode=Mode.INTRADAY, direction=Direction.LONG, entry_price=100.0,
                entry_type=EntryType.MARKET, stop_loss=99.0, target_1=101.6, target_2=103.0,
                strategy="vwap_pullback", confidence=80, confluence_score=78.0, grade=Grade.A, atr=1.0,
                tradeable=True, sector="IT", ltp=100.05, created_at=now - timedelta(minutes=2),
                valid_until=now + timedelta(minutes=10), failed_agents_pct=10.0, debate_bull=75, debate_bear=60,
                room_r=2.5)
    base.update(kw)
    return TradeProposal(**base)


def test_fresh_well_placed_setup_passes(rm: RiskManager, settings: Settings) -> None:
    now = at(10, 30)
    v = rm.evaluate(good(now), state(settings, now))
    assert v.approved, v.message
    for rule in (RiskRule.SIGNAL_EXPIRED, RiskRule.PRICE_MOVED, RiskRule.AGENT_QUORUM, RiskRule.DEBATE_AGAINST,
                 RiskRule.NO_ROOM):
        assert any(c.rule == rule and c.passed for c in v.checks), rule


@pytest.mark.parametrize("update,rule", [
    ({"failed_agents_pct": 40.0}, RiskRule.AGENT_QUORUM),
    ({"debate_bull": 60, "debate_bear": 70}, RiskRule.DEBATE_AGAINST),
    ({"room_r": 0.4}, RiskRule.NO_ROOM),
    ({"valid_until": datetime(2026, 11, 25, 10, 29, tzinfo=IST)}, RiskRule.SIGNAL_EXPIRED),
    ({"created_at": datetime(2026, 11, 25, 10, 23, tzinfo=IST)}, RiskRule.SIGNAL_EXPIRED),   # 7 min old
    ({"ltp": 98.9}, RiskRule.PRICE_MOVED),                                                      # through the stop
    ({"ltp": 100.9}, RiskRule.PRICE_MOVED),                                                     # R:R 0.37 at live
    ({"ltp": 99.5}, RiskRule.PRICE_MOVED),                                                      # stop 0.5 ATR away
])
def test_each_gate_refuses_on_its_own(rm: RiskManager, settings: Settings, update: dict, rule: RiskRule) -> None:
    now = at(10, 30)
    v = rm.evaluate(good(now, **update), state(settings, now))
    assert not v.approved and v.rule == rule, v.message


def test_unknown_inputs_skip_the_gates(rm: RiskManager, settings: Settings) -> None:
    now = at(10, 30)
    p = good(now, failed_agents_pct=None, debate_bull=None, debate_bear=None, room_r=None, valid_until=None, ltp=None)
    v = rm.evaluate(p, state(settings, now))
    assert v.approved, v.message


def test_manual_trades_are_not_subject_to_agent_gates(rm: RiskManager, settings: Settings) -> None:
    now = at(10, 30)
    p = good(now, source=TradeSource.MANUAL, failed_agents_pct=100.0, debate_bull=0, debate_bear=100, room_r=0.1,
             confidence=0, confluence_score=None, grade=None)
    v = rm.evaluate(p, state(settings, now))
    assert v.approved, v.message


def test_limit_entry_only_checks_stop_and_target(rm: RiskManager, settings: Settings) -> None:
    now = at(10, 30)
    ok = rm.evaluate(good(now, entry_type=EntryType.LIMIT, ltp=100.9), state(settings, now))
    assert ok.approved, ok.message        # a LIMIT order fills at its own price
    gone = rm.evaluate(good(now, entry_type=EntryType.LIMIT, ltp=101.7), state(settings, now))
    assert gone.rule == RiskRule.PRICE_MOVED


def test_gates_can_be_turned_off(settings: Settings) -> None:
    s = update_settings(settings, {"risk.debate_veto_enabled": False, "risk.min_room_r": 0.0,
                                   "risk.max_failed_agents_pct": 100.0,
                                   "modes.modes.INTRADAY.reentry_cooldown_minutes": 0})
    rm = RiskManager(s, load_hard_caps())
    now = at(10, 30)
    p = good(now, failed_agents_pct=100.0, debate_bull=10, debate_bear=90, room_r=0.1)
    st = state(s, now, last_sl_exit={"INFY:INTRADAY": now - timedelta(minutes=5)})
    v = rm.evaluate(p, st)
    assert v.approved, v.message


def test_cooldown_expires(rm: RiskManager, settings: Settings) -> None:
    now = at(14, 0)
    st = state(settings, now, last_sl_exit={"INFY:INTRADAY": now - timedelta(minutes=61)})
    assert rm.evaluate(good(now), st).approved
    st2 = state(settings, now, last_sl_exit={"INFY:INTRADAY": now - timedelta(minutes=30)})
    assert rm.evaluate(good(now), st2).rule == RiskRule.REENTRY_COOLDOWN
    # another mode's stop does not block INTRADAY
    st3 = state(settings, now, last_sl_exit={"INFY:SWING": now - timedelta(minutes=5)})
    assert rm.evaluate(good(now), st3).approved
