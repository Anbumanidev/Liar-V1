"""Deterministic risk manager: every rule in sections 5 / 7.5 / 8 (phase 4)."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from capital.protection import CapitalProtection
from config.loader import update_settings
from config.schema import HardCaps, Settings
from core.clock import IST
from core.enums import (
    BatchStatus,
    Direction,
    DrawdownTier,
    EventType,
    Grade,
    Mode,
    Product,
    RiskRule,
    TradeSource,
)
from core.types import CapitalSnapshot, MarketEvent, OpenPositionView, TradeProposal
from risk.manager import RiskManager
from risk.state import ModeStateView, RiskState

NOW = datetime(2026, 11, 24, 10, 30, tzinfo=IST)


def snap(settings: Settings, *, cash: float = 1000.0, blocked: float = 0.0, reserve: float = 0.0,
         unrealized: float = 0.0, hwm: float | None = None, start: float = 1000.0,
         open_risk: float = 0.0, sizing: float | None = None, mode_blocked: dict[str, float] | None = None,
         halted: bool = False) -> CapitalSnapshot:
    prot = CapitalProtection(settings.risk)
    equity = cash + blocked + unrealized + reserve
    hwm = max(hwm if hwm is not None else start, equity)
    abs_f, trail_f = prot.floors(start, hwm)
    dd = prot.drawdown_pct(equity, hwm)
    tier = prot.tier_for(dd, equity, max(abs_f, trail_f))
    sizing = sizing if sizing is not None else start
    return CapitalSnapshot(
        batch_id=1, as_of=NOW, starting_capital=start, base_capital=start, trading_cash=cash, blocked=blocked,
        reserve=reserve, reserve_peak=reserve, unrealized_pnl=unrealized, total_equity=equity, hwm=hwm,
        drawdown_pct=dd, tier=tier, sizing_capital=sizing,
        mode_sizing_capital={m.value: sizing * settings.mode(m).allocation_pct / 100 for m in Mode},
        mode_blocked=mode_blocked or {}, absolute_floor=abs_f, trailing_floor=trail_f, open_risk=open_risk,
        loss_budget=prot.loss_budget(equity, open_risk, (abs_f, trail_f)), halted=halted or tier == DrawdownTier.HALT,
    )


def state(settings: Settings, **kw) -> RiskState:  # type: ignore[no-untyped-def]
    cap = kw.pop("capital", None) or snap(settings)
    return RiskState(now=kw.pop("now", NOW), batch_id=1, capital=cap, **kw)


def prop(**kw) -> TradeProposal:  # type: ignore[no-untyped-def]
    base = dict(batch_id=1, symbol="INFY", mode=Mode.INTRADAY, direction=Direction.LONG, entry_price=100.0,
                stop_loss=99.0, target_1=101.5, target_2=103.0, strategy="vwap_pullback", confidence=80,
                confluence_score=75.0, grade=Grade.A, atr=1.0, tradeable=True, sector="IT", created_at=NOW)
    base.update(kw)
    return TradeProposal(**base)


def pos(symbol: str = "TCS", mode: Mode = Mode.INTRADAY, sector: str | None = "IT", direction=Direction.LONG,
        entry: float = 100.0, qty: int = 2, ltp: float | None = 100.0, sl: float = 98.0) -> OpenPositionView:
    return OpenPositionView(position_id=9, symbol=symbol, mode=mode,
                            product=Product.MIS if mode == Mode.INTRADAY else Product.CNC,
                            direction=direction, qty=qty, avg_entry=entry, stop_loss=sl, ltp=ltp,
                            margin_blocked=qty * entry, sector=sector, opened_at=NOW,
                            overnight=mode != Mode.INTRADAY)


@pytest.fixture
def rm(settings: Settings, caps: HardCaps) -> RiskManager:
    return RiskManager(settings, caps)


def with_settings(settings: Settings, caps: HardCaps, changes: dict) -> RiskManager:  # type: ignore[type-arg]
    return RiskManager(update_settings(settings, changes), caps)


# ---------------------------------------------------------------- happy path + sizing


def test_intraday_long_approved_and_sized(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(), state(settings))
    assert v.approved, v.message
    # 1% of INTRADAY sizing (50% of 1000 = 500) = Rs 5 risk / Rs 1 SL distance = 5 shares
    assert v.qty == 5 and v.max_qty == 5
    assert v.risk_amount == pytest.approx(5.0)
    assert v.expected_net_at_t1 is not None and v.expected_net_at_t1 > 0
    assert all(c.passed for c in v.checks)


def test_quantity_capped_by_mode_capital_and_position_pct(rm: RiskManager, settings: Settings) -> None:
    # tiny SL distance -> risk allows 50 shares, mode capital 500 allows 5 at Rs 100
    v = rm.evaluate(prop(stop_loss=99.9, target_1=100.2, atr=0.12), state(settings))
    assert v.approved, v.message
    assert v.qty == 5


def test_manual_requested_qty_above_max_rejected(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(source=TradeSource.MANUAL, requested_qty=50, confidence=0, confluence_score=None,
                         grade=None), state(settings))
    assert not v.approved and v.rule == RiskRule.QTY_ABOVE_MAX and v.max_qty == 5
    ok = rm.evaluate(prop(source=TradeSource.MANUAL, requested_qty=3, confidence=0, confluence_score=None,
                          grade=None), state(settings))
    assert ok.approved and ok.qty == 3


def test_reserve_never_used_for_sizing(rm: RiskManager, settings: Settings) -> None:
    cap = snap(settings, cash=50.0, blocked=0.0, reserve=950.0, sizing=1000.0)
    v = rm.evaluate(prop(), state(settings, capital=cap))
    assert not v.approved
    assert v.rule in (RiskRule.INSUFFICIENT_CASH, RiskRule.QTY_BELOW_ONE, RiskRule.MAX_POSITION_PCT)


# ---------------------------------------------------------------- hard rules


def test_no_overnight_short(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(mode=Mode.SWING, direction=Direction.SHORT, stop_loss=101.0, target_1=97.0),
                    state(settings))
    assert not v.approved and v.rule == RiskRule.NO_OVERNIGHT_SHORT


def test_intraday_short_allowed(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(direction=Direction.SHORT, stop_loss=101.0, target_1=98.5, target_2=97.0), state(settings))
    assert v.approved, v.message


def test_mode_disabled(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(mode=Mode.BTST, entry_price=600, stop_loss=590, target_1=625, atr=10),
                    state(settings, now=datetime(2026, 11, 24, 15, 0, tzinfo=IST)))
    assert not v.approved and v.rule == RiskRule.MODE_DISABLED


@pytest.mark.parametrize("hh,mm,ok", [(9, 20, False), (9, 30, True), (14, 45, True), (14, 50, False)])
def test_intraday_entry_window(rm: RiskManager, settings: Settings, hh: int, mm: int, ok: bool) -> None:
    at = datetime(2026, 11, 24, hh, mm, tzinfo=IST)
    v = rm.evaluate(prop(created_at=at), state(settings, now=at))   # a fresh decision at that time
    assert v.approved is ok
    if not ok:
        assert v.rule == RiskRule.ENTRY_WINDOW


def test_timing_can_be_skipped_for_evening_scan(rm: RiskManager, settings: Settings) -> None:
    evening = datetime(2026, 11, 24, 16, 30, tzinfo=IST)
    p = prop(mode=Mode.SWING, entry_price=100, stop_loss=97, target_1=106.5, atr=2.0, grade=Grade.A)
    v = rm.evaluate(p, state(settings, now=evening, market_open=False))
    assert not v.approved and v.rule in (RiskRule.MARKET_CLOSED, RiskRule.ENTRY_WINDOW)
    v2 = rm.evaluate(p, state(settings, now=evening, market_open=False), check_timing=False)
    assert v2.rule not in (RiskRule.MARKET_CLOSED, RiskRule.ENTRY_WINDOW)


@pytest.mark.parametrize(
    "kw,rule",
    [
        ({"stop_loss": 100.5}, RiskRule.SL_WRONG_SIDE),
        ({"stop_loss": 99.9, "target_1": 100.2}, RiskRule.SL_TOO_TIGHT),   # 0.1 < 0.3 x 1.5
        ({"stop_loss": 95.0, "target_1": 108.0}, RiskRule.SL_TOO_WIDE),    # 5 > 3 x 1.5
        ({"target_1": 101.0}, RiskRule.RR_BELOW_MIN),                      # 1R < 1.5R
        ({"target_1": 99.5}, RiskRule.TARGET_INVALID),
        ({"tradeable": False}, RiskRule.TRADEABLE_FALSE),
        ({"confidence": 60}, RiskRule.CONFIDENCE_BELOW_MIN),
        ({"confluence_score": 50.0}, RiskRule.CONFLUENCE_BELOW_MIN),
        ({"grade": Grade.C}, RiskRule.GRADE_NOT_ALLOWED),
        ({"upper_circuit": 101.0, "ltp": 100.0}, RiskRule.NEAR_CIRCUIT),
        ({"lower_circuit": 99.0, "ltp": 100.0}, RiskRule.NEAR_CIRCUIT),
    ],
)
def test_single_rule_rejections(rm: RiskManager, settings: Settings, kw: dict, rule: RiskRule) -> None:  # type: ignore[type-arg]
    v = rm.evaluate(prop(**kw), state(settings))
    assert not v.approved and v.rule == rule, v.message
    assert v.message.startswith(rule.value)


def test_blacklist(settings: Settings, caps: HardCaps) -> None:
    rm = with_settings(settings, caps, {"universe.blacklist": ["INFY"]})
    v = rm.evaluate(prop(), state(settings))
    assert v.rule == RiskRule.BLACKLISTED


def test_manual_trade_skips_agent_gates_but_not_risk_rules(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(source=TradeSource.MANUAL, confidence=0, confluence_score=None, grade=None),
                    state(settings, engine_paused=True))
    assert v.approved, v.message
    bad = rm.evaluate(prop(source=TradeSource.MANUAL, confidence=0, confluence_score=None, grade=None,
                           stop_loss=100.5), state(settings))
    assert bad.rule == RiskRule.SL_WRONG_SIDE


def test_engine_paused_blocks_system(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(), state(settings, engine_paused=True))
    assert v.rule == RiskRule.ENGINE_PAUSED


# ---------------------------------------------------------------- portfolio / daily rules


def test_small_capital_max_two_open_positions(rm: RiskManager, settings: Settings) -> None:
    open_ = [pos("TCS", sector="IT"), pos("HDFCBANK", sector="BANK")]
    v = rm.evaluate(prop(sector="AUTO"), state(settings, open_positions=open_))
    assert v.rule == RiskRule.MAX_OPEN_POSITIONS


def test_sector_limit(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(), state(settings, open_positions=[pos("TCS", sector="IT")]))
    assert v.rule == RiskRule.SECTOR_LIMIT


def test_correlation_limit(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(sector="AUTO"), state(settings, open_positions=[pos("TCS", sector="IT")],
                                               correlations={"TCS": 0.85}))
    assert v.rule == RiskRule.CORRELATION_LIMIT


def test_duplicate_and_averaging_down(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(), state(settings, open_positions=[pos("INFY", sector="X", ltp=97.0)]))
    assert v.rule == RiskRule.AVERAGING_DOWN_FORBIDDEN
    v = rm.evaluate(prop(), state(settings, open_positions=[pos("INFY", sector="X", ltp=103.0)]))
    assert v.rule == RiskRule.DUPLICATE_POSITION


def test_max_trades_per_day(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(), state(settings, trades_today={Mode.INTRADAY: 3}))
    assert v.rule == RiskRule.MAX_TRADES_PER_DAY


def test_daily_loss_limit(rm: RiskManager, settings: Settings) -> None:
    cap = snap(settings, cash=970.0)
    v = rm.evaluate(prop(), state(settings, capital=cap, realized_today_total=-30.0,
                                  realized_today={Mode.INTRADAY: -30.0}))
    assert v.rule == RiskRule.DAILY_LOSS_LIMIT


def test_mode_pauses(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(), state(settings, mode_states={Mode.INTRADAY: ModeStateView(paused=True)}))
    assert v.rule == RiskRule.MODE_PAUSED
    v = rm.evaluate(prop(), state(settings, mode_states={
        Mode.INTRADAY: ModeStateView(paused_until=datetime(2026, 11, 24, 11, 0, tzinfo=IST))}))
    assert v.rule == RiskRule.CONSECUTIVE_LOSS_PAUSE
    v = rm.evaluate(prop(), state(settings, mode_states={
        Mode.INTRADAY: ModeStateView(paused_until=datetime(2026, 11, 24, 10, 0, tzinfo=IST))}))
    assert v.approved
    v = rm.evaluate(prop(), state(settings, mode_states={Mode.INTRADAY: ModeStateView(stopped_for_day=date(2026, 11, 24))}))
    assert v.rule == RiskRule.DAILY_PROFIT_LOCK


def test_intraday_daily_profit_lock_only_a_plus_half_size(rm: RiskManager, settings: Settings) -> None:
    st = state(settings, capital=snap(settings, cash=1025.0, sizing=1000.0), realized_today={Mode.INTRADAY: 25.0},
               realized_today_total=25.0)
    v = rm.evaluate(prop(grade=Grade.A), st)
    assert v.rule == RiskRule.DAILY_PROFIT_LOCK
    v = rm.evaluate(prop(grade=Grade.A_PLUS), st)
    assert v.approved and v.size_multiplier == 0.5 and v.qty == 2   # 2.5 risk / 1.0 -> 2


def test_strategy_auto_disabled(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(), state(settings, disabled_strategies={"vwap_pullback:INTRADAY"}))
    assert v.rule == RiskRule.STRATEGY_DISABLED


# ---------------------------------------------------------------- capital protection


def test_halt_blocks_everything(rm: RiskManager, settings: Settings) -> None:
    cap = snap(settings, cash=905.0)   # equity 905 <= 910 absolute floor
    v = rm.evaluate(prop(), state(settings, capital=cap))
    assert v.rule == RiskRule.HALTED
    m = rm.evaluate(prop(source=TradeSource.MANUAL, grade=None, confluence_score=None), state(settings, capital=cap))
    assert m.rule == RiskRule.HALTED


def test_caution_tier_grades_and_half_size(rm: RiskManager, settings: Settings) -> None:
    cap = snap(settings, cash=960.0, hwm=1000.0)   # 4% drawdown -> CAUTION
    assert cap.tier == DrawdownTier.CAUTION
    v = rm.evaluate(prop(grade=Grade.B), state(settings, capital=cap))
    assert v.rule == RiskRule.DRAWDOWN_TIER
    v = rm.evaluate(prop(grade=Grade.A), state(settings, capital=cap))
    assert v.approved and v.size_multiplier == 0.5 and v.qty == 2


def test_defensive_tier_one_position_no_overnight(rm: RiskManager, settings: Settings) -> None:
    cap = snap(settings, cash=930.0, hwm=1000.0)   # 7% -> DEFENSIVE
    assert cap.tier == DrawdownTier.DEFENSIVE
    v = rm.evaluate(prop(grade=Grade.A), state(settings, capital=cap))
    assert v.rule == RiskRule.DRAWDOWN_TIER
    v = rm.evaluate(prop(grade=Grade.A_PLUS, sector="AUTO"), state(settings, capital=cap,
                                                                     open_positions=[pos("TCS", sector="IT")]))
    assert v.rule == RiskRule.MAX_OPEN_POSITIONS
    sw = rm.evaluate(prop(mode=Mode.SWING, grade=Grade.A_PLUS, entry_price=100, stop_loss=97, target_1=106.5,
                          target_2=110.0, atr=2.0), state(settings, capital=cap, now=datetime(2026, 11, 24, 9, 45, tzinfo=IST)))
    assert sw.rule == RiskRule.DRAWDOWN_TIER


def test_loss_budget_exhausted(rm: RiskManager, settings: Settings) -> None:
    cap = snap(settings, cash=1000.0, open_risk=89.9)   # budget 0.1 -> /3 -> 0.03
    v = rm.evaluate(prop(), state(settings, capital=cap))
    assert v.rule == RiskRule.LOSS_BUDGET_EXHAUSTED


def test_overnight_sizing_is_gap_adjusted(rm: RiskManager, settings: Settings, caps: HardCaps) -> None:
    rm2 = with_settings(settings, caps, {"risk.max_charges_pct_of_value": 10.0, "charges.cnc.dp_charge_per_scrip": 0.0})
    # SWING: SL 1% away but gap buffer 3% -> effective risk 3/share; budget 90/3 = 30 -> <= 9 shares
    # normal risk 1% of 500 = 5 / 1 = 5 shares; capital 500/100 = 5 -> 5 shares, budget not binding
    v = rm2.evaluate(prop(mode=Mode.SWING, entry_price=100, stop_loss=99, target_1=102.5, target_2=104, atr=1.0),
                     state(settings, now=datetime(2026, 11, 24, 9, 45, tzinfo=IST)))
    assert v.approved, v.message
    assert v.effective_risk >= v.qty * 3.0
    # with a tight loss budget the gap-adjusted risk binds
    cap = snap(settings, open_risk=80.0)   # budget 10 -> 3.33 -> 1 share at 3/share (+charges)
    v2 = rm2.evaluate(prop(mode=Mode.SWING, entry_price=100, stop_loss=99, target_1=102.5, atr=1.0, sector="AUTO"),
                      state(settings, capital=cap, now=datetime(2026, 11, 24, 9, 45, tzinfo=IST)))
    assert v2.qty <= 1


def test_cnc_small_trade_rejected_by_charges(rm: RiskManager, settings: Settings) -> None:
    # 1 share (risk Rs 5 / SL distance 4): gross Rs 10 at T1 < ~Rs 17 charges (DP Rs 15.34 dominates)
    v = rm.evaluate(prop(mode=Mode.SWING, entry_price=400, stop_loss=396, target_1=410, target_2=430, atr=4.0),
                    state(settings, now=datetime(2026, 11, 24, 9, 45, tzinfo=IST)))
    assert not v.approved and v.rule == RiskRule.CHARGES_EXCEED_EDGE
    assert "% of position value" in v.message or "expected net" in v.message


def test_overnight_exposure(rm: RiskManager, settings: Settings, caps: HardCaps) -> None:
    rm2 = with_settings(settings, caps, {"risk.max_charges_pct_of_value": 10.0, "risk.small_max_open_positions": 5,
                                         "risk.max_positions_per_sector": 5})
    held = pos("TCS", mode=Mode.SWING, sector="IT", qty=5, entry=100.0, ltp=100.0, sl=97.0)
    cap = snap(settings, cash=500.0, blocked=500.0, mode_blocked={"SWING": 500.0})
    v = rm2.evaluate(prop(mode=Mode.SWING, entry_price=100, stop_loss=99, target_1=102.5, atr=1.0),
                     state(settings, capital=cap, open_positions=[held],
                           now=datetime(2026, 11, 24, 9, 45, tzinfo=IST)))
    assert not v.approved and v.rule in (RiskRule.OVERNIGHT_EXPOSURE, RiskRule.ALLOCATION_EXHAUSTED)


def test_event_in_window_blocks_overnight(rm: RiskManager, settings: Settings, caps: HardCaps) -> None:
    rm2 = with_settings(settings, caps, {"risk.max_charges_pct_of_value": 10.0, "charges.cnc.dp_charge_per_scrip": 0.0})
    ev = MarketEvent(symbol="INFY", event_type=EventType.RESULTS, event_date=date(2026, 11, 27))
    v = rm2.evaluate(prop(mode=Mode.SWING, entry_price=100, stop_loss=99, target_1=102.5, atr=1.0,
                          events_in_window=[ev]), state(settings, now=datetime(2026, 11, 24, 9, 45, tzinfo=IST)))
    assert v.rule == RiskRule.EVENT_IN_WINDOW
    other = MarketEvent(symbol="TCS", event_type=EventType.RESULTS, event_date=date(2026, 11, 27))
    v2 = rm2.evaluate(prop(mode=Mode.SWING, entry_price=100, stop_loss=99, target_1=102.5, atr=1.0,
                           events_in_window=[other]), state(settings, now=datetime(2026, 11, 24, 9, 45, tzinfo=IST)))
    assert v2.approved, v2.message


def test_leverage_only_for_mis(settings: Settings, caps: HardCaps) -> None:
    rm = with_settings(settings, caps, {"risk.mis_leverage": 4.0})
    assert rm.leverage_for(Mode.INTRADAY) == 4.0
    for m in (Mode.BTST, Mode.SWING, Mode.POSITIONAL):
        assert rm.leverage_for(m) == 1.0


def test_batch_not_active(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(), state(settings, batch_status=BatchStatus.PAUSED))
    assert v.rule == RiskRule.BATCH_NOT_ACTIVE


def test_all_checks_recorded(rm: RiskManager, settings: Settings) -> None:
    v = rm.evaluate(prop(stop_loss=100.5, confidence=10), state(settings))
    failed = [c.rule for c in v.checks if not c.passed]
    assert RiskRule.SL_WRONG_SIDE in failed and RiskRule.CONFIDENCE_BELOW_MIN in failed
    assert v.rule == RiskRule.CONFIDENCE_BELOW_MIN   # first failure in evaluation order


def test_max_affordable_price(rm: RiskManager, settings: Settings) -> None:
    st = state(settings)
    # INTRADAY: min(mode free 500, cash 1000, 60% of 1000 = 600) = 500
    assert rm.max_affordable_price(st, Mode.INTRADAY) == pytest.approx(500.0)


# ---------------------------------------------------------------- conversion


def test_conversion_rules(rm: RiskManager, settings: Settings) -> None:
    p = pos("INFY", mode=Mode.INTRADAY, qty=2, entry=100.0, ltp=101.0, sl=99.0)
    cap = snap(settings, cash=800.0, blocked=200.0, mode_blocked={"INTRADAY": 200.0})
    ok = rm.evaluate_conversion(p, Mode.SWING, state(settings, capital=cap, open_positions=[p],
                                                     now=datetime(2026, 11, 24, 15, 0, tzinfo=IST)))
    assert ok.approved, ok.message
    late = rm.evaluate_conversion(p, Mode.SWING, state(settings, capital=cap, open_positions=[p],
                                                       now=datetime(2026, 11, 24, 15, 11, tzinfo=IST)))
    assert late.rule == RiskRule.CONVERSION_NOT_ALLOWED
    short = pos("INFY", mode=Mode.INTRADAY, direction=Direction.SHORT, sl=102.0)
    assert rm.evaluate_conversion(short, Mode.SWING, state(settings)).rule == RiskRule.CONVERSION_NOT_ALLOWED
    assert rm.evaluate_conversion(p, Mode.POSITIONAL, state(settings)).rule == RiskRule.MODE_DISABLED


# ---------------------------------------------------------------- floor guarantee (section 8)


@pytest.mark.parametrize("seed", range(60))
def test_no_sequence_of_sl_hits_breaches_floors(settings: Settings, caps: HardCaps, seed: int) -> None:
    """Trades sized by the risk manager, then stopped out (overnight ones gapping by up to the
    gap buffer), can never push equity below either floor."""
    s = update_settings(settings, {"risk.max_charges_pct_of_value": 10.0, "risk.small_max_open_positions": 3,
                                   "risk.max_positions_per_sector": 3, "modes.modes.INTRADAY.max_trades_per_day": 50,
                                   "modes.modes.SWING.max_trades_per_day": 50})
    rm = RiskManager(s, caps)
    prot = CapitalProtection(s.risk, s.charges)
    rng = np.random.default_rng(seed)
    start = 1000.0
    cash, hwm = start, start
    open_pos: list[OpenPositionView] = []
    for step in range(120):
        blocked = sum(p.margin_blocked for p in open_pos)
        equity = cash + blocked
        orisk = prot.open_risk(open_pos)
        abs_f, trail_f = prot.floors(start, hwm)
        assert equity >= max(abs_f, trail_f) - 1e-6, f"floor breached at step {step}"
        cap = snap(s, cash=cash, blocked=blocked, hwm=hwm, open_risk=orisk,
                   mode_blocked={m.value: sum(p.margin_blocked for p in open_pos if p.mode == m) for m in Mode})
        if cap.tier == DrawdownTier.HALT:
            break
        if open_pos and (rng.random() < 0.55 or len(open_pos) >= 3):
            p = open_pos.pop(int(rng.integers(len(open_pos))))
            win = rng.random() < 0.3
            if win:
                exit_px = p.avg_entry + 2 * (p.avg_entry - p.stop_loss)
            elif p.overnight:   # gap through the SL, bounded by the gap buffer
                worst = p.avg_entry * (1 - s.risk.gap_buffer_pct / 100.0)
                exit_px = min(p.stop_loss, float(rng.uniform(worst, p.stop_loss)))
            else:
                exit_px = p.stop_loss
            from paper_engine.charges import round_trip_charges
            ch = round_trip_charges(p.product, p.direction, p.qty, p.avg_entry, exit_px, s.charges).total
            cash += p.margin_blocked + (exit_px - p.avg_entry) * p.qty - ch
            hwm = max(hwm, cash + sum(q.margin_blocked for q in open_pos))
            continue
        mode = Mode.SWING if rng.random() < 0.4 else Mode.INTRADAY
        entry = float(rng.uniform(20, 300))
        sl_pct = float(rng.uniform(0.004, 0.02))
        atr = entry * sl_pct / float(rng.uniform(0.5, 2.5))
        proposal = prop(symbol=f"S{step}", mode=mode, sector=f"SEC{step}", entry_price=entry,
                        stop_loss=entry * (1 - sl_pct), target_1=entry * (1 + 2.5 * sl_pct),
                        target_2=entry * (1 + 4 * sl_pct), atr=atr, grade=Grade.A_PLUS, confidence=90,
                        confluence_score=90.0)
        now = datetime(2026, 11, 24, 9, 45 if mode == Mode.SWING else 11, tzinfo=IST)
        v = rm.evaluate(proposal, RiskState(now=now, batch_id=1, capital=cap, open_positions=open_pos))
        if v.approved:
            margin = v.qty * entry
            cash -= margin
            open_pos.append(OpenPositionView(
                position_id=step, symbol=proposal.symbol, mode=mode,
                product=Product.MIS if mode == Mode.INTRADAY else Product.CNC, direction=Direction.LONG,
                qty=v.qty, avg_entry=entry, stop_loss=proposal.stop_loss, ltp=entry, margin_blocked=margin,
                sector=proposal.sector, opened_at=now, overnight=mode != Mode.INTRADAY))
    # stop out everything at the end (worst case)
    for p in open_pos:
        exit_px = p.avg_entry * (1 - s.risk.gap_buffer_pct / 100.0) if p.overnight else p.stop_loss
        from paper_engine.charges import round_trip_charges
        ch = round_trip_charges(p.product, p.direction, p.qty, p.avg_entry, exit_px, s.charges).total
        cash += p.margin_blocked + (exit_px - p.avg_entry) * p.qty - ch
    abs_f, trail_f = prot.floors(start, hwm)
    assert cash >= max(abs_f, trail_f) - 1e-6
