"""Capital protection tests (section 8): floors, tiers, gap-adjusted risk, budget."""

from __future__ import annotations

from datetime import datetime

import pytest

from capital.protection import CapitalProtection, TierConstraints
from config.schema import Settings
from core.clock import IST
from core.enums import Direction, DrawdownTier, Grade, Mode, Product
from core.types import OpenPositionView


@pytest.fixture
def prot(settings: Settings) -> CapitalProtection:
    return CapitalProtection(settings.risk)


def _pos(
    *,
    qty: int = 10,
    entry: float = 100.0,
    sl: float = 98.0,
    direction: Direction = Direction.LONG,
    mode: Mode = Mode.INTRADAY,
    ltp: float | None = None,
    entry_charges: float = 0.0,
    overnight: bool = False,
) -> OpenPositionView:
    product = Product.MIS if mode == Mode.INTRADAY else Product.CNC
    return OpenPositionView(
        position_id=1, symbol="TEST", mode=mode, product=product, direction=direction, qty=qty,
        avg_entry=entry, stop_loss=sl, ltp=ltp, margin_blocked=qty * entry, entry_charges=entry_charges,
        opened_at=datetime(2026, 11, 24, 10, 0, tzinfo=IST), overnight=overnight,
    )


# ----------------------------------------------------------------------------- floors


def test_floors_absolute_and_trailing(prot: CapitalProtection) -> None:
    abs_f, trail_f = prot.floors(1000.0, 1000.0)
    assert abs_f == pytest.approx(910.0)
    assert trail_f == pytest.approx(910.0)
    abs_f, trail_f = prot.floors(1000.0, 1200.0)
    assert abs_f == pytest.approx(910.0)
    assert trail_f == pytest.approx(1092.0)


def test_stricter_floor_wins(prot: CapitalProtection) -> None:
    assert prot.effective_floor(1000.0, 1200.0) == pytest.approx(1092.0)  # trailing stricter
    # HWM can never be below starting in practice, but the absolute floor still binds if it were
    assert prot.effective_floor(1000.0, 900.0) == pytest.approx(910.0)
    assert prot.loss_budget(1100.0, 0.0, prot.floors(1000.0, 1200.0)) == pytest.approx(8.0)
    assert prot.loss_budget(1000.0, 0.0, prot.floors(1000.0, 1000.0)) == pytest.approx(90.0)


def test_drawdown_pct(prot: CapitalProtection) -> None:
    assert prot.drawdown_pct(1000.0, 1000.0) == 0.0
    assert prot.drawdown_pct(1100.0, 1000.0) == 0.0
    assert prot.drawdown_pct(970.0, 1000.0) == pytest.approx(3.0)
    assert prot.drawdown_pct(5.0, 0.0) == 0.0


# ----------------------------------------------------------------------------- tiers


@pytest.mark.parametrize(
    ("equity", "tier"),
    [
        (1000.0, DrawdownTier.NORMAL),
        (971.0, DrawdownTier.NORMAL),      # 2.9 %
        (970.0, DrawdownTier.CAUTION),     # exactly 3 %
        (941.0, DrawdownTier.CAUTION),     # 5.9 %
        (940.0, DrawdownTier.DEFENSIVE),   # exactly 6 %
        (911.0, DrawdownTier.DEFENSIVE),   # 8.9 %
        (910.0, DrawdownTier.HALT),        # exactly 9 % == floor
        (850.0, DrawdownTier.HALT),
    ],
)
def test_tier_transitions(prot: CapitalProtection, equity: float, tier: DrawdownTier) -> None:
    dd, got = prot.evaluate(1000.0, 1000.0, equity)
    assert got == tier, (equity, dd)


def test_halt_on_floor_even_with_small_drawdown(prot: CapitalProtection) -> None:
    # Equity at/below the floor halts regardless of the computed drawdown number.
    assert prot.tier_for(1.0, 909.0, 910.0) == DrawdownTier.HALT
    assert prot.tier_for(1.0, 910.0, 910.0) == DrawdownTier.HALT
    assert prot.tier_for(1.0, 911.0, 910.0) == DrawdownTier.NORMAL


def test_trailing_tiers_after_new_hwm(prot: CapitalProtection) -> None:
    # HWM 1200: 4 % down = 1152 -> CAUTION even though above starting capital
    assert prot.evaluate(1000.0, 1200.0, 1152.0)[1] == DrawdownTier.CAUTION
    assert prot.evaluate(1000.0, 1200.0, 1092.0)[1] == DrawdownTier.HALT


def test_tier_constraints(prot: CapitalProtection) -> None:
    n = prot.tier_constraints(DrawdownTier.NORMAL)
    assert n == TierConstraints(DrawdownTier.NORMAL, 1.0, tuple(Grade), None, True, False)
    c = prot.tier_constraints(DrawdownTier.CAUTION)
    assert c.size_mult == 0.5 and set(c.allowed_grades) == {Grade.A_PLUS, Grade.A}
    assert c.grade_allowed("A") and not c.grade_allowed(Grade.B)
    assert c.allow_new_overnight and not c.halted and c.max_open_positions is None
    d = prot.tier_constraints(DrawdownTier.DEFENSIVE)
    assert d.size_mult == 0.25 and d.allowed_grades == (Grade.A_PLUS,)
    assert d.max_open_positions == 1 and d.allow_new_overnight is False and not d.halted
    h = prot.tier_constraints(DrawdownTier.HALT)
    assert h.halted and h.size_mult == 0.0 and h.allowed_grades == () and h.max_open_positions == 0


# ----------------------------------------------------------------------------- position risk


def test_sl_risk_long_and_short(prot: CapitalProtection) -> None:
    assert prot.position_risk(_pos(qty=10, entry=100, sl=98)) == pytest.approx(20.0)
    assert prot.position_risk(_pos(qty=10, entry=100, sl=103, direction=Direction.SHORT)) == pytest.approx(30.0)


def test_sl_beyond_entry_in_profit_is_zero_risk(prot: CapitalProtection) -> None:
    # trailed SL above entry on a long with no quote -> 0 (never negative)
    assert prot.position_risk(_pos(qty=10, entry=100, sl=101)) == 0.0
    assert prot.position_risk(_pos(qty=10, entry=100, sl=99, direction=Direction.SHORT)) == 0.0


def test_mark_to_market_gain_counts_as_risk(prot: CapitalProtection) -> None:
    # In profit at 110 with SL 98: equity includes +100 MTM; an SL fill gives back 120.
    assert prot.position_risk(_pos(qty=10, entry=100, sl=98, ltp=110)) == pytest.approx(120.0)
    # In loss at 99: entry-based risk (20) is larger and kept (conservative)
    assert prot.position_risk(_pos(qty=10, entry=100, sl=98, ltp=99)) == pytest.approx(20.0)


def test_gap_adjusted_overnight_risk(prot: CapitalProtection) -> None:
    # tight SL: 10*(100-99) = 10 < gap 10*100*3% = 30 -> 30
    swing = _pos(qty=10, entry=100, sl=99, mode=Mode.SWING)
    assert prot.position_risk(swing) == pytest.approx(30.0)
    assert prot.position_risk(swing, include_gap=False) == pytest.approx(10.0)
    # wide SL: 10*5 = 50 > 30 -> 50
    assert prot.position_risk(_pos(qty=10, entry=100, sl=95, mode=Mode.SWING)) == pytest.approx(50.0)
    # intraday position flagged overnight (e.g. converted) also uses the gap buffer
    assert prot.position_risk(_pos(qty=10, entry=100, sl=99, overnight=True)) == pytest.approx(30.0)
    assert prot.position_risk(_pos(qty=10, entry=100, sl=99), include_gap=True) == pytest.approx(30.0)
    # an SL trailed into profit still carries gap risk overnight
    assert prot.position_risk(_pos(qty=10, entry=100, sl=101, mode=Mode.SWING)) == pytest.approx(30.0)


def test_risk_includes_charges(settings: Settings) -> None:
    p = CapitalProtection(settings.risk, settings.charges)
    pos = _pos(qty=10, entry=500, sl=495, entry_charges=1.96)
    risk = p.position_risk(pos)
    # 50 SL risk + 1.96 accrued + MIS sell charges at 495 (~2.9)
    assert 50 + 1.96 + 2.5 < risk < 50 + 1.96 + 3.5
    assert CapitalProtection(settings.risk).position_risk(pos) == pytest.approx(51.96)


def test_open_risk_sum(prot: CapitalProtection) -> None:
    ps = [_pos(qty=10, entry=100, sl=98), _pos(qty=10, entry=100, sl=99, mode=Mode.SWING)]
    assert prot.open_risk(ps) == pytest.approx(20 + 30)
    assert prot.open_risk([]) == 0.0


# ----------------------------------------------------------------------------- budget


def test_loss_budget_and_max_new_trade_risk(prot: CapitalProtection) -> None:
    floors = prot.floors(1000.0, 1000.0)
    budget = prot.loss_budget(1000.0, 30.0, floors)
    assert budget == pytest.approx(60.0)
    assert prot.max_new_trade_risk(10.0, budget) == pytest.approx(10.0)   # normal risk binds
    assert prot.max_new_trade_risk(25.0, budget) == pytest.approx(20.0)   # budget/3 binds
    assert prot.loss_budget(950.0, 60.0, floors) == 0.0                     # never negative
    assert prot.max_new_trade_risk(10.0, 0.0) == 0.0
    assert prot.loss_budget(1000.0, 0.0, 905.0) == pytest.approx(95.0)      # single float floor
