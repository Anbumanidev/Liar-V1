"""Capital protection: floors, drawdown tiers, open risk and loss budget (section 8).

Definitions (``hard_stop_pct`` defaults to 9 %, MAX_LOSS_PCT 10 % leaves a 1 % buffer):

* **Absolute floor** = ``starting_capital * (1 - hard_stop_pct/100)`` (Rs 910 on Rs 1,000).
* **Trailing floor** = ``hwm * (1 - hard_stop_pct/100)``.
* **Effective floor** = ``max(absolute, trailing)`` - the stricter floor always wins.
* **Drawdown %** = ``(hwm - equity) / hwm * 100`` (0 when equity >= hwm).
* **Tiers**: ``NORMAL`` < caution_pct <= ``CAUTION`` < defensive_pct <= ``DEFENSIVE`` <
  hard_stop_pct <= ``HALT``; also ``HALT`` whenever ``equity <= effective floor``.
* **Position risk** (rupees the position can still take off TOTAL EQUITY if its stop fills),
  see :meth:`CapitalProtection.position_risk`.
* **Loss budget** = ``max(0, equity - open_risk - effective_floor)``.
* **Max new trade risk** = ``min(normal_risk, loss_budget / loss_budget_divisor)``.

Why the budget guarantees the floors (proved by the simulation test): every new trade's full
risk is at most ``budget / divisor < budget``, so after it the remaining budget stays >= 0;
an SL fill reduces equity and open risk by the same amount (budget unchanged); a winning exit
raises equity by ``g`` and the trailing floor by at most ``(1 - hard_stop%) * g`` (budget
grows). Hence ``equity - open_risk >= effective_floor`` is an invariant and equity can never
drop below either floor through stop fills (gap fills bounded by the gap buffer included).

All functions are pure (no I/O, no clock).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from config.schema import ChargesSettings, RiskSettings
from core.enums import Direction, DrawdownTier, Grade
from core.types import OpenPositionView
from modes.base import is_overnight
from paper_engine.charges import exit_side, order_charges

__all__ = ["CapitalProtection", "TierConstraints"]

#: Rupee tolerance for floor comparisons (sub-paisa float noise must not flip a HALT).
_EPS: float = 1e-9


@dataclass(frozen=True)
class TierConstraints:
    """What a drawdown tier allows (section 8).

    Attributes:
        tier: the tier these constraints belong to.
        size_mult: multiplier applied to the normal risk amount (1.0, 0.5, 0.25, 0.0).
        allowed_grades: confluence grades still allowed. ``NORMAL`` lists every grade - the
            mode's own ``min_grade`` still applies on top.
        max_open_positions: cap on total open positions (``None`` = no tier cap).
        allow_new_overnight: whether NEW overnight (CNC) positions may be opened.
        halted: no new trades at all; existing positions must be exited.
    """

    tier: DrawdownTier
    size_mult: float
    allowed_grades: tuple[Grade, ...]
    max_open_positions: int | None
    allow_new_overnight: bool
    halted: bool

    def grade_allowed(self, grade: Grade | str) -> bool:
        """True if ``grade`` is allowed in this tier."""
        return Grade(grade) in self.allowed_grades


class CapitalProtection:
    """Section 8 math, parameterised by ``RiskSettings``.

    Args:
        risk: ``settings.risk``.
        charges: optional ``settings.charges``; when given, :meth:`position_risk` adds the
            estimated exit charges at the stop price to each position's risk.
    """

    def __init__(self, risk: RiskSettings, charges: ChargesSettings | None = None) -> None:
        self.risk = risk
        self.charges = charges

    # ------------------------------------------------------------------ floors / drawdown

    @property
    def hard_stop_fraction(self) -> float:
        """``hard_stop_pct`` as a fraction (0.09)."""
        return self.risk.hard_stop_pct / 100.0

    def floors(self, starting_capital: float, hwm: float) -> tuple[float, float]:
        """``(absolute_floor, trailing_floor)``."""
        keep = 1.0 - self.hard_stop_fraction
        return starting_capital * keep, hwm * keep

    def effective_floor(self, starting_capital: float, hwm: float) -> float:
        """The stricter (higher) of the two floors."""
        return max(self.floors(starting_capital, hwm))

    @staticmethod
    def drawdown_pct(equity: float, hwm: float) -> float:
        """Drawdown from the high-water mark in percent (never negative)."""
        if hwm <= 0:
            return 0.0
        return max(0.0, (hwm - equity) / hwm * 100.0)

    def tier_for(self, drawdown_pct: float, equity: float, effective_floor: float) -> DrawdownTier:
        """Drawdown tier; ``HALT`` also when equity has reached the effective floor."""
        dd = self.risk.drawdown
        if equity <= effective_floor + _EPS or drawdown_pct >= self.risk.hard_stop_pct - _EPS:
            return DrawdownTier.HALT
        if drawdown_pct >= dd.defensive_pct - _EPS:
            return DrawdownTier.DEFENSIVE
        if drawdown_pct >= dd.caution_pct - _EPS:
            return DrawdownTier.CAUTION
        return DrawdownTier.NORMAL

    def evaluate(self, starting_capital: float, hwm: float, equity: float) -> tuple[float, DrawdownTier]:
        """Convenience: ``(drawdown_pct, tier)`` for the given equity."""
        dd = self.drawdown_pct(equity, hwm)
        return dd, self.tier_for(dd, equity, self.effective_floor(starting_capital, hwm))

    # ------------------------------------------------------------------ risk / budget

    def _exit_charges_at(self, pos: OpenPositionView, price: float) -> float:
        if self.charges is None or pos.qty <= 0:
            return 0.0
        return order_charges(pos.product, exit_side(pos.direction), pos.qty, max(price, 0.0), self.charges).total

    def position_risk(self, pos: OpenPositionView, include_gap: bool | None = None) -> float:
        """Rupees this open position can still take off TOTAL EQUITY (never negative).

        ``risk = price_risk + accrued entry charges + estimated exit charges``, where

        * price risk (stop) = ``qty * |ref - SL|`` on the losing side only (0 if the SL is
          already beyond the reference price in the profit direction). It is evaluated with
          ``ref = avg_entry`` (the classic SL risk) AND with ``ref = ltp`` when a quote is known,
          and the larger is used: total equity marks positions to market, so a position in
          profit can give back its open gain plus the entry-to-SL distance.
        * overnight positions (``include_gap`` true; default ``pos.overnight`` or an overnight mode)
          use ``max(stop risk, qty * ref * gap_buffer_pct/100)`` for each reference price, to
          cover a gap through the stop at the next open.
        * accrued entry charges (``pos.entry_charges``) are added in full. TOTAL EQUITY already
          nets them out of unrealized P&L, so this deliberately double-counts them - a small,
          conservative cushion.
        * estimated exit charges at the stop price are added when this object was built with a
          ``ChargesSettings`` (otherwise 0).
        """
        if pos.qty <= 0:
            return 0.0
        if include_gap is None:
            include_gap = bool(pos.overnight) or is_overnight(pos.mode)
        gap = self.risk.gap_buffer_pct / 100.0 if include_gap else 0.0
        refs = [pos.avg_entry] + ([pos.ltp] if pos.ltp is not None and pos.ltp > 0 else [])
        worst = 0.0
        for ref in refs:
            if pos.direction == Direction.LONG:
                stop_risk = max(0.0, (ref - pos.stop_loss) * pos.qty)
            else:
                stop_risk = max(0.0, (pos.stop_loss - ref) * pos.qty)
            gap_risk = pos.qty * ref * gap
            worst = max(worst, stop_risk, gap_risk)
        return worst + max(0.0, pos.entry_charges) + self._exit_charges_at(pos, pos.stop_loss)

    def open_risk(self, positions: Iterable[OpenPositionView], include_gap: bool | None = None) -> float:
        """Sum of :meth:`position_risk` over ``positions``."""
        return sum(self.position_risk(p, include_gap) for p in positions)

    @staticmethod
    def loss_budget(equity: float, open_risk: float, floors: tuple[float, float] | float) -> float:
        """``max(0, equity - open_risk - stricter floor)``. ``floors`` = tuple or one floor."""
        floor = max(floors) if isinstance(floors, tuple) else float(floors)
        return max(0.0, equity - open_risk - floor)

    def max_new_trade_risk(self, normal_risk: float, budget: float) -> float:
        """``min(normal_risk, budget / loss_budget_divisor)`` (never negative)."""
        return max(0.0, min(normal_risk, budget / self.risk.loss_budget_divisor))

    # ------------------------------------------------------------------ tiers

    def tier_constraints(self, tier: DrawdownTier) -> TierConstraints:
        """Constraints for ``tier`` from ``risk.drawdown``."""
        tier = DrawdownTier(tier)
        dd = self.risk.drawdown
        if tier == DrawdownTier.NORMAL:
            return TierConstraints(tier, 1.0, tuple(Grade), None, True, False)
        if tier == DrawdownTier.CAUTION:
            return TierConstraints(tier, dd.caution_size_mult, tuple(dd.caution_grades), None, True, False)
        if tier == DrawdownTier.DEFENSIVE:
            return TierConstraints(
                tier,
                dd.defensive_size_mult,
                tuple(dd.defensive_grades),
                dd.defensive_max_open_positions,
                dd.defensive_allow_new_overnight,
                False,
            )
        return TierConstraints(DrawdownTier.HALT, 0.0, (), 0, False, True)
