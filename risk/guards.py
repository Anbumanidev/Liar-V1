"""Hard-coded trade-management guards (rule 0.10). Used by the risk manager, the paper
engine and every owner action from the dashboard. These rules can NOT be configured off.

* A stop-loss may only move CLOSER to price (tighter), never further away - by the system
  or by the owner.
* No averaging down: never add to an open position in the same symbol (and in particular
  never to a losing one).
* No overnight shorts: CNC modes are long-only (see ``modes.base``).
"""

from __future__ import annotations

from core.enums import Direction, Mode, RiskRule
from core.errors import AveragingDownError, StopLossLooseningError
from core.types import OpenPositionView
from modes.base import assert_direction_allowed


class InvalidStopLossError(ValueError):
    """New stop-loss is structurally invalid (non-positive, or at/through the market price)."""


def is_tighter(direction: Direction, current_sl: float, new_sl: float) -> bool:
    """True if ``new_sl`` is strictly closer to price than ``current_sl`` for ``direction``."""
    if Direction(direction) == Direction.LONG:
        return new_sl > current_sl
    return new_sl < current_sl


def validate_sl_change(direction: Direction, current_sl: float, new_sl: float, ltp: float | None = None) -> None:
    """Validate a stop-loss modification.

    Raises:
        InvalidStopLossError: ``new_sl`` <= 0, or on/through the current price.
        StopLossLooseningError: ``new_sl`` is not strictly tighter than ``current_sl``.
    """
    if new_sl is None or new_sl <= 0:
        raise InvalidStopLossError(f"Stop-loss must be a positive price, got {new_sl!r}")
    if not is_tighter(direction, current_sl, new_sl):
        raise StopLossLooseningError(
            f"{RiskRule.SL_LOOSENING_FORBIDDEN}: stop-loss may only move closer to price "
            f"({direction}: current {current_sl}, requested {new_sl})"
        )
    if ltp is not None:
        if Direction(direction) == Direction.LONG and new_sl >= ltp:
            raise InvalidStopLossError(
                f"Stop-loss {new_sl} is at/above the market price {ltp} for a LONG; close the position instead."
            )
        if Direction(direction) == Direction.SHORT and new_sl <= ltp:
            raise InvalidStopLossError(
                f"Stop-loss {new_sl} is at/below the market price {ltp} for a SHORT; close the position instead."
            )


def tightened_sl(direction: Direction, current_sl: float, candidate_sl: float) -> float:
    """Return the tighter of ``current_sl`` and ``candidate_sl`` (used by trailing logic).

    Trailing can never loosen a stop: if the candidate is looser, the current SL is kept.
    """
    return candidate_sl if is_tighter(direction, current_sl, candidate_sl) else current_sl


def check_add_to_position(
    symbol: str, direction: Direction, open_positions: list[OpenPositionView]
) -> RiskRule | None:
    """Return a veto rule if a new entry would add to an existing position in ``symbol``.

    * Same symbol & same direction & that position is losing -> AVERAGING_DOWN_FORBIDDEN.
    * Any other existing open position in the same symbol -> DUPLICATE_POSITION.
    """
    sym = symbol.strip().upper()
    for pos in open_positions:
        if pos.symbol.strip().upper() != sym:
            continue
        if Direction(pos.direction) == Direction(direction) and pos.ltp is not None and pos.unrealized_gross < 0:
            return RiskRule.AVERAGING_DOWN_FORBIDDEN
        return RiskRule.DUPLICATE_POSITION
    return None


def assert_no_averaging_down(symbol: str, direction: Direction, open_positions: list[OpenPositionView]) -> None:
    """Raise ``AveragingDownError`` when ``check_add_to_position`` would veto."""
    rule = check_add_to_position(symbol, direction, open_positions)
    if rule is not None:
        raise AveragingDownError(f"{rule}: already holding {symbol}; adding to positions is forbidden.")


def assert_entry_direction(mode: Mode, direction: Direction) -> None:
    """Long-only for CNC modes (re-export of the hard-coded mode guard)."""
    assert_direction_allowed(mode, direction)
