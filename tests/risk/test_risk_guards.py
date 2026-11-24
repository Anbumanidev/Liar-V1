"""SL tighten-only, no averaging down, no overnight shorts (rule 0.10, phase 4)."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.clock import IST
from core.enums import Direction, Mode, Product, RiskRule
from core.errors import AveragingDownError, OvernightShortError, StopLossLooseningError
from core.types import OpenPositionView
from risk.guards import (
    InvalidStopLossError,
    assert_entry_direction,
    assert_no_averaging_down,
    check_add_to_position,
    is_tighter,
    tightened_sl,
    validate_sl_change,
)


def _pos(symbol: str = "INFY", direction: Direction = Direction.LONG, entry: float = 100.0,
         ltp: float | None = 101.0) -> OpenPositionView:
    return OpenPositionView(
        position_id=1, symbol=symbol, mode=Mode.INTRADAY, product=Product.MIS, direction=direction, qty=2,
        avg_entry=entry, stop_loss=95.0 if direction == Direction.LONG else 105.0, ltp=ltp,
        margin_blocked=200.0, opened_at=datetime(2026, 11, 24, 10, 0, tzinfo=IST),
    )


def test_long_sl_can_only_rise() -> None:
    validate_sl_change(Direction.LONG, 95.0, 96.0, ltp=100.0)
    with pytest.raises(StopLossLooseningError):
        validate_sl_change(Direction.LONG, 95.0, 94.0, ltp=100.0)
    with pytest.raises(StopLossLooseningError):
        validate_sl_change(Direction.LONG, 95.0, 95.0, ltp=100.0)


def test_short_sl_can_only_fall() -> None:
    validate_sl_change(Direction.SHORT, 105.0, 104.0, ltp=100.0)
    with pytest.raises(StopLossLooseningError):
        validate_sl_change(Direction.SHORT, 105.0, 106.0, ltp=100.0)


def test_sl_through_market_rejected() -> None:
    with pytest.raises(InvalidStopLossError):
        validate_sl_change(Direction.LONG, 95.0, 100.5, ltp=100.0)
    with pytest.raises(InvalidStopLossError):
        validate_sl_change(Direction.SHORT, 105.0, 99.0, ltp=100.0)
    with pytest.raises(InvalidStopLossError):
        validate_sl_change(Direction.LONG, 95.0, 0.0)


def test_trailing_never_loosens() -> None:
    assert tightened_sl(Direction.LONG, 95.0, 97.0) == 97.0
    assert tightened_sl(Direction.LONG, 95.0, 93.0) == 95.0
    assert tightened_sl(Direction.SHORT, 105.0, 103.0) == 103.0
    assert tightened_sl(Direction.SHORT, 105.0, 107.0) == 105.0
    assert is_tighter(Direction.LONG, 1.0, 2.0) and not is_tighter(Direction.SHORT, 1.0, 2.0)


def test_no_averaging_down() -> None:
    losing = _pos(ltp=98.0)
    assert check_add_to_position("INFY", Direction.LONG, [losing]) == RiskRule.AVERAGING_DOWN_FORBIDDEN
    with pytest.raises(AveragingDownError):
        assert_no_averaging_down("infy", Direction.LONG, [losing])
    winning = _pos(ltp=103.0)
    assert check_add_to_position("INFY", Direction.LONG, [winning]) == RiskRule.DUPLICATE_POSITION
    assert check_add_to_position("TCS", Direction.LONG, [losing]) is None


def test_no_overnight_shorts_hard_coded() -> None:
    assert_entry_direction(Mode.INTRADAY, Direction.SHORT)
    for mode in (Mode.BTST, Mode.SWING, Mode.POSITIONAL):
        assert_entry_direction(mode, Direction.LONG)
        with pytest.raises(OvernightShortError):
            assert_entry_direction(mode, Direction.SHORT)
