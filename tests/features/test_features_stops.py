"""Stop-loss candidate generation."""

from __future__ import annotations

import pytest

from core.enums import Direction, Mode, SLMethod
from core.types import Levels, StockFeatures, TimeframeFeatures
from features.stops import stop_loss_candidates
from tests.helpers import ist

AS_OF = ist(2026, 11, 24, 11, 0)


def _features(**over: object) -> StockFeatures:
    levels = Levels(swing_lows=[95.0, 97.0, 101.0], swing_highs=[108.0, 104.0, 99.0], prev_high=106.0, prev_low=94.0)
    tff = TimeframeFeatures(timeframe="5m", as_of=AS_OF, bars=300,
                            values={"vwap": 98.5, "atr_14": 2.0, "close": 100.0}, levels=levels)
    base: dict[str, object] = dict(
        symbol="TEST", mode=Mode.INTRADAY, as_of=AS_OF, ltp=100.0, primary_timeframe="5m",
        timeframes={"5m": tff}, atr=2.0, opening_range_high=102.0, opening_range_low=97.5,
        extra={"prev_day_high": 105.0, "prev_day_low": 93.0},
    )
    base.update(over)
    return StockFeatures(**base)  # type: ignore[arg-type]


def test_long_candidates_all_below_entry() -> None:
    cands = stop_loss_candidates(_features(), Direction.LONG, 100.0)
    assert cands, "expected candidates"
    assert all(c.price < 100.0 for c in cands)
    by = {c.method: [x.price for x in cands if x.method == c.method] for c in cands}
    assert sorted(by[SLMethod.ATR]) == [96.0, 97.0, 98.0]
    assert by[SLMethod.SWING] == [97.0]  # most recent swing low below entry (101 is above)
    assert by[SLMethod.VWAP] == [98.5]
    assert by[SLMethod.OPENING_RANGE] == [97.5]
    assert by[SLMethod.DAILY_STRUCTURE] == [93.0]
    assert [c.distance for c in cands] == sorted(c.distance for c in cands)
    atr1 = next(c for c in cands if c.method == SLMethod.ATR and c.price == 98.0)
    assert atr1.distance == pytest.approx(2.0)
    assert atr1.distance_atr == pytest.approx(1.0)


def test_short_candidates_all_above_entry() -> None:
    cands = stop_loss_candidates(_features(), Direction.SHORT, 100.0)
    assert all(c.price > 100.0 for c in cands)
    methods = {c.method for c in cands}
    assert SLMethod.VWAP not in methods  # VWAP 98.5 is below entry -> wrong side for a short
    assert {c.price for c in cands if c.method == SLMethod.SWING} == {104.0}
    assert {c.price for c in cands if c.method == SLMethod.OPENING_RANGE} == {102.0}
    assert {c.price for c in cands if c.method == SLMethod.DAILY_STRUCTURE} == {105.0}
    assert sorted(c.price for c in cands if c.method == SLMethod.ATR) == [102.0, 103.0, 104.0]


def test_wrong_side_levels_are_excluded() -> None:
    f = _features(opening_range_low=100.5, extra={"prev_day_low": 100.0})
    cands = stop_loss_candidates(f, Direction.LONG, 100.0)
    methods = {c.method for c in cands}
    assert SLMethod.OPENING_RANGE not in methods
    assert SLMethod.DAILY_STRUCTURE not in methods  # equal to entry is not strictly below


def test_tick_rounding_is_protective() -> None:
    f = _features(atr=1.234)
    longs = stop_loss_candidates(f, Direction.LONG, 100.0)
    shorts = stop_loss_candidates(f, Direction.SHORT, 100.0)
    a_long = next(c for c in longs if c.method == SLMethod.ATR and "1x" in c.note)
    a_short = next(c for c in shorts if c.method == SLMethod.ATR and "1x" in c.note)
    assert a_long.price == pytest.approx(98.75)   # 98.766 floored to tick
    assert a_short.price == pytest.approx(101.25)  # 101.234 ceiled to tick
    for c in longs + shorts:
        assert round(c.price / 0.05, 6) == round(c.price / 0.05)


def test_no_atr_no_atr_candidates() -> None:
    f = _features(atr=None)
    cands = stop_loss_candidates(f, Direction.LONG, 100.0)
    assert SLMethod.ATR not in {c.method for c in cands}
    assert all(c.distance_atr is None for c in cands)


def test_daily_structure_fallback_to_intraday_levels() -> None:
    f = _features(extra={})
    cands = stop_loss_candidates(f, Direction.LONG, 100.0)
    ds = [c for c in cands if c.method == SLMethod.DAILY_STRUCTURE]
    assert ds and ds[0].price == 94.0


def test_invalid_entry() -> None:
    assert stop_loss_candidates(_features(), Direction.LONG, 0.0) == []
    assert stop_loss_candidates(_features(), Direction.LONG, float("nan")) == []


def test_no_duplicates() -> None:
    f = _features(opening_range_low=98.0)
    cands = stop_loss_candidates(f, Direction.LONG, 100.0)
    keys = [(c.method, c.price) for c in cands]
    assert len(keys) == len(set(keys))
