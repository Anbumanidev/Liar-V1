"""Return correlation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from features.correlation import correlation_matrix, return_correlation
from tests.helpers import make_daily_candles


def test_identical_and_inverse_series() -> None:
    a = make_daily_candles(date(2026, 11, 24), 60, seed=1)["close"]
    assert return_correlation(a, a, 20) == pytest.approx(1.0)
    rets = a.pct_change()
    inv = (1 - rets.fillna(0)).cumprod() * 100
    assert return_correlation(a, inv, 20) == pytest.approx(-1.0, abs=1e-9)


def test_matches_numpy_on_last_window() -> None:
    a = make_daily_candles(date(2026, 11, 24), 60, seed=1)["close"]
    b = make_daily_candles(date(2026, 11, 24), 60, seed=2)["close"]
    ra, rb = a.pct_change().iloc[-20:], b.pct_change().iloc[-20:]
    exp = np.corrcoef(ra, rb)[0, 1]
    assert return_correlation(a, b, 20) == pytest.approx(exp)


def test_alignment_by_date_ignores_bar_time() -> None:
    a = make_daily_candles(date(2026, 11, 24), 40, seed=1)["close"]
    b = a.copy()
    b.index = b.index + pd.Timedelta(hours=6)  # same dates, different timestamps
    assert return_correlation(a, b, 20) == pytest.approx(1.0)


def test_insufficient_data_and_flat() -> None:
    a = make_daily_candles(date(2026, 11, 24), 10, seed=1)["close"]
    assert return_correlation(a, a, 20) is None
    assert return_correlation(a, a, 20, min_periods=5) == pytest.approx(1.0)
    flat = pd.Series(100.0, index=a.index)
    assert return_correlation(a, flat, 5) is None
    assert return_correlation(a.iloc[:0], a, 5) is None


def test_partial_overlap() -> None:
    a = make_daily_candles(date(2026, 11, 24), 60, seed=1)["close"]
    b = make_daily_candles(date(2026, 11, 24), 60, seed=3)["close"].iloc[-15:]
    assert return_correlation(a, b, 20) is None  # only 14 overlapping returns


def test_correlation_matrix() -> None:
    a = make_daily_candles(date(2026, 11, 24), 40, seed=1)["close"]
    b = make_daily_candles(date(2026, 11, 24), 40, seed=2)["close"]
    m = correlation_matrix({"B": b, "A": a, "C": a}, 20)
    assert set(m) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert m[("A", "C")] == pytest.approx(1.0)
