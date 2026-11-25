"""normalize_ohlcv and session-aligned resample_ohlcv."""

from __future__ import annotations

from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

from core.clock import IST
from core.types import OHLCV_COLUMNS, validate_ohlcv
from data.bars import normalize_ohlcv, resample_ohlcv
from tests.helpers import ist, make_daily_candles, make_intraday_candles, trading_days


def _frame(index: list, closes: list[float] | None = None) -> pd.DataFrame:
    n = len(index)
    c = np.asarray(closes if closes is not None else np.arange(1, n + 1, dtype=float) + 100)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": np.full(n, 10.0)},
                        index=pd.DatetimeIndex(index))


# ------------------------------------------------------------------ normalize


def test_normalize_utc_intraday_index_to_ist() -> None:
    src = make_intraday_candles([date(2026, 11, 23)], 5)
    utc = src.copy()
    utc.index = utc.index.tz_convert("UTC")
    out = normalize_ohlcv(utc)
    validate_ohlcv(out)
    assert str(out.index.tz) == "Asia/Kolkata"
    assert out.index[0] == ist(2026, 11, 23, 9, 15)
    assert out.index.name == "ts"
    pd.testing.assert_frame_equal(out, src, check_freq=False, check_names=False, check_index_type=False)


def test_normalize_naive_index_is_assumed_ist() -> None:
    src = make_intraday_candles([date(2026, 11, 23)], 15)
    naive = src.copy()
    naive.index = naive.index.tz_localize(None)
    out = normalize_ohlcv(naive)
    assert out.index[0] == ist(2026, 11, 23, 9, 15)
    assert out.index[-1] == ist(2026, 11, 23, 15, 15)


def test_normalize_columns_sort_dedupe_and_bad_rows() -> None:
    idx = [ist(2026, 11, 23, 9, 20), ist(2026, 11, 23, 9, 15), ist(2026, 11, 23, 9, 20), ist(2026, 11, 23, 9, 25),
           ist(2026, 11, 23, 9, 30)]
    df = _frame(idx, [101.0, 100.0, 105.0, np.nan, 0.0])
    df["Adj Close"] = df["Close"]
    df["Dividends"] = 0.0
    df["Volume"] = [1.0, np.nan, 3.0, 4.0, 5.0]
    out = normalize_ohlcv(df)
    assert list(out.columns) == list(OHLCV_COLUMNS)
    assert all(out[c].dtype == "float64" for c in OHLCV_COLUMNS)
    assert list(out.index) == [ist(2026, 11, 23, 9, 15), ist(2026, 11, 23, 9, 20)]   # NaN / 0 close dropped
    assert out.loc[ist(2026, 11, 23, 9, 20), "close"] == 105.0                       # duplicate: last wins
    assert out.loc[ist(2026, 11, 23, 9, 15), "volume"] == 0.0                         # missing volume -> 0


def test_normalize_timestamp_column_and_aliases() -> None:
    df = pd.DataFrame({"Datetime": ["2026-11-23 09:15", "2026-11-23 09:16"], "o": [1.0, 2.0], "h": [1.0, 2.0],
                       "l": [1.0, 2.0], "c": [1.0, 2.0]})
    out = normalize_ohlcv(df)
    assert list(out.index) == [ist(2026, 11, 23, 9, 15), ist(2026, 11, 23, 9, 16)]
    assert (out["volume"] == 0.0).all()


def test_normalize_daily_dates_keep_their_calendar_day() -> None:
    days = [date(2026, 11, 20), date(2026, 11, 23)]
    utc_midnight = _frame([pd.Timestamp(d) for d in days]).tz_localize("UTC")
    ny_midnight = _frame([pd.Timestamp(d) for d in days]).tz_localize("America/New_York")
    # IST midnight expressed in UTC (18:30Z of the previous day): an instant, not a date label
    ist_as_utc = _frame([pd.Timestamp(d).tz_localize(IST).tz_convert("UTC") for d in days])
    ist_open_utc = _frame([pd.Timestamp(datetime.combine(d, time(3, 45))).tz_localize("UTC") for d in days])
    naive = _frame([pd.Timestamp(d) for d in days])
    expected = [ist(2026, 11, 20), ist(2026, 11, 23)]
    for frame in (utc_midnight, ny_midnight, ist_as_utc, ist_open_utc, naive):
        assert list(normalize_ohlcv(frame, "1d").index) == expected
    for frame in (utc_midnight, ny_midnight, ist_as_utc, naive):   # auto-detected as daily
        assert list(normalize_ohlcv(frame).index) == expected


def test_normalize_empty_and_missing_columns() -> None:
    assert normalize_ohlcv(None).empty
    assert validate_ohlcv(normalize_ohlcv(pd.DataFrame())) is not None
    with pytest.raises(ValueError):
        normalize_ohlcv(pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex([ist(2026, 11, 23)])))


# ------------------------------------------------------------------ resample


def test_resample_5m_to_15m_aggregates_correctly() -> None:
    src = make_intraday_candles([date(2026, 11, 23)], 5, seed=3)
    out = resample_ohlcv(src, "15m")
    validate_ohlcv(out)
    assert len(out) == 25
    assert out.index[0] == ist(2026, 11, 23, 9, 15) and out.index[1] == ist(2026, 11, 23, 9, 30)
    first = src.loc[ist(2026, 11, 23, 9, 15): ist(2026, 11, 23, 9, 25)]
    row = out.iloc[0]
    assert row["open"] == first["open"].iloc[0]
    assert row["high"] == first["high"].max()
    assert row["low"] == first["low"].min()
    assert row["close"] == first["close"].iloc[-1]
    assert row["volume"] == pytest.approx(first["volume"].sum())


def test_resample_5m_to_1h_session_aligned() -> None:
    days = [date(2026, 11, 20), date(2026, 11, 23)]
    src = make_intraday_candles(days, 5, seed=4)
    out = resample_ohlcv(src, "1h")
    day_bars = out.loc[out.index.date == date(2026, 11, 23)]
    assert [ts.strftime("%H:%M") for ts in day_bars.index] == [
        "09:15", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"]
    last = src.loc[ist(2026, 11, 23, 15, 15):ist(2026, 11, 23, 15, 25)]
    assert len(last) == 3                                    # the 15:15 bar is only 15 minutes long
    assert out.loc[ist(2026, 11, 23, 15, 15), "close"] == last["close"].iloc[-1]
    assert out.loc[ist(2026, 11, 23, 15, 15), "volume"] == pytest.approx(last["volume"].sum())
    assert len(out) == 14


def test_resample_ignores_out_of_session_rows_for_intraday() -> None:
    src = make_intraday_candles([date(2026, 11, 23)], 5)
    extra = src.iloc[:1].copy()
    extra.index = pd.DatetimeIndex([ist(2026, 11, 23, 9, 5)], name="ts")       # pre-open print
    out = resample_ohlcv(pd.concat([extra, src]).sort_index(), "15m")
    assert out.index[0] == ist(2026, 11, 23, 9, 15)
    assert len(out) == 25


def test_resample_intraday_to_daily_and_daily_to_weekly() -> None:
    days = trading_days(date(2026, 11, 16), 6)              # Mon 16 .. Mon 23
    src = make_intraday_candles(days, 5, seed=5)
    daily = resample_ohlcv(src, "1d")
    assert list(daily.index) == [ist(d.year, d.month, d.day) for d in days]
    one = src.loc[src.index.date == date(2026, 11, 18)]
    assert daily.loc[ist(2026, 11, 18), "high"] == one["high"].max()
    assert daily.loc[ist(2026, 11, 18), "close"] == one["close"].iloc[-1]

    weekly = resample_ohlcv(daily, "1wk")
    assert list(weekly.index) == [ist(2026, 11, 16), ist(2026, 11, 23)]
    wk = daily.loc[ist(2026, 11, 16):ist(2026, 11, 20)]
    assert weekly.iloc[0]["open"] == wk["open"].iloc[0]
    assert weekly.iloc[0]["close"] == wk["close"].iloc[-1]
    assert weekly.iloc[0]["volume"] == pytest.approx(wk["volume"].sum())


def test_resample_weekly_labels_first_day_present() -> None:
    daily = make_daily_candles(date(2026, 11, 27), 5)        # Mon 23 .. Fri 27
    without_monday = daily.drop(index=ist(2026, 11, 23))     # Monday holiday
    weekly = resample_ohlcv(without_monday, "1wk")
    assert list(weekly.index) == [ist(2026, 11, 24)]


def test_resample_accepts_unnormalised_input_and_empty() -> None:
    src = make_intraday_candles([date(2026, 11, 23)], 5)
    raw = src.rename(columns=str.capitalize)
    raw.index = raw.index.tz_convert("UTC")
    out = resample_ohlcv(raw, "1h")
    assert out.index[0] == ist(2026, 11, 23, 9, 15) and len(out) == 7
    assert resample_ohlcv(src.iloc[:0], "1h").empty
