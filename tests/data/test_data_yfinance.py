"""YFinanceProvider normalisation with a monkeypatched ``yfinance.Ticker`` (no network)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from core.clock import SimulatedClock
from core.enums import Timeframe
from core.errors import DataUnavailableError
from core.types import OHLCV_COLUMNS, validate_ohlcv
from data.providers.yfinance_provider import YF_SYMBOLS, YFinanceProvider, yf_symbol
from tests.helpers import ist

yf = pytest.importorskip("yfinance")


def _vendor_frame(index: pd.DatetimeIndex, start: float = 100.0) -> pd.DataFrame:
    n = len(index)
    close = [start + i for i in range(n)]
    return pd.DataFrame({"Open": close, "High": [c + 2 for c in close], "Low": [c - 2 for c in close],
                         "Close": close, "Adj Close": [c * 0.99 for c in close], "Volume": [1000 + i for i in range(n)],
                         "Dividends": 0.0, "Stock Splits": 0.0}, index=index)


class FakeTicker:
    """Stands in for ``yfinance.Ticker``; behaviour is configured through class attributes."""

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    fast: dict[str, Any] = {}
    created: list["FakeTicker"] = []
    fail: Exception | None = None

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.history_kwargs: dict[str, Any] = {}
        FakeTicker.created.append(self)

    def history(self, **kwargs: Any) -> pd.DataFrame:
        self.history_kwargs = kwargs
        if FakeTicker.fail is not None:
            raise FakeTicker.fail
        return FakeTicker.frames.get((self.symbol, kwargs["interval"]), pd.DataFrame())

    @property
    def fast_info(self) -> Any:
        if FakeTicker.fail is not None:
            raise FakeTicker.fail
        return FakeTicker.fast[self.symbol]


@pytest.fixture
def ticker(monkeypatch: pytest.MonkeyPatch) -> type[FakeTicker]:
    FakeTicker.frames, FakeTicker.fast, FakeTicker.created, FakeTicker.fail = {}, {}, [], None
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    return FakeTicker


@pytest.fixture
def provider() -> YFinanceProvider:
    return YFinanceProvider(clock=SimulatedClock(ist(2026, 11, 23, 16, 0)))


def test_symbol_mapping() -> None:
    assert yf_symbol("reliance") == "RELIANCE.NS"
    assert yf_symbol("M&M") == "M&M.NS"
    assert yf_symbol("NIFTY50") == "^NSEI"
    assert yf_symbol("BANKNIFTY") == "^NSEBANK"
    assert yf_symbol("INDIAVIX") == "^INDIAVIX"
    assert yf_symbol("SP500") == "^GSPC"
    assert yf_symbol("USDINR") == "INR=X"
    assert yf_symbol("^NSEI") == "^NSEI"                   # already a Yahoo ticker
    assert yf_symbol("TCS.BO") == "TCS.BO"
    required = {"NIFTY50", "BANKNIFTY", "INDIAVIX", "NIFTYIT", "NIFTYPHARMA", "NIFTYFMCG", "NIFTYAUTO", "NIFTYMETAL",
                "NIFTYREALTY", "NIFTYENERGY", "NIFTYPSUBANK", "NIFTYPVTBANK", "NIFTYFINSERVICE", "NIFTYMEDIA",
                "SP500", "NASDAQ", "NIKKEI", "HANGSENG", "DOWJONES", "CRUDE", "USDINR"}
    assert required <= set(YF_SYMBOLS)


def test_daily_bars_normalised_to_0915_ist(ticker: type[FakeTicker], provider: YFinanceProvider) -> None:
    idx = pd.DatetimeIndex(["2026-11-19", "2026-11-20", "2026-11-23"], name="Date").tz_localize("Asia/Kolkata")
    ticker.frames[("RELIANCE.NS", "1d")] = _vendor_frame(idx)
    df = provider.get_candles("RELIANCE", Timeframe.D1, ist(2026, 11, 19, 12, 0), ist(2026, 11, 23, 16, 0))
    validate_ohlcv(df)
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert all(df[c].dtype == "float64" for c in OHLCV_COLUMNS)
    assert list(df.index) == [ist(2026, 11, 19), ist(2026, 11, 20), ist(2026, 11, 23)]
    assert df["close"].tolist() == [100.0, 101.0, 102.0]                  # unadjusted Close, not Adj Close
    kw = ticker.created[-1].history_kwargs
    assert ticker.created[-1].symbol == "RELIANCE.NS"
    assert kw["interval"] == "1d" and kw["auto_adjust"] is False
    assert kw["start"] == date(2026, 11, 19) and kw["end"] == date(2026, 11, 24)   # exclusive end padded


def test_index_symbol_and_foreign_timezone_keep_their_dates(ticker: type[FakeTicker],
                                                            provider: YFinanceProvider) -> None:
    nifty_idx = pd.DatetimeIndex(["2026-11-20", "2026-11-23"]).tz_localize("Asia/Kolkata")
    ticker.frames[("^NSEI", "1d")] = _vendor_frame(nifty_idx, 24000.0)
    spx_idx = pd.DatetimeIndex(["2026-11-19", "2026-11-20"]).tz_localize("America/New_York")
    ticker.frames[("^GSPC", "1d")] = _vendor_frame(spx_idx, 6000.0)
    nifty = provider.get_candles("NIFTY50", Timeframe.D1, ist(2026, 11, 1), ist(2026, 11, 23, 16, 0))
    assert nifty.index[-1] == ist(2026, 11, 23) and nifty["close"].iloc[-1] == 24001.0
    spx = provider.get_candles("SP500", Timeframe.D1, ist(2026, 11, 1), ist(2026, 11, 23, 16, 0))
    assert list(spx.index) == [ist(2026, 11, 19), ist(2026, 11, 20)]    # US dates are not shifted


def test_intraday_bars_converted_to_ist_and_multiindex_columns(ticker: type[FakeTicker],
                                                                provider: YFinanceProvider) -> None:
    idx = pd.date_range("2026-11-23 03:45", periods=6, freq="5min", tz="UTC", name="Datetime")   # 09:15 IST
    frame = _vendor_frame(idx)
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["TCS.NS"]], names=["Price", "Ticker"])
    ticker.frames[("TCS.NS", "5m")] = frame
    df = provider.get_candles("TCS", Timeframe.M5, ist(2026, 11, 23, 9, 15), ist(2026, 11, 23, 9, 35))
    validate_ohlcv(df)
    assert str(df.index.tz) == "Asia/Kolkata"
    assert df.index[0] == ist(2026, 11, 23, 9, 15)
    assert df.index[-1] == ist(2026, 11, 23, 9, 35)                       # end filter (09:40 bar dropped)
    assert len(df) == 5


def test_intraday_start_clamped_to_yahoo_limit(ticker: type[FakeTicker], provider: YFinanceProvider) -> None:
    idx = pd.date_range("2026-11-23 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
    ticker.frames[("INFY.NS", "5m")] = _vendor_frame(idx)
    now = provider._clock.now()
    provider.get_candles("INFY", Timeframe.M5, now - timedelta(days=200), now)
    assert ticker.created[-1].history_kwargs["start"] == now - timedelta(days=59)
    assert provider.max_history(Timeframe.M1) == timedelta(days=7)
    assert provider.max_history(Timeframe.D1) is None


def test_failures_raise_data_unavailable(ticker: type[FakeTicker], provider: YFinanceProvider) -> None:
    with pytest.raises(DataUnavailableError):                              # empty history
        provider.get_candles("ABC", Timeframe.D1, ist(2026, 11, 1), ist(2026, 11, 23))
    assert not provider.supports(Timeframe.W1)
    with pytest.raises(DataUnavailableError):                              # weekly is resampled upstream
        provider.get_candles("ABC", Timeframe.W1, ist(2026, 11, 1), ist(2026, 11, 23))
    assert provider.get_candles("ABC", Timeframe.D1, ist(2026, 11, 23), ist(2026, 11, 1)).empty
    ticker.fail = RuntimeError("rate limited")
    with pytest.raises(DataUnavailableError):
        provider.get_candles("ABC", Timeframe.D1, ist(2026, 11, 1), ist(2026, 11, 23))
    assert provider.get_quote("ABC") is None
    assert provider.healthcheck() is False


def test_quote_from_fast_info(ticker: type[FakeTicker], provider: YFinanceProvider) -> None:
    ticker.fast["RELIANCE.NS"] = {"lastPrice": 2500.5, "previousClose": 2480.0, "open": 2490.0,
                                  "dayHigh": 2510.0, "dayLow": 2485.0, "lastVolume": 12345}
    q = provider.get_quote("reliance")
    assert q is not None
    assert (q.symbol, q.ltp, q.prev_close, q.open, q.high, q.low, q.volume) == \
        ("RELIANCE", 2500.5, 2480.0, 2490.0, 2510.0, 2485.0, 12345.0)
    assert q.timestamp == provider._clock.now() and q.source == "yfinance"
    ticker.fast["NAN.NS"] = {"lastPrice": float("nan")}
    assert provider.get_quote("NAN") is None

    class AttrOnly:                                                        # FastInfo-like without .get
        last_price = 99.0

    ticker.fast["ATTR.NS"] = AttrOnly()
    q2 = provider.get_quote("ATTR")
    assert q2 is not None and q2.ltp == 99.0 and q2.prev_close is None


def test_healthcheck_uses_nifty(ticker: type[FakeTicker], provider: YFinanceProvider) -> None:
    idx = pd.DatetimeIndex(["2026-11-20", "2026-11-23"]).tz_localize("Asia/Kolkata")
    ticker.frames[("^NSEI", "1d")] = _vendor_frame(idx, 24000.0)
    assert provider.healthcheck() is True
