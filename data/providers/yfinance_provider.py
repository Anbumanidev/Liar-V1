"""yfinance provider: historical / fallback data for NSE stocks, indices and global cues.

* Plain NSE tickers get the ``.NS`` suffix; index / global pseudo-symbols map via ``YF_SYMBOLS``.
* ``Ticker.history(start, end, interval, auto_adjust=False)`` (single-level columns).
* Yahoo's intraday history limits (1m ~7 days, 5m/15m ~60 days, 1h ~730 days) are respected by
  clamping ``start`` (logged).
* Intraday indexes are converted to IST; daily bars are re-indexed to 09:15 IST of their LOCAL
  date (so a ``^GSPC`` bar dated 23-Nov stays on 23-Nov). Weekly bars are NOT served natively:
  Yahoo's 1wk alignment is not guaranteed, so ``MarketDataService`` resamples weekly from daily.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from core.clock import Clock, SystemClock, to_ist
from core.enums import Timeframe
from core.errors import DataUnavailableError
from core.types import Quote, empty_ohlcv
from data.bars import normalize_ohlcv
from data.base import DataProvider

#: Project pseudo-symbol -> Yahoo ticker.
YF_SYMBOLS: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
    "NIFTYIT": "^CNXIT",
    "NIFTYPHARMA": "^CNXPHARMA",
    "NIFTYFMCG": "^CNXFMCG",
    "NIFTYAUTO": "^CNXAUTO",
    "NIFTYMETAL": "^CNXMETAL",
    "NIFTYREALTY": "^CNXREALTY",
    "NIFTYENERGY": "^CNXENERGY",
    "NIFTYPSUBANK": "^CNXPSUBANK",
    "NIFTYMEDIA": "^CNXMEDIA",
    "NIFTYFINSERVICE": "NIFTY_FIN_SERVICE.NS",
    "NIFTYPVTBANK": "NIFTY_PVT_BANK.NS",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOWJONES": "^DJI",
    "NIKKEI": "^N225",
    "HANGSENG": "^HSI",
    "CRUDE": "CL=F",
    "USDINR": "INR=X",
}

YF_INTERVALS: dict[Timeframe, str] = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.H1: "1h",
    Timeframe.D1: "1d",
}

#: Conservative history windows (Yahoo: 1m=7d per request / 30d total, 5m/15m=60d, 1h=730d).
YF_MAX_HISTORY: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(days=7),
    Timeframe.M5: timedelta(days=59),
    Timeframe.M15: timedelta(days=59),
    Timeframe.H1: timedelta(days=729),
}


def yf_symbol(symbol: str) -> str:
    """Map a project symbol to its Yahoo ticker (``RELIANCE`` -> ``RELIANCE.NS``)."""
    sym = symbol.strip().upper()
    if sym in YF_SYMBOLS:
        return YF_SYMBOLS[sym]
    if any(ch in sym for ch in (".", "^", "=")):
        return symbol.strip()
    return f"{sym}.NS"


def _default_ticker(symbol: str) -> Any:
    import yfinance as yf  # imported lazily: heavy, and tests monkeypatch yfinance.Ticker

    return yf.Ticker(symbol)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider (free, delayed; no credentials).

    Args:
        clock: used for history clamping and quote timestamps (defaults to wall clock).
        ticker_factory: ``symbol -> yfinance.Ticker``-like object (tests inject fakes).
    """

    name = "yfinance"
    supported_timeframes = frozenset(YF_INTERVALS)

    def __init__(self, clock: Clock | None = None, ticker_factory: Callable[[str], Any] | None = None) -> None:
        self._clock = clock or SystemClock()
        self._ticker_factory = ticker_factory

    def _ticker(self, symbol: str) -> Any:
        factory = self._ticker_factory or _default_ticker
        return factory(yf_symbol(symbol))

    def max_history(self, timeframe: Timeframe) -> timedelta | None:
        return YF_MAX_HISTORY.get(timeframe)

    def clamp_start(self, timeframe: Timeframe, start: datetime) -> datetime:
        """Clamp ``start`` to Yahoo's intraday history limit (relative to the clock)."""
        limit = YF_MAX_HISTORY.get(timeframe)
        start = to_ist(start)
        if limit is None:
            return start
        earliest = self._clock.now() - limit
        if start < earliest:
            logger.info(f"[yfinance] {timeframe.value} history limited to {limit.days}d: start "
                        f"{start:%Y-%m-%d %H:%M} clamped to {earliest:%Y-%m-%d %H:%M}")
            return earliest
        return start

    def get_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        if timeframe not in YF_INTERVALS:
            raise DataUnavailableError(f"yfinance does not serve {timeframe.value} natively (resample)")
        start = self.clamp_start(timeframe, start)
        end = to_ist(end)
        if end <= start:
            return empty_ohlcv()
        # yfinance `end` is exclusive (and date-granular for daily bars): pad it.
        pad = timedelta(days=1) if timeframe == Timeframe.D1 else timedelta(minutes=60)
        try:
            raw = self._ticker(symbol).history(
                start=start.date() if timeframe == Timeframe.D1 else start,
                end=(end + pad).date() if timeframe == Timeframe.D1 else end + pad,
                interval=YF_INTERVALS[timeframe],
                auto_adjust=False,
                actions=False,
            )
        except Exception as exc:  # noqa: BLE001 - vendor library raises many types
            raise DataUnavailableError(f"yfinance history failed for {symbol} {timeframe.value}: {exc}") from exc
        if raw is None or len(raw) == 0:
            raise DataUnavailableError(f"yfinance returned no {timeframe.value} candles for {symbol}")
        if isinstance(raw.columns, pd.MultiIndex):  # yf.download style frames
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(0)
        df = normalize_ohlcv(raw, timeframe)
        lo = pd.Timestamp(start.replace(hour=0, minute=0, second=0, microsecond=0)) \
            if timeframe == Timeframe.D1 else pd.Timestamp(start)
        return df.loc[(df.index >= lo) & (df.index <= pd.Timestamp(end))]

    def get_quote(self, symbol: str) -> Quote | None:
        try:
            fi = self._ticker(symbol).fast_info
            ltp = _safe_float(fi.get("lastPrice") if hasattr(fi, "get") else getattr(fi, "last_price", None))
            if ltp is None or ltp <= 0:
                logger.warning(f"[yfinance] no last price for {symbol}")
                return None

            def g(key: str) -> float | None:
                return _safe_float(fi.get(key) if hasattr(fi, "get") else None)

            return Quote(
                symbol=symbol.upper(),
                ltp=ltp,
                timestamp=self._clock.now(),
                open=g("open"),
                high=g("dayHigh"),
                low=g("dayLow"),
                prev_close=g("previousClose"),
                volume=g("lastVolume"),
                source=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[yfinance] quote failed for {symbol}: {exc}")
            return None

    def healthcheck(self) -> bool:
        try:
            now = self._clock.now()
            df = self.get_candles("NIFTY50", Timeframe.D1, now - timedelta(days=10), now)
            return not df.empty
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[yfinance] healthcheck failed: {exc}")
            return False
