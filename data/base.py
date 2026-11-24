"""Pluggable market-data provider interface (section 1: abstract, pluggable data providers).

Providers are MARKET DATA ONLY. No provider may expose order placement (rule 0.1); broker
providers call ``core.guard.install_order_guards()`` on construction.

Contract for ``get_candles``:
* returns a contract-compliant frame (``core.types.validate_ohlcv``), bar OPEN times in IST,
  daily bars at 09:15 IST, covering ``[start, end]`` as far as the source allows;
* may include bars that are still forming - ``data.service.MarketDataService`` (and
  ``ReplayDataProvider``) apply the closed-bars-only rule;
* raises ``core.errors.DataUnavailableError`` (or returns an empty frame) on failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from core.enums import Timeframe
from core.types import Quote


class DataProvider(ABC):
    """Abstract market-data source."""

    #: short provider id ("yfinance", "angelone", "upstox", "replay")
    name: str = "base"
    #: timeframes the source serves natively (others are resampled by the service)
    supported_timeframes: frozenset[Timeframe] = frozenset(Timeframe)

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        """OHLCV candles for ``symbol`` with bar open times in ``[start, end]``."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote | None:
        """Latest quote, or ``None`` if unavailable."""

    def get_quotes(self, symbols: Iterable[str]) -> dict[str, Quote]:
        """Quotes for several symbols (default: one call per symbol, failures skipped)."""
        out: dict[str, Quote] = {}
        for sym in symbols:
            try:
                q = self.get_quote(sym)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the batch
                logger.warning(f"[{self.name}] quote failed for {sym}: {exc}")
                continue
            if q is not None:
                out[sym] = q
        return out

    @abstractmethod
    def healthcheck(self) -> bool:
        """True if the provider is usable right now (credentials ok / reachable)."""

    def supports(self, timeframe: Timeframe) -> bool:
        """True if ``timeframe`` is served natively."""
        return timeframe in self.supported_timeframes

    def max_history(self, timeframe: Timeframe) -> timedelta | None:
        """Maximum look-back the source serves for ``timeframe`` (``None`` = unlimited/unknown)."""
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name}>"
