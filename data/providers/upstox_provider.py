"""Upstox provider (minimal) - MARKET DATA ONLY via the REST API and httpx.

* Historical candles: ``GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to}/{from}``
  (unit minutes|hours|days), chunked per Upstox range limits.
* LTP: ``GET /v2/market-quote/ltp?instrument_key=...``.
* Instrument keys (``NSE_EQ|<ISIN>``) come from the daily NSE instrument file, cached in
  ``<cache_dir>/upstox/``; indices use ``NSE_INDEX|<name>``.
* Without ``UPSTOX_ACCESS_TOKEN`` the provider is unavailable (healthcheck False, calls raise).

``core.guard.install_order_guards()`` runs in ``__init__`` - the httpx transport refuses any
order URL even if one were ever constructed.
"""

from __future__ import annotations

import gzip
import json
import threading
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote as urlquote

import pandas as pd
from loguru import logger

from core.clock import Clock, SystemClock, to_ist
from core.enums import Timeframe
from core.errors import DataUnavailableError
from core.guard import install_order_guards
from core.types import Quote, empty_ohlcv
from data.bars import normalize_ohlcv
from data.base import DataProvider

if TYPE_CHECKING:  # pragma: no cover
    from config.secrets import Secrets

API_BASE = "https://api.upstox.com"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

#: timeframe -> (unit, interval, max days per request)
UPSTOX_INTERVALS: dict[Timeframe, tuple[str, str, int]] = {
    Timeframe.M1: ("minutes", "1", 28),
    Timeframe.M5: ("minutes", "5", 28),
    Timeframe.M15: ("minutes", "15", 28),
    Timeframe.H1: ("hours", "1", 85),
    Timeframe.D1: ("days", "1", 3000),
}

UPSTOX_INDEX_KEYS: dict[str, str] = {
    "NIFTY50": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "INDIAVIX": "NSE_INDEX|India VIX",
    "NIFTYIT": "NSE_INDEX|Nifty IT",
    "NIFTYPHARMA": "NSE_INDEX|Nifty Pharma",
    "NIFTYFMCG": "NSE_INDEX|Nifty FMCG",
    "NIFTYAUTO": "NSE_INDEX|Nifty Auto",
    "NIFTYMETAL": "NSE_INDEX|Nifty Metal",
    "NIFTYREALTY": "NSE_INDEX|Nifty Realty",
    "NIFTYENERGY": "NSE_INDEX|Nifty Energy",
    "NIFTYPSUBANK": "NSE_INDEX|Nifty PSU Bank",
    "NIFTYPVTBANK": "NSE_INDEX|Nifty Private Bank",
    "NIFTYFINSERVICE": "NSE_INDEX|Nifty Fin Service",
    "NIFTYMEDIA": "NSE_INDEX|Nifty Media",
}


def parse_upstox_instruments(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Upstox NSE instrument rows -> ``{TRADING_SYMBOL: instrument_key}`` for cash equities."""
    out: dict[str, str] = {}
    for row in rows:
        if row.get("segment") != "NSE_EQ":
            continue
        sym = str(row.get("trading_symbol", "")).upper()
        key = row.get("instrument_key")
        if sym and key and str(row.get("instrument_type", "EQ")).upper() in ("EQ", "BE", ""):
            out.setdefault(sym, str(key))
    return out


class UpstoxProvider(DataProvider):
    """Upstox REST market data (historical candles + LTP).

    Args:
        secrets: ``config.secrets.Secrets`` (``upstox_access_token``).
        cache_dir: cache root for the instrument map.
        clock: quote timestamps / cache dating.
        client: an ``httpx.Client`` (tests inject a fake transport).
        instrument_loader: ``() -> list[dict]`` instrument rows (tests).
    """

    name = "upstox"
    supported_timeframes = frozenset(UPSTOX_INTERVALS)

    def __init__(
        self,
        secrets: "Secrets | None",
        cache_dir: Path | None = None,
        clock: Clock | None = None,
        client: Any = None,
        instrument_loader: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        install_order_guards()
        tok = getattr(secrets, "upstox_access_token", None)
        raw = tok.get_secret_value() if hasattr(tok, "get_secret_value") else tok
        self._token: str | None = (raw or "").strip() or None
        self._clock = clock or SystemClock()
        self._cache_dir = cache_dir
        self._client = client
        self._instrument_loader = instrument_loader
        self._keys: dict[str, str] | None = None
        self._keys_day: date | None = None
        self._lock = threading.RLock()
        if not self._token:
            logger.warning("[upstox] UPSTOX_ACCESS_TOKEN missing: provider unavailable")

    @property
    def configured(self) -> bool:
        """True when an access token is present."""
        return self._token is not None

    def _http(self) -> Any:
        if not self.configured:
            raise DataUnavailableError("Upstox access token not configured")
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=20.0)
        return self._client

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        client = self._http()
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        try:
            resp = client.get(f"{API_BASE}{path}", params=params, headers=headers)
        except Exception as exc:  # noqa: BLE001
            raise DataUnavailableError(f"Upstox request failed: {exc}") from exc
        if resp.status_code != 200:
            raise DataUnavailableError(f"Upstox HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        if payload.get("status") not in (None, "success"):
            raise DataUnavailableError(f"Upstox error: {payload.get('errors') or payload}")
        return payload

    def _load_keys(self) -> dict[str, str]:
        today = self._clock.now().date()
        with self._lock:
            if self._keys is not None and self._keys_day == today:
                return self._keys
            cache_file = (self._cache_dir / "upstox" / f"instruments_{today:%Y%m%d}.json") if self._cache_dir else None
            if cache_file is not None and cache_file.exists():
                try:
                    self._keys = json.loads(cache_file.read_text(encoding="utf-8"))
                    self._keys_day = today
                    return self._keys  # type: ignore[return-value]
                except (OSError, ValueError) as exc:
                    logger.warning(f"[upstox] instrument cache unreadable ({exc}); re-downloading")
            try:
                if self._instrument_loader is not None:
                    rows = self._instrument_loader()
                else:
                    import httpx

                    resp = httpx.get(INSTRUMENTS_URL, timeout=60.0)
                    resp.raise_for_status()
                    rows = json.loads(gzip.decompress(resp.content))
            except Exception as exc:  # noqa: BLE001
                raise DataUnavailableError(f"Upstox instrument download failed: {exc}") from exc
            keys = parse_upstox_instruments(rows)
            if cache_file is not None:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(keys), encoding="utf-8")
                except OSError as exc:
                    logger.warning(f"[upstox] could not write instrument cache: {exc}")
            self._keys, self._keys_day = keys, today
            return keys

    def instrument_key(self, symbol: str) -> str:
        """Upstox instrument key for a project symbol."""
        sym = symbol.strip().upper()
        if sym in UPSTOX_INDEX_KEYS:
            return UPSTOX_INDEX_KEYS[sym]
        keys = self._load_keys()
        if sym not in keys:
            raise DataUnavailableError(f"Upstox has no NSE_EQ instrument for {symbol}")
        return keys[sym]

    def get_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        if timeframe not in UPSTOX_INTERVALS:
            raise DataUnavailableError(f"Upstox does not serve {timeframe.value} natively")
        self._http()
        key = urlquote(self.instrument_key(symbol), safe="")
        unit, interval, max_days = UPSTOX_INTERVALS[timeframe]
        start, end = to_ist(start), to_ist(end)
        if end <= start:
            return empty_ohlcv()
        rows: list[list[Any]] = []
        chunk_start = start.date()
        while chunk_start <= end.date():
            chunk_end = min(chunk_start + timedelta(days=max_days), end.date())
            payload = self._get(f"/v3/historical-candle/{key}/{unit}/{interval}/{chunk_end:%Y-%m-%d}/{chunk_start:%Y-%m-%d}")
            rows.extend((payload.get("data") or {}).get("candles") or [])
            chunk_start = chunk_end + timedelta(days=1)
        if end.date() == self._clock.now().date() and timeframe != Timeframe.D1:
            try:  # today's intraday candles live on a separate endpoint
                payload = self._get(f"/v3/historical-candle/intraday/{key}/{unit}/{interval}")
                rows.extend((payload.get("data") or {}).get("candles") or [])
            except DataUnavailableError as exc:
                logger.warning(f"[upstox] intraday candles unavailable for {symbol}: {exc}")
        if not rows:
            raise DataUnavailableError(f"Upstox returned no {timeframe.value} candles for {symbol}")
        raw = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "high", "low", "close", "volume"])
        raw["ts"] = pd.to_datetime(raw["ts"], utc=False)
        df = normalize_ohlcv(raw, timeframe)
        lo = pd.Timestamp(start.replace(hour=0, minute=0)) if timeframe == Timeframe.D1 else pd.Timestamp(start)
        return df.loc[(df.index >= lo) & (df.index <= pd.Timestamp(end))]

    def get_quote(self, symbol: str) -> Quote | None:
        try:
            key = self.instrument_key(symbol)
            payload = self._get("/v2/market-quote/ltp", params={"instrument_key": key})
        except DataUnavailableError as exc:
            logger.warning(f"[upstox] quote failed for {symbol}: {exc}")
            return None
        data = payload.get("data") or {}
        for item in data.values():
            ltp = item.get("last_price")
            if ltp:
                return Quote(symbol=symbol.upper(), ltp=float(ltp), timestamp=self._clock.now(), source=self.name)
        logger.warning(f"[upstox] empty LTP for {symbol}")
        return None

    def healthcheck(self) -> bool:
        if not self.configured:
            return False
        try:
            return self.get_quote("NIFTY50") is not None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[upstox] healthcheck failed: {exc}")
            return False
