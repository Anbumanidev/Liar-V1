"""Angel One SmartAPI provider - MARKET DATA ONLY (rule 0.1).

``core.guard.install_order_guards()`` runs in ``__init__`` so SmartConnect's order methods
(placeOrder, modifyOrder, cancelOrder, convertPosition, GTT ...) raise
``RealTradingForbiddenError`` and the HTTP layer blocks order URLs. This module only calls
``generateSession``, ``getCandleData`` and ``ltpData``.

* Login: ``SmartConnect(api_key).generateSession(client_code, pin, TOTP(secret).now())``.
* Instrument master (symbol -> token) downloaded once per day into ``<cache_dir>/angelone/``.
* ``getCandleData`` per-request range limits are respected by chunking requests.
* Missing secrets -> provider reports unavailable (``healthcheck() is False``) and every
  call raises ``DataUnavailableError`` so ``MarketDataService`` falls back.
"""

from __future__ import annotations

import json
import threading
import time as _time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

ANGEL_INTERVALS: dict[Timeframe, str] = {
    Timeframe.M1: "ONE_MINUTE",
    Timeframe.M5: "FIVE_MINUTE",
    Timeframe.M15: "FIFTEEN_MINUTE",
    Timeframe.H1: "ONE_HOUR",
    Timeframe.D1: "ONE_DAY",
}

#: Max calendar days per getCandleData request (SmartAPI docs), kept slightly conservative.
ANGEL_MAX_DAYS_PER_REQUEST: dict[Timeframe, int] = {
    Timeframe.M1: 30,
    Timeframe.M5: 90,
    Timeframe.M15: 180,
    Timeframe.H1: 365,
    Timeframe.D1: 1900,
}

#: Project index pseudo-symbol -> instrument-master ``symbol`` of the NSE index (AMXIDX).
ANGEL_INDEX_NAMES: dict[str, str] = {
    "NIFTY50": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    "INDIAVIX": "India VIX",
    "NIFTYIT": "Nifty IT",
    "NIFTYPHARMA": "Nifty Pharma",
    "NIFTYFMCG": "Nifty FMCG",
    "NIFTYAUTO": "Nifty Auto",
    "NIFTYMETAL": "Nifty Metal",
    "NIFTYREALTY": "Nifty Realty",
    "NIFTYENERGY": "Nifty Energy",
    "NIFTYPSUBANK": "Nifty PSU Bank",
    "NIFTYPVTBANK": "Nifty Pvt Bank",
    "NIFTYFINSERVICE": "Nifty Fin Service",
    "NIFTYMEDIA": "Nifty Media",
}

_REQUEST_GAP_SECONDS = 0.35  # SmartAPI historical API: ~3 requests/second


def _secret(value: Any) -> str | None:
    if value is None:
        return None
    raw = value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)
    return raw.strip() or None


def parse_instrument_master(rows: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """Instrument-master rows -> ``{PROJECT_SYMBOL: (trading_symbol, token)}`` for NSE cash + indices."""
    out: dict[str, tuple[str, str]] = {}
    index_lookup = {v.upper(): k for k, v in ANGEL_INDEX_NAMES.items()}
    for row in rows:
        if str(row.get("exch_seg", "")).upper() != "NSE":
            continue
        tsym = str(row.get("symbol", ""))
        token = str(row.get("token", ""))
        if not tsym or not token:
            continue
        itype = str(row.get("instrumenttype", "")).upper()
        if itype == "AMXIDX":
            key = index_lookup.get(tsym.upper())
            if key:
                out[key] = (tsym, token)
        elif tsym.upper().endswith("-EQ"):
            out[tsym.upper()[:-3]] = (tsym, token)
    return out


class AngelOneProvider(DataProvider):
    """Angel One SmartAPI historical candles + LTP (data only).

    Args:
        secrets: ``config.secrets.Secrets`` (angel_api_key, angel_client_code, angel_pin,
            angel_totp_secret).
        cache_dir: cache root (``runtime/cache``); instrument master stored under ``angelone/``.
        clock: for quote timestamps and cache dating.
        smart_factory: ``api_key -> SmartConnect``-like (tests inject fakes).
        instrument_loader: ``() -> list[dict]`` instrument master rows (tests inject fakes).
    """

    name = "angelone"
    supported_timeframes = frozenset(ANGEL_INTERVALS)

    def __init__(
        self,
        secrets: "Secrets | None",
        cache_dir: Path | None = None,
        clock: Clock | None = None,
        smart_factory: Callable[[str], Any] | None = None,
        instrument_loader: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        install_order_guards()  # PAPER ONLY: neutralise SDK order methods + order URLs
        self._clock = clock or SystemClock()
        self._cache_dir = cache_dir
        self._smart_factory = smart_factory
        self._instrument_loader = instrument_loader
        self._api_key = _secret(getattr(secrets, "angel_api_key", None))
        self._client_code = _secret(getattr(secrets, "angel_client_code", None))
        self._pin = _secret(getattr(secrets, "angel_pin", None))
        self._totp_secret = _secret(getattr(secrets, "angel_totp_secret", None))
        self._smart: Any = None
        self._tokens: dict[str, tuple[str, str]] | None = None
        self._tokens_day: date | None = None
        self._lock = threading.RLock()
        self._last_request = 0.0
        self.last_error: str | None = None
        if not self.configured:
            logger.warning("[angelone] secrets missing (ANGEL_API_KEY / CLIENT_CODE / PIN / TOTP_SECRET): "
                           "provider unavailable, fallbacks will be used")

    # ------------------------------------------------------------------ session
    @property
    def configured(self) -> bool:
        """True when all four credentials are present."""
        return all((self._api_key, self._client_code, self._pin, self._totp_secret))

    def _login(self) -> Any:
        if not self.configured:
            raise DataUnavailableError("Angel One credentials not configured")
        with self._lock:
            if self._smart is not None:
                return self._smart
            try:
                import pyotp

                if self._smart_factory is not None:
                    smart = self._smart_factory(self._api_key)  # type: ignore[arg-type]
                else:
                    from SmartApi import SmartConnect

                    install_order_guards()  # re-run after the SDK import so its methods are patched
                    smart = SmartConnect(api_key=self._api_key)
                totp = pyotp.TOTP(self._totp_secret).now()  # type: ignore[arg-type]
                resp = smart.generateSession(self._client_code, self._pin, totp)
            except DataUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"login failed: {exc}"
                raise DataUnavailableError(f"Angel One login failed: {exc}") from exc
            if not isinstance(resp, dict) or not resp.get("status"):
                msg = resp.get("message") if isinstance(resp, dict) else resp
                self.last_error = f"login rejected: {msg}"
                raise DataUnavailableError(f"Angel One login rejected: {msg}")
            self._smart = smart
            logger.info("[angelone] SmartAPI session established (market data only)")
            return smart

    def _throttle(self) -> None:
        wait = _REQUEST_GAP_SECONDS - (_time.monotonic() - self._last_request)
        if wait > 0:
            _time.sleep(wait)
        self._last_request = _time.monotonic()

    # ------------------------------------------------------------------ instruments
    def _download_master(self) -> list[dict[str, Any]]:
        import requests

        resp = requests.get(INSTRUMENT_MASTER_URL, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _load_tokens(self) -> dict[str, tuple[str, str]]:
        today = self._clock.now().date()
        with self._lock:
            if self._tokens is not None and self._tokens_day == today:
                return self._tokens
            cache_file = (self._cache_dir / "angelone" / f"tokens_{today:%Y%m%d}.json") if self._cache_dir else None
            if cache_file is not None and cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    self._tokens = {k: (v[0], v[1]) for k, v in data.items()}
                    self._tokens_day = today
                    return self._tokens
                except (OSError, ValueError) as exc:
                    logger.warning(f"[angelone] token cache unreadable ({exc}); re-downloading")
            try:
                rows = self._instrument_loader() if self._instrument_loader else self._download_master()
            except Exception as exc:  # noqa: BLE001
                raise DataUnavailableError(f"Angel One instrument master download failed: {exc}") from exc
            tokens = parse_instrument_master(rows)
            if cache_file is not None:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(tokens), encoding="utf-8")
                except OSError as exc:
                    logger.warning(f"[angelone] could not write token cache: {exc}")
            self._tokens, self._tokens_day = tokens, today
            return tokens

    def instrument(self, symbol: str) -> tuple[str, str]:
        """``(trading_symbol, token)`` for a project symbol; raises if unknown."""
        tokens = self._load_tokens()
        key = symbol.strip().upper()
        if key not in tokens:
            raise DataUnavailableError(f"Angel One has no NSE instrument for {symbol}")
        return tokens[key]

    # ------------------------------------------------------------------ data
    def get_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        if timeframe not in ANGEL_INTERVALS:
            raise DataUnavailableError(f"Angel One does not serve {timeframe.value} natively")
        smart = self._login()
        _tsym, token = self.instrument(symbol)
        start, end = to_ist(start), to_ist(end)
        if end <= start:
            return empty_ohlcv()
        step = timedelta(days=ANGEL_MAX_DAYS_PER_REQUEST[timeframe])
        rows: list[list[Any]] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + step, end)
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": ANGEL_INTERVALS[timeframe],
                "fromdate": chunk_start.strftime("%Y-%m-%d %H:%M"),
                "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
            }
            self._throttle()
            try:
                resp = smart.getCandleData(params)
            except Exception as exc:  # noqa: BLE001
                raise DataUnavailableError(f"Angel One getCandleData failed for {symbol}: {exc}") from exc
            if not isinstance(resp, dict) or not resp.get("status"):
                msg = resp.get("message") if isinstance(resp, dict) else resp
                raise DataUnavailableError(f"Angel One getCandleData error for {symbol}: {msg}")
            rows.extend(resp.get("data") or [])
            chunk_start = chunk_end + timedelta(minutes=1)
        if not rows:
            raise DataUnavailableError(f"Angel One returned no {timeframe.value} candles for {symbol}")
        raw = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        raw["ts"] = pd.to_datetime(raw["ts"], utc=False)
        df = normalize_ohlcv(raw, timeframe)
        lo = pd.Timestamp(start.replace(hour=0, minute=0)) if timeframe == Timeframe.D1 else pd.Timestamp(start)
        return df.loc[(df.index >= lo) & (df.index <= pd.Timestamp(end))]

    def get_quote(self, symbol: str) -> Quote | None:
        try:
            smart = self._login()
            tsym, token = self.instrument(symbol)
            self._throttle()
            resp = smart.ltpData("NSE", tsym, token)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[angelone] quote failed for {symbol}: {exc}")
            self.last_error = str(exc)
            return None
        data = resp.get("data") if isinstance(resp, dict) else None
        if not data or data.get("ltp") in (None, 0):
            logger.warning(f"[angelone] empty LTP for {symbol}: {resp}")
            return None
        return Quote(
            symbol=symbol.upper(),
            ltp=float(data["ltp"]),
            timestamp=self._clock.now(),
            open=float(data["open"]) if data.get("open") else None,
            high=float(data["high"]) if data.get("high") else None,
            low=float(data["low"]) if data.get("low") else None,
            prev_close=float(data["close"]) if data.get("close") else None,
            source=self.name,
        )

    def healthcheck(self) -> bool:
        if not self.configured:
            return False
        try:
            self._login()
            return True
        except DataUnavailableError as exc:
            logger.warning(f"[angelone] healthcheck failed: {exc}")
            return False
