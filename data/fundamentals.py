"""Daily-cached fundamentals: yfinance ``.info`` + NSE surveillance / pledge / event flags.

Units follow ``core.types`` (percent units, ``15.0`` == 15 %):

=========================  ======================  ======================================
FundamentalSnapshot        yfinance ``.info`` key   conversion
=========================  ======================  ======================================
``pe``                     ``trailingPE``           ratio (as is)
``pb``                     ``priceToBook``          ratio (as is)
``roe``                    ``returnOnEquity``       fraction -> percent (x100)
``debt_to_equity``         ``debtToEquity``         yfinance gives percent -> ratio (/100)
``revenue_growth``         ``revenueGrowth``        fraction -> percent (x100)
``earnings_growth``        ``earningsGrowth``       fraction -> percent (x100)
``profit_margin``          ``profitMargins``        fraction -> percent (x100)
``promoter_holding``       ``heldPercentInsiders``  fraction -> percent (PROXY: insiders)
``market_cap``             ``marketCap``            rupees (as is)
``sector`` / ``industry``  ``sector``/``industry``  yfinance text (sector falls back to NSE)
=========================  ======================  ======================================

NSE overlay (``NSEClient`` / ``EventCalendar``, when available): ``in_asm``, ``in_gsm``,
``promoter_pledge_pct``, ``results_today`` (RESULTS event today) and ``corporate_action_today``
(ex-dividend / split / bonus / rights / ex-date today). ``sector_pe`` is the median trailing PE of
at least ``MIN_PEERS`` same-industry (else same-sector) peers already cached today.

Caching: the yfinance part is cached per IST day in memory and, with ``cache_dir``, in
``<cache_dir>/fundamentals/YYYY-MM-DD.json``. Failed fetches are NOT persisted (retried after
``retry_minutes``). The NSE overlay is recomputed (memoised ``nse_ttl_minutes``) so a late NSE
recovery is picked up. ``snapshot`` never raises: on failure it returns a snapshot whose
fundamental fields are ``None`` and ``extra["error"]`` explains why.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.clock import Clock, to_ist
from core.enums import EventType
from core.types import FundamentalSnapshot

if TYPE_CHECKING:  # pragma: no cover
    from data.events import EventCalendar
    from data.nse import NSEClient
    from data.universe import UniverseService

#: Minimum number of peers (excluding the symbol) for a sector/industry PE.
MIN_PEERS = 3

#: Event types that count as "corporate action today".
CORPORATE_ACTION_EVENT_TYPES: frozenset[EventType] = frozenset(
    {EventType.EX_DIVIDEND, EventType.EX_DATE, EventType.SPLIT, EventType.BONUS, EventType.RIGHTS}
)

#: info key -> (snapshot field, multiplier)
_NUMERIC_FIELDS: dict[str, tuple[str, float]] = {
    "trailingPE": ("pe", 1.0),
    "priceToBook": ("pb", 1.0),
    "returnOnEquity": ("roe", 100.0),
    "debtToEquity": ("debt_to_equity", 0.01),
    "revenueGrowth": ("revenue_growth", 100.0),
    "earningsGrowth": ("earnings_growth", 100.0),
    "profitMargins": ("profit_margin", 100.0),
    "heldPercentInsiders": ("promoter_holding", 100.0),
    "marketCap": ("market_cap", 1.0),
}
#: Extra informational values kept in ``FundamentalSnapshot.extra``.
_EXTRA_FIELDS: dict[str, str] = {
    "forwardPE": "forward_pe",
    "trailingEps": "trailing_eps",
    "bookValue": "book_value",
    "beta": "beta",
    "dividendYield": "dividend_yield",
    "fiftyTwoWeekHigh": "high_52w",
    "fiftyTwoWeekLow": "low_52w",
    "heldPercentInstitutions": "institutional_holding_frac",
}


def _num(value: Any) -> float | None:
    """Finite float or ``None`` (yfinance mixes None, NaN, 'Infinity' strings ...)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def map_info(symbol: str, as_of: date, info: dict[str, Any], nse_sector: str | None = None) -> FundamentalSnapshot:
    """Map a yfinance ``.info`` dict onto a ``FundamentalSnapshot`` (see module table).

    Raises:
        ValueError: when ``info`` carries none of the expected fields (unknown symbol).
    """
    values: dict[str, Any] = {}
    for key, (fld, mult) in _NUMERIC_FIELDS.items():
        v = _num(info.get(key))
        if v is not None:
            values[fld] = round(v * mult, 6)
    if values.get("promoter_holding") is not None and not 0 <= values["promoter_holding"] <= 100:
        values.pop("promoter_holding")  # corrupt insider %, drop rather than mislead
    sector = str(info.get("sector") or "").strip() or None
    industry = str(info.get("industry") or "").strip() or None
    if not values and sector is None and industry is None:
        raise ValueError("no fundamentals in yfinance info")
    extra: dict[str, Any] = {"source": "yfinance", "promoter_holding_source": "heldPercentInsiders (proxy)"}
    for key, name in _EXTRA_FIELDS.items():
        v = _num(info.get(key))
        if v is not None:
            extra[name] = v
    if info.get("longName") or info.get("shortName"):
        extra["company"] = str(info.get("longName") or info.get("shortName"))
    if nse_sector:
        extra["nse_sector"] = nse_sector
    return FundamentalSnapshot(symbol=symbol, as_of=as_of, sector=sector or nse_sector, industry=industry,
                               extra=extra, **values)


def yfinance_info(symbol: str) -> dict[str, Any]:
    """Default ``info_fetcher``: ``yfinance.Ticker(<symbol>.NS).info``."""
    import yfinance as yf

    from data.providers.yfinance_provider import yf_symbol

    info = yf.Ticker(yf_symbol(symbol)).info
    return dict(info or {})


class FundamentalsService:
    """Per-day fundamentals snapshot of a symbol.

    Args:
        clock: time source (cache day, retry timers).
        nse: optional ``NSEClient`` (ASM/GSM lists, pledge %, today's results / ex-dates).
        cache_dir: cache root (``runtime/cache``); daily files in ``<cache_dir>/fundamentals/``.
            ``None`` = in-memory only.
        universe: optional ``UniverseService`` (NSE sector fallback, peer grouping).
        info_fetcher: ``symbol -> yfinance-style info dict`` (tests inject fakes; default
            ``yfinance_info``). Receives the plain NSE symbol (``"RELIANCE"``).
        events: optional ``EventCalendar`` (DB events) used together with ``nse`` for
            ``results_today`` / ``corporate_action_today``.
        retry_minutes: back-off before re-fetching a symbol whose fetch failed.
        nse_ttl_minutes: memo lifetime of NSE lookups.
    """

    def __init__(
        self,
        clock: Clock,
        nse: "NSEClient | None" = None,
        cache_dir: Path | None = None,
        universe: "UniverseService | None" = None,
        info_fetcher: Callable[[str], dict[str, Any]] | None = None,
        *,
        events: "EventCalendar | None" = None,
        retry_minutes: float = 30.0,
        nse_ttl_minutes: float = 30.0,
    ) -> None:
        self.clock = clock
        self.nse = nse
        self.cache_dir = (Path(cache_dir) / "fundamentals") if cache_dir else None
        self.universe = universe
        self.events = events
        self._fetch_info = info_fetcher or yfinance_info
        self._retry = timedelta(minutes=retry_minutes)
        self._nse_ttl = timedelta(minutes=nse_ttl_minutes)
        self._day: date | None = None
        self._cache: dict[str, FundamentalSnapshot] = {}
        self._failures: dict[str, tuple[datetime, str]] = {}
        self._memo: dict[str, tuple[datetime, Any]] = {}
        self._lock = threading.RLock()
        self.last_error: str | None = None

    # ------------------------------------------------------------------ public API
    def snapshot(self, symbol: str) -> FundamentalSnapshot:
        """Today's ``FundamentalSnapshot`` of ``symbol`` (never raises)."""
        sym = symbol.strip().upper()
        today: date | None = None
        try:
            now = to_ist(self.clock.now())
            today = now.date()
            base = self._base(sym, today, now)
            snap = self._with_nse(base, sym, today, now)
            return self._with_sector_pe(snap)
        except Exception as exc:  # noqa: BLE001 - fundamentals must never break the pipeline
            self.last_error = f"{sym}: {exc}"
            logger.warning(f"[fundamentals] snapshot failed for {sym}: {exc}")
            return FundamentalSnapshot(symbol=sym, as_of=today or date.min, extra={"error": str(exc)})

    def snapshots(self, symbols: Iterable[str]) -> dict[str, FundamentalSnapshot]:
        """``snapshot`` for several symbols (pre-market warm-up)."""
        return {s.strip().upper(): self.snapshot(s) for s in symbols}

    def cache_path(self, day: date) -> Path | None:
        """Daily cache file of ``day`` (``None`` without ``cache_dir``)."""
        return None if self.cache_dir is None else self.cache_dir / f"{day:%Y-%m-%d}.json"

    def clear_cache(self) -> None:
        """Forget in-memory snapshots, failures and NSE memos (disk files are kept)."""
        with self._lock:
            self._day = None
            self._cache.clear()
            self._failures.clear()
            self._memo.clear()

    # ------------------------------------------------------------------ yfinance part
    def _nse_sector(self, sym: str) -> str | None:
        if self.universe is None:
            return None
        try:
            return self.universe.sector_of(sym)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[fundamentals] sector lookup failed for {sym}: {exc}")
            return None

    def _ensure_day(self, today: date) -> None:
        """Switch the in-memory cache to ``today`` (loading the day's file if present)."""
        if self._day == today:
            return
        self._day = today
        self._cache = {}
        self._failures = {}
        self._memo = {}
        path = self.cache_path(today)
        if path is None or not path.exists():
            return
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            for sym, data in (blob or {}).items():
                try:
                    self._cache[sym] = FundamentalSnapshot.model_validate(data)
                except Exception as exc:  # noqa: BLE001 - skip one bad row, keep the rest
                    logger.warning(f"[fundamentals] bad cached row {sym} in {path}: {exc}")
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning(f"[fundamentals] cache unreadable {path}: {exc}")

    def _persist(self, today: date) -> None:
        path = self.cache_path(today)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({s: snap.model_dump(mode="json") for s, snap in sorted(self._cache.items())}),
                           encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning(f"[fundamentals] could not write cache {path}: {exc}")

    def _base(self, sym: str, today: date, now: datetime) -> FundamentalSnapshot:
        """yfinance-derived snapshot (cached per day; failures retried after ``retry_minutes``)."""
        with self._lock:
            self._ensure_day(today)
            hit = self._cache.get(sym)
            if hit is not None:
                return hit
            failed = self._failures.get(sym)
            if failed is not None and now < failed[0]:
                return self._failed_snapshot(sym, today, failed[1])
        nse_sector = self._nse_sector(sym)
        try:
            info = self._fetch_info(sym)
            if not isinstance(info, dict):
                raise TypeError(f"info fetcher returned {type(info).__name__}, expected dict")
            snap = map_info(sym, today, info, nse_sector)
        except Exception as exc:  # noqa: BLE001 - vendor errors of every kind
            msg = f"yfinance info unavailable: {exc}"
            self.last_error = f"{sym}: {msg}"
            logger.warning(f"[fundamentals] {sym}: {msg} (retry in {self._retry})")
            with self._lock:
                self._failures[sym] = (now + self._retry, msg)
            return self._failed_snapshot(sym, today, msg)
        with self._lock:
            if self._day == today:
                self._cache[sym] = snap
                self._failures.pop(sym, None)
                self._persist(today)
        return snap

    def _failed_snapshot(self, sym: str, today: date, msg: str) -> FundamentalSnapshot:
        nse_sector = self._nse_sector(sym)
        extra: dict[str, Any] = {"error": msg, "source": "none"}
        if nse_sector:
            extra["nse_sector"] = nse_sector
        return FundamentalSnapshot(symbol=sym, as_of=today, sector=nse_sector, extra=extra)

    # ------------------------------------------------------------------ NSE overlay
    def _memoised(self, key: str, now: datetime, fn: Callable[[], Any], default: Any) -> Any:
        with self._lock:
            hit = self._memo.get(key)
            if hit is not None and timedelta(0) <= now - hit[0] < self._nse_ttl:
                return hit[1]
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - NSE data is optional
            logger.warning(f"[fundamentals] NSE lookup {key} failed: {exc}")
            return default
        with self._lock:
            self._memo[key] = (now, value)
        return value

    def _with_nse(self, base: FundamentalSnapshot, sym: str, today: date, now: datetime) -> FundamentalSnapshot:
        update: dict[str, Any] = {}
        extra = dict(base.extra)
        nse = self.nse
        if nse is not None:
            update["in_asm"] = sym in self._memoised("asm", now, lambda: set(nse.asm_symbols()), set())
            update["in_gsm"] = sym in self._memoised("gsm", now, lambda: set(nse.gsm_symbols()), set())
            update["promoter_pledge_pct"] = self._memoised(f"pledge:{sym}", now, lambda: nse.pledge_pct(sym), None)
        extra["nse_checked"] = nse is not None
        results, corp = False, False
        todays: list[Any] = []
        if nse is not None:
            todays += self._memoised("events_today", now, lambda: list(nse.event_calendar(today, today)), [])
            todays += self._memoised("ca_today", now, lambda: list(nse.corporate_actions(today, today)), [])
        if self.events is not None:
            todays += self._memoised(f"db_events:{sym}", now,
                                     lambda: list(self.events.events_for(sym, today, today)), [])  # type: ignore[union-attr]
        for ev in todays:
            if (ev.symbol or "").upper() != sym or ev.event_date != today:
                continue
            if ev.event_type == EventType.RESULTS:
                results = True
            elif ev.event_type in CORPORATE_ACTION_EVENT_TYPES:
                corp = True
        update.update(results_today=results, corporate_action_today=corp, extra=extra)
        return base.model_copy(update=update)

    # ------------------------------------------------------------------ sector PE
    def _with_sector_pe(self, snap: FundamentalSnapshot) -> FundamentalSnapshot:
        """Median trailing PE of today's cached peers (same industry, else same sector)."""
        if snap.sector_pe is not None:
            return snap
        with self._lock:
            peers = [s for s in self._cache.values() if s.symbol != snap.symbol and s.pe is not None and s.pe > 0]
        groups = (("industry", snap.industry, lambda s: s.industry),
                  ("sector", snap.extra.get("nse_sector") or snap.sector,
                   lambda s: s.extra.get("nse_sector") or s.sector))
        for basis, value, key in groups:
            if not value:
                continue
            pes = [float(s.pe) for s in peers if key(s) == value]  # type: ignore[arg-type]
            if len(pes) >= MIN_PEERS:
                extra = {**snap.extra, "sector_pe_basis": f"{basis}:{value} (n={len(pes)})"}
                return snap.model_copy(update={"sector_pe": round(statistics.median(pes), 4), "extra": extra})
        return snap
