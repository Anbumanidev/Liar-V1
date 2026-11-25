"""NSE public-data client: holidays, ASM/GSM, FII/DII, corporate actions, event calendar,
announcements, delivery % (bhavcopy), advance/decline and promoter pledge.

Design
------
* Talks to NSE's public JSON endpoints (the same ones nsepython / jugaad-data wrap) with a
  browser-like ``requests`` session; endpoints and response shapes were verified 2026-11-24.
* EVERY call is cached as JSON under ``<cache_dir>/nse/`` with a per-endpoint TTL measured on
  the injected clock. On network/parse failure the client logs a loguru warning, records an
  ``errors`` row when a ``session_factory`` is available, serves the stale cache if any and
  otherwise returns an EMPTY result (empty set / list / dict / None) - it never raises.
* ``offline=True`` (REPLAY) serves only what is cached - NSE endpoints describe *today*, so
  fetching them while replaying the past would be look-ahead.
* Pure parsing helpers (``parse_*``) are module-level so they can be unit-tested offline.
"""

from __future__ import annotations

import csv
import functools
import io
import json
import re
import threading
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger

from core.clock import IST, Clock, SystemClock, to_ist
from core.enums import EventType
from core.types import Headline, MarketEvent

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm import Session, sessionmaker

NSE_BASE = "https://www.nseindia.com"
NSE_ARCHIVES = "https://nsearchives.nseindia.com"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
#: Equity series whose corporate actions matter (excludes InvIT/REIT/bond series).
EQUITY_SERIES = {"EQ", "BE", "BZ", "SM", "ST", ""}

_TTL_SHORT = timedelta(hours=1)
_TTL_DAY = timedelta(hours=12)
_TTL_WEEK = timedelta(days=7)

# =========================================================================== parsing helpers

_NUM = r"([0-9]+(?:\.[0-9]+)?)"
_CUR = r"(?:rs\.?|re\.?|inr|₹)\s*"
_RE_DIVIDEND = re.compile(r"dividend[^0-9/]*?" + _CUR + _NUM, re.IGNORECASE)
_RE_BONUS = re.compile(r"bonus\s*(\d+)\s*:\s*(\d+)", re.IGNORECASE)
_RE_SPLIT = re.compile(r"split.*?from\s*" + _CUR + _NUM + r".*?to\s*" + _CUR + _NUM, re.IGNORECASE)
_RE_RIGHTS = re.compile(r"rights\s*(\d+)\s*:\s*(\d+)(?:.*?premium\s*" + _CUR + _NUM + r")?", re.IGNORECASE)
_RE_AGM = re.compile(r"annual\s+general\s+meeting|\bagm\b", re.IGNORECASE)
_RE_BUYBACK = re.compile(r"buy\s*-?\s*back", re.IGNORECASE)
_RE_SKIP = re.compile(r"interest\s+payment|distribution|redemption", re.IGNORECASE)


def parse_nse_date(value: Any) -> date | None:
    """``"02-Nov-2026"`` / ``"02-11-2026"`` / ISO -> date (None for ``"-"``/blank/garbage)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("-", "NA", "N.A."):
        return None
    text = text.split(" ")[0]
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_nse_datetime(value: Any) -> datetime | None:
    """``"24-Nov-2026 19:49:52"`` -> IST datetime (None if unparseable)."""
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=IST)
        except ValueError:
            continue
    return None


def _f(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() in ("", "-"):
            return None
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def parse_ca_subject(subject: str) -> list[dict[str, Any]]:
    """Parse an NSE corporate-action subject into event dicts.

    Returns a list of ``{"event_type", "ratio", "amount", "details"}``:
    * dividends -> ``EX_DIVIDEND`` with ``amount`` = Rs per share (several dividends summed);
    * ``Bonus a:b`` (a new for every b held) -> ``BONUS``, ``ratio = (a + b) / b``;
    * ``Face Value Split From Rs X To Rs Y`` -> ``SPLIT``, ``ratio = X / Y``;
    * ``Rights a:b @ Premium Rs p`` -> ``RIGHTS`` (details only);
    * AGM -> ``AGM``; buy-back -> ``EX_DATE`` (kind BUYBACK); interest/distribution -> skipped;
    * anything else -> generic ``EX_DATE``.
    """
    s = subject or ""
    out: list[dict[str, Any]] = []
    amounts = [float(m.group(1)) for m in _RE_DIVIDEND.finditer(s)]
    if amounts:
        out.append({"event_type": EventType.EX_DIVIDEND, "ratio": None, "amount": round(sum(amounts), 4),
                    "details": {"dividends": amounts}})
    if m := _RE_BONUS.search(s):
        a, b = int(m.group(1)), int(m.group(2))
        if b > 0:
            out.append({"event_type": EventType.BONUS, "ratio": (a + b) / b, "amount": None,
                        "details": {"bonus": f"{a}:{b}"}})
    if m := _RE_SPLIT.search(s):
        old, new = float(m.group(1)), float(m.group(2))
        if new > 0:
            out.append({"event_type": EventType.SPLIT, "ratio": old / new, "amount": None,
                        "details": {"face_value_from": old, "face_value_to": new}})
    if m := _RE_RIGHTS.search(s):
        out.append({"event_type": EventType.RIGHTS, "ratio": None, "amount": None,
                    "details": {"rights": f"{m.group(1)}:{m.group(2)}",
                                "premium": float(m.group(3)) if m.group(3) else None}})
    if _RE_AGM.search(s):
        out.append({"event_type": EventType.AGM, "ratio": None, "amount": None, "details": {}})
    if not out and _RE_BUYBACK.search(s):
        out.append({"event_type": EventType.EX_DATE, "ratio": None, "amount": None, "details": {"kind": "BUYBACK"}})
    if not out and s.strip() and not _RE_SKIP.search(s):
        out.append({"event_type": EventType.EX_DATE, "ratio": None, "amount": None, "details": {"kind": "OTHER"}})
    return out


def parse_corporate_actions(rows: list[dict[str, Any]]) -> list[MarketEvent]:
    """NSE ``corporates-corporateActions`` rows -> ``MarketEvent`` list (keyed on ex-date).

    Events of the same ``(symbol, type, ex-date)`` are merged (dividend amounts summed).
    """
    merged: dict[tuple[str, EventType, date], MarketEvent] = {}
    for row in rows or []:
        sym = str(row.get("symbol", "")).strip().upper()
        series = str(row.get("series") or "").strip().upper()
        ex = parse_nse_date(row.get("exDate"))
        if not sym or ex is None or series not in EQUITY_SERIES:
            continue
        subject = str(row.get("subject", "")).strip()
        for ev in parse_ca_subject(subject):
            key = (sym, ev["event_type"], ex)
            details = {**ev["details"], "subject": subject, "record_date": str(parse_nse_date(row.get("recDate")) or ""),
                       "face_value": row.get("faceVal"), "series": series}
            if key in merged and ev["event_type"] == EventType.EX_DIVIDEND:
                prev = merged[key]
                merged[key] = prev.model_copy(update={"amount": (prev.amount or 0) + (ev["amount"] or 0),
                                                      "title": f"{prev.title}; {subject}"})
                continue
            merged[key] = MarketEvent(symbol=sym, event_type=ev["event_type"], event_date=ex, title=subject,
                                      details=details, ratio=ev["ratio"], amount=ev["amount"], source="NSE")
    return sorted(merged.values(), key=lambda e: (e.event_date, e.symbol or ""))


def parse_event_calendar(rows: list[dict[str, Any]]) -> list[MarketEvent]:
    """NSE ``event-calendar`` rows -> RESULTS / BOARD_MEETING events."""
    out: dict[tuple[str, EventType, date], MarketEvent] = {}
    for row in rows or []:
        sym = str(row.get("symbol", "")).strip().upper()
        d = parse_nse_date(row.get("date"))
        if not sym or d is None:
            continue
        purpose = str(row.get("purpose", "")).strip()
        etype = EventType.RESULTS if "result" in purpose.lower() else EventType.BOARD_MEETING
        key = (sym, etype, d)
        out[key] = MarketEvent(symbol=sym, event_type=etype, event_date=d,
                               title=f"Board meeting: {purpose}" if purpose else "Board meeting",
                               details={"purpose": purpose, "description": str(row.get("bm_desc", ""))[:1000],
                                        "company": row.get("company")},
                               source="NSE")
    return sorted(out.values(), key=lambda e: (e.event_date, e.symbol or ""))


def parse_announcements(
    rows: list[dict[str, Any]], since: datetime, until: datetime, symbol: str | None = None
) -> list[Headline]:
    """NSE ``corporate-announcements`` rows -> headlines published in ``[since, until]``."""
    out: list[Headline] = []
    sym_filter = symbol.upper() if symbol else None
    for row in rows or []:
        sym = str(row.get("symbol", "")).strip().upper() or None
        if sym_filter and sym != sym_filter:
            continue
        ts = parse_nse_datetime(row.get("an_dt") or row.get("sort_date") or row.get("exchdisstime"))
        if ts is None or ts < since or ts > until:
            continue
        desc = str(row.get("desc", "")).strip()
        text = str(row.get("attchmntText", "") or "").strip()
        title = f"{sym or row.get('sm_name', '')}: {desc}" if desc else (text[:160] or "NSE announcement")
        out.append(Headline(source="NSE", title=title, url=row.get("attchmntFile") or None, published_at=ts,
                            symbol=sym, summary=text[:2000] or None))
    out.sort(key=lambda h: h.published_at, reverse=True)
    return out


def parse_bhavcopy_delivery(text: str) -> dict[str, float]:
    """``sec_bhavdata_full_DDMMYYYY.csv`` -> ``{SYMBOL: delivery %}`` for EQ series."""
    out: dict[str, float] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        clean = {str(k).strip().upper(): (v or "").strip() for k, v in row.items() if k}
        if clean.get("SERIES") not in ("EQ", "BE"):
            continue
        pct = _f(clean.get("DELIV_PER"))
        if pct is not None:
            out.setdefault(clean.get("SYMBOL", "").upper(), pct)
    return out


def parse_holiday_master(payload: Any, segment: str = "CM") -> dict[date, str]:
    """NSE ``holiday-master?type=trading`` payload -> ``{date: description}`` for one segment."""
    out: dict[date, str] = {}
    rows = payload.get(segment, []) if isinstance(payload, dict) else []
    for row in rows:
        d = parse_nse_date(row.get("tradingDate"))
        if d is not None:
            out[d] = str(row.get("description", "NSE holiday")).strip()
    return out


_RE_PLEDGE_KEY = re.compile(r"(pledg|encumb).*(perc|per|pct|%)|(perc|per|pct|%).*(pledg|encumb)", re.IGNORECASE)
_RE_PROMOTER_SHARES_KEY = re.compile(r"(perc|pct).*promoter.*share", re.IGNORECASE)


def _pledge_key_rank(key: str) -> int | None:
    """Rank of a payload key as the 'pledged % of promoter holding' field (lower is better).

    ``0`` explicit pledged/encumbered percentage, ``1`` the same expressed of TOTAL shares,
    ``2`` NSE-style ``percPromoterShares``; ``None`` otherwise. A promoter HOLDING percentage
    (e.g. ``percPromoterHolding``) is never a pledge figure.
    """
    k = key.lower()
    if "holding" in k:
        return None
    if _RE_PLEDGE_KEY.search(k):
        return 1 if "tot" in k else 0
    if _RE_PROMOTER_SHARES_KEY.search(k):
        return 2
    return None


def parse_pledge(payload: Any) -> float | None:
    """Best-effort promoter-pledge % from ``corporate-pledgedata`` (field names vary; first row = latest)."""
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not rows or not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        best: tuple[int, float] | None = None
        for key, value in row.items():
            rank = _pledge_key_rank(str(key))
            num = _f(value) if rank is not None else None
            if rank is not None and num is not None and (best is None or rank < best[0]):
                best = (rank, num)
        if best is not None:
            return best[1]
    return None


# =========================================================================== client

_F = TypeVar("_F", bound=Callable[..., Any])


def _failsafe(default: Callable[[], Any]) -> Callable[[_F], _F]:
    """Decorator for public ``NSEClient`` calls: any exception (odd payload shapes included) is
    logged / recorded via ``NSEClient._failure`` and ``default()`` is returned - never raises."""

    def deco(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(self: "NSEClient", *args: Any, **kwargs: Any) -> Any:
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - NSE responses are unreliable
                self._failure(f"{fn.__name__} failed: {type(exc).__name__}: {exc}",
                              {"args": [str(a) for a in args]})
                return default()

        return wrapper  # type: ignore[return-value]

    return deco



class NSEClient:
    """Cached, failure-safe NSE public-data client.

    Args:
        cache_dir: cache root (``runtime/cache``); files go to ``<cache_dir>/nse/``.
        clock: time source for TTLs and default date ranges.
        session_factory: optional; failures are recorded in the ``errors`` table.
        http_get: ``(url, params) -> response`` with ``status_code``, ``content``, ``json()``
            (tests inject fakes; default is a browser-like requests session).
        offline: serve cached data only (REPLAY).
        timeout: HTTP timeout seconds.
    """

    def __init__(
        self,
        cache_dir: Path | None,
        clock: Clock | None = None,
        *,
        session_factory: "sessionmaker[Session] | None" = None,
        http_get: Callable[[str, dict[str, Any] | None], Any] | None = None,
        offline: bool = False,
        timeout: float = 20.0,
    ) -> None:
        self.cache_dir = (Path(cache_dir) / "nse") if cache_dir else None
        self.clock = clock or SystemClock()
        self._session_factory = session_factory
        self._http_get = http_get
        self.offline = offline
        self.timeout = timeout
        self._session: Any = None
        self._lock = threading.RLock()
        self.last_error: str | None = None

    # ------------------------------------------------------------------ http + cache
    def _default_get(self, url: str, params: dict[str, Any] | None) -> Any:
        import requests

        with self._lock:
            if self._session is None:
                sess = requests.Session()
                sess.headers.update(_HEADERS)
                try:  # cookie warm-up; NSE often answers 403 here but still sets cookies
                    sess.get(NSE_BASE, timeout=self.timeout)
                except requests.RequestException as exc:
                    logger.debug(f"[nse] warm-up request failed: {exc}")
                self._session = sess
        return self._session.get(url, params=params, timeout=self.timeout)

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        return self.cache_dir / f"{safe}.json"

    def _read_cache(self, key: str) -> tuple[datetime | None, Any]:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None, None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            return datetime.fromisoformat(blob["fetched_at"]), blob["data"]
        except (OSError, ValueError, KeyError) as exc:
            logger.warning(f"[nse] cache unreadable {path}: {exc}")
            return None, None

    def _write_cache(self, key: str, data: Any) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"fetched_at": self.clock.now().isoformat(), "data": data}), encoding="utf-8")
        except (OSError, TypeError) as exc:
            logger.warning(f"[nse] could not write cache {path}: {exc}")

    def _fetch(self, key: str, url: str, params: dict[str, Any] | None, ttl: timedelta, *, as_text: bool = False) -> Any:
        """Cached GET. Returns parsed JSON (or text) or ``None`` on failure without cache."""
        fetched_at, cached = self._read_cache(key)
        now = to_ist(self.clock.now())
        if cached is not None and fetched_at is not None and now - to_ist(fetched_at) < ttl:
            return cached
        if self.offline:
            if cached is None:
                logger.debug(f"[nse] offline and no cache for {key}")
            return cached
        try:
            resp = (self._http_get or self._default_get)(url, params)
            status = getattr(resp, "status_code", 200)
            if status != 200:
                raise RuntimeError(f"HTTP {status}")
            data = resp.content.decode("utf-8-sig", errors="replace") if as_text else resp.json()
        except Exception as exc:  # noqa: BLE001 - NSE is flaky; degrade to cache / empty
            self._failure(f"{key}: {exc}", {"url": url, "params": params})
            return cached
        self._write_cache(key, data)
        return data

    def _failure(self, message: str, context: dict[str, Any]) -> None:
        self.last_error = message
        logger.warning(f"[nse] {message}")
        if self._session_factory is None:
            return
        session = None
        try:
            from storage.audit import record_error

            session = self._session_factory()
            record_error(session, "nse", message, context, severity="WARNING", clock=self.clock)
            session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[nse] could not record error: {exc}")
        finally:
            if session is not None:
                session.close()

    # ------------------------------------------------------------------ API
    @_failsafe(dict)
    def holiday_map(self) -> dict[date, str]:
        """``{date: description}`` of NSE equity (CM) trading holidays as published."""
        data = self._fetch("holidays_trading", f"{NSE_BASE}/api/holiday-master", {"type": "trading"}, _TTL_WEEK)
        try:
            return parse_holiday_master(data) if data else {}
        except Exception as exc:  # noqa: BLE001
            self._failure(f"holiday parse failed: {exc}", {})
            return {}

    @_failsafe(set)
    def holidays(self, year: int) -> set[date]:
        """NSE equity trading holidays of ``year`` (empty on failure)."""
        return {d for d in self.holiday_map() if d.year == year}

    @_failsafe(set)
    def asm_symbols(self) -> set[str]:
        """Symbols currently under (long- or short-term) Additional Surveillance Measure."""
        today = self.clock.now().date()
        data = self._fetch(f"asm_{today:%Y%m%d}", f"{NSE_BASE}/api/reportASM", None, _TTL_DAY)
        out: set[str] = set()
        if isinstance(data, dict):
            for part in data.values():
                rows = part.get("data", []) if isinstance(part, dict) else part if isinstance(part, list) else []
                out |= {str(r.get("symbol", "")).strip().upper() for r in rows if isinstance(r, dict) and r.get("symbol")}
        return out

    @_failsafe(set)
    def gsm_symbols(self) -> set[str]:
        """Symbols currently under Graded Surveillance Measure."""
        today = self.clock.now().date()
        data = self._fetch(f"gsm_{today:%Y%m%d}", f"{NSE_BASE}/api/reportGSM", None, _TTL_DAY)
        rows = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        return {str(r.get("symbol", "")).strip().upper() for r in rows if isinstance(r, dict) and r.get("symbol")}

    @_failsafe(dict)
    def fii_dii(self) -> dict[str, Any]:
        """Latest provisional FII/DII cash-market flows in Rs crore (empty dict on failure).

        Keys: ``date``, ``fii_buy_cr``, ``fii_sell_cr``, ``fii_net_cr``, ``dii_buy_cr``,
        ``dii_sell_cr``, ``dii_net_cr``.
        """
        today = self.clock.now().date()
        data = self._fetch(f"fiidii_{today:%Y%m%d}", f"{NSE_BASE}/api/fiidiiTradeReact", None, _TTL_SHORT)
        out: dict[str, Any] = {}
        for row in data if isinstance(data, list) else []:
            if not isinstance(row, dict):
                continue
            cat = str(row.get("category", "")).upper()
            prefix = "fii" if cat.startswith("FII") else "dii" if cat.startswith("DII") else None
            if prefix is None:
                continue
            out[f"{prefix}_buy_cr"] = _f(row.get("buyValue"))
            out[f"{prefix}_sell_cr"] = _f(row.get("sellValue"))
            out[f"{prefix}_net_cr"] = _f(row.get("netValue"))
            d = parse_nse_date(row.get("date"))
            if d is not None:
                out["date"] = d.isoformat()
        return out

    @_failsafe(list)
    def corporate_actions(self, start: date, end: date) -> list[MarketEvent]:
        """Ex-dates (dividends with amount, splits/bonus with ratio, rights, AGM ...) in ``[start, end]``."""
        params = {"index": "equities", "from_date": f"{start:%d-%m-%Y}", "to_date": f"{end:%d-%m-%Y}"}
        data = self._fetch(f"ca_{start:%Y%m%d}_{end:%Y%m%d}", f"{NSE_BASE}/api/corporates-corporateActions",
                           params, _TTL_DAY)
        try:
            events = parse_corporate_actions(data if isinstance(data, list) else [])
        except Exception as exc:  # noqa: BLE001
            self._failure(f"corporate actions parse failed: {exc}", {})
            return []
        return [e for e in events if start <= e.event_date <= end]

    @_failsafe(list)
    def event_calendar(self, start: date, end: date) -> list[MarketEvent]:
        """Board meetings / results dates in ``[start, end]``."""
        params = {"index": "equities", "from_date": f"{start:%d-%m-%Y}", "to_date": f"{end:%d-%m-%Y}"}
        data = self._fetch(f"events_{start:%Y%m%d}_{end:%Y%m%d}", f"{NSE_BASE}/api/event-calendar", params, _TTL_DAY)
        try:
            events = parse_event_calendar(data if isinstance(data, list) else [])
        except Exception as exc:  # noqa: BLE001
            self._failure(f"event calendar parse failed: {exc}", {})
            return []
        return [e for e in events if start <= e.event_date <= end]

    @_failsafe(list)
    def announcements(self, symbol: str | None, since: datetime) -> list[Headline]:
        """Corporate announcements since ``since`` (up to ``clock.now()``), optionally for one symbol."""
        now = to_ist(self.clock.now())
        since = to_ist(since)
        params: dict[str, Any] = {"index": "equities", "from_date": f"{since:%d-%m-%Y}", "to_date": f"{now:%d-%m-%Y}"}
        key = f"ann_{since:%Y%m%d}_{now:%Y%m%d}"
        if symbol:
            params["symbol"] = symbol.upper()
            key += f"_{symbol.upper()}"
        data = self._fetch(key, f"{NSE_BASE}/api/corporate-announcements", params, timedelta(minutes=15))
        try:
            return parse_announcements(data if isinstance(data, list) else [], since, now, symbol)
        except Exception as exc:  # noqa: BLE001
            self._failure(f"announcements parse failed: {exc}", {})
            return []

    @_failsafe(dict)
    def delivery_map(self, day: date) -> dict[str, float]:
        """``{SYMBOL: delivery %}`` from the full bhavcopy of ``day`` (empty if unavailable)."""
        key = f"delivery_{day:%Y%m%d}"
        fetched_at, cached = self._read_cache(key)
        if cached is not None:
            return {k: float(v) for k, v in cached.items()}
        text = self._fetch(f"bhav_{day:%Y%m%d}", f"{NSE_ARCHIVES}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv",
                           None, _TTL_WEEK, as_text=True)
        if not text:
            return {}
        try:
            parsed = parse_bhavcopy_delivery(text)
        except Exception as exc:  # noqa: BLE001
            self._failure(f"bhavcopy parse failed for {day}: {exc}", {})
            return {}
        if parsed:
            self._write_cache(key, parsed)
            path = self._cache_path(f"bhav_{day:%Y%m%d}")
            if path is not None:  # the raw CSV is large; keep only the parsed map
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.debug(f"[nse] could not remove raw bhavcopy cache: {exc}")
        return parsed

    @_failsafe(lambda: None)
    def delivery_pct(self, symbol: str, day: date) -> float | None:
        """Delivery % of ``symbol`` on ``day`` (bhavcopy), None if unknown."""
        return self.delivery_map(day).get(symbol.upper())

    @_failsafe(lambda: None)
    def advance_decline(self) -> float | None:
        """Advance/decline ratio of NIFTY 500 (fallback: market-wide / NIFTY 50); None on failure."""
        data = self._fetch(f"allindices_{self.clock.now():%Y%m%d%H}", f"{NSE_BASE}/api/allIndices", None,
                           timedelta(minutes=10))
        if not isinstance(data, dict):
            return None
        rows = {str(r.get("index", "")).upper(): r for r in data.get("data", []) if isinstance(r, dict)}
        for source in (rows.get("NIFTY 500"), data, rows.get("NIFTY 50")):
            if not source:
                continue
            adv, dec = _f(source.get("advances")), _f(source.get("declines"))
            if adv is None or dec is None:
                continue
            if dec == 0:
                return float(adv) if adv > 0 else None
            return round(adv / dec, 4)
        return None

    @_failsafe(lambda: None)
    def pledge_pct(self, symbol: str) -> float | None:
        """Promoter shares pledged/encumbered (% of promoter holding), best effort."""
        today = self.clock.now().date()
        params = {"index": "equities", "symbol": symbol.upper()}
        data = self._fetch(f"pledge_{symbol.upper()}_{today:%Y%m%d}", f"{NSE_BASE}/api/corporate-pledgedata",
                           params, _TTL_WEEK)
        try:
            return parse_pledge(data)
        except Exception as exc:  # noqa: BLE001
            self._failure(f"pledge parse failed for {symbol}: {exc}", {})
            return None


def fetch_nse_holidays(cache_dir: Path | None = None) -> dict[date, str]:
    """Fetch NSE's published equity trading holidays (used by ``load_calendar(fetch=True)``)."""
    client = NSEClient(cache_dir or Path(__file__).resolve().parent.parent / "runtime" / "cache")
    return client.holiday_map()
