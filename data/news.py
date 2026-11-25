"""News headlines from Indian financial RSS feeds (+ NSE announcements), matched to symbols.

* Default feeds (``DEFAULT_FEEDS``): Moneycontrol market reports, ET Markets, Business Standard
  markets and Livemint markets, parsed with ``feedparser`` (fetched via ``requests`` so the
  process-wide CA bundle from ``core.net.configure_ssl`` and a timeout apply).
* Symbol matching (``headline_matches``): the ticker (case-sensitive whole word, >= 3 chars, e.g.
  ``TCS``, ``M&M``), the company name from ``UniverseService.company_name`` with legal suffixes
  removed (``Reliance Industries Ltd.`` -> ``Reliance Industries``) plus shortened variants
  (``Maruti Suzuki India`` -> ``Maruti Suzuki``) and a small table of common headline names
  (``SBI``, ``L&T``, ``HUL`` ...). Matching is a heuristic: false positives are possible.
* Caching: every feed is cached in memory for ``ttl_seconds`` (measured on the injected clock).
  With ``cache_dir`` the items are also kept in ``<cache_dir>/news/<feed>.json`` and ACCUMULATED
  for ``retention_days`` - RSS feeds only carry the latest few dozen items, so the archive is what
  lets the 7-day window of the longer modes fill up. A failed fetch serves the stale items.
* Never raises: failures are logged and degrade to stale/empty results. Headlines published
  after ``clock.now()`` are never returned (no look-ahead).
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.clock import IST, Clock, to_ist
from core.types import Headline

if TYPE_CHECKING:  # pragma: no cover
    from data.nse import NSEClient
    from data.universe import UniverseService

#: feed name -> RSS URL (verify: publishers occasionally move their feeds).
DEFAULT_FEEDS: dict[str, str] = {
    "moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "livemint": "https://www.livemint.com/rss/markets",
}

#: Names headlines commonly use that the official company name does not contain.
SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "RELIANCE": ("RIL", "Reliance"),
    "SBIN": ("SBI", "State Bank"),
    "LT": ("L&T", "Larsen"),
    "HINDUNILVR": ("HUL",),
    "BHARTIARTL": ("Airtel",),
    "SUNPHARMA": ("Sun Pharma",),
    "DRREDDY": ("Dr Reddy", "Dr. Reddy"),
    "MARUTI": ("Maruti",),
    "ULTRACEMCO": ("UltraTech",),
    "HCLTECH": ("HCL Tech", "HCLTech"),
    "IOC": ("IOCL", "Indian Oil"),
    "KOTAKBANK": ("Kotak Bank", "Kotak Mahindra"),
    "INDUSINDBK": ("IndusInd",),
    "EICHERMOT": ("Eicher", "Royal Enfield"),
    "TCS": ("Tata Consultancy",),
    "LICI": ("LIC",),
    "BAJFINANCE": ("Bajaj Finance",),
    "POWERGRID": ("Power Grid",),
    "HAL": ("Hindustan Aeronautics",),
    "BEL": ("Bharat Electronics",),
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

#: Legal-form suffixes stripped from company names.
_LEGAL_SUFFIX = re.compile(r"[\s,]*\b(ltd|limited|pvt|private)\b\.?\s*$", re.IGNORECASE)
#: Trailing words dropped to build shorter variants ("Maruti Suzuki India" -> "Maruti Suzuki").
_TRAILING_GENERIC = ("of india", "(india)", "india", "industries", "company", "corporation", "corp", "co.",
                     "enterprises")
#: A shortened variant made only of these words is too generic to identify a company.
_GENERIC_WORDS = frozenset({
    "india", "indian", "life", "insurance", "bank", "power", "steel", "oil", "gas", "coal", "general",
    "state", "national", "corporation", "company", "industries", "finance", "financial", "services",
    "motors", "auto", "cement", "pharma", "chemicals", "energy", "infra", "infrastructure", "housing",
    "holdings", "the", "and", "of", "&", "global", "international", "consumer", "products", "capital",
    "securities", "bharat", "hindustan", "united", "new", "sun", "max",
})
#: Business-group names: a single-word variant equal to one of these is ambiguous.
_GROUP_WORDS = frozenset({"adani", "tata", "bajaj", "hdfc", "icici", "jsw", "mahindra", "birla", "aditya",
                          "godrej", "reliance", "jindal", "kotak", "sbi", "hero", "torrent", "muthoot"})
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# =========================================================================== helpers


def clean_company_name(name: str) -> str:
    """``"Reliance Industries Ltd."`` -> ``"Reliance Industries"`` (legal suffixes removed)."""
    out = (name or "").strip()
    while True:
        new = _LEGAL_SUFFIX.sub("", out).strip().rstrip(",.").strip()
        if new == out:
            return out
        out = new


def company_aliases(name: str | None) -> list[str]:
    """Name variants used to find a company in headlines (most specific first).

    The cleaned full name is always included; shortened variants (trailing generic words such
    as ``India`` / ``Industries`` removed) are kept only if they are still distinctive.
    """
    if not name:
        return []
    full = clean_company_name(name)
    if not full:
        return []
    out = [full]
    current = full
    while True:
        low = current.lower()
        cut = next((g for g in _TRAILING_GENERIC if low.endswith(" " + g)), None)
        if cut is None:
            break
        current = current[: len(current) - len(cut) - 1].strip().rstrip(",&").strip()
        words = [w.lower() for w in current.split()]
        if not words:
            break
        if all(w in _GENERIC_WORDS for w in words):
            continue
        if len(words) == 1 and (words[0] in _GROUP_WORDS or len(words[0]) < 4):
            continue
        out.append(current)
    return list(dict.fromkeys(out))


def _alias_pattern(aliases: Iterable[str], flags: int) -> re.Pattern[str] | None:
    alts = sorted({a for a in aliases if a}, key=len, reverse=True)
    if not alts:
        return None
    body = "|".join(re.escape(a) for a in alts)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", flags)


@dataclass
class SymbolMatcher:
    """Compiled matcher for one symbol: case-sensitive acronyms + case-insensitive names."""

    symbol: str
    aliases: list[str] = field(default_factory=list)
    _exact: re.Pattern[str] | None = None
    _fuzzy: re.Pattern[str] | None = None

    @classmethod
    def build(cls, symbol: str, company: str | None, extra: Iterable[str] = ()) -> "SymbolMatcher":
        """Aliases = ticker (if >= 3 chars) + company-name variants + ``extra`` names."""
        sym = symbol.strip().upper()
        aliases: list[str] = []
        if len(sym.replace("&", "").replace("-", "")) >= 3 or "&" in sym:
            aliases.append(sym)
        aliases += company_aliases(company)
        aliases += [a.strip() for a in extra if a and a.strip()]
        aliases = list(dict.fromkeys(aliases))
        # all-caps tokens (tickers, acronyms) must match case-sensitively: "ITC" but not "itc".
        exact = [a for a in aliases if a.upper() == a and any(ch.isalpha() for ch in a)]
        fuzzy = [a for a in aliases if a not in exact]
        return cls(sym, aliases, _alias_pattern(exact, 0), _alias_pattern(fuzzy, re.IGNORECASE))

    def matches(self, text: str) -> bool:
        """True if any alias occurs in ``text`` as a whole word/phrase."""
        if not text:
            return False
        return bool((self._exact and self._exact.search(text)) or (self._fuzzy and self._fuzzy.search(text)))


def headline_matches(headline: Headline, matcher: SymbolMatcher) -> bool:
    """True if the headline's title or summary mentions the matcher's company."""
    return matcher.matches(headline.title) or matcher.matches(headline.summary or "")


def _strip_html(text: str) -> str:
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text or ""))).strip()


def _entry_get(entry: Any, *keys: str) -> Any:
    for key in keys:
        value = entry.get(key) if hasattr(entry, "get") else getattr(entry, key, None)
        if value:
            return value
    return None


def parse_entry_time(entry: Any) -> datetime | None:
    """Publication time of a feed entry as IST (``None`` if the entry carries no usable date).

    Prefers feedparser's UTC ``published_parsed`` / ``updated_parsed`` tuples; falls back to
    RFC-822 / ISO strings (naive strings are taken as IST).
    """
    parsed = _entry_get(entry, "published_parsed", "updated_parsed")
    if parsed:
        try:
            return datetime(*tuple(parsed)[:6], tzinfo=timezone.utc).astimezone(IST)
        except (TypeError, ValueError) as exc:
            logger.debug(f"[news] bad parsed date {parsed!r}: {exc}")
    raw = _entry_get(entry, "published", "updated", "pubDate", "date")
    if isinstance(raw, datetime):
        return to_ist(raw)
    if not raw:
        return None
    text = str(raw).strip()
    try:
        return to_ist(parsedate_to_datetime(text))
    except (TypeError, ValueError, IndexError):
        pass
    try:
        return to_ist(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        logger.debug(f"[news] unparseable entry date {text!r}")
        return None


def parse_feed_entries(source: str, entries: Iterable[Any]) -> list[Headline]:
    """Feed entries (feedparser dicts or plain dicts) -> ``Headline`` list; undated entries skipped."""
    out: list[Headline] = []
    for entry in entries or []:
        try:
            title = _strip_html(str(_entry_get(entry, "title") or ""))
            published = parse_entry_time(entry)
            if not title or published is None:
                continue
            summary = _strip_html(str(_entry_get(entry, "summary", "description") or ""))[:1000] or None
            link = _entry_get(entry, "link", "id")
            out.append(Headline(source=source, title=title[:500], url=str(link) if link else None,
                                published_at=published, summary=summary))
        except Exception as exc:  # noqa: BLE001 - one malformed entry must not drop the feed
            logger.debug(f"[news] skipping malformed entry from {source}: {exc}")
    return out


def _entries_of(result: Any) -> list[Any]:
    """Entries from a feedparser result, a ``{"entries": [...]}`` dict or a plain list."""
    if result is None:
        return []
    if isinstance(result, list):
        return result
    entries = result.get("entries") if hasattr(result, "get") else None
    if entries is None:
        entries = getattr(result, "entries", None)
    return list(entries or [])


def default_fetcher(url: str, timeout: float = 15.0) -> Any:
    """Download ``url`` with ``requests`` and parse it with ``feedparser``."""
    import feedparser
    import requests

    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", None):
        raise ValueError(f"unparseable feed: {getattr(parsed, 'bozo_exception', 'bozo')}")
    return parsed


def _dedupe_key(h: Headline) -> str:
    return re.sub(r"[^a-z0-9]+", " ", h.title.lower()).strip()


@dataclass
class _FeedCache:
    fetched_at: datetime
    items: list[Headline]


# =========================================================================== service


class NewsService:
    """Headlines for a symbol (or the whole market) since a given time.

    Args:
        clock: time source (cache TTL, no-look-ahead cut-off).
        cache_dir: cache root (``runtime/cache``); feed archives go to ``<cache_dir>/news/``.
            ``None`` = in-memory cache only.
        universe: ``UniverseService`` for company names (ticker-only matching without it).
        feeds: ``{name: rss_url}`` (default ``DEFAULT_FEEDS``); the name becomes ``Headline.source``.
        fetcher: ``url -> feedparser result | {"entries": [...]} | [entry, ...]`` (tests inject
            fakes; default downloads with requests + feedparser).
        nse: optional ``NSEClient``; its corporate announcements are merged for symbol queries.
        ttl_seconds: in-memory freshness of a fetched feed (clock seconds).
        retention_days: how long archived items are kept on disk.
        aliases: extra ``{SYMBOL: [names]}`` merged into ``SYMBOL_ALIASES``.
    """

    def __init__(
        self,
        clock: Clock,
        cache_dir: Path | None = None,
        universe: "UniverseService | None" = None,
        feeds: dict[str, str] | None = None,
        fetcher: Callable[[str], Any] | None = None,
        *,
        nse: "NSEClient | None" = None,
        ttl_seconds: float = 600.0,
        retention_days: int = 10,
        aliases: dict[str, Iterable[str]] | None = None,
    ) -> None:
        self.clock = clock
        self.cache_dir = (Path(cache_dir) / "news") if cache_dir else None
        self.universe = universe
        self.feeds: dict[str, str] = dict(DEFAULT_FEEDS if feeds is None else feeds)
        self._fetcher = fetcher or default_fetcher
        self.nse = nse
        self.ttl = timedelta(seconds=ttl_seconds)
        self.retention = timedelta(days=retention_days)
        self._aliases: dict[str, tuple[str, ...]] = dict(SYMBOL_ALIASES)
        for sym, names in (aliases or {}).items():
            key = sym.strip().upper()
            self._aliases[key] = tuple(dict.fromkeys([*self._aliases.get(key, ()), *names]))
        self._cache: dict[str, _FeedCache] = {}
        self._matchers: dict[str, SymbolMatcher] = {}
        self._lock = threading.RLock()
        self.last_error: str | None = None

    # ------------------------------------------------------------------ public API
    def headlines(self, symbol: str | None, since: datetime) -> list[Headline]:
        """Headlines published in ``[since, clock.now()]``, newest first.

        Args:
            symbol: plain NSE ticker -> only headlines mentioning it (``Headline.symbol`` set);
                ``None`` -> every headline (market news, ``symbol`` left ``None``).
            since: start of the window (tz-aware; naive is taken as IST).

        Returns:
            De-duplicated headlines; an empty list on any failure (never raises).
        """
        try:
            now = to_ist(self.clock.now())
            since = to_ist(since)
            items: list[Headline] = []
            for name, url in self.feeds.items():
                items.extend(self._feed_items(name, url, now))
            sym = symbol.strip().upper() if symbol else None
            if sym and self.nse is not None:
                items.extend(self._nse_items(sym, since))
            window = [h for h in items if since <= h.published_at <= now]
            if sym:
                matcher = self.matcher(sym)
                window = [h if h.symbol == sym else h.model_copy(update={"symbol": sym})
                          for h in window if h.symbol == sym or (h.symbol is None and headline_matches(h, matcher))]
            window.sort(key=lambda h: h.published_at, reverse=True)
            seen: dict[str, int] = {}
            out: list[Headline] = []
            for h in window:
                key = _dedupe_key(h)
                if key in seen:
                    # keep the newest copy, but fill fields it lacks from older copies
                    i = seen[key]
                    if out[i].summary is None and h.summary:
                        out[i] = out[i].model_copy(update={"summary": h.summary})
                    continue
                seen[key] = len(out)
                out.append(h)
            return out
        except Exception as exc:  # noqa: BLE001 - news must never break the pipeline
            self.last_error = f"headlines({symbol}): {exc}"
            logger.warning(f"[news] headlines failed for {symbol or 'market'}: {exc}")
            return []

    def matcher(self, symbol: str) -> SymbolMatcher:
        """The (cached) ``SymbolMatcher`` of ``symbol``."""
        sym = symbol.strip().upper()
        with self._lock:
            m = self._matchers.get(sym)
            if m is None:
                company = None
                if self.universe is not None:
                    try:
                        company = self.universe.company_name(sym)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"[news] company name lookup failed for {sym}: {exc}")
                m = SymbolMatcher.build(sym, company, self._aliases.get(sym, ()))
                self._matchers[sym] = m
            return m

    def clear_cache(self) -> None:
        """Forget in-memory feed items and matchers (disk archives are kept)."""
        with self._lock:
            self._cache.clear()
            self._matchers.clear()

    # ------------------------------------------------------------------ feeds
    def _feed_items(self, name: str, url: str, now: datetime) -> list[Headline]:
        """Items of one feed: fresh cache, else fetch (+ merge into the archive), else stale."""
        with self._lock:
            entry = self._cache.get(name)
            if entry is None:
                entry = self._read_disk(name)
                if entry is not None:
                    self._cache[name] = entry
            if entry is not None and timedelta(0) <= now - entry.fetched_at < self.ttl:
                return entry.items
            try:
                fresh = parse_feed_entries(name, _entries_of(self._fetcher(url)))
            except Exception as exc:  # noqa: BLE001 - network / parse failure: serve stale items
                self.last_error = f"{name}: {exc}"
                logger.warning(f"[news] feed {name} failed ({exc}); serving "
                               f"{len(entry.items) if entry else 0} cached items")
                if entry is not None:  # back off for one TTL instead of hammering a dead feed
                    self._cache[name] = _FeedCache(now, entry.items)
                    return entry.items
                self._cache[name] = _FeedCache(now, [])
                return []
            merged: dict[str, Headline] = {}
            for h in [*(entry.items if entry else []), *fresh]:
                if now - h.published_at <= self.retention:
                    merged[h.url or _dedupe_key(h)] = h
            items = sorted(merged.values(), key=lambda h: h.published_at, reverse=True)
            new_entry = _FeedCache(now, items)
            self._cache[name] = new_entry
            self._write_disk(name, new_entry)
            return items

    def _nse_items(self, symbol: str, since: datetime) -> list[Headline]:
        try:
            return list(self.nse.announcements(symbol, since)) if self.nse is not None else []
        except Exception as exc:  # noqa: BLE001 - NSEClient is failure-safe; belt and braces
            logger.warning(f"[news] NSE announcements failed for {symbol}: {exc}")
            return []

    # ------------------------------------------------------------------ disk archive
    def _path(self, name: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.json"

    def _read_disk(self, name: str) -> _FeedCache | None:
        path = self._path(name)
        if path is None or not path.exists():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            items = [Headline.model_validate(x) for x in blob.get("items", [])]
            return _FeedCache(to_ist(datetime.fromisoformat(blob["fetched_at"])), items)
        except Exception as exc:  # noqa: BLE001 - corrupt archive: ignore it, refetch
            logger.warning(f"[news] cache unreadable {path}: {exc}")
            return None

    def _write_disk(self, name: str, entry: _FeedCache) -> None:
        path = self._path(name)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"fetched_at": entry.fetched_at.isoformat(),
                                       "items": [h.model_dump(mode="json") for h in entry.items]}),
                           encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning(f"[news] could not write cache {path}: {exc}")
