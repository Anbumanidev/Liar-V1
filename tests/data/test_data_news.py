"""NewsService: feed parsing, symbol matching, time window, de-duplication, caching and failure paths."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from core.clock import IST, SimulatedClock
from core.types import Headline
from data.news import (
    DEFAULT_FEEDS,
    NewsService,
    SymbolMatcher,
    clean_company_name,
    company_aliases,
    parse_entry_time,
    parse_feed_entries,
)
from data.universe import UniverseService
from tests.helpers import ist

NOW = ist(2026, 11, 23, 12, 0)
SINCE = ist(2026, 11, 22, 0, 0)

MC = [
    {"title": "Reliance Industries shares rise 3% on Jio tariff hike", "link": "https://mc/1",
     "published": "Mon, 23 Nov 2026 10:00:00 +0530", "summary": "<p>RIL gained <b>3%</b> &amp; more</p>"},
    {"title": "TCS bags $1 billion deal from a European bank", "link": "https://mc/2",
     "published_parsed": time.strptime("2026-11-23 04:00:00", "%Y-%m-%d %H:%M:%S")},          # 09:30 IST
    {"title": "Maruti Suzuki November sales jump 12%", "link": "https://mc/3",
     "published": "2026-11-23T08:00:00+05:30"},
    {"title": "SBI raises lending rates by 10 bps", "link": "https://mc/4",
     "published": "Sun, 22 Nov 2026 18:00:00 +0530"},
    {"title": "itc hotels demerger: what investors should know", "link": "https://mc/5",
     "published": "Sun, 22 Nov 2026 19:00:00 +0530"},
    {"title": "Old news about Infosys", "link": "https://mc/6", "published": "Mon, 16 Nov 2026 10:00:00 +0530"},
    {"title": "Infosys guidance (published in the future)", "link": "https://mc/7",
     "published": "Mon, 23 Nov 2026 13:00:00 +0530"},
    {"title": "Undated story about Infosys", "link": "https://mc/8"},
    {"title": "Infosys wins large deal", "link": "https://mc/9", "published": "Mon, 23 Nov 2026 11:00:00 +0530"},
]
ET = [
    {"title": "Reliance Industries shares rise 3% on Jio tariff hike", "link": "https://et/1",
     "published": "Mon, 23 Nov 2026 10:05:00 +0530"},                                          # duplicate story
    {"title": "Sensex, Nifty end flat ahead of F&O expiry", "link": "https://et/2",
     "published": "Mon, 23 Nov 2026 11:30:00 +0530", "description": "Market wrap"},
]
FEEDS = {"mc": "https://mc/rss", "et": "https://et/rss"}


class FakeFetcher:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.calls: list[str] = []

    def __call__(self, url: str) -> Any:
        self.calls.append(url)
        value = self.data[url]
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture(scope="module")
def universe() -> UniverseService:
    return UniverseService()


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def fetcher() -> FakeFetcher:
    return FakeFetcher({"https://mc/rss": {"entries": MC}, "https://et/rss": ET})


@pytest.fixture
def news(clock: SimulatedClock, universe: UniverseService, fetcher: FakeFetcher) -> NewsService:
    return NewsService(clock, universe=universe, feeds=FEEDS, fetcher=fetcher)


def _titles(hs: list[Headline]) -> list[str]:
    return [h.title for h in hs]


# ------------------------------------------------------------------ matching + window


def test_symbol_matching_by_company_name_ticker_and_alias(news: NewsService) -> None:
    rel = news.headlines("reliance", SINCE)
    assert _titles(rel) == ["Reliance Industries shares rise 3% on Jio tariff hike"]          # de-duplicated
    assert rel[0].symbol == "RELIANCE" and rel[0].summary == "RIL gained 3% & more"
    assert rel[0].published_at == ist(2026, 11, 23, 10, 5) and rel[0].source == "et"            # newest copy kept
    assert _titles(news.headlines("TCS", SINCE)) == ["TCS bags $1 billion deal from a European bank"]
    assert _titles(news.headlines("MARUTI", SINCE)) == ["Maruti Suzuki November sales jump 12%"]
    assert _titles(news.headlines("SBIN", SINCE)) == ["SBI raises lending rates by 10 bps"]
    assert news.headlines("ITC", SINCE) == []                       # all-caps names match case-sensitively
    assert news.headlines("HDFCBANK", SINCE) == []


def test_time_window_excludes_old_future_and_undated(news: NewsService) -> None:
    infy = news.headlines("INFY", SINCE)
    assert _titles(infy) == ["Infosys wins large deal"]
    assert all(SINCE <= h.published_at <= NOW for h in infy)
    assert _titles(news.headlines("INFY", ist(2026, 11, 1))) == ["Infosys wins large deal", "Old news about Infosys"]


def test_market_headlines_newest_first(news: NewsService) -> None:
    hs = news.headlines(None, SINCE)
    assert all(h.symbol is None for h in hs)
    times = [h.published_at for h in hs]
    assert times == sorted(times, reverse=True)
    assert len(hs) == 7                       # 9 MC + 2 ET, minus old / future / undated / duplicate
    assert hs[0].title.startswith("Sensex") and hs[0].summary == "Market wrap"
    assert all(h.published_at.tzinfo is not None and h.published_at.utcoffset() == timedelta(hours=5, minutes=30)
               for h in hs)


def test_nse_announcements_are_merged_for_symbol_queries(clock: SimulatedClock, universe: UniverseService,
                                                         fetcher: FakeFetcher) -> None:
    class FakeNSE:
        def __init__(self) -> None:
            self.calls: list[tuple[str | None, datetime]] = []

        def announcements(self, symbol: str | None, since: datetime) -> list[Headline]:
            self.calls.append((symbol, since))
            return [Headline(source="NSE", title="TCS: Outcome of Board Meeting", symbol="TCS",
                             published_at=ist(2026, 11, 23, 11, 45))]

    nse = FakeNSE()
    svc = NewsService(clock, universe=universe, feeds=FEEDS, fetcher=fetcher, nse=nse)  # type: ignore[arg-type]
    hs = svc.headlines("TCS", SINCE)
    assert _titles(hs) == ["TCS: Outcome of Board Meeting", "TCS bags $1 billion deal from a European bank"]
    assert nse.calls == [("TCS", SINCE)]
    svc.headlines(None, SINCE)
    assert len(nse.calls) == 1                # market-wide queries do not pull every NSE filing


# ------------------------------------------------------------------ caching + failures


def test_feeds_cached_for_ttl(news: NewsService, fetcher: FakeFetcher, clock: SimulatedClock) -> None:
    news.headlines("TCS", SINCE)
    news.headlines("INFY", SINCE)
    assert len(fetcher.calls) == 2                                   # one fetch per feed
    clock.advance(timedelta(minutes=9))
    news.headlines(None, SINCE)
    assert len(fetcher.calls) == 2
    clock.advance(timedelta(minutes=2))                              # default TTL 10 min expired
    news.headlines(None, SINCE)
    assert len(fetcher.calls) == 4


def test_feed_failures_degrade_gracefully(clock: SimulatedClock, universe: UniverseService) -> None:
    fetcher = FakeFetcher({"https://mc/rss": {"entries": MC}, "https://et/rss": ConnectionError("timeout")})
    svc = NewsService(clock, universe=universe, feeds=FEEDS, fetcher=fetcher)
    assert _titles(svc.headlines("TCS", SINCE)) == ["TCS bags $1 billion deal from a European bank"]
    assert svc.last_error is not None and "et" in svc.last_error
    fetcher.data["https://mc/rss"] = RuntimeError("feed moved")
    clock.advance(timedelta(minutes=11))
    assert _titles(svc.headlines("TCS", SINCE)) == ["TCS bags $1 billion deal from a European bank"]   # stale
    dead = NewsService(clock, universe=universe, feeds=FEEDS,
                       fetcher=FakeFetcher({u: OSError("offline") for u in FEEDS.values()}))
    assert dead.headlines("TCS", SINCE) == [] and dead.headlines(None, SINCE) == []


def test_never_raises(universe: UniverseService, fetcher: FakeFetcher) -> None:
    class BrokenClock:
        def now(self) -> datetime:
            raise RuntimeError("clock broken")

    assert NewsService(BrokenClock(), feeds=FEEDS, fetcher=fetcher).headlines("TCS", SINCE) == []

    class BrokenUniverse:
        def company_name(self, symbol: str) -> str:
            raise KeyError(symbol)

    svc = NewsService(SimulatedClock(NOW), universe=BrokenUniverse(), feeds=FEEDS, fetcher=fetcher)  # type: ignore[arg-type]
    assert _titles(svc.headlines("TCS", SINCE)) == ["TCS bags $1 billion deal from a European bank"]  # ticker only


def test_disk_archive_survives_restart_and_accumulates(clock: SimulatedClock, universe: UniverseService,
                                                       tmp_path: Path) -> None:
    fetcher = FakeFetcher({"https://mc/rss": {"entries": MC}, "https://et/rss": ET})
    NewsService(clock, tmp_path, universe, FEEDS, fetcher).headlines("TCS", SINCE)
    assert (tmp_path / "news" / "mc.json").exists() and (tmp_path / "news" / "et.json").exists()
    cold = FakeFetcher({"https://mc/rss": {"entries": MC}, "https://et/rss": ET})
    restarted = NewsService(clock, tmp_path, universe, FEEDS, cold)
    assert _titles(restarted.headlines("TCS", SINCE)) == ["TCS bags $1 billion deal from a European bank"]
    assert cold.calls == []                                                   # fresh disk cache reused
    clock.advance(timedelta(minutes=30))
    cold.data["https://mc/rss"] = {"entries": [{"title": "HDFC Bank hits record high", "link": "https://mc/10",
                                                "published": "Mon, 23 Nov 2026 12:20:00 +0530"}]}
    assert _titles(restarted.headlines("TCS", SINCE)) == ["TCS bags $1 billion deal from a European bank"]
    assert _titles(restarted.headlines("HDFCBANK", SINCE)) == ["HDFC Bank hits record high"]
    clock.advance(timedelta(days=12))                                         # beyond retention (10 days)
    cold.data["https://mc/rss"] = {"entries": []}
    assert restarted.headlines("TCS", SINCE) == []


def test_real_feedparser_document_offline(clock: SimulatedClock, universe: UniverseService) -> None:
    feedparser = pytest.importorskip("feedparser")
    rss = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
      <item><title>Infosys &amp; TCS lead IT rally</title><link>https://x/1</link>
        <pubDate>Mon, 23 Nov 2026 05:30:00 GMT</pubDate><description>&lt;b&gt;IT up&lt;/b&gt;</description></item>
      <item><title>No date</title><link>https://x/2</link></item>
    </channel></rss>"""
    svc = NewsService(clock, universe=universe, feeds={"x": "https://x/rss"},
                      fetcher=lambda url: feedparser.parse(rss))
    hs = svc.headlines("INFY", SINCE)
    assert _titles(hs) == ["Infosys & TCS lead IT rally"]
    assert hs[0].published_at == ist(2026, 11, 23, 11, 0) and hs[0].summary == "IT up"
    assert _titles(svc.headlines("TCS", SINCE)) == ["Infosys & TCS lead IT rally"]


# ------------------------------------------------------------------ helpers


def test_company_name_aliases() -> None:
    assert clean_company_name("Reliance Industries Ltd.") == "Reliance Industries"
    assert clean_company_name("Foo Pvt. Ltd.") == "Foo"
    assert clean_company_name("State Bank of India") == "State Bank of India"
    assert company_aliases("Maruti Suzuki India Ltd.") == ["Maruti Suzuki India", "Maruti Suzuki"]
    assert company_aliases("Titan Company Ltd.") == ["Titan Company", "Titan"]
    assert company_aliases("Coal India Ltd.") == ["Coal India"]                 # "Coal" alone is too generic
    assert company_aliases("Adani Enterprises Ltd.") == ["Adani Enterprises"]   # "Adani" alone is ambiguous
    assert company_aliases("Life Insurance Corporation of India") == ["Life Insurance Corporation of India"]
    assert company_aliases("Power Grid Corporation of India Ltd.")[-1] == "Power Grid"
    assert company_aliases(None) == [] and company_aliases("") == []


def test_symbol_matcher_rules() -> None:
    m = SymbolMatcher.build("TCS", "Tata Consultancy Services Ltd.")
    assert m.matches("TCS Q2 profit beats estimates")
    assert not m.matches("tcs q2")                                    # acronyms are case-sensitive
    assert m.matches("tata consultancy services to hire 40,000")       # names are case-insensitive
    assert not m.matches("TCSL shares")                                # whole words only
    mm = SymbolMatcher.build("M&M", "Mahindra & Mahindra Ltd.")
    assert mm.matches("M&M SUV sales rise") and mm.matches("mahindra & mahindra results")
    lt = SymbolMatcher.build("LT", "Larsen & Toubro Ltd.", ["L&T"])
    assert lt.matches("L&T wins order") and not lt.matches("LT Foods rallies")   # 2-letter ticker not used
    dr = SymbolMatcher.build("DRREDDY", None, ["Dr. Reddy"])
    assert dr.matches("Dr. Reddy's recalls drug")


def test_parse_entry_time_variants() -> None:
    assert parse_entry_time({"published": "Mon, 23 Nov 2026 05:30:00 GMT"}) == ist(2026, 11, 23, 11, 0)
    assert parse_entry_time({"published": "2026-11-23T06:00:00Z"}) == ist(2026, 11, 23, 11, 30)
    assert parse_entry_time({"updated": "2026-11-23 09:00:00"}) == ist(2026, 11, 23, 9, 0)        # naive = IST
    st = time.strptime("2026-11-23 00:00:00", "%Y-%m-%d %H:%M:%S")
    assert parse_entry_time({"updated_parsed": st}) == datetime(2026, 11, 23, 5, 30, tzinfo=IST)
    assert parse_entry_time({"published": "not a date"}) is None
    assert parse_entry_time({}) is None
    parsed = parse_feed_entries("s", [{"title": "  "}, {"title": "ok", "published": "2026-11-23T09:00:00+05:30"},
                                      "garbage"])
    assert [h.title for h in parsed] == ["ok"]


def test_default_feeds_configured() -> None:
    assert set(DEFAULT_FEEDS) == {"moneycontrol", "et_markets", "business_standard", "livemint"}
    assert all(url.startswith("https://") for url in DEFAULT_FEEDS.values())


@pytest.mark.network
def test_live_feeds_smoke() -> None:                                  # RUN_NETWORK_TESTS=1 only
    from core.clock import SystemClock

    svc = NewsService(SystemClock(), universe=UniverseService())
    hs = svc.headlines(None, SystemClock().now() - timedelta(days=2))
    assert isinstance(hs, list)
