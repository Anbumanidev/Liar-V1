"""FundamentalsService: yfinance mapping, NSE overlay, daily cache, failure safety."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from core.clock import IST, SimulatedClock
from core.enums import EventType
from core.types import MarketEvent
from data.fundamentals import FundamentalsService

INFO = {"trailingPE": 25.0, "priceToBook": 4.0, "returnOnEquity": 0.18, "debtToEquity": 45.0,
        "revenueGrowth": 0.12, "earningsGrowth": 0.2, "profitMargins": 0.15, "heldPercentInsiders": 0.5,
        "sector": "Technology", "industry": "IT Services", "marketCap": 5e12}


class FakeNSE:
    def asm_symbols(self) -> set[str]:
        return {"BADCO"}

    def gsm_symbols(self) -> set[str]:
        return set()

    def pledge_pct(self, symbol: str) -> float | None:
        return 12.5 if symbol == "INFY" else None

    def event_calendar(self, start: date, end: date) -> list[MarketEvent]:
        return [MarketEvent(symbol="INFY", event_type=EventType.RESULTS, event_date=start)]

    def corporate_actions(self, start: date, end: date) -> list[MarketEvent]:
        return []


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(datetime(2026, 11, 25, 8, 30, tzinfo=IST))


def test_mapping_units(clock: SimulatedClock) -> None:
    svc = FundamentalsService(clock, info_fetcher=lambda s: dict(INFO))
    snap = svc.snapshot("infy")
    assert snap.symbol == "INFY" and snap.as_of == date(2026, 11, 25)
    assert snap.pe == 25.0 and snap.pb == 4.0
    assert snap.roe == pytest.approx(18.0) and snap.revenue_growth == pytest.approx(12.0)
    assert snap.debt_to_equity == pytest.approx(0.45)          # yfinance percent -> ratio
    assert snap.promoter_holding == pytest.approx(50.0)
    assert snap.sector == "Technology"


def test_nse_overlay(clock: SimulatedClock) -> None:
    svc = FundamentalsService(clock, nse=FakeNSE(), info_fetcher=lambda s: dict(INFO))  # type: ignore[arg-type]
    infy = svc.snapshot("INFY")
    assert infy.promoter_pledge_pct == 12.5 and infy.results_today is True and infy.in_asm is False
    assert svc.snapshot("BADCO").in_asm is True


def test_daily_cache_and_disk(tmp_path: Path, clock: SimulatedClock) -> None:
    calls: list[str] = []

    def fetch(sym: str) -> dict:  # type: ignore[type-arg]
        calls.append(sym)
        return dict(INFO)

    svc = FundamentalsService(clock, cache_dir=tmp_path, info_fetcher=fetch)
    svc.snapshot("INFY")
    svc.snapshot("INFY")
    assert calls == ["INFY"]
    assert (tmp_path / "fundamentals" / "2026-11-25.json").exists()
    fresh = FundamentalsService(clock, cache_dir=tmp_path, info_fetcher=fetch)   # reloads today's disk cache
    assert fresh.snapshot("INFY").pe == 25.0 and calls == ["INFY"]
    clock.advance(timedelta(days=1))
    svc.snapshot("INFY")
    assert calls == ["INFY", "INFY"]                              # new day -> refetch


def test_failure_never_raises(clock: SimulatedClock) -> None:
    def boom(sym: str) -> dict:  # type: ignore[type-arg]
        raise RuntimeError("yahoo down")

    snap = FundamentalsService(clock, info_fetcher=boom).snapshot("INFY")
    assert snap.symbol == "INFY" and snap.pe is None
    assert "error" in snap.extra or snap.extra.get("source") is not None
