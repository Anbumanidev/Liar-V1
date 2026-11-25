"""NSEClient: failure safety (never raises), caching/offline behaviour and the pure parsers."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from core.clock import IST, SimulatedClock
from core.enums import EventType
from data.nse import (
    NSEClient,
    parse_announcements,
    parse_bhavcopy_delivery,
    parse_ca_subject,
    parse_corporate_actions,
    parse_event_calendar,
    parse_holiday_master,
    parse_nse_date,
    parse_nse_datetime,
    parse_pledge,
)
from storage.models import ErrorLog
from tests.helpers import ist

D0, D1 = date(2026, 11, 1), date(2026, 11, 30)
SINCE = ist(2026, 11, 20)


class Resp:
    def __init__(self, payload: Any = None, status: int = 200, text: str | None = None) -> None:
        self.status_code = status
        self._payload = payload
        self.content = (text if text is not None else json.dumps(payload)).encode("utf-8")

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _all_calls(c: NSEClient) -> dict[str, Callable[[], Any]]:
    return {
        "holiday_map": c.holiday_map, "holidays": lambda: c.holidays(2026), "asm": c.asm_symbols,
        "gsm": c.gsm_symbols, "fii_dii": c.fii_dii, "corporate_actions": lambda: c.corporate_actions(D0, D1),
        "event_calendar": lambda: c.event_calendar(D0, D1), "announcements": lambda: c.announcements(None, SINCE),
        "announcements_sym": lambda: c.announcements("ABC", SINCE), "delivery_map": lambda: c.delivery_map(D1),
        "delivery_pct": lambda: c.delivery_pct("ABC", D1), "advance_decline": c.advance_decline,
        "pledge_pct": lambda: c.pledge_pct("ABC"),
    }


def _assert_all_empty(c: NSEClient) -> None:
    for name, call in _all_calls(c).items():
        out = call()
        assert out in (None, {}, set(), []), f"{name} returned {out!r}"


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(ist(2026, 11, 24, 8, 30))


# ------------------------------------------------------------------ failure safety


def test_network_failure_returns_empty_everywhere(clock: SimulatedClock,
                                                  session_factory: sessionmaker[Session]) -> None:
    def boom(url: str, params: dict[str, Any] | None) -> Any:
        raise ConnectionError("NSE unreachable")

    c = NSEClient(None, clock, http_get=boom, session_factory=session_factory)
    _assert_all_empty(c)
    assert c.last_error is not None and "NSE unreachable" in c.last_error
    with session_factory() as s:
        assert s.execute(select(func.count()).select_from(ErrorLog).where(ErrorLog.component == "nse")).scalar_one() > 0


def test_default_http_layer_failure_is_safe(monkeypatch: pytest.MonkeyPatch, clock: SimulatedClock) -> None:
    def boom(self: NSEClient, url: str, params: dict[str, Any] | None) -> Any:
        raise TimeoutError("read timed out")

    monkeypatch.setattr(NSEClient, "_default_get", boom)
    _assert_all_empty(NSEClient(None, clock))


def test_even_an_internal_fetch_error_never_escapes(monkeypatch: pytest.MonkeyPatch, clock: SimulatedClock) -> None:
    def broken(self: NSEClient, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bug in fetch")

    monkeypatch.setattr(NSEClient, "_fetch", broken)
    c = NSEClient(None, clock)
    _assert_all_empty(c)
    assert c.last_error is not None and "bug in fetch" in c.last_error


@pytest.mark.parametrize("payload", [["junk", 1], [None], {"data": "junk"}, {"data": [None, 1]}, "text", 5,
                                     {"a": 5}, ValueError("bad json")])
def test_malformed_payloads_never_raise(clock: SimulatedClock, payload: Any) -> None:
    c = NSEClient(None, clock, http_get=lambda url, params: Resp(payload))
    for call in _all_calls(c).values():
        call()


def test_http_error_status_is_a_failure(clock: SimulatedClock) -> None:
    c = NSEClient(None, clock, http_get=lambda url, params: Resp({"data": [{"symbol": "X"}]}, status=403))
    assert c.gsm_symbols() == set()
    assert c.last_error is not None and "403" in c.last_error


# ------------------------------------------------------------------ happy paths via a routed fake


def _router(calls: list[str]) -> Callable[[str, dict[str, Any] | None], Resp]:
    payloads: dict[str, Any] = {
        "holiday-master": {"CM": [{"tradingDate": "26-Jan-2026", "description": "Republic Day"},
                                  {"tradingDate": "26-Jan-2027", "description": "Republic Day"}]},
        "reportASM": {"longterm": {"data": [{"symbol": "aaa"}]}, "shortterm": {"data": [{"symbol": "BBB"}]}},
        "reportGSM": {"data": [{"symbol": "CCC"}, {"nosymbol": 1}]},
        "fiidiiTradeReact": [{"category": "FII/FPI *", "date": "23-Nov-2026", "buyValue": "12,000.50",
                              "sellValue": "13,000", "netValue": "-999.50"},
                             {"category": "DII **", "date": "23-Nov-2026", "buyValue": "9,000",
                              "sellValue": "8,000", "netValue": "1,000"}],
        "corporates-corporateActions": [
            {"symbol": "ABC", "series": "EQ", "exDate": "26-Nov-2026", "recDate": "26-Nov-2026",
             "subject": "Interim Dividend - Rs 5.50 Per Share", "faceVal": "10"}],
        "event-calendar": [{"symbol": "ABC", "date": "25-Nov-2026", "purpose": "Financial Results",
                            "bm_desc": "To consider results", "company": "ABC Ltd"}],
        "corporate-announcements": [
            {"symbol": "ABC", "desc": "Outcome of Board Meeting", "an_dt": "23-Nov-2026 18:00:00",
             "attchmntText": "Results approved", "attchmntFile": "https://x/y.pdf"},
            {"symbol": "XYZ", "desc": "Press release", "an_dt": "22-Nov-2026 10:00:00"},
            {"symbol": "ABC", "desc": "Old", "an_dt": "01-Nov-2026 10:00:00"}],
        "allIndices": {"data": [{"index": "NIFTY 50", "advances": "30", "declines": "20"},
                                {"index": "NIFTY 500", "advances": "300", "declines": "200"}]},
        "corporate-pledgedata": {"data": [{"percPromoterHolding": "55.0", "percPromoterShares": "12.5"}]},
    }
    bhav = ("SYMBOL, SERIES, DATE1, DELIV_QTY, DELIV_PER\n ABC, EQ, 30-Nov-2026, 100, 45.67\n"
            " ABC, BL, 30-Nov-2026, -, -\n XYZ, BE, 30-Nov-2026, 5, 55.10\n")

    def get(url: str, params: dict[str, Any] | None) -> Resp:
        calls.append(url)
        if "sec_bhavdata_full" in url:
            return Resp(text=bhav)
        for key, payload in payloads.items():
            if url.endswith(key):
                return Resp(payload)
        return Resp(status=404)

    return get


def test_client_parses_every_endpoint(clock: SimulatedClock) -> None:
    calls: list[str] = []
    c = NSEClient(None, clock, http_get=_router(calls))
    assert c.holidays(2026) == {date(2026, 1, 26)}
    assert c.asm_symbols() == {"AAA", "BBB"}
    assert c.gsm_symbols() == {"CCC"}
    flows = c.fii_dii()
    assert flows["fii_net_cr"] == -999.5 and flows["dii_buy_cr"] == 9000.0 and flows["date"] == "2026-11-23"
    ca = c.corporate_actions(D0, D1)
    assert len(ca) == 1 and ca[0].event_type == EventType.EX_DIVIDEND and ca[0].amount == 5.5
    ev = c.event_calendar(D0, D1)
    assert len(ev) == 1 and ev[0].event_type == EventType.RESULTS and ev[0].event_date == date(2026, 11, 25)
    anns = c.announcements("abc", SINCE)
    assert [h.title for h in anns] == ["ABC: Outcome of Board Meeting"]
    assert anns[0].symbol == "ABC" and anns[0].url == "https://x/y.pdf" and anns[0].source == "NSE"
    assert len(c.announcements(None, SINCE)) == 2
    assert c.delivery_pct("abc", D1) == 45.67 and c.delivery_map(D1) == {"ABC": 45.67, "XYZ": 55.1}
    assert c.advance_decline() == 1.5
    assert c.pledge_pct("ABC") == 12.5


def test_cache_ttl_stale_fallback_and_offline(clock: SimulatedClock, tmp_path: Path) -> None:
    calls: list[str] = []
    c = NSEClient(tmp_path, clock, http_get=_router(calls))
    assert c.holiday_map()
    assert c.holiday_map()
    assert len(calls) == 1                                        # within TTL: served from cache
    assert (tmp_path / "nse" / "holidays_trading.json").exists()
    fresh = NSEClient(tmp_path, clock, http_get=_router(calls))
    assert fresh.holiday_map() and len(calls) == 1                # disk cache survives restarts

    def boom(url: str, params: dict[str, Any] | None) -> Any:
        raise ConnectionError("down")

    clock.advance(timedelta(days=8))                              # TTL (7 days) expired
    stale = NSEClient(tmp_path, clock, http_get=boom)
    assert stale.holiday_map() == {date(2026, 1, 26): "Republic Day", date(2027, 1, 26): "Republic Day"}
    assert stale.last_error is not None                           # failure recorded, stale data served

    offline_calls: list[str] = []
    off = NSEClient(tmp_path, clock, http_get=_router(offline_calls), offline=True)
    assert off.holiday_map()                                      # cached
    assert off.gsm_symbols() == set()                             # nothing cached: empty, no request
    assert offline_calls == []


def test_delivery_map_keeps_only_parsed_cache(clock: SimulatedClock, tmp_path: Path) -> None:
    calls: list[str] = []
    c = NSEClient(tmp_path, clock, http_get=_router(calls))
    assert c.delivery_map(D1)["ABC"] == 45.67
    assert (tmp_path / "nse" / "delivery_20261130.json").exists()
    assert not (tmp_path / "nse" / "bhav_20261130.json").exists()
    assert c.delivery_map(D1) and len(calls) == 1


# ------------------------------------------------------------------ parsers


def test_parse_dates() -> None:
    assert parse_nse_date("02-Nov-2026") == date(2026, 11, 2)
    assert parse_nse_date("02-11-2026") == date(2026, 11, 2)
    assert parse_nse_date("2026-11-02") == date(2026, 11, 2)
    assert parse_nse_date("24-Nov-2026 19:49:52") == date(2026, 11, 24)
    assert parse_nse_date("-") is None and parse_nse_date("") is None and parse_nse_date("garbage") is None
    assert parse_nse_datetime("24-Nov-2026 19:49:52") == datetime(2026, 11, 24, 19, 49, 52, tzinfo=IST)
    assert parse_nse_datetime(None) is None


def test_parse_ca_subject_variants() -> None:
    div = parse_ca_subject("Final Dividend - Rs 2 Per Share And Special Dividend - Rs 3.50 Per Share")
    assert div[0]["event_type"] == EventType.EX_DIVIDEND and div[0]["amount"] == 5.5
    assert parse_ca_subject("Dividend - Re 0.50 Per Share")[0]["amount"] == 0.5
    bonus = parse_ca_subject("Bonus 1:1")
    assert bonus[0]["event_type"] == EventType.BONUS and bonus[0]["ratio"] == 2.0
    split = parse_ca_subject("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share")
    assert split[0]["event_type"] == EventType.SPLIT and split[0]["ratio"] == 10.0
    rights = parse_ca_subject("Rights 1:5 @ Premium Rs 100/-")
    assert rights[0]["event_type"] == EventType.RIGHTS and rights[0]["details"]["premium"] == 100.0
    assert parse_ca_subject("Annual General Meeting")[0]["event_type"] == EventType.AGM
    assert parse_ca_subject("Buy Back")[0]["details"] == {"kind": "BUYBACK"}
    assert parse_ca_subject("Interest Payment") == []
    assert parse_ca_subject("Capital Reduction")[0]["event_type"] == EventType.EX_DATE


def test_parse_corporate_actions_merges_and_filters() -> None:
    rows = [
        {"symbol": "abc", "series": "EQ", "exDate": "26-Nov-2026", "subject": "Interim Dividend - Rs 2 Per Share"},
        {"symbol": "ABC", "series": "EQ", "exDate": "26-Nov-2026", "subject": "Special Dividend - Rs 1 Per Share"},
        {"symbol": "ABC", "series": "N1", "exDate": "26-Nov-2026", "subject": "Interest Payment"},
        {"symbol": "DEF", "series": "EQ", "exDate": "-", "subject": "Bonus 1:2"},
        {"symbol": "GHI", "series": "EQ", "exDate": "27-Nov-2026", "subject": "Bonus 1:2"},
    ]
    events = parse_corporate_actions(rows)
    assert [(e.symbol, e.event_type) for e in events] == [("ABC", EventType.EX_DIVIDEND), ("GHI", EventType.BONUS)]
    assert events[0].amount == 3.0 and events[1].ratio == 1.5


def test_parse_event_calendar_and_announcements() -> None:
    ev = parse_event_calendar([{"symbol": "a", "date": "25-Nov-2026", "purpose": "Financial Results"},
                               {"symbol": "B", "date": "26-Nov-2026", "purpose": "Fund Raising"},
                               {"symbol": "", "date": "26-Nov-2026", "purpose": "x"}])
    assert [(e.symbol, e.event_type) for e in ev] == [("A", EventType.RESULTS), ("B", EventType.BOARD_MEETING)]
    rows = [{"symbol": "A", "desc": "d1", "an_dt": "23-Nov-2026 10:00:00"},
            {"symbol": "A", "desc": "d2", "an_dt": "24-Nov-2026 09:00:00"},
            {"symbol": "A", "desc": "future", "an_dt": "25-Nov-2026 09:00:00"},
            {"symbol": "B", "desc": "other", "an_dt": "24-Nov-2026 09:00:00"}]
    out = parse_announcements(rows, ist(2026, 11, 23), ist(2026, 11, 24, 12, 0), "a")
    assert [h.title for h in out] == ["A: d2", "A: d1"]                    # newest first, window applied


def test_parse_bhavcopy_holidays_and_pledge() -> None:
    text = "SYMBOL, SERIES, DELIV_PER\n AAA, EQ, 40.5\n AAA, EQ, 99\n BBB, N1, 10\n CCC, BE, -\n"
    assert parse_bhavcopy_delivery(text) == {"AAA": 40.5}
    hm = parse_holiday_master({"CM": [{"tradingDate": "02-Oct-2026", "description": "Gandhi Jayanti"}],
                               "FO": [{"tradingDate": "03-Oct-2026"}]})
    assert hm == {date(2026, 10, 2): "Gandhi Jayanti"}
    assert parse_holiday_master(["bad"]) == {}
    # a promoter HOLDING percentage must never be mistaken for the pledge percentage
    assert parse_pledge({"data": [{"percPromoterHolding": "50.1"}]}) is None
    assert parse_pledge({"data": [{"percPromoterHolding": "50.1", "percPromoterShares": "4.2",
                                   "percTotShares": "2.1"}]}) == 4.2
    assert parse_pledge([{"pledgedPercTotal": "3", "percSharesPledged": "7"}]) == 7.0
    assert parse_pledge({"data": []}) is None and parse_pledge(None) is None
