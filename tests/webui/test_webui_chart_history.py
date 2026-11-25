"""Chart history: ``range`` (clamped to the timeframe's provider history), ``before`` scroll-back pages,
``has_more`` / ``earliest_available``, session-anchored buckets (never across days) and the price-behaviour
snapshot exposed on decisions and trades."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core.clock import IST
from storage.models import DecisionRow
from tests.webui.webui_kit import NOW
from webui.candles import HISTORY_DAYS, TIMEFRAMES, earliest_available, range_start, start_for_bars
from webui.util import IST_OFFSET_S


def _wall(t: int) -> datetime:
    """Chart time (IST-shifted epoch) -> naive IST wall-clock datetime."""
    return datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None)


def _chart_time(dt: datetime) -> int:
    return int(dt.timestamp()) + IST_OFFSET_S


def _anchored(c: dict[str, Any], minutes: int) -> None:
    """Every bar opens on a session-anchored bucket boundary inside 09:15-15:30 (so none spans two days)."""
    for b in c["candles"]:
        w = _wall(b["time"])
        off = w.hour * 60 + w.minute - 555
        assert 0 <= off < 375 and off % minutes == 0, w


# =========================================================================== helpers


def test_webui_history_limits_and_range_starts(ctx: Any) -> None:
    cal = ctx.get_calendar()
    assert HISTORY_DAYS["10m"] == HISTORY_DAYS["5m"] == 59 and HISTORY_DAYS["5h"] == HISTORY_DAYS["1h"] == 729
    assert set(HISTORY_DAYS) == set(TIMEFRAMES)
    e1 = earliest_available(TIMEFRAMES["1m"], NOW)
    assert NOW - timedelta(days=7) < e1 <= NOW - timedelta(days=7) + timedelta(minutes=1)
    ed = earliest_available(TIMEFRAMES["1d"], NOW)
    assert (ed.hour, ed.minute) == (9, 15) and (NOW - ed).days in (4382, 4383)
    # 1D = today's session (market open at 14:00); 5D = 5 sessions back (weekend skipped; the test calendar
    # has no exchange holidays); before 09:15 "1D" is the previous session
    assert range_start("1D", TIMEFRAMES["5m"], NOW, cal) == datetime(2026, 11, 25, 9, 15, tzinfo=IST)
    assert range_start("5D", TIMEFRAMES["5m"], NOW, cal) == datetime(2026, 11, 19, 9, 15, tzinfo=IST)
    before_open = datetime(2026, 11, 25, 8, 0, tzinfo=IST)
    assert range_start("1D", TIMEFRAMES["5m"], before_open, cal) == datetime(2026, 11, 24, 9, 15, tzinfo=IST)
    assert range_start("1M", TIMEFRAMES["1h"], NOW, cal) == datetime(2026, 10, 26, 9, 15, tzinfo=IST)
    assert range_start("MAX", TIMEFRAMES["5m"], NOW, cal) == earliest_available(TIMEFRAMES["5m"], NOW)
    # enough sessions for N bars of a timeframe (+1 spare session)
    assert start_for_bars(TIMEFRAMES["10m"], NOW, 40, cal) == datetime(2026, 11, 20, 9, 15, tzinfo=IST)
    # the weekday fallback works without a calendar
    assert range_start("1D", TIMEFRAMES["5m"], NOW, None) == datetime(2026, 11, 25, 9, 15, tzinfo=IST)


# =========================================================================== range


def test_webui_candles_range_clamped_to_history(authed: TestClient) -> None:
    c = authed.get("/api/candles?symbol=SBIN&tf=1m&range=1M").json()
    assert c["range"] == "1M" and c["clamped"] is True and c["has_more"] is False
    earliest = datetime.fromisoformat(c["earliest_available"])
    assert NOW - timedelta(days=7) <= earliest <= NOW - timedelta(days=7) + timedelta(minutes=1)
    assert datetime.fromisoformat(c["start"]) == earliest
    assert c["history_days"] == 7 and "7 days" in c["history_note"]
    assert c["candles"] and all(_wall(b["time"]) >= earliest.replace(tzinfo=None) for b in c["candles"])
    # MAX is "everything the provider keeps": clamped by definition, nothing older to load
    m = authed.get("/api/candles?symbol=ITC&tf=1d&range=MAX").json()
    assert m["clamped"] is True and m["has_more"] is False and m["intraday"] is False and m["bars"] == 120
    # a range within the limit but older than the provider's data: has_more False (first bar far after the start)
    o = authed.get("/api/candles?symbol=SBIN&tf=5m&range=1M").json()
    assert o["clamped"] is False and o["has_more"] is False and o["bars"] == 5 * 75


def test_webui_candles_range_1d_one_session_no_cross_day(authed: TestClient) -> None:
    c = authed.get("/api/candles?symbol=SBIN&tf=10m&range=1D").json()
    assert c["range"] == "1D" and c["candles"]
    assert {_wall(b["time"]).date() for b in c["candles"]} == {datetime(2026, 11, 25).date()}
    _anchored(c, 10)
    assert c["has_more"] is True                         # earlier sessions exist
    five = authed.get("/api/candles?symbol=SBIN&tf=30m&range=5D").json()
    days = sorted({_wall(b["time"]).date() for b in five["candles"]})
    assert len(days) == 5 and five["bars"] == 5 * 13    # 12 full 30m buckets + the 15:15 stub per session
    _anchored(five, 30)
    assert len(five["ema9"]) == five["bars"] - 8 and five["vwap"]


def test_webui_candles_range_validation(authed: TestClient) -> None:
    assert authed.get("/api/candles?symbol=SBIN&tf=5m&range=2Y").status_code == 422
    assert authed.get("/api/candles?symbol=SBIN&tf=5m&range=max").json()["range"] == "MAX"   # case-insensitive
    assert authed.get("/api/candles?symbol=SBIN&tf=5m&before=12").status_code == 422
    assert authed.get("/api/candles?symbol=SBIN&tf=5m&bars=5000").status_code == 200
    assert authed.get("/api/candles?symbol=SBIN&tf=5m&bars=5001").status_code == 422


# =========================================================================== before (scroll-back)


def test_webui_candles_before_pages_back_without_gaps(authed: TestClient, ctx: Any) -> None:
    latest = authed.get("/api/candles?symbol=SBIN&tf=10m&bars=60").json()
    assert latest["bars"] == 60 and latest["has_more"] is True and latest["earliest_available"]
    first = latest["candles"][0]["time"]
    page = authed.get(f"/api/candles?symbol=SBIN&tf=10m&bars=40&before={first}").json()
    assert page["bars"] == 40 and page["has_more"] is True and page["before"]
    times = [b["time"] for b in page["candles"]]
    assert times == sorted(times) and times[-1] < first
    _anchored(page, 10)
    # contiguous with the newer page: the bar right before `first` is the last bar of this page
    full = authed.get("/api/candles?symbol=SBIN&tf=10m&range=MAX").json()
    all_times = [b["time"] for b in full["candles"]]
    i = all_times.index(first)
    assert all_times[i - 40:i] == times
    # page down to the start of the available data
    rest = authed.get(f"/api/candles?symbol=SBIN&tf=10m&bars=500&before={times[0]}").json()
    assert rest["has_more"] is False and rest["bars"] == i - 40
    assert [b["time"] for b in rest["candles"]] == all_times[: i - 40]
    empty = authed.get(f"/api/candles?symbol=SBIN&tf=10m&bars=100&before={all_times[0]}").json()
    assert empty["bars"] == 0 and empty["has_more"] is False
    # 5h pages keep the 09:15 / 14:15 session anchors
    h5 = authed.get("/api/candles?symbol=SBIN&tf=5h&range=1D").json()
    assert h5["bars"] == 2
    older = authed.get(f"/api/candles?symbol=SBIN&tf=5h&bars=10&before={h5['candles'][0]['time']}").json()
    assert older["bars"] == 8 and older["has_more"] is False
    assert {(_wall(b["time"]).hour, _wall(b["time"]).minute) for b in older["candles"]} == {(9, 15), (14, 15)}
    assert older["candles"][-1]["time"] < h5["candles"][0]["time"]


def test_webui_candles_before_markers_stay_on_their_page(authed: TestClient) -> None:
    """A fill after the page boundary must not be snapped onto the older page's last bar."""
    ten = _chart_time(datetime(2026, 11, 25, 10, 0, tzinfo=IST))
    page = authed.get(f"/api/candles?symbol=ITC&tf=5m&bars=30&before={ten}").json()
    assert page["candles"][-1]["time"] < ten
    assert page["markers"] == []                     # ITC entry at 10:02 belongs to the newer page
    newer = authed.get("/api/candles?symbol=ITC&tf=5m&range=1D").json()
    entry = [m for m in newer["markers"] if m["kind"] == "entry"]
    assert len(entry) == 1 and _wall(entry[0]["time"]) == datetime(2026, 11, 25, 10, 0)


def test_webui_candles_trade_window_has_more(authed: TestClient) -> None:
    c = authed.get("/api/candles?symbol=ITC&tf=5m&from=2026-11-25T09:30:00%2B05:30&to=2026-11-25T11:30:00%2B05:30").json()
    assert c["has_more"] is True and c["clamped"] is False      # older sessions can be scrolled into
    few = authed.get("/api/candles?symbol=SBIN&tf=1d&bars=500").json()
    assert few["bars"] == 120 and few["has_more"] is False       # provider has fewer bars than asked


# =========================================================================== price behaviour


BEHAVIOUR = {"range_pos": 0.93, "range_vs_adr": 1.42, "vwap": 805.5, "vwap_ext_atr": 2.1, "day_high": 815.0,
             "day_low": 795.0, "day_open": 798.0, "adr": 14.1, "atr": 3.2, "pdh": 810.0, "pdl": 790.0, "pdc": 797.0,
             "gap_pct": 0.13, "continuation_rate": 0.35, "samples": 17, "direction": "LONG",
             "notes": ["price at 93% of the day range", "<script>alert(1)</script> untrusted"], "verdict": "CHASING"}


def test_webui_decision_behaviour(authed: TestClient, web_env: dict[str, Any]) -> None:
    with web_env["factory"]() as s:
        d = DecisionRow(batch_id=web_env["ids"]["batch"], symbol="SBIN", mode="INTRADAY", decision="BUY", direction="LONG",
                        confidence=70, entry_price=812.0, stop_loss=805.0, target_1=826.0, final_action="RISK_REJECTED",
                        risk_rule="CHASING_HIGH",
                        context_json={"proposal": {"behaviour": BEHAVIOUR, "ltp": 812.4, "range_pos": 0.93}},
                        created_at=NOW - timedelta(minutes=3))
        partial = DecisionRow(batch_id=web_env["ids"]["batch"], symbol="BEL", mode="INTRADAY", decision="BUY",
                              confidence=70, final_action="NO_TRADE",
                              context_json={"proposal": {"range_pos": 0.2, "behaviour_samples": 9, "continuation_rate": 0.6}},
                              created_at=NOW - timedelta(minutes=2))
        s.add_all([d, partial])
        s.commit()
        did, pid = d.id, partial.id
    b = authed.get(f"/api/decisions/{did}").json()["behaviour"]
    assert b["verdict"] == "CHASING" and b["direction"] == "LONG" and b["range_pos"] == 0.93 and b["samples"] == 17
    assert b["entry"] == 812.0 and b["ltp"] == 812.4 and b["pdh"] == 810.0 and b["vwap_ext_atr"] == 2.1
    assert b["notes"][1] == "<script>alert(1)</script> untrusted"    # plain text; the UI inserts it as a text node
    p = authed.get(f"/api/decisions/{pid}").json()["behaviour"]       # only the proposal's top-level fields
    assert p["range_pos"] == 0.2 and p["samples"] == 9 and p["continuation_rate"] == 0.6 and p["verdict"] is None
    assert p["notes"] == [] and p["day_high"] is None
    # older decisions: no snapshot -> null (the UI renders nothing)
    assert authed.get(f"/api/decisions/{web_env['ids']['d_exec']}").json()["behaviour"] is None
    t = authed.get(f"/api/trades/{web_env['ids']['itc']}").json()
    assert t["decision"]["behaviour"] is None and "risk_rule" in t["decision"]


@pytest.mark.parametrize("ctxj", [None, {}, {"proposal": None}, {"proposal": {"behaviour": "garbage"}},
                                  {"proposal": {"behaviour": {"notes": None, "verdict": "MAYBE"}}}])
def test_webui_behaviour_view_absent_or_garbage(ctxj: Any) -> None:
    from webui.queries import behaviour_view

    assert behaviour_view(ctxj) is None
