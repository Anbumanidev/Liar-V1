"""Web dashboard charts (session-anchored resampling, overlays, markers) and the SSE stream."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from core.clock import IST
from storage.models import DecisionRow
from tests.helpers import make_intraday_candles, trading_days
from tests.webui.webui_kit import NOW
from webui.candles import resample_session
from webui.util import IST_OFFSET_S


def _wall(t: int) -> datetime:
    """Chart time (IST-shifted epoch) -> naive IST wall-clock datetime."""
    return datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None)


def test_webui_resample_10m_anchored_and_no_cross_day() -> None:
    days = trading_days(datetime(2026, 11, 23).date(), 2)
    src = make_intraday_candles(days, freq_minutes=5, seed=3)
    out = resample_session(src, 10)
    assert len(out) == 2 * 38  # 37 full 10m buckets + the 15:25 stub per session
    local = pd.DatetimeIndex(out.index).tz_convert(IST)
    minutes = local.hour * 60 + local.minute - (9 * 60 + 15)
    assert ((minutes % 10) == 0).all() and minutes.min() == 0 and minutes.max() == 370
    for d in days:
        day_src = src[pd.DatetimeIndex(src.index).date == d]
        day_out = out[local.date == d]
        assert day_out["volume"].sum() == pytest.approx(day_src["volume"].sum())
        last = day_out.iloc[-1]
        assert last["volume"] == day_src.iloc[-1]["volume"]  # 15:25 stub never merges into the next day
        assert day_out.iloc[0]["open"] == day_src.iloc[0]["open"]
        assert day_out["high"].max() == day_src["high"].max()
    # a source starting mid-bucket drops the leading partial bucket
    assert resample_session(src.iloc[1:], 10).index[0] == out.index[1]


def test_webui_resample_5h_two_buckets_per_session() -> None:
    days = trading_days(datetime(2026, 11, 23).date(), 3)
    src = make_intraday_candles(days, freq_minutes=60, seed=5)   # 09:15 ... 15:15 (7 bars/day)
    out = resample_session(src, 300)
    local = pd.DatetimeIndex(out.index).tz_convert(IST)
    assert list(local.strftime("%H:%M")) == ["09:15", "14:15"] * 3
    first_day = src[pd.DatetimeIndex(src.index).date == days[0]]
    assert out.iloc[0]["volume"] == pytest.approx(first_day.iloc[:5]["volume"].sum())
    assert out.iloc[1]["volume"] == pytest.approx(first_day.iloc[5:]["volume"].sum())
    assert out.iloc[1]["close"] == first_day.iloc[-1]["close"]


def test_webui_candles_api_10m(authed: TestClient) -> None:
    r = authed.get("/api/candles?symbol=SBIN&tf=10m&bars=60")
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["tf"] == "10m" and c["source_tf"] == "5m" and c["intraday"] is True and c["bars"] == 60
    times = [b["time"] for b in c["candles"]]
    assert times == sorted(times)
    for t in times:
        w = _wall(t)
        off = w.hour * 60 + w.minute - 555
        assert 0 <= off < 375 and off % 10 == 0
    assert len(c["ema9"]) == 60 - 8 and len(c["ema21"]) == 60 - 20 and c["vwap"]
    assert c["quote"]["ltp"] == 812.0
    kinds = {line["kind"] for line in c["lines"]}
    assert kinds == {"entry", "sl", "t1", "t2"}  # open SBIN position
    # IST shift: epoch of the bar opening at 09:15 IST + 19800 lands on 09:15 "UTC"
    first = _wall(times[0])
    real = datetime.fromtimestamp(times[0] - IST_OFFSET_S, tz=IST)
    assert (first.hour, first.minute) == (real.hour, real.minute)


def test_webui_candles_api_5h_daily_and_markers(authed: TestClient) -> None:
    c = authed.get("/api/candles?symbol=ITC&tf=5h&bars=10").json()
    walls = [_wall(b["time"]) for b in c["candles"]]
    assert {(w.hour, w.minute) for w in walls} == {(9, 15), (14, 15)}
    markers = c["markers"]
    assert {m["kind"] for m in markers} == {"entry", "exit"}
    assert any(m["text"].startswith("BUY 2 @ 300.00") for m in markers)
    times = {b["time"] for b in c["candles"]}
    assert all(m["time"] in times for m in markers)  # snapped to existing bars
    d = authed.get("/api/candles?symbol=ITC&tf=1d&bars=30").json()
    assert d["bars"] == 30 and d["vwap"] is None and d["intraday"] is False
    w = authed.get("/api/candles?symbol=ITC&tf=1wk&bars=10").json()
    assert w["source_tf"] == "1wk"
    assert authed.get("/api/candles?symbol=ITC&tf=7m").status_code == 422
    assert authed.get("/api/candles?symbol=..%2Fetc&tf=5m").status_code == 422


def test_webui_candles_window_for_trade(authed: TestClient, ctx: Any) -> None:
    r = authed.get("/api/candles?symbol=ITC&tf=5m&from=2026-11-25T09:30:00%2B05:30&to=2026-11-25T11:30:00%2B05:30")
    c = r.json()
    walls = [_wall(b["time"]) for b in c["candles"]]
    assert walls[0] >= datetime(2026, 11, 25, 9, 25) and walls[-1] <= datetime(2026, 11, 25, 11, 30)
    assert len(c["markers"]) == 3  # entry + partial + final exit
    # cached: second call does not hit the data service again
    n = len(ctx.data.calls)
    authed.get("/api/candles?symbol=ITC&tf=5m&from=2026-11-25T09:30:00%2B05:30&to=2026-11-25T11:30:00%2B05:30")
    assert len(ctx.data.calls) == n


def _events(body: str) -> list[tuple[str, str]]:
    out = []
    for block in body.split("\n\n"):
        lines = block.strip().splitlines()
        ev = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in lines if ln.startswith("data: ")), "")
        if ev:
            out.append((ev, data))
    return out


def test_webui_stream_yields_events(authed: TestClient, web_env: dict[str, Any]) -> None:
    with authed.stream("GET", "/api/stream") as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        body = b""
        for chunk in r.iter_bytes():
            body += chunk
            if len(body) > 200_000:
                break
    events = _events(body.decode())
    names = [e for e, _ in events]
    assert names[0] == "hello" and "status" in names and "approvals" in names
    assert "decision" not in names  # no backlog without cursors


def test_webui_stream_resumes_from_cursor(authed: TestClient, web_env: dict[str, Any]) -> None:
    with web_env["factory"]() as s:  # type: Session
        s.add(DecisionRow(batch_id=web_env["ids"]["batch"], symbol="LT", mode="INTRADAY", decision="NO_TRADE",
                          confidence=0, context_json={"filtered_reason": "no strategy setup"},
                          final_action="FILTERED", created_at=NOW - timedelta(seconds=1)))
        s.commit()
        last_before = web_env["ids"]["d_exec"] + 5  # the 6 seeded decisions
    with authed.stream("GET", f"/api/stream?decision={last_before}&order=0&trade=0") as r:
        body = r.read().decode()
    events = _events(body)
    decisions = [d for e, d in events if e == "decision"]
    assert len(decisions) == 1 and '"symbol":"LT"' in decisions[0] and "no strategy setup" in decisions[0]
    assert sum(1 for e, _ in events if e == "order") == 7 and sum(1 for e, _ in events if e == "trade") == 4
    assert any(e == "log" for e, _ in events) is False  # log offset starts at the end of the file


def test_webui_stream_ends_when_session_idles_out(web_env: dict[str, Any], monkeypatch: Any) -> None:
    from core.clock import SimulatedClock
    from tests.webui.webui_kit import login, make_ctx
    from webui import stream as stream_mod
    from webui.server import create_app

    clock = SimulatedClock(NOW)
    ctx = make_ctx(web_env, clock=clock, stream_max_iterations=20)
    real_poll = stream_mod.poll_once

    def ticking_poll(c: Any, st: Any) -> Any:
        clock.advance(timedelta(minutes=10))  # each poll "takes" 10 minutes of idle time
        return real_poll(c, st)

    monkeypatch.setattr(stream_mod, "poll_once", ticking_poll)
    with TestClient(create_app(ctx)) as c:
        login(c)
        with c.stream("GET", "/api/stream") as r:
            body = r.read().decode()
    names = [e for e, _ in _events(body)]
    assert names[-1] == "session_expired" and names.count("status") >= 1
    ctx.close()
