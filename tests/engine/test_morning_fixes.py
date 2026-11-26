"""2026-11-26 morning: 0 executions all session. Root causes fixed and pinned here.

1. "Room" counted micro 5-minute swing points and levels at the entry -> ~0R on every pullback.
2. Volume scored one 5-minute bar's relative volume (0.24 -> 1.23 on consecutive bars) -> ~0 points.
3. The cycle log said "no stock has a setup" while setups existed below the confluence bar.
4. The PC slept (10:20-10:47) and the dashboard logged out every 30 min; Ollama was down at start.
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from core.clock import IST
from core.enums import Direction
from core.types import ConfluenceResult, Levels


# --------------------------------------------------------------------------- 1. room ignores noise levels


class _Feat:
    def __init__(self, levels: dict[str, Levels]) -> None:
        self._lv = levels
        self.timeframes = {k: None for k in levels}
        self.week52_high = 400.0
        self.week52_low = 150.0
        self.prev_close = 220.0
        self.primary_timeframe = "5m"
        self.atr = 0.376

    def tf(self, name: str) -> Any:
        lv = self._lv.get(name)
        return SimpleNamespace(levels=lv) if lv is not None else None


def test_room_ignores_micro_swings_and_levels_at_the_entry() -> None:
    from strategies.base import nearest_opposing_level

    # ONGC 26 Nov 11:10: entry 223.80, five 5m swing highs within 0.2 of the price
    feats = _Feat({
        "5m": Levels(swing_highs=[223.78, 223.79, 223.62, 223.80, 223.85], resistances=[224.04, 225.0]),
        "15m": Levels(swing_highs=[224.13, 224.9], resistances=[224.78]),
        "1h": Levels(r1=224.73, swing_highs=[225.01]),
    })
    old = nearest_opposing_level(feats, 223.80, Direction.LONG, ["5m", "15m", "1h"])
    assert old == pytest.approx(223.85)                      # the noise the old room score used
    new = nearest_opposing_level(feats, 223.80, Direction.LONG, ["5m", "15m", "1h"],
                                 min_gap=0.25 * feats.atr, skip_swings_of="5m")
    assert new == pytest.approx(224.04)                      # first REAL resistance
    short = nearest_opposing_level(_Feat({"5m": Levels(swing_lows=[223.7], supports=[222.9])}), 223.80,
                                   Direction.SHORT, ["5m"], min_gap=0.1, skip_swings_of="5m")
    assert short == pytest.approx(222.9)


# --------------------------------------------------------------------------- 2. cumulative relative volume


def _bars(day: date, vols: list[float]) -> pd.DataFrame:
    idx = [datetime.combine(day, time(9, 15), tzinfo=IST) + timedelta(minutes=5 * i) for i in range(len(vols))]
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": vols},
                        index=pd.DatetimeIndex(idx))


def test_cumulative_rvol_is_steady_where_one_bar_is_noisy() -> None:
    from features.builder import cumulative_rvol

    days = [date(2026, 11, d) for d in (17, 18, 19, 20, 23, 24)]
    hist = [_bars(d, [1000.0] * 75) for d in days[:-1]]
    today = _bars(days[-1], [2000.0, 200.0, 3000.0, 100.0, 1700.0])   # spiky bars, 1.4x normal in total
    frame = pd.concat(hist + [today])
    assert cumulative_rvol(frame, days[-1]) == pytest.approx(7000 / 5000)
    assert cumulative_rvol(frame.iloc[:150], days[-1]) is None          # not enough history (no today)
    assert cumulative_rvol(pd.concat(hist[:2] + [today]), days[-1]) is None   # < 3 prior sessions


# --------------------------------------------------------------------------- 3. honest cycle message


def test_near_miss_names_the_best_setup_and_what_held_it_back() -> None:
    from scheduler.jobs import _near_miss

    def res(sym: str, score: float, comp: dict[str, float], strat: str) -> Any:
        return SimpleNamespace(symbol=sym, confluence=ConfluenceResult(
            score=score, grade="C", components=comp, notes=[f"strategy {strat} (desc)", "direction LONG"]))

    low = [res("ONGC", 44.6, {"trend": 12.5, "volume": 1.0, "room": 0.0}, "ema_9_21_pullback"),
           res("ITC", 57.6, {"trend": 20.0, "volume": 4.2, "regime": 10.0, "market": 12.5, "candles": 5.0,
                             "room": 2.5, "sentiment": 2.5}, "ema_9_21_pullback")]
    msg = _near_miss(low)
    assert msg.startswith("ITC ema_9_21_pullback 57.6 (weak: ")
    assert "low volume" in msg and "resistance close" in msg


# --------------------------------------------------------------------------- 4. machine / dashboard / ollama


def test_keep_awake_sets_and_releases_the_execution_state(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import core.power as power

    calls: list[int] = []
    monkeypatch.setattr(power, "_set", lambda flags: calls.append(flags) or True)
    assert power.keep_awake(display=True)
    power.release()
    assert calls[0] == power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED | power.ES_DISPLAY_REQUIRED
    assert calls[1] == power.ES_CONTINUOUS
    calls.clear()
    power.keep_awake(display=False)
    assert calls[0] == power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
def test_real_execution_state_call_works_on_windows() -> None:
    import core.power as power

    assert power.keep_awake(display=False) is True
    assert power.release() is True


def test_keep_live_session_survives_30_minutes_idle() -> None:
    from webui.security import SessionManager

    sm = SessionManager(b"k" * 32, idle=timedelta(minutes=30), absolute=timedelta(days=1))
    t0 = 1_000_000.0
    normal, _ = sm.issue("owner", t0)
    live, payload = sm.issue("owner", t0, keep_live=True)
    later = t0 + 3 * 3600                                   # 3 hours without clicks
    assert sm.verify(normal, later)[0] is None and "30 minutes" in sm.verify(normal, later)[1]
    assert sm.verify(live, later)[0] is not None
    assert sm.verify(live, t0 + 17 * 3600)[0] is None       # but never longer than 16 h
    assert sm.verify(sm.refresh(payload, later), later + 60)[0]["kl"] is True


class _LLM:
    def __init__(self, up_after: int) -> None:
        self.calls = 0
        self.up_after = up_after

    def healthcheck(self) -> bool:
        self.calls += 1
        return self.calls > self.up_after


def test_ollama_not_started_when_already_up(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import agents.ollama_service as svc

    started: list[Any] = []
    monkeypatch.setattr(svc.subprocess, "Popen", lambda *a, **k: started.append(a))
    assert svc.ensure_running(_LLM(0), "http://127.0.0.1:11434", tmp_path) is True and started == []


def test_ollama_started_when_down_and_local(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import agents.ollama_service as svc

    started: list[Any] = []
    monkeypatch.setattr(svc.subprocess, "Popen", lambda args, **k: started.append(args))
    monkeypatch.setattr(svc, "find_ollama", lambda: "C:/ollama/ollama.exe")
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)
    assert svc.ensure_running(_LLM(2), "http://127.0.0.1:11434", tmp_path) is True
    assert started == [["C:/ollama/ollama.exe", "serve"]]
    assert svc.ensure_running(_LLM(99), "http://10.0.0.5:11434", tmp_path, wait_seconds=0) is False   # remote: never
