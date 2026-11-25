"""strategies.next_day: deterministic end-of-day ranking of tomorrow's INTRADAY candidates."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from core.clock import IST
from features.indicators import atr, ema
from strategies import next_day
from strategies.base import round_tick
from strategies.next_day import build_next_day_plan, evaluate_symbol

END = date(2026, 11, 25)                      # Wednesday; the plan is for Thu 26 Nov
AT = datetime(2026, 11, 25, 15, 40, tzinfo=IST)


def _days(end: date, n: int) -> list[date]:
    out: list[date] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _frame(closes: list[float], highs: list[float], lows: list[float], vols: list[float],
           end: date = END) -> pd.DataFrame:
    idx = [datetime.combine(d, time(9, 15), tzinfo=IST) for d in _days(end, len(closes))]
    c = np.asarray(closes, dtype=float)
    opens = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": c, "volume": vols},
                        index=pd.DatetimeIndex(idx))


def breakout_frame(scale: float = 1.0) -> pd.DataFrame:
    """Steady uptrend, 10-day tight base just under the high, last day an NR7 inside day."""
    c = [100 + 0.5 * i for i in range(90)] + [144.5, 145, 144.8, 145.2, 144.9, 145.1, 145.0, 145.3, 145.2, 145.25]
    h = [x + 1.0 for x in c]
    lo = [x - 1.0 for x in c]
    h[-1], lo[-1] = c[-1] + 0.2, c[-1] - 0.3
    v = [1e6] * len(c)
    v[-1] = 1.2e6
    return _frame([x * scale for x in c], [x * scale for x in h], [x * scale for x in lo], v)


def pullback_frame() -> pd.DataFrame:
    """Uptrend, three down days, last day dips to EMA20 on light volume and closes above it."""
    c = [100 + 0.55 * i for i in range(95)]
    c += [c[-1] - 1.0, c[-1] - 2.2, c[-1] - 3.2]
    h = [x + 1.0 for x in c]
    lo = [x - 1.0 for x in c]
    e = float(ema(pd.Series(c), 20).iloc[-1])
    c.append(e + 0.8)
    h.append(e + 1.3)
    lo.append(e - 0.2)
    return _frame(c, h, lo, [1e6] * (len(c) - 1) + [0.6e6])


def breakdown_frame() -> pd.DataFrame:
    """Mirror of the breakout: downtrend, tight base just above the 20-day low."""
    c = [250 - 0.5 * i for i in range(90)] + [205.5, 205, 205.2, 204.8, 205.1, 204.9, 205.0, 204.7, 204.8, 204.75]
    h = [x + 1.0 for x in c]
    lo = [x - 1.0 for x in c]
    h[-1], lo[-1] = c[-1] + 0.3, c[-1] - 0.2
    v = [1e6] * len(c)
    v[-1] = 1.3e6
    return _frame(c, h, lo, v)


def nifty_frame() -> pd.DataFrame:
    c = list(np.linspace(24000, 24200, 100))
    return _frame(c, [x + 50 for x in c], [x - 50 for x in c], [1e9] * 100)


class FakeData:
    def __init__(self, frames: dict[str, pd.DataFrame], raise_for: set[str] | None = None) -> None:
        self.frames = frames
        self.raise_for = raise_for or set()
        self.calendar = None

    def candles(self, symbol: str, timeframe: Any, lookback_bars: int = 300, start: Any = None) -> pd.DataFrame:
        if symbol in self.raise_for:
            raise RuntimeError("provider down")
        f = self.frames.get(symbol)
        return f.tail(lookback_bars).copy() if f is not None else pd.DataFrame()


# --------------------------------------------------------------------------- setups + levels


def test_breakout_is_detected_with_exact_levels() -> None:
    df = breakout_frame()
    out = evaluate_symbol("abc", df, nifty_ret=1.0, max_price=1000.0, for_date=date(2026, 11, 26))
    assert out is not None and out["symbol"] == "ABC"
    assert out["setup"] == "BREAKOUT" and out["direction"] == "LONG"
    a = float(atr(df, 14).iloc[-1])
    hi20 = float(df["high"].iloc[-21:-1].max())
    h, lo = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
    trig = round_tick(max(h, hi20) + 0.1 * a, "up")
    stop = round_tick(max(lo, trig - 1.2 * a), "down")
    assert out["trigger"] == trig and out["stop"] == stop
    assert out["target"] == round_tick(trig + 2 * (trig - stop))
    assert out["rr"] == pytest.approx(2.0, abs=0.05)
    assert out["high20"] == round(hi20, 2) and out["pdh"] == round(h, 2) and out["pdl"] == round(lo, 2)
    assert out["atr"] == round(a, 2) and out["ema20"] == round(float(ema(df["close"], 20).iloc[-1]), 2)
    assert out["volume_ratio"] == pytest.approx(1.2, abs=0.01)
    assert any("NR7" in r for r in out["reasons"]) and any("inside day" in r for r in out["reasons"])
    assert out["affordable"] is True and out["for_date"] == "2026-11-26"
    assert 55 <= out["score"] <= 100
    for k in ("trigger", "stop", "target"):                                 # tick-rounded to 0.05
        assert round(out[k] / 0.05, 6) == round(out[k] / 0.05)


def test_pullback_is_detected_with_exact_levels() -> None:
    df = pullback_frame()
    out = evaluate_symbol("PB", df, nifty_ret=0.5, max_price=None)
    assert out is not None and out["setup"] == "PULLBACK" and out["direction"] == "LONG"
    a = float(atr(df, 14).iloc[-1])
    e20 = float(ema(df["close"], 20).iloc[-1])
    h, lo = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
    trig = round_tick(h + 0.05 * a, "up")
    stop = round_tick(min(lo, e20 - 0.3 * a), "down")
    assert (out["trigger"], out["stop"]) == (trig, stop)
    assert out["target"] == round_tick(trig + 2 * (trig - stop))
    assert out["volume_ratio"] == pytest.approx(0.6, abs=0.01)
    assert any("pulled back to EMA20" in r for r in out["reasons"])
    assert out["affordable"] is True                                       # no max price -> not checked


def test_breakdown_is_a_short_mirror() -> None:
    df = breakdown_frame()
    out = evaluate_symbol("BD", df, nifty_ret=1.0)
    assert out is not None and out["setup"] == "BREAKDOWN" and out["direction"] == "SHORT"
    a = float(atr(df, 14).iloc[-1])
    lo20 = float(df["low"].iloc[-21:-1].min())
    h, lo = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
    trig = round_tick(min(lo, lo20) - 0.1 * a, "down")
    stop = round_tick(min(h, trig + 1.2 * a), "up")
    assert (out["trigger"], out["stop"]) == (trig, stop) and out["stop"] > out["trigger"] > out["target"]
    assert out["target"] == round_tick(trig - 2 * (stop - trig))
    assert out["rs_20d"] is not None and out["rs_20d"] < 0
    assert any("intraday only" in r for r in out["reasons"])


def test_no_setup_in_a_flat_market() -> None:
    c = [100.0 + (0.3 if i % 2 else -0.3) for i in range(100)]
    assert evaluate_symbol("FLAT", _frame(c, [x + 1 for x in c], [x - 1 for x in c], [1e6] * 100)) is None


# --------------------------------------------------------------------------- the whole plan


def test_plan_ranks_skips_bad_data_and_flags_affordability() -> None:
    frames = {"BO": breakout_frame(), "PB": pullback_frame(), "BD": breakdown_frame(), "NIFTY50": nifty_frame(),
              "SHORT": breakout_frame().tail(40), "STALE": breakout_frame().iloc[:-10]}
    stale = breakout_frame()
    stale.index = stale.index - pd.Timedelta(days=30)
    frames["STALE"] = stale
    data = FakeData(frames, raise_for={"BOOM"})
    out = build_next_day_plan(data, ["bo", "PB", "BD", "SHORT", "STALE", "MISSING", "BOOM", "BO"], AT, None,
                              max_price=147.0)
    syms = [c["symbol"] for c in out]
    assert sorted(syms) == ["BD", "BO", "PB"]                              # < 60 bars / stale / missing / error
    assert [c["score"] for c in out] == sorted((c["score"] for c in out), reverse=True)
    by = {c["symbol"]: c for c in out}
    assert by["BO"]["affordable"] is True and by["PB"]["affordable"] is False   # trigger 148.x > 147
    assert by["BO"]["for_date"] == "2026-11-26"                             # next weekday (no calendar)
    assert by["BO"]["rs_20d"] is not None
    assert set(by["BO"]) >= {"symbol", "setup", "direction", "trigger", "stop", "target", "rr", "score", "reasons",
                             "close", "atr", "ema20", "pdh", "pdl", "high20", "low20", "volume_ratio", "rs_20d",
                             "affordable", "for_date"}


def test_plan_respects_universe_price_and_turnover_filters(settings) -> None:  # type: ignore[no-untyped-def]
    from config.loader import update_settings

    data = FakeData({"BO": breakout_frame(), "NIFTY50": nifty_frame()})
    assert [c["symbol"] for c in build_next_day_plan(data, ["BO"], AT, settings)] == ["BO"]
    strict = update_settings(settings, {"universe.min_avg_daily_turnover_cr": 50.0})   # BO trades ~Rs 15 cr/day
    assert build_next_day_plan(data, ["BO"], AT, strict) == []
    pricey = update_settings(settings, {"universe.min_price": 500.0})
    assert build_next_day_plan(data, ["BO"], AT, pricey) == []


def test_missing_nifty_gives_neutral_rs_and_never_raises() -> None:
    out = build_next_day_plan(FakeData({"BO": breakout_frame()}, raise_for={"NIFTY50"}), ["BO"], AT, None)
    assert len(out) == 1 and out[0]["rs_20d"] is None
    assert build_next_day_plan(FakeData({}), [], AT, None) == []


def test_ties_put_affordable_candidates_first(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(symbol: str, df: Any, **kw: Any) -> dict[str, Any]:
        return {"symbol": symbol, "score": 70.0 if symbol != "LOW" else 60.0, "affordable": symbol in ("ZZZ", "LOW")}

    monkeypatch.setattr(next_day, "evaluate_symbol", fake)
    frames = {s: breakout_frame() for s in ("AAA", "ZZZ", "LOW")}
    out = build_next_day_plan(FakeData(frames), ["AAA", "ZZZ", "LOW"], AT, None)
    assert [c["symbol"] for c in out] == ["ZZZ", "AAA", "LOW"]
