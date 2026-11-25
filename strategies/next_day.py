"""End-of-day ranking of tomorrow's INTRADAY candidates (deterministic, CLOSED daily bars only).

Called by ``scheduler.jobs.job_next_day_plan`` at 15:40, after the 15:30 close, so today's daily
bar is included. Pure arithmetic - no LLM is involved; the plan only decides which stocks the
INTRADAY cycles analyse first the next morning. Every trade still goes through the full pipeline
and the deterministic risk manager; nothing here places orders.

Per symbol (>= 60 daily bars, else skipped): close, EMA20/EMA50, ATR14 (Wilder), the prior
20-session high/low (excluding today), today's range, NR7 (today's range is the smallest of the
last 7), inside day, volume ratio (today vs the prior 20-day average), relative strength vs NIFTY
over 20 sessions (percentage points) and the last session's high/low (``pdh``/``pdl`` = tomorrow's
previous-day high/low).

Setups (the best-scoring one per symbol is kept). "Uptrend" = close > EMA20 > EMA50 with EMA20 at
least 0.1 ATR above EMA50 and higher than 5 sessions ago ("downtrend" mirrored) - EMAs stacked by a
hair in a flat market do not count:

* BREAKOUT (LONG): uptrend and the close within 1 ATR of the 20-day high (at most
  1 ATR above it). trigger = max(today high, 20d high) + 0.1 ATR; stop = max(today low, trigger -
  1.2 ATR); target = trigger + 2 x (trigger - stop). Range compression (NR7 / inside day) scores.
* PULLBACK (LONG): uptrend, today's low within 0.5 ATR of EMA20 (or below it) and the
  close above EMA20, volume ratio < 1.2. trigger = today high + 0.05 ATR; stop = min(today low,
  EMA20 - 0.3 ATR); target = trigger + 2R.
* BREAKDOWN (SHORT, intraday only): downtrend and the close within 1 ATR of the 20-day
  low. trigger = min(today low, 20d low) - 0.1 ATR; stop = min(today high, trigger + 1.2 ATR);
  target = trigger - 2R.

Prices are tick-rounded to 0.05 (trigger/stop away from the market, target to nearest).

Score (0-100) = 25 x trend + 20 x RS + 20 x compression + 15 x volume + 10 x proximity + 10 x liquidity,
each component clipped to [0, 1] (``clip(x) = min(1, max(0, x))``; "dir" = +1 LONG / -1 SHORT):

* trend       = 0.6 (EMAs stacked in the trade direction - required by every setup)
                + 0.4 x clip(dir x (EMA20 - EMA20 five sessions ago) / ATR / 0.5)
* RS          = clip((dir x rs_20d + 2) / 10)   (-2 pts -> 0, +8 pts -> 1; 0.5 when NIFTY is missing)
* compression = 0.5 x NR7 + 0.3 x inside day + 0.2 x clip((1 - range / ATR) / 0.5)
* volume      = BREAKOUT/BREAKDOWN: clip((volume ratio - 0.8) / 0.7)   (participation)
                PULLBACK:           clip((1.2 - volume ratio) / 0.5)   (a quiet pullback)
* proximity   = clip(1 - |trigger - close| / ATR / 1.5)   (closer to the trigger is better)
* liquidity   = clip((log10(20-day average turnover in Rs crore) - 1) / 2)   (Rs 10 cr -> 0, Rs 1000 cr -> 1)

Universe price / turnover thresholds (``settings.universe``) are applied here because the daily
bars are already loaded; ``affordable`` = ``max_price is None or trigger <= max_price``.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from core.clock import to_ist
from core.enums import Timeframe
from features.indicators import atr as atr_series
from features.indicators import ema
from strategies.base import round_tick

MIN_BARS = 60
LOOKBACK_BARS = 120
MIN_EMA_GAP_ATR = 0.1      # EMA20 must be at least 0.1 ATR beyond EMA50 (and rising/falling) for a trend
STALE_DAYS = 7
CRORE = 1e7

WEIGHTS = {"trend": 25.0, "rs": 20.0, "compression": 20.0, "volume": 15.0, "proximity": 10.0, "liquidity": 10.0}


def _clip(x: float | None) -> float:
    if x is None or not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _r2(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else round(float(x), 2)


def _next_session(data: Any, d: date) -> date:
    cal = getattr(data, "calendar", None)
    if cal is not None and hasattr(cal, "next_trading_day"):
        try:
            return cal.next_trading_day(d)
        except Exception:  # noqa: BLE001
            pass
    n = d + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n


def _pct_change(close: pd.Series, n: int) -> float | None:
    if len(close) <= n:
        return None
    base = float(close.iloc[-1 - n])
    return (float(close.iloc[-1]) / base - 1.0) * 100.0 if base > 0 else None


def _stats(df: pd.DataFrame, nifty_ret: float | None) -> dict[str, Any] | None:
    """Every per-symbol number the setups need (None when the data is unusable)."""
    df = df[["open", "high", "low", "close", "volume"]].astype("float64")
    if len(df) < MIN_BARS or df.iloc[-1].isna().any():
        return None
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    e20, e50 = ema(close, 20), ema(close, 50)
    a14 = atr_series(df, 14)
    c, h, lo = float(close.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1])
    atr_v, ema20, ema50 = float(a14.iloc[-1]), float(e20.iloc[-1]), float(e50.iloc[-1])
    if not all(math.isfinite(x) for x in (c, h, lo, atr_v, ema20, ema50)) or atr_v <= 0 or c <= 0:
        return None
    prior = df.iloc[-21:-1]
    ranges = (high - low).iloc[-7:]
    rng = h - lo
    avg_vol = float(vol.iloc[-21:-1].mean())
    vr = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 and math.isfinite(avg_vol) else None
    ret20 = _pct_change(close, 20)
    rs = (ret20 - nifty_ret) if ret20 is not None and nifty_ret is not None else None
    recent = df.iloc[-20:]
    turnover_cr = float((recent["close"] * recent["volume"]).mean()) / CRORE
    ema20_prev = float(e20.iloc[-6]) if len(e20) > 6 and math.isfinite(float(e20.iloc[-6])) else ema20
    return {
        "close": c, "high": h, "low": lo, "atr": atr_v, "ema20": ema20, "ema50": ema50,
        "ema20_slope_atr": (ema20 - ema20_prev) / atr_v,
        "high20": float(prior["high"].max()), "low20": float(prior["low"].min()),
        "range": rng, "nr7": bool(len(ranges) == 7 and rng <= float(ranges.min()) + 1e-9),
        "inside": bool(h <= float(high.iloc[-2]) and lo >= float(low.iloc[-2])),
        "volume_ratio": vr, "rs_20d": rs, "turnover_cr": turnover_cr,
        "pdh": h, "pdl": lo, "last_day": to_ist(pd.Timestamp(df.index[-1]).to_pydatetime()).date(),
    }


def _score(st: dict[str, Any], setup: str, direction: int, trigger: float) -> tuple[float, dict[str, float]]:
    a = st["atr"]
    trend = 0.6 + 0.4 * _clip(direction * st["ema20_slope_atr"] / 0.5)
    rs = 0.5 if st["rs_20d"] is None else _clip((direction * st["rs_20d"] + 2.0) / 10.0)
    comp = 0.5 * st["nr7"] + 0.3 * st["inside"] + 0.2 * _clip((1.0 - st["range"] / a) / 0.5)
    vr = st["volume_ratio"]
    if vr is None:
        volume = 0.0
    elif setup == "PULLBACK":
        volume = _clip((1.2 - vr) / 0.5)
    else:
        volume = _clip((vr - 0.8) / 0.7)
    proximity = _clip(1.0 - abs(trigger - st["close"]) / a / 1.5)
    t = st["turnover_cr"]
    liquidity = _clip((math.log10(t) - 1.0) / 2.0) if t > 0 else 0.0
    parts = {"trend": trend, "rs": rs, "compression": _clip(comp), "volume": volume, "proximity": proximity,
             "liquidity": liquidity}
    return sum(WEIGHTS[k] * v for k, v in parts.items()), parts


def _common_reasons(st: dict[str, Any], direction: int) -> list[str]:
    out: list[str] = []
    if st["nr7"]:
        out.append("NR7: narrowest range of the last 7 sessions (compression)")
    if st["inside"]:
        out.append("inside day (range inside the previous session)")
    if st["volume_ratio"] is not None:
        out.append(f"volume {st['volume_ratio']:.2f}x the 20-day average")
    if st["rs_20d"] is not None:
        side = "stronger" if st["rs_20d"] >= 0 else "weaker"
        out.append(f"20-day relative strength vs NIFTY {st['rs_20d']:+.1f} pts ({side})")
    out.append(f"avg turnover Rs {st['turnover_cr']:.0f} cr/day")
    return out


def _setups(st: dict[str, Any]) -> list[dict[str, Any]]:
    c, h, lo, a, e20, e50 = st["close"], st["high"], st["low"], st["atr"], st["ema20"], st["ema50"]
    hi20, lo20 = st["high20"], st["low20"]
    out: list[dict[str, Any]] = []
    # a real trend, not EMAs stacked by a hair in a flat market
    up = c > e20 > e50 and e20 - e50 >= MIN_EMA_GAP_ATR * a and st["ema20_slope_atr"] > 0
    down = c < e20 < e50 and e50 - e20 >= MIN_EMA_GAP_ATR * a and st["ema20_slope_atr"] < 0
    if up and hi20 - a <= c <= hi20 + a:
        trig = round_tick(max(h, hi20) + 0.1 * a, "up")
        stop = round_tick(max(lo, trig - 1.2 * a), "down")
        reasons = [f"uptrend: close {c:.2f} > EMA20 {e20:.2f} > EMA50 {e50:.2f}",
                   (f"close {(hi20 - c) / a:.2f} ATR below the 20-day high {hi20:.2f}" if c <= hi20
                    else f"closed above the 20-day high {hi20:.2f}")]
        out.append({"setup": "BREAKOUT", "dir": 1, "trigger": trig, "stop": stop, "reasons": reasons})
    if up and lo <= e20 + 0.5 * a and c > e20 and st["volume_ratio"] is not None and st["volume_ratio"] < 1.2:
        trig = round_tick(h + 0.05 * a, "up")
        stop = round_tick(min(lo, e20 - 0.3 * a), "down")
        reasons = [f"uptrend: close {c:.2f} > EMA20 {e20:.2f} > EMA50 {e50:.2f}",
                   f"pulled back to EMA20 (low {lo:.2f}, {(lo - e20) / a:+.2f} ATR from it) and closed above it"]
        out.append({"setup": "PULLBACK", "dir": 1, "trigger": trig, "stop": stop, "reasons": reasons})
    if down and lo20 - a <= c <= lo20 + a:
        trig = round_tick(min(lo, lo20) - 0.1 * a, "down")
        stop = round_tick(min(h, trig + 1.2 * a), "up")
        reasons = [f"downtrend: close {c:.2f} < EMA20 {e20:.2f} < EMA50 {e50:.2f}",
                   (f"close {(c - lo20) / a:.2f} ATR above the 20-day low {lo20:.2f}" if c >= lo20
                    else f"closed below the 20-day low {lo20:.2f}"),
                   "SHORT: intraday only (MIS), squared off the same day"]
        out.append({"setup": "BREAKDOWN", "dir": -1, "trigger": trig, "stop": stop, "reasons": reasons})
    return out


def evaluate_symbol(symbol: str, daily: pd.DataFrame, *, nifty_ret: float | None = None,
                    max_price: float | None = None, for_date: date | None = None) -> dict[str, Any] | None:
    """Best next-day setup for one symbol from its closed daily bars (None = no setup / bad data)."""
    st = _stats(daily, nifty_ret)
    if st is None:
        return None
    best: tuple[float, dict[str, Any], dict[str, float]] | None = None
    for cand in _setups(st):
        d, trig, stop = cand["dir"], cand["trigger"], cand["stop"]
        risk = (trig - stop) * d
        if risk <= 0:
            continue
        target = round_tick(trig + d * 2.0 * risk)
        score, parts = _score(st, cand["setup"], d, trig)
        cand = {**cand, "target": target, "rr": abs(target - trig) / risk}
        if best is None or score > best[0]:
            best = (score, cand, parts)
    if best is None:
        return None
    score, cand, parts = best
    reasons = cand["reasons"] + _common_reasons(st, cand["dir"])
    reasons.append("score " + ", ".join(f"{k} {WEIGHTS[k] * v:.0f}/{WEIGHTS[k]:.0f}" for k, v in parts.items()))
    trig = cand["trigger"]
    return {
        "symbol": symbol.strip().upper(), "setup": cand["setup"], "direction": "LONG" if cand["dir"] > 0 else "SHORT",
        "trigger": _r2(trig), "stop": _r2(cand["stop"]), "target": _r2(cand["target"]), "rr": _r2(cand["rr"]),
        "score": _r2(score), "reasons": reasons,
        "close": _r2(st["close"]), "atr": _r2(st["atr"]), "ema20": _r2(st["ema20"]),
        "pdh": _r2(st["pdh"]), "pdl": _r2(st["pdl"]), "high20": _r2(st["high20"]), "low20": _r2(st["low20"]),
        "volume_ratio": _r2(st["volume_ratio"]), "rs_20d": _r2(st["rs_20d"]),
        "affordable": bool(max_price is None or trig <= max_price),
        "for_date": (for_date or _next_session(None, st["last_day"])).isoformat(),
    }


def build_next_day_plan(data: Any, symbols: list[str], now: datetime, settings: Any, *,
                        nifty_symbol: str = "NIFTY50", max_price: float | None = None,
                        for_date: date | None = None) -> list[dict[str, Any]]:
    """Rank ``symbols`` for the next session (see the module docstring). Never raises.

    Args:
        data: ``MarketDataService`` (``candles(symbol, timeframe, lookback_bars=)`` -> closed bars).
        symbols: candidate symbols (blacklist / sector filters already applied by the caller).
        now: as-of time (call after the 15:30 close so today's bar is included).
        settings: ``Settings`` (``universe.min_price`` / ``max_price`` / ``min_avg_daily_turnover_cr``)
            or None to skip those filters.
        nifty_symbol: index used for relative strength.
        max_price: INTRADAY max affordable price from the risk manager (None = not checked).
        for_date: the session the plan is for (default: next trading day after ``now``'s date).

    Returns:
        Candidate dicts (``plan_json``), highest score first; affordable first on equal scores.
    """
    now = to_ist(now)
    target_day = for_date or _next_session(data, now.date())
    u = getattr(settings, "universe", None)
    nifty_ret: float | None = None
    try:
        idx = data.candles(nifty_symbol, Timeframe.D1, lookback_bars=40)
        if idx is not None and len(idx) > 20:
            nifty_ret = _pct_change(idx["close"].astype("float64"), 20)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[next_day] {nifty_symbol} daily bars unavailable: {exc}")
    out: list[dict[str, Any]] = []
    for raw in dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()):
        try:
            df = data.candles(raw, Timeframe.D1, lookback_bars=LOOKBACK_BARS)
            if df is None or len(df) < MIN_BARS:
                continue
            last_day = to_ist(pd.Timestamp(df.index[-1]).to_pydatetime()).date()
            if (now.date() - last_day).days > STALE_DAYS:
                continue
            last_close = float(df["close"].iloc[-1])
            if u is not None:
                if not np.isfinite(last_close) or last_close < float(u.min_price):
                    continue
                if u.max_price is not None and last_close > float(u.max_price):
                    continue
                recent = df.tail(20)
                turnover_cr = float((recent["close"] * recent["volume"]).mean()) / CRORE
                if turnover_cr < float(u.min_avg_daily_turnover_cr):
                    continue
            cand = evaluate_symbol(raw, df, nifty_ret=nifty_ret, max_price=max_price, for_date=target_day)
            if cand is not None:
                out.append(cand)
        except Exception as exc:  # noqa: BLE001 - one bad symbol never breaks the plan
            logger.warning(f"[next_day] {raw}: {exc}")
    out.sort(key=lambda c: (-(c["score"] or 0.0), not c["affordable"], c["symbol"]))
    return out


__all__ = ["build_next_day_plan", "evaluate_symbol", "WEIGHTS", "MIN_BARS"]
