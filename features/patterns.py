"""Candlestick patterns + price-action helpers (opening range, gaps, inside bars, structure).

Candlestick detection uses TA-Lib ``CDL*`` functions when ``talib`` imports, otherwise a
pure-python fallback covering engulfing, hammer, shooting star, doji, morning/evening star
and harami. ``inside_bar`` is always detected in pure python (it is not a TA-Lib pattern).

Only CLOSED bars may be passed in (the data layer guarantees it); hits are reported for the
last ``lookback`` bars of the frame, oldest first.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from core.clock import IST, parse_hhmm
from core.enums import Signal, Timeframe
from core.types import PatternHit
from features.swings import fractal_flags, structure_from_pivots

try:  # TA-Lib is optional at runtime; the fallback keeps the system working without it.
    import talib as _talib

    TALIB_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - environment dependent
    _talib = None
    TALIB_AVAILABLE = False
    logger.warning(f"TA-Lib unavailable, using pure-python candlestick fallback: {exc}")

#: TA-Lib function -> (pattern name, fixed direction or None to use the sign of the output).
TALIB_PATTERNS: dict[str, tuple[str, Signal | None]] = {
    "CDLENGULFING": ("engulfing", None),
    "CDLHAMMER": ("hammer", Signal.BULLISH),
    "CDLINVERTEDHAMMER": ("inverted_hammer", Signal.BULLISH),
    "CDLSHOOTINGSTAR": ("shooting_star", Signal.BEARISH),
    "CDLDOJI": ("doji", Signal.NEUTRAL),
    "CDLMORNINGSTAR": ("morning_star", Signal.BULLISH),
    "CDLEVENINGSTAR": ("evening_star", Signal.BEARISH),
    "CDLHARAMI": ("harami", None),
    "CDLPIERCING": ("piercing", Signal.BULLISH),
    "CDLDARKCLOUDCOVER": ("dark_cloud_cover", Signal.BEARISH),
    "CDL3WHITESOLDIERS": ("three_white_soldiers", Signal.BULLISH),
    "CDL3BLACKCROWS": ("three_black_crows", Signal.BEARISH),
    "CDLMARUBOZU": ("marubozu", None),
    "CDLHANGINGMAN": ("hanging_man", Signal.BEARISH),
    "CDLDRAGONFLYDOJI": ("dragonfly_doji", Signal.BULLISH),
    "CDLGRAVESTONEDOJI": ("gravestone_doji", Signal.BEARISH),
}

# Fallback thresholds (documented, intentionally conservative).
DOJI_BODY_FRAC = 0.1          # body <= 10 % of range
SHADOW_BODY_MULT = 2.0        # hammer/shooting star: long shadow >= 2x body
SHORT_SHADOW_FRAC = 0.25      # opposite shadow <= 25 % of range
TREND_BARS = 3                # prior trend: close[i-1] vs close[i-1-TREND_BARS]
STAR_BODY_FRAC = 0.3          # star body <= 30 % of the first candle's body
BODY_AVG_BARS = 10            # "long body" = above the 10-bar average body


def _direction_from_value(value: int, fixed: Signal | None) -> Signal:
    if fixed is not None:
        return fixed
    if value > 0:
        return Signal.BULLISH
    if value < 0:
        return Signal.BEARISH
    return Signal.NEUTRAL


def _signed_strength(value: int, direction: Signal) -> int:
    """Strength in [-100, 100] whose sign follows ``direction`` (0 for NEUTRAL patterns).

    TA-Lib returns e.g. +100 for a gravestone doji although the pattern is bearish; the
    stored strength is normalised so bullish > 0, bearish < 0, neutral == 0.
    """
    mag = min(100, abs(int(value)))
    if direction == Signal.BULLISH:
        return mag
    if direction == Signal.BEARISH:
        return -mag
    return 0


def _to_dt(ts: Any) -> datetime:
    dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


# --------------------------------------------------------------------------- fallback detectors


def fallback_patterns(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    """Pure-python candlestick detectors. Returns ``name -> int array`` (+100/-100/0, TA-Lib style).

    Rules (``body=|C-O|``, ``range=H-L``, ``avg_body`` = mean body of the prior 10 bars):

    * doji: ``range > 0`` and ``body <= 0.1 * range`` (+100, neutral).
    * hammer: lower shadow >= 2x body, upper shadow <= 25 % of range, body > 0, prior
      ``TREND_BARS`` closes falling (+100).
    * shooting_star: upper shadow >= 2x body, lower shadow <= 25 % of range, prior closes
      rising (-100).
    * engulfing: opposite-colour previous candle whose body is fully covered by the current
      body (strictly larger body) -> +100 bullish / -100 bearish.
    * harami: previous body longer than ``avg_body``, current body strictly inside it ->
      +100 when the previous candle was bearish, -100 when bullish.
    * morning_star: long bearish candle, then a small-body star whose body sits below the
      first close, then a bullish candle closing above the first body's midpoint (+100);
      evening_star mirrors (-100).
    """
    o, h, lo, c = (np.asarray(x, dtype="float64") for x in (o, h, lo, c))
    n = len(c)
    body = np.abs(c - o)
    rng = h - lo
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - lo
    avg_body = pd.Series(body).shift(1).rolling(BODY_AVG_BARS, min_periods=3).mean().to_numpy()
    res = {k: np.zeros(n, dtype=int) for k in
           ("doji", "hammer", "shooting_star", "engulfing", "harami", "morning_star", "evening_star")}
    for i in range(n):
        if rng[i] > 0 and body[i] <= DOJI_BODY_FRAC * rng[i]:
            res["doji"][i] = 100
        prior_down = i - 1 - TREND_BARS >= 0 and c[i - 1] < c[i - 1 - TREND_BARS]
        prior_up = i - 1 - TREND_BARS >= 0 and c[i - 1] > c[i - 1 - TREND_BARS]
        if rng[i] > 0 and body[i] > 0:
            if lower[i] >= SHADOW_BODY_MULT * body[i] and upper[i] <= SHORT_SHADOW_FRAC * rng[i] and prior_down:
                res["hammer"][i] = 100
            if upper[i] >= SHADOW_BODY_MULT * body[i] and lower[i] <= SHORT_SHADOW_FRAC * rng[i] and prior_up:
                res["shooting_star"][i] = -100
        if i >= 1:
            p_bear, p_bull = c[i - 1] < o[i - 1], c[i - 1] > o[i - 1]
            bull, bear = c[i] > o[i], c[i] < o[i]
            if p_bear and bull and o[i] <= c[i - 1] and c[i] >= o[i - 1] and body[i] > body[i - 1]:
                res["engulfing"][i] = 100
            elif p_bull and bear and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body[i] > body[i - 1]:
                res["engulfing"][i] = -100
            ab = avg_body[i - 1] if not np.isnan(avg_body[i - 1]) else np.nan
            if (not np.isnan(ab) and body[i - 1] > ab and body[i] < body[i - 1]
                    and max(o[i], c[i]) < max(o[i - 1], c[i - 1])
                    and min(o[i], c[i]) > min(o[i - 1], c[i - 1])):
                if p_bear:
                    res["harami"][i] = 100
                elif p_bull:
                    res["harami"][i] = -100
        if i >= 2:
            ab = avg_body[i - 2]
            long_first = not np.isnan(ab) and body[i - 2] > ab
            small_star = body[i - 1] <= STAR_BODY_FRAC * body[i - 2]
            mid_first = (o[i - 2] + c[i - 2]) / 2.0
            if long_first and small_star:
                if (c[i - 2] < o[i - 2] and max(o[i - 1], c[i - 1]) <= c[i - 2]
                        and c[i] > o[i] and c[i] > mid_first):
                    res["morning_star"][i] = 100
                if (c[i - 2] > o[i - 2] and min(o[i - 1], c[i - 1]) >= c[i - 2]
                        and c[i] < o[i] and c[i] < mid_first):
                    res["evening_star"][i] = -100
    return res


_FALLBACK_DIRECTIONS: dict[str, Signal | None] = {
    "doji": Signal.NEUTRAL,
    "hammer": Signal.BULLISH,
    "shooting_star": Signal.BEARISH,
    "engulfing": None,
    "harami": None,
    "morning_star": Signal.BULLISH,
    "evening_star": Signal.BEARISH,
}


def _talib_outputs(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> dict[str, tuple[np.ndarray, Signal | None]]:
    out: dict[str, tuple[np.ndarray, Signal | None]] = {}
    for fn_name, (name, fixed) in TALIB_PATTERNS.items():
        fn: Callable[..., np.ndarray] | None = getattr(_talib, fn_name, None)
        if fn is None:
            logger.warning(f"TA-Lib lacks {fn_name}; skipped")
            continue
        try:
            out[name] = (np.asarray(fn(o, h, lo, c), dtype=int), fixed)
        except Exception as exc:  # TA-Lib raises generic exceptions on bad input
            logger.warning(f"TA-Lib {fn_name} failed: {exc}")
    return out


def detect_patterns(
    df: pd.DataFrame,
    timeframe: str | Timeframe,
    lookback: int = 3,
    *,
    use_talib: bool | None = None,
) -> list[PatternHit]:
    """Detect candlestick patterns on the last ``lookback`` closed bars.

    Args:
        df: OHLCV frame of CLOSED bars (full history helps TA-Lib's body averages).
        timeframe: label stored on each hit (``"5m"``, ``"1d"`` ...).
        lookback: number of most recent bars to report hits for.
        use_talib: force (True) or disable (False) TA-Lib; default = use it when available.

    Returns:
        PatternHit list ordered by bar time then name. ``strength`` is signed by direction
        (bullish > 0, bearish < 0, neutral 0). Includes ``inside_bar`` (NEUTRAL, strength 0)
        price-action hits.
    """
    if df is None or len(df) == 0 or lookback <= 0:
        return []
    tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    lo = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    want_talib = TALIB_AVAILABLE if use_talib is None else (use_talib and TALIB_AVAILABLE)
    if want_talib:
        outputs = _talib_outputs(o, h, lo, c)
    else:
        outputs = {name: (arr, _FALLBACK_DIRECTIONS[name]) for name, arr in fallback_patterns(o, h, lo, c).items()}

    n = len(df)
    start = max(0, n - lookback)
    hits: list[PatternHit] = []
    for i in range(start, n):
        bar_time = _to_dt(df.index[i])
        for name, (arr, fixed) in outputs.items():
            val = int(arr[i])
            if val == 0:
                continue
            direction = _direction_from_value(val, fixed)
            hits.append(PatternHit(
                name=name,
                timeframe=tf,
                direction=direction,
                strength=_signed_strength(val, direction),
                bar_time=bar_time,
            ))
        if i >= 1 and h[i] <= h[i - 1] and lo[i] >= lo[i - 1]:
            hits.append(PatternHit(name="inside_bar", timeframe=tf, direction=Signal.NEUTRAL,
                                   strength=0, bar_time=bar_time))
    hits.sort(key=lambda p: (p.bar_time, p.name))
    return hits


# --------------------------------------------------------------------------- price action


def _infer_bar_minutes(df: pd.DataFrame) -> float | None:
    if len(df) < 2:
        return None
    diffs = pd.Series(df.index).diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return float(diffs.min().total_seconds() / 60.0)


def opening_range(
    df_intraday: pd.DataFrame,
    day: date,
    start: str = "09:15",
    end: str = "09:30",
    bar_minutes: float | None = None,
) -> tuple[float, float] | None:
    """High/low of the opening range ``[start, end)`` on ``day``.

    Returns None when there is no data for the window, when the bar size is larger than the
    window, or when the window is not complete yet (its last bar has not closed - prevents
    look-ahead on a partial range). ``bar_minutes`` overrides the bar size inferred from the
    index spacing (needed when the frame holds a single bar).
    """
    if df_intraday is None or len(df_intraday) == 0:
        return None
    t0 = datetime.combine(day, parse_hhmm(start), tzinfo=IST)
    t1 = datetime.combine(day, parse_hhmm(end), tzinfo=IST)
    if t1 <= t0:
        raise ValueError("opening range end must be after start")
    window = df_intraday[(df_intraday.index >= t0) & (df_intraday.index < t1)]
    if window.empty:
        return None
    minutes = bar_minutes if bar_minutes is not None else _infer_bar_minutes(df_intraday)
    if minutes is None or minutes > (t1 - t0).total_seconds() / 60.0:
        return None
    last_close = window.index[-1].to_pydatetime() + timedelta(minutes=minutes)
    if last_close < t1:
        return None
    if window.index[0] > t0:
        return None
    return float(window["high"].max()), float(window["low"].min())


def gap_pct(daily_df: pd.DataFrame, intraday_df: pd.DataFrame | None = None) -> float | None:
    """Gap of the latest session's open vs the previous session's close, in percent.

    The latest session is the most recent date present in ``intraday_df`` (when given and
    non-empty) or in ``daily_df``. Its open comes from the first intraday bar of that date
    (preferred) or the daily bar; the previous close is the last daily close strictly before
    that date (falling back to the last intraday close of an earlier date).
    """
    has_intra = intraday_df is not None and len(intraday_df) > 0
    has_daily = daily_df is not None and len(daily_df) > 0
    if not has_intra and not has_daily:
        return None
    if has_intra:
        session = intraday_df.index[-1].date()
        today_bars = intraday_df[intraday_df.index.date == session]
        today_open = float(today_bars["open"].iloc[0])
    else:
        session = daily_df.index[-1].date()
        today_open = float(daily_df["open"].iloc[-1])
    prev_close: float | None = None
    if has_daily:
        prior = daily_df[daily_df.index.date < session]
        if len(prior):
            prev_close = float(prior["close"].iloc[-1])
    if prev_close is None and has_intra:
        prior_i = intraday_df[intraday_df.index.date < session]
        if len(prior_i):
            prev_close = float(prior_i["close"].iloc[-1])
    if prev_close is None or prev_close == 0:
        return None
    return (today_open - prev_close) / prev_close * 100.0


def inside_bar(df: pd.DataFrame) -> bool:
    """True when the last bar's high/low are inside the previous bar's range."""
    if df is None or len(df) < 2:
        return False
    h, lo = df["high"].to_numpy(dtype="float64"), df["low"].to_numpy(dtype="float64")
    return bool(h[-1] <= h[-2] and lo[-1] >= lo[-2])


def trend_structure(df: pd.DataFrame, swing_lookback: int = 2) -> int:
    """+1 higher-high/higher-low, -1 lower-high/lower-low, 0 otherwise.

    Uses the last two CONFIRMED fractal swings (``left = right = swing_lookback``).
    """
    if df is None or len(df) < 2 * swing_lookback + 1:
        return 0
    h, lo = df["high"].to_numpy(dtype="float64"), df["low"].to_numpy(dtype="float64")
    is_h, is_l = fractal_flags(h, lo, swing_lookback, swing_lookback)
    return structure_from_pivots([float(x) for x in h[is_h]], [float(x) for x in lo[is_l]])


__all__ = [
    "TALIB_AVAILABLE",
    "TALIB_PATTERNS",
    "detect_patterns",
    "fallback_patterns",
    "gap_pct",
    "inside_bar",
    "opening_range",
    "trend_structure",
]

