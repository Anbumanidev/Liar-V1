"""FeatureBuilder: turns candles into :class:`core.types.StockFeatures` for one symbol / mode.

``build`` fetches candles from the market data service (``data.candles(symbol, timeframe,
lookback_bars=...)``, CLOSED bars only as of the service clock) and delegates to the pure
:func:`build_from_frames`, which is what unit tests and the replay engine exercise directly.

No look-ahead: :func:`build_from_frames` additionally drops any bar that had not closed at
``as_of`` (intraday: ``min(open + timeframe, 15:30)``; daily: 15:30 of its date; weekly:
15:30 of the week's last trading day), so a misbehaving provider cannot leak future data.

Field rules (all deterministic):

* ``ltp`` - close of the most recently CLOSED bar across the supplied frames.
* ``atr`` / ``rvol`` / ``regime`` - from the primary timeframe.
* ``gap_pct`` - session open vs previous daily close (None when the session at ``as_of`` has
  no bars yet, e.g. pre-market).
* ``opening_range_high/low`` - 09:15-09:30 (configurable) from the finest intraday frame for
  the ``as_of`` session; None until the window has fully closed.
* ``day_high/day_low`` - the ``as_of`` session's range (intraday bars, else today's daily bar).
* ``prev_close`` - last daily close strictly before the ``as_of`` session.
* ``avg_daily_volume_20`` / ``avg_daily_turnover_20`` - mean over the last 20 closed daily bars
  (turnover = close x volume, rupees).
* ``week52_high/low`` - over closed daily bars within 365 days before ``as_of``.
* ``rs_vs_nifty`` - ``(stock return - NIFTY return) * 100`` over the last 20 sessions
  (55 for POSITIONAL), on common dates.
* Levels per timeframe: pivots/CPR from the previous session (intraday), previous week (1d),
  previous month (1wk), plus swings and clustered S/R from that timeframe.
* ``extra``: ``prev_day_high``, ``prev_day_low``, ``rs_lookback``, ``ltp_bar_time``,
  ``missing_timeframes``, ``inside_cpr`` (primary timeframe).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd
from loguru import logger

from config.schema import ModeConfig
from core.clock import IST, to_ist
from core.enums import TIMEFRAME_MINUTES, Mode, Regime, Timeframe
from core.errors import DataUnavailableError
from core.types import MarketContext, StockFeatures, TimeframeFeatures
from features.indicators import VALUE_KEYS, compute_indicators, is_intraday_timeframe, latest_values
from features.levels import compute_levels, inside_cpr
from features.patterns import detect_patterns, gap_pct, opening_range
from features.regime import detect_regime
from storage.audit import record_error

INDEX_SYMBOL = "NIFTY50"
SESSION_CLOSE = time(15, 30)

#: Bars requested per timeframe by :meth:`FeatureBuilder.build` (enough for EMA200 warm-up and
#: ~20 sessions of intraday time-of-day rvol).
DEFAULT_LOOKBACK_BARS: dict[str, int] = {
    Timeframe.M1.value: 1500,
    Timeframe.M5.value: 1500,
    Timeframe.M15.value: 600,
    Timeframe.H1.value: 400,
    Timeframe.D1.value: 400,
    Timeframe.W1.value: 260,
}

RS_LOOKBACK_DEFAULT = 20
RS_LOOKBACK_POSITIONAL = 55
AVG_DAILY_BARS = 20
PATTERN_LOOKBACK = 3


class CandleSource(Protocol):
    """The subset of ``data.service.MarketDataService`` the builder needs."""

    def candles(self, symbol: str, timeframe: Timeframe, lookback_bars: int = 300,
                start: datetime | None = None) -> pd.DataFrame:
        """Closed candles as of the service clock."""
        ...


# --------------------------------------------------------------------------- helpers


def _tf_str(tf: str | Timeframe) -> str:
    return tf.value if isinstance(tf, Timeframe) else str(tf)


def _week_last_day(d: date, calendar: Any | None) -> date:
    if calendar is not None and hasattr(calendar, "week_last_trading_day"):
        try:
            return calendar.week_last_trading_day(d)
        except Exception as exc:  # calendar is an external dependency; degrade to Friday
            logger.warning(f"calendar.week_last_trading_day failed ({exc}); assuming Friday")
    return d + timedelta(days=4 - d.weekday()) if d.weekday() <= 4 else d


def bar_close_times(index: pd.DatetimeIndex, timeframe: str, calendar: Any | None = None) -> list[datetime]:
    """Close time of every bar (see module docstring)."""
    out: list[datetime] = []
    tf = _tf_str(timeframe)
    intraday = is_intraday_timeframe(tf)
    minutes = TIMEFRAME_MINUTES.get(Timeframe(tf)) if tf in {t.value for t in Timeframe} else None
    for ts in index:
        dt = to_ist(pd.Timestamp(ts).to_pydatetime())
        session_end = datetime.combine(dt.date(), SESSION_CLOSE, tzinfo=IST)
        if intraday and minutes:
            out.append(min(dt + timedelta(minutes=minutes), session_end))
        elif tf == Timeframe.W1.value:
            out.append(datetime.combine(_week_last_day(dt.date(), calendar), SESSION_CLOSE, tzinfo=IST))
        else:
            out.append(session_end)
    return out


def closed_only(df: pd.DataFrame | None, timeframe: str, as_of: datetime, calendar: Any | None = None) -> pd.DataFrame:
    """Return ``df`` sorted with every bar that had not closed at ``as_of`` removed."""
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    closes = bar_close_times(df.index, timeframe, calendar)
    mask = np.array([c <= as_of for c in closes], dtype=bool)
    return df[mask]


def _finest_intraday(frames: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame] | None:
    best: tuple[int, str, pd.DataFrame] | None = None
    for tf, df in frames.items():
        if not is_intraday_timeframe(tf) or df is None or len(df) == 0:
            continue
        minutes = TIMEFRAME_MINUTES[Timeframe(tf)]
        if best is None or minutes < best[0]:
            best = (minutes, tf, df)
    return (best[1], best[2]) if best else None


def _infer_intraday_tf(df: pd.DataFrame) -> str:
    """Timeframe label of an intraday frame from its smallest bar spacing (default 1m)."""
    if len(df) >= 2:
        diffs = pd.Series(df.index).diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if not diffs.empty:
            step = int(diffs.min().total_seconds() // 60)
            for t, m in TIMEFRAME_MINUTES.items():
                if m == step and is_intraday_timeframe(t):
                    return t.value
    return Timeframe.M1.value


def relative_strength(stock_close: pd.Series, index_close: pd.Series, lookback: int) -> float | None:
    """``(stock return - index return) * 100`` over ``lookback`` sessions on common dates."""
    if stock_close is None or index_close is None or len(stock_close) == 0 or len(index_close) == 0:
        return None
    s = pd.Series(stock_close.to_numpy(dtype="float64"), index=pd.Index(stock_close.index.date))
    i = pd.Series(index_close.to_numpy(dtype="float64"), index=pd.Index(index_close.index.date))
    s = s[~s.index.duplicated(keep="last")]
    i = i[~i.index.duplicated(keep="last")]
    joined = pd.concat([s.rename("s"), i.rename("i")], axis=1, join="inner").dropna()
    if len(joined) < lookback + 1:
        return None
    a, b = joined.iloc[-lookback - 1], joined.iloc[-1]
    if a["s"] <= 0 or a["i"] <= 0:
        return None
    return float(((b["s"] / a["s"] - 1.0) - (b["i"] / a["i"] - 1.0)) * 100.0)


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def empty_timeframe_features(timeframe: str, as_of: datetime) -> TimeframeFeatures:
    """TimeframeFeatures with every value None (no data)."""
    return TimeframeFeatures(timeframe=timeframe, as_of=as_of, bars=0, values={k: None for k in VALUE_KEYS})


def timeframe_features(
    df: pd.DataFrame,
    timeframe: str,
    *,
    daily: pd.DataFrame | None,
    price: float,
    as_of: datetime,
    session_date: date,
    market: MarketContext | None = None,
    pattern_lookback: int = PATTERN_LOOKBACK,
) -> TimeframeFeatures:
    """Compute indicators, patterns, levels and regime for one timeframe frame."""
    if df is None or len(df) == 0:
        return empty_timeframe_features(timeframe, as_of)
    ind = compute_indicators(df, timeframe)
    values = latest_values(ind)
    patterns = detect_patterns(df, timeframe, lookback=pattern_lookback)
    levels = compute_levels(daily, df, price, timeframe=timeframe, session_date=session_date)
    tff = TimeframeFeatures(
        timeframe=timeframe,
        as_of=to_ist(pd.Timestamp(df.index[-1]).to_pydatetime()),
        bars=len(df),
        values=values,
        patterns=patterns,
        levels=levels,
    )
    tff.regime = detect_regime(
        tff,
        index_trend=market.nifty_trend if market is not None else None,
        vix_elevated=bool(market.vix_elevated) if market is not None else False,
        inside_cpr=inside_cpr(values.get("close"), levels),
    )
    return tff


# --------------------------------------------------------------------------- pure core


def cumulative_rvol(intraday: pd.DataFrame, session_date: Any, sessions: int = 10, min_sessions: int = 3
                    ) -> float | None:
    """Today's volume so far / the average volume of the prior sessions up to the same time of day.

    A single 5-minute bar's relative volume swings wildly (ONGC 0.24 -> 1.23 on consecutive bars on
    2026-11-26) and made the confluence volume component ~0 all day; the cumulative ratio answers
    the actual question - is this stock trading more than usual today?
    """
    try:
        dates = np.array(intraday.index.date)
        today = intraday[dates == session_date]
        if today.empty:
            return None
        last = today.index[-1]
        cut = last.hour * 60 + last.minute
        tod = np.array(intraday.index.hour * 60 + intraday.index.minute)
        prior_days = sorted({d for d in dates if d < session_date})[-sessions:]
        totals = [float(intraday.loc[(dates == d) & (tod <= cut), "volume"].sum()) for d in prior_days]
        totals = [x for x in totals if x > 0]
        if len(totals) < min_sessions:
            return None
        avg = sum(totals) / len(totals)
        return round(float(today["volume"].sum()) / avg, 4) if avg > 0 else None
    except Exception:  # noqa: BLE001 - advisory feature
        return None


def build_from_frames(
    symbol: str,
    mode: Mode,
    frames: dict[str, pd.DataFrame],
    daily: pd.DataFrame | None,
    as_of: datetime,
    primary_timeframe: str | Timeframe,
    index_daily: pd.DataFrame | None = None,
    market: MarketContext | None = None,
    intraday_base: pd.DataFrame | None = None,
    *,
    calendar: Any | None = None,
    opening_range_window: tuple[str, str] = ("09:15", "09:30"),
    pattern_lookback: int = PATTERN_LOOKBACK,
    rs_lookback: int | None = None,
) -> StockFeatures:
    """Pure feature computation from candle frames (see module docstring).

    Args:
        symbol: NSE ticker.
        mode: trading mode the features are for.
        frames: ``{timeframe label: candles}`` for the mode's timeframes.
        daily: daily candles (defaults to ``frames["1d"]``).
        as_of: evaluation instant (IST); bars not closed by then are ignored.
        primary_timeframe: the mode's primary timeframe.
        index_daily: NIFTY50 daily candles for relative strength.
        market: optional market context (index trend / VIX feed the regime).
        intraday_base: optional finer intraday frame (e.g. 5m) for opening range / day range;
            defaults to the finest intraday frame in ``frames``.
        calendar: optional NSE calendar (``week_last_trading_day``) for weekly bar closes.

    Raises:
        DataUnavailableError: when no closed bar exists in any frame (no price at all).
    """
    as_of = to_ist(as_of)
    primary = _tf_str(primary_timeframe)
    session_date = as_of.date()

    clean: dict[str, pd.DataFrame] = {}
    for tf_key, df in frames.items():
        tf = _tf_str(tf_key)
        clean[tf] = closed_only(df, tf, as_of, calendar) if df is not None else None  # type: ignore[assignment]
    daily_src = daily if daily is not None and len(daily) else clean.get(Timeframe.D1.value)
    daily_c = closed_only(daily_src, Timeframe.D1.value, as_of, calendar) if daily_src is not None else None
    index_c = closed_only(index_daily, Timeframe.D1.value, as_of, calendar) if index_daily is not None else None

    base_tf: str | None = None
    base: pd.DataFrame | None = None
    if intraday_base is not None and len(intraday_base):
        base_tf = _infer_intraday_tf(intraday_base)
        base = closed_only(intraday_base, base_tf, as_of, calendar)
    else:
        found = _finest_intraday(clean)
        if found is not None:
            base_tf, base = found

    # ---- ltp: latest closed bar across all frames
    best: tuple[datetime, float, datetime] | None = None
    candidates: list[tuple[str, pd.DataFrame | None]] = list(clean.items())
    candidates.append((Timeframe.D1.value, daily_c))
    if base is not None and base_tf is not None:
        candidates.append((base_tf, base))
    for tf, df in candidates:
        if df is None or len(df) == 0:
            continue
        close_t = bar_close_times(df.index[-1:], tf, calendar)[0]
        if best is None or close_t > best[0]:
            best = (close_t, float(df["close"].iloc[-1]), to_ist(pd.Timestamp(df.index[-1]).to_pydatetime()))
    if best is None:
        raise DataUnavailableError(f"{symbol}: no closed candles available as of {as_of.isoformat()}")
    ltp = best[1]

    # ---- per-timeframe features
    timeframes: dict[str, TimeframeFeatures] = {}
    missing: list[str] = []
    for tf, df in clean.items():
        if df is None or len(df) == 0:
            missing.append(tf)
            timeframes[tf] = empty_timeframe_features(tf, as_of)
            continue
        try:
            timeframes[tf] = timeframe_features(
                df, tf, daily=daily_c, price=ltp, as_of=as_of, session_date=session_date,
                market=market, pattern_lookback=pattern_lookback,
            )
        except Exception as exc:  # one bad timeframe must not kill the whole symbol
            record_error(None, "features.builder", exc, {"symbol": symbol, "timeframe": tf})
            timeframes[tf] = empty_timeframe_features(tf, as_of)
            missing.append(tf)

    prim = timeframes.get(primary)
    if prim is None:
        missing.append(primary)

    # ---- session facts
    has_daily = daily_c is not None and len(daily_c) > 0
    daily_dates = np.array(daily_c.index.date) if has_daily else np.array([])
    today_intra = None
    if base is not None and len(base):
        today_intra = base[np.array(base.index.date) == session_date]
        if today_intra.empty:
            today_intra = None
    daily_today = daily_c[daily_dates == session_date] if has_daily else None
    if daily_today is not None and daily_today.empty:
        daily_today = None

    gap: float | None = None
    if today_intra is not None:
        gap = gap_pct(daily_c if has_daily else None, base[np.array(base.index.date) <= session_date])
    elif daily_today is not None:
        gap = gap_pct(daily_c)

    or_high = or_low = None
    if today_intra is not None:
        rng = opening_range(base, session_date, *opening_range_window)
        if rng is not None:
            or_high, or_low = rng

    day_high = day_low = None
    if today_intra is not None:
        day_high, day_low = float(today_intra["high"].max()), float(today_intra["low"].min())
    elif daily_today is not None:
        day_high, day_low = float(daily_today["high"].iloc[-1]), float(daily_today["low"].iloc[-1])

    prev_close = prev_day_high = prev_day_low = None
    if has_daily:
        prior = daily_c[daily_dates < session_date]
        if len(prior):
            prev_close = float(prior["close"].iloc[-1])
            prev_day_high = float(prior["high"].iloc[-1])
            prev_day_low = float(prior["low"].iloc[-1])
    if prev_close is None and base is not None and len(base):
        prior_i = base[np.array(base.index.date) < session_date]
        if len(prior_i):
            last_day = prior_i.index[-1].date()
            pd_bars = prior_i[np.array(prior_i.index.date) == last_day]
            prev_close = float(pd_bars["close"].iloc[-1])
            prev_day_high = float(pd_bars["high"].max())
            prev_day_low = float(pd_bars["low"].min())

    avg_vol = avg_turnover = w52_high = w52_low = None
    if has_daily:
        last20 = daily_c.iloc[-AVG_DAILY_BARS:]
        if len(last20) >= AVG_DAILY_BARS:
            avg_vol = _f(last20["volume"].mean())
            avg_turnover = _f((last20["close"] * last20["volume"]).mean())
        year = daily_c[daily_c.index >= as_of - timedelta(days=365)]
        if len(year):
            w52_high, w52_low = _f(year["high"].max()), _f(year["low"].min())

    rs_n = rs_lookback or (RS_LOOKBACK_POSITIONAL if mode == Mode.POSITIONAL else RS_LOOKBACK_DEFAULT)
    rs = relative_strength(daily_c["close"], index_c["close"], rs_n) if has_daily and index_c is not None and len(index_c) else None

    rvol_cum = cumulative_rvol(base, session_date) if base is not None and len(base) else None

    prim_inside_cpr = False
    if prim is not None:
        prim_inside_cpr = inside_cpr(prim.values.get("close"), prim.levels)

    return StockFeatures(
        symbol=symbol,
        mode=mode,
        as_of=as_of,
        ltp=ltp,
        primary_timeframe=primary,
        timeframes=timeframes,
        atr=prim.values.get("atr_14") if prim else None,
        gap_pct=_f(gap),
        opening_range_high=or_high,
        opening_range_low=or_low,
        day_high=day_high,
        day_low=day_low,
        prev_close=prev_close,
        rvol=prim.values.get("rvol") if prim else None,
        avg_daily_volume_20=avg_vol,
        avg_daily_turnover_20=avg_turnover,
        week52_high=w52_high,
        week52_low=w52_low,
        rs_vs_nifty=_f(rs),
        regime=prim.regime if prim else Regime.UNDEFINED,
        extra={
            "prev_day_high": prev_day_high,
            "prev_day_low": prev_day_low,
            "rs_lookback": rs_n,
            "ltp_bar_time": best[2].isoformat(),
            "missing_timeframes": sorted(set(missing)),
            "inside_cpr": prim_inside_cpr,
            "index_trend": (market.nifty_trend.value if market is not None else None),
            "rvol_cum": rvol_cum,
        },
    )


# --------------------------------------------------------------------------- service wrapper


class FeatureBuilder:
    """Fetches candles from the market data service and builds :class:`StockFeatures`."""

    def __init__(
        self,
        data: CandleSource,
        calendar: Any | None = None,
        *,
        lookback_bars: dict[str, int] | None = None,
        index_symbol: str = INDEX_SYMBOL,
        opening_range_window: tuple[str, str] = ("09:15", "09:30"),
        pattern_lookback: int = PATTERN_LOOKBACK,
    ) -> None:
        self.data = data
        self.calendar = calendar
        self.lookback_bars = {**DEFAULT_LOOKBACK_BARS, **(lookback_bars or {})}
        self.index_symbol = index_symbol
        self.opening_range_window = opening_range_window
        self.pattern_lookback = pattern_lookback
        self._index_cache: tuple[date, pd.DataFrame] | None = None

    def _fetch(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame | None:
        try:
            df = self.data.candles(symbol, timeframe, lookback_bars=self.lookback_bars.get(timeframe.value, 300))
        except Exception as exc:  # data layer should not raise, but never trust it blindly
            record_error(None, "features.builder", exc, {"symbol": symbol, "timeframe": timeframe.value})
            return None
        if df is None or len(df) == 0:
            logger.warning(f"features: no {timeframe.value} candles for {symbol}")
            return None
        return df

    def _index_daily(self, as_of: datetime) -> pd.DataFrame | None:
        if self._index_cache is not None and self._index_cache[0] == as_of.date():
            return self._index_cache[1]
        df = self._fetch(self.index_symbol, Timeframe.D1)
        if df is not None:
            self._index_cache = (as_of.date(), df)
        return df

    def build(
        self,
        symbol: str,
        mode: Mode,
        mode_cfg: ModeConfig,
        as_of: datetime,
        market: MarketContext | None = None,
    ) -> StockFeatures:
        """Fetch the mode's timeframes (+ daily, + NIFTY50 daily) and build features.

        Raises:
            DataUnavailableError: when no candles at all could be obtained for ``symbol``.
        """
        frames: dict[str, pd.DataFrame] = {}
        for tf in mode_cfg.timeframes:
            df = self._fetch(symbol, tf)
            frames[tf.value] = df if df is not None else pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], dtype="float64")
        daily = frames.get(Timeframe.D1.value)
        if daily is None or len(daily) == 0:
            daily = self._fetch(symbol, Timeframe.D1)
        intraday_base = None
        if mode in (Mode.INTRADAY, Mode.BTST) and Timeframe.M5.value not in frames:
            intraday_base = self._fetch(symbol, Timeframe.M5)
        index_daily = None if symbol == self.index_symbol else self._index_daily(as_of)
        return build_from_frames(
            symbol, mode, frames, daily, as_of, mode_cfg.primary_timeframe,
            index_daily=index_daily if index_daily is not None else (daily if symbol == self.index_symbol else None),
            market=market, intraday_base=intraday_base, calendar=self.calendar,
            opening_range_window=self.opening_range_window, pattern_lookback=self.pattern_lookback,
        )

    def build_from_frames(self, *args: Any, **kwargs: Any) -> StockFeatures:
        """Instance alias of the module-level :func:`build_from_frames` (uses this builder's calendar)."""
        kwargs.setdefault("calendar", self.calendar)
        kwargs.setdefault("opening_range_window", self.opening_range_window)
        kwargs.setdefault("pattern_lookback", self.pattern_lookback)
        return build_from_frames(*args, **kwargs)


__all__ = [
    "CandleSource",
    "DEFAULT_LOOKBACK_BARS",
    "FeatureBuilder",
    "bar_close_times",
    "build_from_frames",
    "closed_only",
    "relative_strength",
    "timeframe_features",
]

