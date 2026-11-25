"""Builders for crafted ``StockContext`` objects and scripted LLM answers (agents tests).

Everything is synthetic and deterministic; nothing touches the network or Ollama.

* ``intraday_ctx()`` - RELIANCE-like INTRADAY LONG setup at 100.00 (ATR 0.80 on 5m), with
  stop-loss candidates (valid and invalid), FinBERT-scored headlines, market context and a
  clean fundamentals snapshot. Strategy ``opening_range_breakout``: entry 100, SL 98.8,
  T1 102.0 (1.67R), T2 103.0.
* ``swing_ctx()`` - SWING LONG daily setup at 500.00 (ATR 10 on 1d) with a RESULTS event in
  the holding window.
* ``good_answers()`` - one valid scripted LLM answer per sub-agent (FakeLLMClient format).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from core.clock import IST
from core.enums import (
    AgentStatus,
    Direction,
    EntryType,
    EventType,
    Grade,
    Mode,
    Regime,
    Signal,
    SLMethod,
    TrailingMethod,
)
from core.types import (
    AgentReport,
    ConfluenceResult,
    FundamentalSnapshot,
    Headline,
    Levels,
    MarketContext,
    MarketEvent,
    PatternHit,
    Quote,
    SLCandidate,
    StockContext,
    StockFeatures,
    StrategySignal,
    TimeframeFeatures,
)

NOW = datetime(2026, 11, 24, 10, 30, tzinfo=IST)  # Tuesday, inside the intraday window
SWING_NOW = datetime(2026, 11, 24, 16, 15, tzinfo=IST)  # evening swing scan


def _tf(name: str, as_of: datetime, values: dict[str, float | None], levels: Levels | None = None,
        patterns: list[PatternHit] | None = None, regime: Regime = Regime.TRENDING_UP) -> TimeframeFeatures:
    return TimeframeFeatures(timeframe=name, as_of=as_of, bars=250, values=values, levels=levels,
                             patterns=patterns or [], regime=regime)


def bullish_values(close: float, atr: float, *, vwap: bool = True) -> dict[str, float | None]:
    """Indicator values of a clean uptrend around ``close``."""
    vals: dict[str, float | None] = {
        "close": close, "open": close - 0.3 * atr, "high": close + 0.1 * atr, "low": close - 0.4 * atr,
        "ema_9": close - 0.5 * atr, "ema_20": close - 0.9 * atr, "ema_21": close - 1.0 * atr,
        "ema_50": close - 2.0 * atr, "ema_200": close - 4.0 * atr,
        "rsi_14": 63.0, "macd": 0.4, "macd_signal": 0.3, "macd_hist": 0.1, "supertrend": close - 1.5 * atr,
        "supertrend_dir": 1.0, "adx_14": 29.0, "plus_di": 27.0, "minus_di": 14.0,
        "bb_upper": close + 1.5 * atr, "bb_mid": close - 0.5 * atr, "bb_lower": close - 2.5 * atr,
        "bb_width_pct": 3.2, "bb_squeeze": 0.0, "atr_14": atr, "atr_pct": atr / close * 100,
        "atr_expansion": 1.1, "volume": 18_000.0, "avg_volume_20": 10_000.0, "rvol": 1.8,
        "trend_structure": 1.0, "inside_bar": 0.0, "rsi_bull_div": 0.0, "rsi_bear_div": 0.0,
    }
    if vwap:
        vals["vwap"] = close - 0.6 * atr
        vals["vwap_dev_pct"] = round(0.6 * atr / close * 100, 3)
    return vals


def intraday_levels() -> Levels:
    return Levels(pivot=99.0, r1=101.5, r2=103.0, r3=104.5, s1=98.0, s2=97.0, s3=96.0, cpr_bc=98.8,
                  cpr_tc=99.2, cpr_width_pct=0.4, prev_high=100.8, prev_low=97.5, prev_close=99.0,
                  supports=[99.0, 98.2], resistances=[101.2, 102.5], swing_highs=[101.2], swing_lows=[98.9])


def intraday_signal(**kw: Any) -> StrategySignal:
    base: dict[str, Any] = dict(
        strategy="opening_range_breakout", mode=Mode.INTRADAY, direction=Direction.LONG, entry_price=100.0,
        entry_type=EntryType.MARKET, stop_loss=98.8, target_1=102.0, target_2=103.0, strength=70,
        valid_until=NOW + timedelta(minutes=20), trailing_rule=TrailingMethod.ATR,
        notes=["ORB above 99.80 with volume"],
    )
    base.update(kw)
    return StrategySignal(**base)


def intraday_sl_candidates() -> list[SLCandidate]:
    """Candidates vs entry 100 / ATR 0.8 (VWAP 99.5 is too tight, prior-day 97.5 too wide)."""
    def c(method: SLMethod, price: float, note: str) -> SLCandidate:
        d = abs(100.0 - price)
        return SLCandidate(method=method, price=price, distance=d, distance_atr=round(d / 0.8, 4), note=note)

    return [
        c(SLMethod.VWAP, 99.5, "session VWAP (5m)"),
        c(SLMethod.ATR, 99.2, "1x ATR"),
        c(SLMethod.SWING, 98.9, "latest swing low (5m)"),
        c(SLMethod.ATR, 98.8, "1.5x ATR"),
        c(SLMethod.OPENING_RANGE, 98.6, "opening range low"),
        c(SLMethod.ATR, 98.4, "2x ATR"),
        c(SLMethod.DAILY_STRUCTURE, 97.5, "prior day low"),
    ]


def market_ctx(as_of: datetime = NOW, *, trend: Signal = Signal.BULLISH, vix_elevated: bool = False) -> MarketContext:
    return MarketContext(as_of=as_of, nifty_trend=trend, nifty_change_pct=0.6, banknifty_trend=trend,
                         banknifty_change_pct=0.4, india_vix=13.5, vix_20d_avg=14.0, vix_elevated=vix_elevated,
                         advance_decline_ratio=1.5, fii_net_cr=850.0, dii_net_cr=-120.0,
                         sector_strength={"ENERGY": 0.9, "IT": -0.4, "BANK": 0.2},
                         global_cues={"SP500": 0.5, "NIKKEI": 0.3}, regime=Regime.TRENDING_UP)


def clean_fundamentals(**kw: Any) -> FundamentalSnapshot:
    base: dict[str, Any] = dict(symbol="RELIANCE", as_of=NOW.date(), sector="ENERGY", industry="Oil & Gas",
                                market_cap=1.8e13, pe=22.0, sector_pe=25.0, pb=2.1, roe=18.0, roce=16.0,
                                debt_to_equity=0.4, revenue_growth=12.0, earnings_growth=16.0, profit_margin=11.0,
                                promoter_holding=50.3, promoter_holding_change=0.2, promoter_pledge_pct=0.0)
    base.update(kw)
    return FundamentalSnapshot(**base)


def headlines(as_of: datetime = NOW, symbol: str = "RELIANCE") -> list[Headline]:
    """Two positive headlines in the last 24 h and a negative one 3 days old (outside INTRADAY's window)."""
    return [
        Headline(source="ET Markets", title=f"{symbol} wins large order", symbol=symbol,
                 published_at=as_of - timedelta(hours=3), sentiment_label=Signal.BULLISH, sentiment_score=0.72),
        Headline(source="Moneycontrol", title="Margins improve in Q3", symbol=symbol,
                 published_at=as_of - timedelta(hours=6), sentiment_label=Signal.BULLISH, sentiment_score=0.41),
        Headline(source="Livemint", title="Older negative news", symbol=symbol,
                 published_at=as_of - timedelta(days=3), sentiment_label=Signal.BEARISH, sentiment_score=-0.9),
    ]


def intraday_features() -> StockFeatures:
    lv = intraday_levels()
    pats = [PatternHit(name="CDLENGULFING", timeframe="5m", direction=Signal.BULLISH, strength=100,
                       bar_time=NOW - timedelta(minutes=5))]
    tfs = {
        "5m": _tf("5m", NOW, bullish_values(100.0, 0.8), lv, pats),
        "15m": _tf("15m", NOW, bullish_values(100.0, 1.2), lv),
        "1h": _tf("1h", NOW, bullish_values(100.0, 2.0), lv),
    }
    return StockFeatures(symbol="RELIANCE", mode=Mode.INTRADAY, as_of=NOW, ltp=100.0, primary_timeframe="5m",
                         timeframes=tfs, atr=0.8, gap_pct=0.4, opening_range_high=99.8, opening_range_low=98.6,
                         day_high=100.2, day_low=98.5, prev_close=99.0, rvol=1.8, avg_daily_volume_20=5e6,
                         avg_daily_turnover_20=5e8, week52_high=110.0, week52_low=80.0, rs_vs_nifty=1.5,
                         regime=Regime.TRENDING_UP)


def intraday_ctx(**kw: Any) -> StockContext:
    """INTRADAY LONG context (see module docstring)."""
    base: dict[str, Any] = dict(
        symbol="RELIANCE", mode=Mode.INTRADAY, as_of=NOW, batch_id=1, run_id=7, sector="ENERGY",
        quote=Quote(symbol="RELIANCE", ltp=100.0, timestamp=NOW, bid=99.95, ask=100.05, prev_close=99.0,
                    upper_circuit=110.0, lower_circuit=90.0),
        features=intraday_features(), market=market_ctx(), fundamentals=clean_fundamentals(),
        headlines=headlines(), sentiment_score=0.55, events=[], strategy_signals=[intraday_signal()],
        sl_candidates=intraday_sl_candidates(),
        confluence=ConfluenceResult(score=78.0, grade=Grade.A, components={"trend": 20.0, "volume": 12.0}),
        regime=Regime.TRENDING_UP,
        capital_summary={"trading_balance": 1000.0, "reserve": 0.0, "tier": "NORMAL", "loss_budget": 90.0},
    )
    base.update(kw)
    return StockContext(**base)


# --------------------------------------------------------------------------- swing


def swing_signal(**kw: Any) -> StrategySignal:
    base: dict[str, Any] = dict(
        strategy="swing_pullback_ema", mode=Mode.SWING, direction=Direction.LONG, entry_price=500.0,
        entry_type=EntryType.LIMIT, stop_loss=485.0, target_1=530.0, target_2=545.0, strength=65,
        trailing_rule=TrailingMethod.EMA20_DAILY, max_holding_days=8,
    )
    base.update(kw)
    return StrategySignal(**base)


def swing_features() -> StockFeatures:
    lv = Levels(pivot=495.0, r1=515.0, r2=532.0, s1=488.0, s2=476.0, prev_high=503.0, prev_low=492.0,
                prev_close=498.0, supports=[490.0, 480.0], resistances=[520.0, 540.0], swing_lows=[486.0],
                swing_highs=[521.0])
    tfs = {
        "1h": _tf("1h", SWING_NOW, bullish_values(500.0, 4.0), lv),
        "1d": _tf("1d", SWING_NOW, bullish_values(500.0, 10.0, vwap=False), lv),
    }
    return StockFeatures(symbol="TCS", mode=Mode.SWING, as_of=SWING_NOW, ltp=500.0, primary_timeframe="1d",
                         timeframes=tfs, atr=10.0, day_high=503.0, day_low=492.0, prev_close=496.0,
                         delivery_pct=55.0, delivery_pct_avg_20=42.0, avg_daily_volume_20=2e6,
                         avg_daily_turnover_20=1e9, week52_high=560.0, week52_low=400.0, rs_vs_nifty=2.5,
                         regime=Regime.TRENDING_UP)


def swing_ctx(**kw: Any) -> StockContext:
    """SWING LONG context with a RESULTS event 3 days after entry (blocking)."""
    base: dict[str, Any] = dict(
        symbol="TCS", mode=Mode.SWING, as_of=SWING_NOW, batch_id=1, run_id=8, sector="IT",
        quote=Quote(symbol="TCS", ltp=500.0, timestamp=SWING_NOW, prev_close=496.0),
        features=swing_features(), market=market_ctx(SWING_NOW),
        fundamentals=clean_fundamentals(symbol="TCS", sector="IT"), headlines=headlines(SWING_NOW, "TCS"),
        events=[MarketEvent(symbol="TCS", event_type=EventType.RESULTS, event_date=date(2026, 11, 27),
                            title="Q3 results"),
                MarketEvent(symbol=None, event_type=EventType.MACRO, event_date=date(2026, 12, 4),
                            title="RBI MPC policy")],
        strategy_signals=[swing_signal()],
        sl_candidates=[
            SLCandidate(method=SLMethod.SWING, price=486.0, distance=14.0, distance_atr=1.4, note="swing low"),
            SLCandidate(method=SLMethod.ATR, price=485.0, distance=15.0, distance_atr=1.5, note="1.5x ATR"),
            SLCandidate(method=SLMethod.ATR, price=480.0, distance=20.0, distance_atr=2.0, note="2x ATR"),
        ],
        confluence=ConfluenceResult(score=72.0, grade=Grade.B),
        regime=Regime.TRENDING_UP,
    )
    base.update(kw)
    return StockContext(**base)


# --------------------------------------------------------------------------- scripted answers


def answer(signal: str = "BULLISH", confidence: int = 70, **kw: Any) -> dict[str, Any]:
    """A valid common-field answer plus agent specific keys."""
    out: dict[str, Any] = {"signal": signal, "confidence": confidence,
                           "key_findings": ["finding quoted from the facts"], "reasoning": "short reasoning"}
    out.update(kw)
    return out


def good_answers() -> dict[str, list[dict[str, Any]]]:
    """One valid answer per sub-agent for ``intraday_ctx`` / ``swing_ctx``."""
    return {
        "fundamental": [answer("NEUTRAL", 50, tradeable=True, concerns=[])],
        "technical": [answer("BULLISH", 72, trend_alignment="ALIGNED_UP")],
        "price_action": [answer("BULLISH", 66, key_pattern="CDLENGULFING")],
        "volume_liquidity": [answer("BULLISH", 64, volume_confirms=True, liquidity_ok=True)],
        "market_context": [answer("BULLISH", 60, risk_off=False)],
        "news_sentiment": [answer("BULLISH", 58)],
        "event_risk": [answer("BEARISH", 80, event_risk=True, blocking_events=["E1"])],
        "entry_timing": [answer("BULLISH", 62, entry_type="MARKET", candidate_index=0)],
        "stop_loss": [answer("BULLISH", 60, candidate_index=3, price=98.8)],
        "exit_target": [answer("BULLISH", 61, target_1_index=0, target_2_index=None, trailing_rule="ATR",
                               max_holding_days=0)],
    }


def report(agent: str, signal: Signal = Signal.NEUTRAL, confidence: int = 0, *, mode: Mode = Mode.INTRADAY,
           status: AgentStatus = AgentStatus.OK, tradeable: bool | None = None,
           extra: dict[str, Any] | None = None, findings: list[str] | None = None) -> AgentReport:
    """Hand-made AgentReport for aggregate / debate / meta tests."""
    return AgentReport(agent=agent, symbol="RELIANCE", mode=mode, timestamp=NOW, signal=signal,
                       confidence=confidence, status=status, tradeable=tradeable, extra=dict(extra or {}),
                       key_findings=findings or [f"{agent} finding"])
