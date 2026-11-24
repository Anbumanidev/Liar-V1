"""Shared domain types (Pydantic v2). These are the CONTRACT between modules.

Conventions
-----------
* Prices are ``float`` rupees, quantities ``int`` shares, percentages are expressed in
  percent units (``1.0`` == 1 %), never fractions, unless a field name ends in ``_frac``.
* Every datetime is timezone-aware IST (see ``core.clock``).
* OHLCV candles are ``pandas.DataFrame`` objects, NOT Pydantic models. Contract:
  ``DatetimeIndex`` (tz-aware IST, ascending, unique, bar OPEN time), float columns
  ``open, high, low, close, volume``. Only CLOSED bars are ever returned by the data
  layer (no look-ahead). Use ``validate_ohlcv`` to check.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.enums import (
    AgentStatus,
    Direction,
    DrawdownTier,
    EntryType,
    EventType,
    Grade,
    Mode,
    Product,
    Regime,
    RiskRule,
    Signal,
    SLMethod,
    TradeDecision,
    TradeSource,
    TrailingMethod,
)

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the candle DataFrame contract and return it unchanged.

    Raises:
        ValueError: if the frame violates the contract.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV frame must have a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("OHLCV index must be timezone-aware (IST)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be ascending")
    if df.index.has_duplicates:
        raise ValueError("OHLCV index must be unique")
    return df


def empty_ohlcv() -> pd.DataFrame:
    """An empty, contract-compliant candle frame."""
    idx = pd.DatetimeIndex([], tz="Asia/Kolkata", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS}, index=idx)


class _Model(BaseModel):
    model_config = ConfigDict(use_enum_values=False, arbitrary_types_allowed=True, extra="forbid")


# --------------------------------------------------------------------------- market data


class Quote(_Model):
    """Latest price snapshot for one symbol."""

    symbol: str
    ltp: float
    timestamp: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    prev_close: float | None = None
    volume: float | None = None
    bid: float | None = None
    ask: float | None = None
    upper_circuit: float | None = None
    lower_circuit: float | None = None
    source: str = "unknown"

    @property
    def spread_pct(self) -> float | None:
        """Bid/ask spread in percent of mid, if known."""
        if self.bid and self.ask and self.bid > 0 and self.ask >= self.bid:
            mid = (self.bid + self.ask) / 2
            return (self.ask - self.bid) / mid * 100
        return None

    @property
    def change_pct(self) -> float | None:
        if self.prev_close:
            return (self.ltp - self.prev_close) / self.prev_close * 100
        return None


class Headline(_Model):
    """A news headline / announcement, optionally scored by FinBERT."""

    source: str
    title: str
    url: str | None = None
    published_at: datetime
    symbol: str | None = None
    summary: str | None = None
    sentiment_label: Signal | None = None
    sentiment_score: float | None = None  # -1 .. +1 (positive - negative probability)


class MarketEvent(_Model):
    """Corporate or macro event (results, ex-dates, RBI policy, ...)."""

    symbol: str | None = None  # None = market-wide (macro / holiday / index rebalance)
    event_type: EventType
    event_date: date
    title: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    ratio: float | None = None   # split/bonus factor: new_shares_per_old_share (e.g. 2.0 for 1:1 bonus)
    amount: float | None = None  # dividend per share
    source: str = "unknown"


class FundamentalSnapshot(_Model):
    """Daily-cached fundamentals for one symbol. Missing values are None."""

    symbol: str
    as_of: date
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    pe: float | None = None
    sector_pe: float | None = None
    pb: float | None = None
    roe: float | None = None
    roce: float | None = None
    debt_to_equity: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    profit_margin: float | None = None
    promoter_holding: float | None = None
    promoter_holding_change: float | None = None
    promoter_pledge_pct: float | None = None
    in_asm: bool = False
    in_gsm: bool = False
    results_today: bool = False
    corporate_action_today: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class MarketContext(_Model):
    """Broad market state (Market Context Agent input)."""

    as_of: datetime
    nifty_trend: Signal = Signal.NEUTRAL
    nifty_change_pct: float | None = None
    banknifty_trend: Signal = Signal.NEUTRAL
    banknifty_change_pct: float | None = None
    india_vix: float | None = None
    vix_20d_avg: float | None = None
    vix_elevated: bool = False
    advance_decline_ratio: float | None = None
    fii_net_cr: float | None = None
    dii_net_cr: float | None = None
    sector_strength: dict[str, float] = Field(default_factory=dict)  # sector -> % change / RS
    global_cues: dict[str, float] = Field(default_factory=dict)      # e.g. {"SP500": 0.4}
    regime: Regime = Regime.UNDEFINED
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- features


class PatternHit(_Model):
    """Candlestick / price-action pattern detected on a closed bar."""

    name: str
    timeframe: str
    direction: Signal
    strength: int = Field(ge=-100, le=100)  # TA-Lib style (+100 bullish, -100 bearish)
    bar_time: datetime


class Levels(_Model):
    """Pivot / CPR / support-resistance levels for one timeframe."""

    pivot: float | None = None
    r1: float | None = None
    r2: float | None = None
    r3: float | None = None
    s1: float | None = None
    s2: float | None = None
    s3: float | None = None
    cpr_bc: float | None = None
    cpr_tc: float | None = None
    cpr_width_pct: float | None = None
    prev_high: float | None = None
    prev_low: float | None = None
    prev_close: float | None = None
    supports: list[float] = Field(default_factory=list)      # nearest first
    resistances: list[float] = Field(default_factory=list)   # nearest first
    swing_highs: list[float] = Field(default_factory=list)
    swing_lows: list[float] = Field(default_factory=list)


class TimeframeFeatures(_Model):
    """Latest indicator values for one symbol on one timeframe (last CLOSED bar).

    ``values`` keys (None when not computable):
    ``ema_9, ema_20, ema_21, ema_50, ema_200, vwap, rsi_14, macd, macd_signal, macd_hist,
    supertrend, supertrend_dir (+1/-1), adx_14, plus_di, minus_di, bb_upper, bb_mid,
    bb_lower, bb_width_pct, bb_squeeze (0/1), atr_14, atr_pct, atr_expansion (ratio vs 20-bar
    avg ATR), volume, avg_volume_20, rvol, vwap_dev_pct, close, high, low, open,
    trend_structure (+1 HH/HL, -1 LH/LL, 0), inside_bar (0/1), rsi_bull_div (0/1),
    rsi_bear_div (0/1)``
    """

    timeframe: str
    as_of: datetime
    bars: int
    values: dict[str, float | None] = Field(default_factory=dict)
    patterns: list[PatternHit] = Field(default_factory=list)
    levels: Levels | None = None
    regime: Regime = Regime.UNDEFINED

    def v(self, key: str, default: float | None = None) -> float | None:
        """Convenience accessor for ``values``."""
        val = self.values.get(key)
        return default if val is None else val


class StockFeatures(_Model):
    """All deterministic features for one symbol in one mode."""

    symbol: str
    mode: Mode
    as_of: datetime
    ltp: float
    primary_timeframe: str
    timeframes: dict[str, TimeframeFeatures] = Field(default_factory=dict)
    atr: float | None = None                  # ATR(14) of the mode's primary timeframe
    gap_pct: float | None = None              # today's open vs previous close
    opening_range_high: float | None = None   # 09:15-09:30 (intraday)
    opening_range_low: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    prev_close: float | None = None
    rvol: float | None = None                 # relative volume vs same-time average / 20d avg
    delivery_pct: float | None = None
    delivery_pct_avg_20: float | None = None
    avg_daily_volume_20: float | None = None
    avg_daily_turnover_20: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    rs_vs_nifty: float | None = None          # % outperformance vs NIFTY over lookback
    regime: Regime = Regime.UNDEFINED          # regime on primary timeframe
    extra: dict[str, Any] = Field(default_factory=dict)

    def tf(self, timeframe: str) -> TimeframeFeatures | None:
        return self.timeframes.get(timeframe)


class SLCandidate(_Model):
    """Deterministic stop-loss candidate (Stop-Loss Agent picks one; code validates)."""

    method: SLMethod
    price: float
    distance: float           # |entry - price|
    distance_atr: float | None = None
    note: str = ""


# --------------------------------------------------------------------------- strategies


class StrategySignal(_Model):
    """A setup produced by a deterministic strategy module."""

    strategy: str
    mode: Mode
    direction: Direction
    entry_price: float
    entry_type: EntryType = EntryType.LIMIT
    stop_loss: float
    target_1: float
    target_2: float | None = None
    strength: int = Field(default=50, ge=0, le=100)
    regime_fit: bool = True
    valid_until: datetime | None = None
    trailing_rule: TrailingMethod = TrailingMethod.NONE
    max_holding_days: int | None = None
    notes: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def rr(self) -> float:
        """Reward:risk to target_1."""
        risk = abs(self.entry_price - self.stop_loss)
        return abs(self.target_1 - self.entry_price) / risk if risk > 0 else 0.0


class ConfluenceResult(_Model):
    """Deterministic confluence score (section 7.3)."""

    score: float = Field(ge=0, le=100)
    grade: Grade
    components: dict[str, float] = Field(default_factory=dict)  # component -> points
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- agent I/O


class AgentReport(_Model):
    """Output of one Layer-1 sub-agent (section 4)."""

    agent: str
    symbol: str
    mode: Mode
    timestamp: datetime
    signal: Signal = Signal.NEUTRAL
    confidence: int = Field(default=0, ge=0, le=100)
    key_findings: list[str] = Field(default_factory=list)
    data_used: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    tradeable: bool | None = None          # Fundamental agent: TRADEABLE true/false
    status: AgentStatus = AgentStatus.OK
    error: str | None = None
    model: str | None = None
    prompt_version: int | None = None
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)  # agent specific (entry zone, chosen SL, targets ...)

    @property
    def effective_signal(self) -> Signal:
        """FAILED agents vote NEUTRAL (rule 0.4)."""
        return Signal.NEUTRAL if self.status != AgentStatus.OK else self.signal


class DebateSide(_Model):
    stance: str
    top_arguments: list[str] = Field(default_factory=list)
    rebuttals: list[str] = Field(default_factory=list)
    conviction: int = Field(default=0, ge=0, le=100)


class DebateResult(_Model):
    bull: DebateSide | None = None
    bear: DebateSide | None = None
    rounds: int = 0
    transcript: list[dict[str, Any]] = Field(default_factory=list)  # [{round, speaker, content}]
    status: AgentStatus = AgentStatus.OK
    error: str | None = None
    latency_ms: int | None = None


class MetaDecision(_Model):
    """Meta Decision Agent output (section 4, Layer 3). Quantity is NEVER here."""

    decision: TradeDecision
    symbol: str
    mode: Mode
    recommended_mode_reason: str | None = None
    strategy: str | None = None
    entry_price: float | None = None
    entry_type: EntryType = EntryType.LIMIT
    stop_loss: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    trailing_rule: TrailingMethod = TrailingMethod.NONE
    max_holding_days: int | None = None
    confidence: int = Field(default=0, ge=0, le=100)
    time_validity_minutes: int | None = None
    reasons_for: list[str] = Field(default_factory=list)
    reasons_against_considered: list[str] = Field(default_factory=list)
    why_not: str | None = None
    status: AgentStatus = AgentStatus.OK
    error: str | None = None
    model: str | None = None
    prompt_version: int | None = None
    latency_ms: int | None = None

    @classmethod
    def no_trade(cls, symbol: str, mode: Mode, why: str, *, failed: bool = False) -> "MetaDecision":
        """Graceful-degradation default (section 16)."""
        return cls(
            decision=TradeDecision.NO_TRADE,
            symbol=symbol,
            mode=mode,
            why_not=why,
            status=AgentStatus.FAILED if failed else AgentStatus.OK,
            error=why if failed else None,
        )


class StockContext(_Model):
    """Everything known about one symbol for one mode at one instant (agent input)."""

    symbol: str
    mode: Mode
    as_of: datetime
    batch_id: int | None = None
    run_id: int | None = None
    sector: str | None = None
    quote: Quote | None = None
    features: StockFeatures | None = None
    market: MarketContext | None = None
    fundamentals: FundamentalSnapshot | None = None
    headlines: list[Headline] = Field(default_factory=list)
    sentiment_score: float | None = None     # aggregate FinBERT score -1..+1
    events: list[MarketEvent] = Field(default_factory=list)  # events inside the holding window
    strategy_signals: list[StrategySignal] = Field(default_factory=list)
    sl_candidates: list[SLCandidate] = Field(default_factory=list)
    confluence: ConfluenceResult | None = None
    regime: Regime = Regime.UNDEFINED
    capital_summary: dict[str, Any] = Field(default_factory=dict)  # for meta agent (read-only)
    mode_settings: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- risk / execution


class ChargesBreakdown(_Model):
    """Statutory + broker charges for one order or a round trip (section 9)."""

    brokerage: float = 0.0
    stt: float = 0.0
    exchange_txn: float = 0.0
    sebi: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0
    dp: float = 0.0
    ipft: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.brokerage + self.stt + self.exchange_txn + self.sebi + self.stamp_duty
            + self.gst + self.dp + self.ipft,
            4,
        )

    def __add__(self, other: "ChargesBreakdown") -> "ChargesBreakdown":
        return ChargesBreakdown(
            **{k: getattr(self, k) + getattr(other, k) for k in type(self).model_fields}
        )

    def as_dict(self) -> dict[str, float]:
        d = self.model_dump()
        d["total"] = self.total
        return d


class TradeProposal(_Model):
    """A fully specified trade idea submitted to the risk manager (system or manual)."""

    batch_id: int
    symbol: str
    mode: Mode
    direction: Direction
    entry_price: float
    entry_type: EntryType = EntryType.LIMIT
    stop_loss: float
    target_1: float
    target_2: float | None = None
    strategy: str | None = None
    confidence: int = Field(default=0, ge=0, le=100)
    confluence_score: float | None = None
    grade: Grade | None = None
    regime: Regime = Regime.UNDEFINED
    atr: float | None = None                  # ATR of the mode's primary timeframe
    tradeable: bool | None = True             # Fundamental agent verdict
    source: TradeSource = TradeSource.SYSTEM
    sector: str | None = None
    requested_qty: int | None = None          # manual trades / approval-queue modifications
    trailing_rule: TrailingMethod = TrailingMethod.NONE
    max_holding_days: int | None = None
    upper_circuit: float | None = None
    lower_circuit: float | None = None
    ltp: float | None = None
    events_in_window: list[MarketEvent] = Field(default_factory=list)
    decision_id: int | None = None
    approval_id: int | None = None
    created_at: datetime

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def rr(self) -> float:
        r = self.risk_per_share
        return abs(self.target_1 - self.entry_price) / r if r > 0 else 0.0

    @property
    def product(self) -> Product:
        return Product.MIS if self.mode == Mode.INTRADAY else Product.CNC


class RiskCheck(_Model):
    rule: RiskRule
    passed: bool
    detail: str = ""


class RiskVerdict(_Model):
    """Risk manager result. ``approved=False`` always carries the first failing ``rule``."""

    approved: bool
    rule: RiskRule | None = None
    message: str = ""
    qty: int = 0
    max_qty: int = 0
    risk_amount: float = 0.0            # rupees at risk for qty (SL distance)
    effective_risk: float = 0.0         # gap-adjusted risk used for the loss budget
    risk_budget: float = 0.0            # max rupees allowed to risk on this trade
    position_value: float = 0.0
    charges_estimate: ChargesBreakdown | None = None
    expected_net_at_t1: float | None = None
    size_multiplier: float = 1.0
    tier: DrawdownTier = DrawdownTier.NORMAL
    checks: list[RiskCheck] = Field(default_factory=list)

    @classmethod
    def reject(cls, rule: RiskRule, message: str, **kw: Any) -> "RiskVerdict":
        return cls(approved=False, rule=rule, message=message, **kw)


class OpenPositionView(_Model):
    """Read-only snapshot of an open paper position (risk manager / dashboard input)."""

    position_id: int
    symbol: str
    mode: Mode
    product: Product
    direction: Direction
    qty: int
    avg_entry: float
    stop_loss: float
    target_1: float | None = None
    target_2: float | None = None
    ltp: float | None = None
    margin_blocked: float
    entry_charges: float = 0.0
    sector: str | None = None
    opened_at: datetime
    overnight: bool = False

    @property
    def sl_risk(self) -> float:
        """Rupees lost if the SL fills exactly (never negative)."""
        if self.direction == Direction.LONG:
            return max(0.0, (self.avg_entry - self.stop_loss) * self.qty)
        return max(0.0, (self.stop_loss - self.avg_entry) * self.qty)

    @property
    def unrealized_gross(self) -> float:
        if self.ltp is None:
            return 0.0
        sign = 1 if self.direction == Direction.LONG else -1
        return sign * (self.ltp - self.avg_entry) * self.qty


class CapitalSnapshot(_Model):
    """Point-in-time capital state for one batch (sections 6 and 8)."""

    batch_id: int
    as_of: datetime
    starting_capital: float
    base_capital: float
    trading_cash: float
    blocked: float
    reserve: float
    reserve_peak: float
    unrealized_pnl: float = 0.0
    total_equity: float
    hwm: float
    drawdown_pct: float = 0.0
    tier: DrawdownTier = DrawdownTier.NORMAL
    sizing_capital: float
    mode_sizing_capital: dict[str, float] = Field(default_factory=dict)
    mode_blocked: dict[str, float] = Field(default_factory=dict)
    realized_today: float = 0.0
    absolute_floor: float = 0.0
    trailing_floor: float = 0.0
    open_risk: float = 0.0
    loss_budget: float = 0.0
    halted: bool = False

    @property
    def trading_balance(self) -> float:
        """TRADING account = cash + cost basis (margin) of open positions."""
        return self.trading_cash + self.blocked

    @property
    def effective_floor(self) -> float:
        return max(self.absolute_floor, self.trailing_floor)
