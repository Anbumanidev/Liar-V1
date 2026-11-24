"""Typed settings schema (Pydantic v2).

Two kinds of configuration exist:

* ``HardCaps`` - ``config/hard_caps.yaml``. Editable ONLY on disk. Never from the dashboard.
  Some invariants are hard-coded on top of the file (paper-only, no overnight shorts,
  SL-tighten-only, no averaging down, MAX_LOSS_PCT <= 10): if the file tries to relax
  them the system refuses to start.
* ``Settings`` - everything else. Seeded from the YAML files in ``config/``, then stored
  and versioned in the ``settings_versions`` table (see ``config.store``). The dashboard
  edits Settings; every edit must pass ``validate_against_caps``.

Units: percentages are percent units (1.0 == 1 %). Times are "HH:MM" IST strings.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.clock import parse_hhmm
from core.enums import EmailType, ExecutionPolicy, Grade, Mode, SelectionPolicy, Timeframe, TrailingMethod

#: Absolute ceiling for MAX_LOSS_PCT. hard_caps.yaml can lower it, never raise it.
ABSOLUTE_MAX_LOSS_PCT: float = 10.0

AGENT_NAMES: tuple[str, ...] = (
    "fundamental",
    "technical",
    "price_action",
    "volume_liquidity",
    "market_context",
    "news_sentiment",
    "event_risk",
    "entry_timing",
    "stop_loss",
    "exit_target",
)
DEBATE_AGENTS: tuple[str, ...] = ("bull", "bear")
META_AGENT: str = "meta"
ALL_LLM_AGENTS: tuple[str, ...] = AGENT_NAMES + DEBATE_AGENTS + (META_AGENT,)


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _check_hhmm(v: str) -> str:
    parse_hhmm(v)
    return v


# =========================================================================== hard caps


class HardCaps(_Cfg):
    """config/hard_caps.yaml - the only settings the dashboard can NOT change."""

    paper_only: bool = True
    max_loss_pct: float = Field(default=10.0, gt=0, le=ABSOLUTE_MAX_LOSS_PCT)
    max_mis_leverage: float = Field(default=5.0, ge=1.0)
    max_risk_per_trade_pct: float = Field(default=2.0, gt=0, le=5.0)
    max_daily_loss_limit_pct: float = Field(default=5.0, gt=0, le=10.0)
    max_position_pct: float = Field(default=100.0, gt=0, le=100.0)
    max_overnight_exposure_pct: float = Field(default=100.0, ge=0, le=100.0)
    min_rr_floor: float = Field(default=1.0, ge=0.5)
    max_slippage_pct: float = Field(default=1.0, ge=0)
    allow_overnight_shorts: bool = False
    sl_tighten_only: bool = True
    allow_averaging_down: bool = False

    @model_validator(mode="after")
    def _hard_coded_invariants(self) -> "HardCaps":
        errors = []
        if self.paper_only is not True:
            errors.append("paper_only must be true (PAPER TRADING ONLY)")
        if self.allow_overnight_shorts is not False:
            errors.append("allow_overnight_shorts must be false (no overnight shorts in Indian cash market)")
        if self.sl_tighten_only is not True:
            errors.append("sl_tighten_only must be true (stop-loss may only move closer to price)")
        if self.allow_averaging_down is not False:
            errors.append("allow_averaging_down must be false")
        if errors:
            raise ValueError("; ".join(errors))
        return self


# =========================================================================== modes


class TimeWindow(_Cfg):
    start: str
    end: str

    _v_start = field_validator("start")(classmethod(lambda cls, v: _check_hhmm(v)))
    _v_end = field_validator("end")(classmethod(lambda cls, v: _check_hhmm(v)))

    @model_validator(mode="after")
    def _ordered(self) -> "TimeWindow":
        if parse_hhmm(self.start) >= parse_hhmm(self.end):
            raise ValueError(f"window start {self.start} must be before end {self.end}")
        return self


class ModeConfig(_Cfg):
    """Per-mode pluggable configuration (section 3)."""

    enabled: bool = False
    allocation_pct: float = Field(default=0.0, ge=0, le=100)
    execution_policy: ExecutionPolicy = ExecutionPolicy.MANUAL_APPROVE
    approval_timeout_minutes: int | None = Field(default=None, ge=1, le=24 * 60)
    approval_until_next_session: str | None = None  # e.g. "09:45" for SWING/POSITIONAL
    timeframes: list[Timeframe]
    primary_timeframe: Timeframe
    entry_windows: list[TimeWindow] = Field(default_factory=list)
    max_trades_per_day: int = Field(default=3, ge=0, le=50)
    min_holding_days: int = Field(default=0, ge=0)
    max_holding_days: int = Field(default=0, ge=0, le=180)
    confidence_threshold: int = Field(default=70, ge=0, le=100)
    min_confluence: float = Field(default=60.0, ge=0, le=100)
    min_rr: float = Field(default=2.0, ge=0.5, le=10)
    min_grade: Grade = Grade.B
    agent_weights: dict[str, float] = Field(default_factory=dict)
    agents_enabled: dict[str, bool] = Field(default_factory=dict)
    universe: str = "NIFTY50"  # NIFTY50 | NIFTY100 | NIFTY200 | FNO | CUSTOM
    time_stop_minutes: int | None = Field(default=None, ge=5, le=375)
    time_stop_min_r: float = Field(default=0.5, ge=0, le=5)
    trailing: TrailingMethod = TrailingMethod.NONE
    trail_atr_mult: float = Field(default=1.5, gt=0, le=10)
    partial_exit_at_r: float = Field(default=1.0, gt=0, le=10)
    partial_exit_pct: float = Field(default=50.0, ge=0, le=100)
    move_sl_to_breakeven_at_partial: bool = True
    exit_before_events: bool = True
    consecutive_loss_limit: int = Field(default=3, ge=1, le=20)
    consecutive_loss_pause_minutes: int = Field(default=60, ge=0)
    consecutive_loss_pause_days: int = Field(default=0, ge=0)
    max_symbols_per_cycle: int = Field(default=10, ge=1, le=200)

    @field_validator("approval_until_next_session")
    @classmethod
    def _v_until(cls, v: str | None) -> str | None:
        return _check_hhmm(v) if v else v

    @model_validator(mode="after")
    def _primary_in_timeframes(self) -> "ModeConfig":
        if self.primary_timeframe not in self.timeframes:
            raise ValueError("primary_timeframe must be one of timeframes")
        if self.min_holding_days > self.max_holding_days and self.max_holding_days > 0:
            raise ValueError("min_holding_days must be <= max_holding_days")
        return self


class ModesSettings(_Cfg):
    selection_policy: SelectionPolicy = SelectionPolicy.FIXED
    modes: dict[Mode, ModeConfig]

    @model_validator(mode="after")
    def _allocations(self) -> "ModesSettings":
        missing = [m for m in Mode if m not in self.modes]
        if missing:
            raise ValueError(f"missing mode configs: {missing}")
        total = sum(cfg.allocation_pct for cfg in self.modes.values())
        if total > 100.0 + 1e-9:
            raise ValueError(f"mode allocations sum to {total:.2f}% (> 100%)")
        return self

    def enabled_modes(self) -> list[Mode]:
        return [m for m, c in self.modes.items() if c.enabled]


# =========================================================================== risk


class DrawdownTierSettings(_Cfg):
    caution_pct: float = Field(default=3.0, gt=0)
    defensive_pct: float = Field(default=6.0, gt=0)
    caution_size_mult: float = Field(default=0.5, gt=0, le=1)
    defensive_size_mult: float = Field(default=0.25, gt=0, le=1)
    caution_grades: list[Grade] = Field(default_factory=lambda: [Grade.A_PLUS, Grade.A])
    defensive_grades: list[Grade] = Field(default_factory=lambda: [Grade.A_PLUS])
    defensive_max_open_positions: int = Field(default=1, ge=0)
    defensive_allow_new_overnight: bool = False


class RiskSettings(_Cfg):
    """config/risk.yaml (section 5 + section 8)."""

    risk_per_trade_pct: float = Field(default=1.0, gt=0)
    max_open_positions: int = Field(default=5, ge=1)
    max_position_pct: float = Field(default=40.0, gt=0, le=100)
    small_capital_threshold: float = Field(default=10_000.0, ge=0)
    small_max_open_positions: int = Field(default=2, ge=1)
    small_max_position_pct: float = Field(default=60.0, gt=0, le=100)
    max_positions_per_sector: int = Field(default=1, ge=1)
    mis_leverage: float = Field(default=1.0, ge=1.0)
    daily_loss_limit_pct: float = Field(default=3.0, gt=0)
    intraday_no_entry_before: str = "09:30"
    intraday_no_entry_after: str = "14:45"
    intraday_squareoff: str = "15:15"
    conversion_cutoff: str = "15:10"
    max_overnight_exposure_pct: float = Field(default=50.0, ge=0, le=100)
    gap_buffer_pct: float = Field(default=3.0, ge=0, le=20)
    avoid_events_in_window: bool = True
    avoid_event_types: list[str] = Field(
        default_factory=lambda: ["RESULTS", "BOARD_MEETING", "EX_DIVIDEND", "EX_DATE", "SPLIT", "BONUS", "RIGHTS"]
    )
    circuit_buffer_pct: float = Field(default=2.0, ge=0, le=10)
    sl_min_atr: float = Field(default=0.3, gt=0)
    sl_max_atr: float = Field(default=3.0, gt=0)
    max_charges_pct_of_value: float = Field(default=1.0, gt=0, le=10)
    max_correlation: float = Field(default=0.7, gt=0, le=1)
    correlation_lookback_days: int = Field(default=20, ge=5)
    loss_budget_divisor: float = Field(default=3.0, ge=1)
    max_loss_pct: float = Field(default=10.0, gt=0, le=ABSOLUTE_MAX_LOSS_PCT)
    hard_stop_pct: float = Field(default=9.0, gt=0)
    drawdown: DrawdownTierSettings = Field(default_factory=DrawdownTierSettings)
    intraday_daily_profit_lock_pct: float = Field(default=2.0, gt=0)
    intraday_profit_giveback_pct: float = Field(default=50.0, gt=0, le=100)
    stale_price_seconds: int = Field(default=300, ge=10)

    _v_times = field_validator(
        "intraday_no_entry_before", "intraday_no_entry_after", "intraday_squareoff", "conversion_cutoff"
    )(classmethod(lambda cls, v: _check_hhmm(v)))

    @model_validator(mode="after")
    def _consistency(self) -> "RiskSettings":
        if self.hard_stop_pct >= self.max_loss_pct:
            raise ValueError("hard_stop_pct must be below max_loss_pct (buffer for slippage/gaps)")
        if not (self.drawdown.caution_pct < self.drawdown.defensive_pct < self.hard_stop_pct):
            raise ValueError("drawdown tiers must satisfy caution_pct < defensive_pct < hard_stop_pct")
        if self.sl_min_atr >= self.sl_max_atr:
            raise ValueError("sl_min_atr must be < sl_max_atr")
        return self


# =========================================================================== capital


class CapitalSettings(_Cfg):
    """config/capital.yaml (section 6)."""

    starting_capital: float = Field(default=1000.0, gt=0)
    profit_lock_pct: float = Field(default=30.0, ge=0, le=100)
    reserve_protected_pct: float = Field(default=50.0, ge=0, le=100)
    sizing_increase_threshold_pct: float = Field(default=10.0, ge=0)
    sizing_increase_days: int = Field(default=3, ge=1)
    sizing_step_mode: str = Field(default="TO_TRADING", pattern="^(TO_TRADING|STEP_PCT)$")
    sizing_step_pct: float = Field(default=10.0, gt=0)


class EvaluationSettings(_Cfg):
    """Self-evaluation (section 7.6)."""

    auto_disable_min_trades: int = Field(default=30, ge=5)
    kelly_enabled: bool = False
    kelly_fraction: float = Field(default=0.25, gt=0, le=1)
    kelly_min_trades: int = Field(default=50, ge=10)


# =========================================================================== charges


class ProductCharges(_Cfg):
    """Charges model for one product (MIS or CNC). Rates in percent of turnover."""

    brokerage_model: str = Field(default="PCT_CAP", pattern="^(FLAT|PCT_CAP|ZERO)$")
    brokerage_flat: float = Field(default=20.0, ge=0)          # per executed order
    brokerage_pct: float = Field(default=0.03, ge=0)           # % of order value (PCT_CAP)
    brokerage_cap: float = Field(default=20.0, ge=0)           # max per order (PCT_CAP)
    stt_buy_pct: float = Field(default=0.0, ge=0)
    stt_sell_pct: float = Field(default=0.025, ge=0)
    exchange_txn_pct: float = Field(default=0.00297, ge=0)     # NSE transaction charge
    sebi_per_crore: float = Field(default=10.0, ge=0)          # Rs per 1 crore turnover
    stamp_buy_pct: float = Field(default=0.003, ge=0)
    gst_pct: float = Field(default=18.0, ge=0)                 # on brokerage + exchange + SEBI (+ DP)
    dp_charge_per_scrip: float = Field(default=0.0, ge=0)      # CNC sell day, per scrip, before GST
    ipft_pct: float = Field(default=0.0001, ge=0)              # NSE investor protection fund (0.0001%)
    round_stt_stamp: bool = True                               # STT/stamp rounded to nearest rupee


class ChargesSettings(_Cfg):
    """config/charges.yaml. VERIFY CURRENT RATES before relying on results."""

    last_verified: date
    note: str = "Verify current statutory rates and your broker's tariff; update last_verified."
    mis: ProductCharges
    cnc: ProductCharges


# =========================================================================== engine / models / agents


class EngineSettings(_Cfg):
    poll_interval_seconds: float = Field(default=3.0, ge=1, le=10)
    monitor_interval_seconds: int = Field(default=60, ge=5, le=600)
    slippage_pct: float = Field(default=0.05, ge=0)
    data_provider: str = Field(default="yfinance", pattern="^(yfinance|angelone|upstox|replay)$")
    fallback_providers: list[str] = Field(default_factory=lambda: ["yfinance"])
    llm_backend: str = Field(default="ollama", pattern="^(ollama|rules)$")
    max_parallel_symbols: int = Field(default=2, ge=1, le=16)
    max_parallel_agents: int = Field(default=4, ge=1, le=16)
    use_os_cert_store: bool = True
    heartbeat_stale_seconds: int = Field(default=30, ge=5)


class ModelProfile(_Cfg):
    reasoning: str   # Bull, Bear, Meta
    sub_agent: str   # Layer-1 agents


class ModelsSettings(_Cfg):
    """config/models.yaml (section 2)."""

    ollama_host: str = "http://127.0.0.1:11434"
    active_profile: str = "low"
    profiles: dict[str, ModelProfile]
    assignments: dict[str, str] = Field(default_factory=dict)  # agent -> model override
    finbert_model: str = "ProsusAI/finbert"
    finbert_device: str = "auto"  # auto | cpu | cuda
    temperature: float = Field(default=0.2, ge=0, le=2)
    num_ctx: int = Field(default=8192, ge=1024)
    request_timeout_seconds: int = Field(default=120, ge=5)
    max_retries: int = Field(default=2, ge=0, le=5)
    keep_alive: str = "15m"
    reasoning_think: bool = False
    latency_warn_seconds: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _profile_exists(self) -> "ModelsSettings":
        if self.active_profile not in self.profiles:
            raise ValueError(f"active_profile {self.active_profile!r} not in profiles")
        unknown = set(self.assignments) - set(ALL_LLM_AGENTS)
        if unknown:
            raise ValueError(f"unknown agents in assignments: {sorted(unknown)}")
        return self

    def model_for(self, agent: str) -> str:
        """Resolve the Ollama model for ``agent`` (explicit assignment > profile role)."""
        if agent in self.assignments and self.assignments[agent]:
            return self.assignments[agent]
        profile = self.profiles[self.active_profile]
        return profile.reasoning if agent in DEBATE_AGENTS + (META_AGENT,) else profile.sub_agent


class AgentsSettings(_Cfg):
    enabled: dict[str, bool] = Field(default_factory=lambda: {a: True for a in ALL_LLM_AGENTS})
    debate_rounds: int = Field(default=2, ge=0, le=5)
    debate_enabled: bool = True
    prompt_versions: dict[str, int] = Field(default_factory=dict)  # agent -> pinned version (absent = latest)

    @field_validator("enabled")
    @classmethod
    def _known(cls, v: dict[str, bool]) -> dict[str, bool]:
        unknown = set(v) - set(ALL_LLM_AGENTS)
        if unknown:
            raise ValueError(f"unknown agents: {sorted(unknown)}")
        return v


class StrategyConfig(_Cfg):
    enabled: bool = True
    modes: list[Mode] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class StrategiesSettings(_Cfg):
    strategies: dict[str, StrategyConfig] = Field(default_factory=dict)


# =========================================================================== schedule / notifications / universe


class ScheduleSettings(_Cfg):
    """config/schedule.yaml (section 10). All IST, trading days only."""

    premarket: str = "08:30"
    preopen_check: str = "09:08"
    watchlist_build: str = "09:00"
    opening_range_start: str = "09:15"
    opening_range_end: str = "09:30"
    cnc_entry_window: TimeWindow = Field(default_factory=lambda: TimeWindow(start="09:20", end="10:30"))
    intraday_cycle_start: str = "09:30"
    intraday_cycle_end: str = "14:45"
    intraday_cycle_minutes: int = Field(default=5, ge=1, le=60)
    btst_scan_window: TimeWindow = Field(default_factory=lambda: TimeWindow(start="14:30", end="15:15"))
    btst_scan_minutes: int = Field(default=15, ge=1, le=60)
    intraday_squareoff: str = "15:15"
    cnc_time_stop: str = "15:20"
    eod_processing: str = "15:30"
    eod_report: str = "15:45"
    swing_scan: str = "16:00"
    swing_scan_deadline: str = "18:00"
    approval_timeout_check_seconds: int = Field(default=15, ge=5)
    holiday_overrides: list[date] = Field(default_factory=list)       # extra non-trading days
    special_sessions: list[date] = Field(default_factory=list)        # trading on otherwise-holiday dates

    _v_times = field_validator(
        "premarket", "preopen_check", "watchlist_build", "opening_range_start", "opening_range_end",
        "intraday_cycle_start", "intraday_cycle_end", "intraday_squareoff", "cnc_time_stop",
        "eod_processing", "eod_report", "swing_scan", "swing_scan_deadline",
    )(classmethod(lambda cls, v: _check_hhmm(v)))

    @model_validator(mode="after")
    def _within_market(self) -> "ScheduleSettings":
        sq = parse_hhmm(self.intraday_squareoff)
        if not (parse_hhmm("09:15") < sq <= parse_hhmm("15:25")):
            raise ValueError("intraday_squareoff must be within NSE hours (09:15-15:25)")
        if parse_hhmm(self.intraday_cycle_end) > sq:
            raise ValueError("intraday_cycle_end must be <= intraday_squareoff")
        return self


class NotificationSettings(_Cfg):
    enabled_types: dict[EmailType, bool] = Field(default_factory=lambda: {t: True for t in EmailType})
    recipients: list[str] = Field(default_factory=list)
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None
    urgent_ignores_quiet_hours: bool = True
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=5.0, ge=0)
    subject_prefix: str = "[PAPER]"


class UniverseSettings(_Cfg):
    custom_symbols: list[str] = Field(default_factory=list)
    whitelist: list[str] = Field(default_factory=list)
    blacklist: list[str] = Field(default_factory=list)
    whitelist_only: bool = False
    min_price: float = Field(default=20.0, ge=0)
    max_price: float | None = None
    min_avg_daily_turnover_cr: float = Field(default=5.0, ge=0)
    max_spread_pct: float = Field(default=0.3, ge=0)
    excluded_sectors: list[str] = Field(default_factory=list)

    @field_validator("custom_symbols", "whitelist", "blacklist")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        return sorted({s.strip().upper() for s in v if s.strip()})


class ReplaySettings(_Cfg):
    btst_short_delivery_probability: float = Field(default=0.02, ge=0, le=1)
    seed: int = 42
    step_minutes: int = Field(default=1, ge=1, le=60)


# =========================================================================== root


class Settings(_Cfg):
    """All dashboard-editable settings. Snapshot stored per version in settings_versions."""

    modes: ModesSettings
    risk: RiskSettings
    capital: CapitalSettings
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    charges: ChargesSettings
    engine: EngineSettings = Field(default_factory=EngineSettings)
    models: ModelsSettings
    agents: AgentsSettings = Field(default_factory=AgentsSettings)
    strategies: StrategiesSettings = Field(default_factory=StrategiesSettings)
    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    replay: ReplaySettings = Field(default_factory=ReplaySettings)

    def mode(self, mode: Mode) -> ModeConfig:
        return self.modes.modes[mode]
