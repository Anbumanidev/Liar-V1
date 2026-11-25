"""Shared enumerations used across every module.

All enums are ``StrEnum`` so they serialise to plain strings in JSON / SQLite
and compare equal to their string value.
"""

from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    """Trading mode (section 3)."""

    INTRADAY = "INTRADAY"
    BTST = "BTST"
    SWING = "SWING"
    POSITIONAL = "POSITIONAL"


class Product(StrEnum):
    """Broker product type. MIS = intraday margin, CNC = delivery."""

    MIS = "MIS"
    CNC = "CNC"


class Direction(StrEnum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class OrderSide(StrEnum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Paper order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


# Entry type requested by the meta agent / user has the same values as OrderType.
EntryType = OrderType


class OrderPurpose(StrEnum):
    """Why an order exists."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    PARTIAL_EXIT = "PARTIAL_EXIT"


class OrderStatus(StrEnum):
    """Paper order lifecycle."""

    PENDING = "PENDING"      # created, waiting for next tick/candle (market) or price-through (limit)
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class Signal(StrEnum):
    """Sub-agent vote."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TradeDecision(StrEnum):
    """Meta agent decision."""

    BUY = "BUY"
    SELL_SHORT = "SELL_SHORT"
    NO_TRADE = "NO_TRADE"


class ExecutionPolicy(StrEnum):
    """Per-mode execution policy (section 3)."""

    AUTO = "AUTO"
    MANUAL_APPROVE = "MANUAL_APPROVE"
    SIGNALS_ONLY = "SIGNALS_ONLY"


class SelectionPolicy(StrEnum):
    """Mode selection policy (section 3)."""

    FIXED = "FIXED"
    AUTO = "AUTO"


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"
    TIMEOUT = "TIMEOUT"


class ExitReason(StrEnum):
    """Why (part of) a position was closed (section 12)."""

    SL = "SL"
    TARGET = "TARGET"
    TRAIL = "TRAIL"
    TIME = "TIME"
    SQUAREOFF = "SQUAREOFF"
    GAP = "GAP"
    EVENT = "EVENT"
    HALT = "HALT"
    MANUAL = "MANUAL"
    KILL_SWITCH = "KILL_SWITCH"
    BATCH_CLOSE = "BATCH_CLOSE"


class TradeSource(StrEnum):
    SYSTEM = "SYSTEM"
    MANUAL = "MANUAL"


class BatchStatus(StrEnum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    HALTED = "HALTED"
    CLOSING = "CLOSING"   # old batch whose CNC positions run to completion
    CLOSED = "CLOSED"


class Regime(StrEnum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNDEFINED = "UNDEFINED"


class Grade(StrEnum):
    """Confluence grade. C = below the minimum grade (not tradeable)."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"


class DrawdownTier(StrEnum):
    """Drawdown tiers from the high-water mark (section 8)."""

    NORMAL = "NORMAL"        # 0-3 %
    CAUTION = "CAUTION"      # 3-6 %  size x0.5, A/A+ only
    DEFENSIVE = "DEFENSIVE"  # 6-9 %  size x0.25, A+ only, max 1 open, no new overnight
    HALT = "HALT"            # >= 9 % or floor hit


class AgentStatus(StrEnum):
    OK = "OK"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class LedgerEntryType(StrEnum):
    """Capital ledger movements (section 6)."""

    INITIAL_CAPITAL = "INITIAL_CAPITAL"
    BLOCK = "BLOCK"                    # cash -> blocked (position opened)
    RELEASE = "RELEASE"                # blocked -> cash (position closed, cost basis returned)
    REALIZED_PNL = "REALIZED_PNL"      # net realized P&L credited/debited to cash
    PROFIT_LOCK = "PROFIT_LOCK"        # TRADING -> RESERVE (EOD)
    LOSS_ABSORB = "LOSS_ABSORB"        # informational EOD marker (amount 0 movement)
    REFILL = "REFILL"                  # RESERVE -> TRADING (capped, EOD)
    MANUAL_TRANSFER = "MANUAL_TRANSFER"  # RESERVE -> TRADING (confirmed by owner)
    MANUAL_ADJUSTMENT = "MANUAL_ADJUSTMENT"
    SIZING_CHANGE = "SIZING_CHANGE"    # informational, amount = delta of sizing capital
    DIVIDEND = "DIVIDEND"              # counted as realized P&L
    CORPORATE_ACTION = "CORPORATE_ACTION"  # informational


class CapitalAccountName(StrEnum):
    TRADING = "TRADING"
    BLOCKED = "BLOCKED"
    RESERVE = "RESERVE"
    SIZING = "SIZING"


class CommandType(StrEnum):
    """Dashboard -> engine control commands (section 14.2)."""

    ENGINE_START = "ENGINE_START"
    ENGINE_PAUSE = "ENGINE_PAUSE"
    ENGINE_RESUME = "ENGINE_RESUME"
    ENGINE_STOP = "ENGINE_STOP"
    MODE_PAUSE = "MODE_PAUSE"
    MODE_RESUME = "MODE_RESUME"
    KILL_SWITCH = "KILL_SWITCH"
    RELOAD_SETTINGS = "RELOAD_SETTINGS"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MODIFY_APPROVE = "MODIFY_APPROVE"
    BULK_REJECT = "BULK_REJECT"
    MANUAL_TRADE = "MANUAL_TRADE"
    ANALYZE_NOW = "ANALYZE_NOW"
    CLOSE_POSITION = "CLOSE_POSITION"
    TIGHTEN_SL = "TIGHTEN_SL"
    UPDATE_TARGETS = "UPDATE_TARGETS"
    UPDATE_TRAILING = "UPDATE_TRAILING"
    CONVERT_POSITION = "CONVERT_POSITION"
    UPDATE_HOLDING = "UPDATE_HOLDING"
    SET_EVENT_PIN = "SET_EVENT_PIN"
    RESUME_HALT = "RESUME_HALT"
    RESERVE_TRANSFER = "RESERVE_TRANSFER"
    NEW_BATCH = "NEW_BATCH"
    CLOSE_BATCH = "CLOSE_BATCH"
    REENABLE_STRATEGY = "REENABLE_STRATEGY"
    SEND_TEST_EMAIL = "SEND_TEST_EMAIL"
    RUN_JOB = "RUN_JOB"


class CommandStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    REJECTED = "REJECTED"   # refused by risk manager / validation; result explains the rule


class EngineState(StrEnum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    HALTED = "HALTED"


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"
    W1 = "1wk"


TIMEFRAME_MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
    Timeframe.D1: 375,       # one NSE session (09:15-15:30)
    Timeframe.W1: 375 * 5,
}


class RunType(StrEnum):
    PREMARKET = "PREMARKET"
    WATCHLIST = "WATCHLIST"
    INTRADAY_CYCLE = "INTRADAY_CYCLE"
    BTST_SCAN = "BTST_SCAN"
    SWING_SCAN = "SWING_SCAN"
    POSITIONAL_SCAN = "POSITIONAL_SCAN"
    ANALYZE_NOW = "ANALYZE_NOW"
    EOD = "EOD"
    REPLAY = "REPLAY"


class RejectionSource(StrEnum):
    """Who/what rejected a setup (shadow tracking, section 7.6)."""

    META = "META"                  # meta agent said NO_TRADE / below confidence
    RISK = "RISK"                  # risk manager veto
    USER = "USER"                  # owner rejected in approval queue
    TIMEOUT = "TIMEOUT"            # approval expired
    SIGNALS_ONLY = "SIGNALS_ONLY"  # mode policy: never executed
    FILTER = "FILTER"              # confluence/strategy filter before meta


class ShadowOutcome(StrEnum):
    OPEN = "OPEN"
    NOT_TRIGGERED = "NOT_TRIGGERED"
    SL = "SL"
    TARGET_1 = "TARGET_1"
    TARGET_2 = "TARGET_2"
    TIME = "TIME"


class EmailType(StrEnum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    EOD = "EOD"
    SWING_SCAN = "SWING_SCAN"
    URGENT = "URGENT"
    TEST = "TEST"


class EventType(StrEnum):
    RESULTS = "RESULTS"
    BOARD_MEETING = "BOARD_MEETING"
    EX_DIVIDEND = "EX_DIVIDEND"
    EX_DATE = "EX_DATE"          # generic ex-date (bonus/split/rights)
    SPLIT = "SPLIT"
    BONUS = "BONUS"
    RIGHTS = "RIGHTS"
    AGM = "AGM"
    INDEX_REBALANCE = "INDEX_REBALANCE"
    MACRO = "MACRO"              # RBI policy, budget, US Fed ...
    HOLIDAY = "HOLIDAY"
    OTHER = "OTHER"


class SLMethod(StrEnum):
    ATR = "ATR"
    SWING = "SWING"
    VWAP = "VWAP"
    OPENING_RANGE = "OPENING_RANGE"
    DAILY_STRUCTURE = "DAILY_STRUCTURE"
    STRATEGY = "STRATEGY"
    MANUAL = "MANUAL"


class TrailingMethod(StrEnum):
    NONE = "NONE"
    ATR = "ATR"
    EMA9_5M = "EMA9_5M"
    EMA20_DAILY = "EMA20_DAILY"
    ATR_DAILY_CLOSE = "ATR_DAILY_CLOSE"


class RiskRule(StrEnum):
    """Every rule code the deterministic risk manager can veto with (section 5).

    The dashboard shows these codes verbatim when the owner is blocked.
    """

    PAPER_ONLY = "PAPER_ONLY"
    BATCH_NOT_ACTIVE = "BATCH_NOT_ACTIVE"
    HALTED = "HALTED"
    ENGINE_PAUSED = "ENGINE_PAUSED"
    MODE_DISABLED = "MODE_DISABLED"
    MODE_PAUSED = "MODE_PAUSED"
    ALLOCATION_EXHAUSTED = "ALLOCATION_EXHAUSTED"
    ENTRY_WINDOW = "ENTRY_WINDOW"
    MARKET_CLOSED = "MARKET_CLOSED"
    MAX_TRADES_PER_DAY = "MAX_TRADES_PER_DAY"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    NO_OVERNIGHT_SHORT = "NO_OVERNIGHT_SHORT"
    TRADEABLE_FALSE = "TRADEABLE_FALSE"
    CONFLUENCE_BELOW_MIN = "CONFLUENCE_BELOW_MIN"
    CONFIDENCE_BELOW_MIN = "CONFIDENCE_BELOW_MIN"
    GRADE_NOT_ALLOWED = "GRADE_NOT_ALLOWED"
    RR_BELOW_MIN = "RR_BELOW_MIN"
    SL_MISSING = "SL_MISSING"
    SL_WRONG_SIDE = "SL_WRONG_SIDE"
    SL_TOO_TIGHT = "SL_TOO_TIGHT"
    SL_TOO_WIDE = "SL_TOO_WIDE"
    SL_LOOSENING_FORBIDDEN = "SL_LOOSENING_FORBIDDEN"
    TARGET_INVALID = "TARGET_INVALID"
    NEAR_CIRCUIT = "NEAR_CIRCUIT"
    CHARGES_EXCEED_EDGE = "CHARGES_EXCEED_EDGE"
    CHARGES_TOO_HIGH = "CHARGES_TOO_HIGH"
    QTY_BELOW_ONE = "QTY_BELOW_ONE"
    QTY_ABOVE_MAX = "QTY_ABOVE_MAX"
    MAX_OPEN_POSITIONS = "MAX_OPEN_POSITIONS"
    MAX_POSITION_PCT = "MAX_POSITION_PCT"
    SECTOR_LIMIT = "SECTOR_LIMIT"
    CORRELATION_LIMIT = "CORRELATION_LIMIT"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"
    AVERAGING_DOWN_FORBIDDEN = "AVERAGING_DOWN_FORBIDDEN"
    OVERNIGHT_EXPOSURE = "OVERNIGHT_EXPOSURE"
    EVENT_IN_WINDOW = "EVENT_IN_WINDOW"
    LOSS_BUDGET_EXHAUSTED = "LOSS_BUDGET_EXHAUSTED"
    DRAWDOWN_TIER = "DRAWDOWN_TIER"
    CONSECUTIVE_LOSS_PAUSE = "CONSECUTIVE_LOSS_PAUSE"
    DAILY_PROFIT_LOCK = "DAILY_PROFIT_LOCK"
    LEVERAGE_NOT_ALLOWED = "LEVERAGE_NOT_ALLOWED"
    BLACKLISTED = "BLACKLISTED"
    STRATEGY_DISABLED = "STRATEGY_DISABLED"
    STALE_PRICE = "STALE_PRICE"
    INVALID_PRICE = "INVALID_PRICE"
    CONVERSION_NOT_ALLOWED = "CONVERSION_NOT_ALLOWED"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    # ---- execution quality (added 2026-11-25 after three stale / chased / quorum-less losses)
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"          # decision older than the signal validity / max age
    PRICE_MOVED = "PRICE_MOVED"                # live price drifted: R:R or SL distance no longer valid
    AGENT_QUORUM = "AGENT_QUORUM"              # too many LLM agents failed - no informed decision
    DEBATE_AGAINST = "DEBATE_AGAINST"          # bear case clearly won the debate
    NO_ROOM = "NO_ROOM"                        # opposing level too close to the entry
    REENTRY_COOLDOWN = "REENTRY_COOLDOWN"      # same symbol stopped out recently


class Autonomy(StrEnum):
    """Global trading autonomy (overrides the per-mode execution policy unless PER_MODE).

    Applies to PAPER trades only - nothing in this system can place a real order.
    """

    PER_MODE = "PER_MODE"                    # use each mode's execution_policy
    FULL_AUTO = "FULL_AUTO"                  # every risk-approved trade executes (no human)
    HYBRID = "HYBRID"                        # auto only for top setups, the rest to the approval queue
    HUMAN_IN_THE_LOOP = "HUMAN_IN_THE_LOOP"  # every trade waits for the owner's approval
    SIGNALS_ONLY = "SIGNALS_ONLY"            # log + email, never execute


class TaxHead(StrEnum):
    """Indian income-tax head of a closed trade (Income-tax Act, 2025)."""

    SPECULATIVE = "SPECULATIVE"   # intraday equity (squared off same day): business income at slab rates
    STCG = "STCG"                 # listed equity held <= 12 months, STT paid: 20 % (old s.111A, new s.196)
    LTCG = "LTCG"                 # held > 12 months: 12.5 % above Rs 1.25 lakh per FY (old s.112A, new s.198)
    BUSINESS = "BUSINESS"         # delivery trades treated as non-speculative business income (owner choice)
