"""Project-wide exception types."""

from __future__ import annotations


class TradingSystemError(Exception):
    """Base class for all project errors."""


class RealTradingForbiddenError(TradingSystemError):
    """Raised whenever anything attempts to place/modify/cancel a REAL broker order.

    This system is PAPER TRADING ONLY (rule 0.1). This error must never be caught
    and silenced.
    """


class HardCapViolationError(TradingSystemError):
    """A setting exceeds a hard cap from config/hard_caps.yaml."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


class ConfigError(TradingSystemError):
    """Invalid or missing configuration."""


class DataUnavailableError(TradingSystemError):
    """Market data could not be obtained from any provider."""


class StopLossLooseningError(TradingSystemError):
    """Attempt to move a stop-loss further away from price (rule 0.10)."""


class AveragingDownError(TradingSystemError):
    """Attempt to add to a losing position (rule 0.10)."""


class OvernightShortError(TradingSystemError):
    """Attempt to hold a short position overnight / short in a CNC mode."""


class LedgerImbalanceError(TradingSystemError):
    """The capital ledger invariant (section 6.4) is broken."""


class ReserveAccessError(TradingSystemError):
    """Something tried to treat RESERVE as tradeable capital (section 6.4)."""


class AgentOutputError(TradingSystemError):
    """LLM output failed Pydantic validation after all retries."""
