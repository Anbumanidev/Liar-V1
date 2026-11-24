"""Capital-module exceptions (all subclass ``core.errors.TradingSystemError``)."""

from __future__ import annotations

from core.errors import ReserveAccessError, TradingSystemError


class CapitalError(TradingSystemError):
    """Generic invalid capital operation (bad amount, unknown batch, double init ...)."""


class AccountNotFoundError(CapitalError):
    """No ``capital_accounts`` row exists for the batch."""


class InsufficientCashError(ReserveAccessError):
    """A BLOCK asked for more than TRADING cash.

    Subclasses ``ReserveAccessError`` on purpose: the only place the missing money could come
    from is RESERVE, which is never available for trading (section 6.4).
    """


class UnconfirmedTransferError(CapitalError):
    """A RESERVE -> TRADING manual transfer (or manual adjustment) was not confirmed."""
