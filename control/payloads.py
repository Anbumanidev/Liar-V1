"""Typed payloads for dashboard -> engine control commands (section 14.2).

The dashboard validates a payload with these models before writing a ``control_commands``
row; the engine re-validates when it executes. Results are written back as
``CommandResult`` JSON in ``control_commands.result_json``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.enums import CommandType, Direction, EntryType, Mode, RiskRule, TrailingMethod


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyPayload(_P):
    pass


class ModePayload(_P):
    mode: Mode
    reason: str | None = None


class KillSwitchPayload(_P):
    exit_cnc: bool = False          # also exit all CNC positions (at next available price)
    reason: str = "Owner kill switch"


class ApprovePayload(_P):
    approval_id: int


class RejectPayload(_P):
    approval_id: int
    reason: str | None = None


class ModifyApprovePayload(_P):
    """Modify then approve. Any None field keeps the queued value. Re-checked by risk manager."""

    approval_id: int
    mode: Mode | None = None              # "approve as different mode"
    entry_type: EntryType | None = None
    entry_price: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    target_1: float | None = Field(default=None, gt=0)
    target_2: float | None = Field(default=None, gt=0)
    qty: int | None = Field(default=None, ge=1)   # capped at the risk-manager maximum
    reason: str | None = None


class BulkRejectPayload(_P):
    approval_ids: list[int]
    reason: str | None = None


class ManualTradePayload(_P):
    symbol: str
    mode: Mode
    direction: Direction
    entry_type: EntryType = EntryType.MARKET
    entry_price: float | None = Field(default=None, gt=0)  # required for LIMIT; reference for MARKET
    qty: int = Field(ge=1)
    stop_loss: float = Field(gt=0)                          # mandatory
    target_1: float = Field(gt=0)
    target_2: float | None = Field(default=None, gt=0)
    analyze_first: bool = False
    note: str | None = None


class AnalyzeNowPayload(_P):
    symbol: str
    mode: Mode
    route: bool = False   # False: analysis only (shown on dashboard). True: follow mode execution policy.


class ClosePositionPayload(_P):
    position_id: int
    qty: int | None = Field(default=None, ge=1)   # None = close fully
    reason: str | None = None


class TightenSLPayload(_P):
    position_id: int
    new_stop_loss: float = Field(gt=0)


class UpdateTargetsPayload(_P):
    position_id: int
    target_1: float | None = Field(default=None, gt=0)
    target_2: float | None = Field(default=None, gt=0)


class UpdateTrailingPayload(_P):
    position_id: int
    trailing_rule: TrailingMethod
    trail_atr_mult: float | None = Field(default=None, gt=0)


class ConvertPositionPayload(_P):
    position_id: int
    to_mode: Mode


class UpdateHoldingPayload(_P):
    position_id: int
    max_holding_days: int = Field(ge=0, le=180)


class EventPinPayload(_P):
    position_id: int
    pinned: bool


class ResumeHaltPayload(_P):
    confirm_text: str   # must be "CONFIRM"


class ReserveTransferPayload(_P):
    amount: float = Field(gt=0)
    confirm_text: str   # must be "CONFIRM"
    note: str | None = None


class NewBatchPayload(_P):
    name: str = Field(min_length=1, max_length=120)
    starting_capital: float = Field(default=1000.0, gt=0)
    copy_settings_from_version_id: int | None = None   # None = current settings
    open_cnc_handling: str = Field(default="EXIT", pattern="^(EXIT|RUN_TO_COMPLETION)$")
    confirm_text: str   # must be "CONFIRM" or the current batch name


class CloseBatchPayload(_P):
    batch_id: int
    confirm_text: str


class StrategyPayload(_P):
    strategy: str
    mode: Mode


class RunJobPayload(_P):
    job: str   # premarket | watchlist | intraday_cycle | btst_scan | swing_scan | positional_scan | eod | eod_report


PAYLOAD_MODELS: dict[CommandType, type[_P]] = {
    CommandType.ENGINE_START: EmptyPayload,
    CommandType.ENGINE_PAUSE: EmptyPayload,
    CommandType.ENGINE_RESUME: EmptyPayload,
    CommandType.ENGINE_STOP: EmptyPayload,
    CommandType.MODE_PAUSE: ModePayload,
    CommandType.MODE_RESUME: ModePayload,
    CommandType.KILL_SWITCH: KillSwitchPayload,
    CommandType.RELOAD_SETTINGS: EmptyPayload,
    CommandType.APPROVE: ApprovePayload,
    CommandType.REJECT: RejectPayload,
    CommandType.MODIFY_APPROVE: ModifyApprovePayload,
    CommandType.BULK_REJECT: BulkRejectPayload,
    CommandType.MANUAL_TRADE: ManualTradePayload,
    CommandType.ANALYZE_NOW: AnalyzeNowPayload,
    CommandType.CLOSE_POSITION: ClosePositionPayload,
    CommandType.TIGHTEN_SL: TightenSLPayload,
    CommandType.UPDATE_TARGETS: UpdateTargetsPayload,
    CommandType.UPDATE_TRAILING: UpdateTrailingPayload,
    CommandType.CONVERT_POSITION: ConvertPositionPayload,
    CommandType.UPDATE_HOLDING: UpdateHoldingPayload,
    CommandType.SET_EVENT_PIN: EventPinPayload,
    CommandType.RESUME_HALT: ResumeHaltPayload,
    CommandType.RESERVE_TRANSFER: ReserveTransferPayload,
    CommandType.NEW_BATCH: NewBatchPayload,
    CommandType.CLOSE_BATCH: CloseBatchPayload,
    CommandType.REENABLE_STRATEGY: StrategyPayload,
    CommandType.SEND_TEST_EMAIL: EmptyPayload,
    CommandType.RUN_JOB: RunJobPayload,
}

#: Commands that are destructive and require a typed confirmation in the dashboard.
DESTRUCTIVE_COMMANDS: frozenset[CommandType] = frozenset(
    {CommandType.KILL_SWITCH, CommandType.ENGINE_STOP, CommandType.NEW_BATCH, CommandType.CLOSE_BATCH,
     CommandType.RESUME_HALT, CommandType.RESERVE_TRANSFER, CommandType.BULK_REJECT}
)


class CommandResult(_P):
    """What the engine writes to ``control_commands.result_json``."""

    ok: bool
    message: str = ""
    rule: RiskRule | None = None           # exact risk rule when the owner was blocked
    data: dict[str, Any] = Field(default_factory=dict)


def validate_payload(command_type: CommandType, payload: dict[str, Any]) -> _P:
    """Validate a raw payload dict for ``command_type`` (raises ``pydantic.ValidationError``)."""
    model = PAYLOAD_MODELS[CommandType(command_type)]
    return model.model_validate(payload or {})
