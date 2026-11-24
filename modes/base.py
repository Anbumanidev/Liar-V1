"""Static, HARD-CODED properties of each trading mode (section 3).

Configurable parts (timeframes, windows, policies, weights ...) live in ``Settings.modes``.
What is here can NOT be changed by configuration:

* INTRADAY is MIS and may go long or short (squared off same day).
* BTST / SWING / POSITIONAL are CNC and LONG ONLY - overnight shorts are impossible in the
  Indian cash market.
* Leverage is only ever allowed for MIS.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.enums import Direction, Mode, Product, Timeframe
from core.errors import OvernightShortError


@dataclass(frozen=True)
class ModeSpec:
    mode: Mode
    product: Product
    allowed_directions: frozenset[Direction]
    overnight: bool
    leverage_allowed: bool
    default_timeframes: tuple[Timeframe, ...]
    description: str


MODE_SPECS: dict[Mode, ModeSpec] = {
    Mode.INTRADAY: ModeSpec(
        mode=Mode.INTRADAY,
        product=Product.MIS,
        allowed_directions=frozenset({Direction.LONG, Direction.SHORT}),
        overnight=False,
        leverage_allowed=True,
        default_timeframes=(Timeframe.M5, Timeframe.M15, Timeframe.H1),
        description="Same day, square-off 15:15, MIS, long + short",
    ),
    Mode.BTST: ModeSpec(
        mode=Mode.BTST,
        product=Product.CNC,
        allowed_directions=frozenset({Direction.LONG}),
        overnight=True,
        leverage_allowed=False,
        default_timeframes=(Timeframe.M15, Timeframe.H1, Timeframe.D1),
        description="Buy today 14:45-15:20, sell next trading day, CNC, long only",
    ),
    Mode.SWING: ModeSpec(
        mode=Mode.SWING,
        product=Product.CNC,
        allowed_directions=frozenset({Direction.LONG}),
        overnight=True,
        leverage_allowed=False,
        default_timeframes=(Timeframe.H1, Timeframe.D1),
        description="2-10 trading days, CNC, long only",
    ),
    Mode.POSITIONAL: ModeSpec(
        mode=Mode.POSITIONAL,
        product=Product.CNC,
        allowed_directions=frozenset({Direction.LONG}),
        overnight=True,
        leverage_allowed=False,
        default_timeframes=(Timeframe.D1, Timeframe.W1),
        description="Weeks to ~3 months, CNC, long only",
    ),
}


def get_mode_spec(mode: Mode) -> ModeSpec:
    """Static spec for ``mode``."""
    return MODE_SPECS[Mode(mode)]


def product_for(mode: Mode) -> Product:
    return get_mode_spec(mode).product


def is_overnight(mode: Mode) -> bool:
    return get_mode_spec(mode).overnight


def direction_allowed(mode: Mode, direction: Direction) -> bool:
    return Direction(direction) in get_mode_spec(mode).allowed_directions


def assert_direction_allowed(mode: Mode, direction: Direction) -> None:
    """Raise ``OvernightShortError`` for shorts in CNC modes (hard-coded guard)."""
    if not direction_allowed(mode, direction):
        raise OvernightShortError(
            f"{direction} is not allowed in {mode}: BTST/SWING/POSITIONAL are long-only (no overnight shorts)."
        )


def conversion_allowed(from_mode: Mode, to_mode: Mode, direction: Direction) -> bool:
    """INTRADAY -> SWING/POSITIONAL conversion, long positions only (section 3)."""
    return (
        Mode(from_mode) == Mode.INTRADAY
        and Mode(to_mode) in (Mode.SWING, Mode.POSITIONAL)
        and Direction(direction) == Direction.LONG
    )
