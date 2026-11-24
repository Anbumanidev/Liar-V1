"""Strategy registry (section 7.2): registration, parameter resolution and batch execution.

Strategies register themselves with the :func:`register` class decorator when their module
is imported. :func:`_ensure_loaded` imports every built-in strategy module on first use, so
callers only need ``from strategies.registry import run_strategies``.

Disabled strategies (``disabled`` argument) may be given as plain names (``"vwap_pullback"``,
disabled in every mode) or as ``"<name>:<MODE>"`` (disabled in one mode only), matching the
rows of ``strategy_states`` (see :func:`strategies.performance.disabled_strategies`).
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Collection
from datetime import datetime
from typing import Any, TypeVar

from loguru import logger

from config.schema import Settings
from core.enums import Mode
from core.logging import log_event
from core.types import MarketContext, StockFeatures, StrategySignal
from modes.base import direction_allowed
from strategies.base import Strategy, sign

_T = TypeVar("_T", bound=type[Strategy])

_REGISTRY: dict[str, Strategy] = {}
_LOCK = threading.RLock()
_LOADED = False

#: Built-in strategy modules imported on first registry access.
BUILTIN_MODULES: tuple[str, ...] = (
    "strategies.intraday.opening_range_breakout",
    "strategies.intraday.vwap_pullback",
    "strategies.intraday.ema_9_21_pullback",
    "strategies.intraday.cpr_breakout",
    "strategies.intraday.gap_and_go",
    "strategies.intraday.gap_fill",
    "strategies.intraday.relative_strength_momentum",
    "strategies.intraday.mean_reversion",
    "strategies.intraday.supertrend_adx_15m",
    "strategies.btst",
    "strategies.swing.swing_base_breakout",
    "strategies.swing.swing_ema_pullback",
    "strategies.swing.swing_reversal_divergence",
    "strategies.swing.swing_rs_leader",
    "strategies.positional.positional_weekly_trend",
    "strategies.positional.positional_52w_breakout",
    "strategies.positional.positional_sector_rotation",
)


class _FixedClock:
    """Clock pinned to the cycle time so JSONL error logs land in the simulated day (replay)."""

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now


def register(cls: _T) -> _T:
    """Class decorator: instantiate and register a :class:`Strategy` subclass.

    Raises:
        ValueError: on a missing/duplicate name or an incomplete declaration.
    """
    inst = cls()
    if not inst.name:
        raise ValueError(f"{cls.__name__} has no name")
    if not inst.modes:
        raise ValueError(f"strategy {inst.name} declares no modes")
    for m in inst.modes:
        if not inst.windows.get(m):
            raise ValueError(f"strategy {inst.name} declares no time window for {m}")
    names = [p.name for p in inst.param_specs]
    if len(names) != len(set(names)):
        raise ValueError(f"strategy {inst.name} has duplicate parameter names")
    with _LOCK:
        existing = _REGISTRY.get(inst.name)
        if existing is not None and type(existing).__qualname__ != cls.__qualname__:
            raise ValueError(f"duplicate strategy name {inst.name!r}")
        _REGISTRY[inst.name] = inst
    return cls


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        for mod in BUILTIN_MODULES:
            importlib.import_module(mod)
        _LOADED = True


def all_strategies() -> list[Strategy]:
    """Every registered strategy, sorted by name."""
    _ensure_loaded()
    with _LOCK:
        return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def get_strategy(name: str) -> Strategy:
    """The strategy called ``name``.

    Raises:
        KeyError: unknown strategy.
    """
    _ensure_loaded()
    with _LOCK:
        return _REGISTRY[name]


def strategies_for(mode: Mode) -> list[Strategy]:
    """Strategies that declare support for ``mode`` (sorted by name)."""
    m = Mode(mode)
    return [s for s in all_strategies() if m in s.modes]


def resolve_params(name: str, settings: Settings | None) -> dict[str, Any]:
    """Effective parameters for strategy ``name``.

    Coded defaults are overlaid with ``settings.strategies.strategies[name].params``; every
    value is coerced and clamped to its ParamSpec range. Unknown or uninterpretable params
    are ignored with a warning.
    """
    strat = get_strategy(name)
    params = strat.default_params()
    cfg = settings.strategies.strategies.get(name) if settings is not None else None
    if cfg is None:
        return params
    for key, raw in cfg.params.items():
        spec = strat.spec(key)
        if spec is None:
            logger.warning(f"strategy {name}: unknown parameter {key!r} ignored")
            continue
        try:
            value = spec.clamp(raw)
        except ValueError as exc:
            logger.warning(f"strategy {name}: invalid value for {key!r} ({exc}); using default {spec.default}")
            continue
        if value != spec.coerce(raw):
            logger.warning(f"strategy {name}: {key}={raw!r} clamped to {value!r} (range {spec.min}..{spec.max})")
        params[key] = value
    return params


def _is_disabled(name: str, mode: Mode, disabled: Collection[str]) -> bool:
    return name in disabled or f"{name}:{Mode(mode).value}" in disabled


def strategy_enabled(name: str, mode: Mode, settings: Settings | None, disabled: Collection[str] = ()) -> bool:
    """Whether strategy ``name`` may run in ``mode``.

    False when the strategy does not support the mode, is disabled in settings (``enabled:
    false`` or ``modes`` not listing the mode) or appears in ``disabled`` (auto-disabled by
    self-evaluation, plain name or ``name:MODE``).
    """
    try:
        strat = get_strategy(name)
    except KeyError:
        return False
    m = Mode(mode)
    if m not in strat.modes or _is_disabled(name, m, disabled):
        return False
    cfg = settings.strategies.strategies.get(name) if settings is not None else None
    if cfg is None:
        return True
    if not cfg.enabled:
        return False
    return not cfg.modes or m in cfg.modes


def _signal_valid(sig: StrategySignal, mode: Mode, min_rr: float) -> str | None:
    """Return a reason string if ``sig`` violates a hard rule, else ``None``."""
    if sig.mode != mode:
        return f"mode mismatch {sig.mode} != {mode}"
    if not direction_allowed(mode, sig.direction):
        return f"{sig.direction} not allowed in {mode}"
    s = sign(sig.direction)
    if (sig.entry_price - sig.stop_loss) * s <= 0:
        return "stop-loss on wrong side / zero distance"
    if (sig.target_1 - sig.entry_price) * s <= 0:
        return "target_1 on wrong side"
    if sig.rr < min_rr - 1e-6:
        return f"R:R {sig.rr:.2f} below mode minimum {min_rr}"
    return None


def run_strategies(
    features: StockFeatures,
    market: MarketContext | None,
    mode: Mode,
    settings: Settings | None,
    now: datetime,
    disabled: Collection[str] = frozenset(),
) -> list[StrategySignal]:
    """Run every enabled strategy for ``mode`` on ``features``. Never raises.

    A failing strategy is logged (loguru + JSONL ``errors``) and skipped; a signal that
    violates a hard rule (direction, SL side, minimum R:R) is dropped and logged.

    Returns:
        Signals sorted by strength (desc), then strategy name.
    """
    m = Mode(mode)
    out: list[StrategySignal] = []
    try:
        candidates = strategies_for(m)
    except Exception as exc:  # registry import failure must not break the cycle
        logger.exception(f"strategy registry failed to load: {exc}")
        log_event("errors", {"component": "strategies.registry", "message": str(exc)}, clock=_FixedClock(now))
        return out
    if settings is not None:
        min_rr = settings.mode(m).min_rr
        sl_min, sl_max = settings.risk.sl_min_atr, settings.risk.sl_max_atr
    else:
        from strategies.base import DEFAULT_MIN_RR, DEFAULT_SL_MAX_ATR, DEFAULT_SL_MIN_ATR

        min_rr, sl_min, sl_max = DEFAULT_MIN_RR[m], DEFAULT_SL_MIN_ATR, DEFAULT_SL_MAX_ATR
    for strat in candidates:
        if not strategy_enabled(strat.name, m, settings, disabled):
            continue
        try:
            params = resolve_params(strat.name, settings)
            sig = strat.generate(
                features, market, params, now, mode=m, min_rr=min_rr, sl_min_atr=sl_min, sl_max_atr=sl_max
            )
        except Exception as exc:
            logger.exception(f"strategy {strat.name} failed for {features.symbol} ({m}): {exc}")
            log_event(
                "errors",
                {"component": f"strategies.{strat.name}", "symbol": features.symbol, "mode": m.value,
                 "message": str(exc)},
                clock=_FixedClock(now),
            )
            continue
        if sig is None:
            continue
        reason = _signal_valid(sig, m, min_rr)
        if reason is not None:
            logger.error(f"strategy {strat.name} produced an invalid signal for {features.symbol}: {reason}")
            log_event(
                "errors",
                {"component": f"strategies.{strat.name}", "symbol": features.symbol, "mode": m.value,
                 "message": f"invalid signal dropped: {reason}"},
                clock=_FixedClock(now),
            )
            continue
        out.append(sig)
    out.sort(key=lambda s: (-s.strength, s.strategy))
    return out
