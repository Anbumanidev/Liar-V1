"""Strategy framework (section 7.2).

Every strategy is a small, deterministic, pure class:

* it declares the ``modes`` it supports, the ``regimes`` it is designed for, its IST time
  ``windows`` per mode and a list of :class:`ParamSpec` (default + allowed range);
* :meth:`Strategy.generate` receives ONLY already-computed features
  (:class:`~core.types.StockFeatures` / :class:`~core.types.TimeframeFeatures` values and
  :class:`~core.types.MarketContext`) and returns a :class:`~core.types.StrategySignal` or ``None``.

Strategies never fetch data, never call an LLM and never look at the wall clock: time
comes from the ``now`` argument (taken from a ``Clock`` by the caller).

Common rules enforced by :meth:`Strategy.build_signal` for every strategy:

* the direction must be allowed for the mode (BTST/SWING/POSITIONAL are LONG only);
* the stop-loss is on the correct side of the entry and never at zero distance; it is
  widened to at least ``sl_min_atr`` x ATR and the setup is dropped when it would need
  more than ``sl_max_atr`` x ATR (the risk manager would veto it anyway);
* T1 is at least ``min_rr`` x risk away from the entry (the mode's minimum R:R); when the
  next opposing level lies farther away (but within a sane cap) T1 is moved out to it;
* T2 lies beyond T1 (next level or ``t2_r`` x risk);
* prices are rounded to the NSE tick (0.05) in the conservative direction so rounding can
  never push the R:R below the minimum.

Optional ``StockFeatures.extra`` keys read by strategies (all optional, ``None`` safe):

* ``sector`` (str) - NSE sector of the symbol (sector strength lookups in MarketContext);
* ``fundamental_score`` (0-100) - fundamentals score for positional strategies/confluence;
* ``range_high_20`` / ``range_low_20`` (float) - high/low of the 20 daily bars BEFORE the
  last closed bar (consolidation base). Strategies fall back to daily swing levels.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, ClassVar

from core.clock import parse_hhmm, to_ist
from core.enums import Direction, EntryType, Mode, Regime, Signal, TrailingMethod
from core.types import Levels, MarketContext, StockFeatures, StrategySignal, TimeframeFeatures
from modes.base import assert_direction_allowed, direction_allowed

#: NSE equity tick size.
TICK: float = 0.05

#: Mode minimum R:R used when the caller does not pass one (section 4, Exit/Target agent).
DEFAULT_MIN_RR: dict[Mode, float] = {
    Mode.INTRADAY: 1.5,
    Mode.BTST: 2.0,
    Mode.SWING: 2.0,
    Mode.POSITIONAL: 2.0,
}

#: Default SL distance bounds in ATR of the primary timeframe (mirror RiskSettings defaults).
DEFAULT_SL_MIN_ATR: float = 0.3
DEFAULT_SL_MAX_ATR: float = 3.0

#: Default trailing rule per mode (section 7.4).
DEFAULT_TRAILING: dict[Mode, TrailingMethod] = {
    Mode.INTRADAY: TrailingMethod.ATR,
    Mode.BTST: TrailingMethod.NONE,
    Mode.SWING: TrailingMethod.EMA20_DAILY,
    Mode.POSITIONAL: TrailingMethod.ATR_DAILY_CLOSE,
}

#: Time window used by daily-data strategies (SWING / POSITIONAL): next-morning re-validation
#: inside the CNC entry window and the post-close scan (16:00-18:00) / on-demand analysis.
CNC_SCAN_WINDOWS: list[tuple[str, str]] = [("09:15", "10:30"), ("15:30", "23:59")]


# =========================================================================== params


@dataclass(frozen=True)
class ParamSpec:
    """One tunable strategy parameter with its allowed range (editable from the dashboard).

    Attributes:
        name: parameter key.
        default: coded default value.
        min: inclusive lower bound (``None`` = unbounded; ignored for ``bool``).
        max: inclusive upper bound (``None`` = unbounded; ignored for ``bool``).
        kind: ``"float"``, ``"int"`` or ``"bool"``.
        description: human readable explanation (shown in the dashboard).
    """

    name: str
    default: float | int | bool
    min: float | None = None
    max: float | None = None
    kind: str = "float"
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("float", "int", "bool"):
            raise ValueError(f"ParamSpec {self.name}: unknown kind {self.kind!r}")
        if self.kind != "bool":
            if self.min is not None and self.max is not None and self.min > self.max:
                raise ValueError(f"ParamSpec {self.name}: min {self.min} > max {self.max}")
            if self.clamp(self.default) != self.default:
                raise ValueError(f"ParamSpec {self.name}: default {self.default} outside [{self.min}, {self.max}]")

    def coerce(self, value: Any) -> float | int | bool:
        """Convert ``value`` to this parameter's type.

        Raises:
            ValueError: if the value cannot be interpreted.
        """
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
                return value.strip().lower() in ("true", "1", "yes")
            raise ValueError(f"{self.name}: cannot interpret {value!r} as bool")
        if isinstance(value, bool):
            raise ValueError(f"{self.name}: boolean given for numeric parameter")
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.name}: cannot interpret {value!r} as number") from exc
        if not math.isfinite(num):
            raise ValueError(f"{self.name}: non-finite value {value!r}")
        return int(round(num)) if self.kind == "int" else num

    def clamp(self, value: Any) -> float | int | bool:
        """Coerce and clamp ``value`` into ``[min, max]``."""
        v = self.coerce(value)
        if self.kind == "bool":
            return v
        if self.min is not None and v < self.min:
            v = self.min
        if self.max is not None and v > self.max:
            v = self.max
        return int(round(v)) if self.kind == "int" else float(v)


# =========================================================================== helpers


def round_tick(price: float, mode: str = "nearest") -> float:
    """Round ``price`` to the NSE tick. ``mode``: ``nearest`` | ``down`` | ``up``."""
    units = price / TICK
    if mode == "down":
        n = math.floor(units + 1e-9)
    elif mode == "up":
        n = math.ceil(units - 1e-9)
    else:
        n = round(units)
    return round(n * TICK, 2)


def sign(direction: Direction) -> int:
    """+1 for LONG, -1 for SHORT."""
    return 1 if direction == Direction.LONG else -1


def val(features: StockFeatures, timeframe: str, key: str) -> float | None:
    """Value ``key`` of ``timeframe`` (None when the timeframe or value is missing)."""
    tf = features.tf(timeframe)
    if tf is None:
        return None
    v = tf.values.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def all_present(*values: float | None) -> bool:
    """True when no value is None."""
    return all(v is not None for v in values)


def in_windows(now: datetime, windows: Iterable[tuple[str, str]]) -> bool:
    """True when the IST wall time of ``now`` lies inside any [start, end] window (inclusive)."""
    t = to_ist(now).time().replace(tzinfo=None)
    for start, end in windows:
        if parse_hhmm(start) <= t <= parse_hhmm(end):
            return True
    return False


def collect_levels(features: StockFeatures, timeframes: Sequence[str] | None = None) -> list[float]:
    """All known price levels (pivots, CPR, S/R, swing points, prev day H/L/C, 52w H/L)."""
    out: list[float] = []
    tfs = timeframes if timeframes is not None else list(features.timeframes)
    for name in tfs:
        tf = features.tf(name)
        if tf is None or tf.levels is None:
            continue
        out.extend(_levels_of(tf.levels))
    for v in (features.week52_high, features.week52_low, features.prev_close):
        if v is not None:
            out.append(float(v))
    return [x for x in out if x is not None and math.isfinite(x) and x > 0]


def _levels_of(lv: Levels) -> list[float]:
    vals: list[float | None] = [
        lv.pivot, lv.r1, lv.r2, lv.r3, lv.s1, lv.s2, lv.s3, lv.cpr_bc, lv.cpr_tc,
        lv.prev_high, lv.prev_low, lv.prev_close,
    ]
    vals.extend(lv.supports)
    vals.extend(lv.resistances)
    vals.extend(lv.swing_highs)
    vals.extend(lv.swing_lows)
    return [float(v) for v in vals if v is not None]


def levels_beyond(levels: Iterable[float], price: float, direction: Direction) -> list[float]:
    """Levels strictly beyond ``price`` in ``direction``, nearest first."""
    s = sign(direction)
    beyond = sorted({round(x, 4) for x in levels if (x - price) * s > 1e-9}, key=lambda x: abs(x - price))
    return beyond


def nearest_opposing_level(
    features: StockFeatures, price: float, direction: Direction, timeframes: Sequence[str] | None = None
) -> float | None:
    """Nearest level in the trade's path (resistance for LONG, support for SHORT)."""
    lv = levels_beyond(collect_levels(features, timeframes), price, direction)
    return lv[0] if lv else None


def latest_patterns(tf: TimeframeFeatures | None, direction: Signal, lookback_bars: int = 3) -> list[int]:
    """Strengths (absolute) of patterns on ``tf`` in ``direction`` on its most recent bars."""
    if tf is None or not tf.patterns:
        return []
    times = sorted({p.bar_time for p in tf.patterns}, reverse=True)[:lookback_bars]
    return [abs(p.strength) for p in tf.patterns if p.direction == direction and p.bar_time in times]


def pick_stop(
    direction: Direction,
    entry: float,
    candidates: Sequence[float | None],
    atr: float,
    *,
    min_atr: float,
    max_atr: float,
    fallback_atr: float,
) -> float | None:
    """Choose a structural stop from ``candidates`` (priority order).

    The first candidate on the correct side of ``entry`` whose distance lies within
    ``[min_atr, max_atr]`` x ATR wins; a candidate that is too tight is widened to
    ``min_atr`` x ATR. When none qualifies, ``fallback_atr`` x ATR is used (if it is
    within the bounds). Returns ``None`` when no valid stop exists.
    """
    if atr <= 0 or not math.isfinite(atr):
        return None
    s = sign(direction)
    for c in candidates:
        if c is None or not math.isfinite(c):
            continue
        dist = (entry - c) * s
        if dist <= 0:
            continue
        if dist < min_atr * atr:
            return entry - s * min_atr * atr
        if dist <= max_atr * atr:
            return c
    if min_atr <= fallback_atr <= max_atr:
        return entry - s * fallback_atr * atr
    return None


def build_targets(
    direction: Direction,
    entry: float,
    stop: float,
    min_rr: float,
    levels: Iterable[float],
    *,
    t1_max_r: float | None = None,
    t2_r: float = 3.0,
) -> tuple[float, float]:
    """Deterministic T1/T2 (unrounded).

    * T1 = ``entry + s * min_rr * risk``; moved out to the next level beyond it if that level
      is within ``t1_max_r`` R (default ``min_rr + 1.5``).
    * T2 = the next level beyond T1 within ``t2_r + 1`` R, else ``max(t2_r, T1_r + 1)`` R.
    """
    s = sign(direction)
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("zero risk")
    cap = t1_max_r if t1_max_r is not None else min_rr + 1.5
    t1 = entry + s * min_rr * risk
    lv = levels_beyond(levels, entry, direction)
    for x in lv:
        r = (x - entry) * s / risk
        if r >= min_rr - 1e-9:
            if r <= cap:
                t1 = x
            break
    t1_r = (t1 - entry) * s / risk
    t2 = entry + s * max(t2_r, t1_r + 1.0) * risk
    for x in lv:
        r = (x - entry) * s / risk
        if r >= t1_r + 0.5 - 1e-9:
            if r <= max(t2_r, t1_r + 1.0) + 1.0:
                t2 = x
            break
    return t1, t2


# =========================================================================== strategy base


@dataclass
class Setup:
    """Raw setup returned by a strategy's :meth:`Strategy.evaluate` before normalisation."""

    direction: Direction
    entry: float
    stop: float
    entry_type: EntryType = EntryType.LIMIT
    target_levels: list[float] = field(default_factory=list)
    strength: float = 50.0
    notes: list[str] = field(default_factory=list)
    natural_target: float | None = None   # e.g. mean-reversion target (must give >= min R:R)
    trailing_rule: TrailingMethod | None = None
    t2_r: float | None = None


class Strategy(ABC):
    """Base class for all strategies. Subclasses set the class attributes and implement
    :meth:`evaluate`; :meth:`generate` applies the shared rules (mode, window, direction,
    SL bounds, minimum R:R, rounding) and builds the :class:`StrategySignal`.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    modes: ClassVar[tuple[Mode, ...]] = ()
    regimes: ClassVar[frozenset[Regime]] = frozenset()
    #: When True, the strategy is silent outside ``regimes`` (e.g. mean reversion).
    strict_regime: ClassVar[bool] = False
    #: When True, trading against the regime's trend direction still counts as a fit
    #: (reversal / fade strategies such as divergence reversals and gap fills).
    countertrend: ClassVar[bool] = False
    windows: ClassVar[dict[Mode, list[tuple[str, str]]]] = {}
    param_specs: ClassVar[tuple[ParamSpec, ...]] = ()
    #: Validity of an intraday signal, in minutes from ``now``.
    intraday_validity_minutes: ClassVar[int] = 15

    # ------------------------------------------------------------------ declarations

    def default_params(self) -> dict[str, Any]:
        """Coded defaults ``{name: default}``."""
        return {p.name: p.default for p in self.param_specs}

    def spec(self, name: str) -> ParamSpec | None:
        """The ParamSpec called ``name``."""
        for p in self.param_specs:
            if p.name == name:
                return p
        return None

    def supports(self, mode: Mode) -> bool:
        return Mode(mode) in self.modes

    def in_window(self, mode: Mode, now: datetime) -> bool:
        """True if ``now`` (IST) is inside one of the strategy's windows for ``mode``."""
        return in_windows(now, self.windows.get(Mode(mode), []))

    def regime_fit(self, regime: Regime, direction: Direction) -> bool:
        """Whether ``regime`` suits this strategy for a trade in ``direction``."""
        if regime not in self.regimes:
            return False
        if self.countertrend:
            return True
        if direction == Direction.LONG and regime == Regime.TRENDING_DOWN:
            return False
        if direction == Direction.SHORT and regime == Regime.TRENDING_UP:
            return False
        return True

    # ------------------------------------------------------------------ API

    def generate(
        self,
        features: StockFeatures,
        market: MarketContext | None,
        params: dict[str, Any],
        now: datetime,
        *,
        mode: Mode | None = None,
        min_rr: float | None = None,
        sl_min_atr: float = DEFAULT_SL_MIN_ATR,
        sl_max_atr: float = DEFAULT_SL_MAX_ATR,
    ) -> StrategySignal | None:
        """Evaluate the strategy on ``features`` at ``now``.

        Args:
            features: deterministic features of one symbol (mode = ``features.mode`` unless
                ``mode`` is given).
            market: broad market context (may be ``None``).
            params: parameters (missing keys fall back to coded defaults; values are clamped).
            now: current IST time (from a Clock); used for the time windows / validity.
            mode: override the mode (defaults to ``features.mode``).
            min_rr: mode minimum R:R (defaults to :data:`DEFAULT_MIN_RR`).
            sl_min_atr / sl_max_atr: SL distance bounds in ATR (RiskSettings).

        Returns:
            A validated :class:`StrategySignal` or ``None`` (no setup / outside window).
        """
        m = Mode(mode or features.mode)
        if not self.supports(m) or not self.in_window(m, now):
            return None
        p = self._merge_params(params)
        rr_min = float(min_rr if min_rr is not None else DEFAULT_MIN_RR[m])
        if self.strict_regime and features.regime not in self.regimes:
            return None
        setup = self.evaluate(features, market, p, m)
        if setup is None:
            return None
        return self.build_signal(
            setup, features, m, now, p, min_rr=rr_min, sl_min_atr=sl_min_atr, sl_max_atr=sl_max_atr
        )

    @abstractmethod
    def evaluate(
        self, features: StockFeatures, market: MarketContext | None, params: dict[str, Any], mode: Mode
    ) -> Setup | None:
        """Strategy-specific rules. Return a :class:`Setup` or ``None``."""

    # ------------------------------------------------------------------ shared logic

    def _merge_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        out = self.default_params()
        for k, v in (params or {}).items():
            spec = self.spec(k)
            if spec is None:
                continue
            try:
                out[k] = spec.clamp(v)
            except ValueError:
                out[k] = spec.default
        return out

    def build_signal(
        self,
        setup: Setup,
        features: StockFeatures,
        mode: Mode,
        now: datetime,
        params: dict[str, Any],
        *,
        min_rr: float,
        sl_min_atr: float = DEFAULT_SL_MIN_ATR,
        sl_max_atr: float = DEFAULT_SL_MAX_ATR,
    ) -> StrategySignal | None:
        """Normalise a :class:`Setup` into a validated :class:`StrategySignal`.

        Returns ``None`` when the setup cannot satisfy the shared rules.

        Raises:
            OvernightShortError: if a SHORT setup is produced for a CNC mode (programming error).
        """
        direction = setup.direction
        assert_direction_allowed(mode, direction)
        s = sign(direction)
        atr = features.atr
        if setup.entry <= 0 or not math.isfinite(setup.entry) or not math.isfinite(setup.stop):
            return None
        entry = round_tick(setup.entry)
        stop = setup.stop
        if (entry - stop) * s <= 0:
            return None
        if atr is not None and atr > 0:
            dist = (entry - stop) * s
            if dist > sl_max_atr * atr + 1e-9:
                return None
            if dist < sl_min_atr * atr:
                stop = entry - s * sl_min_atr * atr
        # round SL away from the entry so the realised risk is never smaller than computed
        stop = round_tick(stop, "down" if direction == Direction.LONG else "up")
        if (entry - stop) * s < TICK - 1e-9 or stop <= 0:
            return None
        risk = abs(entry - stop)
        if setup.natural_target is not None:
            room = (setup.natural_target - entry) * s
            if room < min_rr * risk - 1e-9:
                return None
        t2_r = setup.t2_r if setup.t2_r is not None else float(params.get("t2_r", 3.0))
        levels = list(setup.target_levels)
        if setup.natural_target is not None:
            levels.append(setup.natural_target)
        t1, t2 = build_targets(direction, entry, stop, min_rr, levels, t2_r=t2_r)
        rnd = "up" if direction == Direction.LONG else "down"
        t1 = round_tick(t1, rnd)
        t2 = round_tick(t2, rnd)
        if (t2 - t1) * s <= 0:
            t2 = round_tick(t1 + s * risk, rnd)
        if t1 <= 0 or t2 <= 0:
            return None
        valid_until = self._valid_until(mode, now)
        max_hold = self._max_holding_days(mode, params)
        trailing = setup.trailing_rule or DEFAULT_TRAILING[mode]
        regime_fit = self.regime_fit(features.regime, direction)
        strength = int(max(0, min(100, round(setup.strength - (0 if regime_fit else 15)))))
        notes = list(setup.notes)
        if not regime_fit:
            notes.append(f"regime {features.regime} outside strategy regimes")
        sig = StrategySignal(
            strategy=self.name,
            mode=mode,
            direction=direction,
            entry_price=entry,
            entry_type=setup.entry_type,
            stop_loss=stop,
            target_1=t1,
            target_2=t2,
            strength=strength,
            regime_fit=regime_fit,
            valid_until=valid_until,
            trailing_rule=trailing,
            max_holding_days=max_hold,
            notes=notes,
            params=dict(params),
        )
        if sig.rr < min_rr - 1e-6:  # defensive; rounding is conservative so this cannot happen
            return None
        return sig

    def _valid_until(self, mode: Mode, now: datetime) -> datetime | None:
        now = to_ist(now)
        if mode == Mode.INTRADAY:
            return now + timedelta(minutes=self.intraday_validity_minutes)
        if mode == Mode.BTST:
            return now.replace(hour=15, minute=20, second=0, microsecond=0) if now.time() < parse_hhmm("15:20") else now
        # SWING / POSITIONAL: validity is governed by approval_until_next_session (next session's
        # CNC entry window); the strategy cannot know the exchange calendar.
        return None

    def _max_holding_days(self, mode: Mode, params: dict[str, Any]) -> int | None:
        if mode == Mode.INTRADAY:
            return 0
        if mode == Mode.BTST:
            return 1
        v = params.get("max_holding_days")
        return int(v) if v is not None else None


def long_short_allowed(mode: Mode) -> tuple[Direction, ...]:
    """Directions a strategy may emit in ``mode``."""
    return tuple(d for d in (Direction.LONG, Direction.SHORT) if direction_allowed(mode, d))


def strength_from(*parts: tuple[float | None, float, float, float]) -> float:
    """Combine ``(value, lo, hi, weight)`` parts into a 0-100 strength (linear in [lo, hi])."""
    score = 40.0
    for value, lo, hi, weight in parts:
        if value is None or hi == lo:
            continue
        frac = (value - lo) / (hi - lo)
        score += weight * max(0.0, min(1.0, frac))
    return max(0.0, min(100.0, score))
