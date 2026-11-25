"""Layer-3 Meta Decision Agent (section 4).

The meta agent receives the sub-agent reports, the weighted vote, the strategy candidates,
the confluence score, the regime, the capital summary, the mode settings and the debate, and
returns a ``MetaDecision``. **The LLM chooses among deterministic options only** (rule 0.3):

* ``decision``: BUY | SELL_SHORT | NO_TRADE;
* ``strategy_index``: one of ``ctx.strategy_signals`` (indices of that list);
* ``entry_type``: MARKET (entry = live LTP) or LIMIT (entry = the strategy's entry, or the
  entry-timing agent's code-validated LIMIT price for the reference setup);
* ``stop_loss_index`` / ``target_1_index`` / ``target_2_index``: options built by code from
  the stop-loss and exit/target agents' reports (their code-validated candidates) and the
  strategy signals;
* ``recommended_mode`` (+ reason) under the AUTO selection policy only.

Code maps the indices to numbers and validates (also inside the Pydantic schema, so the LLM
is re-prompted with the exact error, <= the client's retries):

* the decision matches the chosen setup's direction; SELL_SHORT only where the mode allows
  shorts (INTRADAY, ``modes.base.direction_allowed``);
* the stop is on the protective side of the entry (and within the ATR distance limits of
  ``settings.risk`` for the analysed mode); target_1 is on the profit side with
  R:R >= ``settings.mode(mode).min_rr``; target_2 (optional) lies beyond target_1;
* AUTO: the recommended mode is enabled (``enabled_modes``), allows the direction and comes
  with a reason. FIXED: a recommendation is ignored.

After validation: confidence < ``settings.mode(mode).confidence_threshold`` -> NO_TRADE with
``why_not`` (confidence is capped when the weighted vote opposes the trade). Deterministic
gates (``agents.gates.trade_gate``: no usable strategy signal, TRADEABLE=false, blocking
event, confluence below the mode minimum) return NO_TRADE without calling the model.
Trailing rule, max holding days and time validity are set by code. **Quantity is never part
of the output** (the risk manager sizes trades).

Graceful degradation (section 16): any failure (LLM failed after retries, prompt error,
unexpected exception) -> ``MetaDecision.no_trade(..., failed=True)``; ``run_meta`` never
raises. NO_TRADE decisions carry the would-be prices of the best valid option when one exists
(strategy / entry / stop / targets) so shadow tracking can follow the rejected setup;
``decision`` is always authoritative. ``deterministic_meta`` is the ``rules`` backend.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agents.aggregate import VoteSummary, code_facts
from agents.base import DIRECTION_SIGNAL, OPPOSITE, current_price, mode_rules_text, r2
from agents.debate.debate import (
    agent_digest,
    market_digest,
    overnight_risk,
    setup_costs,
    setup_digest,
    vote_digest,
)
from agents.gates import strongest_usable, trade_gate, usable_signals
from agents.schemas import clamp_int, output_schema_summary, truncate_list, truncate_text
from agents.sub_agents.exit_target import ALLOWED_TRAILING
from config.schema import META_AGENT
from core.clock import parse_hhmm, to_ist
from core.enums import (
    AgentStatus,
    Direction,
    EntryType,
    Mode,
    SelectionPolicy,
    TradeDecision,
    TrailingMethod,
)
from core.logging import log_event, to_jsonable
from core.types import AgentReport, DebateResult, MetaDecision, StockContext, StrategySignal
from modes.base import direction_allowed

if TYPE_CHECKING:  # pragma: no cover
    from agents.llm import LLMClient
    from agents.prompt_store import PromptStore
    from config.schema import Settings

#: Prices closer than this (rupees) are the same option.
PRICE_TOL = 0.01
#: Confidence cap when the weighted vote points the other way.
VOTE_OPPOSES_CONF_CAP = 50
#: Default validity of an INTRADAY decision without a strategy ``valid_until``.
INTRADAY_VALIDITY_MINUTES = 15
#: BTST entries must happen before this time.
BTST_ENTRY_CUTOFF = "15:20"
MAX_REASONS = 5
MAX_REASON_CHARS = 240

_DECISION_SYNONYMS: dict[str, str] = {
    "LONG": "BUY", "GO_LONG": "BUY", "BUY_LONG": "BUY",
    "SHORT": "SELL_SHORT", "SELL": "SELL_SHORT", "SELLSHORT": "SELL_SHORT", "SHORT_SELL": "SELL_SHORT",
    "NOTRADE": "NO_TRADE", "NONE": "NO_TRADE", "HOLD": "NO_TRADE", "WAIT": "NO_TRADE", "SKIP": "NO_TRADE",
    "AVOID": "NO_TRADE", "PASS": "NO_TRADE",
}
_DECISION_FOR: dict[Direction, TradeDecision] = {Direction.LONG: TradeDecision.BUY,
                                                 Direction.SHORT: TradeDecision.SELL_SHORT}


class MetaValidationError(ValueError):
    """The meta agent's choice violates a code rule."""


# --------------------------------------------------------------------------- LLM schema


def _none_like(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().upper() in ("", "NULL", "NONE", "N/A", "NA"))


class MetaOutput(BaseModel):
    """LLM answer of the meta agent: choices among code-built options. No prices, no quantity."""

    model_config = ConfigDict(extra="ignore")

    decision: TradeDecision = Field(description="BUY, SELL_SHORT or NO_TRADE")
    strategy_index: int | None = Field(default=None, description="index of the chosen strategy option (null for NO_TRADE)")
    entry_type: EntryType = Field(default=EntryType.LIMIT, description="MARKET (enter at the LTP) or LIMIT (strategy limit_entry_price)")
    stop_loss_index: int | None = Field(default=None, description="index of the chosen stop_loss option")
    target_1_index: int | None = Field(default=None, description="index of the chosen target option for target 1")
    target_2_index: int | None = Field(default=None, description="index of a farther target option, or null")
    recommended_mode: Mode | None = Field(default=None, description="AUTO policy only: best enabled mode, else null")
    recommended_mode_reason: str | None = Field(default=None, description="AUTO policy only: why that mode fits")
    confidence: int = Field(ge=0, le=100, description="0-100 confidence that this decision is right")
    reasons_for: list[str] = Field(default_factory=list, description=f"Up to {MAX_REASONS} reasons for the decision, quoting facts")
    reasons_against_considered: list[str] = Field(
        default_factory=list, description=f"Up to {MAX_REASONS} counter-arguments you weighed (Bear, dissenting agents, costs)")
    why_not: str | None = Field(default=None, description="For NO_TRADE: the main reason")

    @field_validator("decision", mode="before")
    @classmethod
    def _v_decision(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().upper().replace("-", "_").replace(" ", "_")
            compact = key.replace("_", "")
            return _DECISION_SYNONYMS.get(key) or _DECISION_SYNONYMS.get(compact) or key
        return v

    @field_validator("entry_type", mode="before")
    @classmethod
    def _v_entry_type(cls, v: Any) -> Any:
        if _none_like(v):
            return EntryType.LIMIT
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("recommended_mode", mode="before")
    @classmethod
    def _v_mode(cls, v: Any) -> Any:
        if _none_like(v):
            return None
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("strategy_index", "stop_loss_index", "target_1_index", "target_2_index", mode="before")
    @classmethod
    def _v_index(cls, v: Any) -> Any:
        return None if _none_like(v) else v

    @field_validator("confidence", mode="before")
    @classmethod
    def _v_conf(cls, v: Any) -> Any:
        return clamp_int(v, 0, 100)

    @field_validator("reasons_for", "reasons_against_considered", mode="before")
    @classmethod
    def _v_reasons(cls, v: Any) -> Any:
        if v is None:
            return []
        return truncate_list(v, MAX_REASONS, MAX_REASON_CHARS)

    @field_validator("recommended_mode_reason", "why_not", mode="before")
    @classmethod
    def _v_text(cls, v: Any) -> Any:
        if _none_like(v):
            return None
        return truncate_text(v, 400) if isinstance(v, str) else v


# --------------------------------------------------------------------------- options


@dataclass(frozen=True)
class PriceOption:
    """One code-computed price the LLM may choose by index."""

    index: int
    price: float
    sources: tuple[str, ...]


@dataclass
class MetaOptions:
    """Deterministic option lists offered to the meta agent (indices are what the LLM picks)."""

    strategies: list[StrategySignal]
    usable: list[int]
    excluded: dict[int, str]
    reference: int | None
    ltp: float | None
    atr: float | None
    limit_entries: dict[int, float]
    stops: list[PriceOption]
    targets: list[PriceOption]
    agent_stop: float | None = None
    agent_targets: tuple[float | None, float | None] = (None, None)
    agent_entry_type: EntryType | None = None
    agent_trailing: TrailingMethod | None = None
    agent_holding_days: int | None = None


@dataclass
class Resolved:
    """A validated trade choice mapped to numbers."""

    strategy_index: int
    signal: StrategySignal
    decision: TradeDecision
    mode: Mode
    entry_type: EntryType
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float | None
    rr: float
    notes: list[str] = field(default_factory=list)


def _report(reports: Sequence[AgentReport], name: str) -> AgentReport | None:
    rep = next((r for r in reports if r.agent == name), None)
    return None if rep is None or rep.status == AgentStatus.SKIPPED else rep


def _float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _atr(ctx: StockContext) -> float | None:
    f = ctx.features
    if f is None:
        return None
    if f.atr:
        return float(f.atr)
    tf = f.tf(f.primary_timeframe)
    return tf.v("atr_14") if tf else None


def _dedupe(raw: list[tuple[float, str]], ref: float | None) -> list[PriceOption]:
    """Merge prices within ``PRICE_TOL`` (sources joined); sort nearest to ``ref`` first."""
    merged: list[tuple[float, list[str]]] = []
    for price, source in raw:
        for p, sources in merged:
            if abs(p - price) <= PRICE_TOL:
                if source not in sources:
                    sources.append(source)
                break
        else:
            merged.append((price, [source]))
    if ref is not None:
        merged.sort(key=lambda m: abs(m[0] - ref))
    return [PriceOption(index=i, price=round(p, 4), sources=tuple(s)) for i, (p, s) in enumerate(merged)]


def build_options(ctx: StockContext, reports: Sequence[AgentReport]) -> MetaOptions:
    """Build every option list from the strategy signals and the agents' code-validated extras."""
    sigs = list(ctx.strategy_signals)
    usable, excluded = usable_signals(ctx)
    best = strongest_usable(ctx)
    ref = best[0] if best else None
    ref_sig = sigs[ref] if ref is not None else None
    ref_entry = ref_sig.entry_price if ref_sig else None

    stops_raw: list[tuple[float, str]] = []
    targets_raw: list[tuple[float, str]] = []
    agent_stop: float | None = None
    agent_t1 = agent_t2 = None
    agent_entry_type: EntryType | None = None
    agent_trailing: TrailingMethod | None = None
    agent_days: int | None = None
    limit_entries = {i: sigs[i].entry_price for i in usable}

    sl = _report(reports, "stop_loss")
    sl_facts = code_facts(sl) if sl is not None else {}
    if ref_sig is not None and sl_facts.get("setup_strategy") == ref_sig.strategy:
        agent_stop = _float(sl_facts.get("stop_loss"))
        if agent_stop is not None:
            stops_raw.append((agent_stop, f"stop_loss agent choice ({sl_facts.get('method')})"))
        for c in sl_facts.get("candidates") or []:
            p = _float(c.get("price"))
            if p is not None and c.get("valid"):
                stops_raw.append((p, f"stop_loss candidate ({c.get('method')})"))
    for i in usable:
        stops_raw.append((sigs[i].stop_loss, f"strategy {sigs[i].strategy} stop"))

    et = _report(reports, "exit_target")
    et_facts = code_facts(et) if et is not None else {}
    if ref_sig is not None and et_facts.get("setup_strategy") == ref_sig.strategy:
        agent_t1, agent_t2 = _float(et_facts.get("target_1")), _float(et_facts.get("target_2"))
        if agent_t1 is not None:
            targets_raw.append((agent_t1, "exit_target agent target_1"))
        if agent_t2 is not None:
            targets_raw.append((agent_t2, "exit_target agent target_2"))
        for c in et_facts.get("candidates") or []:
            p = _float(c.get("price"))
            if p is not None:
                targets_raw.append((p, f"exit_target candidate ({c.get('label')})"))
        try:
            agent_trailing = TrailingMethod(et_facts["trailing_rule"]) if et_facts.get("trailing_rule") else None
        except ValueError:
            agent_trailing = None
        days = et_facts.get("max_holding_days")
        agent_days = int(days) if isinstance(days, (int, float)) else None
    for i in usable:
        targets_raw.append((sigs[i].target_1, f"strategy {sigs[i].strategy} target_1"))
        if sigs[i].target_2 is not None:
            targets_raw.append((sigs[i].target_2, f"strategy {sigs[i].strategy} target_2"))

    en = _report(reports, "entry_timing")
    en_facts = code_facts(en) if en is not None else {}
    if ref_sig is not None and en_facts.get("setup_strategy") == ref_sig.strategy:
        etype = str(en_facts.get("entry_type") or "")
        if etype in (EntryType.MARKET.value, EntryType.LIMIT.value):
            agent_entry_type = EntryType(etype)
        price = _float(en_facts.get("entry_price"))
        if etype == EntryType.LIMIT.value and price is not None and ref is not None:
            limit_entries[ref] = price

    return MetaOptions(
        strategies=sigs, usable=usable, excluded=excluded, reference=ref, ltp=current_price(ctx), atr=_atr(ctx),
        limit_entries=limit_entries, stops=_dedupe(stops_raw, ref_entry), targets=_dedupe(targets_raw, ref_entry),
        agent_stop=agent_stop, agent_targets=(agent_t1, agent_t2), agent_entry_type=agent_entry_type,
        agent_trailing=agent_trailing, agent_holding_days=agent_days,
    )


# --------------------------------------------------------------------------- validation


def _final_mode(choice: MetaOutput, ctx: StockContext, settings: "Settings", enabled_modes: Sequence[Mode],
                direction: Direction, errors: list[str], notes: list[str]) -> Mode:
    rec = choice.recommended_mode
    if rec is None or rec == ctx.mode:
        return ctx.mode
    if settings.modes.selection_policy != SelectionPolicy.AUTO:
        notes.append(f"recommended_mode {rec.value} ignored: selection policy is FIXED")
        return ctx.mode
    if rec not in set(enabled_modes):
        errors.append(f"recommended_mode {rec.value} is not enabled (enabled: "
                      f"{', '.join(m.value for m in enabled_modes) or 'none'})")
    elif not direction_allowed(rec, direction):
        errors.append(f"recommended_mode {rec.value} does not allow {direction.value} trades (no overnight shorts)")
    elif not (choice.recommended_mode_reason or "").strip():
        errors.append("recommended_mode_reason is required when recommending a different mode")
    else:
        return rec
    return ctx.mode


def _option(options: Sequence[PriceOption], idx: int | None, name: str, errors: list[str]) -> PriceOption | None:
    if idx is None:
        errors.append(f"{name} is required for a trade")
        return None
    if not 0 <= idx < len(options):
        errors.append(f"{name} {idx} does not exist (valid: 0-{len(options) - 1})" if options
                      else f"{name}: there are no options")
        return None
    return options[idx]


def resolve_choice(choice: MetaOutput, opts: MetaOptions, ctx: StockContext, settings: "Settings",
                   enabled_modes: Sequence[Mode]) -> Resolved | None:
    """Map a BUY / SELL_SHORT choice to numbers and validate it (None for NO_TRADE).

    Raises:
        MetaValidationError: listing every violated rule.
    """
    if choice.decision == TradeDecision.NO_TRADE:
        return None
    errors: list[str] = []
    notes: list[str] = []
    idx = choice.strategy_index
    if idx is None or idx not in opts.usable:
        why = opts.excluded.get(idx) if idx is not None else None
        errors.append(f"strategy_index {idx} is not a usable strategy option"
                      + (f" ({why})" if why else "") + f"; usable: {opts.usable}")
        raise MetaValidationError("; ".join(errors))
    sig = opts.strategies[idx]
    direction = sig.direction
    expected = _DECISION_FOR[direction]
    if choice.decision != expected:
        errors.append(f"decision {choice.decision.value} does not match the {direction.value} setup "
                      f"{sig.strategy} (use {expected.value})")
    mode = _final_mode(choice, ctx, settings, enabled_modes, direction, errors, notes)
    if TradeDecision.SELL_SHORT in (choice.decision, expected) and not direction_allowed(mode, Direction.SHORT):
        errors.append(f"SELL_SHORT is not allowed in {mode.value}: SELL_SHORT only in INTRADAY (no overnight shorts)")

    if choice.entry_type == EntryType.MARKET:
        entry = opts.ltp
        if entry is None:
            errors.append("MARKET entry needs a live price (none available); use LIMIT")
    else:
        entry = opts.limit_entries.get(idx, sig.entry_price)
    sl = _option(opts.stops, choice.stop_loss_index, "stop_loss_index", errors)
    t1 = _option(opts.targets, choice.target_1_index, "target_1_index", errors)
    if entry is None or sl is None or t1 is None or errors:
        raise MetaValidationError("; ".join(errors))

    long = direction == Direction.LONG
    risk = (entry - sl.price) if long else (sl.price - entry)
    if risk <= 0:
        errors.append(f"stop {sl.price:.2f} (option {sl.index}) is not {'below' if long else 'above'} the "
                      f"{choice.entry_type.value} entry {entry:.2f}")
    elif mode == ctx.mode and opts.atr:
        lo, hi = settings.risk.sl_min_atr, settings.risk.sl_max_atr
        d_atr = risk / opts.atr
        if d_atr < lo:
            errors.append(f"stop {sl.price:.2f} is {d_atr:.2f} ATR from entry (< minimum {lo} ATR: too tight)")
        elif d_atr > hi:
            errors.append(f"stop {sl.price:.2f} is {d_atr:.2f} ATR from entry (> maximum {hi} ATR: too wide)")
    reward = (t1.price - entry) if long else (entry - t1.price)
    rr = reward / risk if risk > 0 else 0.0
    min_rr = settings.mode(mode).min_rr
    if reward <= 0:
        errors.append(f"target_1 {t1.price:.2f} (option {t1.index}) is not on the profit side of entry {entry:.2f}")
    elif risk > 0 and rr + 1e-9 < min_rr:
        errors.append(f"reward:risk {rr:.2f} to target_1 {t1.price:.2f} with stop {sl.price:.2f} is below the "
                      f"{mode.value} minimum {min_rr}")
    if errors:
        raise MetaValidationError("; ".join(errors))

    t2_price: float | None = None
    if choice.target_2_index is not None:
        if 0 <= choice.target_2_index < len(opts.targets):
            t2 = opts.targets[choice.target_2_index]
            beyond = (t2.price > t1.price) if long else (t2.price < t1.price)
            if beyond:
                t2_price = t2.price
            else:
                notes.append(f"target_2 option {t2.index} ({t2.price:.2f}) is not beyond target_1; dropped")
        else:
            notes.append(f"target_2_index {choice.target_2_index} does not exist; dropped")
    return Resolved(strategy_index=idx, signal=sig, decision=expected, mode=mode, entry_type=choice.entry_type,
                    entry_price=float(entry), stop_loss=sl.price, target_1=t1.price, target_2=t2_price,
                    rr=round(rr, 3), notes=notes)


def suggest_choice(opts: MetaOptions, ctx: StockContext, settings: "Settings") -> MetaOutput | None:
    """Deterministic valid choice: reference setup with the agents' picks, else the nearest valid combo."""
    order = ([opts.reference] if opts.reference is not None else []) + [
        i for i in sorted(opts.usable, key=lambda k: -opts.strategies[k].strength) if i != opts.reference]
    for idx in order:
        sig = opts.strategies[idx]
        etypes = [opts.agent_entry_type if (idx == opts.reference and opts.agent_entry_type) else sig.entry_type]
        etypes += [e for e in (EntryType.LIMIT, EntryType.MARKET) if e not in etypes]

        def _pref(options: list[PriceOption], wanted: list[float | None], rest: bool = True) -> list[int]:
            first = [o.index for w in wanted if w is not None for o in options if abs(o.price - w) <= PRICE_TOL]
            return list(dict.fromkeys(first + ([o.index for o in options] if rest else [])))

        agent_ok = idx == opts.reference
        stops = _pref(opts.stops, [opts.agent_stop if agent_ok else None, sig.stop_loss])
        t1s = _pref(opts.targets, [opts.agent_targets[0] if agent_ok else None, sig.target_1])
        for etype in etypes:
            for s_i in stops:
                for t_i in t1s:
                    choice = MetaOutput(decision=_DECISION_FOR[sig.direction], strategy_index=idx, entry_type=etype,
                                        stop_loss_index=s_i, target_1_index=t_i, confidence=0)
                    try:
                        res = resolve_choice(choice, opts, ctx, settings, [ctx.mode])
                    except MetaValidationError:
                        continue
                    assert res is not None
                    t2_pref = [opts.agent_targets[1] if agent_ok else None, sig.target_2]
                    for t2 in _pref(opts.targets, t2_pref, rest=False):
                        price = opts.targets[t2].price
                        if (price > res.target_1) if sig.direction == Direction.LONG else (price < res.target_1):
                            choice = choice.model_copy(update={"target_2_index": t2})
                            break
                    return choice
    return None


def bound_output_model(check: Callable[[MetaOutput], None]) -> type[MetaOutput]:
    """``MetaOutput`` subclass whose validation also runs ``check`` (re-prompt on rule errors)."""

    class MetaAnswer(MetaOutput):
        @model_validator(mode="after")
        def _validate_choice(self) -> "MetaAnswer":
            check(self)  # raises MetaValidationError (a ValueError) -> Pydantic ValidationError
            return self

    return MetaAnswer


# --------------------------------------------------------------------------- deterministic pieces


def holding_and_trailing(mode: Mode, strategy_index: int, opts: MetaOptions, ctx: StockContext,
                         settings: "Settings") -> tuple[TrailingMethod, int]:
    """Trailing rule and max holding days for the final mode (exit_target agent's when it applies)."""
    cfg = settings.mode(mode)
    allowed = ALLOWED_TRAILING[mode]
    sig = opts.strategies[strategy_index]
    same = mode == ctx.mode and strategy_index == opts.reference
    if same and opts.agent_trailing in allowed:
        trailing = opts.agent_trailing
    elif sig.trailing_rule in allowed and sig.mode == mode:
        trailing = sig.trailing_rule
    elif cfg.trailing in allowed:
        trailing = cfg.trailing
    else:
        trailing = TrailingMethod.NONE
    lo, hi = cfg.min_holding_days, cfg.max_holding_days
    if same and opts.agent_holding_days is not None:
        days = opts.agent_holding_days
    elif sig.max_holding_days is not None and sig.mode == mode:
        days = sig.max_holding_days
    else:
        days = hi
    return trailing, int(max(lo, min(hi, days)))  # type: ignore[return-value]


def time_validity_minutes(mode: Mode, sig: StrategySignal, as_of: datetime) -> int | None:
    """Minutes the decision stays valid: the strategy's ``valid_until``, else per-mode defaults.

    INTRADAY 15 min; BTST until 15:20 IST; SWING / POSITIONAL None (valid for the next
    session's CNC entry window - expiry is handled by the approval queue).
    """
    now = to_ist(as_of)
    if sig.valid_until is not None:
        mins = math.ceil((to_ist(sig.valid_until) - now).total_seconds() / 60)
        if mins >= 1:
            return int(mins)
    if mode == Mode.INTRADAY:
        return INTRADAY_VALIDITY_MINUTES
    if mode == Mode.BTST:
        cutoff = datetime.combine(now.date(), parse_hhmm(BTST_ENTRY_CUTOFF), tzinfo=now.tzinfo)
        return max(1, math.ceil((cutoff - now) / timedelta(minutes=1)))
    return None


def deterministic_confidence(ctx: StockContext, sig: StrategySignal, vote: VoteSummary | None,
                             debate: DebateResult | None) -> int:
    """0.6 x confluence + 0.4 x direction-aligned vote (mapped to 0-100) +/- debate balance / 10."""
    c = ctx.confluence.score if ctx.confluence is not None else 50.0
    sign = 1.0 if sig.direction == Direction.LONG else -1.0
    aligned = max(-100.0, min(100.0, sign * (vote.score if vote is not None else 0.0)))
    conf = 0.6 * c + 0.4 * (aligned + 100.0) / 2.0
    if debate is not None and debate.status == AgentStatus.OK and debate.bull and debate.bear:
        conf += (debate.bull.conviction - debate.bear.conviction) / 10.0
    return int(max(0, min(100, round(conf))))


def _shadow_prices(res: Resolved | None, opts: MetaOptions | None, ctx: StockContext,
                   settings: "Settings") -> dict[str, Any]:
    """Would-be prices for a NO_TRADE (shadow tracking): resolved > suggested > strongest signal."""
    if res is None and opts is not None:
        sug = suggest_choice(opts, ctx, settings)
        if sug is not None:
            try:
                res = resolve_choice(sug, opts, ctx, settings, [ctx.mode])
            except MetaValidationError:
                res = None
    if res is not None:
        return {"strategy": res.signal.strategy, "entry_price": res.entry_price, "entry_type": res.entry_type,
                "stop_loss": res.stop_loss, "target_1": res.target_1, "target_2": res.target_2}
    best = strongest_usable(ctx)
    if best is None:
        return {}
    s = best[1]
    return {"strategy": s.strategy, "entry_price": s.entry_price, "entry_type": s.entry_type,
            "stop_loss": s.stop_loss, "target_1": s.target_1, "target_2": s.target_2}


def _no_trade(ctx: StockContext, why: str, settings: "Settings", *, failed: bool = False,
              res: Resolved | None = None, opts: MetaOptions | None = None, **kw: Any) -> MetaDecision:
    """NO_TRADE with shadow prices; ``kw`` sets extra MetaDecision fields."""
    base = MetaDecision.no_trade(ctx.symbol, ctx.mode, truncate_text(why, 1000), failed=failed)
    try:
        prices = _shadow_prices(res, opts, ctx, settings)
    except Exception as exc:  # noqa: BLE001 - shadow prices are best effort
        logger.warning(f"meta: shadow prices unavailable for {ctx.symbol}: {exc}")
        prices = {}
    return base.model_copy(update={**prices, **kw})


def _vote_crosscheck(conf: int, direction: Direction, vote: VoteSummary | None, notes: list[str]) -> int:
    if vote is not None and vote.signal == OPPOSITE[DIRECTION_SIGNAL[direction]] and conf > VOTE_OPPOSES_CONF_CAP:
        notes.append(f"weighted vote {vote.score:+.1f} opposes the {direction.value} trade; confidence capped "
                     f"at {VOTE_OPPOSES_CONF_CAP}")
        return VOTE_OPPOSES_CONF_CAP
    return conf


def _trade_decision(res: Resolved, ctx: StockContext, opts: MetaOptions, settings: "Settings", confidence: int,
                    reasons_for: list[str], reasons_against: list[str], *, recommended_reason: str | None,
                    model: str | None, prompt_version: int | None) -> MetaDecision:
    """Trade decision, or NO_TRADE when the confidence is below the final mode's threshold."""
    threshold = settings.mode(res.mode).confidence_threshold
    if confidence < threshold:
        return _no_trade(ctx, f"confidence {confidence} is below the {res.mode.value} threshold {threshold}",
                         settings, res=res, confidence=confidence, reasons_for=reasons_for,
                         reasons_against_considered=reasons_against, model=model, prompt_version=prompt_version)
    trailing, days = holding_and_trailing(res.mode, res.strategy_index, opts, ctx, settings)
    return MetaDecision(
        decision=res.decision, symbol=ctx.symbol, mode=res.mode,
        recommended_mode_reason=recommended_reason, strategy=res.signal.strategy,
        entry_price=round(res.entry_price, 4), entry_type=res.entry_type, stop_loss=round(res.stop_loss, 4),
        target_1=round(res.target_1, 4), target_2=round(res.target_2, 4) if res.target_2 is not None else None,
        trailing_rule=trailing, max_holding_days=days, confidence=confidence,
        time_validity_minutes=time_validity_minutes(res.mode, res.signal, ctx.as_of),
        reasons_for=reasons_for, reasons_against_considered=reasons_against, status=AgentStatus.OK,
        model=model, prompt_version=prompt_version,
    )


# --------------------------------------------------------------------------- rules backend


def deterministic_meta(
    ctx: StockContext,
    reports: Sequence[AgentReport],
    vote: VoteSummary | None,
    debate: DebateResult | None,
    settings: "Settings",
    *,
    enabled_modes: Sequence[Mode] | None = None,
) -> MetaDecision:
    """Rule-based meta decision (``rules`` backend; also used when the meta agent is disabled).

    Strongest usable strategy signal, if the gates pass (confluence grade >= mode minimum,
    TRADEABLE, no blocking event), the weighted vote is aligned with the direction and a
    valid stop / target combination exists (agents' picks first, then the signal's own).
    Confidence: ``deterministic_confidence``. No mode switching (FIXED behaviour).
    """
    gate = trade_gate(ctx, reports, vote, settings)
    if gate is not None:
        return _no_trade(ctx, gate, settings, model="rules")
    opts = build_options(ctx, reports)
    sug = suggest_choice(opts, ctx, settings)
    if sug is None:
        return _no_trade(ctx, "no option combination satisfies the stop side, ATR distance and min R:R",
                         settings, opts=opts, model="rules")
    res = resolve_choice(sug, opts, ctx, settings, [ctx.mode])
    assert res is not None
    want = DIRECTION_SIGNAL[res.signal.direction]
    if vote is None or vote.signal != want:
        score = f"{vote.score:+.1f}" if vote is not None else "unavailable"
        return _no_trade(ctx, f"weighted vote {score} is not aligned with the {res.signal.direction.value} setup",
                         settings, res=res, model="rules")
    conf = deterministic_confidence(ctx, res.signal, vote, debate)
    reasons_for = [f"{res.signal.strategy} {res.signal.direction.value}: entry {res.entry_price:.2f}, stop "
                   f"{res.stop_loss:.2f}, target {res.target_1:.2f} ({res.rr:.2f}R)"]
    if ctx.confluence is not None:
        reasons_for.append(f"confluence {ctx.confluence.score:.1f} grade {ctx.confluence.grade.value}")
    reasons_for.append(f"weighted vote {vote.score:+.1f} ({vote.bullish} bullish / {vote.bearish} bearish)")
    against: list[str] = []
    if debate is not None and debate.status == AgentStatus.OK and debate.bear:
        against = list(debate.bear.top_arguments[:3])
    if vote.failed:
        against.append(f"{vote.failed} agent(s) failed: {', '.join(vote.failed_agents[:4])}")
    return _trade_decision(res, ctx, opts, settings, conf, reasons_for, truncate_list(against, MAX_REASONS,
                           MAX_REASON_CHARS), recommended_reason=None, model="rules", prompt_version=None)


# --------------------------------------------------------------------------- prompt payloads


def debate_digest(debate: DebateResult | None) -> dict[str, Any]:
    """The debate in prompt form."""
    if debate is None:
        return {"status": "NOT_RUN"}
    out: dict[str, Any] = {"status": debate.status.value, "rounds": debate.rounds}
    if debate.status != AgentStatus.OK:
        out["note"] = debate.error
        return out
    for name, side in (("bull", debate.bull), ("bear", debate.bear)):
        if side is not None:
            out[name] = {"stance": side.stance, "top_arguments": side.top_arguments[:5],
                         "rebuttals": side.rebuttals[:3], "conviction": side.conviction}
    return out


def meta_facts(ctx: StockContext, reports: Sequence[AgentReport], vote: VoteSummary | None,
               debate: DebateResult | None, settings: "Settings", enabled_modes: Sequence[Mode],
               opts: MetaOptions) -> dict[str, Any]:
    """Deterministic facts for the meta prompt."""
    cfg = settings.mode(ctx.mode)
    ref = opts.strategies[opts.reference] if opts.reference is not None else None
    conf = ctx.confluence
    facts: dict[str, Any] = {
        "symbol": ctx.symbol, "mode": ctx.mode.value, "ltp": r2(opts.ltp), "atr": r2(opts.atr),
        "regime": ctx.regime.value, "sector": ctx.sector,
        "confluence": ({"score": r2(conf.score, 1), "grade": conf.grade.value,
                        "top_components": dict(sorted(conf.components.items(), key=lambda kv: -kv[1])[:6])}
                       if conf else None),
        "mode_settings": {"min_rr": cfg.min_rr, "confidence_threshold": cfg.confidence_threshold,
                          "min_grade": cfg.min_grade.value,
                          "allowed_decisions": [_DECISION_FOR[d].value for d in (Direction.LONG, Direction.SHORT)
                                                if direction_allowed(ctx.mode, d)] + [TradeDecision.NO_TRADE.value]},
        "weighted_vote": vote_digest(vote),
        "agents": agent_digest(reports),
        "debate": debate_digest(debate),
        "market": market_digest(ctx),
        "costs_reference_setup": setup_costs(ref, settings) if ref else None,
        "overnight_and_event_risk": overnight_risk(ctx, ref, reports, vote, settings),
        "capital_summary": dict(ctx.capital_summary),
    }
    if settings.modes.selection_policy == SelectionPolicy.AUTO:
        facts["enabled_modes"] = {
            m.value: {"min_rr": settings.mode(m).min_rr, "confidence_threshold": settings.mode(m).confidence_threshold,
                      "holding_days": [settings.mode(m).min_holding_days, settings.mode(m).max_holding_days],
                      "allows_short": direction_allowed(m, Direction.SHORT)}
            for m in enabled_modes
        }
    return to_jsonable(facts)


def options_payload(opts: MetaOptions, suggested: MetaOutput | None, ctx: StockContext,
                    settings: "Settings") -> dict[str, Any]:
    """The option lists (with code-computed distances / R:R) the LLM chooses from."""
    sug_res: Resolved | None = None
    if suggested is not None:
        try:
            sug_res = resolve_choice(suggested, opts, ctx, settings, [ctx.mode])
        except MetaValidationError:
            sug_res = None
    entry = sug_res.entry_price if sug_res else (opts.strategies[opts.reference].entry_price
                                                 if opts.reference is not None else opts.ltp)
    stop = sug_res.stop_loss if sug_res else None
    strategies = []
    for i in opts.usable:
        s = opts.strategies[i]
        d = setup_digest(s)
        d.update({"index": i, "decision_if_chosen": _DECISION_FOR[s.direction].value,
                  "market_entry_price": r2(opts.ltp), "limit_entry_price": r2(opts.limit_entries.get(i))})
        strategies.append(d)
    stops = []
    for o in opts.stops:
        item: dict[str, Any] = {"index": o.index, "price": r2(o.price), "sources": list(o.sources)}
        if entry:
            item["distance_from_suggested_entry"] = r2(abs(entry - o.price), 4)
            if opts.atr:
                item["distance_atr"] = r2(abs(entry - o.price) / opts.atr, 2)
        stops.append(item)
    targets = []
    for o in opts.targets:
        item = {"index": o.index, "price": r2(o.price), "sources": list(o.sources)}
        if entry and stop is not None and abs(entry - stop) > 0:
            item["rr_with_suggested_entry_and_stop"] = r2(abs(o.price - entry) / abs(entry - stop), 2)
        targets.append(item)
    out: dict[str, Any] = {
        "strategies": strategies,
        "excluded_strategies": [{"index": i, "strategy": opts.strategies[i].strategy, "reason": why}
                                for i, why in opts.excluded.items()],
        "entry_types": {"MARKET": f"enter now at the LTP {r2(opts.ltp)}",
                        "LIMIT": "resting order at the chosen strategy's limit_entry_price"},
        "stop_loss_options": stops,
        "target_options": targets,
        "suggested_choice": None,
    }
    if suggested is not None and sug_res is not None:
        out["suggested_choice"] = {
            "decision": suggested.decision.value, "strategy_index": suggested.strategy_index,
            "entry_type": suggested.entry_type.value, "stop_loss_index": suggested.stop_loss_index,
            "target_1_index": suggested.target_1_index, "target_2_index": suggested.target_2_index,
            "entry_price": r2(sug_res.entry_price), "stop_loss": r2(sug_res.stop_loss),
            "target_1": r2(sug_res.target_1), "rr_to_target_1": r2(sug_res.rr, 2),
        }
    return to_jsonable(out)


# --------------------------------------------------------------------------- entry point


class _FixedClock:
    """Clock frozen at the analysis instant (JSONL lands in the analysed day's folder)."""

    def __init__(self, at: datetime) -> None:
        self._at = to_ist(at)

    def now(self) -> datetime:
        return self._at


def _log(ctx: StockContext, decision: MetaDecision, notes: list[str]) -> None:
    log_event("meta", {"symbol": ctx.symbol, "mode": ctx.mode.value, "run_id": ctx.run_id, "batch_id": ctx.batch_id,
                       "decision": decision.model_dump(mode="json"), "validation_notes": notes},
              _FixedClock(ctx.as_of))


def run_meta(
    ctx: StockContext,
    reports: Sequence[AgentReport],
    vote: VoteSummary | None,
    debate: DebateResult | None,
    settings: "Settings",
    llm: "LLMClient",
    prompts: "PromptStore | None",
    *,
    enabled_modes: Sequence[Mode],
) -> MetaDecision:
    """Meta decision for ``ctx`` (see module docstring). NEVER raises; failures -> NO_TRADE(failed)."""
    t0 = time.perf_counter()
    notes: list[str] = []
    meta_info: dict[str, Any] = {}
    opts: MetaOptions | None = None

    def _done(decision: MetaDecision) -> MetaDecision:
        decision = decision.model_copy(update={"latency_ms": int((time.perf_counter() - t0) * 1000)})
        try:
            _log(ctx, decision, notes)
        except Exception as exc:  # noqa: BLE001 - logging must never break the pipeline
            logger.error(f"Could not log meta decision for {ctx.symbol}: {exc}")
        return decision

    try:
        gate = trade_gate(ctx, reports, vote, settings)
        if gate is not None:
            return _done(_no_trade(ctx, gate, settings))
        opts = build_options(ctx, reports)
        suggested = suggest_choice(opts, ctx, settings)
        if suggested is None:
            return _done(_no_trade(ctx, "no option combination satisfies the stop side, ATR distance and min R:R",
                                   settings, opts=opts))
        if getattr(llm, "backend", None) == "rules":
            return _done(deterministic_meta(ctx, reports, vote, debate, settings, enabled_modes=enabled_modes))
        if not settings.agents.enabled.get(META_AGENT, True):
            notes.append("meta agent disabled in settings: deterministic decision")
            return _done(deterministic_meta(ctx, reports, vote, debate, settings, enabled_modes=enabled_modes))
        if prompts is None:
            raise RuntimeError("no PromptStore available for the LLM backend")

        modes_enabled = list(enabled_modes)
        model = settings.models.model_for(META_AGENT)
        tpl = prompts.get(META_AGENT, settings.agents.prompt_versions.get(META_AGENT))
        meta_info = {"model": model, "prompt_version": tpl.version}
        variables = {
            "agent": META_AGENT, "symbol": ctx.symbol, "mode": ctx.mode.value, "as_of": ctx.as_of.isoformat(),
            "mode_rules": mode_rules_text(ctx.mode, settings), "output_schema": output_schema_summary(MetaOutput),
            "facts_json": json.dumps(meta_facts(ctx, reports, vote, debate, settings, modes_enabled, opts),
                                     ensure_ascii=False, indent=1),
            "options_json": json.dumps(options_payload(opts, suggested, ctx, settings), ensure_ascii=False, indent=1),
            "selection_policy": settings.modes.selection_policy.value,
            "enabled_modes": ", ".join(m.value for m in modes_enabled) or "none",
        }
        system, user = prompts.render_prompt(tpl, **variables)
        assert opts is not None
        checked_opts = opts

        def _check(choice: MetaOutput) -> None:
            resolve_choice(choice, checked_opts, ctx, settings, modes_enabled)

        res = llm.generate(META_AGENT, model, system, user, bound_output_model(_check),
                           temperature=settings.models.temperature, think=settings.models.reasoning_think)
        meta_info = {"model": res.model or model, "prompt_version": tpl.version}
        if res.skipped:
            return _done(deterministic_meta(ctx, reports, vote, debate, settings, enabled_modes=enabled_modes))
        if not res.ok or res.parsed is None:
            notes.extend(res.errors[-3:])
            return _done(_no_trade(ctx, f"meta agent FAILED: {res.error or 'no parsed output'}", settings,
                                   failed=True, opts=opts, **meta_info))

        out: MetaOutput = res.parsed
        resolved = resolve_choice(out, opts, ctx, settings, modes_enabled)
        reasons_for = list(out.reasons_for)
        against = list(out.reasons_against_considered)
        if resolved is None:  # the LLM chose NO_TRADE
            return _done(_no_trade(ctx, out.why_not or "meta agent chose NO_TRADE", settings, opts=opts,
                                   confidence=out.confidence, reasons_for=reasons_for,
                                   reasons_against_considered=against, **meta_info))
        notes.extend(resolved.notes)
        conf = _vote_crosscheck(out.confidence, resolved.signal.direction, vote, notes)
        auto = settings.modes.selection_policy == SelectionPolicy.AUTO
        decision = _trade_decision(
            resolved, ctx, opts, settings, conf, reasons_for, against,
            recommended_reason=(out.recommended_mode_reason if auto else None), **meta_info)
        logger.info(f"Meta {ctx.symbol}/{ctx.mode.value}: {decision.decision.value} ({decision.mode.value}) "
                    f"confidence {decision.confidence}" + (f" - {decision.why_not}" if decision.why_not else ""))
        return _done(decision)
    except Exception as exc:  # noqa: BLE001 - graceful degradation: any failure -> NO_TRADE (section 16)
        if isinstance(exc, MetaValidationError):
            logger.warning(f"Meta decision rejected for {ctx.symbol}/{ctx.mode.value}: {exc}")
        else:
            logger.exception(f"Meta agent FAILED for {ctx.symbol}/{ctx.mode.value}: {exc}")
        return _done(_no_trade(ctx, f"meta agent FAILED: {type(exc).__name__}: {exc}", settings, failed=True,
                               opts=opts, **meta_info))
