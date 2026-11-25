"""Layer-1 sub-agent base class (section 4).

A sub-agent is *deterministic feature interpretation + optional LLM interpretation*:

1. ``build_payload(ctx)`` - the facts, computed in Python (rule 0.3: the LLM never computes
   numbers; every number, comparison and percentage it sees is pre-computed here).
2. ``deterministic_report(ctx)`` - a rule-based reading of the same facts. Used as the whole
   answer for ``llm_backend: rules`` and as a cross-check for the LLM answer.
3. LLM call with the agent's versioned prompt; answer validated by ``OutputModel``.
4. ``finalize(ctx, parsed)`` - code validation of the LLM answer (clamp, reject numbers that
   are not among the deterministic candidates, enforce code-owned verdicts).

``run`` NEVER raises: any failure produces a ``FAILED`` report that votes NEUTRAL with
confidence 0 (rule 0.4). Code-owned safety facts (e.g. fundamental TRADEABLE=false, chosen
stop-loss candidate) are kept in a FAILED report because they are not LLM opinions.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from agents.prompt_store import PromptRenderError
from agents.schemas import (
    MAX_FINDING_CHARS,
    MAX_FINDINGS,
    MAX_REASONING_CHARS,
    AgentOutput,
    output_schema_summary,
    truncate_list,
    truncate_text,
)
from core.enums import AgentStatus, Direction, Mode, Signal, Timeframe
from core.logging import to_jsonable
from core.types import AgentReport, StockContext, StrategySignal
from modes.base import get_mode_spec

if TYPE_CHECKING:  # pragma: no cover
    from agents.llm import LLMClient
    from agents.prompt_store import PromptStore
    from config.schema import ModeConfig, Settings

#: Signal implied by a setup direction.
DIRECTION_SIGNAL: dict[Direction, Signal] = {Direction.LONG: Signal.BULLISH, Direction.SHORT: Signal.BEARISH}
OPPOSITE: dict[Signal, Signal] = {
    Signal.BULLISH: Signal.BEARISH, Signal.BEARISH: Signal.BULLISH, Signal.NEUTRAL: Signal.NEUTRAL,
}
#: Default min R:R when no settings are available (section 4, agent 10).
DEFAULT_MIN_RR: dict[Mode, float] = {Mode.INTRADAY: 1.5, Mode.BTST: 2.0, Mode.SWING: 2.0, Mode.POSITIONAL: 2.0}
#: Intraday timeframes (VWAP / opening range only make sense here).
INTRADAY_TIMEFRAMES: frozenset[str] = frozenset({Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1})


# --------------------------------------------------------------------------- small pure helpers


def r2(x: Any, nd: int = 2) -> float | None:
    """Round a number for display in payloads (None-safe, NaN -> None)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return round(f, nd)


def pct_diff(a: float | None, b: float | None) -> float | None:
    """``(a - b) / b * 100`` rounded to 2 dp; None when not computable."""
    if a is None or b is None or b == 0:
        return None
    return round((float(a) - float(b)) / float(b) * 100.0, 2)


def mode_config(ctx: StockContext, settings: "Settings | None") -> "ModeConfig | None":
    """The ``ModeConfig`` of ``ctx.mode`` (None without settings)."""
    if settings is None:
        return None
    try:
        return settings.mode(ctx.mode)
    except KeyError:
        logger.warning(f"No mode config for {ctx.mode}")
        return None


def mode_timeframes(ctx: StockContext, settings: "Settings | None") -> list[str]:
    """Timeframes of the mode being analysed that are present in ``ctx.features``.

    Order follows the mode config (``settings.mode(m).timeframes``); without settings it
    falls back to ``ctx.mode_settings['timeframes']`` and then the static mode defaults.
    """
    cfg = mode_config(ctx, settings)
    if cfg is not None:
        wanted = [str(t) for t in cfg.timeframes]
    elif ctx.mode_settings.get("timeframes"):
        wanted = [str(t) for t in ctx.mode_settings["timeframes"]]
    else:
        wanted = [str(t) for t in get_mode_spec(ctx.mode).default_timeframes]
    have = ctx.features.timeframes if ctx.features else {}
    return [t for t in wanted if t in have]


def primary_timeframe(ctx: StockContext, settings: "Settings | None") -> str | None:
    """Primary timeframe of the mode (config > features)."""
    cfg = mode_config(ctx, settings)
    if cfg is not None:
        return str(cfg.primary_timeframe)
    if ctx.features is not None:
        return ctx.features.primary_timeframe
    return None


def min_rr_for(ctx: StockContext, settings: "Settings | None") -> float:
    """Minimum reward:risk of the mode (settings, then ``ctx.mode_settings``, then defaults)."""
    cfg = mode_config(ctx, settings)
    if cfg is not None:
        return float(cfg.min_rr)
    if ctx.mode_settings.get("min_rr") is not None:
        return float(ctx.mode_settings["min_rr"])
    return DEFAULT_MIN_RR[ctx.mode]


def current_price(ctx: StockContext) -> float | None:
    """Last traded price (quote first, then features)."""
    if ctx.quote is not None and ctx.quote.ltp > 0:
        return float(ctx.quote.ltp)
    if ctx.features is not None and ctx.features.ltp > 0:
        return float(ctx.features.ltp)
    return None


def reference_setup(ctx: StockContext) -> StrategySignal | None:
    """Strongest strategy signal of ``ctx.mode`` (ties -> higher R:R, then name)."""
    sigs = [s for s in ctx.strategy_signals if s.mode == ctx.mode]
    if not sigs:
        return None
    return sorted(sigs, key=lambda s: (-s.strength, -s.rr, s.strategy))[0]


def mode_rules_text(mode: Mode, settings: "Settings | None") -> str:
    """Short, human-readable rules of ``mode`` injected into every prompt (``mode_rules``)."""
    spec = get_mode_spec(mode)
    dirs = "+".join(sorted(d.value for d in spec.allowed_directions))
    parts = [f"{mode.value}", f"product {spec.product.value}", f"directions {dirs}"]
    if settings is not None:
        cfg = settings.mode(mode)
        if mode == Mode.INTRADAY:
            parts.append(f"holding same day, square-off {settings.risk.intraday_squareoff} IST")
        else:
            parts.append(f"holding {cfg.min_holding_days}-{cfg.max_holding_days} trading days")
        tfs = ", ".join(str(t) for t in cfg.timeframes)
        parts.append(f"timeframes {tfs} (primary {cfg.primary_timeframe})")
        parts.append(f"min reward:risk {cfg.min_rr}")
        parts.append(f"confidence threshold {cfg.confidence_threshold}")
    else:
        parts.append(spec.description)
    if mode != Mode.INTRADAY:
        parts.append("no overnight shorts")
    return "; ".join(parts)


# --------------------------------------------------------------------------- base class


class SubAgent(ABC):
    """Abstract Layer-1 specialist.

    Subclasses set ``name`` (one of ``config.schema.AGENT_NAMES`` or a new registered
    name), ``modes`` (applicable modes), ``OutputModel`` and implement ``build_payload``
    and ``deterministic_report``; most also override ``finalize``.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    modes: ClassVar[frozenset[Mode]] = frozenset(Mode)
    OutputModel: ClassVar[type[AgentOutput]] = AgentOutput
    #: Confidence cap when the LLM signal is the opposite of a confident rule-based signal.
    contradiction_cap: ClassVar[int] = 50
    #: Rule-based confidence at/above which a contradiction triggers the cap.
    contradiction_min_conf: ClassVar[int] = 60

    # ------------------------------------------------------------------ applicability

    def applies_to(self, mode: Mode) -> bool:
        """True if this agent is meaningful for ``mode``."""
        return Mode(mode) in self.modes

    def enabled_for(self, settings: "Settings", mode: Mode) -> tuple[bool, str]:
        """Global switch (``settings.agents.enabled``) and per-mode switch (``agents_enabled``)."""
        if not settings.agents.enabled.get(self.name, True):
            return False, "agent disabled in settings"
        cfg = settings.mode(mode)
        if not cfg.agents_enabled.get(self.name, True):
            return False, f"agent disabled for {mode.value}"
        return True, ""

    # ------------------------------------------------------------------ hooks

    @abstractmethod
    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        """Deterministic facts given to the LLM (JSON-serialisable, numbers pre-computed)."""

    @abstractmethod
    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        """Rule-based interpretation of ``build_payload``'s facts (status OK)."""

    def precheck(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport | None:
        """Return a final report to short-circuit the LLM (e.g. 'no setup'), else None."""
        return None

    def finalize(
        self,
        ctx: StockContext,
        parsed: AgentOutput,
        settings: "Settings | None" = None,
        det: AgentReport | None = None,
    ) -> AgentReport:
        """Code validation of the LLM answer. Default: common fields + rule cross-check."""
        det = det if det is not None else self.deterministic_report(ctx, settings)
        rep = self.report_from_output(ctx, parsed)
        return self.crosscheck(rep, det)

    # ------------------------------------------------------------------ report builders

    def make_report(
        self,
        ctx: StockContext,
        *,
        signal: Signal = Signal.NEUTRAL,
        confidence: int | float = 0,
        key_findings: list[str] | None = None,
        reasoning: str = "",
        tradeable: bool | None = None,
        status: AgentStatus = AgentStatus.OK,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
        data_used: dict[str, Any] | None = None,
    ) -> AgentReport:
        """Build an ``AgentReport`` with clamped confidence and trimmed text."""
        return AgentReport(
            agent=self.name,
            symbol=ctx.symbol,
            mode=ctx.mode,
            timestamp=ctx.as_of,
            signal=Signal(signal),
            confidence=int(max(0, min(100, round(float(confidence))))),
            key_findings=truncate_list(list(key_findings or []), MAX_FINDINGS, MAX_FINDING_CHARS),
            reasoning=truncate_text(reasoning, MAX_REASONING_CHARS),
            tradeable=tradeable,
            status=status,
            error=error,
            extra=dict(extra or {}),
            data_used=dict(data_used or {}),
        )

    def report_from_output(self, ctx: StockContext, parsed: AgentOutput, **overrides: Any) -> AgentReport:
        """Map the common LLM fields onto a report (``overrides`` win)."""
        kw: dict[str, Any] = {
            "signal": parsed.signal,
            "confidence": parsed.confidence,
            "key_findings": list(parsed.key_findings),
            "reasoning": parsed.reasoning,
        }
        kw.update(overrides)
        return self.make_report(ctx, **kw)

    def crosscheck(self, rep: AgentReport, det: AgentReport) -> AgentReport:
        """Record the rule-based view and cap confidence when the LLM flatly contradicts it."""
        extra = dict(rep.extra)
        extra["deterministic"] = {"signal": det.signal.value, "confidence": det.confidence}
        conf = rep.confidence
        if (
            det.signal != Signal.NEUTRAL
            and rep.signal == OPPOSITE[det.signal]
            and det.confidence >= self.contradiction_min_conf
            and conf > self.contradiction_cap
        ):
            conf = self.contradiction_cap
            extra["crosscheck"] = (
                f"LLM {rep.signal.value} contradicts rules {det.signal.value} ({det.confidence}); "
                f"confidence capped at {self.contradiction_cap}"
            )
        return rep.model_copy(update={"extra": extra, "confidence": conf})

    def skipped_report(self, ctx: StockContext, reason: str) -> AgentReport:
        """SKIPPED (not applicable / disabled): neutral, confidence 0."""
        return self.make_report(ctx, status=AgentStatus.SKIPPED, key_findings=[reason], error=None,
                                extra={"skip_reason": reason})

    def failed_report(self, ctx: StockContext, error: str, det: AgentReport | None = None) -> AgentReport:
        """FAILED: NEUTRAL, confidence 0 (rule 0.4). Code-owned facts from ``det`` are kept.

        ``tradeable`` and the deterministic ``extra`` (under ``extra['fallback']``) are kept
        because they are code verdicts, not LLM opinions; the vote itself is neutral.
        """
        extra: dict[str, Any] = {}
        tradeable: bool | None = None
        if det is not None:
            tradeable = det.tradeable
            extra["fallback"] = dict(det.extra)
            extra["deterministic"] = {"signal": det.signal.value, "confidence": det.confidence}
        return self.make_report(
            ctx, status=AgentStatus.FAILED, error=truncate_text(error, 1000), tradeable=tradeable,
            key_findings=[f"{self.name} agent FAILED: neutral vote"], extra=extra,
        )

    # ------------------------------------------------------------------ prompts

    def prompt_vars(self, ctx: StockContext, payload: dict[str, Any], settings: "Settings | None") -> dict[str, Any]:
        """Template variables for this agent's prompt."""
        return {
            "agent": self.name,
            "symbol": ctx.symbol,
            "mode": ctx.mode.value,
            "as_of": ctx.as_of.isoformat(),
            "payload_json": json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
            "mode_rules": mode_rules_text(ctx.mode, settings),
            "output_schema": output_schema_summary(self.OutputModel),
        }

    # ------------------------------------------------------------------ main entry point

    def run(
        self,
        ctx: StockContext,
        llm: "LLMClient",
        settings: "Settings",
        prompts: "PromptStore | None",
    ) -> AgentReport:
        """Analyse ``ctx``. NEVER raises; failures -> FAILED (neutral, confidence 0)."""
        t0 = time.perf_counter()
        det: AgentReport | None = None
        payload: dict[str, Any] = {}
        meta: dict[str, Any] = {}  # model / prompt version / tokens once known

        def _ms() -> int:
            return int((time.perf_counter() - t0) * 1000)

        try:
            if not self.applies_to(ctx.mode):
                return self.skipped_report(ctx, f"not applicable in {ctx.mode.value}")
            enabled, why = self.enabled_for(settings, ctx.mode)
            if not enabled:
                return self.skipped_report(ctx, why)

            payload = to_jsonable(self.build_payload(ctx, settings))
            pre = self.precheck(ctx, settings)
            if pre is not None:
                return pre.model_copy(update={"data_used": payload, "latency_ms": _ms()})

            det = self.deterministic_report(ctx, settings)
            if getattr(llm, "backend", None) == "rules":
                return self._rules(det, payload, _ms())
            if prompts is None:
                raise RuntimeError("no PromptStore available for the LLM backend")

            model = settings.models.model_for(self.name)
            pinned = settings.agents.prompt_versions.get(self.name)
            tpl = prompts.get(self.name, pinned)
            meta = {"model": model, "prompt_version": tpl.version}
            system, user = prompts.render_prompt(tpl, **self.prompt_vars(ctx, payload, settings))
            res = llm.generate(
                self.name, model, system, user, self.OutputModel,
                temperature=settings.models.temperature, think=False,
            )
            meta = {
                "model": res.model or model, "prompt_version": tpl.version, "tokens_in": res.tokens_in,
                "tokens_out": res.tokens_out, "data_used": payload,
            }
            if res.skipped:
                return self._rules(det, payload, _ms())
            if not res.ok or res.parsed is None:
                rep = self.failed_report(ctx, res.error or "LLM returned no parsed output", det)
                extra = dict(rep.extra)
                extra["attempts"] = res.attempts
                return rep.model_copy(update={**meta, "extra": extra, "latency_ms": _ms()})

            rep = self.finalize(ctx, res.parsed, settings, det=det)
            extra = dict(rep.extra)
            extra["attempts"] = res.attempts
            return rep.model_copy(update={**meta, "extra": extra, "status": AgentStatus.OK, "latency_ms": _ms()})
        except Exception as exc:  # noqa: BLE001 - one agent failing must never crash the loop
            if isinstance(exc, PromptRenderError):
                # A broken (dashboard-edited) prompt is a configuration error: no traceback spam.
                logger.error(f"Agent {self.name} FAILED for {ctx.symbol}/{ctx.mode.value}: {exc}")
            else:
                logger.exception(f"Agent {self.name} failed for {ctx.symbol}/{ctx.mode.value}: {exc}")
            try:
                rep = self.failed_report(ctx, f"{type(exc).__name__}: {exc}", det)
            except Exception as inner:  # noqa: BLE001 - last-resort minimal report
                logger.error(f"Agent {self.name} could not build a FAILED report: {inner}")
                rep = AgentReport(agent=self.name or "unknown", symbol=ctx.symbol, mode=ctx.mode,
                                  timestamp=ctx.as_of, status=AgentStatus.FAILED, error=str(exc)[:1000])
            return rep.model_copy(update={**meta, "data_used": payload, "latency_ms": _ms()})

    def _rules(self, det: AgentReport, payload: dict[str, Any], latency_ms: int) -> AgentReport:
        """Report for the 'rules' backend: the deterministic interpretation, status OK."""
        return det.model_copy(update={"model": "rules", "data_used": payload, "latency_ms": latency_ms,
                                      "status": AgentStatus.OK})


class SetupAgent(SubAgent):
    """Base for agents that refine the strongest strategy setup (entry / stop / target).

    Without a strategy setup in ``ctx.strategy_signals`` they return NEUTRAL 'no setup'
    without calling the LLM.
    """

    def precheck(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport | None:
        if reference_setup(ctx) is None:
            return self.make_report(ctx, key_findings=["no setup: no strategy signal for this mode"],
                                    extra={"no_setup": True})
        if current_price(ctx) is None:
            return self.make_report(ctx, key_findings=["no price available"], extra={"no_price": True})
        return None

    @staticmethod
    def restrict_signal(signal: Signal, setup: StrategySignal) -> Signal:
        """Setup-refining agents may only support the setup's direction or stay NEUTRAL."""
        aligned = DIRECTION_SIGNAL[setup.direction]
        return signal if signal in (aligned, Signal.NEUTRAL) else Signal.NEUTRAL
