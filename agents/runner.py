"""Parallel execution of the Layer-1 sub-agents for one stock / mode (section 4).

``run_sub_agents`` runs every registered agent (non-applicable / disabled agents return a
SKIPPED report so the audit trail shows them), logs per-agent latency to the JSONL
category ``agents`` and warns when the stock's wall time exceeds
``settings.models.latency_warn_seconds`` (section 2). It persists nothing; see
``agents.persist.save_agent_reports``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from agents.registry import all_agents
from core.clock import Clock, to_ist
from core.enums import AgentStatus
from core.logging import log_event
from core.types import AgentReport, StockContext

if TYPE_CHECKING:  # pragma: no cover
    from agents.base import SubAgent
    from agents.llm import LLMClient
    from agents.prompt_store import PromptStore
    from config.schema import Settings


class _FixedClock:
    """Clock frozen at the analysis instant (JSONL lands in the analysed day's folder)."""

    def __init__(self, at: datetime) -> None:
        self._at = to_ist(at)

    def now(self) -> datetime:
        return self._at


def _safe_run(agent: "SubAgent", ctx: StockContext, llm: "LLMClient", settings: "Settings",
              prompts: "PromptStore | None") -> AgentReport:
    """``agent.run`` already never raises; this is a belt-and-braces guard for custom agents."""
    try:
        return agent.run(ctx, llm, settings, prompts)
    except Exception as exc:  # noqa: BLE001 - one agent must never crash the stock's analysis
        logger.exception(f"Agent {agent.name} raised out of run(): {exc}")
        return AgentReport(agent=agent.name, symbol=ctx.symbol, mode=ctx.mode, timestamp=ctx.as_of,
                           status=AgentStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:1000])


def run_sub_agents(
    ctx: StockContext,
    settings: "Settings",
    llm: "LLMClient",
    prompts: "PromptStore | None",
    max_workers: int | None = None,
    *,
    agents: Sequence["SubAgent"] | None = None,
    clock: Clock | None = None,
) -> list[AgentReport]:
    """Run the sub-agents for ``ctx`` in parallel and return their reports (registry order).

    Args:
        ctx: the stock / mode context.
        settings: active settings.
        llm: LLM backend (``build_llm``).
        prompts: prompt store (may be None for the rules backend).
        max_workers: thread pool size (default ``settings.engine.max_parallel_agents``).
        agents: override the agent list (default: every registered agent).
        clock: clock for the JSONL log folder (default: frozen at ``ctx.as_of``).

    Returns:
        One ``AgentReport`` per agent; never raises for agent failures.
    """
    agent_list = list(agents) if agents is not None else all_agents()
    workers = max(1, int(max_workers or settings.engine.max_parallel_agents))
    log_clock: Clock = clock or _FixedClock(ctx.as_of)
    t0 = time.perf_counter()

    reports: list[AgentReport] = []
    if agent_list:
        with ThreadPoolExecutor(max_workers=min(workers, len(agent_list)),
                                thread_name_prefix=f"agents-{ctx.symbol}") as pool:
            futures = [pool.submit(_safe_run, a, ctx, llm, settings, prompts) for a in agent_list]
            for agent, fut in zip(agent_list, futures):
                try:
                    reports.append(fut.result())
                except Exception as exc:  # noqa: BLE001 - executor-level failure
                    logger.exception(f"Agent {agent.name} future failed: {exc}")
                    reports.append(AgentReport(agent=agent.name, symbol=ctx.symbol, mode=ctx.mode,
                                               timestamp=ctx.as_of, status=AgentStatus.FAILED,
                                               error=f"{type(exc).__name__}: {exc}"[:1000]))
    wall_s = time.perf_counter() - t0

    for rep in reports:
        log_event("agents", {
            "symbol": ctx.symbol, "mode": ctx.mode.value, "run_id": ctx.run_id, "batch_id": ctx.batch_id,
            "agent": rep.agent, "status": rep.status.value, "signal": rep.signal.value,
            "confidence": rep.confidence, "model": rep.model, "prompt_version": rep.prompt_version,
            "latency_ms": rep.latency_ms, "tokens_in": rep.tokens_in, "tokens_out": rep.tokens_out,
            "error": rep.error,
        }, log_clock)
    counts = {s.value: sum(1 for r in reports if r.status == s) for s in AgentStatus}
    log_event("agents", {
        "symbol": ctx.symbol, "mode": ctx.mode.value, "run_id": ctx.run_id, "batch_id": ctx.batch_id,
        "agent": "_summary", "wall_ms": int(wall_s * 1000), "workers": workers, **counts,
    }, log_clock)

    warn_s = settings.models.latency_warn_seconds
    if wall_s > warn_s:
        slow = sorted(reports, key=lambda r: -(r.latency_ms or 0))[:3]
        logger.warning(
            f"Sub-agents for {ctx.symbol}/{ctx.mode.value} took {wall_s:.1f}s (> {warn_s:.0f}s); slowest: "
            + ", ".join(f"{r.agent}={r.latency_ms}ms" for r in slow)
        )
    else:
        logger.debug(f"Sub-agents for {ctx.symbol}/{ctx.mode.value} finished in {wall_s:.2f}s {counts}")
    return reports
