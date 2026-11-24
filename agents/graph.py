"""LangGraph analysis pipeline: one symbol x one mode -> decision (sections 4, 5, 7).

Graph (checkpointed with an in-memory saver)::

    context -> strategies --(no setup / confluence too low)--> persist -> route -> END
                          \\-> agents -> debate -> meta --(NO_TRADE)--> persist -> route
                                                        \\-> risk -> persist -> route

* context    - features (mode timeframes), market context, quote, fundamentals, FinBERT-scored
               headlines, events in the holding window, sector, read-only capital summary.
* strategies - deterministic strategy signals, SL candidates, confluence (pre-filter: no LLM
               time is spent on setups that cannot pass; "Analyze now" bypasses the filter).
* agents     - the 10 Layer-1 sub-agents in parallel + weighted vote; confluence recomputed
               with the fundamental score and event risk.
* debate     - Bull vs Bear with rebuttal rounds.
* meta       - Meta decision (LLM chooses among deterministic options; code maps to prices).
* risk       - deterministic risk manager (final veto) on the meta proposal.
* persist    - decisions / agent_reports / debates / stocks_analyzed rows + JSONL.
* route      - execution policy (AUTO / MANUAL_APPROVE / SIGNALS_ONLY) or analysis only,
               shadow tracking of everything not traded.

Any failure degrades to NO_TRADE (section 16); the graph never raises to the caller.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TypedDict

from loguru import logger
from sqlalchemy.orm import Session, sessionmaker

from approvals.router import FINAL_ANALYZED, FINAL_FILTERED, FINAL_NO_TRADE, route_decision
from config.schema import HardCaps, Settings
from core.clock import Clock, to_ist
from core.enums import (
    AgentStatus,
    Direction,
    EntryType,
    Mode,
    RejectionSource,
    RunType,
    Timeframe,
    TradeDecision,
    TradeSource,
)
from core.errors import DataUnavailableError
from core.logging import log_event, to_jsonable
from core.types import (
    AgentReport,
    ConfluenceResult,
    DebateResult,
    MarketContext,
    MetaDecision,
    RiskVerdict,
    StockContext,
    StrategySignal,
    TradeProposal,
)
from modes.base import is_overnight
from storage.audit import record_error
from storage.models import DecisionRow, Run, StockAnalyzed


@dataclass
class PipelineServices:
    """Everything the pipeline needs (built once per process by control.bootstrap)."""

    settings: Settings
    caps: HardCaps
    clock: Clock
    session_factory: sessionmaker[Session]
    calendar: Any
    data: Any
    features: Any                     # features.builder.FeatureBuilder
    llm: Any                          # agents.llm.LLMClient
    prompts: Any | None               # agents.prompt_store.PromptStore
    risk: Any                         # risk.manager.RiskManager
    capital: Any                      # capital.accounts.CapitalManager
    broker: Any                       # paper_engine.broker.PaperBroker
    queue: Any                        # approvals.queue.ApprovalQueue
    shadow: Any                       # shadow.tracker.ShadowTracker
    universe: Any | None = None       # data.universe.UniverseService
    events: Any | None = None         # data.events.EventCalendar
    fundamentals: Any | None = None   # data.fundamentals.FundamentalsService
    news: Any | None = None           # data.news.NewsService
    sentiment: Any | None = None      # features.sentiment.FinBERTScorer
    nse: Any | None = None            # data.nse.NSEClient (market breadth / flows)
    notifier: Any | None = None
    market_cache: dict[str, Any] = field(default_factory=dict)


class PipelineState(TypedDict, total=False):
    symbol: str
    mode: str
    batch_id: int
    run_id: int | None
    force: bool
    route: bool
    check_timing: bool
    started: float
    ctx: StockContext
    signals: list[StrategySignal]
    best: StrategySignal | None
    confluence: ConfluenceResult | None
    reports: list[AgentReport]
    vote: Any
    debate: DebateResult | None
    meta: MetaDecision | None
    proposal: TradeProposal | None
    verdict: RiskVerdict | None
    filtered_reason: str | None
    decision_id: int | None
    final_action: str | None
    error: str | None


@dataclass
class PipelineResult:
    """What ``AnalysisPipeline.analyze`` returns."""

    symbol: str
    mode: Mode
    decision_id: int | None
    final_action: str
    meta: MetaDecision | None = None
    verdict: RiskVerdict | None = None
    confluence: ConfluenceResult | None = None
    filtered_reason: str | None = None
    error: str | None = None
    latency_ms: int = 0


# =========================================================================== runs


def open_run(session: Session, run_type: RunType, *, batch_id: int | None, mode: Mode | None, clock: Clock,
             settings_version_id: int | None = None, is_replay: bool = False) -> Run:
    """Create a ``runs`` row (status RUNNING)."""
    run = Run(batch_id=batch_id, run_type=RunType(run_type).value, mode=Mode(mode).value if mode else None,
              started_at=clock.now(), status="RUNNING", settings_version_id=settings_version_id, is_replay=is_replay)
    session.add(run)
    session.flush()
    return run


def close_run(session: Session, run: Run, *, clock: Clock, n_symbols: int = 0, summary: dict[str, Any] | None = None,
              error: str | None = None) -> None:
    """Finish a run row."""
    run.finished_at = clock.now()
    run.status = "FAILED" if error else "DONE"
    run.n_symbols = n_symbols
    run.summary_json = to_jsonable(summary or {})
    run.error = error
    session.flush()


# =========================================================================== pipeline


class AnalysisPipeline:
    """Builds and runs the LangGraph state graph for one symbol/mode at a time."""

    def __init__(self, services: PipelineServices) -> None:
        self.s = services
        self._graph = self._build_graph()

    def update_settings(self, settings: Settings) -> None:
        self.s.settings = settings

    # ------------------------------------------------------------------ graph

    def _build_graph(self) -> Any:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(PipelineState)
        g.add_node("context", self._node_context)
        g.add_node("strategies", self._node_strategies)
        g.add_node("agents", self._node_agents)
        g.add_node("debate", self._node_debate)
        g.add_node("meta", self._node_meta)
        g.add_node("risk", self._node_risk)
        g.add_node("persist", self._node_persist)
        g.add_node("route", self._node_route)
        g.add_edge(START, "context")
        g.add_conditional_edges("context", lambda st: "persist" if st.get("filtered_reason") else "strategies",
                                {"persist": "persist", "strategies": "strategies"})
        g.add_conditional_edges("strategies", lambda st: "persist" if st.get("filtered_reason") else "agents",
                                {"persist": "persist", "agents": "agents"})
        g.add_edge("agents", "debate")
        g.add_edge("debate", "meta")
        g.add_conditional_edges("meta", self._after_meta, {"risk": "risk", "persist": "persist"})
        g.add_edge("risk", "persist")
        g.add_edge("persist", "route")
        g.add_edge("route", END)
        return g.compile(checkpointer=InMemorySaver())

    @staticmethod
    def _after_meta(st: PipelineState) -> str:
        meta = st.get("meta")
        return "risk" if meta is not None and meta.decision != TradeDecision.NO_TRADE else "persist"

    # ------------------------------------------------------------------ public

    def analyze(
        self,
        symbol: str,
        mode: Mode,
        *,
        batch_id: int,
        run_id: int | None = None,
        force: bool = False,
        route: bool = True,
        check_timing: bool = True,
    ) -> PipelineResult:
        """Analyse ``symbol`` for ``mode`` and (optionally) route the decision. Never raises.

        Args:
            force: skip the deterministic pre-filter (owner "Analyze now").
            route: apply the execution policy; False = analysis only (decision ANALYZED).
            check_timing: risk timing rules (False for the evening SWING/POSITIONAL scan).
        """
        t0 = time.perf_counter()
        sym = symbol.strip().upper()
        state: PipelineState = {"symbol": sym, "mode": Mode(mode).value, "batch_id": batch_id, "run_id": run_id,
                                "force": force, "route": route, "check_timing": check_timing, "started": t0}
        cfg = {"configurable": {"thread_id": f"{run_id}:{sym}:{Mode(mode).value}:{to_ist(self.s.clock.now()).isoformat()}"},
               "recursion_limit": 25}
        try:
            out: PipelineState = self._graph.invoke(state, config=cfg)
        except Exception as exc:  # noqa: BLE001 - graceful degradation: NO_TRADE
            logger.error(f"[pipeline] {sym} {Mode(mode).value} crashed: {exc}")
            with self._session() as session:
                record_error(session, "agents.graph", exc, {"symbol": sym, "mode": Mode(mode).value},
                             batch_id=batch_id, clock=self.s.clock)
                session.commit()
            return PipelineResult(sym, Mode(mode), None, FINAL_NO_TRADE, error=str(exc),
                                  latency_ms=int((time.perf_counter() - t0) * 1000))
        latency = int((time.perf_counter() - t0) * 1000)
        if latency > self.s.settings.models.latency_warn_seconds * 1000:
            logger.warning(f"[pipeline] {sym} {Mode(mode).value} full cycle took {latency / 1000:.1f}s "
                           f"(> {self.s.settings.models.latency_warn_seconds:.0f}s)")
        return PipelineResult(
            symbol=sym, mode=Mode(mode), decision_id=out.get("decision_id"),
            final_action=out.get("final_action") or FINAL_NO_TRADE, meta=out.get("meta"),
            verdict=out.get("verdict"), confluence=out.get("confluence"),
            filtered_reason=out.get("filtered_reason"), error=out.get("error"), latency_ms=latency,
        )

    def _session(self) -> Session:
        return self.s.session_factory()

    # ------------------------------------------------------------------ nodes

    def _market(self, now: datetime) -> MarketContext | None:
        key = now.strftime("%Y-%m-%d %H:%M")
        cached = self.s.market_cache.get("market")
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from features.market import build_market_context

            ctx = build_market_context(self.s.data, now, nse=self.s.nse)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[pipeline] market context failed: {exc}")
            ctx = None
        self.s.market_cache["market"] = (key, ctx)
        return ctx

    def _node_context(self, st: PipelineState) -> dict[str, Any]:
        s = self.s
        settings = s.settings
        mode = Mode(st["mode"])
        mcfg = settings.mode(mode)
        now = to_ist(s.clock.now())
        sym = st["symbol"]
        market = self._market(now)
        try:
            features = s.features.build(sym, mode, mcfg, now, market)
        except DataUnavailableError as exc:
            return {"filtered_reason": f"no market data: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"filtered_reason": f"feature build failed: {exc}", "error": traceback.format_exc(limit=3)}
        quote = None
        try:
            quote = s.data.quote(sym)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[pipeline] quote {sym}: {exc}")
        fundamentals = None
        if s.fundamentals is not None:
            try:
                fundamentals = s.fundamentals.snapshot(sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[pipeline] fundamentals {sym}: {exc}")
        headlines: list[Any] = []
        sentiment: float | None = None
        if s.news is not None:
            since = now - (timedelta(hours=24) if mode == Mode.INTRADAY else timedelta(days=7))
            try:
                headlines = list(s.news.headlines(sym, since))
                if s.sentiment is not None and headlines:
                    headlines = s.sentiment.score(headlines)
                from features.sentiment import aggregate

                sentiment = aggregate(headlines)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[pipeline] news/sentiment {sym}: {exc}")
        events: list[Any] = []
        if s.events is not None and s.calendar is not None and is_overnight(mode):
            try:
                events = s.events.events_in_window(sym, mode, now.date(), mcfg.max_holding_days or 1, s.calendar)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[pipeline] events {sym}: {exc}")
        sector = s.universe.sector_of(sym) if s.universe is not None else None
        capital_summary: dict[str, Any] = {}
        try:
            with self._session() as session:
                from paper_engine.views import open_position_views

                snap = s.capital.snapshot(session, st["batch_id"], open_position_views(session, st["batch_id"]),
                                          update_hwm=False)
                session.rollback()   # read-only: never persist side effects from here
            capital_summary = {"trading_balance": round(snap.trading_balance, 2), "reserve": round(snap.reserve, 2),
                               "total_equity": round(snap.total_equity, 2), "tier": snap.tier.value,
                               "drawdown_pct": round(snap.drawdown_pct, 2), "loss_budget": round(snap.loss_budget, 2),
                               "mode_sizing_capital": snap.mode_sizing_capital}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[pipeline] capital summary failed: {exc}")
        ctx = StockContext(
            symbol=sym, mode=mode, as_of=now, batch_id=st["batch_id"], run_id=st.get("run_id"), sector=sector,
            quote=quote, features=features, market=market, fundamentals=fundamentals, headlines=headlines,
            sentiment_score=sentiment, events=events, regime=features.regime, capital_summary=capital_summary,
            mode_settings=mcfg.model_dump(mode="json"),
        )
        return {"ctx": ctx}

    def _disabled(self, batch_id: int) -> set[str]:
        try:
            from risk.state import disabled_strategy_keys

            with self._session() as session:
                return disabled_strategy_keys(session, batch_id)[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[pipeline] strategy states unavailable: {exc}")
            return set()

    def _node_strategies(self, st: PipelineState) -> dict[str, Any]:
        from features.stops import stop_loss_candidates
        from strategies.confluence import compute_confluence
        from strategies.registry import run_strategies

        s = self.s
        ctx = st["ctx"]
        mode = ctx.mode
        mcfg = s.settings.mode(mode)
        signals = run_strategies(ctx.features, ctx.market, mode, s.settings, ctx.as_of, self._disabled(st["batch_id"]))
        best = signals[0] if signals else None
        sl_c = stop_loss_candidates(ctx.features, best.direction, best.entry_price) if best else []
        event_risk = any(e.symbol == ctx.symbol for e in ctx.events)
        conf = compute_confluence(ctx.features, ctx.market, best, mode, mcfg, sentiment=ctx.sentiment_score,
                                  event_risk=event_risk)
        ctx = ctx.model_copy(update={"strategy_signals": signals, "sl_candidates": sl_c, "confluence": conf})
        out: dict[str, Any] = {"ctx": ctx, "signals": signals, "best": best, "confluence": conf}
        if not st.get("force"):
            if best is None:
                out["filtered_reason"] = "no strategy setup"
            elif conf.score < mcfg.min_confluence:
                out["filtered_reason"] = f"confluence {conf.score:.1f} < {mcfg.min_confluence} (grade {conf.grade.value})"
        return out

    def _node_agents(self, st: PipelineState) -> dict[str, Any]:
        from agents.aggregate import weighted_vote
        from agents.runner import run_sub_agents
        from strategies.confluence import compute_confluence

        s = self.s
        ctx = st["ctx"]
        mcfg = s.settings.mode(ctx.mode)
        reports = run_sub_agents(ctx, s.settings, s.llm, s.prompts, s.settings.engine.max_parallel_agents,
                                 clock=s.clock)
        vote = weighted_vote(reports, mcfg.agent_weights)
        conf = compute_confluence(ctx.features, ctx.market, st.get("best"), ctx.mode, mcfg,
                                  sentiment=ctx.sentiment_score, fundamental_score=vote.fundamental_score,
                                  event_risk=bool(vote.event_risk))
        ctx = ctx.model_copy(update={"confluence": conf})
        return {"ctx": ctx, "reports": reports, "vote": vote, "confluence": conf}

    def _node_debate(self, st: PipelineState) -> dict[str, Any]:
        from agents.debate.debate import run_debate

        s = self.s
        return {"debate": run_debate(st["ctx"], st.get("reports", []), st.get("vote"), s.settings, s.llm, s.prompts)}

    def _node_meta(self, st: PipelineState) -> dict[str, Any]:
        from agents.meta.meta import run_meta

        s = self.s
        meta = run_meta(st["ctx"], st.get("reports", []), st.get("vote"), st.get("debate"), s.settings, s.llm,
                        s.prompts, enabled_modes=s.settings.modes.enabled_modes())
        return {"meta": meta}

    def _atr_for_mode(self, symbol: str, mode: Mode, ctx: StockContext) -> float | None:
        if ctx.mode == mode and ctx.features is not None:
            return ctx.features.atr
        try:
            from features.indicators import atr as atr_fn

            df = self.s.data.candles(symbol, Timeframe(self.s.settings.mode(mode).primary_timeframe), lookback_bars=60)
            if df is None or len(df) < 15:
                return None
            val = float(atr_fn(df, 14).iloc[-1])
            return val if val == val and val > 0 else None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[pipeline] ATR {symbol} {mode.value}: {exc}")
            return None

    def _node_risk(self, st: PipelineState) -> dict[str, Any]:
        from risk.state import build_risk_state

        s = self.s
        ctx = st["ctx"]
        meta = st["meta"]
        mode = Mode(meta.mode)
        vote = st.get("vote")
        conf = st.get("confluence")
        direction = Direction.LONG if meta.decision == TradeDecision.BUY else Direction.SHORT
        events = ctx.events
        if mode != ctx.mode and s.events is not None and is_overnight(mode):
            try:
                events = s.events.events_in_window(ctx.symbol, mode, ctx.as_of.date(),
                                                   s.settings.mode(mode).max_holding_days or 1, s.calendar)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[pipeline] events for AUTO mode {mode.value}: {exc}")
        quote = ctx.quote
        proposal = TradeProposal(
            batch_id=st["batch_id"], symbol=ctx.symbol, mode=mode, direction=direction,
            entry_price=float(meta.entry_price), entry_type=meta.entry_type, stop_loss=float(meta.stop_loss),
            target_1=float(meta.target_1), target_2=meta.target_2, strategy=meta.strategy, confidence=meta.confidence,
            confluence_score=conf.score if conf else None, grade=conf.grade if conf else None, regime=ctx.regime,
            atr=self._atr_for_mode(ctx.symbol, mode, ctx),
            tradeable=None if vote is None else bool(vote.tradeable), source=TradeSource.SYSTEM, sector=ctx.sector,
            trailing_rule=meta.trailing_rule, max_holding_days=meta.max_holding_days,
            upper_circuit=quote.upper_circuit if quote else None, lower_circuit=quote.lower_circuit if quote else None,
            ltp=quote.ltp if quote else None, events_in_window=events, created_at=ctx.as_of,
        )
        with self._session() as session:
            state = build_risk_state(session, batch_id=st["batch_id"], capital_manager=s.capital, clock=s.clock,
                                     calendar=s.calendar, data=s.data, symbol=ctx.symbol,
                                     correlation_lookback=s.settings.risk.correlation_lookback_days)
            session.rollback()
        verdict = s.risk.evaluate(proposal, state, check_timing=st.get("check_timing", True))
        return {"proposal": proposal, "verdict": verdict}

    def _node_persist(self, st: PipelineState) -> dict[str, Any]:
        from agents.persist import save_agent_reports, save_debate

        s = self.s
        ctx: StockContext | None = st.get("ctx")
        mode = Mode(st["mode"])
        meta = st.get("meta")
        verdict = st.get("verdict")
        conf = st.get("confluence")
        best = st.get("best")
        now = to_ist(s.clock.now())
        decision_kind = meta.decision if meta is not None else TradeDecision.NO_TRADE
        final = FINAL_FILTERED if st.get("filtered_reason") else FINAL_NO_TRADE
        latency = int((time.perf_counter() - st.get("started", time.perf_counter())) * 1000)
        vote = st.get("vote")
        context_json = {
            "filtered_reason": st.get("filtered_reason"),
            "signals": [x.model_dump(mode="json") for x in st.get("signals", [])],
            "confluence": conf.model_dump(mode="json") if conf else None,
            "sl_candidates": [c.model_dump(mode="json") for c in (ctx.sl_candidates if ctx else [])],
            "vote": vote.model_dump(mode="json") if vote is not None else None,
            "tradeable": None if vote is None else bool(vote.tradeable),
            "market": ctx.market.model_dump(mode="json") if ctx and ctx.market else None,
            "events": [e.model_dump(mode="json") for e in (ctx.events if ctx else [])],
            "proposal": st["proposal"].model_dump(mode="json") if st.get("proposal") else None,
        }
        with self._session() as session:
            try:
                row = DecisionRow(
                    run_id=st.get("run_id"), batch_id=st["batch_id"], symbol=st["symbol"],
                    mode=(meta.mode.value if meta is not None else mode.value),
                    strategy=(meta.strategy if meta is not None and meta.strategy else (best.strategy if best else None)),
                    regime=ctx.regime.value if ctx else None, confluence_score=conf.score if conf else None,
                    grade=conf.grade.value if conf else None, decision=TradeDecision(decision_kind).value,
                    direction=(best.direction.value if best else None) if meta is None or meta.decision == TradeDecision.NO_TRADE
                    else (Direction.LONG.value if meta.decision == TradeDecision.BUY else Direction.SHORT.value),
                    confidence=meta.confidence if meta is not None else 0,
                    entry_price=meta.entry_price if meta is not None and meta.entry_price else (best.entry_price if best else None),
                    stop_loss=meta.stop_loss if meta is not None and meta.stop_loss else (best.stop_loss if best else None),
                    target_1=meta.target_1 if meta is not None and meta.target_1 else (best.target_1 if best else None),
                    target_2=meta.target_2 if meta is not None and meta.target_2 else (best.target_2 if best else None),
                    meta_json=meta.model_dump(mode="json") if meta is not None else None,
                    context_json=to_jsonable(context_json),
                    risk_approved=verdict.approved if verdict is not None else None,
                    risk_rule=verdict.rule.value if verdict is not None and verdict.rule else None,
                    risk_json=verdict.model_dump(mode="json") if verdict is not None else None,
                    qty=verdict.qty if verdict is not None else None, final_action=final, source="SYSTEM",
                    settings_version_id=None, latency_ms=latency, created_at=now,
                )
                session.add(row)
                session.flush()
                if st.get("reports"):
                    save_agent_reports(session, st["reports"], st.get("run_id"), st["batch_id"], decision_id=row.id)
                if st.get("debate") is not None and st["debate"].status != AgentStatus.SKIPPED:
                    save_debate(session, st["debate"], symbol=st["symbol"], mode=mode, created_at=now,
                                run_id=st.get("run_id"), batch_id=st["batch_id"], decision_id=row.id)
                session.add(StockAnalyzed(
                    run_id=st.get("run_id"), batch_id=st["batch_id"], symbol=st["symbol"], mode=mode.value, as_of=now,
                    stage="ANALYZED", included=st.get("filtered_reason") is None,
                    reason=st.get("filtered_reason") or (meta.why_not if meta is not None and meta.why_not else None),
                    regime=ctx.regime.value if ctx else None, confluence=conf.score if conf else None,
                    grade=conf.grade.value if conf else None,
                    features_json=_feature_digest(ctx),
                ))
                session.commit()
                decision_id = row.id
            except Exception as exc:  # noqa: BLE001
                session.rollback()
                record_error(session, "agents.graph.persist", exc, {"symbol": st["symbol"]}, batch_id=st["batch_id"],
                             clock=s.clock)
                session.commit()
                return {"decision_id": None, "final_action": FINAL_NO_TRADE, "error": f"persist failed: {exc}"}
        log_event("decisions", {"decision_id": decision_id, "symbol": st["symbol"], "mode": mode.value,
                                "decision": TradeDecision(decision_kind).value,
                                "filtered": st.get("filtered_reason"),
                                "confidence": meta.confidence if meta is not None else None,
                                "confluence": conf.score if conf else None,
                                "risk": verdict.rule.value if verdict is not None and verdict.rule else
                                ("APPROVED" if verdict is not None else None),
                                "latency_ms": latency}, clock=s.clock)
        return {"decision_id": decision_id, "final_action": final}

    def _node_route(self, st: PipelineState) -> dict[str, Any]:
        s = self.s
        decision_id = st.get("decision_id")
        if decision_id is None:
            return {}
        meta = st.get("meta")
        best = st.get("best")
        conf = st.get("confluence")
        with self._session() as session:
            try:
                row = session.get(DecisionRow, decision_id)
                proposal, verdict = st.get("proposal"), st.get("verdict")
                if proposal is not None and verdict is not None:
                    if not st.get("route", True):
                        row.final_action = FINAL_ANALYZED
                    else:
                        summary = _approval_summary(st)
                        final = route_decision(session, settings=s.settings, decision=row, proposal=proposal,
                                               verdict=verdict, broker=s.broker, queue=s.queue, shadow=s.shadow,
                                               notifier=s.notifier, summary=summary, clock=s.clock)
                        row.final_action = final
                else:
                    reason = st.get("filtered_reason") or (meta.why_not if meta is not None else None)
                    if not st.get("route", True):
                        row.final_action = FINAL_ANALYZED
                    if best is not None and st.get("route", True) and s.shadow is not None:
                        source = RejectionSource.FILTER if st.get("filtered_reason") else RejectionSource.META
                        s.shadow.create(session, batch_id=st["batch_id"], symbol=st["symbol"],
                                        mode=best.mode, direction=best.direction, entry=best.entry_price,
                                        stop_loss=best.stop_loss, target_1=best.target_1, target_2=best.target_2,
                                        source=source, reason=reason, decision_id=decision_id, strategy=best.strategy,
                                        grade=conf.grade.value if conf else None,
                                        confidence=meta.confidence if meta is not None else None,
                                        entry_type=best.entry_type, max_holding_days=best.max_holding_days)
                session.commit()
                return {"final_action": row.final_action}
            except Exception as exc:  # noqa: BLE001
                session.rollback()
                record_error(session, "agents.graph.route", exc, {"decision_id": decision_id},
                             batch_id=st["batch_id"], clock=s.clock)
                session.commit()
                return {"final_action": FINAL_NO_TRADE, "error": f"route failed: {exc}"}


def _feature_digest(ctx: StockContext | None) -> dict[str, Any] | None:
    if ctx is None or ctx.features is None:
        return None
    f = ctx.features
    primary = f.tf(f.primary_timeframe)
    keys = ("close", "ema_9", "ema_21", "ema_50", "ema_200", "vwap", "rsi_14", "adx_14", "atr_14", "rvol", "supertrend_dir")
    return to_jsonable({
        "ltp": f.ltp, "atr": f.atr, "gap_pct": f.gap_pct, "rvol": f.rvol, "regime": f.regime.value,
        "rs_vs_nifty": f.rs_vs_nifty, "primary": {k: primary.v(k) for k in keys} if primary else None,
    })


def _approval_summary(st: PipelineState) -> dict[str, Any]:
    """Top bull/bear points etc. for the approval queue + email."""
    debate = st.get("debate")
    meta = st.get("meta")
    conf = st.get("confluence")
    verdict = st.get("verdict")
    return to_jsonable({
        "bull": (debate.bull.top_arguments[:3] if debate is not None and debate.bull else []),
        "bear": (debate.bear.top_arguments[:3] if debate is not None and debate.bear else []),
        "reasons_for": meta.reasons_for[:5] if meta is not None else [],
        "reasons_against": meta.reasons_against_considered[:5] if meta is not None else [],
        "confluence": conf.score if conf else None,
        "grade": conf.grade.value if conf else None,
        "charges": verdict.charges_estimate.total if verdict is not None and verdict.charges_estimate else None,
        "expected_net_t1": verdict.expected_net_at_t1 if verdict is not None else None,
    })


__all__ = ["AnalysisPipeline", "PipelineResult", "PipelineServices", "close_run", "open_run"]
