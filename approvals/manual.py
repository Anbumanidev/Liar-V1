"""Owner's manual paper trades (section 14.3.4).

A manual trade is built into a ``TradeProposal`` tagged ``MANUAL`` and goes through the SAME
deterministic risk manager as system trades (mandatory SL, R:R, sizing caps, capital
protection ...). Agent-only gates (confidence / confluence / strategy) do not apply, but if
the owner chose "Analyze first", the latest analysis (grade, confluence, TRADEABLE) is
attached so tier and quality rules can use it. Blocked -> the exact rule is returned.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.schema import Settings
from control.payloads import CommandResult, ManualTradePayload
from core.clock import Clock, to_ist
from core.enums import EntryType, Grade, Mode, Regime, Timeframe, TradeSource
from core.types import MarketEvent, TradeProposal
from modes.base import is_overnight
from risk.state import build_risk_state
from storage.audit import audit
from storage.models import DecisionRow

EventsFn = Callable[[str, Mode, datetime], list[MarketEvent]]


class ManualTrades:
    """Builds, vets and submits the owner's manual paper trades."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        calendar: Any,
        data: Any,
        risk: Any,
        broker: Any,
        capital: Any,
        *,
        sector_fn: Callable[[str], str | None] | None = None,
        events_fn: EventsFn | None = None,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.calendar = calendar
        self.data = data
        self.risk = risk
        self.broker = broker
        self.capital = capital
        self.sector_fn = sector_fn
        self.events_fn = events_fn

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    def _atr(self, symbol: str, mode: Mode) -> float | None:
        if self.data is None:
            return None
        try:
            from features.indicators import atr as atr_fn

            df = self.data.candles(symbol, Timeframe(self.settings.mode(mode).primary_timeframe), lookback_bars=60)
            if df is None or len(df) < 15:
                return None
            val = float(atr_fn(df, 14).iloc[-1])
            return val if val == val and val > 0 else None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[manual] ATR {symbol} failed: {exc}")
            return None

    def _ltp(self, symbol: str) -> float | None:
        if self.data is None:
            return None
        try:
            q = self.data.quote(symbol)
            return float(q.ltp) if q is not None and q.ltp else None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[manual] quote {symbol} failed: {exc}")
            return None

    def latest_analysis(self, session: Session, batch_id: int, symbol: str, mode: Mode,
                        within_minutes: int = 60) -> DecisionRow | None:
        """Most recent decision for symbol/mode (e.g. from 'Analyze first') within the window."""
        since = to_ist(self.clock.now()) - timedelta(minutes=within_minutes)
        return session.execute(
            select(DecisionRow).where(DecisionRow.batch_id == batch_id, DecisionRow.symbol == symbol,
                                      DecisionRow.mode == Mode(mode).value, DecisionRow.created_at >= since)
            .order_by(DecisionRow.id.desc()).limit(1)
        ).scalar_one_or_none()

    def build_proposal(self, session: Session, payload: ManualTradePayload, *, batch_id: int) -> TradeProposal:
        """Manual payload -> ``TradeProposal`` (source MANUAL, requested qty)."""
        now = to_ist(self.clock.now())
        symbol = payload.symbol.strip().upper()
        mode = Mode(payload.mode)
        ltp = self._ltp(symbol)
        entry = payload.entry_price if payload.entry_type == EntryType.LIMIT or ltp is None else ltp
        if entry is None:
            raise ValueError("entry price required (no live price available for a MARKET order)")
        analysis = self.latest_analysis(session, batch_id, symbol, mode) if payload.analyze_first else None
        grade = Grade(analysis.grade) if analysis is not None and analysis.grade else None
        tradeable = True
        if analysis is not None and analysis.context_json:
            tradeable = analysis.context_json.get("tradeable", True) is not False
        events: list[MarketEvent] = []
        if is_overnight(mode) and self.events_fn is not None:
            try:
                events = self.events_fn(symbol, mode, now)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[manual] events lookup failed for {symbol}: {exc}")
        return TradeProposal(
            batch_id=batch_id, symbol=symbol, mode=mode, direction=payload.direction, entry_price=float(entry),
            entry_type=payload.entry_type, stop_loss=payload.stop_loss, target_1=payload.target_1,
            target_2=payload.target_2, strategy="manual", confidence=0,
            confluence_score=analysis.confluence_score if analysis is not None else None, grade=grade,
            regime=Regime(analysis.regime) if analysis is not None and analysis.regime else Regime.UNDEFINED,
            atr=self._atr(symbol, mode), tradeable=tradeable, source=TradeSource.MANUAL,
            sector=self.sector_fn(symbol) if self.sector_fn else None, requested_qty=payload.qty,
            max_holding_days=self.settings.mode(mode).max_holding_days or None, ltp=ltp,
            events_in_window=events, decision_id=analysis.id if analysis is not None else None, created_at=now,
        )

    def submit(self, session: Session, payload: ManualTradePayload, *, actor: str, batch_id: int) -> CommandResult:
        """Vet with the risk manager and submit. Never raises."""
        try:
            proposal = self.build_proposal(session, payload, batch_id=batch_id)
        except Exception as exc:  # noqa: BLE001
            return CommandResult(ok=False, message=f"manual trade invalid: {exc}")
        state = build_risk_state(session, batch_id=batch_id, capital_manager=self.capital, clock=self.clock,
                                 calendar=self.calendar, data=self.data, symbol=proposal.symbol,
                                 correlation_lookback=self.settings.risk.correlation_lookback_days)
        check_timing = not is_overnight(proposal.mode) or proposal.mode == Mode.BTST
        verdict = self.risk.evaluate(proposal, state, check_timing=check_timing)
        if not verdict.approved:
            audit(session, actor, "MANUAL_TRADE_BLOCKED", proposal.symbol,
                  {"rule": verdict.rule.value if verdict.rule else None, "message": verdict.message,
                   "payload": payload.model_dump(mode="json")}, batch_id=batch_id, clock=self.clock)
            return CommandResult(ok=False, message=verdict.message, rule=verdict.rule,
                                 data={"max_qty": verdict.max_qty,
                                       "checks": [c.model_dump(mode="json") for c in verdict.checks if not c.passed]})
        order = self.broker.submit_entry(session, proposal, verdict, decision_id=proposal.decision_id,
                                         extra={"manual_note": payload.note})
        audit(session, actor, "MANUAL_TRADE", proposal.symbol,
              {"order_id": order.id, "qty": verdict.qty, "payload": payload.model_dump(mode="json")},
              batch_id=batch_id, clock=self.clock)
        return CommandResult(ok=True, message=f"manual {proposal.direction.value} {verdict.qty} {proposal.symbol} "
                                              f"submitted (order {order.id})",
                             data={"order_id": order.id, "qty": verdict.qty})
