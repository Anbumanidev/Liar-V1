"""Position monitoring loop (sections 7.4, 8, 9).

Called every tick/minute by the engine (and every simulated minute in REPLAY). For each open
position it replays the CLOSED 1-minute bars since the last check, in order, conservatively:

1. stop / gap - a bar opening beyond the stop fills at the open (GAP on the first bar of a new
   session for overnight positions), a touched stop fills at the stop minus slippage. If a bar
   touches both the stop and a target, the stop is assumed to have filled first.
2. partial - book ``partial_exit_pct`` % at ``partial_exit_at_r`` R (default 50 % at 1R) and
   move the stop to breakeven (qty 1: breakeven only).
3. final target - ``target_2`` when a partial plan exists, else ``target_1``.

Then: trailing stops (never loosen), time stops, 15:15 MIS square-off, CNC max-holding time
stop at 15:20, BTST next-morning exit, pre-event exits, owner/HALT pending exits, holding-day
counts, MTM, HWM/tier tracking and the capital-protection HALT (exit everything, block new
trades, urgent email). Positions are evaluated with the settings frozen on them at entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Callable

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.schema import ModeConfig, Settings
from core.clock import Clock, ist_datetime, parse_hhmm, to_ist
from core.enums import (
    BatchStatus,
    Direction,
    DrawdownTier,
    ExitReason,
    Mode,
    PositionStatus,
    Product,
    Timeframe,
    TrailingMethod,
)
from core.logging import log_event
from core.types import MarketEvent
from paper_engine.broker import OrderRejected, PaperBroker, get_mode_state
from paper_engine.fills import Bar, favorable_excursion, round_tick, stop_fill, target_fill
from paper_engine.views import open_position_views
from risk.guards import is_tighter
from risk.state import realized_by_mode
from storage.audit import record_error
from storage.models import Batch, Position

EventsFn = Callable[[str, date, date], list[MarketEvent]]


@dataclass
class MonitorReport:
    """What one monitor pass did."""

    checked: int = 0
    opened: list[int] = field(default_factory=list)
    exits: list[dict[str, Any]] = field(default_factory=list)
    sl_moves: int = 0
    halted_batches: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PositionMonitor:
    """Applies every exit / trade-management rule to open paper positions."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        calendar: Any,
        data: Any,
        broker: PaperBroker,
        capital: Any,
        *,
        events_fn: EventsFn | None = None,
        notifier: Any | None = None,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.calendar = calendar
        self.data = data
        self.broker = broker
        self.capital = capital
        self.events_fn = events_fn
        self.notifier = notifier

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------ entry point

    def check(self, session: Session, now: datetime | None = None, *, batch_ids: list[int] | None = None) -> MonitorReport:
        """One full monitoring pass. Never raises."""
        now = to_ist(now or self.clock.now())
        report = MonitorReport()
        market_open = bool(self.calendar.is_market_open(now)) if self.calendar is not None else True
        if market_open:
            try:
                report.opened = [p.id for p in self.broker.process_pending_orders(session, now)]
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"fills: {exc}")
                record_error(session, "paper_engine.monitor", exc, {"stage": "fills"}, clock=self.clock)
        stmt = select(Position).where(Position.status == PositionStatus.OPEN.value)
        if batch_ids:
            stmt = stmt.where(Position.batch_id.in_(batch_ids))
        positions = list(session.execute(stmt.order_by(Position.id)).scalars())
        for pos in positions:
            report.checked += 1
            try:
                self._check_position(session, pos, now, market_open, report)
            except Exception as exc:  # noqa: BLE001 - one position never stops the loop
                report.errors.append(f"{pos.symbol}: {exc}")
                record_error(session, "paper_engine.monitor", exc, {"position_id": pos.id}, batch_id=pos.batch_id,
                             clock=self.clock)
        for bid in self._batches_to_protect(session, batch_ids):
            try:
                self._protect(session, bid, now, market_open, report)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"protect batch {bid}: {exc}")
                record_error(session, "paper_engine.monitor", exc, {"stage": "protect", "batch_id": bid},
                             batch_id=bid, clock=self.clock)
        session.flush()
        return report

    def _batches_to_protect(self, session: Session, batch_ids: list[int] | None) -> list[int]:
        stmt = select(Batch.id).where(Batch.status.in_([BatchStatus.ACTIVE.value, BatchStatus.PAUSED.value,
                                                        BatchStatus.HALTED.value]), Batch.deleted.is_(False))
        ids = [int(i) for i in session.execute(stmt).scalars()]
        return [i for i in ids if batch_ids is None or i in batch_ids]

    # ------------------------------------------------------------------ per position

    def _mode_cfg(self, pos: Position) -> ModeConfig:
        frozen = (pos.settings_json or {}).get("mode")
        if frozen:
            try:
                return ModeConfig.model_validate(frozen)
            except ValueError as exc:
                logger.warning(f"[monitor] frozen settings of position {pos.id} invalid ({exc}); using current")
        return self.settings.mode(Mode(pos.mode))

    def _notes(self, pos: Position) -> dict[str, Any]:
        return dict(pos.notes_json or {})

    def _set_note(self, pos: Position, key: str, value: Any) -> None:
        notes = self._notes(pos)
        notes[key] = value
        pos.notes_json = notes

    def _exit(self, session: Session, pos: Position, qty: int | None, reason: ExitReason, price: float | None,
              now: datetime, report: MonitorReport, *, apply_slippage: bool = True) -> bool:
        try:
            trade = self.broker.exit_position(session, pos, qty, reason, price=price, now=now,
                                              apply_slippage=apply_slippage)
        except OrderRejected as exc:
            report.errors.append(f"exit {pos.symbol}: {exc}")
            logger.error(f"[monitor] exit {pos.symbol} failed: {exc}")
            return False
        report.exits.append({"position_id": pos.id, "trade_id": trade.id, "reason": trade.exit_reason,
                             "qty": trade.qty, "price": trade.exit_price, "partial": trade.is_partial})
        if reason == ExitReason.GAP and self.notifier is not None:
            try:
                self.notifier.urgent(
                    f"Gap through stop-loss: {pos.symbol}",
                    f"{pos.symbol} ({pos.mode}) opened at {trade.exit_price}, beyond its stop-loss; "
                    f"exited at the open. Net P&L Rs {trade.net_pnl:.2f}.",
                    {"position_id": pos.id, "trade_id": trade.id},
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[monitor] urgent email failed: {exc}")
        return True

    def _move_sl(self, session: Session, pos: Position, new_sl: float, by: str, now: datetime,
                 report: MonitorReport) -> bool:
        """System stop move: only ever tighter (rule 0.10)."""
        new_sl = round_tick(new_sl)
        if not is_tighter(Direction(pos.direction), pos.stop_loss, new_sl):
            return False
        old = pos.stop_loss
        pos.stop_loss = new_sl
        hist = list(pos.sl_history_json or [])
        hist.append({"ts": now.isoformat(), "old": old, "new": new_sl, "by": by})
        pos.sl_history_json = hist
        report.sl_moves += 1
        log_event("orders", {"event": "sl_moved", "position_id": pos.id, "old": old, "new": new_sl, "by": by},
                  clock=self.clock)
        return True

    def _last_sl_mover(self, pos: Position) -> str | None:
        hist = pos.sl_history_json or []
        return hist[-1].get("by") if hist else None

    def _check_position(self, session: Session, pos: Position, now: datetime, market_open: bool,
                        report: MonitorReport) -> None:
        if pos.status != PositionStatus.OPEN.value:
            return
        pos.holding_days = self.broker._holding_days(pos.opened_at, now)
        if not market_open:
            return
        cfg = self._mode_cfg(pos)
        direction = Direction(pos.direction)
        long_ = direction == Direction.LONG
        slip = self.settings.engine.slippage_pct

        last_bar_raw = self._notes(pos).get("last_bar_ts")
        last_bar_ts = to_ist(datetime.fromisoformat(last_bar_raw)) if last_bar_raw else None
        since = last_bar_ts + timedelta(seconds=1) if last_bar_ts else to_ist(pos.opened_at).replace(second=0, microsecond=0)
        bars = self.broker._bars_since(pos.symbol, since)
        prev_day = last_bar_ts.date() if last_bar_ts else to_ist(pos.opened_at).date()

        # owner / HALT exits requested while the market was closed: first available price
        pending = self._notes(pos).get("pending_exit")
        if pending:
            price = bars[0][1].open if bars else None
            reason = ExitReason(pending.get("reason", ExitReason.MANUAL.value))
            if price is not None or self.broker.market_price(pos.symbol) is not None:
                if self._exit(session, pos, None, reason, price, now, report):
                    return

        init_risk = abs(pos.avg_entry - pos.initial_stop_loss)
        for ts, bar in bars:
            new_session = ts.date() != prev_day
            prev_day = ts.date()
            hit = stop_fill(direction, pos.stop_loss, bar, slip)
            if hit is not None:
                price, gapped = hit
                if gapped and new_session and pos.overnight:
                    reason = ExitReason.GAP
                elif self._last_sl_mover(pos) in ("TRAIL", "BREAKEVEN"):
                    reason = ExitReason.TRAIL
                else:
                    reason = ExitReason.SL
                self._set_note(pos, "last_bar_ts", ts.isoformat())
                self._exit(session, pos, None, reason, price, now, report, apply_slippage=False)
                return
            # partial at R + breakeven
            if not pos.partial_done and init_risk > 0 and cfg.partial_exit_pct > 0:
                level = pos.avg_entry + (1 if long_ else -1) * cfg.partial_exit_at_r * init_risk
                px = target_fill(direction, level, bar)
                if px is not None:
                    part = int(pos.qty * cfg.partial_exit_pct / 100.0)
                    if pos.qty >= 2 and part >= 1:
                        self._exit(session, pos, min(part, pos.qty - 1), ExitReason.TARGET, px, now, report,
                                   apply_slippage=False)
                    pos.partial_done = True
                    if cfg.move_sl_to_breakeven_at_partial and self._move_sl(session, pos, pos.avg_entry,
                                                                             "BREAKEVEN", now, report):
                        pos.breakeven_moved = True
            # final target
            final_t = pos.target_2 if (pos.target_2 and cfg.partial_exit_pct > 0) else pos.target_1
            if final_t:
                px = target_fill(direction, final_t, bar)
                if px is not None:
                    self._set_note(pos, "last_bar_ts", ts.isoformat())
                    self._exit(session, pos, None, ExitReason.TARGET, px, now, report, apply_slippage=False)
                    return
            fav = favorable_excursion(direction, bar)
            pos.highest_price = max(pos.highest_price or bar.high, bar.high)
            pos.lowest_price = min(pos.lowest_price or bar.low, bar.low)
            pos.last_price = bar.close
            pos.last_price_at = ts
            self._set_note(pos, "last_bar_ts", ts.isoformat())
            if init_risk > 0 and abs(fav - pos.avg_entry) >= init_risk and (fav > pos.avg_entry) == long_:
                self._set_note(pos, "reached_1r", True)
        if not bars:
            mp = self.broker.market_price(pos.symbol)
            if mp is not None:
                pos.last_price = mp[0]
                pos.last_price_at = mp[1]

        if pos.status != PositionStatus.OPEN.value:
            return
        self._trail(session, pos, cfg, now, report)
        self._time_exits(session, pos, cfg, now, report)

    # ------------------------------------------------------------------ trailing

    def _trail(self, session: Session, pos: Position, cfg: ModeConfig, now: datetime, report: MonitorReport) -> None:
        rule = TrailingMethod(pos.trailing_rule or TrailingMethod.NONE.value)
        if rule == TrailingMethod.NONE or self.data is None:
            return
        if not (pos.breakeven_moved or self._notes(pos).get("reached_1r")):
            return  # trail only once the trade has worked (>= 1R)
        long_ = pos.direction == Direction.LONG.value
        mult = cfg.trail_atr_mult
        from features.indicators import atr as atr_fn
        from features.indicators import ema as ema_fn

        candidate: float | None = None
        if rule in (TrailingMethod.ATR, TrailingMethod.EMA9_5M):
            df = self.data.candles(pos.symbol, Timeframe.M5, lookback_bars=60)
            if df is None or len(df) < 15:
                return
            if rule == TrailingMethod.ATR:
                a = float(atr_fn(df, 14).iloc[-1])
                extreme = pos.highest_price if long_ else pos.lowest_price
                if extreme is None or not a == a:
                    return
                candidate = extreme - mult * a if long_ else extreme + mult * a
            else:
                candidate = float(ema_fn(df["close"], 9).iloc[-1])
        else:
            df = self.data.candles(pos.symbol, Timeframe.D1, lookback_bars=60)
            if df is None or len(df) < 21:
                return
            if rule == TrailingMethod.EMA20_DAILY:
                candidate = float(ema_fn(df["close"], 20).iloc[-1])
            else:
                a = float(atr_fn(df, 14).iloc[-1])
                close = float(df["close"].iloc[-1])
                candidate = close - mult * a if long_ else close + mult * a
        if candidate is None or candidate != candidate:
            return
        ref = pos.last_price or pos.avg_entry
        if (long_ and candidate >= ref) or (not long_ and candidate <= ref):
            return  # would be at/through the market: leave the stop where it is
        self._move_sl(session, pos, candidate, "TRAIL", now, report)

    # ------------------------------------------------------------------ time-based exits

    def _time_exits(self, session: Session, pos: Position, cfg: ModeConfig, now: datetime, report: MonitorReport) -> None:
        t = now.time()
        mode = Mode(pos.mode)
        sched = self.settings.schedule
        if pos.product == Product.MIS.value:
            if t >= parse_hhmm(sched.intraday_squareoff):
                self._exit(session, pos, None, ExitReason.SQUAREOFF, None, now, report)
                return
            if cfg.time_stop_minutes:
                age_min = (now - to_ist(pos.opened_at)).total_seconds() / 60.0
                init_risk = abs(pos.avg_entry - pos.initial_stop_loss)
                if age_min >= cfg.time_stop_minutes and init_risk > 0:
                    long_ = pos.direction == Direction.LONG.value
                    best = pos.highest_price if long_ else pos.lowest_price
                    best = best if best is not None else pos.avg_entry
                    progress = ((best - pos.avg_entry) if long_ else (pos.avg_entry - best)) / init_risk
                    if progress < cfg.time_stop_min_r:
                        self._exit(session, pos, None, ExitReason.TIME, None, now, report)
                return
            return
        # CNC
        if mode == Mode.BTST:
            if pos.holding_days >= max(1, pos.max_holding_days or 1) and t >= parse_hhmm(sched.cnc_entry_window.end):
                self._exit(session, pos, None, ExitReason.TIME, None, now, report)
            return
        max_days = pos.max_holding_days or cfg.max_holding_days
        if max_days and pos.holding_days >= max_days and t >= parse_hhmm(sched.cnc_time_stop):
            self._exit(session, pos, None, ExitReason.TIME, None, now, report)
            return
        if cfg.exit_before_events and not pos.pin_no_event_exit and self.events_fn is not None \
                and t >= parse_hhmm(sched.cnc_time_stop):
            nxt = self.calendar.next_trading_day(now.date()) if self.calendar is not None else now.date() + timedelta(days=1)
            avoid = {x.upper() for x in self.settings.risk.avoid_event_types}
            try:
                events = self.events_fn(pos.symbol, now.date() + timedelta(days=1), nxt)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[monitor] event lookup failed for {pos.symbol}: {exc}")
                events = []
            hits = [e for e in events if e.event_type.value.upper() in avoid
                    and (e.symbol is None or e.symbol.upper() == pos.symbol.upper())]
            if hits:
                self._set_note(pos, "event_exit", [f"{e.event_type.value} {e.event_date}" for e in hits])
                self._exit(session, pos, None, ExitReason.EVENT, None, now, report)

    # ------------------------------------------------------------------ capital protection

    def _protect(self, session: Session, batch_id: int, now: datetime, market_open: bool, report: MonitorReport) -> None:
        views = open_position_views(session, batch_id)
        realized = realized_by_mode(session, batch_id, now.date())
        snap = self.capital.snapshot(session, batch_id, views, sum(realized.values()), update_hwm=True)
        acct = self.capital.account(session, batch_id)
        batch = session.get(Batch, batch_id)
        if snap.tier == DrawdownTier.HALT and not acct.halted:
            reason = (f"Capital protection HALT: equity Rs {snap.total_equity:.2f}, drawdown {snap.drawdown_pct:.2f}% "
                      f"(floor Rs {snap.effective_floor:.2f})")
            self.capital.set_halted(session, batch_id, reason)
            if batch is not None:
                batch.status = BatchStatus.HALTED.value
                batch.halted_at = now
                batch.halt_reason = reason
            self.broker.close_positions(session, reason=ExitReason.HALT, batch_id=batch_id, market_open=market_open)
            report.halted_batches.append(batch_id)
            log_event("capital", {"event": "HALT", "batch_id": batch_id, "reason": reason}, clock=self.clock)
            if self.notifier is not None:
                try:
                    self.notifier.urgent("HALT - capital protection triggered", reason,
                                         {"batch_id": batch_id, "equity": snap.total_equity, "hwm": snap.hwm})
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[monitor] urgent HALT email failed: {exc}")
        elif acct.halted and market_open:
            # halted with positions still open (queued exits): make sure they go
            for p in session.execute(select(Position).where(Position.batch_id == batch_id,
                                                            Position.status == PositionStatus.OPEN.value)).scalars():
                if not self._notes(p).get("pending_exit"):
                    self.broker.request_exit(session, p, ExitReason.HALT)
        self._profit_lock_tracking(session, batch_id, now, snap, realized, views)

    def _profit_lock_tracking(self, session: Session, batch_id: int, now: datetime, snap: Any,
                              realized: dict[Mode, float], views: list[Any]) -> None:
        """Intraday: if open+realized day profit falls back 50% from its peak (after the lock), stop."""
        if Mode.INTRADAY not in realized and not any(v.mode == Mode.INTRADAY for v in views):
            return
        row = get_mode_state(session, batch_id, Mode.INTRADAY, now)
        today = now.date()
        if row.day_peak_date != today:
            row.day_peak_date = today
            row.day_peak_profit = 0.0
        day_pnl = realized.get(Mode.INTRADAY, 0.0) + sum(
            v.unrealized_gross - v.entry_charges for v in views if v.mode == Mode.INTRADAY)
        row.day_peak_profit = max(row.day_peak_profit or 0.0, day_pnl)
        lock_level = self.settings.risk.intraday_daily_profit_lock_pct / 100.0 * max(0.0, snap.trading_balance)
        giveback = self.settings.risk.intraday_profit_giveback_pct / 100.0
        if (row.stopped_for_day != today and lock_level > 0 and row.day_peak_profit >= lock_level
                and day_pnl <= row.day_peak_profit * (1 - giveback)):
            row.stopped_for_day = today
            log_event("risk", {"event": "intraday_stopped_for_day", "batch_id": batch_id, "peak": row.day_peak_profit,
                               "now": day_pnl}, clock=self.clock)
        row.updated_at = now


def first_bar_of_day(ts: datetime) -> bool:
    """True if ``ts`` is the 09:15 bar."""
    return to_ist(ts).time() == time(9, 15)


__all__ = ["MonitorReport", "PositionMonitor", "first_bar_of_day", "ist_datetime"]
