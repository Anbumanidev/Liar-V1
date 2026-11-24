"""Paper broker: orders, fills, positions, exits and owner edits (section 9).

PAPER ONLY. Nothing here talks to a broker; fills are simulated from market data.

Money flow (see capital.accounts): an entry fill BLOCKs the position margin from TRADING
cash and accrues the entry charges on the position; every exit leg RELEASEs the proportional
margin and books REALIZED_PNL = gross - (accrued entry charges of the leg + exit charges).

The broker never decides whether a trade is allowed: callers pass an APPROVED ``RiskVerdict``
(the deterministic risk manager has the final veto). Hard-coded guards are still enforced
here: long-only CNC, SL tighten-only, no entry through the stop.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from capital.errors import CapitalError
from config.schema import HardCaps, Settings
from core.clock import IST, Clock, ist_datetime, parse_hhmm, to_ist
from core.enums import (
    Direction,
    EntryType,
    ExitReason,
    Mode,
    OrderPurpose,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionStatus,
    Product,
    Timeframe,
    TradeSource,
    TrailingMethod,
)
from core.errors import RealTradingForbiddenError, StopLossLooseningError
from core.logging import log_event, to_jsonable
from core.types import RiskVerdict, TradeProposal
from modes.base import assert_direction_allowed, get_mode_spec, is_overnight
from paper_engine.charges import entry_side, exit_side, order_charges
from paper_engine.fills import Bar, limit_fill_price, round_tick, slipped_price
from risk.guards import InvalidStopLossError, validate_sl_change
from storage.audit import audit, record_error
from storage.models import ModeState, Order, Position, Trade

#: Trailing rules allowed per product.
ALLOWED_TRAILING: dict[Product, tuple[TrailingMethod, ...]] = {
    Product.MIS: (TrailingMethod.NONE, TrailingMethod.ATR, TrailingMethod.EMA9_5M),
    Product.CNC: (TrailingMethod.NONE, TrailingMethod.EMA20_DAILY, TrailingMethod.ATR_DAILY_CLOSE),
}


class OrderRejected(Exception):
    """An order could not be filled/opened (reason in the message)."""


class PaperBroker:
    """Simulated broker over the trading database."""

    def __init__(
        self,
        settings: Settings,
        caps: HardCaps,
        clock: Clock,
        capital: Any,
        data: Any | None = None,
        calendar: Any | None = None,
        notifier: Any | None = None,
    ) -> None:
        """
        Args:
            capital: ``capital.accounts.CapitalManager``.
            data: ``data.service.MarketDataService`` (fills / prices).
            calendar: ``data.calendar.NSECalendar`` (order windows, holding days).
            notifier: ``notifications.service.Notifier`` (entry/exit emails), optional.
        """
        if caps.paper_only is not True:  # defence in depth (core.guard asserts at startup)
            raise RealTradingForbiddenError("PaperBroker requires paper_only=true")
        self.settings = settings
        self.caps = caps
        self.clock = clock
        self.capital = capital
        self.data = data
        self.calendar = calendar
        self.notifier = notifier

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    def _now(self, now: datetime | None = None) -> datetime:
        return to_ist(now) if now is not None else to_ist(self.clock.now())

    # ================================================================== helpers

    def leverage_for(self, mode: Mode) -> float:
        if Mode(mode) != Mode.INTRADAY:
            return 1.0
        return max(1.0, min(self.settings.risk.mis_leverage, self.caps.max_mis_leverage))

    def _next_session_day(self, d: date) -> date:
        if self.calendar is None:
            nd = d + timedelta(days=1)
            while nd.weekday() >= 5:
                nd += timedelta(days=1)
            return nd
        return self.calendar.next_trading_day(d)

    def _is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 if self.calendar is None else self.calendar.is_trading_day(d)

    def default_order_window(self, mode: Mode, now: datetime, entry_type: EntryType,
                             time_validity_minutes: int | None = None) -> tuple[datetime | None, datetime]:
        """``(not_before, valid_until)`` for a new entry order.

        * INTRADAY: now .. min(now + validity (default 5 min MARKET / 30 min LIMIT), no-entry-after).
        * BTST: now .. today's BTST entry-window end.
        * SWING/POSITIONAL: inside today's CNC entry window if open now, else the next
          session's window (evening-scan orders execute 09:20-10:30 next session).
        """
        mode = Mode(mode)
        now = to_ist(now)
        mcfg = self.settings.mode(mode)
        if mode == Mode.INTRADAY:
            default = 5 if entry_type == EntryType.MARKET else 30
            end = now + timedelta(minutes=time_validity_minutes or default)
            cutoff = ist_datetime(now.date(), self.settings.risk.intraday_no_entry_after)
            return None, min(end, max(cutoff, now + timedelta(minutes=1)))
        windows = mcfg.entry_windows or []
        if mode == Mode.BTST:
            w_end = parse_hhmm(windows[-1].end) if windows else time(15, 20)
            return None, ist_datetime(now.date(), w_end)
        w_start = parse_hhmm(windows[0].start) if windows else parse_hhmm(self.settings.schedule.cnc_entry_window.start)
        w_end = parse_hhmm(windows[-1].end) if windows else parse_hhmm(self.settings.schedule.cnc_entry_window.end)
        if self._is_trading_day(now.date()) and now.time() < w_end:
            start = ist_datetime(now.date(), w_start)
            return (start if now < start else None), ist_datetime(now.date(), w_end)
        nd = self._next_session_day(now.date())
        return ist_datetime(nd, w_start), ist_datetime(nd, w_end)

    def _bars_since(self, symbol: str, since: datetime) -> list[tuple[datetime, Bar]]:
        """Closed 1-minute bars (fallback 5-minute) with open time >= ``since``."""
        if self.data is None:
            return []
        for tf in (Timeframe.M1, Timeframe.M5):
            try:
                df = self.data.candles(symbol, tf, start=since)
            except Exception as exc:  # noqa: BLE001 - data failure: degrade to quotes
                logger.warning(f"[broker] candles {symbol} {tf.value} failed: {exc}")
                continue
            if df is not None and not df.empty:
                df = df[df.index >= since]
                return [
                    (ts.to_pydatetime(), Bar(float(r.open), float(r.high), float(r.low), float(r.close)))
                    for ts, r in df.iterrows()
                ]
        return []

    def market_price(self, symbol: str) -> tuple[float, datetime] | None:
        """Latest price for ``symbol`` (quote ltp) or None."""
        if self.data is None:
            return None
        try:
            q = self.data.quote(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[broker] quote {symbol} failed: {exc}")
            return None
        if q is None or not q.ltp or q.ltp <= 0:
            return None
        return float(q.ltp), to_ist(q.timestamp)

    # ================================================================== entries

    def submit_entry(
        self,
        session: Session,
        proposal: TradeProposal,
        verdict: RiskVerdict,
        *,
        decision_id: int | None = None,
        approval_id: int | None = None,
        not_before: datetime | None = None,
        valid_until: datetime | None = None,
        time_validity_minutes: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Order:
        """Create a PENDING paper entry order for an APPROVED risk verdict."""
        if not verdict.approved or verdict.qty < 1:
            raise OrderRejected(f"risk verdict not approved: {verdict.message}")
        assert_direction_allowed(proposal.mode, proposal.direction)
        now = self._now()
        nb, vu = self.default_order_window(proposal.mode, now, proposal.entry_type, time_validity_minutes)
        not_before = not_before if not_before is not None else nb
        valid_until = valid_until if valid_until is not None else vu
        mode = Mode(proposal.mode)
        product = get_mode_spec(mode).product
        side = entry_side(proposal.direction)
        meta = {
            "proposal": proposal.model_dump(mode="json"),
            "verdict": {"qty": verdict.qty, "max_qty": verdict.max_qty, "risk_amount": verdict.risk_amount,
                        "effective_risk": verdict.effective_risk, "tier": verdict.tier.value},
            "leverage": self.leverage_for(mode),
            "not_before": not_before.isoformat() if not_before else None,
            "mode_settings": self.settings.mode(mode).model_dump(mode="json"),
            **(extra or {}),
        }
        order = Order(
            batch_id=proposal.batch_id,
            approval_id=approval_id if approval_id is not None else proposal.approval_id,
            decision_id=decision_id if decision_id is not None else proposal.decision_id,
            symbol=proposal.symbol,
            mode=mode.value,
            product=product.value,
            side=side.value,
            order_type=OrderType(proposal.entry_type).value,
            purpose=OrderPurpose.ENTRY.value,
            qty=int(verdict.qty),
            limit_price=round_tick(proposal.entry_price) if proposal.entry_type == EntryType.LIMIT else None,
            status=OrderStatus.PENDING.value,
            created_at=now,
            valid_until=valid_until,
            source=proposal.source.value,
            is_paper=True,
            meta_json=to_jsonable(meta),
        )
        session.add(order)
        session.flush()
        log_event("orders", {"event": "submitted", "order_id": order.id, "symbol": order.symbol, "mode": order.mode,
                             "side": order.side, "type": order.order_type, "qty": order.qty,
                             "limit": order.limit_price, "valid_until": valid_until, "not_before": not_before,
                             "source": order.source}, clock=self.clock)
        return order

    def pending_orders(self, session: Session, batch_id: int | None = None) -> list[Order]:
        stmt = select(Order).where(Order.status == OrderStatus.PENDING.value, Order.purpose == OrderPurpose.ENTRY.value)
        if batch_id is not None:
            stmt = stmt.where(Order.batch_id == batch_id)
        return list(session.execute(stmt.order_by(Order.id)).scalars())

    def cancel_order(self, session: Session, order: Order, reason: str, *, status: OrderStatus = OrderStatus.CANCELLED) -> None:
        order.status = status.value
        order.reason = reason
        session.flush()
        log_event("orders", {"event": status.value.lower(), "order_id": order.id, "symbol": order.symbol,
                             "reason": reason}, clock=self.clock)

    def process_pending_orders(self, session: Session, now: datetime | None = None) -> list[Position]:
        """Try to fill every PENDING entry order. Returns newly opened positions."""
        now = self._now(now)
        opened: list[Position] = []
        for order in self.pending_orders(session):
            try:
                pos = self._try_fill(session, order, now)
            except Exception as exc:  # noqa: BLE001 - one bad order never stops the loop
                record_error(session, "paper_engine.broker", exc, {"order_id": order.id}, batch_id=order.batch_id,
                             clock=self.clock)
                self.cancel_order(session, order, f"fill error: {exc}", status=OrderStatus.REJECTED)
                continue
            if pos is not None:
                opened.append(pos)
        return opened

    def _try_fill(self, session: Session, order: Order, now: datetime) -> Position | None:
        nb_raw = (order.meta_json or {}).get("not_before")
        not_before = to_ist(datetime.fromisoformat(nb_raw)) if nb_raw else None
        if not_before is not None and now < not_before:
            return None
        start = max(to_ist(order.created_at), not_before) if not_before else to_ist(order.created_at)
        bars = [(ts, b) for ts, b in self._bars_since(order.symbol, start)
                if order.valid_until is None or ts <= to_ist(order.valid_until)]
        side = OrderSide(order.side)
        fill: tuple[float, datetime] | None = None
        for ts, bar in bars:
            if order.order_type == OrderType.MARKET.value:
                fill = (slipped_price(side, bar.open, self.settings.engine.slippage_pct), ts)
                break
            price = limit_fill_price(side, float(order.limit_price), bar)
            if price is not None:
                fill = (price, ts)
                break
        if fill is None and not bars and order.order_type == OrderType.MARKET.value:
            mp = self.market_price(order.symbol)
            if mp is not None and mp[1] >= start:
                fill = (slipped_price(side, mp[0], self.settings.engine.slippage_pct), mp[1])
        if fill is None:
            if order.valid_until is not None and now >= to_ist(order.valid_until):
                self.cancel_order(session, order, "not filled before validity ended", status=OrderStatus.EXPIRED)
            return None
        return self._open_position(session, order, fill[0], fill[1])

    def _open_position(self, session: Session, order: Order, price: float, fill_time: datetime) -> Position | None:
        meta = order.meta_json or {}
        prop = TradeProposal.model_validate(meta["proposal"])
        direction = Direction(prop.direction)
        mode = Mode(order.mode)
        assert_direction_allowed(mode, direction)
        sl = prop.stop_loss
        if (direction == Direction.LONG and price <= sl) or (direction == Direction.SHORT and price >= sl):
            self.cancel_order(session, order, f"fill {price} is through the stop-loss {sl}; entry skipped",
                              status=OrderStatus.REJECTED)
            return None
        leverage = float(meta.get("leverage", 1.0))
        product = Product(order.product)
        margin = round(order.qty * price / leverage, 4)
        entry_ch = order_charges(product, entry_side(direction), order.qty, price, self.settings.charges)
        mode_settings = meta.get("mode_settings") or self.settings.mode(mode).model_dump(mode="json")
        pos = Position(
            batch_id=order.batch_id, symbol=order.symbol, mode=mode.value, product=product.value,
            direction=direction.value, status=PositionStatus.OPEN.value, qty=order.qty, initial_qty=order.qty,
            avg_entry=price, stop_loss=sl, initial_stop_loss=sl, target_1=prop.target_1, target_2=prop.target_2,
            trailing_rule=TrailingMethod(prop.trailing_rule).value,
            max_holding_days=prop.max_holding_days if prop.max_holding_days is not None
            else (mode_settings.get("max_holding_days") or None),
            holding_days=0, opened_at=fill_time, strategy=prop.strategy, regime=prop.regime.value,
            grade=prop.grade.value if prop.grade else None, source=prop.source.value, sector=prop.sector,
            margin_blocked=margin, leverage=leverage, entry_charges=entry_ch.total, highest_price=price,
            lowest_price=price, last_price=price, last_price_at=fill_time, overnight=is_overnight(mode),
            decision_id=order.decision_id, approval_id=order.approval_id,
            settings_json=to_jsonable({"mode": mode_settings,
                                       "risk": {"gap_buffer_pct": self.settings.risk.gap_buffer_pct}}),
            sl_history_json=[{"ts": fill_time.isoformat(), "old": None, "new": sl, "by": "ENTRY"}],
            notes_json={},
        )
        session.add(pos)
        session.flush()
        try:
            self.capital.block(session, order.batch_id, margin, position_id=pos.id, mode=mode.value,
                               note=f"entry {order.symbol} {order.qty} @ {price}")
        except CapitalError as exc:
            session.delete(pos)
            session.flush()
            self.cancel_order(session, order, f"capital: {exc}", status=OrderStatus.REJECTED)
            return None
        order.status = OrderStatus.FILLED.value
        order.filled_at = fill_time
        order.fill_price = price
        order.position_id = pos.id
        ref = float(order.limit_price) if order.limit_price else prop.entry_price
        order.slippage = round(abs(price - ref), 4) if order.order_type == OrderType.MARKET.value else 0.0
        order.charges_json = entry_ch.as_dict()
        session.flush()
        log_event("orders", {"event": "filled", "order_id": order.id, "position_id": pos.id, "symbol": pos.symbol,
                             "mode": pos.mode, "direction": pos.direction, "qty": pos.qty, "price": price,
                             "margin": margin, "entry_charges": entry_ch.total}, clock=self.clock)
        self._notify("entry", pos, order)
        return pos

    # ================================================================== exits

    def _dp_already_charged(self, session: Session, pos: Position, day: date) -> bool:
        if pos.product != Product.CNC.value:
            return False
        start = datetime.combine(day, time(0, 0), tzinfo=IST)
        rows = session.execute(
            select(Order.id).where(
                Order.batch_id == pos.batch_id, Order.symbol == pos.symbol, Order.product == Product.CNC.value,
                Order.side == OrderSide.SELL.value, Order.status == OrderStatus.FILLED.value,
                Order.filled_at >= start, Order.filled_at < start + timedelta(days=1),
            ).limit(1)
        ).first()
        return rows is not None

    def _holding_days(self, opened: datetime, now: datetime) -> int:
        d0, d1 = to_ist(opened).date(), to_ist(now).date()
        if self.calendar is None:
            return max(0, (d1 - d0).days)
        return max(0, self.calendar.count_trading_days(d0, d1))

    def exit_position(
        self,
        session: Session,
        position: Position,
        qty: int | None = None,
        reason: ExitReason = ExitReason.MANUAL,
        *,
        price: float | None = None,
        now: datetime | None = None,
        actor: str = "SYSTEM",
        apply_slippage: bool = True,
    ) -> Trade:
        """Close ``qty`` (default all) of ``position`` at ``price`` (default market + slippage).

        Raises:
            OrderRejected: position not open, bad qty, or no price available.
        """
        if position.status != PositionStatus.OPEN.value or position.qty <= 0:
            raise OrderRejected(f"position {position.id} is not open")
        qty = position.qty if qty is None else int(qty)
        if qty < 1 or qty > position.qty:
            raise OrderRejected(f"invalid exit qty {qty} (open {position.qty})")
        now = self._now(now)
        direction = Direction(position.direction)
        side = exit_side(direction)
        if price is None:
            mp = self.market_price(position.symbol)
            if mp is None:
                if position.last_price is None:
                    raise OrderRejected(f"no price available to exit {position.symbol}")
                ref = float(position.last_price)
            else:
                ref = mp[0]
            price = slipped_price(side, ref, self.settings.engine.slippage_pct) if apply_slippage else round_tick(ref)
        product = Product(position.product)
        include_dp = None if not self._dp_already_charged(session, position, now.date()) else False
        exit_ch = order_charges(product, side, qty, price, self.settings.charges, include_dp=include_dp)
        full = qty == position.qty
        frac = qty / position.qty
        entry_share = position.entry_charges if full else round(position.entry_charges * frac, 6)
        margin_share = position.margin_blocked if full else round(position.margin_blocked * frac, 6)
        sign = 1.0 if direction == Direction.LONG else -1.0
        gross = sign * (price - position.avg_entry) * qty
        charges = entry_share + exit_ch.total
        net = gross - charges
        init_risk = abs(position.avg_entry - position.initial_stop_loss)
        r_mult = (sign * (price - position.avg_entry) / init_risk) if init_risk > 0 else None

        order = Order(
            batch_id=position.batch_id, position_id=position.id, symbol=position.symbol, mode=position.mode,
            product=position.product, side=side.value, order_type=OrderType.MARKET.value,
            purpose=(OrderPurpose.EXIT if full else OrderPurpose.PARTIAL_EXIT).value, qty=qty,
            status=OrderStatus.FILLED.value, created_at=now, filled_at=now, fill_price=price,
            charges_json=exit_ch.as_dict(), source=TradeSource.MANUAL.value if actor != "SYSTEM" else TradeSource.SYSTEM.value,
            reason=ExitReason(reason).value, is_paper=True,
        )
        session.add(order)
        session.flush()
        trade = Trade(
            batch_id=position.batch_id, position_id=position.id, symbol=position.symbol, mode=position.mode,
            product=position.product, direction=position.direction, qty=qty, entry_time=position.opened_at,
            entry_price=position.avg_entry, exit_time=now, exit_price=price, stop_loss=position.initial_stop_loss,
            target_1=position.target_1, target_2=position.target_2, exit_reason=ExitReason(reason).value,
            is_partial=not full, gross_pnl=round(gross, 4), charges=round(charges, 4), net_pnl=round(net, 4),
            r_multiple=round(r_mult, 4) if r_mult is not None else None,
            charges_json={"entry_share": round(entry_share, 4), **exit_ch.as_dict()},
            source=position.source, strategy=position.strategy, regime=position.regime, grade=position.grade,
            holding_minutes=int((now - to_ist(position.opened_at)).total_seconds() // 60),
            holding_days=self._holding_days(position.opened_at, now),
            decision_id=position.decision_id, approval_id=position.approval_id,
        )
        session.add(trade)
        session.flush()
        self.capital.release(session, position.batch_id, margin_share, net, position_id=position.id,
                             trade_id=trade.id, mode=position.mode,
                             note=f"{ExitReason(reason).value} {position.symbol} {qty} @ {price}")
        position.qty -= qty
        position.margin_blocked = 0.0 if full else max(0.0, position.margin_blocked - margin_share)
        position.entry_charges = 0.0 if full else max(0.0, position.entry_charges - entry_share)
        position.realized_gross += gross
        position.realized_charges += charges
        position.last_price = price
        position.last_price_at = now
        if full:
            position.status = PositionStatus.CLOSED.value
            position.closed_at = now
        session.flush()
        if full:
            self._update_mode_state_on_close(session, position, now)
        log_event("trades", {"trade_id": trade.id, "position_id": position.id, "symbol": position.symbol,
                             "mode": position.mode, "qty": qty, "exit_price": price, "reason": trade.exit_reason,
                             "gross": trade.gross_pnl, "charges": trade.charges, "net": trade.net_pnl,
                             "partial": not full, "actor": actor}, clock=self.clock)
        if actor != "SYSTEM":
            audit(session, actor, "POSITION_EXIT", f"position:{position.id}",
                  {"qty": qty, "price": price, "reason": trade.exit_reason}, batch_id=position.batch_id, clock=self.clock)
        self._notify("exit", trade)
        return trade

    def _update_mode_state_on_close(self, session: Session, pos: Position, now: datetime) -> None:
        """Consecutive-loss tracking and the automatic pause (section 7.5)."""
        mode = Mode(pos.mode)
        mcfg = self.settings.mode(mode)
        row = get_mode_state(session, pos.batch_id, mode, now)
        total_net = pos.realized_gross - pos.realized_charges
        if total_net < 0:
            row.consecutive_losses = (row.consecutive_losses or 0) + 1
        else:
            row.consecutive_losses = 0
        if row.consecutive_losses >= mcfg.consecutive_loss_limit:
            if mode == Mode.INTRADAY or mcfg.consecutive_loss_pause_days == 0:
                until = now + timedelta(minutes=max(mcfg.consecutive_loss_pause_minutes, 1))
            else:
                end_day = now.date()
                for _ in range(mcfg.consecutive_loss_pause_days):
                    end_day = self._next_session_day(end_day)
                until = ist_datetime(end_day, "00:00")
            row.paused_until = until
            row.pause_reason = f"{row.consecutive_losses} consecutive losses"
            row.consecutive_losses = 0
            log_event("risk", {"event": "mode_auto_pause", "mode": mode.value, "until": until,
                               "batch_id": pos.batch_id}, clock=self.clock)
        row.updated_at = now
        session.flush()

    def request_exit(self, session: Session, position: Position, reason: ExitReason, *, actor: str = "SYSTEM") -> None:
        """Mark a position to be exited at the next available price (market closed now)."""
        notes = dict(position.notes_json or {})
        notes["pending_exit"] = {"reason": ExitReason(reason).value, "actor": actor, "requested_at": self._now().isoformat()}
        position.notes_json = notes
        session.flush()
        log_event("orders", {"event": "exit_requested", "position_id": position.id, "symbol": position.symbol,
                             "reason": ExitReason(reason).value}, clock=self.clock)

    def close_positions(
        self,
        session: Session,
        *,
        reason: ExitReason,
        batch_id: int | None = None,
        include_mis: bool = True,
        include_cnc: bool = True,
        actor: str = "SYSTEM",
        market_open: bool = True,
    ) -> list[Trade]:
        """Exit every matching open position now (or queue it if the market is closed)."""
        stmt = select(Position).where(Position.status == PositionStatus.OPEN.value)
        if batch_id is not None:
            stmt = stmt.where(Position.batch_id == batch_id)
        trades: list[Trade] = []
        for pos in session.execute(stmt).scalars().all():
            is_mis = pos.product == Product.MIS.value
            if (is_mis and not include_mis) or (not is_mis and not include_cnc):
                continue
            if not market_open:
                self.request_exit(session, pos, reason, actor=actor)
                continue
            try:
                trades.append(self.exit_position(session, pos, None, reason, actor=actor))
            except OrderRejected as exc:
                logger.error(f"[broker] could not exit {pos.symbol}: {exc}; queued for next price")
                self.request_exit(session, pos, reason, actor=actor)
        return trades

    # ================================================================== owner edits

    def tighten_sl(self, session: Session, position: Position, new_sl: float, *, actor: str,
                   by: str = "OWNER", ltp: float | None = None) -> None:
        """Move the stop closer to price only (rule 0.10). Raises on loosening."""
        if position.status != PositionStatus.OPEN.value:
            raise OrderRejected("position is not open")
        if ltp is None:
            mp = self.market_price(position.symbol)
            ltp = mp[0] if mp else position.last_price
        validate_sl_change(Direction(position.direction), position.stop_loss, float(new_sl), ltp)
        old = position.stop_loss
        position.stop_loss = round_tick(float(new_sl))
        hist = list(position.sl_history_json or [])
        hist.append({"ts": self._now().isoformat(), "old": old, "new": position.stop_loss, "by": by})
        position.sl_history_json = hist
        session.flush()
        if by == "OWNER" or actor != "SYSTEM":
            audit(session, actor, "SL_TIGHTEN", f"position:{position.id}", {"old": old, "new": position.stop_loss},
                  batch_id=position.batch_id, clock=self.clock)
        log_event("orders", {"event": "sl_moved", "position_id": position.id, "old": old,
                             "new": position.stop_loss, "by": by}, clock=self.clock)

    def update_targets(self, session: Session, position: Position, target_1: float | None,
                       target_2: float | None, *, actor: str) -> None:
        """Change targets (must stay on the profit side of the entry; T2 beyond T1)."""
        long_ = position.direction == Direction.LONG.value
        t1 = position.target_1 if target_1 is None else float(target_1)
        t2 = position.target_2 if target_2 is None else float(target_2)
        for t in (t1, t2):
            if t is not None and ((long_ and t <= position.avg_entry) or (not long_ and t >= position.avg_entry)):
                raise OrderRejected(f"target {t} must be beyond the entry {position.avg_entry}")
        if t1 is not None and t2 is not None and ((long_ and t2 < t1) or (not long_ and t2 > t1)):
            raise OrderRejected("target_2 must be beyond target_1")
        old = {"t1": position.target_1, "t2": position.target_2}
        position.target_1, position.target_2 = t1, t2
        session.flush()
        audit(session, actor, "TARGETS_UPDATE", f"position:{position.id}", {"old": old, "new": {"t1": t1, "t2": t2}},
              batch_id=position.batch_id, clock=self.clock)

    def update_trailing(self, session: Session, position: Position, rule: TrailingMethod,
                        trail_atr_mult: float | None, *, actor: str) -> None:
        rule = TrailingMethod(rule)
        allowed = ALLOWED_TRAILING[Product(position.product)]
        if rule not in allowed:
            raise OrderRejected(f"trailing rule {rule.value} not allowed for {position.product} "
                                f"({', '.join(a.value for a in allowed)})")
        old = position.trailing_rule
        position.trailing_rule = rule.value
        if trail_atr_mult is not None:
            st = dict(position.settings_json or {})
            mode_cfg = dict(st.get("mode") or {})
            mode_cfg["trail_atr_mult"] = float(trail_atr_mult)
            st["mode"] = mode_cfg
            position.settings_json = st
        session.flush()
        audit(session, actor, "TRAILING_UPDATE", f"position:{position.id}",
              {"old": old, "new": rule.value, "mult": trail_atr_mult}, batch_id=position.batch_id, clock=self.clock)

    def update_holding(self, session: Session, position: Position, max_holding_days: int, *, actor: str) -> None:
        if position.product != Product.CNC.value:
            raise OrderRejected("holding period applies to CNC positions only")
        if max_holding_days < 1:
            raise OrderRejected("max holding days must be >= 1")
        old = position.max_holding_days
        position.max_holding_days = int(max_holding_days)
        session.flush()
        audit(session, actor, "HOLDING_UPDATE", f"position:{position.id}", {"old": old, "new": max_holding_days},
              batch_id=position.batch_id, clock=self.clock)

    def set_event_pin(self, session: Session, position: Position, pinned: bool, *, actor: str) -> None:
        """'Do not auto-exit before event' (still subject to SL and HALT)."""
        position.pin_no_event_exit = bool(pinned)
        session.flush()
        audit(session, actor, "EVENT_PIN", f"position:{position.id}", {"pinned": bool(pinned)},
              batch_id=position.batch_id, clock=self.clock)

    def convert_position(self, session: Session, position: Position, to_mode: Mode, verdict: RiskVerdict,
                         *, actor: str) -> Position:
        """INTRADAY -> SWING/POSITIONAL conversion with an approved conversion verdict."""
        if not verdict.approved:
            raise OrderRejected(f"conversion vetoed: {verdict.message}")
        to_mode = Mode(to_mode)
        if position.mode != Mode.INTRADAY.value or to_mode not in (Mode.SWING, Mode.POSITIONAL) \
                or position.direction != Direction.LONG.value:
            raise OrderRejected("only LONG INTRADAY -> SWING/POSITIONAL conversions are allowed")
        required = position.qty * position.avg_entry
        extra = round(required - position.margin_blocked, 4)
        if extra > 0:
            self.capital.block(session, position.batch_id, extra, position_id=position.id, mode=to_mode.value,
                               note=f"conversion margin {position.symbol} -> {to_mode.value}")
        old_mode = position.mode
        cfg = self.settings.mode(to_mode)
        position.converted_from_mode = old_mode
        position.mode = to_mode.value
        position.product = Product.CNC.value
        position.leverage = 1.0
        position.margin_blocked = round(position.margin_blocked + max(0.0, extra), 4)
        position.overnight = True
        position.max_holding_days = cfg.max_holding_days
        position.trailing_rule = cfg.trailing.value if cfg.trailing in ALLOWED_TRAILING[Product.CNC] else TrailingMethod.NONE.value
        # charges model switches to CNC: re-accrue the buy-side charges as delivery
        position.entry_charges = order_charges(Product.CNC, OrderSide.BUY, position.qty, position.avg_entry,
                                               self.settings.charges).total
        st = dict(position.settings_json or {})
        st["mode"] = cfg.model_dump(mode="json")
        position.settings_json = to_jsonable(st)
        session.flush()
        audit(session, actor, "POSITION_CONVERT", f"position:{position.id}",
              {"from": old_mode, "to": to_mode.value, "extra_margin": extra}, batch_id=position.batch_id, clock=self.clock)
        log_event("orders", {"event": "converted", "position_id": position.id, "from": old_mode, "to": to_mode.value},
                  clock=self.clock)
        return position

    # ================================================================== notify

    def _notify(self, kind: str, *args: Any) -> None:
        if self.notifier is None:
            return
        try:
            getattr(self.notifier, kind)(*args)
        except Exception as exc:  # noqa: BLE001 - email must never block trading
            logger.error(f"[broker] notifier.{kind} failed: {exc}")


def get_mode_state(session: Session, batch_id: int, mode: Mode, now: datetime) -> ModeState:
    """Get-or-create the ``mode_states`` row for ``batch_id``/``mode``."""
    row = session.execute(
        select(ModeState).where(ModeState.batch_id == batch_id, ModeState.mode == Mode(mode).value)
    ).scalar_one_or_none()
    if row is None:
        row = ModeState(batch_id=batch_id, mode=Mode(mode).value, paused=False, consecutive_losses=0,
                        day_peak_profit=0.0, updated_at=now)
        session.add(row)
        session.flush()
    return row


__all__ = ["ALLOWED_TRAILING", "OrderRejected", "PaperBroker", "get_mode_state",
           "InvalidStopLossError", "StopLossLooseningError"]
