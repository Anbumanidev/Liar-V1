"""Shadow tracking (section 7.6).

Every setup that was NOT traded - rejected by the meta agent, the risk manager, the owner,
an approval timeout, a SIGNALS_ONLY policy or the confluence filter - is followed to its
would-be outcome WITHOUT capital, so the dashboard can compare "my decisions vs the system".

Rules (deterministic, conservative, recomputed from scratch on every update so it is
replay-safe):

* Entry: triggered when price trades at/through the entry within the entry validity window
  (immediately if the setup was a MARKET entry). Never triggered -> ``NOT_TRIGGERED``.
* After entry: stop first if a bar touches both stop and target. ``target_1`` hit -> stop moves
  to the entry; if ``target_2`` exists keep tracking for it (``TARGET_2``), else ``TARGET_1``.
* Expiry (mode horizon: INTRADAY square-off, BTST next morning, SWING/POSITIONAL max holding
  days at 15:20) -> ``TIME`` at the last close.
* Intraday/BTST shadows are evaluated on 1-minute bars (5-minute fallback); SWING/POSITIONAL on
  daily bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.schema import Settings
from core.clock import Clock, ist_datetime, to_ist
from core.enums import Direction, EntryType, Mode, RejectionSource, ShadowOutcome, Timeframe
from core.logging import log_event
from modes.base import product_for
from paper_engine.charges import round_trip_charges
from storage.models import ShadowTrade


@dataclass
class ShadowResult:
    outcome: ShadowOutcome
    entered_at: datetime | None = None
    exit_price: float | None = None
    exit_time: datetime | None = None


class ShadowTracker:
    """Creates and updates ``shadow_trades`` rows."""

    def __init__(self, settings: Settings, clock: Clock, calendar: Any, data: Any | None) -> None:
        self.settings = settings
        self.clock = clock
        self.calendar = calendar
        self.data = data

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------ horizon

    def _next_day(self, d: Any) -> Any:
        return self.calendar.next_trading_day(d) if self.calendar is not None else d + timedelta(days=1)

    def expiry_for(self, mode: Mode, created: datetime, max_holding_days: int | None = None) -> datetime:
        """When a shadow of ``mode`` created at ``created`` stops being followed."""
        mode = Mode(mode)
        created = to_ist(created)
        sched = self.settings.schedule
        if mode == Mode.INTRADAY:
            end = ist_datetime(created.date(), sched.intraday_squareoff)
            return end if created < end else ist_datetime(self._next_day(created.date()), sched.intraday_squareoff)
        if mode == Mode.BTST:
            return ist_datetime(self._next_day(created.date()), sched.cnc_entry_window.end)
        days = max_holding_days or self.settings.mode(mode).max_holding_days or 10
        d = created.date()
        for _ in range(max(1, days)):
            d = self._next_day(d)
        return ist_datetime(d, sched.cnc_time_stop)

    # ------------------------------------------------------------------ create

    def create(
        self,
        session: Session,
        *,
        batch_id: int | None,
        symbol: str,
        mode: Mode,
        direction: Direction,
        entry: float | None,
        stop_loss: float | None,
        target_1: float | None,
        target_2: float | None = None,
        qty: int = 0,
        source: RejectionSource,
        rule: str | None = None,
        reason: str | None = None,
        decision_id: int | None = None,
        approval_id: int | None = None,
        strategy: str | None = None,
        grade: str | None = None,
        confidence: int | None = None,
        entry_type: EntryType = EntryType.LIMIT,
        max_holding_days: int | None = None,
    ) -> ShadowTrade | None:
        """Record a rejected setup. Returns None if its prices are not a valid setup."""
        if entry is None or stop_loss is None or target_1 is None or entry <= 0:
            return None
        long_ = Direction(direction) == Direction.LONG
        if (long_ and not (stop_loss < entry < target_1)) or (not long_ and not (target_1 < entry < stop_loss)):
            logger.debug(f"[shadow] skip invalid setup {symbol} e={entry} sl={stop_loss} t1={target_1}")
            return None
        now = to_ist(self.clock.now())
        row = ShadowTrade(
            batch_id=batch_id, decision_id=decision_id, approval_id=approval_id, symbol=symbol.upper(),
            mode=Mode(mode).value, direction=Direction(direction).value, entry_price=float(entry),
            stop_loss=float(stop_loss), target_1=float(target_1), target_2=float(target_2) if target_2 else None,
            qty=max(0, int(qty or 0)), rejection_source=RejectionSource(source).value, rejection_rule=rule,
            rejection_reason=(reason or "")[:2000] or None, strategy=strategy, grade=grade, confidence=confidence,
            created_at=now, expires_at=self.expiry_for(Mode(mode), now, max_holding_days),
            entered_at=now if EntryType(entry_type) == EntryType.MARKET else None,
            status="OPEN", outcome=ShadowOutcome.OPEN.value,
        )
        session.add(row)
        session.flush()
        log_event("shadow", {"event": "created", "id": row.id, "symbol": row.symbol, "mode": row.mode,
                             "source": row.rejection_source, "rule": rule}, clock=self.clock)
        return row

    # ------------------------------------------------------------------ evaluate

    def _bars(self, row: ShadowTrade) -> pd.DataFrame:
        if self.data is None:
            return pd.DataFrame()
        start = to_ist(row.created_at)
        mode = Mode(row.mode)
        tfs = (Timeframe.M1, Timeframe.M5) if mode in (Mode.INTRADAY, Mode.BTST) else (Timeframe.D1,)
        for tf in tfs:
            try:
                if tf == Timeframe.D1:
                    df = self.data.candles(row.symbol, tf, lookback_bars=200)
                    df = df[df.index.date > start.date()] if not df.empty else df
                else:
                    df = self.data.candles(row.symbol, tf, start=start.replace(second=0, microsecond=0))
                    df = df[df.index >= start.replace(second=0, microsecond=0)] if not df.empty else df
            except Exception as exc:  # noqa: BLE001 - data failure: try next timeframe
                logger.warning(f"[shadow] candles {row.symbol} {tf.value}: {exc}")
                continue
            if df is not None and not df.empty:
                if row.expires_at is not None:
                    df = df[df.index <= to_ist(row.expires_at)]
                return df
        return pd.DataFrame()

    @staticmethod
    def simulate(row: ShadowTrade, df: pd.DataFrame, now: datetime) -> ShadowResult:
        """Pure outcome simulation over bars ``df`` (open/high/low/close)."""
        long_ = row.direction == Direction.LONG.value
        entered_at = to_ist(row.entered_at) if row.entered_at else None
        stop, entry = row.stop_loss, row.entry_price
        t1_hit = False
        last_close: float | None = None
        last_ts: datetime | None = None
        for ts, bar in df.iterrows():
            ts = ts.to_pydatetime()
            o, h, lo, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
            last_close, last_ts = c, ts
            if entered_at is None:
                touched = lo <= entry if long_ else h >= entry
                if not touched:
                    continue
                entered_at = ts
            # stop first (conservative)
            if (long_ and lo <= stop) or (not long_ and h >= stop):
                px = min(o, stop) if long_ else max(o, stop)
                if t1_hit:
                    return ShadowResult(ShadowOutcome.TARGET_1, entered_at, row.target_1, ts)
                return ShadowResult(ShadowOutcome.SL, entered_at, px, ts)
            if not t1_hit and ((long_ and h >= row.target_1) or (not long_ and lo <= row.target_1)):
                t1_hit = True
                if row.target_2 is None:
                    return ShadowResult(ShadowOutcome.TARGET_1, entered_at, row.target_1, ts)
                stop = entry  # breakeven after T1
            if t1_hit and row.target_2 is not None and ((long_ and h >= row.target_2) or (not long_ and lo <= row.target_2)):
                return ShadowResult(ShadowOutcome.TARGET_2, entered_at, row.target_2, ts)
        expired = row.expires_at is not None and now >= to_ist(row.expires_at)
        if expired:
            if entered_at is None:
                return ShadowResult(ShadowOutcome.NOT_TRIGGERED)
            if t1_hit:
                return ShadowResult(ShadowOutcome.TARGET_1, entered_at, row.target_1, last_ts)
            return ShadowResult(ShadowOutcome.TIME, entered_at, last_close, last_ts)
        return ShadowResult(ShadowOutcome.OPEN, entered_at)

    def update(self, session: Session, now: datetime | None = None) -> int:
        """Advance every OPEN shadow; returns how many were closed. Never raises."""
        now = to_ist(now or self.clock.now())
        closed = 0
        rows = session.execute(select(ShadowTrade).where(ShadowTrade.status == "OPEN")).scalars().all()
        for row in rows:
            try:
                res = self.simulate(row, self._bars(row), now)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[shadow] update {row.id} failed: {exc}")
                continue
            if res.entered_at is not None and row.entered_at is None:
                row.entered_at = res.entered_at
            if res.outcome == ShadowOutcome.OPEN:
                continue
            self._close(row, res)
            closed += 1
        session.flush()
        return closed

    def _close(self, row: ShadowTrade, res: ShadowResult) -> None:
        row.status = "CLOSED"
        row.outcome = res.outcome.value
        row.exit_price = res.exit_price
        row.exit_time = res.exit_time
        risk = abs(row.entry_price - row.stop_loss)
        if res.exit_price is not None and risk > 0:
            sign = 1.0 if row.direction == Direction.LONG.value else -1.0
            row.r_multiple = round(sign * (res.exit_price - row.entry_price) / risk, 4)
            if row.qty > 0:
                ch = round_trip_charges(product_for(Mode(row.mode)), Direction(row.direction), row.qty,
                                        row.entry_price, res.exit_price, self.settings.charges).total
                row.hypothetical_net_pnl = round(sign * (res.exit_price - row.entry_price) * row.qty - ch, 4)
        log_event("shadow", {"event": "closed", "id": row.id, "symbol": row.symbol, "outcome": row.outcome,
                             "r": row.r_multiple, "net": row.hypothetical_net_pnl}, clock=self.clock)
