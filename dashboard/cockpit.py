"""Data for the Home cockpit page (pure functions over a Session; unit-testable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config.schema import HardCaps, Settings
from config.secrets import Secrets
from core.clock import IST
from core.enums import BatchStatus, PositionStatus
from storage.models import Batch, CapitalAccount, ErrorLog, Position, Trade


@dataclass
class Cockpit:
    batch_name: str | None = None
    batch_status: str | None = None
    halted: bool = False
    halt_reason: str | None = None
    trading: float | None = None
    reserve: float | None = None
    equity: float | None = None
    hwm: float | None = None
    drawdown_pct: float | None = None
    tier: str | None = None
    today_net: float = 0.0
    today_trades: int = 0
    unrealized: float = 0.0
    open_positions: int = 0
    errors_today: int = 0
    starting_capital: float | None = None
    total_return_pct: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _unrealized(p: Position) -> float:
    if p.last_price is None:
        return 0.0
    sign = 1.0 if p.direction == "LONG" else -1.0
    return sign * (p.last_price - p.avg_entry) * p.qty - (p.entry_charges or 0.0)


def cockpit(session: Session, batch_id: int | None, now: datetime) -> Cockpit:
    """Headline numbers for the selected batch."""
    c = Cockpit()
    if batch_id is None:
        return c
    b = session.get(Batch, batch_id)
    if b is not None:
        c.batch_name, c.batch_status = b.name, b.status
        c.halted = b.status == BatchStatus.HALTED.value
        c.halt_reason = b.halt_reason
    acct = session.get(CapitalAccount, batch_id)
    positions = list(session.execute(select(Position).where(Position.batch_id == batch_id,
                                                            Position.status == PositionStatus.OPEN.value)).scalars())
    c.open_positions = len(positions)
    c.unrealized = round(sum(_unrealized(p) for p in positions), 2)
    start = datetime.combine(now.date(), time(0, 0), tzinfo=IST)
    n, net = session.execute(select(func.count(Trade.id), func.coalesce(func.sum(Trade.net_pnl), 0.0))
                             .where(Trade.batch_id == batch_id, Trade.exit_time >= start,
                                    Trade.exit_time < start + timedelta(days=1))).one()
    c.today_trades, c.today_net = int(n), round(float(net), 2)
    if acct is not None:
        c.trading = round(acct.trading_cash + acct.blocked, 2)
        c.reserve = round(acct.reserve, 2)
        c.equity = round(acct.trading_cash + acct.blocked + acct.reserve + c.unrealized, 2)
        c.hwm = round(max(acct.hwm, c.equity), 2)
        c.drawdown_pct = round((c.hwm - c.equity) / c.hwm * 100.0, 2) if c.hwm else 0.0
        c.tier = acct.tier
        c.halted = c.halted or bool(acct.halted)
        c.halt_reason = c.halt_reason or acct.halt_reason
        c.starting_capital = acct.starting_capital
        if acct.starting_capital:
            c.total_return_pct = round((c.equity - acct.starting_capital) / acct.starting_capital * 100.0, 2)
    c.errors_today = int(session.scalar(select(func.count(ErrorLog.id)).where(ErrorLog.ts >= start)) or 0)
    return c


@dataclass
class CheckItem:
    name: str
    ok: bool | None          # None = unknown / informational
    detail: str
    fix: str = ""


def setup_checklist(settings: Settings, caps: HardCaps, secrets: Secrets, *, engine_alive: bool,
                    engine_data_ok: bool | None, ollama_models: list[str] | None, today: date) -> list[CheckItem]:
    """First-run / health checklist in plain language."""
    items = [CheckItem("Paper trading only", caps.paper_only is True, "Real orders are impossible.")]
    items.append(CheckItem("Engine running", engine_alive,
                           "Heartbeat is fresh." if engine_alive else "No recent heartbeat from the engine.",
                           "Start it in a terminal:  .\\tasks.ps1 run   (or: python main.py engine)"))
    if settings.engine.llm_backend == "rules":
        items.append(CheckItem("AI models", None, "LLM backend is 'rules' (deterministic agents, no Ollama needed).",
                               "Switch Settings > Agents & models to use local LLMs."))
    else:
        prof = settings.models.profiles[settings.models.active_profile]
        wanted = {prof.sub_agent, prof.reasoning} | set(settings.models.assignments.values())
        if ollama_models is None:
            items.append(CheckItem("AI models (Ollama)", False, "Ollama is not reachable.",
                                   "Install from ollama.com/download, keep it running, then  .\\tasks.ps1 pull-models"))
        else:
            missing = sorted(m for m in wanted if m and m not in ollama_models)
            items.append(CheckItem("AI models (Ollama)", not missing,
                                   "All configured models installed." if not missing else f"Missing: {', '.join(missing)}",
                                   ".\\tasks.ps1 pull-models"))
    email_ok = bool(secrets.smtp_user and secrets.smtp_app_password and (secrets.alert_email_to or settings.notifications.recipients))
    items.append(CheckItem("Email alerts", email_ok, "Gmail configured." if email_ok else "SMTP not configured.",
                           "Put SMTP_USER, SMTP_APP_PASSWORD (Gmail App Password) and ALERT_EMAIL_TO in .env"))
    items.append(CheckItem("Market data", engine_data_ok,
                           f"Provider: {settings.engine.data_provider}" + ("" if engine_data_ok is not False
                                                                         else " - engine reports the feed DOWN"),
                           "Check internet / proxy; Angel One keys in .env for better intraday data."))
    age = (today - settings.charges.last_verified).days
    items.append(CheckItem("Charges rates verified", age <= 180, f"Last verified {age} days ago.",
                           "Check your broker's charges page and update config/charges.yaml last_verified."))
    items.append(CheckItem("Tax slab set", None, f"Using slab rate {settings.tax.slab_rate_pct:g}% for intraday income.",
                           "Analytics > Tax & charges > Tax settings"))
    return items
