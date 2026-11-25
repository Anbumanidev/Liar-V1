"""Global result scope shared by every page: Today (a date) | Batch (one batch) | Overall (all batches).

Query params understood by the scoped endpoints: ``scope=today|batch|all`` + ``date=YYYY-MM-DD`` (today)
or ``batch_id=N`` (batch; default = the active batch, else the latest one).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from fastapi import HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from storage.models import Batch
from webui import queries as q
from webui.util import NotAvailable, day_bounds, iso, optional_attr, parse_day

SCOPES = ("today", "batch", "all")


@dataclass(frozen=True)
class Scope:
    kind: str                     # today | batch | all
    day: date | None              # today scope only
    batch_id: int | None          # batch scope (or an explicit extra filter on today)
    lo: datetime | None
    hi: datetime | None
    label: str
    batch_name: str | None = None
    batch_status: str | None = None

    @property
    def is_today(self) -> bool:
        return self.kind == "today"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "date": self.day.isoformat() if self.day else None, "batch_id": self.batch_id,
                "batch_name": self.batch_name, "batch_status": self.batch_status, "label": self.label,
                "from": iso(self.lo), "to": iso(self.hi)}


def resolve_scope(session: Session, now: datetime, scope: str | None, day: str | None, batch_id: int | None,
                  *, default: str = "today") -> Scope:
    kind = (scope or default).strip().lower()
    if kind in ("overall", "total"):
        kind = "all"
    if kind not in SCOPES:
        raise HTTPException(status_code=422, detail=f"scope must be one of {', '.join(SCOPES)}")
    if batch_id is not None and not (1 <= batch_id < 2**63):
        raise HTTPException(status_code=422, detail="invalid batch_id")
    if kind == "today":
        d = parse_day(day, now.date())
        if not (date(2000, 1, 1) <= d <= date(2100, 12, 31)):
            raise HTTPException(status_code=422, detail="date out of range")
        lo, hi = day_bounds(d)
        label = "Today" if d == now.date() else d.strftime("%d %b %Y")
        b = session.get(Batch, batch_id) if batch_id is not None else None
        return Scope("today", d, batch_id, lo, hi, label, b.name if b else None, b.status if b else None)
    if kind == "batch":
        b = session.get(Batch, batch_id) if batch_id is not None else (q.current_batch(session) or session.execute(
            select(Batch).where(Batch.deleted.is_(False)).order_by(Batch.id.desc()).limit(1)).scalar_one_or_none())
        if b is None or b.deleted:
            raise HTTPException(status_code=404, detail="no batch found" if batch_id is None
                                else f"batch {batch_id} not found")
        return Scope("batch", None, b.id, None, None, b.name, b.name, b.status)
    return Scope("all", None, None, None, None, "Overall (all batches)")


def funding_or_fallback(session: Session, unrealized_by_batch: dict[int, float]) -> tuple[dict[str, Any], str]:
    """``capital.funding.funding_summary`` when available, else the approximate fallback."""
    try:
        return optional_attr("capital.funding", "funding_summary")(session), "funding"
    except NotAvailable:
        pass
    except Exception as exc:  # noqa: BLE001 - keep pages usable
        logger.warning(f"[webui] funding_summary failed, using fallback: {exc}")
        session.rollback()
    return q.batches_fallback(session, unrealized_by_batch), "fallback"
