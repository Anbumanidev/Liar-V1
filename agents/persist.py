"""Persistence of sub-agent reports into ``agent_reports`` (rule 0.7: everything is logged)."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from core.logging import to_jsonable
from core.types import AgentReport
from storage.models import AgentReportRow


def report_to_row(
    report: AgentReport,
    run_id: int | None,
    batch_id: int | None,
    decision_id: int | None = None,
) -> AgentReportRow:
    """Map one ``AgentReport`` to an (unsaved) ``AgentReportRow``."""
    return AgentReportRow(
        run_id=run_id,
        batch_id=batch_id,
        decision_id=decision_id,
        symbol=report.symbol,
        mode=report.mode.value,
        agent=report.agent,
        signal=report.signal.value,
        confidence=int(report.confidence),
        status=report.status.value,
        model=report.model,
        prompt_version=report.prompt_version,
        latency_ms=report.latency_ms,
        tokens_in=report.tokens_in,
        tokens_out=report.tokens_out,
        report_json=to_jsonable(report.model_dump(mode="json")),
        error=report.error,
        created_at=report.timestamp,
    )


def save_agent_reports(
    session: Session,
    reports: Sequence[AgentReport],
    run_id: int | None,
    batch_id: int | None,
    decision_id: int | None = None,
) -> list[AgentReportRow]:
    """Add one ``agent_reports`` row per report and flush (the caller commits).

    Returns:
        The new rows (ids assigned by the flush).
    """
    rows = [report_to_row(r, run_id, batch_id, decision_id) for r in reports]
    session.add_all(rows)
    session.flush()
    return rows


def link_reports_to_decision(session: Session, rows: Sequence[AgentReportRow], decision_id: int) -> None:
    """Set ``decision_id`` on rows saved before the decision row existed (caller commits)."""
    for row in rows:
        row.decision_id = decision_id
    session.flush()
