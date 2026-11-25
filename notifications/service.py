"""Notifier: turns trading events into PAPER-TRADING e-mails (section 13).

Usage (engine bootstrap)::

    notifier = Notifier(ctx.session_factory, ctx.settings, ctx.secrets, ctx.clock)
    notifier.pending_approval(item)       # ApprovalItem row
    notifier.entry(position, order)       # Position + filled entry Order rows
    notifier.exit(trade)                  # Trade row (one closed leg)
    notifier.eod(build_eod_report(session, batch_id, day, clock))
    notifier.swing_scan(candidates)
    notifier.urgent("HALT", "Equity hit the trailing floor", {"equity": 905.2})
    notifier.test()

Behaviour:

* Per-type toggles: ``settings.notifications.enabled_types[EmailType]``.
* Quiet hours ``quiet_hours_start``-``quiet_hours_end`` (IST, overnight ranges such as
  22:00-07:00 supported). URGENT bypasses them when ``urgent_ignores_quiet_hours``;
  TEST (an explicit owner action) always bypasses them.
* URGENT de-duplication: an identical (title, message) urgent e-mail within
  ``urgent_dedupe_minutes`` is skipped (avoids a mail storm while a feed stays down).
* ``async_send=True``: rendering + SMTP run on ONE background daemon thread fed by a
  bounded queue, so the trading loop never waits for e-mail. Methods then return a
  ``QUEUED`` outcome. ``flush()`` waits for the queue (tests / shutdown).
* Every decision (SENT / FAILED / SKIPPED) is written to ``notifications_log`` and to the
  ``notifications`` JSONL log. Nothing here raises into the caller.

Context dicts are extracted from the ORM rows synchronously (so detached/expired rows
are never touched from the worker thread); all numbers are computed here in Python.
"""

from __future__ import annotations

import html as html_lib
import json
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Protocol

from jinja2 import Environment
from loguru import logger
from sqlalchemy.orm import Session, sessionmaker

from config.schema import NotificationSettings, Settings
from config.secrets import Secrets
from core.clock import Clock, parse_hhmm, to_ist
from core.enums import Direction, EmailType, ExitReason, Mode, Product
from core.logging import log_event, to_jsonable
from modes.base import product_for
from notifications.email import (
    STATUS_FAILED,
    STATUS_SENT,
    STATUS_SKIPPED,
    Attachment,
    EmailSender,
    SendResult,
)
from notifications.reports import equity_chart_png
from notifications.templating import (
    build_environment,
    fmt_date,
    fmt_dt,
    fmt_duration,
    fmt_inr,
    fmt_num,
    fmt_price,
    render_template,
)
from storage.audit import record_error
from storage.db import session_scope
from storage.models import NotificationLog

STATUS_QUEUED = "QUEUED"

EXIT_REASON_LABELS: dict[str, str] = {
    ExitReason.SL.value: "Stop-loss hit",
    ExitReason.TARGET.value: "Target hit",
    ExitReason.TRAIL.value: "Trailing stop hit",
    ExitReason.TIME.value: "Time stop",
    ExitReason.SQUAREOFF.value: "Intraday square-off",
    ExitReason.GAP.value: "Gap through stop-loss",
    ExitReason.EVENT.value: "Pre-event exit",
    ExitReason.HALT.value: "Capital-protection HALT",
    ExitReason.MANUAL.value: "Manual exit (owner)",
    ExitReason.KILL_SWITCH.value: "Kill switch",
    ExitReason.BATCH_CLOSE.value: "Batch closed",
}

_CHARGE_LABELS: dict[str, str] = {
    "brokerage": "Brokerage",
    "stt": "STT",
    "exchange_txn": "Exchange transaction",
    "sebi": "SEBI fee",
    "stamp_duty": "Stamp duty",
    "gst": "GST",
    "dp": "DP charge",
    "ipft": "IPFT",
}

_ACCENTS: dict[EmailType, str] = {
    EmailType.PENDING_APPROVAL: "#c98500",
    EmailType.ENTRY: "#2a78d6",
    EmailType.EXIT: "#52514e",
    EmailType.EOD: "#2a78d6",
    EmailType.SWING_SCAN: "#6250d6",
    EmailType.URGENT: "#c0262d",
    EmailType.TEST: "#1baf7a",
}

_TEMPLATES: dict[EmailType, str] = {
    EmailType.PENDING_APPROVAL: "pending_approval",
    EmailType.ENTRY: "entry",
    EmailType.EXIT: "exit",
    EmailType.EOD: "eod",
    EmailType.SWING_SCAN: "swing_scan",
    EmailType.URGENT: "urgent",
    EmailType.TEST: "test",
}

CHART_CID = "equity_chart"


# =========================================================================== small helpers


class SenderLike(Protocol):
    """Anything with ``EmailSender.send``'s signature (tests inject fakes)."""

    def send(
        self,
        subject: str,
        html: str,
        recipients: Sequence[str] | str | None = None,
        attachments: Sequence[Attachment] | None = None,
        text: str | None = None,
    ) -> SendResult: ...


@dataclass
class NotificationOutcome:
    """What happened to one notification request.

    ``status``: ``SENT`` | ``FAILED`` | ``SKIPPED`` (sync mode or gated) | ``QUEUED`` (async).
    """

    email_type: EmailType
    status: str
    subject: str
    reason: str | None = None
    attempts: int = 0
    recipients: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when sent (or queued for sending)."""
        return self.status in (STATUS_SENT, STATUS_QUEUED)


@dataclass
class _Job:
    email_type: EmailType
    subject: str
    template: str
    context: dict[str, Any]
    batch_id: int | None = None
    attachment_builder: Callable[[dict[str, Any]], list[Attachment]] | None = None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or mapping access (ORM rows, Pydantic models and dicts are all accepted)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _first(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        v = _get(obj, n)
        if v is not None:
            return v
    return default


def _float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_value(value: Any) -> str | None:
    """Enum / plain value -> string."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def reward_risk(entry: float | None, stop: float | None, target: float | None) -> float | None:
    """|target - entry| / |entry - stop| (None when undefined)."""
    if entry is None or stop is None or target is None:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    return round(abs(target - entry) / risk, 2)


def side_label(direction: str | None) -> str:
    """LONG -> BUY, SHORT -> SELL SHORT."""
    if direction == Direction.SHORT.value:
        return "SELL SHORT"
    if direction == Direction.LONG.value:
        return "BUY"
    return direction or "?"


def in_quiet_hours(now: datetime | time, start: str | None, end: str | None) -> bool:
    """True when ``now`` (IST) falls in the quiet window ``[start, end)``.

    Overnight windows (start > end, e.g. 22:00-07:00) wrap past midnight. A window with
    a missing bound, or start == end, is treated as "no quiet hours".
    """
    if not start or not end:
        return False
    try:
        s, e = parse_hhmm(start), parse_hhmm(end)
    except ValueError as exc:
        logger.warning(f"Invalid quiet hours {start!r}-{end!r}: {exc}; ignoring")
        return False
    t = to_ist(now).time() if isinstance(now, datetime) else now
    t = t.replace(tzinfo=None)
    if s == e:
        return False
    if s < e:
        return s <= t < e
    return t >= s or t < e


def extract_points(summary: Mapping[str, Any] | None, side: str, limit: int = 3) -> list[str]:
    """Top bull/bear points from ``approval_queue.summary_json``.

    Accepted shapes (first match wins): ``{"bull_points": [...]}``, ``{"top_bull": [...]}``,
    ``{"bull_arguments": [...]}``, ``{"bull": [...]}``, ``{"bull": {"top_arguments": [...]}}``,
    ``{"debate": {"bull": {"top_arguments": [...]}}}``. Items may be strings or dicts with
    ``text`` / ``point`` / ``argument``.
    """
    if not summary:
        return []
    candidates: list[Any] = [
        summary.get(f"{side}_points"),
        summary.get(f"top_{side}"),
        summary.get(f"top_{side}_points"),
        summary.get(f"{side}_arguments"),
        summary.get(side),
    ]
    debate = summary.get("debate")
    if isinstance(debate, Mapping):
        candidates.append(debate.get(side))
    for cand in candidates:
        if isinstance(cand, Mapping):
            cand = cand.get("top_arguments") or cand.get("points") or cand.get("arguments")
        if isinstance(cand, str):
            cand = [cand]
        if isinstance(cand, (list, tuple)) and cand:
            out: list[str] = []
            for item in cand:
                if isinstance(item, Mapping):
                    item = item.get("text") or item.get("point") or item.get("argument") or json.dumps(item)
                text = " ".join(str(item).split())
                if not text:
                    continue
                out.append(text if len(text) <= 220 else text[:217] + "...")
                if len(out) >= limit:
                    break
            if out:
                return out
    return []


def charges_breakdown(charges_json: Any) -> list[tuple[str, float]]:
    """Flatten ``charges_json`` (flat ChargesBreakdown dict or {entry: {...}, exit: {...}}) to labelled rows."""
    if not isinstance(charges_json, Mapping):
        return []
    totals: dict[str, float] = {}

    def _acc(d: Mapping[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, Mapping):
                _acc(value)
            elif key in _CHARGE_LABELS:
                v = _float(value)
                if v is not None:
                    totals[key] = totals.get(key, 0.0) + v

    _acc(charges_json)
    return [(_CHARGE_LABELS[k], round(v, 4)) for k, v in totals.items() if abs(v) > 1e-9]


def _charges_total(charges_json: Any) -> float | None:
    rows = charges_breakdown(charges_json)
    if rows:
        return round(sum(v for _, v in rows), 2)
    if isinstance(charges_json, Mapping):
        return _float(charges_json.get("total"))
    return None


def _mask_email(addr: str | None) -> str:
    if not addr or "@" not in addr:
        return addr or "—"
    user, _, domain = addr.partition("@")
    return f"{user[:2]}***@{domain}"


# =========================================================================== notifier


class Notifier:
    """E-mail notification service (section 13). Never raises into trading code.

    Args:
        session_factory: SQLAlchemy sessionmaker (``notifications_log`` writes).
        settings: active ``Settings`` (``settings.notifications`` is used).
        secrets: ``Secrets`` (SMTP credentials, ALERT_EMAIL_TO).
        clock: time source (quiet hours, deadlines, log timestamps).
        sender: optional sender (default ``EmailSender(secrets, settings.notifications)``).
        async_send: send on a background thread (default True).
        batch_id: default batch id for log rows when the event does not carry one.
        urgent_dedupe_minutes: suppress identical URGENT e-mails within this window (0 = off).
        queue_size: maximum queued e-mails in async mode (overflow is dropped + logged).
        env: optional Jinja2 environment (tests / custom templates).
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None,
        settings: Settings,
        secrets: Secrets | None,
        clock: Clock,
        sender: SenderLike | None = None,
        async_send: bool = True,
        *,
        batch_id: int | None = None,
        urgent_dedupe_minutes: float = 15.0,
        queue_size: int = 500,
        env: Environment | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.secrets = secrets
        self.clock = clock
        self.sender: SenderLike = sender if sender is not None else EmailSender(secrets, settings.notifications)
        self.async_send = async_send
        self.batch_id = batch_id
        self.urgent_dedupe_minutes = urgent_dedupe_minutes
        self.env = env if env is not None else build_environment()

        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue(maxsize=max(1, queue_size))
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._idle = threading.Condition()
        self._pending = 0
        self._closed = False
        self._recent_urgent: dict[tuple[str, str], datetime] = {}
        self._urgent_lock = threading.Lock()

    # ------------------------------------------------------------------ configuration

    @property
    def cfg(self) -> NotificationSettings:
        """Current notification settings."""
        return self.settings.notifications

    def update_settings(self, settings: Settings) -> None:
        """Apply reloaded settings (RELOAD_SETTINGS); the default sender follows them."""
        self.settings = settings
        if isinstance(self.sender, EmailSender):
            self.sender.cfg = settings.notifications

    def type_enabled(self, email_type: EmailType) -> bool:
        """Per-type toggle (missing entries default to enabled)."""
        return bool(self.cfg.enabled_types.get(email_type, True))

    def is_quiet_now(self) -> bool:
        """True when the clock is inside the configured quiet hours."""
        return in_quiet_hours(self.clock.now(), self.cfg.quiet_hours_start, self.cfg.quiet_hours_end)

    def _gate(self, email_type: EmailType) -> str | None:
        """Reason to skip this e-mail type now, or None to send."""
        if not self.type_enabled(email_type):
            return f"{email_type.value} e-mails disabled in settings"
        if self.is_quiet_now():
            if email_type == EmailType.TEST:
                return None
            if email_type == EmailType.URGENT and self.cfg.urgent_ignores_quiet_hours:
                return None
            return f"quiet hours {self.cfg.quiet_hours_start}-{self.cfg.quiet_hours_end}"
        return None

    def _base_context(self, email_type: EmailType, title: str, kicker: str, subtitle: str | None = None) -> dict[str, Any]:
        return {
            "title": title,
            "kicker": kicker,
            "subtitle": subtitle,
            "accent": _ACCENTS.get(email_type),
            "generated_at": self.clock.now(),
            "email_type": email_type.value,
        }

    # ------------------------------------------------------------------ public API

    def pending_approval(self, item: Any) -> NotificationOutcome:
        """PENDING APPROVAL e-mail for an ``approval_queue`` row (approval itself only in the dashboard)."""
        try:
            p = self._approval_context(item)
            subject = (
                f"APPROVAL NEEDED: {p['side']} {p['symbol']} ({p['mode']}) "
                f"- decide by {fmt_dt(p['expires_at'], '%H:%M')}"
            )
            ctx = self._base_context(
                EmailType.PENDING_APPROVAL,
                f"Approval needed: {p['side']} {p['qty']} {p['symbol']}",
                "Pending approval",
                f"{p['mode']} · {p['product']} · decide in the dashboard by {fmt_dt(p['expires_at'])}",
            )
            ctx["p"] = p
            return self._dispatch(_Job(EmailType.PENDING_APPROVAL, subject, _TEMPLATES[EmailType.PENDING_APPROVAL],
                                       ctx, batch_id=_get(item, "batch_id", self.batch_id)))
        except Exception as exc:  # noqa: BLE001
            return self._internal_failure(EmailType.PENDING_APPROVAL, "pending approval", exc)

    def entry(self, position: Any, order: Any = None) -> NotificationOutcome:
        """ENTRY e-mail for a newly filled paper position."""
        try:
            e = self._entry_context(position, order)
            subject = f"ENTRY {e['side']} {e['qty']} {e['symbol']} @ {fmt_price(e['fill_price'])} ({e['mode']})"
            ctx = self._base_context(
                EmailType.ENTRY,
                f"Entered {e['side']} {e['qty']} {e['symbol']} @ {fmt_price(e['fill_price'])}",
                "Paper entry filled",
                f"{e['mode']} · {e['product']} · {e['strategy'] or 'manual'}",
            )
            ctx["e"] = e
            return self._dispatch(_Job(EmailType.ENTRY, subject, _TEMPLATES[EmailType.ENTRY], ctx,
                                       batch_id=_get(position, "batch_id", self.batch_id)))
        except Exception as exc:  # noqa: BLE001
            return self._internal_failure(EmailType.ENTRY, "entry", exc)

    def exit(self, trade: Any) -> NotificationOutcome:
        """EXIT e-mail for one closed leg (``trades`` row)."""
        try:
            x = self._exit_context(trade)
            subject = (
                f"EXIT {x['symbol']} ({x['mode']}) {x['exit_reason']}: net {fmt_inr(x['net'], signed=True)}"
                + (" [partial]" if x["is_partial"] else "")
            )
            ctx = self._base_context(
                EmailType.EXIT,
                f"Exited {x['symbol']}: net {fmt_inr(x['net'], signed=True)}",
                "Paper exit" + (" (partial)" if x["is_partial"] else ""),
                f"{x['mode']} · {x['exit_reason_label']} · held {x['holding']}",
            )
            ctx["x"] = x
            return self._dispatch(_Job(EmailType.EXIT, subject, _TEMPLATES[EmailType.EXIT], ctx,
                                       batch_id=_get(trade, "batch_id", self.batch_id)))
        except Exception as exc:  # noqa: BLE001
            return self._internal_failure(EmailType.EXIT, "exit", exc)

    def eod(self, report: Mapping[str, Any]) -> NotificationOutcome:
        """EOD e-mail from ``notifications.reports.build_eod_report`` output (with equity chart)."""
        try:
            r = normalize_eod_report(report)
            cap = r["capital"]
            day_label = fmt_date(r["day"]) if r["day"] else "today"
            subject = (
                f"EOD {day_label}: net {fmt_inr(r['totals']['net'], signed=True)}"
                f" | equity {fmt_inr(cap.get('total_equity'))}"
                + (" | HALTED" if cap.get("halted") else "")
            )
            subtitle = f"Batch {r['batch_name'] or r['batch_id']} · {r['totals']['trades']} trades"
            ctx = self._base_context(EmailType.EOD, f"End of day {day_label}", "Daily report", subtitle)
            ctx["r"] = r
            ctx["chart_cid"] = None
            ctx["limits"] = {"trades": 25, "not_traded": 30}
            points = list(r["equity_series"])
            baseline = _float(cap.get("starting_capital"))

            def _chart(c: dict[str, Any]) -> list[Attachment]:
                png = equity_chart_png(points, baseline=baseline)
                c["chart_cid"] = CHART_CID
                return [Attachment.inline_png("equity_chart.png", png, CHART_CID)]

            return self._dispatch(_Job(EmailType.EOD, subject, _TEMPLATES[EmailType.EOD], ctx,
                                       batch_id=r.get("batch_id") or self.batch_id, attachment_builder=_chart))
        except Exception as exc:  # noqa: BLE001
            return self._internal_failure(EmailType.EOD, "eod", exc)

    def swing_scan(
        self,
        candidates: Sequence[Any],
        *,
        for_day: date | None = None,
        batch_id: int | None = None,
    ) -> NotificationOutcome:
        """SWING SCAN e-mail: next-session candidates (dicts, Pydantic models or rows)."""
        try:
            items = [self._candidate_context(c) for c in candidates]
            when = for_day.strftime("%d %b %Y") if for_day else "next session"
            subject = f"Swing scan: {len(items)} candidate{'s' if len(items) != 1 else ''} for {when}"
            if items:
                subject += " - " + ", ".join(c["symbol"] for c in items[:5]) + ("..." if len(items) > 5 else "")
            ctx = self._base_context(EmailType.SWING_SCAN, f"Swing scan for {when}", "Evening scan",
                                     f"{len(items)} candidate(s)")
            ctx["candidates"] = items
            return self._dispatch(_Job(EmailType.SWING_SCAN, subject, _TEMPLATES[EmailType.SWING_SCAN], ctx,
                                       batch_id=batch_id if batch_id is not None else self.batch_id))
        except Exception as exc:  # noqa: BLE001
            return self._internal_failure(EmailType.SWING_SCAN, "swing scan", exc)

    def urgent(
        self,
        title: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        *,
        batch_id: int | None = None,
    ) -> NotificationOutcome:
        """URGENT e-mail (HALT, gap through SL on an overnight position, data feed down)."""
        try:
            subject = f"URGENT: {title}"
            key = (title.strip(), message.strip())
            now = self.clock.now()
            if self.urgent_dedupe_minutes > 0:
                with self._urgent_lock:
                    last = self._recent_urgent.get(key)
                    if last is not None and now - last < timedelta(minutes=self.urgent_dedupe_minutes):
                        return self._skip(EmailType.URGENT, subject,
                                          f"duplicate of an urgent e-mail sent at {fmt_dt(last, '%H:%M')}",
                                          batch_id if batch_id is not None else self.batch_id)
            rows = [(str(k).replace("_", " ").capitalize(), self._fmt_detail(v)) for k, v in (details or {}).items()]
            ctx = self._base_context(EmailType.URGENT, title, "Urgent alert")
            ctx["message"] = message
            ctx["details"] = rows
            outcome = self._dispatch(_Job(EmailType.URGENT, subject, _TEMPLATES[EmailType.URGENT], ctx,
                                          batch_id=batch_id if batch_id is not None else self.batch_id))
            if outcome.status != STATUS_SKIPPED and self.urgent_dedupe_minutes > 0:
                with self._urgent_lock:
                    self._recent_urgent[key] = now
                    cutoff = now - timedelta(minutes=self.urgent_dedupe_minutes)
                    for k in [k for k, ts in self._recent_urgent.items() if ts < cutoff]:
                        del self._recent_urgent[k]
            return outcome
        except Exception as exc:  # noqa: BLE001
            return self._internal_failure(EmailType.URGENT, "urgent", exc)

    def test(self) -> NotificationOutcome:
        """TEST e-mail (dashboard 'send test e-mail'). Bypasses quiet hours."""
        try:
            cfg = self.cfg
            resolver = getattr(self.sender, "resolve_recipients", None)
            recipients = resolver() if callable(resolver) else list(cfg.recipients)
            quiet = (f"{cfg.quiet_hours_start}-{cfg.quiet_hours_end} IST"
                     if cfg.quiet_hours_start and cfg.quiet_hours_end else "off")
            smtp_user = self.secrets.smtp_user if self.secrets is not None else None
            rows = [
                ("Recipients", ", ".join(recipients) if recipients else "none configured"),
                ("SMTP server", f"{cfg.smtp_host}:{cfg.smtp_port} (STARTTLS)"),
                ("SMTP user", _mask_email(smtp_user)),
                ("Quiet hours", quiet),
                ("Urgent ignores quiet hours", "yes" if cfg.urgent_ignores_quiet_hours else "no"),
                ("Retries", f"{cfg.retries} (every {fmt_num(cfg.retry_delay_seconds, 0)} s)"),
                ("Subject prefix", cfg.subject_prefix),
                ("Engine clock", fmt_dt(self.clock.now(), "%d %b %Y %H:%M:%S")),
            ]
            ctx = self._base_context(EmailType.TEST, "Test e-mail", "Notification test",
                                     "SMTP delivery check from Liar-V1")
            ctx["rows"] = rows
            ctx["enabled_types"] = [(t.value, self.type_enabled(t)) for t in EmailType]
            return self._dispatch(_Job(EmailType.TEST, "Test e-mail", _TEMPLATES[EmailType.TEST], ctx,
                                       batch_id=self.batch_id))
        except Exception as exc:  # noqa: BLE001
            return self._internal_failure(EmailType.TEST, "test", exc)

    # ------------------------------------------------------------------ async machinery

    def flush(self, timeout: float | None = 30.0) -> bool:
        """Wait until every queued e-mail has been processed. Returns False on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._pending == 0, timeout=timeout)

    def close(self, timeout: float = 30.0) -> None:
        """Drain the queue and stop the worker thread (idempotent)."""
        self.flush(timeout)
        with self._thread_lock:
            self._closed = True
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            try:
                self._queue.put(None, timeout=1.0)
            except queue.Full:
                logger.warning("Notifier queue full on close; worker will exit when the process ends")
            thread.join(timeout=timeout)

    @property
    def pending(self) -> int:
        """Number of e-mails queued or in progress."""
        with self._idle:
            return self._pending

    def _ensure_worker(self) -> None:
        with self._thread_lock:
            if self._thread is None or not self._thread.is_alive():
                self._closed = False
                self._thread = threading.Thread(target=self._worker, name="notifier-email", daemon=True)
                self._thread.start()

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
            try:
                task()
            except Exception as exc:  # noqa: BLE001 - worker must survive anything
                logger.exception(f"Notifier worker task failed: {exc}")
            finally:
                with self._idle:
                    self._pending -= 1
                    if self._pending <= 0:
                        self._pending = 0
                        self._idle.notify_all()

    def _submit(self, task: Callable[[], None]) -> bool:
        """Queue ``task`` for the worker. Never blocks; returns False when the queue is full."""
        self._ensure_worker()
        with self._idle:
            self._pending += 1
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            with self._idle:
                self._pending -= 1
                if self._pending <= 0:
                    self._pending = 0
                    self._idle.notify_all()
            return False

    # ------------------------------------------------------------------ dispatch

    def _dispatch(self, job: _Job) -> NotificationOutcome:
        reason = self._gate(job.email_type)
        if reason is not None:
            return self._skip(job.email_type, job.subject, reason, job.batch_id)
        if not self.async_send:
            return self._process(job)
        if self._submit(lambda: self._process(job)):
            return NotificationOutcome(job.email_type, STATUS_QUEUED, job.subject)
        reason = "notification queue full; e-mail dropped"
        logger.error(f"{reason}: {job.subject!r}")
        self._write_log(job.email_type, job.subject, STATUS_FAILED, 0, reason, [], job.batch_id)
        return NotificationOutcome(job.email_type, STATUS_FAILED, job.subject, reason=reason)

    def _skip(self, email_type: EmailType, subject: str, reason: str, batch_id: int | None) -> NotificationOutcome:
        logger.info(f"E-mail {email_type.value} skipped: {reason} ({subject!r})")
        if self.async_send:
            if not self._submit(lambda: self._write_log(email_type, subject, STATUS_SKIPPED, 0, reason, [], batch_id)):
                logger.error("Notifier queue full; SKIPPED log entry dropped")
        else:
            self._write_log(email_type, subject, STATUS_SKIPPED, 0, reason, [], batch_id)
        return NotificationOutcome(email_type, STATUS_SKIPPED, subject, reason=reason)

    def _render(self, job: _Job) -> tuple[str, list[Attachment]]:
        attachments: list[Attachment] = []
        if job.attachment_builder is not None:
            try:
                attachments = job.attachment_builder(job.context)
            except Exception as exc:  # noqa: BLE001 - e-mail goes out without the chart
                self._report_error(f"attachment build failed for {job.email_type.value}", exc)
                attachments = []
        try:
            html = render_template(self.env, job.template, job.context)
        except Exception as exc:  # noqa: BLE001 - degrade to a plain fallback body
            self._report_error(f"template {job.template} failed", exc)
            html = fallback_html(job.context.get("title") or job.subject, job.context)
            attachments = [a for a in attachments if not a.inline]
        return html, attachments

    def _process(self, job: _Job) -> NotificationOutcome:
        """Render + send + log one job (worker thread in async mode). Never raises."""
        try:
            html, attachments = self._render(job)
        except Exception as exc:  # noqa: BLE001
            self._report_error("render crashed", exc)
            html, attachments = fallback_html(job.subject, {}), []
        try:
            result = self.sender.send(job.subject, html, attachments=attachments)
        except Exception as exc:  # noqa: BLE001 - custom senders might raise
            self._report_error(f"sender raised for {job.email_type.value}", exc)
            result = SendResult(ok=False, attempts=1, error=f"{type(exc).__name__}: {exc}", status=STATUS_FAILED)
        status = result.status if result.status in (STATUS_SENT, STATUS_FAILED, STATUS_SKIPPED) else (
            STATUS_SENT if result.ok else STATUS_FAILED)
        subject = result.subject or job.subject
        self._write_log(job.email_type, subject, status, result.attempts, result.error, result.recipients,
                        job.batch_id)
        return NotificationOutcome(job.email_type, status, subject, reason=result.error, attempts=result.attempts,
                                   recipients=list(result.recipients))

    def _internal_failure(self, email_type: EmailType, what: str, exc: BaseException) -> NotificationOutcome:
        self._report_error(f"building {what} e-mail failed", exc)
        reason = f"internal error: {type(exc).__name__}: {exc}"
        self._write_log(email_type, f"{email_type.value} (not built)", STATUS_FAILED, 0, reason, [], self.batch_id)
        return NotificationOutcome(email_type, STATUS_FAILED, f"{email_type.value}", reason=reason)

    def _report_error(self, message: str, exc: BaseException) -> None:
        try:
            record_error(None, "notifications", exc, {"message": message}, clock=self.clock)
        except Exception as inner:  # noqa: BLE001
            logger.error(f"notifications: {message}: {exc} (record_error failed: {inner})")

    def _write_log(
        self,
        email_type: EmailType,
        subject: str,
        status: str,
        attempts: int,
        error: str | None,
        recipients: Sequence[str],
        batch_id: int | None,
    ) -> None:
        """Persist one ``notifications_log`` row + JSONL event. Never raises."""
        now = self.clock.now()
        rcpt = ",".join(recipients)
        log_event(
            "notifications",
            {"email_type": email_type.value, "status": status, "subject": subject, "recipients": list(recipients),
             "attempts": attempts, "error": error, "batch_id": batch_id},
            clock=self.clock,
        )
        if self.session_factory is None:
            return
        try:
            with session_scope(self.session_factory) as session:
                session.add(
                    NotificationLog(
                        ts=now,
                        email_type=email_type.value,
                        recipients=rcpt,
                        subject=subject[:1000],
                        status=status,
                        attempts=int(attempts),
                        error=(error[:4000] if error else None),
                        batch_id=batch_id,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Could not write notifications_log row ({status} {subject!r}): {exc}")

    # ------------------------------------------------------------------ context builders

    def _approval_context(self, item: Any) -> dict[str, Any]:
        direction = _str_value(_get(item, "direction")) or Direction.LONG.value
        mode = _str_value(_get(item, "mode")) or "?"
        entry = _float(_get(item, "entry_price"))
        sl = _float(_get(item, "stop_loss"))
        t1 = _float(_get(item, "target_1"))
        t2 = _float(_get(item, "target_2"))
        qty = int(_get(item, "qty", 0) or 0)
        risk_ps = abs(entry - sl) if entry is not None and sl is not None else None
        risk_json = _get(item, "risk_json") or {}
        summary = _get(item, "summary_json") or {}
        proposal = _get(item, "proposal_json") or {}
        est_charges = _charges_total(risk_json.get("charges_estimate")) if isinstance(risk_json, Mapping) else None
        expires = _get(item, "expires_at")
        time_left = None
        if isinstance(expires, datetime):
            remaining = to_ist(expires) - self.clock.now()
            if remaining.total_seconds() > 0:
                time_left = _fmt_remaining(remaining)
        try:
            product = product_for(Mode(mode)).value
        except ValueError:
            product = "?"
        reasoning = None
        if isinstance(summary, Mapping):
            reasoning = summary.get("reasoning") or summary.get("meta_reasoning") or summary.get("summary")
            if reasoning is not None:
                reasoning = " ".join(str(reasoning).split())[:600]
        return {
            "id": _get(item, "id"),
            "decision_id": _get(item, "decision_id"),
            "symbol": _get(item, "symbol", "?"),
            "mode": mode,
            "product": product,
            "direction": direction,
            "side": side_label(direction),
            "entry_type": _str_value(_get(item, "entry_type")) or "LIMIT",
            "entry": entry,
            "stop_loss": sl,
            "target_1": t1,
            "target_2": t2,
            "qty": qty,
            "max_qty": int(_get(item, "max_qty", qty) or qty),
            "confidence": _get(item, "confidence"),
            "grade": _str_value(_get(item, "grade")),
            "strategy": _get(item, "strategy"),
            "regime": _str_value(proposal.get("regime")) if isinstance(proposal, Mapping) else None,
            "risk_per_share": risk_ps,
            "risk_amount": risk_ps * qty if risk_ps is not None else None,
            "sl_pct": (risk_ps / entry * 100.0) if risk_ps is not None and entry else None,
            "position_value": entry * qty if entry is not None else None,
            "rr_t1": reward_risk(entry, sl, t1),
            "rr_t2": reward_risk(entry, sl, t2),
            "est_charges": est_charges,
            "bull_points": extract_points(summary if isinstance(summary, Mapping) else None, "bull"),
            "bear_points": extract_points(summary if isinstance(summary, Mapping) else None, "bear"),
            "reasoning": reasoning,
            "created_at": _get(item, "created_at"),
            "expires_at": expires,
            "time_left": time_left,
        }

    def _entry_context(self, position: Any, order: Any) -> dict[str, Any]:
        direction = _str_value(_get(position, "direction")) or Direction.LONG.value
        qty = int(_first(order, "qty", default=None) or _get(position, "initial_qty") or _get(position, "qty", 0))
        fill = _float(_first(order, "fill_price", default=None)) or _float(_get(position, "avg_entry"))
        sl = _float(_get(position, "stop_loss"))
        t1 = _float(_get(position, "target_1"))
        product = _str_value(_get(position, "product")) or "?"
        risk_ps = abs(fill - sl) if fill is not None and sl is not None else None
        return {
            "position_id": _get(position, "id"),
            "order_id": _get(order, "id"),
            "symbol": _get(position, "symbol", "?"),
            "mode": _str_value(_get(position, "mode")) or "?",
            "product": product,
            "direction": direction,
            "side": side_label(direction),
            "qty": qty,
            "fill_price": fill,
            "order_type": _str_value(_get(order, "order_type")) or "MARKET",
            "limit_price": _float(_get(order, "limit_price")),
            "filled_at": _first(order, "filled_at", default=None) or _get(position, "opened_at"),
            "slippage": _float(_get(order, "slippage")),
            "stop_loss": sl,
            "target_1": t1,
            "target_2": _float(_get(position, "target_2")),
            "margin_blocked": _float(_get(position, "margin_blocked")),
            "entry_charges": _float(_get(position, "entry_charges")),
            "risk_amount": risk_ps * qty if risk_ps is not None else None,
            "rr_t1": reward_risk(fill, sl, t1),
            "strategy": _get(position, "strategy"),
            "grade": _str_value(_get(position, "grade")),
            "regime": _str_value(_get(position, "regime")),
            "trailing_rule": _str_value(_get(position, "trailing_rule")),
            "max_holding_days": _get(position, "max_holding_days"),
            "source": _str_value(_get(position, "source")) or "SYSTEM",
            "overnight": product == Product.CNC.value,
        }

    def _exit_context(self, trade: Any) -> dict[str, Any]:
        entry_time = _get(trade, "entry_time")
        exit_time = _get(trade, "exit_time")
        holding = None
        if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
            holding = to_ist(exit_time) - to_ist(entry_time)
        elif _get(trade, "holding_minutes") is not None:
            holding = timedelta(minutes=int(_get(trade, "holding_minutes")))
        entry_price = _float(_get(trade, "entry_price"))
        qty = int(_get(trade, "qty", 0) or 0)
        net = _float(_get(trade, "net_pnl"))
        reason = _str_value(_get(trade, "exit_reason")) or "?"
        cost = entry_price * qty if entry_price is not None else 0.0
        return {
            "trade_id": _get(trade, "id"),
            "position_id": _get(trade, "position_id"),
            "symbol": _get(trade, "symbol", "?"),
            "mode": _str_value(_get(trade, "mode")) or "?",
            "product": _str_value(_get(trade, "product")) or "?",
            "direction": _str_value(_get(trade, "direction")) or "?",
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": _float(_get(trade, "exit_price")),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "stop_loss": _float(_get(trade, "stop_loss")),
            "exit_reason": reason,
            "exit_reason_label": EXIT_REASON_LABELS.get(reason, reason),
            "is_partial": bool(_get(trade, "is_partial", False)),
            "holding": fmt_duration(holding) if holding is not None else "—",
            "holding_days": _get(trade, "holding_days"),
            "gross": _float(_get(trade, "gross_pnl")),
            "charges": _float(_get(trade, "charges")),
            "net": net,
            "return_pct": (net / cost * 100.0) if net is not None and cost > 0 else None,
            "r_multiple": _float(_get(trade, "r_multiple")),
            "charges_breakdown": charges_breakdown(_get(trade, "charges_json")),
            "strategy": _get(trade, "strategy"),
            "grade": _str_value(_get(trade, "grade")),
            "regime": _str_value(_get(trade, "regime")),
            "source": _str_value(_get(trade, "source")) or "SYSTEM",
        }

    def _candidate_context(self, cand: Any) -> dict[str, Any]:
        if hasattr(cand, "model_dump"):
            cand = cand.model_dump()
        direction = _str_value(_first(cand, "direction", default=None))
        decision = _str_value(_first(cand, "decision", "side", default=None))
        if direction is None:
            direction = Direction.SHORT.value if decision in ("SELL_SHORT", "SELL SHORT", "SHORT") else Direction.LONG.value
        entry = _float(_first(cand, "entry", "entry_price", "price"))
        sl = _float(_first(cand, "stop_loss", "sl"))
        t1 = _float(_first(cand, "target_1", "t1", "target"))
        notes = _first(cand, "notes", "reason", "reasoning", "summary")
        if notes is not None:
            notes = " ".join(str(notes).split())[:400]
        return {
            "symbol": str(_first(cand, "symbol", default="?")),
            "mode": _str_value(_first(cand, "mode", default=None)) or Mode.SWING.value,
            "direction": direction,
            "side": side_label(direction),
            "entry": entry,
            "stop_loss": sl,
            "target_1": t1,
            "target_2": _float(_first(cand, "target_2", "t2")),
            "sl_pct": (abs(entry - sl) / entry * 100.0) if entry and sl is not None else None,
            "rr": reward_risk(entry, sl, t1),
            "confidence": _first(cand, "confidence"),
            "grade": _str_value(_first(cand, "grade")),
            "strategy": _first(cand, "strategy"),
            "confluence": _float(_first(cand, "confluence", "confluence_score")),
            "notes": notes,
        }

    @staticmethod
    def _fmt_detail(value: Any) -> str:
        if isinstance(value, datetime):
            return fmt_dt(value, "%d %b %Y %H:%M:%S")
        if isinstance(value, bool) or value is None:
            return str(value)
        if isinstance(value, float):
            return fmt_num(value, 2)
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, Mapping):
            return json.dumps(to_jsonable(value), ensure_ascii=False)
        return str(getattr(value, "value", value))


# =========================================================================== module helpers


def _fmt_remaining(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    if minutes >= 24 * 60:
        return f"{minutes // (24 * 60)}d {(minutes % (24 * 60)) // 60}h"
    if minutes >= 60:
        return f"{minutes // 60}h {minutes % 60:02d}m"
    return f"{max(minutes, 0)} min"


_EOD_TOTALS_DEFAULT: dict[str, Any] = {"trades": 0, "legs": 0, "wins": 0, "losses": 0, "gross": 0.0,
                                       "charges": 0.0, "net": 0.0, "win_rate": None}
_EOD_CAPITAL_DEFAULT: dict[str, Any] = {
    "available": False, "starting_capital": None, "base_capital": None, "trading_cash": None, "blocked": None,
    "trading_balance": None, "reserve": None, "reserve_peak": None, "unrealized_pnl": 0.0, "total_equity": None,
    "hwm": None, "drawdown_pct": None, "tier": "UNKNOWN", "sizing_capital": None, "mode_sizing": {},
    "realized_today": 0.0, "dividends_today": 0.0, "profit_locked": 0.0, "loss_absorbed": 0.0, "refilled": 0.0,
    "manual_transfers": 0.0, "manual_adjustments": 0.0, "sizing_change": 0.0, "halted": False,
    "halt_reason": None, "eod_processed": False, "source": "none",
}
_EOD_APPROVALS_DEFAULT: dict[str, Any] = {"total": 0, "approved": 0, "modified": 0, "rejected": 0, "timeout": 0,
                                          "pending": 0, "items": []}
_EOD_SHADOW_DEFAULT: dict[str, Any] = {"total": 0, "open": 0, "closed": 0, "target_hits": 0, "sl_hits": 0,
                                       "hypothetical_net": 0.0, "by_source": [], "cumulative_by_source": [],
                                       "items": []}


def normalize_eod_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fill every key the EOD template needs (tolerates partial/hand-built report dicts)."""
    r: dict[str, Any] = dict(report or {})
    r.setdefault("batch_id", None)
    r.setdefault("batch_name", None)
    r.setdefault("day", None)
    r["totals"] = {**_EOD_TOTALS_DEFAULT, **(r.get("totals") or {})}
    r["capital"] = {**_EOD_CAPITAL_DEFAULT, **(r.get("capital") or {})}
    r["approvals"] = {**_EOD_APPROVALS_DEFAULT, **(r.get("approvals") or {})}
    r["shadow"] = {**_EOD_SHADOW_DEFAULT, **(r.get("shadow") or {})}
    failures = r.get("agent_failures") or {}
    r["agent_failures"] = {"total": failures.get("total", 0), "by_agent": list(failures.get("by_agent") or [])}
    for key in ("modes", "trades", "open_positions", "not_traded", "equity_series"):
        r[key] = list(r.get(key) or [])
    for row in r["modes"]:
        row.setdefault("win_rate", None)
        for k in ("trades", "wins", "losses"):
            row.setdefault(k, 0)
        for k in ("gross", "charges", "net"):
            row.setdefault(k, 0.0)
    for t in r["trades"]:
        t.setdefault("is_partial", False)
        t.setdefault("exit_reason", "?")
        t.setdefault("net", None)
    for p in r["open_positions"]:
        for k in ("mode", "product"):
            p.setdefault(k, "?")
        p.setdefault("price_missing", p.get("last_price") is None)
        for k in ("qty", "avg_entry", "last_price", "unrealized_net", "unrealized_pct", "days_held"):
            p.setdefault(k, None)
    for s in r["not_traded"]:
        s.setdefault("reason", "—")
        s.setdefault("mode", "?")
    for row in r["shadow"]["by_source"]:
        for k in ("count", "target_hits", "sl_hits"):
            row.setdefault(k, 0)
        row.setdefault("hypothetical_net", 0.0)
    for row in r["shadow"]["cumulative_by_source"]:
        for k in ("closed", "target_hits", "sl_hits"):
            row.setdefault(k, 0)
        row.setdefault("hypothetical_net", 0.0)
    for f in r["agent_failures"]["by_agent"]:
        f.setdefault("count", 0)
        f["symbols"] = list(f.get("symbols") or [])
        f.setdefault("last_error", None)
    r.setdefault("open_unrealized_total", 0.0)
    r.setdefault("not_traded_total", len(r["not_traded"]))
    r.setdefault("empty", False)
    if r.get("next_day_plan"):   # optional section (scheduler.jobs.job_eod_report); absent = not shown
        plan_rows = []
        for p in r["next_day_plan"]:
            row = dict(p or {})
            for k in ("symbol", "setup", "direction", "trigger", "stop", "target", "rr", "score", "status"):
                row.setdefault(k, None)
            row["reasons"] = list(row.get("reasons") or [])
            row["affordable"] = bool(row.get("affordable", True))
            plan_rows.append(row)
        r["next_day_plan"] = plan_rows
        r.setdefault("next_day_plan_for", None)
        r.setdefault("next_day_plan_name", None)
        r.setdefault("next_day_plan_check", "09:20")
    return r


def fallback_html(title: str, context: Mapping[str, Any]) -> str:
    """Minimal HTML body used when a template fails to render (content as escaped JSON)."""
    data = {k: v for k, v in context.items() if k not in ("accent",)}
    try:
        body = json.dumps(to_jsonable(data), indent=2, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        body = f"(context not serialisable: {exc})"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'></head>"
        "<body style='font-family:Arial,sans-serif;margin:0;padding:12px;'>"
        "<div style='background:#8a4b00;color:#fff;padding:6px 10px;font-weight:700;font-size:12px;'>"
        "PAPER TRADING &middot; simulated &middot; no real orders</div>"
        f"<h1 style='font-size:18px;'>{html_lib.escape(str(title))}</h1>"
        "<p style='font-size:12px;color:#52514e;'>The formatted template failed to render; raw data follows.</p>"
        f"<pre style='white-space:pre-wrap;font-size:12px;'>{html_lib.escape(body)}</pre>"
        "</body></html>"
    )
