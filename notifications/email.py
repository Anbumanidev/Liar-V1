"""SMTP e-mail sender (section 13).

Gmail with an App Password over STARTTLS::

    sender = EmailSender(secrets, settings.notifications)
    result = sender.send("EOD report", html, attachments=[Attachment.inline_png("equity.png", png)])

Guarantees:

* ``send`` NEVER raises. Every failure is logged and reported through ``SendResult``.
* Missing credentials or recipients -> ``status="SKIPPED"`` (logged), not an exception.
* Transient failures are retried ``cfg.retries`` times (so ``retries + 1`` attempts in
  total) with ``cfg.retry_delay_seconds`` between attempts. Permanent failures
  (authentication refused, every recipient refused) are not retried.
* Messages are ``multipart/alternative`` (plain text + HTML). Inline images are
  ``multipart/related`` with the HTML part and referenced as ``cid:<content_id>``.

The retry delay is a real wall-clock pause between network attempts (not trading
time), so it uses an injectable ``sleep`` callable rather than the trading ``Clock``.
"""

from __future__ import annotations

import re
import smtplib
import ssl
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from html.parser import HTMLParser

from loguru import logger

from config.schema import NotificationSettings
from config.secrets import Secrets

#: Delivery status values written to ``notifications_log.status``.
STATUS_SENT = "SENT"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"

_EMAIL_RE = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")


@dataclass(frozen=True)
class Attachment:
    """A binary MIME part (e.g. the equity-chart PNG).

    Attributes:
        filename: file name shown to the recipient.
        content: raw bytes.
        mimetype: ``maintype/subtype`` (default ``image/png``).
        inline: True = embed in the HTML (``multipart/related``) and reference it with
            ``<img src="cid:{content_id}">``; False = regular attachment.
        content_id: the ``cid`` used by the HTML (without angle brackets). Required for
            inline parts; defaults to the filename stem.
    """

    filename: str
    content: bytes
    mimetype: str = "image/png"
    inline: bool = True
    content_id: str | None = None

    @property
    def cid(self) -> str:
        """Content-ID without angle brackets."""
        if self.content_id:
            return self.content_id
        return re.sub(r"[^A-Za-z0-9_.-]", "_", self.filename.rsplit(".", 1)[0]) or "part"

    @classmethod
    def inline_png(cls, filename: str, content: bytes, content_id: str | None = None) -> "Attachment":
        """Convenience constructor for an inline PNG image."""
        return cls(filename=filename, content=content, mimetype="image/png", inline=True, content_id=content_id)


@dataclass
class SendResult:
    """Outcome of one ``EmailSender.send`` call.

    Attributes:
        ok: True when the SMTP server accepted the message.
        attempts: number of SMTP attempts made (0 when skipped).
        error: last error message (None on success).
        status: ``SENT`` | ``FAILED`` | ``SKIPPED``.
        recipients: resolved recipient list.
        subject: the final subject line (with prefix).
    """

    ok: bool
    attempts: int
    error: str | None = None
    status: str = STATUS_FAILED
    recipients: list[str] = field(default_factory=list)
    subject: str = ""


class _TextExtractor(HTMLParser):
    """Very small HTML -> plain-text converter for the text/plain fallback part."""

    _BLOCK_TAGS = {"p", "div", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li", "table", "section", "hr"}
    _SKIP_TAGS = {"style", "script", "head", "title"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")
        elif tag in ("td", "th"):
            self._parts.append("  ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in raw.split("\n")]
        out: list[str] = []
        for ln in lines:
            if ln or (out and out[-1]):
                out.append(ln)
        return "\n".join(out).strip() + "\n"


def html_to_text(html: str) -> str:
    """Derive a readable plain-text version of an HTML e-mail body."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - fallback text must never break sending
        logger.warning(f"html_to_text failed, using tag-stripped text: {exc}")
        return re.sub(r"<[^>]+>", " ", html)
    return parser.text()


def parse_recipients(value: str | Sequence[str] | None) -> list[str]:
    """Split a comma/semicolon separated address string (or list) into unique valid addresses."""
    if value is None:
        return []
    items: list[str] = []
    raw = [value] if isinstance(value, str) else list(value)
    for chunk in raw:
        items.extend(p.strip() for p in re.split(r"[,;]", str(chunk)))
    out: list[str] = []
    for addr in items:
        if not addr:
            continue
        if not _EMAIL_RE.match(addr):
            logger.warning(f"Ignoring invalid e-mail recipient {addr!r}")
            continue
        if addr.lower() not in {a.lower() for a in out}:
            out.append(addr)
    return out


class EmailSender:
    """Sends HTML e-mails through SMTP + STARTTLS (Gmail App Password).

    Args:
        secrets: ``Secrets`` with ``smtp_user``, ``smtp_app_password``, ``alert_email_to``.
        cfg: ``NotificationSettings`` (host, port, retries, delay, recipients, subject prefix).
        sleep: pause function between retries (injectable for tests).
        timeout: socket timeout in seconds for each SMTP connection.
    """

    def __init__(
        self,
        secrets: Secrets | None,
        cfg: NotificationSettings,
        *,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 30.0,
    ) -> None:
        self.secrets = secrets
        self.cfg = cfg
        self._sleep = sleep
        self.timeout = timeout

    # ------------------------------------------------------------------ configuration

    @property
    def smtp_user(self) -> str | None:
        """SMTP login (also used as the From address)."""
        user = self.secrets.smtp_user if self.secrets is not None else None
        return user.strip() if user and user.strip() else None

    def _password(self) -> str | None:
        if self.secrets is None or self.secrets.smtp_app_password is None:
            return None
        value = self.secrets.smtp_app_password.get_secret_value()
        # Gmail shows App Passwords as 4 groups of 4 characters; spaces are not part of it.
        value = value.replace(" ", "").strip()
        return value or None

    @property
    def credentials_configured(self) -> bool:
        """True when both SMTP user and App Password are present."""
        return self.smtp_user is not None and self._password() is not None

    def resolve_recipients(self, override: Sequence[str] | str | None = None) -> list[str]:
        """Recipients: explicit override, else ``cfg.recipients``, else ``secrets.alert_email_to``."""
        if override:
            found = parse_recipients(override)
            if found:
                return found
        found = parse_recipients(self.cfg.recipients)
        if found:
            return found
        if self.secrets is not None:
            return parse_recipients(self.secrets.alert_email_to)
        return []

    def format_subject(self, subject: str) -> str:
        """Prefix the subject with ``cfg.subject_prefix`` (e.g. ``[PAPER]``) exactly once."""
        subject = " ".join(subject.split())
        prefix = (self.cfg.subject_prefix or "").strip()
        if prefix and not subject.startswith(prefix):
            return f"{prefix} {subject}"
        return subject

    # ------------------------------------------------------------------ message building

    def build_message(
        self,
        subject: str,
        html: str,
        recipients: list[str],
        attachments: Sequence[Attachment] | None = None,
        text: str | None = None,
    ) -> EmailMessage:
        """Build the MIME message (``multipart/alternative`` text + HTML, optional images)."""
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.smtp_user or "liar-v1@localhost"
        msg["To"] = ", ".join(recipients)
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="liar-v1.paper")
        msg["X-Paper-Trading"] = "true"
        msg.set_content(text if text is not None else html_to_text(html))
        msg.add_alternative(html, subtype="html")

        attachments = list(attachments or [])
        inline = [a for a in attachments if a.inline]
        regular = [a for a in attachments if not a.inline]
        if inline:
            html_part = msg.get_payload()[1]
            for att in inline:
                maintype, _, subtype = att.mimetype.partition("/")
                html_part.add_related(
                    att.content,
                    maintype=maintype or "application",
                    subtype=subtype or "octet-stream",
                    cid=f"<{att.cid}>",
                    filename=att.filename,
                    disposition="inline",
                )
        for att in regular:
            maintype, _, subtype = att.mimetype.partition("/")
            msg.add_attachment(
                att.content,
                maintype=maintype or "application",
                subtype=subtype or "octet-stream",
                filename=att.filename,
            )
        return msg

    # ------------------------------------------------------------------ sending

    def _deliver(self, msg: EmailMessage, recipients: list[str]) -> None:
        """One SMTP attempt. Raises on failure (caught by ``send``)."""
        password = self._password()
        user = self.smtp_user
        if user is None or password is None:  # pragma: no cover - guarded by send()
            raise RuntimeError("SMTP credentials missing")
        context = ssl.create_default_context()
        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=self.timeout) as smtp:
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
            smtp.login(user, password)
            refused = smtp.send_message(msg, from_addr=user, to_addrs=recipients)
            if refused:
                logger.warning(f"SMTP server refused some recipients: {sorted(refused)}")

    @staticmethod
    def _is_permanent(exc: BaseException) -> bool:
        """Errors that will not succeed on retry (bad credentials, all recipients refused)."""
        return isinstance(exc, (smtplib.SMTPAuthenticationError, smtplib.SMTPRecipientsRefused))

    def send(
        self,
        subject: str,
        html: str,
        recipients: Sequence[str] | str | None = None,
        attachments: Sequence[Attachment] | None = None,
        text: str | None = None,
    ) -> SendResult:
        """Send one e-mail. Never raises.

        Args:
            subject: subject without prefix (the configured prefix is added).
            html: HTML body.
            recipients: override recipients (default: configured recipients).
            attachments: optional inline/attached binary parts.
            text: plain-text alternative (derived from the HTML when omitted).

        Returns:
            ``SendResult`` (``SKIPPED`` when credentials/recipients are missing).
        """
        try:
            final_subject = self.format_subject(subject)
            to = self.resolve_recipients(recipients)
        except Exception as exc:  # noqa: BLE001 - never raise from send
            logger.error(f"E-mail preparation failed: {exc}")
            return SendResult(ok=False, attempts=0, error=f"prepare failed: {exc}", status=STATUS_FAILED,
                              subject=subject)

        if not self.credentials_configured:
            reason = "SMTP credentials not configured (SMTP_USER / SMTP_APP_PASSWORD in .env)"
            logger.warning(f"E-mail skipped ({final_subject!r}): {reason}")
            return SendResult(ok=False, attempts=0, error=reason, status=STATUS_SKIPPED, recipients=to,
                              subject=final_subject)
        if not to:
            reason = "no recipients configured (notifications.recipients / ALERT_EMAIL_TO)"
            logger.warning(f"E-mail skipped ({final_subject!r}): {reason}")
            return SendResult(ok=False, attempts=0, error=reason, status=STATUS_SKIPPED, recipients=to,
                              subject=final_subject)

        try:
            msg = self.build_message(final_subject, html, to, attachments=attachments, text=text)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"E-mail build failed ({final_subject!r}): {exc}")
            return SendResult(ok=False, attempts=0, error=f"build failed: {exc}", status=STATUS_FAILED,
                              recipients=to, subject=final_subject)

        max_attempts = max(1, int(self.cfg.retries) + 1)
        delay = max(0.0, float(self.cfg.retry_delay_seconds))
        last_error: str | None = None
        attempts = 0
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                self._deliver(msg, to)
            except Exception as exc:  # noqa: BLE001 - network failures are expected; log + retry
                last_error = f"{type(exc).__name__}: {exc}"
                permanent = self._is_permanent(exc)
                logger.warning(
                    f"E-mail attempt {attempt}/{max_attempts} failed ({final_subject!r}): {last_error}"
                    + (" [permanent, not retrying]" if permanent else "")
                )
                if permanent or attempt >= max_attempts:
                    break
                if delay > 0:
                    try:
                        self._sleep(delay)
                    except Exception as sleep_exc:  # noqa: BLE001
                        logger.warning(f"Retry sleep interrupted: {sleep_exc}")
                continue
            logger.info(f"E-mail sent ({final_subject!r}) to {len(to)} recipient(s) after {attempt} attempt(s)")
            return SendResult(ok=True, attempts=attempt, error=None, status=STATUS_SENT, recipients=to,
                              subject=final_subject)

        logger.error(f"E-mail FAILED after {attempts} attempt(s) ({final_subject!r}): {last_error}")
        return SendResult(ok=False, attempts=attempts, error=last_error, status=STATUS_FAILED, recipients=to,
                          subject=final_subject)
