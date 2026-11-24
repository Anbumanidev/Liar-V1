"""EmailSender: SMTP mocked; retries, never-raise, skip on missing config, MIME structure."""

from __future__ import annotations

import smtplib
from email import message_from_bytes
from email.message import EmailMessage
from typing import Any

import pytest
from pydantic import SecretStr

from config.schema import NotificationSettings
from config.secrets import Secrets
from notifications.email import (
    STATUS_FAILED,
    STATUS_SENT,
    STATUS_SKIPPED,
    Attachment,
    EmailSender,
    html_to_text,
    parse_recipients,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class FakeSMTP:
    """Stand-in for ``smtplib.SMTP``: records calls, fails the first ``fail_times`` attempts."""

    fail_times: int = 0
    fail_exc: BaseException | None = None
    fail_on_connect: bool = False
    instances: list["FakeSMTP"] = []
    sent: list[EmailMessage] = []
    attempts: int = 0

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        FakeSMTP.attempts += 1
        self.host, self.port, self.timeout = host, port, timeout
        self.calls: list[tuple[str, Any]] = []
        FakeSMTP.instances.append(self)
        if FakeSMTP.fail_on_connect and FakeSMTP.attempts <= FakeSMTP.fail_times:
            raise OSError("connection refused")

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.calls.append(("quit", None))

    def ehlo(self) -> None:
        self.calls.append(("ehlo", None))

    def starttls(self, context: Any = None) -> None:
        self.calls.append(("starttls", context))

    def login(self, user: str, password: str) -> None:
        self.calls.append(("login", (user, password)))
        if FakeSMTP.attempts <= FakeSMTP.fail_times and not FakeSMTP.fail_on_connect:
            raise FakeSMTP.fail_exc or smtplib.SMTPServerDisconnected("temporary failure")

    def send_message(self, msg: EmailMessage, from_addr: str | None = None, to_addrs: Any = None) -> dict:
        self.calls.append(("send_message", (from_addr, list(to_addrs or []))))
        FakeSMTP.sent.append(msg)
        return {}


@pytest.fixture(autouse=True)
def fake_smtp(monkeypatch: pytest.MonkeyPatch) -> type[FakeSMTP]:
    FakeSMTP.fail_times = 0
    FakeSMTP.fail_exc = None
    FakeSMTP.fail_on_connect = False
    FakeSMTP.instances = []
    FakeSMTP.sent = []
    FakeSMTP.attempts = 0
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


def _secrets(user: str | None = "bot@gmail.com", password: str | None = "abcd efgh ijkl mnop",
             to: str | None = "owner@example.com") -> Secrets:
    return Secrets.model_construct(
        smtp_user=user,
        smtp_app_password=SecretStr(password) if password is not None else None,
        alert_email_to=to,
    )


def _cfg(**kw: Any) -> NotificationSettings:
    base: dict[str, Any] = {"retries": 3, "retry_delay_seconds": 2.0}
    base.update(kw)
    return NotificationSettings(**base)


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def test_send_success_first_attempt_uses_starttls_and_app_password() -> None:
    sleep = SleepRecorder()
    sender = EmailSender(_secrets(), _cfg(), sleep=sleep)
    res = sender.send("Hello", "<p>Hi <b>there</b></p>")
    assert res.ok and res.status == STATUS_SENT and res.attempts == 1 and res.error is None
    assert res.recipients == ["owner@example.com"]
    assert res.subject == "[PAPER] Hello"
    smtp = FakeSMTP.instances[0]
    assert (smtp.host, smtp.port) == ("smtp.gmail.com", 587)
    names = [c[0] for c in smtp.calls]
    assert names.index("starttls") < names.index("login") < names.index("send_message")
    assert ("login", ("bot@gmail.com", "abcdefghijklmnop")) in smtp.calls  # spaces stripped
    assert sleep.calls == []


def test_message_is_multipart_alternative_with_text_fallback() -> None:
    sender = EmailSender(_secrets(), _cfg())
    sender.send("Struct", "<html><body><h1>Title</h1><p>Body &amp; more</p></body></html>")
    msg = FakeSMTP.sent[0]
    raw = message_from_bytes(msg.as_bytes())
    assert raw.get_content_type() == "multipart/alternative"
    parts = [p.get_content_type() for p in raw.walk()]
    assert "text/plain" in parts and "text/html" in parts
    plain = next(p for p in raw.walk() if p.get_content_type() == "text/plain").get_payload(decode=True).decode()
    assert "Title" in plain and "Body & more" in plain and "<p>" not in plain
    assert raw["Subject"] == "[PAPER] Struct"
    assert raw["To"] == "owner@example.com"
    assert raw["X-Paper-Trading"] == "true"


def test_inline_png_is_related_to_html_with_content_id() -> None:
    sender = EmailSender(_secrets(), _cfg())
    res = sender.send("Chart", '<img src="cid:equity_chart">',
                      attachments=[Attachment.inline_png("equity_chart.png", PNG, "equity_chart")])
    assert res.ok
    raw = message_from_bytes(FakeSMTP.sent[0].as_bytes())
    types = [p.get_content_type() for p in raw.walk()]
    assert "multipart/related" in types and "image/png" in types
    img = next(p for p in raw.walk() if p.get_content_type() == "image/png")
    assert img["Content-ID"] == "<equity_chart>"
    assert img.get_payload(decode=True) == PNG


def test_regular_attachment_makes_mixed_message() -> None:
    sender = EmailSender(_secrets(), _cfg())
    sender.send("Att", "<p>x</p>", attachments=[Attachment("report.png", PNG, inline=False)])
    raw = message_from_bytes(FakeSMTP.sent[0].as_bytes())
    assert raw.get_content_type() == "multipart/mixed"
    att = next(p for p in raw.walk() if p.get_content_type() == "image/png")
    assert att.get_filename() == "report.png"


def test_retries_then_succeeds() -> None:
    FakeSMTP.fail_times = 2
    sleep = SleepRecorder()
    sender = EmailSender(_secrets(), _cfg(retries=3, retry_delay_seconds=1.5), sleep=sleep)
    res = sender.send("Retry", "<p>x</p>")
    assert res.ok and res.attempts == 3 and res.status == STATUS_SENT
    assert sleep.calls == [1.5, 1.5]
    assert len(FakeSMTP.sent) == 1


def test_all_attempts_fail_never_raises() -> None:
    FakeSMTP.fail_times = 100
    sleep = SleepRecorder()
    sender = EmailSender(_secrets(), _cfg(retries=3, retry_delay_seconds=0.5), sleep=sleep)
    res = sender.send("Down", "<p>x</p>")
    assert not res.ok and res.status == STATUS_FAILED
    assert res.attempts == 4  # 1 + 3 retries
    assert "SMTPServerDisconnected" in (res.error or "")
    assert len(sleep.calls) == 3
    assert FakeSMTP.sent == []


def test_connection_errors_are_retried() -> None:
    FakeSMTP.fail_times = 1
    FakeSMTP.fail_on_connect = True
    sender = EmailSender(_secrets(), _cfg(retries=2, retry_delay_seconds=0), sleep=SleepRecorder())
    res = sender.send("Conn", "<p>x</p>")
    assert res.ok and res.attempts == 2


def test_zero_retries_means_single_attempt() -> None:
    FakeSMTP.fail_times = 5
    sender = EmailSender(_secrets(), _cfg(retries=0), sleep=SleepRecorder())
    res = sender.send("Once", "<p>x</p>")
    assert res.attempts == 1 and res.status == STATUS_FAILED


def test_authentication_error_is_not_retried() -> None:
    FakeSMTP.fail_times = 100
    FakeSMTP.fail_exc = smtplib.SMTPAuthenticationError(535, b"bad credentials")
    sleep = SleepRecorder()
    sender = EmailSender(_secrets(), _cfg(), sleep=sleep)
    res = sender.send("Auth", "<p>x</p>")
    assert res.status == STATUS_FAILED and res.attempts == 1
    assert "SMTPAuthenticationError" in (res.error or "")
    assert sleep.calls == []


@pytest.mark.parametrize(
    "secrets",
    [_secrets(user=None), _secrets(password=None), _secrets(user="  "), _secrets(password="   ")],
)
def test_missing_credentials_skip_without_smtp(secrets: Secrets) -> None:
    res = EmailSender(secrets, _cfg()).send("NoCreds", "<p>x</p>")
    assert res.status == STATUS_SKIPPED and not res.ok and res.attempts == 0
    assert "credentials" in (res.error or "")
    assert FakeSMTP.instances == []


def test_no_secrets_object_skips() -> None:
    res = EmailSender(None, _cfg()).send("None", "<p>x</p>")
    assert res.status == STATUS_SKIPPED


def test_missing_recipients_skip() -> None:
    res = EmailSender(_secrets(to=None), _cfg()).send("NoTo", "<p>x</p>")
    assert res.status == STATUS_SKIPPED and "recipients" in (res.error or "")
    assert FakeSMTP.instances == []


def test_recipient_resolution_order() -> None:
    sender = EmailSender(_secrets(to="a@x.com, b@x.com"), _cfg())
    assert sender.resolve_recipients() == ["a@x.com", "b@x.com"]
    sender_cfg = EmailSender(_secrets(to="a@x.com"), _cfg(recipients=["c@y.com"]))
    assert sender_cfg.resolve_recipients() == ["c@y.com"]
    assert sender_cfg.resolve_recipients(["d@z.com"]) == ["d@z.com"]
    res = sender_cfg.send("Override", "<p>x</p>", recipients="e@z.com;f@z.com")
    assert res.recipients == ["e@z.com", "f@z.com"]
    assert FakeSMTP.instances[0].calls[-2][1][1] == ["e@z.com", "f@z.com"]


def test_subject_prefix_added_once() -> None:
    sender = EmailSender(_secrets(), _cfg(subject_prefix="[PAPER]"))
    assert sender.format_subject("EOD   report") == "[PAPER] EOD report"
    assert sender.format_subject("[PAPER] already") == "[PAPER] already"
    assert EmailSender(_secrets(), _cfg(subject_prefix="")).format_subject("x") == "x"


def test_build_failure_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = EmailSender(_secrets(), _cfg())

    def boom(*a: Any, **k: Any) -> EmailMessage:
        raise ValueError("bad mime")

    monkeypatch.setattr(sender, "build_message", boom)
    res = sender.send("Build", "<p>x</p>")
    assert res.status == STATUS_FAILED and "bad mime" in (res.error or "")


def test_parse_recipients_filters_invalid_and_duplicates() -> None:
    assert parse_recipients("a@x.com, A@x.com; bad-address, ,b@y.org") == ["a@x.com", "b@y.org"]
    assert parse_recipients(None) == []
    assert parse_recipients(["a@x.com", "c@z.in"]) == ["a@x.com", "c@z.in"]


def test_html_to_text_skips_style_and_keeps_rows() -> None:
    text = html_to_text("<style>.a{color:red}</style><table><tr><td>Net</td><td>+12</td></tr></table><p>End</p>")
    assert "color" not in text
    assert "Net" in text and "+12" in text and "End" in text
