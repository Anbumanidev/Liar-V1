"""Tests for dashboard.auth (lockout, idle timeout, config IO) and scripts/set_password.py."""

from __future__ import annotations

import importlib.util
import io
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.clock import IST
from dashboard import auth
from storage.models import AuditLog, LoginAttempt

NOW = datetime(2026, 11, 24, 10, 0, tzinfo=IST)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def auth_cfg() -> auth.AuthConfig:
    h = auth.hash_password("correct horse battery", rounds=4)
    return auth.AuthConfig(users={"admin": auth.AuthUser("admin", "Owner", h)}, cookie_name="c", cookie_key="k",
                           cookie_expiry_days=1)


def _login(session: Session, cfg: auth.AuthConfig, pw: str, at: datetime, user: str = "admin") -> auth.LoginResult:
    return auth.authenticate(session, cfg, user, pw, at, client="test")


def test_dashboard_hash_and_verify() -> None:
    h = auth.hash_password("s3cret-password", rounds=4)
    assert h.startswith("$2") and "s3cret" not in h
    assert auth.verify_password("s3cret-password", h)
    assert not auth.verify_password("wrong", h)
    assert not auth.verify_password("x", "not-a-hash")
    with pytest.raises(ValueError):
        auth.hash_password("")


def test_dashboard_auth_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "secrets" / "dashboard_auth.yaml"
    assert auth.load_auth_config(path) is None
    doc = auth.build_auth_document("Admin", "Owner", auth.hash_password("pw-1234567", rounds=4))
    auth.write_auth_config(path, doc)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # streamlit-authenticator compatible layout
    assert set(raw) == {"credentials", "cookie"}
    assert raw["credentials"]["usernames"]["admin"]["password"].startswith("$2")
    assert len(raw["cookie"]["key"]) == 64 and raw["cookie"]["expiry_days"] == 1
    cfg = auth.load_auth_config(path)
    assert cfg is not None and cfg.get_user("ADMIN ").name == "Owner"
    path.write_text("credentials: {usernames: {bob: {password: plaintext}}}", encoding="utf-8")
    assert auth.load_auth_config(path) is None  # no bcrypt hash -> rejected
    path.write_text(": : not yaml : [", encoding="utf-8")
    assert auth.load_auth_config(path) is None
    with pytest.raises(ValueError):
        auth.build_auth_document("admin", "x", "plaintext")


def test_dashboard_login_success_and_logging(session: Session, auth_cfg: auth.AuthConfig) -> None:
    res = _login(session, auth_cfg, "correct horse battery", NOW, user=" Admin ")
    assert res.ok and res.username == "admin" and res.reason == auth.REASON_OK
    rows = session.execute(select(LoginAttempt)).scalars().all()
    assert len(rows) == 1 and rows[0].success and rows[0].client == "test"
    assert session.execute(select(AuditLog.action)).scalars().all() == ["LOGIN_SUCCESS"]


def test_dashboard_lockout_after_five_failures(session: Session, auth_cfg: auth.AuthConfig,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    for i in range(4):
        res = _login(session, auth_cfg, "wrong", NOW + timedelta(seconds=i))
        assert not res.ok and res.locked_until is None and res.remaining_attempts == 4 - i
    fifth = _login(session, auth_cfg, "wrong", NOW + timedelta(seconds=4))
    assert not fifth.ok and fifth.locked_until == NOW + timedelta(minutes=15)

    # While locked, even the correct password is refused and bcrypt is NOT evaluated.
    calls: list[str] = []
    real_verify = auth.verify_password
    monkeypatch.setattr(auth, "verify_password", lambda pw, h: calls.append(pw) or real_verify(pw, h))
    locked = _login(session, auth_cfg, "correct horse battery", NOW + timedelta(minutes=5))
    assert not locked.ok and locked.reason == auth.REASON_LOCKED and calls == []
    assert locked.locked_until == NOW + timedelta(minutes=15)

    # Attempts during the lockout do not extend it; after 15 minutes the window has slid.
    after = _login(session, auth_cfg, "correct horse battery", NOW + timedelta(minutes=15, seconds=1))
    assert after.ok
    reasons = [r.reason for r in session.execute(select(LoginAttempt).order_by(LoginAttempt.id)).scalars()]
    assert reasons == ["BAD_PASSWORD"] * 5 + ["LOCKED_OUT", "OK"]
    actions = session.execute(select(AuditLog.action)).scalars().all()
    assert actions.count("LOGIN_FAILED") == 5 and actions.count("LOGIN_LOCKED") == 1


def test_dashboard_success_resets_failure_count(session: Session, auth_cfg: auth.AuthConfig) -> None:
    for i in range(4):
        _login(session, auth_cfg, "wrong", NOW + timedelta(seconds=i))
    assert _login(session, auth_cfg, "correct horse battery", NOW + timedelta(seconds=10)).ok
    res = _login(session, auth_cfg, "wrong", NOW + timedelta(seconds=20))
    assert res.locked_until is None and res.remaining_attempts == 4


def test_dashboard_unknown_user_counts_and_is_per_user(session: Session, auth_cfg: auth.AuthConfig) -> None:
    for i in range(5):
        res = _login(session, auth_cfg, "pw", NOW + timedelta(seconds=i), user="mallory")
        assert res.reason == auth.REASON_UNKNOWN_USER
    assert auth.lockout_until(session, "mallory", NOW + timedelta(seconds=10)) is not None
    assert auth.lockout_until(session, "admin", NOW + timedelta(seconds=10)) is None
    assert _login(session, auth_cfg, "correct horse battery", NOW + timedelta(seconds=11)).ok


def test_dashboard_login_edge_cases(session: Session, auth_cfg: auth.AuthConfig) -> None:
    assert auth.authenticate(session, None, "admin", "x", NOW).reason == auth.REASON_NO_CONFIG
    assert _login(session, auth_cfg, "", NOW).reason == auth.REASON_EMPTY
    assert _login(session, auth_cfg, "x", NOW, user="  ").reason == auth.REASON_EMPTY
    # empty / misconfiguration attempts do not count towards a lockout
    assert auth.recent_failures(session, "admin", NOW) == []


def test_dashboard_idle_timeout_logic() -> None:
    assert auth.is_idle_expired(None, NOW)
    assert not auth.is_idle_expired(NOW - timedelta(minutes=29, seconds=59), NOW)
    assert auth.is_idle_expired(NOW - timedelta(minutes=30), NOW)
    assert auth.is_idle_expired(NOW - timedelta(hours=2), NOW)
    assert not auth.is_idle_expired(NOW - timedelta(minutes=5), NOW, timeout=timedelta(minutes=10))
    # auto-refresh counter changed -> not user activity; unchanged or absent -> activity
    assert auth.is_user_interaction(None, None)
    assert auth.is_user_interaction(3, 3)
    assert not auth.is_user_interaction(4, 3)


# --------------------------------------------------------------------------- scripts/set_password.py


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("liar_set_password", ROOT / "scripts" / "set_password.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_set_password_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    out = tmp_path / "secrets" / "auth.yaml"
    secret = "a-very-long-Pass#1"
    monkeypatch.setattr(sys, "stdin", io.StringIO(secret + "\n"))
    assert script.main(["--username", "Owner", "--password-stdin", "--auth-file", str(out)]) == 0
    printed = capsys.readouterr()
    assert secret not in printed.out + printed.err
    cfg = auth.load_auth_config(out)
    assert cfg is not None and list(cfg.users) == ["owner"]
    assert auth.verify_password(secret, cfg.users["owner"].password_hash)
    assert secret not in out.read_text(encoding="utf-8")
    key1 = cfg.cookie_key

    # change password: replaces the single admin user, new cookie key
    monkeypatch.setattr(sys, "stdin", io.StringIO("another-Long-pass-2\n"))
    assert script.main(["--username", "admin", "--password-stdin", "--auth-file", str(out)]) == 0
    cfg2 = auth.load_auth_config(out)
    assert list(cfg2.users) == ["admin"] and cfg2.cookie_key != key1

    # weak password -> nothing written
    monkeypatch.setattr(sys, "stdin", io.StringIO("short\n"))
    assert script.main(["--password-stdin", "--auth-file", str(tmp_path / "x.yaml")]) == 2
    assert not (tmp_path / "x.yaml").exists()
    assert script.validate_password("aaaaaaaaaaaa") is not None
    assert script.validate_password(" padded-password ") is not None
