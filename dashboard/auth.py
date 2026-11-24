"""Owner-only authentication for the dashboard (section 14.1).

Design choice (documented): the secrets file is **compatible with streamlit-authenticator**
(``credentials.usernames.<user>.{name, password}`` with a bcrypt hash, ``cookie.{name, key,
expiry_days}``), but the login flow is implemented here with ``bcrypt`` directly instead of
``streamlit_authenticator.Authenticate.login``. Reasons:

* the lockout rule must be evaluated from the ``login_attempts`` table *before* the password
  is even checked - streamlit-authenticator checks the password itself and keeps its failed
  attempt counter only in the browser session (trivially reset by reloading the page);
* every attempt must be written to ``login_attempts`` + ``audit_log``;
* streamlit-authenticator's re-authentication cookie would silently log the user back in and
  defeat the 30-minute idle timeout. Here a browser reload or idle timeout always requires a
  fresh login (the cookie section is kept only for file compatibility).

Rules implemented:

* single admin user (``scripts/set_password.py`` creates/changes it);
* 15-minute lockout after 5 failed logins (counted from ``login_attempts`` within the last
  15 minutes, since the last successful login);
* 30-minute idle timeout (last activity kept in ``st.session_state``; auto-refresh reruns of
  live pages do not count as activity);
* every attempt is logged (``login_attempts`` row + ``audit_log`` row + JSONL ``audit``).

Pure helpers (hashing, config IO, lockout, idle) do not import Streamlit, so the CLI script
and unit tests can use them; the Streamlit UI functions import it lazily.
"""

from __future__ import annotations

import functools
import os
import secrets as pysecrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import bcrypt
import yaml
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.clock import Clock
from storage.audit import audit, record_error
from storage.models import LoginAttempt

#: Failed logins allowed inside :data:`LOCKOUT_WINDOW` before the account is locked.
MAX_FAILED_LOGINS = 5
#: Lockout / counting window.
LOCKOUT_WINDOW = timedelta(minutes=15)
#: Idle timeout for an authenticated session.
IDLE_TIMEOUT = timedelta(minutes=30)
#: Minimum password length enforced by ``scripts/set_password.py``.
MIN_PASSWORD_LENGTH = 10

REASON_OK = "OK"
REASON_BAD_PASSWORD = "BAD_PASSWORD"
REASON_UNKNOWN_USER = "UNKNOWN_USER"
REASON_LOCKED = "LOCKED_OUT"
REASON_EMPTY = "EMPTY_CREDENTIALS"
REASON_NO_CONFIG = "NO_AUTH_CONFIG"
#: Reasons that count towards the lockout (attempts made while locked do not extend it).
COUNTED_FAILURE_REASONS = (REASON_BAD_PASSWORD, REASON_UNKNOWN_USER)



@functools.cache
def _dummy_hash() -> str:
    """A valid bcrypt hash used to keep the timing of unknown-user attempts identical."""
    return bcrypt.hashpw(b"timing-equaliser", bcrypt.gensalt(rounds=12)).decode()


# =========================================================================== password hashing


def hash_password(password: str, rounds: int = 12) -> str:
    """Return a bcrypt hash (``$2b$...``) of ``password``."""
    if not password:
        raise ValueError("password must not be empty")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=rounds)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time bcrypt check. Malformed hashes return False (and are logged)."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        logger.error(f"Invalid bcrypt hash in dashboard auth file: {exc}")
        return False


# =========================================================================== auth file (YAML)


@dataclass(frozen=True)
class AuthUser:
    username: str
    name: str
    password_hash: str


@dataclass(frozen=True)
class AuthConfig:
    """Parsed streamlit-authenticator compatible secrets file."""

    users: dict[str, AuthUser]
    cookie_name: str
    cookie_key: str
    cookie_expiry_days: float

    def get_user(self, username: str) -> AuthUser | None:
        return self.users.get(normalize_username(username))


def normalize_username(username: str) -> str:
    """Usernames are case-insensitive and stripped (streamlit-authenticator lower-cases them)."""
    return (username or "").strip().lower()


def load_auth_config(path: Path) -> AuthConfig | None:
    """Load the auth YAML. Returns None if the file is missing or invalid (logged)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        users_raw = (raw.get("credentials") or {}).get("usernames") or {}
        users: dict[str, AuthUser] = {}
        for uname, info in users_raw.items():
            info = info or {}
            pw = str(info.get("password") or "")
            if not pw.startswith("$2"):
                logger.error(f"Auth file {path}: user {uname!r} has no bcrypt hash; ignored")
                continue
            key = normalize_username(str(uname))
            users[key] = AuthUser(username=key, name=str(info.get("name") or key), password_hash=pw)
        cookie = raw.get("cookie") or {}
        if not users:
            logger.error(f"Auth file {path} contains no valid users")
            return None
        return AuthConfig(
            users=users,
            cookie_name=str(cookie.get("name") or "liar_v1_dashboard"),
            cookie_key=str(cookie.get("key") or ""),
            cookie_expiry_days=float(cookie.get("expiry_days") or 1),
        )
    except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError) as exc:
        logger.error(f"Could not read dashboard auth file {path}: {exc}")
        return None


def build_auth_document(
    username: str, name: str, password_hash: str, *, cookie_key: str | None = None,
    cookie_name: str = "liar_v1_dashboard", expiry_days: float = 1,
) -> dict[str, Any]:
    """Build the streamlit-authenticator compatible YAML document for a single admin user."""
    uname = normalize_username(username)
    if not uname:
        raise ValueError("username must not be empty")
    if not password_hash.startswith("$2"):
        raise ValueError("password_hash must be a bcrypt hash")
    return {
        "credentials": {
            "usernames": {
                uname: {
                    "name": name or uname,
                    "password": password_hash,
                    "email": None,
                    "roles": ["admin"],
                }
            }
        },
        "cookie": {
            "name": cookie_name,
            "key": cookie_key or pysecrets.token_hex(32),
            "expiry_days": expiry_days,
        },
    }


def write_auth_config(path: Path, document: dict[str, Any]) -> Path:
    """Atomically write the auth YAML (creates the directory, restrictive permissions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("# Dashboard credentials (bcrypt). Created by scripts/set_password.py. NEVER commit.\n")
        yaml.safe_dump(document, fh, sort_keys=False)
    try:
        os.chmod(tmp, 0o600)
    except OSError as exc:  # Windows ignores most mode bits
        logger.warning(f"Could not restrict permissions on {tmp}: {exc}")
    os.replace(tmp, path)
    return path


# =========================================================================== lockout


def _last_success_at(session: Session, username: str, since: datetime) -> datetime | None:
    return session.execute(
        select(func.max(LoginAttempt.ts)).where(
            LoginAttempt.username == username, LoginAttempt.success.is_(True), LoginAttempt.ts >= since
        )
    ).scalar_one_or_none()


def recent_failures(session: Session, username: str, now: datetime) -> list[datetime]:
    """Timestamps of counted failed logins for ``username`` in the lockout window.

    Only failures after the most recent successful login in the window count, and attempts
    rejected because of an existing lockout do not extend it.
    """
    username = normalize_username(username)
    since = now - LOCKOUT_WINDOW
    last_ok = _last_success_at(session, username, since)
    if last_ok is not None:
        since = max(since, last_ok)
    rows = session.execute(
        select(LoginAttempt.ts)
        .where(
            LoginAttempt.username == username,
            LoginAttempt.success.is_(False),
            LoginAttempt.reason.in_(COUNTED_FAILURE_REASONS),
            LoginAttempt.ts > since,
            LoginAttempt.ts <= now,
        )
        .order_by(LoginAttempt.ts.asc())
    ).scalars()
    return list(rows)


def lockout_until(session: Session, username: str, now: datetime) -> datetime | None:
    """If ``username`` is locked out at ``now``, return when the lockout ends; else None."""
    failures = recent_failures(session, username, now)
    if len(failures) < MAX_FAILED_LOGINS:
        return None
    # Locked until the window slides past enough failures to drop below the limit.
    pivot = failures[len(failures) - MAX_FAILED_LOGINS]
    until = pivot + LOCKOUT_WINDOW
    return until if until > now else None


def record_attempt(
    session: Session,
    username: str,
    success: bool,
    reason: str,
    now: datetime,
    *,
    client: str | None = None,
    clock: Clock | None = None,
) -> LoginAttempt:
    """Persist a ``login_attempts`` row and an audit entry (caller commits)."""
    uname = normalize_username(username)[:64] or "(empty)"
    row = LoginAttempt(ts=now, username=uname, success=success, reason=reason[:200],
                       client=(client or None) and client[:200])
    session.add(row)
    action = "LOGIN_SUCCESS" if success else ("LOGIN_LOCKED" if reason == REASON_LOCKED else "LOGIN_FAILED")
    audit(session, uname, action, f"user:{uname}", {"reason": reason, "client": client}, clock=clock)
    session.flush()
    return row


@dataclass(frozen=True)
class LoginResult:
    ok: bool
    username: str
    reason: str
    locked_until: datetime | None = None
    remaining_attempts: int | None = None


def authenticate(
    session: Session,
    config: AuthConfig | None,
    username: str,
    password: str,
    now: datetime,
    *,
    client: str | None = None,
    clock: Clock | None = None,
) -> LoginResult:
    """Check credentials with lockout. Every attempt is recorded. Caller commits.

    Order: lockout check FIRST (password is not evaluated while locked), then bcrypt.
    """
    uname = normalize_username(username)
    if config is None:
        record_attempt(session, uname, False, REASON_NO_CONFIG, now, client=client, clock=clock)
        return LoginResult(False, uname, REASON_NO_CONFIG)
    if not uname or not password:
        record_attempt(session, uname, False, REASON_EMPTY, now, client=client, clock=clock)
        return LoginResult(False, uname, REASON_EMPTY)
    until = lockout_until(session, uname, now)
    if until is not None:
        record_attempt(session, uname, False, REASON_LOCKED, now, client=client, clock=clock)
        return LoginResult(False, uname, REASON_LOCKED, locked_until=until, remaining_attempts=0)
    user = config.get_user(uname)
    if user is None:
        verify_password(password, _dummy_hash())  # equalise timing
        reason = REASON_UNKNOWN_USER
        ok = False
    else:
        ok = verify_password(password, user.password_hash)
        reason = REASON_OK if ok else REASON_BAD_PASSWORD
    record_attempt(session, uname, ok, reason, now, client=client, clock=clock)
    if ok:
        return LoginResult(True, uname, REASON_OK)
    failures = len(recent_failures(session, uname, now))
    remaining = max(0, MAX_FAILED_LOGINS - failures)
    locked = lockout_until(session, uname, now)
    return LoginResult(False, uname, reason, locked_until=locked, remaining_attempts=remaining)


# =========================================================================== idle timeout


def is_idle_expired(last_activity: datetime | None, now: datetime, timeout: timedelta = IDLE_TIMEOUT) -> bool:
    """True when the session has been idle for at least ``timeout`` (or has no activity)."""
    if last_activity is None:
        return True
    return now - last_activity >= timeout


def is_user_interaction(autorefresh_count: Any, last_seen_count: Any) -> bool:
    """A rerun caused by ``st_autorefresh`` (counter changed) is NOT user activity."""
    if autorefresh_count is None:
        return True
    return autorefresh_count == last_seen_count


# =========================================================================== Streamlit UI

#: Session-state key of the approvals page auto-refresh component.
AUTOREFRESH_KEY = "approvals_autorefresh"
_AUTOREFRESH_SEEN_KEY = "_auth_autorefresh_seen"
_NOTICE_KEY = "_auth_notice"


def _client_label() -> str | None:
    import streamlit as st

    try:
        ip = st.context.ip_address
        ua = st.context.headers.get("User-Agent") if st.context.headers else None
        return " | ".join(x for x in (ip, (ua or "")[:120]) if x) or None
    except Exception as exc:  # noqa: BLE001 - context info is best effort
        logger.debug(f"No client info available: {exc}")
        return None


def logout(ctx: Any, reason: str = "LOGOUT") -> None:
    """Clear the authenticated session and audit it."""
    import streamlit as st

    from dashboard.state import AUTH_LAST_ACTIVITY_KEY, AUTH_USER_KEY

    user = st.session_state.get(AUTH_USER_KEY)
    if user:
        try:
            with ctx.session() as session:
                audit(session, user, reason, f"user:{user}", {}, clock=ctx.clock)
                session.commit()
        except Exception as exc:  # noqa: BLE001
            record_error(None, "dashboard.auth", exc, {"action": reason})
    st.session_state.pop(AUTH_USER_KEY, None)
    st.session_state.pop(AUTH_LAST_ACTIVITY_KEY, None)
    ctx.user = None


def check_session(ctx: Any) -> str | None:
    """Return the authenticated user if the session is valid (handles idle timeout).

    Updates last activity unless this rerun was triggered by the approvals auto-refresh.
    """
    import streamlit as st

    from dashboard.state import AUTH_LAST_ACTIVITY_KEY, AUTH_USER_KEY

    user = st.session_state.get(AUTH_USER_KEY)
    if not user:
        return None
    now = ctx.clock.now()
    last = st.session_state.get(AUTH_LAST_ACTIVITY_KEY)
    if is_idle_expired(last, now):
        logout(ctx, "LOGOUT_IDLE_TIMEOUT")
        st.session_state[_NOTICE_KEY] = "Session expired after 30 minutes of inactivity. Please log in again."
        return None
    count = st.session_state.get(AUTOREFRESH_KEY)
    if is_user_interaction(count, st.session_state.get(_AUTOREFRESH_SEEN_KEY)):
        st.session_state[AUTH_LAST_ACTIVITY_KEY] = now
    st.session_state[_AUTOREFRESH_SEEN_KEY] = count
    ctx.user = user
    return str(user)


def ensure_authenticated(ctx: Any) -> str:
    """Page-level guard (defence in depth): stop rendering unless logged in."""
    import streamlit as st

    user = check_session(ctx)
    if not user:
        st.warning("Please log in.")
        st.stop()
    return str(user)


def render_login(ctx: Any) -> None:
    """Login form (or set-up instructions when no credentials file exists)."""
    import streamlit as st

    from dashboard.state import AUTH_LAST_ACTIVITY_KEY, AUTH_USER_KEY

    st.title("Liar-V1 - PAPER trading")
    notice = st.session_state.pop(_NOTICE_KEY, None)
    if notice:
        st.info(notice)
    auth_file = Path(ctx.secrets.dashboard_auth_file)
    config = load_auth_config(auth_file)
    if config is None:
        st.error("Dashboard credentials are not set up (or the secrets file is invalid).")
        st.markdown(
            "Create the admin user from a terminal in the project folder, then reload this page:\n\n"
            "```\n.venv\\Scripts\\python.exe scripts/set_password.py\n```\n\n"
            f"Credentials file: `{auth_file}`"
        )
        return
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Log in", type="primary", width="stretch")
    if not submitted:
        return
    now = ctx.clock.now()
    try:
        with ctx.session() as session:
            result = authenticate(session, config, username, password, now, client=_client_label(), clock=ctx.clock)
            session.commit()
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.auth", exc, {"action": "login"})
        st.error("Login failed because of a database error. See logs.")
        return
    if result.ok:
        st.session_state[AUTH_USER_KEY] = result.username
        st.session_state[AUTH_LAST_ACTIVITY_KEY] = now
        ctx.user = result.username
        st.rerun()
    elif result.reason == REASON_LOCKED:
        until = result.locked_until.strftime("%H:%M:%S IST") if result.locked_until else "later"
        st.error(f"Too many failed logins. Locked until {until}.")
    elif result.locked_until is not None:
        st.error(f"Invalid credentials. Account locked for 15 minutes (until "
                 f"{result.locked_until.strftime('%H:%M:%S IST')}).")
    else:
        st.error(f"Invalid credentials. {result.remaining_attempts} attempt(s) left before a 15-minute lockout.")


def logout_button(ctx: Any) -> None:
    """Sidebar logout button."""
    import streamlit as st

    if st.sidebar.button("Log out", icon=":material/logout:", width="stretch", key="logout_btn"):
        logout(ctx, "LOGOUT")
        st.rerun()
