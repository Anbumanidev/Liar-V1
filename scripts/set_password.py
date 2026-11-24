"""Create or change the dashboard admin user (section 14.1).

Usage (from the project root)::

    .venv\\Scripts\\python.exe scripts/set_password.py                      # interactive (getpass)
    .venv\\Scripts\\python.exe scripts/set_password.py --username owner
    echo SECRET | .venv\\Scripts\\python.exe scripts/set_password.py --password-stdin   # automation

Writes a streamlit-authenticator compatible YAML (bcrypt hash + random cookie key) to
``DASHBOARD_AUTH_FILE`` (default ``secrets/dashboard_auth.yaml``, git-ignored). There is a
single admin user: running the script again replaces it. The password is never printed or logged.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.secrets import Secrets  # noqa: E402
from dashboard.auth import (  # noqa: E402
    MIN_PASSWORD_LENGTH,
    build_auth_document,
    hash_password,
    normalize_username,
    write_auth_config,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set the Liar-V1 dashboard admin password (bcrypt).")
    parser.add_argument("--username", default=None, help="admin username (default: prompt, 'admin')")
    parser.add_argument("--name", default=None, help="display name (default: username)")
    parser.add_argument("--password-stdin", action="store_true",
                        help="read the password from the first line of stdin (no prompt, no echo)")
    parser.add_argument("--auth-file", type=Path, default=None,
                        help="output YAML (default: DASHBOARD_AUTH_FILE from .env or secrets/dashboard_auth.yaml)")
    parser.add_argument("--cookie-expiry-days", type=float, default=1.0,
                        help="cookie expiry kept for streamlit-authenticator compatibility (default 1)")
    return parser.parse_args(argv)


def _read_password(from_stdin: bool) -> str:
    if from_stdin:
        line = sys.stdin.readline()
        return line.rstrip("\r\n")
    first = getpass.getpass("New dashboard password: ")
    second = getpass.getpass("Repeat password: ")
    if first != second:
        raise SystemExit("Passwords do not match. Nothing was written.")
    return first


def validate_password(password: str) -> str | None:
    """Return an error message for a weak password, else None."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if password.strip() != password:
        return "Password must not start or end with whitespace."
    if len(set(password)) < 4:
        return "Password is too repetitive."
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    username = args.username
    if username is None:
        username = "admin" if args.password_stdin else (input("Admin username [admin]: ").strip() or "admin")
    username = normalize_username(username)
    if not username or not username.replace("_", "").replace("-", "").replace(".", "").isalnum():
        print("Invalid username (use letters, digits, _ - .).", file=sys.stderr)
        return 2
    password = _read_password(args.password_stdin)
    error = validate_password(password)
    if error:
        print(error + " Nothing was written.", file=sys.stderr)
        return 2
    auth_file = Path(args.auth_file) if args.auth_file else Path(Secrets().dashboard_auth_file)
    if not auth_file.is_absolute():
        auth_file = _ROOT / auth_file
    document = build_auth_document(username, args.name or username, hash_password(password),
                                   expiry_days=args.cookie_expiry_days)
    write_auth_config(auth_file, document)
    print(f"Saved dashboard credentials for user '{username}' to {auth_file}")
    print("Start the dashboard with:  .venv\\Scripts\\python.exe -m streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
