"""Secrets from ``.env`` (git-ignored). NEVER hard-code credentials (rule 0.9)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Secrets(BaseSettings):
    """All credentials and machine-specific paths. Loaded from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # storage
    database_url: str = Field(default=f"sqlite:///{(PROJECT_ROOT / 'runtime' / 'trading.db').as_posix()}")
    runtime_dir: Path = PROJECT_ROOT / "runtime"
    log_dir: Path = PROJECT_ROOT / "logs"

    # email (Gmail App Password)
    smtp_user: str | None = None
    smtp_app_password: SecretStr | None = None
    alert_email_to: str | None = None

    # Angel One SmartAPI (market data ONLY)
    angel_api_key: SecretStr | None = None
    angel_client_code: str | None = None
    angel_pin: SecretStr | None = None
    angel_totp_secret: SecretStr | None = None

    # Upstox (market data ONLY)
    upstox_api_key: SecretStr | None = None
    upstox_api_secret: SecretStr | None = None
    upstox_access_token: SecretStr | None = None

    # dashboard
    dashboard_auth_file: Path = PROJECT_ROOT / "secrets" / "dashboard_auth.yaml"
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8501

    # TLS
    ca_bundle_path: str | None = None

    # optional: Ollama host override
    ollama_host: str | None = None


def load_secrets() -> Secrets:
    """Load secrets from the environment and ``.env``."""
    return Secrets()
