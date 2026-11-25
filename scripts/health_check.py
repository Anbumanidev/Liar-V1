"""Quick health check of the local setup (no trading, no orders - ever).

    python scripts/health_check.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check(name: str, fn) -> bool:  # type: ignore[no-untyped-def]
    try:
        detail = fn()
        print(f"[ OK ] {name}: {detail}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {exc}")
        return False


def main() -> int:
    from config.loader import load_default_settings, load_hard_caps
    from config.secrets import load_secrets
    from core.guard import startup_safety_check
    from core.net import configure_ssl

    secrets = load_secrets()
    caps = load_hard_caps()
    settings = load_default_settings()
    results = [
        check("paper-trading guard", lambda: f"{len(startup_safety_check(caps))} order targets blocked"),
        check("TLS CA bundle", lambda: configure_ssl(secrets.runtime_dir, ca_bundle=secrets.ca_bundle_path) or "defaults"),
    ]

    def db() -> str:
        from storage.db import create_all_for_url

        sf = create_all_for_url(secrets.database_url)
        with sf() as s:
            s.execute(__import__("sqlalchemy").text("select 1"))
        return secrets.database_url

    results.append(check("database", db))

    def ollama_check() -> str:
        import ollama

        models = [getattr(m, "model", None) or m.get("model") for m in ollama.Client(host=settings.models.ollama_host).list().models]
        prof = settings.models.profiles[settings.models.active_profile]
        missing = [m for m in (prof.sub_agent, prof.reasoning) if m not in models]
        if missing:
            raise RuntimeError(f"reachable, but missing models {missing} (run scripts/pull_models.py)")
        return f"{len(models)} models installed"

    results.append(check("ollama", ollama_check))

    def data_check() -> str:
        from data.providers.yfinance_provider import YFinanceProvider

        q = YFinanceProvider(clock=None).get_quote("NIFTY50")
        if q is None:
            raise RuntimeError("no quote for NIFTY50")
        return f"NIFTY50 {q.ltp} at {q.timestamp:%Y-%m-%d %H:%M}"

    results.append(check("market data (yfinance)", data_check))
    results.append(check("email credentials", lambda: "configured" if secrets.smtp_user and secrets.smtp_app_password
                         else (_ for _ in ()).throw(RuntimeError("SMTP_USER / SMTP_APP_PASSWORD missing in .env"))))
    results.append(check("dashboard password", lambda: str(secrets.dashboard_auth_file) if secrets.dashboard_auth_file.exists()
                         else (_ for _ in ()).throw(RuntimeError("run python scripts/set_password.py"))))
    print(f"\n{sum(results)}/{len(results)} checks passed at {datetime.now():%Y-%m-%d %H:%M}")
    return 0 if all(results[:3]) else 1


if __name__ == "__main__":
    sys.exit(main())
