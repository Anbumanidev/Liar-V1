"""Pull the Ollama models of a profile in config/models.yaml (open-weight, local only).

    python scripts/pull_models.py --profile low      # default; fits a 6 GB GPU
    python scripts/pull_models.py --profile high
    python scripts/pull_models.py --all              # every model referenced anywhere
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import load_default_settings  # noqa: E402


def models_for(profile: str | None, all_models: bool) -> list[str]:
    s = load_default_settings().models
    names: list[str] = []
    profiles = s.profiles.values() if all_models else [s.profiles[profile or s.active_profile]]
    for p in profiles:
        names += [p.sub_agent, p.reasoning]
    names += list(s.assignments.values())
    return list(dict.fromkeys(n for n in names if n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    try:
        import ollama
    except ImportError:
        print("The 'ollama' python package is missing: pip install -r requirements.txt")
        return 1
    settings = load_default_settings().models
    client = ollama.Client(host=settings.ollama_host)
    try:
        client.list()
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot reach Ollama at {settings.ollama_host}: {exc}\nInstall it from https://ollama.com/download "
              "and make sure it is running (the tray app, or `ollama serve`).")
        return 1
    ok = True
    for name in models_for(args.profile, args.all):
        print(f"pulling {name} ...", flush=True)
        try:
            last = ""
            for part in client.pull(name, stream=True):
                status = getattr(part, "status", None) or (part.get("status") if isinstance(part, dict) else "")
                if status and status != last:
                    print(f"  {status}", flush=True)
                    last = status
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAILED: {exc}")
    print("done" if ok else "some models failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
