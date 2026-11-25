"""Back up the SQLite trading database (online, consistent copy via the SQLite backup API).

    python scripts/backup_db.py                 # -> runtime/backups/trading_YYYYmmdd_HHMMSS.db
    python scripts/backup_db.py --keep 30       # also prune to the newest 30 backups
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.secrets import load_secrets  # noqa: E402
from core.clock import IST  # noqa: E402


def backup(db_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"trading_{datetime.now(IST):%Y%m%d_%H%M%S}.db"
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(out))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0, help="keep only the newest N backups (0 = keep all)")
    args = ap.parse_args()
    secrets = load_secrets()
    url = secrets.database_url
    if not url.startswith("sqlite:///"):
        print("Only SQLite is backed up by this script; use pg_dump for PostgreSQL.")
        return 1
    db = Path(url.removeprefix("sqlite:///"))
    if not db.exists():
        print(f"database not found: {db}")
        return 1
    dest = secrets.runtime_dir / "backups"
    out = backup(db, dest)
    print(f"backup written: {out}")
    if args.keep > 0:
        files = sorted(dest.glob("trading_*.db"))
        for old in files[: -args.keep]:
            old.unlink()
            print(f"pruned {old.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
