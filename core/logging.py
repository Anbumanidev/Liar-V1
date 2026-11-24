"""Logging: loguru for human logs + raw JSONL event logs in ``logs/YYYY-MM-DD/``.

Usage::

    from core.logging import setup_logging, log_event
    setup_logging(process_name="engine")
    log_event("decisions", {"symbol": "RELIANCE", ...}, clock=ctx.clock)

JSONL categories used across the system (one file per category per day):
``agents``, ``debates``, ``decisions``, ``risk``, ``orders``, ``trades``, ``capital``,
``approvals``, ``control``, ``settings``, ``errors``, ``scans``, ``notifications``,
``shadow``, ``replay``.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from core.clock import IST, Clock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

_jsonl_lock = threading.Lock()
_log_dir: Path = DEFAULT_LOG_DIR
_configured = False


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, set):
        return sorted(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:  # pragma: no cover
        pass
    return str(obj)


def setup_logging(
    process_name: str = "engine",
    log_dir: Path | None = None,
    level: str = "INFO",
    console: bool = True,
) -> None:
    """Configure loguru sinks: console + ``logs/YYYY-MM-DD/<process>.log`` (rotated daily)."""
    global _log_dir, _configured
    _log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    logger.remove()
    if console:
        logger.add(sys.stderr, level=level, enqueue=False, backtrace=False, diagnose=False)
    pattern = str(_log_dir / "{time:YYYY-MM-DD}" / f"{process_name}.log")
    logger.add(
        pattern,
        level="DEBUG",
        rotation="00:00",
        retention="180 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    _configured = True


def get_log_dir() -> Path:
    """Directory that holds the dated log folders."""
    return _log_dir


def set_log_dir(path: Path) -> None:
    """Override the JSONL log directory (tests / replay)."""
    global _log_dir
    _log_dir = Path(path)


def log_event(category: str, payload: dict[str, Any], clock: Clock | None = None) -> None:
    """Append one JSON object to ``logs/<YYYY-MM-DD>/<category>.jsonl``.

    The date folder follows the (possibly simulated) clock so REPLAY logs land in the
    replayed day's folder. Never raises: logging failures are reported via loguru.
    """
    now = clock.now() if clock is not None else datetime.now(IST)
    record = {"ts": now.isoformat(), "category": category, **payload}
    try:
        folder = _log_dir / now.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=_json_default, ensure_ascii=False)
        with _jsonl_lock:
            with open(folder / f"{category}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 - logging must never break trading
        logger.error(f"JSONL log write failed for {category}: {exc}")


def read_events(day: date, category: str, log_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read back the JSONL events for ``day``/``category`` (dashboard log viewer)."""
    folder = (Path(log_dir) if log_dir else _log_dir) / day.strftime("%Y-%m-%d")
    path = folder / f"{category}.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping corrupt JSONL line in {path}: {exc}")
    return out


def to_jsonable(obj: Any) -> Any:
    """Round-trip ``obj`` through JSON using the project's encoder (for DB JSON columns)."""
    return json.loads(json.dumps(obj, default=_json_default))
