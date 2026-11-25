"""Log tailing for the Health & logs page and the SSE ``log`` events.

* ``logs/<YYYY-MM-DD>/engine.log`` (loguru text: ``ts | LEVEL | module:func:line - message``),
  continuation lines (tracebacks) are folded into the previous entry.
* ``logs/<YYYY-MM-DD>/<category>.jsonl`` (one JSON object per line, see ``core.logging``).

Only whitelisted file names are ever opened (no path traversal); dates are parsed as dates.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

#: ``file=`` values accepted by ``/api/logs`` (engine = engine.log, others = <name>.jsonl).
LOG_FILES: tuple[str, ...] = (
    "engine", "errors", "decisions", "orders", "trades", "risk", "scans", "agents", "debates", "meta",
    "approvals", "control", "capital", "shadow", "settings", "notifications", "audit", "web",
)
TEXT_LOGS = frozenset({"engine", "web"})

_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)(?: [+-]\d{2}:?\d{2})?\s*\|\s*(?P<level>[A-Z]+)\s*\|\s*"
    r"(?P<source>[^\s]+)\s+-\s+(?P<msg>.*)$")
MAX_RECORD_CHARS = 4000


def log_path(log_dir: Path, day: date, name: str) -> Path:
    if name not in LOG_FILES:
        raise ValueError(f"unknown log {name!r}")
    fname = f"{name}.log" if name in TEXT_LOGS else f"{name}.jsonl"
    return Path(log_dir) / day.strftime("%Y-%m-%d") / fname


def tail_lines(path: Path, max_lines: int, max_bytes: int = 768 * 1024) -> list[str]:
    """Last ``max_lines`` lines of a text file without reading the whole file."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    start = max(0, size - max_bytes)
    with open(path, "rb") as fh:
        fh.seek(start)
        raw = fh.read()
    lines = raw.decode("utf-8", errors="replace").splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # first line is probably partial
    return lines[-max_lines:] if max_lines > 0 else lines


def parse_text_lines(lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        m = _LINE_RE.match(line)
        if m:
            out.append({"ts": m.group("ts").replace(" ", "T"), "level": m.group("level"),
                        "source": m.group("source"), "message": m.group("msg")})
        elif out:
            out[-1]["message"] += "\n" + line
        elif line.strip():
            out.append({"ts": None, "level": "", "source": "", "message": line})
    for e in out:
        if len(e["message"]) > MAX_RECORD_CHARS:
            e["message"] = e["message"][:MAX_RECORD_CHARS] + "…"
    return out


def parse_jsonl_lines(lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) > MAX_RECORD_CHARS:
            try:
                rec = json.loads(line)
                brief = {k: v for k, v in rec.items() if isinstance(v, (str, int, float, bool)) or v is None}
                brief = {k: (v[:300] + "…" if isinstance(v, str) and len(v) > 300 else v) for k, v in brief.items()}
                brief["_truncated"] = True
                out.append(brief)
            except json.JSONDecodeError:
                out.append({"_raw": line[:MAX_RECORD_CHARS] + "…", "_truncated": True})
            continue
        try:
            rec = json.loads(line)
            out.append(rec if isinstance(rec, dict) else {"value": rec})
        except json.JSONDecodeError:
            out.append({"_raw": line})
    return out


def read_log(log_dir: Path, day: date, name: str, lines: int = 200, query: str | None = None,
             level: str | None = None) -> dict[str, Any]:
    path = log_path(log_dir, day, name)
    n = max(1, min(int(lines), 2000))
    raw = tail_lines(path, n * (4 if (query or level) else 1) + (50 if name in TEXT_LOGS else 0))
    if name in TEXT_LOGS:
        entries = parse_text_lines(raw)
        if level:
            lv = level.upper()
            entries = [e for e in entries if e["level"] == lv or (lv == "WARNING" and e["level"] in ("ERROR",
                                                                                                    "CRITICAL"))]
    else:
        entries = parse_jsonl_lines(raw)
        if level:
            lv = level.upper()
            entries = [e for e in entries if str(e.get("severity") or e.get("level") or "").upper() == lv]
    if query:
        q = query.lower()
        entries = [e for e in entries if q in json.dumps(e, ensure_ascii=False, default=str).lower()]
    entries = entries[-n:]
    return {"file": name, "kind": "text" if name in TEXT_LOGS else "jsonl", "date": day.isoformat(),
            "exists": path.exists(), "count": len(entries), "entries": entries}


class LogFollower:
    """Incrementally read lines appended to today's engine.log (offset tracking for SSE)."""

    def __init__(self, log_dir: Path, name: str = "engine") -> None:
        self.log_dir = Path(log_dir)
        self.name = name
        self.day: date | None = None
        self.offset = 0
        self._partial = ""

    def start(self, day: date) -> None:
        self.day = day
        path = log_path(self.log_dir, day, self.name)
        try:
            self.offset = path.stat().st_size
        except OSError:
            self.offset = 0
        self._partial = ""

    def poll(self, day: date, max_bytes: int = 64 * 1024) -> list[dict[str, Any]]:
        if self.day != day:
            self.day = day
            self.offset = 0
            self._partial = ""
        path = log_path(self.log_dir, day, self.name)
        try:
            size = path.stat().st_size
        except OSError:
            return []
        if size < self.offset:  # rotated / truncated
            self.offset = 0
            self._partial = ""
        if size == self.offset:
            return []
        start = self.offset
        if size - start > max_bytes:
            start = size - max_bytes
            self._partial = ""
        with open(path, "rb") as fh:
            fh.seek(start)
            raw = fh.read(size - start)
        self.offset = size
        textbuf = self._partial + raw.decode("utf-8", errors="replace")
        parts = textbuf.split("\n")
        self._partial = parts.pop() if not textbuf.endswith("\n") else ""
        lines = [p.rstrip("\r") for p in parts if p.strip()]
        return parse_text_lines(lines)


def file_size(path: Path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
