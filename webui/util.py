"""Small shared helpers for the web API: rounding, IST time formatting, parsing, lazy modules."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from typing import Any

from fastapi import HTTPException

from core.clock import IST, to_ist

#: Lightweight Charts renders timestamps as UTC. Every chart ``time`` this API returns is the
#: bar's UTC epoch shifted by +05:30 so the chart axis reads IST wall-clock time.
IST_OFFSET_S = 19_800

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-_.^]{0,23}$")
NOT_AVAILABLE = "not available yet"


class NotAvailable(Exception):
    """A lazily imported module/function (written in parallel) does not exist yet."""


def r2(value: Any) -> float | None:
    """Round to 2 dp (money/prices); None/NaN/inf -> None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    out = round(f, 2)
    return 0.0 if out == 0 else out


def r4(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 4)


def iso(value: datetime | date | None) -> str | None:
    """IST ISO-8601 string (seconds precision) for datetimes; ``YYYY-MM-DD`` for dates."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_ist(value).isoformat(timespec="seconds")
    return value.isoformat()


def chart_time(value: datetime) -> int:
    """Epoch seconds shifted to IST wall-clock (see :data:`IST_OFFSET_S`)."""
    return int(to_ist(value).timestamp()) + IST_OFFSET_S


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """``[00:00, next day 00:00)`` IST-aware bounds of ``day``."""
    lo = datetime.combine(day, time.min, tzinfo=IST)
    return lo, lo + timedelta(days=1)


def parse_day(value: str | None, default: date) -> date:
    """``YYYY-MM-DD`` -> date (HTTP 422 on garbage)."""
    if not value:
        return default
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid date {value!r}; use YYYY-MM-DD") from exc


def parse_dt(value: str | None) -> datetime | None:
    """ISO datetime (naive = IST) or date -> IST-aware datetime; HTTP 422 on garbage."""
    if not value:
        return None
    text = value.strip()
    try:
        if len(text) == 10:
            return datetime.combine(date.fromisoformat(text), time.min, tzinfo=IST)
        return to_ist(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid datetime {value!r}") from exc


def clean_symbol(value: str | None) -> str:
    """Validated upper-case NSE symbol (HTTP 422 otherwise)."""
    sym = (value or "").strip().upper().removesuffix(".NS")
    if not _SYMBOL_RE.match(sym):
        raise HTTPException(status_code=422, detail=f"invalid symbol {value!r}")
    return sym


def is_symbol(value: str | None) -> bool:
    return bool(value) and bool(_SYMBOL_RE.match(str(value).strip().upper()))


def optional_attr(module: str, name: str) -> Callable[..., Any]:
    """Import ``module.name`` lazily. Raises :class:`NotAvailable` when missing.

    Used for modules another engineer writes in parallel (``capital.funding``,
    ``analytics.postmortem``, ``analytics.diagnostics``); endpoints map it to HTTP 501.
    """
    try:
        mod = importlib.import_module(module)
        fn = getattr(mod, name)
    except (ImportError, AttributeError) as exc:
        raise NotAvailable(f"{module}.{name}: {exc}") from exc
    if not callable(fn):
        raise NotAvailable(f"{module}.{name} is not callable")
    return fn


def not_available() -> HTTPException:
    return HTTPException(status_code=501, detail=NOT_AVAILABLE)


def scalars_only(data: Any, limit: int = 40) -> dict[str, Any]:
    """Keep the scalar top-level items of a JSON dict (status/regime blobs for the status bar)."""
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in list(data.items())[:limit]:
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
    return out


def text(value: Any, limit: int = 600) -> str | None:
    """Plain string (truncated) for LLM/DB text shown in the UI."""
    if value is None:
        return None
    s = value if isinstance(value, str) else str(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def str_list(value: Any, limit: int = 12, item_limit: int = 400) -> list[str]:
    if not value:
        return []
    if isinstance(value, (str, bytes)):
        return [text(value, item_limit) or ""]
    out: list[str] = []
    try:
        for item in list(value)[:limit]:
            if item is None:
                continue
            out.append(text(item if isinstance(item, str) else str(item), item_limit) or "")
    except TypeError:
        return [text(value, item_limit) or ""]
    return out
