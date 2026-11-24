"""Jinja2 rendering for the HTML e-mails (templates live in ``notifications/templates``).

All numbers shown in e-mails are computed in Python before rendering; the template
filters below only FORMAT values (Indian digit grouping, IST times, signs, colours).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from core.clock import to_ist

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

#: Placeholder for missing values.
DASH = "—"

# Colours (inline CSS, light e-mail surface). Text tokens, not series colours.
COLOR_POS = "#0b7a35"
COLOR_NEG = "#c0262d"
COLOR_NEUTRAL = "#52514e"


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
    return False


def _group_indian(int_part: str) -> str:
    """Indian digit grouping: 1234567 -> 12,34,567."""
    if len(int_part) <= 3:
        return int_part
    head, tail = int_part[:-3], int_part[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def fmt_num(value: Any, decimals: int = 2, signed: bool = False) -> str:
    """Format a number with Indian digit grouping (``None`` -> em dash)."""
    if not _is_number(value):
        return DASH
    v = float(value)
    rounded = round(abs(v), decimals)
    text = f"{rounded:.{decimals}f}"
    int_part, _, frac = text.partition(".")
    body = _group_indian(int_part) + (f".{frac}" if frac else "")
    if rounded == 0:
        return body
    if v < 0:
        return f"-{body}"
    return f"+{body}" if signed else body


def fmt_inr(value: Any, decimals: int = 2, signed: bool = False) -> str:
    """Format rupees: ``₹1,23,456.78`` / ``-₹12.50`` / ``+₹3.00`` (signed)."""
    if not _is_number(value):
        return DASH
    v = float(value)
    rounded = round(abs(v), decimals)
    body = fmt_num(rounded, decimals)
    if rounded == 0:
        return f"₹{body}"
    if v < 0:
        return f"-₹{body}"
    return f"+₹{body}" if signed else f"₹{body}"


def fmt_pct(value: Any, decimals: int = 2, signed: bool = False) -> str:
    """Format a percentage value already in percent units (``3.2`` -> ``3.20%``)."""
    if not _is_number(value):
        return DASH
    return f"{fmt_num(value, decimals, signed=signed)}%"


def fmt_price(value: Any) -> str:
    """Format a share price (2 decimals, no currency sign)."""
    return fmt_num(value, 2)


def fmt_dt(value: Any, fmt: str = "%d %b %Y %H:%M") -> str:
    """Format a datetime in IST (``24 Nov 2026 09:15 IST``)."""
    if isinstance(value, datetime):
        return f"{to_ist(value).strftime(fmt)} IST"
    if isinstance(value, date):
        return value.strftime("%d %b %Y")
    if isinstance(value, str) and value:
        try:
            return f"{to_ist(datetime.fromisoformat(value)).strftime(fmt)} IST"
        except ValueError:
            return value
    return DASH


def fmt_time(value: Any) -> str:
    """Format only the IST wall time (``09:45``)."""
    return fmt_dt(value, "%H:%M")


def fmt_date(value: Any) -> str:
    """Format a date (``24 Nov 2026``)."""
    if isinstance(value, datetime):
        return to_ist(value).strftime("%d %b %Y")
    if isinstance(value, date):
        return value.strftime("%d %b %Y")
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10]).strftime("%d %b %Y")
        except ValueError:
            return value
    return DASH


def fmt_duration(value: Any) -> str:
    """Humanise a duration given as ``timedelta`` or minutes (``2d 3h 05m``)."""
    if isinstance(value, timedelta):
        minutes = int(value.total_seconds() // 60)
    elif _is_number(value):
        minutes = int(value)
    else:
        return DASH
    if minutes < 0:
        minutes = 0
    days, rem = divmod(minutes, 24 * 60)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {mins:02d}m"
    if hours:
        return f"{hours}h {mins:02d}m"
    return f"{mins}m"


def pnl_color(value: Any) -> str:
    """Text colour for a P&L value (green positive, red negative, grey zero/None)."""
    if not _is_number(value) or abs(float(value)) < 0.005:
        return COLOR_NEUTRAL
    return COLOR_POS if float(value) > 0 else COLOR_NEG


def build_environment(templates_dir: Path | None = None) -> Environment:
    """Jinja2 environment with HTML autoescaping, StrictUndefined and formatting filters."""
    env = Environment(
        loader=FileSystemLoader(str(templates_dir or TEMPLATES_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html", "j2"), default_for_string=True),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        {
            "inr": fmt_inr,
            "num": fmt_num,
            "pct": fmt_pct,
            "price": fmt_price,
            "dt": fmt_dt,
            "hhmm": fmt_time,
            "d": fmt_date,
            "duration": fmt_duration,
            "pnl_color": pnl_color,
        }
    )
    env.globals.update({"DASH": DASH, "COLOR_POS": COLOR_POS, "COLOR_NEG": COLOR_NEG,
                        "COLOR_NEUTRAL": COLOR_NEUTRAL})
    return env


def render_template(env: Environment, name: str, context: dict[str, Any]) -> str:
    """Render ``<name>.html.j2`` with ``context``. Raises Jinja errors (callers degrade)."""
    template = env.get_template(f"{name}.html.j2")
    return template.render(**context)
