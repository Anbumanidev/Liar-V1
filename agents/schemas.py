"""Pydantic schemas for LLM agent outputs (rule 0.4: every LLM answer is validated JSON).

``AgentOutput`` holds the fields every Layer-1 sub-agent must return. Agent-specific
output models subclass it (see ``agents/sub_agents/*``). Debate / meta agents can reuse
``clamp_int`` / ``truncate_list`` / ``normalize_signal`` for their own schemas.

Design choices for small local models (qwen3:4b class):

* Validation is *lenient on cosmetics* (case of enum strings, too many findings, overly
  long reasoning are fixed in code) and *strict on structure* (missing / wrongly typed
  fields trigger the re-prompt loop in ``agents.llm``).
* ``extra="ignore"``: an LLM adding an unexpected key must not burn a retry.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.enums import Signal

#: Max key findings kept per report and max characters per finding.
MAX_FINDINGS: int = 6
MAX_FINDING_CHARS: int = 200
#: Max characters of free-text reasoning kept per report.
MAX_REASONING_CHARS: int = 800


def clamp_int(value: Any, lo: int, hi: int) -> Any:
    """Coerce a numeric value to an ``int`` inside ``[lo, hi]``; leave non-numerics for Pydantic.

    LLMs sometimes return ``"85"``, ``85.0`` or ``120`` for a 0-100 confidence; this keeps
    those answers valid instead of burning a retry on a cosmetic issue.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        txt = value.strip().rstrip("%").strip()
        try:
            value = float(txt)
        except ValueError:
            return value  # let Pydantic report it
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return lo
        return int(max(lo, min(hi, round(value))))
    return value


def truncate_text(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters (adds an ellipsis when cut)."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def truncate_list(items: Any, max_items: int, max_chars: int) -> Any:
    """Keep at most ``max_items`` non-empty strings of at most ``max_chars`` each."""
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return items
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        txt = truncate_text(str(item), max_chars)
        if txt:
            out.append(txt)
        if len(out) >= max_items:
            break
    return out


def normalize_signal(value: Any) -> Any:
    """Accept ``bullish`` / ``Bullish`` / ``BULLISH`` (and BUY/SELL synonyms)."""
    if isinstance(value, str):
        v = value.strip().upper()
        synonyms = {"BUY": "BULLISH", "LONG": "BULLISH", "SELL": "BEARISH", "SHORT": "BEARISH",
                    "HOLD": "NEUTRAL", "NONE": "NEUTRAL"}
        return synonyms.get(v, v)
    return value


class AgentOutput(BaseModel):
    """Fields every Layer-1 sub-agent LLM answer must contain."""

    model_config = ConfigDict(extra="ignore")

    signal: Signal = Field(description="BULLISH, BEARISH or NEUTRAL for the stock in this mode")
    confidence: int = Field(ge=0, le=100, description="0-100 confidence in the signal")
    key_findings: list[str] = Field(
        default_factory=list,
        description=f"At most {MAX_FINDINGS} short findings, each quoting the provided facts",
    )
    reasoning: str = Field(default="", description=f"Short reasoning, <= {MAX_REASONING_CHARS} characters")

    @field_validator("signal", mode="before")
    @classmethod
    def _v_signal(cls, v: Any) -> Any:
        return normalize_signal(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _v_conf(cls, v: Any) -> Any:
        return clamp_int(v, 0, 100)

    @field_validator("key_findings", mode="before")
    @classmethod
    def _v_findings(cls, v: Any) -> Any:
        return truncate_list(v, MAX_FINDINGS, MAX_FINDING_CHARS)

    @field_validator("reasoning", mode="before")
    @classmethod
    def _v_reasoning(cls, v: Any) -> Any:
        if v is None:
            return ""
        return truncate_text(str(v), MAX_REASONING_CHARS) if isinstance(v, str) else v


def output_schema_summary(model: type[BaseModel]) -> str:
    """Compact, human-readable description of ``model``'s JSON fields for prompts.

    Example line: ``- confidence (integer): 0-100 confidence in the signal``.
    """
    schema = model.model_json_schema()
    props: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines: list[str] = []
    for name, spec in props.items():
        typ = spec.get("type")
        if typ is None and "anyOf" in spec:
            typ = " | ".join(str(s.get("type", s.get("$ref", "?")).split("/")[-1]) for s in spec["anyOf"])
        if typ is None and "$ref" in spec:
            ref = spec["$ref"].split("/")[-1]
            enum = schema.get("$defs", {}).get(ref, {}).get("enum")
            typ = " | ".join(enum) if enum else ref
        if "enum" in spec:
            typ = " | ".join(str(e) for e in spec["enum"])
        desc = spec.get("description", "")
        req = "" if name in required else " (optional)"
        lines.append(f"- {name} ({typ}){req}: {desc}".rstrip(": "))
    return "\n".join(lines)
