"""Load YAML configuration, hard caps, and enforce caps on Settings.

* ``load_hard_caps()`` - reads ``config/hard_caps.yaml``; raises if paper-only etc. are relaxed.
* ``load_default_settings()`` - builds ``Settings`` from ``config/<section>.yaml`` files.
* ``validate_against_caps()`` / ``clamp_to_caps()`` - enforce hard caps (dashboard edits).
* ``get_path()`` / ``set_path()`` - dotted-path helpers used by the dashboard and diffing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config.schema import HardCaps, Settings
from core.enums import Mode
from core.errors import ConfigError, HardCapViolationError

CONFIG_DIR = Path(__file__).resolve().parent

#: Settings section -> YAML file name in ``config/``.
SECTION_FILES: dict[str, str] = {
    "modes": "modes.yaml",
    "risk": "risk.yaml",
    "capital": "capital.yaml",
    "evaluation": "evaluation.yaml",
    "charges": "charges.yaml",
    "engine": "engine.yaml",
    "models": "models.yaml",
    "agents": "agents.yaml",
    "strategies": "strategies.yaml",
    "schedule": "schedule.yaml",
    "notifications": "notifications.yaml",
    "universe": "universe.yaml",
    "replay": "replay.yaml",
}


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping (empty file -> ``{}``)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at top level")
    return data


def load_hard_caps(path: Path | None = None) -> HardCaps:
    """Load and validate ``hard_caps.yaml``. Raises ``ConfigError`` on any violation."""
    path = path or CONFIG_DIR / "hard_caps.yaml"
    try:
        return HardCaps.model_validate(read_yaml(path))
    except ValueError as exc:
        raise ConfigError(f"hard_caps.yaml rejected: {exc}") from exc


def load_default_settings(config_dir: Path | None = None) -> Settings:
    """Build ``Settings`` from the YAML files (the seed for settings version 1)."""
    config_dir = config_dir or CONFIG_DIR
    raw: dict[str, Any] = {}
    for section, filename in SECTION_FILES.items():
        path = config_dir / filename
        if path.exists():
            raw[section] = read_yaml(path)
    try:
        return Settings.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid settings in {config_dir}: {exc}") from exc


# --------------------------------------------------------------------------- dotted paths


def get_path(data: dict[str, Any], path: str) -> Any:
    """Get a value from nested dicts by dotted path (``"risk.mis_leverage"``)."""
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set a value in nested dicts by dotted path (creates nothing; path must exist)."""
    parts = path.split(".")
    cur: Any = data
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        raise KeyError(path)
    cur[parts[-1]] = value


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    """JSON-compatible dict of settings (stable for diffing / DB storage)."""
    return settings.model_dump(mode="json")


def settings_from_dict(data: dict[str, Any]) -> Settings:
    """Validate a dict into ``Settings``."""
    return Settings.model_validate(data)


def update_settings(settings: Settings, changes: dict[str, Any]) -> Settings:
    """Return a NEW validated Settings with ``{dotted_path: value}`` changes applied."""
    data = copy.deepcopy(settings_to_dict(settings))
    for path, value in changes.items():
        set_path(data, path, value)
    return settings_from_dict(data)


# --------------------------------------------------------------------------- hard caps


@dataclass(frozen=True)
class CapRule:
    """``path`` must be <= (kind="max") or >= (kind="min") ``caps.<cap_attr>``."""

    path: str
    cap_attr: str
    kind: str  # "max" | "min"


_STATIC_RULES: tuple[CapRule, ...] = (
    CapRule("risk.max_loss_pct", "max_loss_pct", "max"),
    CapRule("risk.mis_leverage", "max_mis_leverage", "max"),
    CapRule("risk.risk_per_trade_pct", "max_risk_per_trade_pct", "max"),
    CapRule("risk.daily_loss_limit_pct", "max_daily_loss_limit_pct", "max"),
    CapRule("risk.max_position_pct", "max_position_pct", "max"),
    CapRule("risk.small_max_position_pct", "max_position_pct", "max"),
    CapRule("risk.max_overnight_exposure_pct", "max_overnight_exposure_pct", "max"),
    CapRule("engine.slippage_pct", "max_slippage_pct", "max"),
)


def cap_rules() -> list[CapRule]:
    """All cap rules, including per-mode ones."""
    rules = list(_STATIC_RULES)
    for mode in Mode:
        rules.append(CapRule(f"modes.modes.{mode.value}.min_rr", "min_rr_floor", "min"))
    return rules


def get_cap(path: str, caps: HardCaps) -> tuple[str, float] | None:
    """Return ``(kind, cap_value)`` for a settings path, or None if uncapped (dashboard hint)."""
    for rule in cap_rules():
        if rule.path == path:
            return rule.kind, float(getattr(caps, rule.cap_attr))
    return None


def clamp_value(path: str, value: float, caps: HardCaps) -> float:
    """Clamp a single numeric dashboard input to its hard cap (if any)."""
    cap = get_cap(path, caps)
    if cap is None:
        return value
    kind, limit = cap
    return min(value, limit) if kind == "max" else max(value, limit)


def validate_against_caps(settings: Settings, caps: HardCaps) -> list[str]:
    """Return human-readable violations (empty list = OK)."""
    data = settings_to_dict(settings)
    violations: list[str] = []
    for rule in cap_rules():
        value = float(get_path(data, rule.path))
        limit = float(getattr(caps, rule.cap_attr))
        if rule.kind == "max" and value > limit + 1e-12:
            violations.append(f"{rule.path}={value} exceeds hard cap {rule.cap_attr}={limit}")
        if rule.kind == "min" and value < limit - 1e-12:
            violations.append(f"{rule.path}={value} is below hard floor {rule.cap_attr}={limit}")
    if settings.risk.hard_stop_pct >= caps.max_loss_pct:
        violations.append(
            f"risk.hard_stop_pct={settings.risk.hard_stop_pct} must be below max_loss_pct cap {caps.max_loss_pct}"
        )
    return violations


def assert_within_caps(settings: Settings, caps: HardCaps) -> None:
    """Raise ``HardCapViolationError`` if any cap is violated."""
    violations = validate_against_caps(settings, caps)
    if violations:
        raise HardCapViolationError(violations)


def clamp_to_caps(settings: Settings, caps: HardCaps) -> tuple[Settings, list[str]]:
    """Return settings clamped to caps plus a list of adjustments made."""
    data = copy.deepcopy(settings_to_dict(settings))
    adjustments: list[str] = []
    for rule in cap_rules():
        value = float(get_path(data, rule.path))
        clamped = clamp_value(rule.path, value, caps)
        if clamped != value:
            set_path(data, rule.path, clamped)
            adjustments.append(f"{rule.path}: {value} -> {clamped} (hard cap {rule.cap_attr})")
    risk = data["risk"]
    if risk["hard_stop_pct"] >= risk["max_loss_pct"]:
        new = max(risk["max_loss_pct"] - 1.0, risk["max_loss_pct"] * 0.9)
        adjustments.append(f"risk.hard_stop_pct: {risk['hard_stop_pct']} -> {new} (must be < max_loss_pct)")
        risk["hard_stop_pct"] = new
    return settings_from_dict(data), adjustments
