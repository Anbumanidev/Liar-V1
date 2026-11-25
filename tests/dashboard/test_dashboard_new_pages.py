"""Home cockpit, Tax & charges, Risk lab pages + cockpit data + autonomy switch."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from config.loader import load_default_settings, load_hard_caps
from config.secrets import Secrets
from core.clock import IST
from dashboard.cockpit import cockpit, setup_checklist
from storage.models import ControlCommand, SettingsVersion

TIMEOUT = 90


def _page_script(module_name: str) -> None:
    import importlib

    importlib.import_module(module_name).render()


def _app(page: str) -> AppTest:
    at = AppTest.from_function(_page_script, kwargs={"module_name": f"dashboard.pages.{page}"},
                               default_timeout=TIMEOUT)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_last_activity"] = datetime.now(IST)
    return at


@pytest.mark.parametrize("page", ["home", "analytics_tax", "analytics_risk_lab"])
def test_new_pages_render(page: str, seeded: dict[str, int]) -> None:
    at = _app(page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_home_autonomy_switch_saves_settings(seeded: dict[str, int], dash_factory) -> None:  # type: ignore[no-untyped-def]
    at = _app("home")
    at.run()
    radio = next(r for r in at.radio if r.key == "home_autonomy_choice")
    radio.set_value("HUMAN_IN_THE_LOOP").run()
    next(b for b in at.button if b.key == "home_autonomy_apply").click().run()
    assert not at.exception, [e.value for e in at.exception]
    with dash_factory() as s:
        latest = s.execute(select(SettingsVersion).order_by(SettingsVersion.id.desc())).scalars().first()
        assert latest.settings_json["modes"]["autonomy"] == "HUMAN_IN_THE_LOOP"
        assert s.execute(select(ControlCommand).where(ControlCommand.command_type == "RELOAD_SETTINGS")).first()


def test_full_auto_needs_confirmation(seeded: dict[str, int]) -> None:
    at = _app("home")
    at.run()
    next(r for r in at.radio if r.key == "home_autonomy_choice").set_value("FULL_AUTO").run()
    apply = next(b for b in at.button if b.key == "home_autonomy_apply")
    assert apply.disabled          # until CONFIRM is typed


def test_tax_calculator(seeded: dict[str, int]) -> None:
    at = _app("analytics_tax")
    at.run()
    calc = next(b for b in at.button if b.label == "Calculate")
    calc.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Tax head" in m.value for m in at.markdown)


def test_cockpit_and_checklist(session) -> None:  # type: ignore[no-untyped-def]
    c = cockpit(session, None, datetime(2026, 11, 25, 10, tzinfo=IST))
    assert c.equity is None and c.open_positions == 0
    items = setup_checklist(load_default_settings(), load_hard_caps(), Secrets(_env_file=None),
                            engine_alive=False, engine_data_ok=None, ollama_models=None, today=date(2026, 11, 25))
    names = {i.name: i for i in items}
    assert names["Paper trading only"].ok is True and names["Engine running"].ok is False
    assert names["AI models (Ollama)"].ok is False and "pull-models" in names["AI models (Ollama)"].fix
