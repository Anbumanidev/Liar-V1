"""Streamlit AppTest smoke tests: every control page renders on a seeded temp DB.

Authentication is injected into ``session_state`` by the tests only - there is no
production bypass. ``DATABASE_URL`` points at a temp SQLite file (see conftest).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from streamlit.testing.v1 import AppTest

from core.clock import IST
from dashboard.app import CONTROL_PAGES
from storage.models import ControlCommand, SettingsVersion

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 60


def _page_script(module_name: str) -> None:
    import importlib

    importlib.import_module(module_name).render()


def _app(page: str, *, logged_in: bool = True) -> AppTest:
    at = AppTest.from_function(_page_script, kwargs={"module_name": f"dashboard.pages.{page}"},
                               default_timeout=TIMEOUT)
    if logged_in:
        at.session_state["auth_user"] = "admin"
        at.session_state["auth_last_activity"] = datetime.now(IST)
    return at


def _commands(factory: sessionmaker[Session], ctype: str) -> list[ControlCommand]:
    with factory() as s:
        return list(s.execute(select(ControlCommand).where(ControlCommand.command_type == ctype)).scalars())


def test_dashboard_all_control_pages_exist() -> None:
    assert len(CONTROL_PAGES) == 14
    for name in CONTROL_PAGES:
        path = ROOT / "dashboard" / "pages" / f"{name}.py"
        assert path.exists(), name
        import importlib

        module = importlib.import_module(f"dashboard.pages.{name}")
        assert isinstance(module.TITLE, str) and module.ICON and callable(module.render)


@pytest.mark.parametrize("page", CONTROL_PAGES)
def test_dashboard_page_renders(page: str, seeded: dict[str, int]) -> None:
    at = _app(page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.subheader, "page header missing"


@pytest.mark.parametrize("page", ["control_engine", "control_approvals", "control_system"])
def test_dashboard_page_renders_on_empty_db(page: str, dash_db_url: str) -> None:
    at = _app(page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_page_requires_login(seeded: dict[str, int]) -> None:
    at = _app("control_engine", logged_in=False)
    at.run()
    assert not at.exception
    assert any("log in" in w.value.lower() for w in at.warning)
    assert not at.button  # nothing rendered past the guard


def test_dashboard_idle_session_is_logged_out(seeded: dict[str, int]) -> None:
    at = _app("control_engine")
    at.session_state["auth_last_activity"] = datetime(2020, 1, 1, tzinfo=IST)
    at.run()
    assert not at.exception
    assert "auth_user" not in at.session_state
    assert not at.button


def test_dashboard_engine_pause_enqueues_command(seeded: dict[str, int],
                                                 dash_factory: sessionmaker[Session]) -> None:
    at = _app("control_engine")
    at.run()
    at.button(key="eng_pause").click().run()
    assert not at.exception
    cmds = _commands(dash_factory, "ENGINE_PAUSE")
    assert len(cmds) == 1 and cmds[0].requested_by == "admin" and cmds[0].batch_id == seeded["batch"]
    # kill switch is disabled until CONFIRM is typed
    assert at.button(key="kill_btn").disabled
    at.text_input(key="kill_confirm").input("CONFIRM").run()
    assert not at.button(key="kill_btn").disabled
    at.checkbox(key="kill_exit_cnc").check().run()
    at.button(key="kill_btn").click().run()
    kill = _commands(dash_factory, "KILL_SWITCH")
    assert len(kill) == 1 and kill[0].payload_json["exit_cnc"] is True


def test_dashboard_approve_enqueues_command(seeded: dict[str, int], dash_factory: sessionmaker[Session]) -> None:
    at = _app("control_approvals")
    at.session_state["appr_pause_refresh"] = True
    at.run()
    assert not at.exception
    assert any("RELIANCE" in m.value for m in at.markdown)
    at.button(key=f"ap_{seeded['approval']}").click().run()
    cmds = _commands(dash_factory, "APPROVE")
    assert len(cmds) == 1 and cmds[0].payload_json == {"approval_id": seeded["approval"]}


def test_dashboard_modes_allocation_over_100_is_rejected(seeded: dict[str, int],
                                                         dash_factory: sessionmaker[Session]) -> None:
    at = _app("control_modes")
    at.run()
    before = _count_versions(dash_factory)
    at.slider(key="alloc_BTST").set_value(30.0)
    at.button(key="FormSubmitter:modes_alloc_form-Save modes & allocations").click().run()
    assert not at.exception
    assert any("must be <= 100" in e.value for e in at.error)
    assert _count_versions(dash_factory) == before
    # a valid change creates a version and RELOAD_SETTINGS
    at.slider(key="alloc_BTST").set_value(0.0)
    at.slider(key="alloc_SWING").set_value(40.0)
    at.button(key="FormSubmitter:modes_alloc_form-Save modes & allocations").click().run()
    assert not at.exception
    assert _count_versions(dash_factory) == before + 1
    assert len(_commands(dash_factory, "RELOAD_SETTINGS")) == 1


def _count_versions(factory: sessionmaker[Session]) -> int:
    with factory() as s:
        return len(s.execute(select(SettingsVersion)).scalars().all())


def test_dashboard_app_shows_setup_instructions_without_auth_file(dash_db_url: str) -> None:
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("set_password.py" in m.value for m in at.markdown)


def test_dashboard_app_login_flow(dash_db_url: str, seeded: dict[str, int], tmp_path: Path) -> None:
    from dashboard import auth

    path = tmp_path / "secrets" / "dashboard_auth.yaml"
    auth.write_auth_config(path, auth.build_auth_document("admin", "Owner",
                                                          auth.hash_password("a-very-long-Pass#1", rounds=4)))
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    at.text_input[0].input("admin")
    at.text_input[1].input("wrong-password")
    at.button[0].click().run()
    assert any("Invalid credentials" in e.value for e in at.error)
    assert "auth_user" not in at.session_state
    at.text_input[1].input("a-very-long-Pass#1")
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["auth_user"] == "admin"
    assert at.sidebar.selectbox(key="sidebar_batch_select").value == seeded["batch"]
