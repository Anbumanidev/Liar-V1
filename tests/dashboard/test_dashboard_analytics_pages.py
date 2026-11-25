"""Streamlit AppTest smoke tests for the 9 analytics pages (section 14.4).

Every page's ``render()`` must run without an exception AND without an error box (pages catch
their own exceptions and show ``st.error``, so both are checked) on: an empty DB, a DB with a
bare batch and nothing else, the realistic analytics seed, and the control-pages seed.
Authentication is injected into ``session_state`` by the tests only.
"""

from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker
from streamlit.testing.v1 import AppTest

from core.clock import IST
from dashboard.app import ANALYTICS_PAGES, page_modules
from storage.models import Batch
from tests.dashboard.test_dashboard_analytics_seed import seed_analytics_db, seed_days

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 60


def _page_script(module_name: str) -> None:
    import importlib

    importlib.import_module(module_name).render()


def _app(page: str) -> AppTest:
    at = AppTest.from_function(_page_script, kwargs={"module_name": f"dashboard.pages.{page}"},
                               default_timeout=TIMEOUT)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_last_activity"] = datetime.now(IST)
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.subheader, "page header missing"


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


@pytest.fixture
def analytics_seeded(dash_factory: sessionmaker[Session]) -> dict[str, int]:
    with dash_factory() as s:
        return seed_analytics_db(s, datetime.now(IST))


@pytest.fixture
def bare_batch(dash_factory: sessionmaker[Session]) -> int:
    with dash_factory() as s:
        b = Batch(name="bare", starting_capital=1000.0, status="ACTIVE", created_at=datetime.now(IST))
        s.add(b)
        s.commit()
        return b.id


def test_dashboard_analytics_pages_registered() -> None:
    assert len(ANALYTICS_PAGES) == 11
    for name in ANALYTICS_PAGES:
        assert (ROOT / "dashboard" / "pages" / f"{name}.py").exists(), name
        module = importlib.import_module(f"dashboard.pages.{name}")
        assert isinstance(module.TITLE, str) and module.TITLE and module.ICON and callable(module.render)
    loaded = page_modules(ANALYTICS_PAGES, only_existing=True)
    assert [name for name, _ in loaded] == list(ANALYTICS_PAGES)  # all, in order, nothing extra


@pytest.mark.parametrize("page", ANALYTICS_PAGES)
def test_dashboard_analytics_page_renders_on_empty_db(page: str, dash_db_url: str) -> None:
    at = _app(page)
    at.run()
    _ok(at)


@pytest.mark.parametrize("page", ANALYTICS_PAGES)
def test_dashboard_analytics_page_renders_on_bare_batch(page: str, bare_batch: int) -> None:
    at = _app(page)
    at.run()
    _ok(at)


@pytest.mark.parametrize("page", ANALYTICS_PAGES)
def test_dashboard_analytics_page_renders_on_seeded_db(page: str, analytics_seeded: dict[str, int]) -> None:
    at = _app(page)
    at.run()
    _ok(at)
    assert at.session_state["selected_batch_id"] == analytics_seeded["batch"]


@pytest.mark.parametrize("page", ANALYTICS_PAGES)
def test_dashboard_analytics_page_renders_on_control_seed(page: str, seeded: dict[str, int]) -> None:
    at = _app(page)
    at.run()
    _ok(at)


def test_dashboard_analytics_page_requires_login(analytics_seeded: dict[str, int]) -> None:
    at = AppTest.from_function(_page_script, kwargs={"module_name": "dashboard.pages.analytics_overview"},
                               default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert any("log in" in w.value.lower() for w in at.warning)
    assert not at.metric


def test_dashboard_analytics_overview_content(analytics_seeded: dict[str, int]) -> None:
    at = _app("analytics_overview")
    at.run()
    _ok(at)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["TRADING"] == "Rs 988.01" and metrics["RESERVE"] == "Rs 8.29"
    assert metrics["TOTAL EQUITY"] == "Rs 1,007.90" and metrics["High-water mark"] == "Rs 1,034.00"
    assert metrics["Tier"] == "NORMAL" and metrics["Net P&L"] == "Rs -3.70" and metrics["Win rate"] == "75.0%"
    assert "Net P&L per mode: INTRADAY Rs +61.30 (3) | SWING Rs -65.00 (1)" in _text(at)
    at.toggle(key="ov_zoom").set_value(False).run()
    _ok(at)


def test_dashboard_analytics_live_content(analytics_seeded: dict[str, int]) -> None:
    at = _app("analytics_live")
    at.session_state["live_pause_refresh"] = True
    at.run()
    _ok(at)
    text = _text(at)
    assert "Open positions (1)" in text and "Pending approvals (1)" in text
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Open-position risk"] == "Rs 20.40" and metrics["Stricter floor"] == "Rs 940.94"
    assert metrics["Loss budget remaining"] == "Rs 46.56"


def test_dashboard_analytics_trades_drilldown(analytics_seeded: dict[str, int]) -> None:
    at = _app("analytics_trades")
    at.run()
    _ok(at)
    picker = at.selectbox(key=f"tr_pick_{analytics_seeded['batch']}")
    assert picker.value == analytics_seeded["t5"]                 # newest exit first (manual NHPC)
    picker.set_value(analytics_seeded["t2"]).run()               # the modified ITC swing trade
    _ok(at)
    text = _text(at)
    assert "ITC" in text and "Approval #" in text and "wider stop below base" in text
    picker = at.selectbox(key=f"tr_pick_{analytics_seeded['batch']}")
    picker.set_value(analytics_seeded["t1"]).run()               # agents + debate + approval
    _ok(at)
    assert "Volume confirms the breakout" in _text(at)
    at.multiselect(key=f"tr_modes_{analytics_seeded['batch']}").set_value(["SWING"]).run()
    at.multiselect(key=f"tr_src_{analytics_seeded['batch']}").set_value(["MANUAL"]).run()
    _ok(at)
    assert "No trades match the filters." in _text(at)    # the only SWING trade is a SYSTEM trade


def test_dashboard_analytics_daywise_pick_day(analytics_seeded: dict[str, int]) -> None:
    at = _app("analytics_daywise")
    at.run()
    _ok(at)
    d1 = seed_days(datetime.now(IST))["D1"]
    at.date_input(key=f"dw_day_{analytics_seeded['batch']}").set_value(d1).run()
    _ok(at)
    assert f"{d1:%A %d %b %Y}" in _text(at)
    assert at.dataframe, "timeline table missing"


def test_dashboard_analytics_rejected_and_me_vs_system_content(analytics_seeded: dict[str, int]) -> None:
    at = _app("analytics_rejected")
    at.run()
    _ok(at)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Not traded"] == "9" and metrics["Profit missed"] == "Rs 36.00"
    assert metrics["Loss avoided"] == "Rs 30.00"
    mvs = _app("analytics_me_vs_system")
    mvs.run()
    _ok(mvs)
    metrics = {m.label: m.value for m in mvs.metric}
    assert metrics["My P&L"] == "Rs -12.50" and metrics["System alone"] == "Rs -12.00"
    assert metrics["My edge"] == "Rs -0.50" and metrics["Compared"] == "6 / 7"


def test_dashboard_analytics_batches_compare_and_diff(analytics_seeded: dict[str, int]) -> None:
    at = _app("analytics_batches")
    at.run()
    _ok(at)
    assert at.multiselect(key="bt_compare").value == [analytics_seeded["batch"], analytics_seeded["old_batch"]]
    assert "1 setting(s) differ." in _text(at)
    at.radio(key="bt_y").set_value("equity").run()
    _ok(at)
    at.selectbox(key="bt_diff_b").set_value(analytics_seeded["batch"]).run()
    _ok(at)
    assert "Pick two different batches." in _text(at)


def test_dashboard_analytics_ledger_filters(analytics_seeded: dict[str, int]) -> None:
    at = _app("analytics_ledger")
    at.run()
    _ok(at)
    assert "Running balances (22 of 22 entries)" in _text(at)
    at.multiselect(key=f"lg_types_{analytics_seeded['batch']}").set_value(["PROFIT_LOCK"]).run()
    _ok(at)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Profit locked"] == "Rs 18.39" and metrics["Refilled"] == "Rs 0.00"
    assert "Running balances (2 of 22 entries)" in _text(at)


def test_dashboard_analytics_performance_tabs(analytics_seeded: dict[str, int]) -> None:
    at = _app("analytics_performance")
    at.run()
    _ok(at)
    at.radio(key="pf_metric_strategy").set_value("profit_factor").run()
    _ok(at)
    assert "call(s)" in _text(at)
