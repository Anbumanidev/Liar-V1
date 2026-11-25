"""Liar-V1 owner dashboard (section 14). PAPER TRADING ONLY.

Run (binds to localhost only)::

    .venv\\Scripts\\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1

* Login gate before ANY page is built (pages are only reachable through ``st.navigation``
  after login, and every page's ``render()`` re-checks the session - defence in depth).
* Sidebar: batch selector (filters every page; default = current batch), engine heartbeat
  badge, logout.
* Navigation is built from page modules in ``dashboard/pages``: each exposes ``TITLE``,
  ``ICON`` and ``render()``. Analytics pages (``analytics_*.py``) are registered only when
  their file exists; control pages (``control_*.py``) are always present.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402
from loguru import logger  # noqa: E402

from dashboard import services as svc  # noqa: E402
from dashboard.auth import check_session, logout_button, render_login  # noqa: E402
from dashboard.components import heartbeat_badge  # noqa: E402
from dashboard.state import DashboardContext, get_ctx, set_selected_batch  # noqa: E402
from storage.audit import record_error  # noqa: E402

PAGES_DIR = Path(__file__).resolve().parent / "pages"

#: Analytics pages (section 14.4), in display order. Registered only if the file exists.
ANALYTICS_PAGES: tuple[str, ...] = (
    "analytics_overview", "analytics_live", "analytics_ledger", "analytics_daywise", "analytics_trades",
    "analytics_rejected", "analytics_me_vs_system", "analytics_performance", "analytics_batches",
    "analytics_tax", "analytics_risk_lab",
)

#: Landing page (cockpit).
HOME_PAGE = "home"

#: User-facing navigation groups (page module names, in display order).
NAV_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Home", (HOME_PAGE,)),
    ("Trading", ("control_approvals", "control_positions", "control_manual_trade", "analytics_live")),
    ("Analytics", ("analytics_overview", "analytics_trades", "analytics_rejected", "analytics_me_vs_system",
                   "analytics_performance", "analytics_tax", "analytics_risk_lab", "analytics_ledger",
                   "analytics_daywise", "analytics_batches")),
    ("Settings", ("control_modes", "control_risk", "control_capital", "control_strategies", "control_agents",
                  "control_universe", "control_schedule", "control_notifications")),
    ("System", ("control_engine", "control_batches", "control_system")),
)
#: Control pages (section 14.3), in display order. All must exist.
CONTROL_PAGES: tuple[str, ...] = (
    "control_engine", "control_approvals", "control_positions", "control_manual_trade", "control_modes",
    "control_universe", "control_strategies", "control_agents", "control_risk", "control_capital",
    "control_schedule", "control_notifications", "control_batches", "control_system",
)


def _load_page_module(name: str) -> ModuleType | None:
    """Import ``dashboard.pages.<name>`` and check it exposes TITLE/ICON/render."""
    try:
        module = importlib.import_module(f"dashboard.pages.{name}")
    except Exception as exc:  # noqa: BLE001 - one broken page must not take the dashboard down
        record_error(None, "dashboard.app", exc, {"page": name})
        return None
    if not callable(getattr(module, "render", None)):
        logger.error(f"Page module {name} has no render()")
        return None
    return module


def page_modules(names: tuple[str, ...], *, only_existing: bool) -> list[tuple[str, ModuleType]]:
    """Load the named page modules (skipping missing files when ``only_existing``)."""
    out: list[tuple[str, ModuleType]] = []
    for name in names:
        if only_existing and not (PAGES_DIR / f"{name}.py").exists():
            continue
        module = _load_page_module(name)
        if module is not None:
            out.append((name, module))
    # Extra analytics pages dropped in later (not in the expected list) are registered too.
    if only_existing:
        for path in sorted(PAGES_DIR.glob("analytics_*.py")):
            if path.stem not in names:
                module = _load_page_module(path.stem)
                if module is not None:
                    out.append((path.stem, module))
    return out


def build_navigation() -> dict[str, list[st.Page]]:
    """Group -> pages mapping for ``st.navigation`` (Home is the default landing page)."""
    loaded: dict[str, ModuleType] = {}
    for name, module in page_modules(ANALYTICS_PAGES, only_existing=True) + page_modules(CONTROL_PAGES,
                                                                                          only_existing=False):
        loaded[name] = module
    home = _load_page_module(HOME_PAGE) if (PAGES_DIR / f"{HOME_PAGE}.py").exists() else None
    if home is not None:
        loaded[HOME_PAGE] = home
    sections: dict[str, list[st.Page]] = {}
    placed: set[str] = set()
    default_set = False
    groups = list(NAV_GROUPS)
    extra = tuple(n for n in loaded if not any(n in names for _, names in NAV_GROUPS))
    if extra:
        groups.append(("More", extra))
    for section, names in groups:
        pages: list[st.Page] = []
        for name in names:
            module = loaded.get(name)
            if module is None or name in placed:
                continue
            placed.add(name)
            pages.append(st.Page(module.render, title=str(getattr(module, "TITLE", name)),
                                 icon=getattr(module, "ICON", None), url_path=name, default=not default_set))
            default_set = True
        if pages:
            sections[section] = pages
    return sections


def render_sidebar(ctx: DashboardContext) -> None:
    """Batch selector + heartbeat badge + logout."""
    sb = st.sidebar
    sb.markdown("**Liar-V1** :orange[PAPER]")
    try:
        with ctx.session() as session:
            options = svc.batch_options(session)
            settings_row = svc.load_settings(session, ctx.caps, ctx.clock)
            session.commit()
            view = svc.engine_status(session, ctx.clock.now(), svc.heartbeat_stale_seconds(settings_row[1]))
    except Exception as exc:  # noqa: BLE001
        record_error(None, "dashboard.sidebar", exc)
        sb.error("DB unavailable - see logs.")
        logout_button(ctx)
        return
    if options:
        ids = [o.id for o in options]
        labels = {o.id: o.label for o in options}
        current = ctx.batch_id if ctx.batch_id in ids else ids[0]
        chosen = sb.selectbox("Batch", ids, index=ids.index(current), format_func=lambda i: labels[i],
                              key="sidebar_batch_select")
        if chosen != ctx.batch_id:
            set_selected_batch(chosen)
            ctx.batch_id = chosen
    else:
        sb.caption("No batches yet.")
    heartbeat_badge(view, sb)
    if view.activity:
        sb.caption(view.activity[:120])
    logout_button(ctx)


def main() -> None:
    st.set_page_config(
        page_title="Liar-V1 (PAPER)",
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    ctx = get_ctx()
    user = check_session(ctx)
    if not user:
        login = st.navigation([st.Page(lambda: render_login(ctx), title="Login", icon=":material/lock:",
                                       url_path="login")], position="hidden")
        login.run()
        return
    render_sidebar(ctx)
    nav = st.navigation(build_navigation())
    nav.run()


main()
