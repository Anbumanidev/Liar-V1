"""Dashboard page modules.

Each module exposes ``TITLE`` (str), ``ICON`` (emoji or ``:material/<name>:``) and ``render()``.
``control_*.py`` are the 14 control panels (section 14.3); ``analytics_*.py`` are the analytics
pages (section 14.4) and are registered by ``dashboard/app.py`` only when their file exists.
Modules must not execute anything at import time (no Streamlit calls at module level).
"""
