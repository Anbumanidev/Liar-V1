"""Owner-only web dashboard (FastAPI + plain HTML/CSS/JS). PAPER TRADING ONLY.

The web process never trades: every write goes through the control-command queue
(``control.commands.enqueue_command``) that the engine executes and the risk manager vets,
or through versioned settings (``config.store.apply_changes`` + ``RELOAD_SETTINGS``).

Entry points: ``python -m webui`` or ``webui.server.run(host="127.0.0.1", port=8600)``.
"""
