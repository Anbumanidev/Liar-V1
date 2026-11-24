# Liar-V1 - PAPER trading system. Targets: setup, pull-models, run, dashboard, test, replay.
# Windows without make: use  .\tasks.ps1 <target>  (same targets).

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif

PROFILE ?= low
START ?=
END ?=
MODES ?= INTRADAY,SWING
SYMBOLS ?=
LLM ?= rules

.PHONY: setup pull-models run dashboard test test-all replay walk-forward init-db set-password backup health

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) main.py init-db

pull-models:
	$(PY) scripts/pull_models.py --profile $(PROFILE)

init-db:
	$(PY) main.py init-db

set-password:
	$(PY) scripts/set_password.py

run:
	$(PY) main.py engine

dashboard:
	$(PY) -m streamlit run dashboard/app.py --server.address 127.0.0.1

test:
	$(PY) -m pytest tests -q

test-all:
	RUN_NETWORK_TESTS=1 RUN_OLLAMA_TESTS=1 $(PY) -m pytest tests -q

replay:
	$(PY) main.py replay --start $(START) --end $(END) --modes $(MODES) --llm $(LLM) $(if $(SYMBOLS),--symbols $(SYMBOLS),)

walk-forward:
	$(PY) main.py walk-forward --start $(START) --end $(END) --modes $(MODES) --llm $(LLM)

backup:
	$(PY) scripts/backup_db.py

health:
	$(PY) scripts/health_check.py
