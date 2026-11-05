# Prefer the project venv when present (needs `pip install -e ".[dev]"`).
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: report test lint

report:
	PYTHONPATH=src $(PY) -m futures_tca.report --out reports

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
