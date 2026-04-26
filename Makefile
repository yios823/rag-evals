.PHONY: setup ingest eval dashboard demo test lint clean

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev,dashboard,app,observability]"
	@echo "✓ setup complete — copy .env.example to .env and add ANTHROPIC_API_KEY"

ingest:
	$(PY) -m rag_evals.ingest --corpus data/corpus --out data/index

eval:
	$(PY) -m rag_evals.eval.run --golden data/golden.jsonl --out data/results.json

dashboard:
	$(PY) dashboards/render.py --results data/results.json --out dashboards/build

demo:
	$(PY) app/gradio_app.py

test:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check src tests

clean:
	rm -rf data/index data/results.json dashboards/build .pytest_cache .ruff_cache
