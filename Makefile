PYTHON := .venv/bin/python

.PHONY: install lint test train run

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src/ tests/

test:
	$(PYTHON) -m pytest tests/ -v

train:
	$(PYTHON) -m churn_nn.train

run:
	$(PYTHON) -m uvicorn churn_nn.api.app:app --reload --port 8000
