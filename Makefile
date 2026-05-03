.PHONY: install lint test train run

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v

train:
	python -m churn_nn.train

run:
	uvicorn churn_nn.api.app:app --reload --port 8000
