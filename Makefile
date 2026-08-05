.PHONY: help install install-dev data features train evaluate forecast api app test lint format clean docker-up docker-down

help:
	@echo "install       Install runtime deps"
	@echo "install-dev   Editable install with dev tooling"
	@echo "data          Download raw demand/weather/holiday data"
	@echo "features      Build the feature matrix"
	@echo "train         Train the model ladder with walk-forward validation"
	@echo "evaluate      Score models and write reports/model_comparison.csv"
	@echo "forecast      Generate the 48h forecast with prediction intervals"
	@echo "api           Run the FastAPI service"
	@echo "app           Run the Streamlit dashboard"
	@echo "test          Run pytest with coverage"
	@echo "lint / format Run ruff / black"
	@echo "docker-up     Build and start the full stack"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"
	pre-commit install || true

data:
	python scripts/download_data.py

features:
	python scripts/prepare_data.py

train:
	python scripts/train_models.py

evaluate:
	python scripts/evaluate_models.py

forecast:
	python scripts/generate_forecast.py

api:
	uvicorn api.main:app --reload --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000}

app:
	streamlit run app/streamlit_app.py --server.port $${STREAMLIT_PORT:-8501}

test:
	pytest

lint:
	ruff check src app api scripts tests

format:
	black src app api scripts tests
	ruff check --fix src app api scripts tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage

docker-up:
	docker compose up --build

docker-down:
	docker compose down
