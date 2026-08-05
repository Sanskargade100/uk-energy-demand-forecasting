# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for prophet/lightgbm/xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY api ./api
COPY app ./app
COPY scripts ./scripts
COPY configs ./configs
RUN pip install -e .

EXPOSE 8000 8501

# Default: serve the API. docker-compose overrides the command per service.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
