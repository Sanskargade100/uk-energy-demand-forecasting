# UK Energy Demand Forecasting System

Forecast Great Britain's electricity demand for every 30-minute settlement period over
the next 24–48 hours, with calibrated **80%** and **95%** prediction intervals.

The target variable is **National Demand (ND)** in MW — the demand met by the GB
transmission network, as defined and published by NESO in their Historic Demand Data
under the NESO Open Data Licence.

## Problem definition

The task is a **multi-horizon point-and-interval forecast** of GB National Demand.

### Target

```
target            = ND        (NESO National Demand)
unit              = MW
frequency         = 30 minutes (half-hourly settlement periods)
forecast_horizon  = 96 settlement periods
```

There are **48 half-hour periods in a normal day**, so the horizon maps to:

```
24-hour forecast  = 48 predictions
48-hour forecast  = 96 predictions
```

(On clock-change days a settlement day has 46 or 50 periods; the code keys off the
`Europe/London` calendar rather than assuming a fixed 48.)

### Model inputs

At prediction time — the moment the forecast would have been issued — the system may use:

- Previous demand values (lags and rolling statistics up to and including the last observed period)
- Calendar information (time of day, settlement period, day of week, month, seasonal encodings)
- Bank-holiday information (GB: England, Scotland, Wales)
- Weather **forecasts** available at prediction time
- Renewable-generation **forecasts** where available

**Leakage constraint.** The system must **never** use actual future demand, nor weather or
generation *observations* that were unavailable when the prediction would have been made.
Any weather/renewable feature at a future timestamp must come from a forecast, not a
back-filled actual. Walk-forward validation enforces this: each fold only ever sees data up
to its own origin.

### Initial scope

Roughly **five years** of half-hourly data:

```
Training and validation : 2020–2024
Final test period        : 2025
```

These dates can be adjusted to match data availability, but the **final test period (2025)
stays completely untouched until model selection is finished** — it is used once, after the
winning model and hyperparameters are locked, to report unbiased performance.

## What this project demonstrates

- Time-series data collection, validation and feature engineering (calendar, weather, holiday, lag/rolling)
- Time-series EDA: trend, seasonality, missing values and anomaly detection
- A model ladder compared on the **same walk-forward splits**:
  - Seasonal naïve baseline
  - SARIMAX / Prophet (statistical)
  - XGBoost / LightGBM (machine learning)
  - LSTM / Temporal Fusion Transformer (deep learning)
- Walk-forward (rolling-origin) validation — never a random split
- Prediction intervals and explainability with SHAP
- A **Streamlit** dashboard and a **FastAPI** prediction endpoint
- Docker, automated tests and GitHub Actions CI

## Repository layout

```
configs/    YAML configuration for data, features, models, logging
data/        raw → interim → processed → external (contents git-ignored)
notebooks/   investigation only — reusable logic lives in src/
src/energy_forecasting/   the importable package (data, features, models, evaluation, pipelines, utils)
app/         Streamlit multi-page dashboard
api/         FastAPI service
scripts/     thin CLI entry points that call into src/
models/      serialized model artifacts (git-ignored)
reports/     figures, model_comparison.csv, final_report.md
tests/       pytest suite
.github/     CI workflows (tests, docker)
```

> **Notebooks are for investigation.** Any logic worth reusing must move into `src/energy_forecasting/`, not stay hidden in a notebook.

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # or: pip install -r requirements.txt
cp .env.example .env           # add API keys

# 2. Data → features → train → forecast
make data
make features
make train
make forecast

# 3. Serve
make api         # FastAPI at http://localhost:8000/docs
make app         # Streamlit at http://localhost:8501
```

Or run the whole stack with Docker:

```bash
docker compose up --build
```

## Data source

NESO (National Energy System Operator) Historic Demand Data — half-hourly GB demand
and embedded renewable generation. See `data/README.md` for the exact dataset,
columns and licence terms.

## License

Code released under the MIT License (see `LICENSE`). NESO demand data is used under the
NESO Open Data Licence and is **not** redistributed in this repository.
