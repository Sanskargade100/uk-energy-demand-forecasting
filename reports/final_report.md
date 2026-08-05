# UK Energy Demand Forecasting — Final Report

_Draft. Populated by `scripts/evaluate_models.py` and manual write-up._

## 1. Objective
Forecast GB National Demand (ND, MW) for every 30-minute period over the next 24–48 hours,
with 80% and 95% prediction intervals.

## 2. Data
Source, coverage, resolution, missing-value and anomaly summary.

## 3. Exploratory analysis
Trend, daily/weekly/annual seasonality, holiday effects, weather sensitivity.

## 4. Features
Calendar, holiday, weather (population-weighted), lag and rolling features.

## 5. Models compared
Seasonal naïve → SARIMAX/Prophet → XGBoost/LightGBM → LSTM/TFT, on identical
walk-forward splits. See `reports/model_comparison.csv`.

## 6. Validation
Walk-forward (rolling-origin) design and rationale.

## 7. Prediction intervals
Method and empirical coverage vs. nominal (80% / 95%).

## 8. Explainability
SHAP global and local findings for the winning model.

## 9. Deployment
Streamlit dashboard, FastAPI endpoint, Docker, CI.

## 10. Limitations & next steps
