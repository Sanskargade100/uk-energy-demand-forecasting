"""Evaluate all models on the untouched test year and write the comparison table.

Runs the seasonal-naive baselines and the trained XGBoost (and optionally SARIMAX)
at daily origins across the test period, scores them on identical origins/horizons,
and writes:

* ``reports/model_comparison.csv``  — MAE / RMSE / WAPE / Peak MAE / training time.
* ``reports/interval_coverage.csv`` — achieved 80%/95% coverage + width (XGBoost).
* ``reports/error_by_horizon.csv``  — MAE vs horizon for the best model.

Usage
-----
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --per-day 4 --with-sarimax
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from energy_forecasting.evaluation import runner
from energy_forecasting.evaluation.backtesting import forecast_origins
from energy_forecasting.evaluation.metrics import error_by
from energy_forecasting.models.intervals import evaluate_intervals
from energy_forecasting.logging_config import get_logger, setup_logging
from energy_forecasting.models.xgboost_model import XGBoostForecaster
from energy_forecasting.settings import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, load_config

logger = get_logger(__name__)


def _test_range():
    try:
        d = load_config("data")["dates"]
        return d["final_test_start"], "2026-01-01"
    except Exception:  # noqa: BLE001
        return "2025-01-01", "2026-01-01"


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate models on the test year.")
    parser.add_argument("--per-day", type=int, default=1, help="Forecast origins per day.")
    parser.add_argument("--with-sarimax", action="store_true")
    args = parser.parse_args(argv)

    processed = pd.read_parquet(PROCESSED_DIR / "energy_demand_30min.parquet")
    fm = pd.read_parquet(PROCESSED_DIR / "feature_matrix.parquet")
    actuals = runner.demand_series(processed)

    test_start, test_end = _test_range()
    origins = forecast_origins(test_start, test_end, per_day=args.per_day)
    logger.info("Evaluating on %d origins from %s to %s.", len(origins), test_start, test_end)

    # Forecast functions per model (all scored on the same origins).
    fns = runner.baseline_forecast_fns(actuals, horizon=96)

    training_seconds: dict[str, float] = {}
    xgb = XGBoostForecaster.load()
    fns["xgboost"] = lambda origin: xgb.forecast_origin(fm, origin)
    try:
        meta = json.loads((MODELS_DIR / "xgboost_metadata.json").read_text())
        training_seconds["xgboost"] = None
    except Exception:  # noqa: BLE001
        pass

    if args.with_sarimax:
        logger.info("SARIMAX evaluation enabled (slower).")
        # SARIMAX predict needs a future exog frame; handled by its own predict().
        sx = joblib.load(MODELS_DIR / "sarimax.joblib")

        def sarimax_fn(origin):
            future = fm[(fm["timestamp_utc"] > origin)].head(96)
            if len(future) < 96:
                return None
            preds = sx.predict(future.set_index("timestamp_utc"))
            return preds["point_forecast_mw"]

        fns["sarimax"] = sarimax_fn

    # Collect predictions and score.
    preds = runner.collect_predictions(fns, origins, actuals, horizon=96)
    table = runner.score_models(preds, training_seconds)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    logger.info("Model comparison:\n%s", table.to_string(index=False))

    # Interval coverage for XGBoost (using calibrated conformal intervals).
    ci_path = MODELS_DIR / "conformal_intervals.joblib"
    if ci_path.exists():
        ci = joblib.load(ci_path)
        xgb_preds = preds[preds["model"] == "xgboost"].copy()
        banded = ci.apply(
            xgb_preds.rename(columns={"y_pred": "point_forecast_mw"})
        )
        for lvl in ("80", "95"):
            banded[f"lower_{lvl}_mw"] = banded[f"lower_{lvl}"]
            banded[f"upper_{lvl}_mw"] = banded[f"upper_{lvl}"]
        banded["y_true"] = xgb_preds["y_true"].to_numpy()
        cov = evaluate_intervals(banded.rename(columns={
            "lower_80": "lower_80", "upper_80": "upper_80", "lower_95": "lower_95", "upper_95": "upper_95"
        }))
        cov.to_csv(REPORTS_DIR / "interval_coverage.csv", index=False)
        logger.info("Interval coverage (XGBoost):\n%s", cov.to_string(index=False))

    # Error by horizon for the best model.
    best = table.iloc[0]["model"]
    best_eh = error_by(preds[preds["model"] == best], "forecast_horizon")
    best_eh.to_csv(REPORTS_DIR / "error_by_horizon.csv", index=False)
    logger.info("Best model: %s (MAE=%.1f MW). Wrote reports.", best, table.iloc[0]["mae"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
