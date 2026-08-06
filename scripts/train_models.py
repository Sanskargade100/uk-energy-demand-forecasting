"""Train models on the real feature matrix and save artifacts.

Trains XGBoost on train+validation (chronological), calibrates conformal
intervals on the validation year, and saves both. Baselines need no training.
SARIMAX (optional, ``--with-sarimax``) is fit on a recent window.

The **final test year is never used here** — only for evaluation afterwards.

Usage
-----
    python scripts/train_models.py                 # XGBoost + conformal intervals
    python scripts/train_models.py --with-sarimax
    python scripts/train_models.py --origin-step 2 # subsample origins (faster/smaller)
"""

from __future__ import annotations

import argparse
import datetime as dt

import joblib
import pandas as pd

from energy_forecasting.features.build_features import build_feature_matrix, FEATURES_FILE
from energy_forecasting.logging_config import get_logger, setup_logging
from energy_forecasting.models.intervals import ConformalIntervals
from energy_forecasting.models.xgboost_model import XGBoostForecaster, make_horizon_samples
from energy_forecasting.settings import MODELS_DIR, PROCESSED_DIR, load_config

logger = get_logger(__name__)


def _splits():
    try:
        d = load_config("data")["dates"]
        train = ("2020-01-01", d["final_test_start"])  # everything before final test
        val = ("2024-01-01", d["final_test_start"])    # last full year for early stop/calibration
        test = (d["final_test_start"], "2026-01-01")
    except Exception:  # noqa: BLE001
        train, val, test = ("2020-01-01", "2024-01-01"), ("2024-01-01", "2025-01-01"), ("2025-01-01", "2026-01-01")
    # Train excludes val for early stopping; combine for the final refit if desired.
    return train, val, test


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Train models on the feature matrix.")
    parser.add_argument("--origin-step", type=int, default=2, help="Subsample training origins.")
    parser.add_argument("--with-sarimax", action="store_true")
    args = parser.parse_args(argv)

    # Feature matrix (build if missing).
    if FEATURES_FILE.exists():
        fm = pd.read_parquet(FEATURES_FILE)
    else:
        logger.info("Feature matrix missing; building it now.")
        fm = build_feature_matrix(save=True)

    (tr0, tr1), (va0, va1), _ = _splits()
    train = fm[(fm["timestamp_utc"] >= pd.Timestamp(tr0, tz="UTC")) & (fm["timestamp_utc"] < pd.Timestamp(va0, tz="UTC"))]
    val = fm[(fm["timestamp_utc"] >= pd.Timestamp(va0, tz="UTC")) & (fm["timestamp_utc"] < pd.Timestamp(va1, tz="UTC"))]
    logger.info("Train rows=%d, Val rows=%d", len(train), len(val))

    # --- XGBoost ---
    xgb = XGBoostForecaster(origin_step=args.origin_step)
    xgb.fit(train, val)
    xgb.save()
    logger.info("XGBoost saved. Val RMSE=%.1f MW", xgb.val_rmse_ or float("nan"))

    # --- Conformal intervals on validation residuals ---
    val_samples = make_horizon_samples(val, range(1, 97), origin_step=args.origin_step).dropna(subset=["y"])
    val_samples["y_pred"] = xgb.model_.predict(val_samples[xgb.features_])
    val_samples = val_samples.rename(columns={"y": "y_true"})
    ci = ConformalIntervals(by="group").fit(val_samples)
    joblib.dump(ci, MODELS_DIR / "conformal_intervals.joblib")
    logger.info("Conformal intervals calibrated on %d validation samples.", len(val_samples))

    # --- SARIMAX (optional) ---
    if args.with_sarimax:
        from energy_forecasting.models.sarimax_model import SARIMAXForecaster

        processed = pd.read_parquet(PROCESSED_DIR / "energy_demand_30min.parquet")
        recent = processed[processed["timestamp_utc"] >= pd.Timestamp(va1, tz="UTC") - pd.Timedelta(days=90)]
        recent = recent[recent["timestamp_utc"] < pd.Timestamp(va1, tz="UTC")]
        sx = SARIMAXForecaster().fit(recent)
        sx.save()
        logger.info("SARIMAX fitted on %d recent obs and saved.", len(recent))

    logger.info("Training complete at %s", dt.datetime.now().isoformat(timespec="seconds"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
