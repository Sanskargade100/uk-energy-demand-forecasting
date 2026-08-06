"""XGBoost forecaster using the horizon-feature framing.

One gradient-boosted model serves all 96 horizons: each training row is an
(origin, horizon) pair with demand-history features known at the origin, calendar
and weather for the forecasted timestamp, and ``forecast_horizon`` as a feature
(see :mod:`energy_forecasting.features.supervised`).

Discipline
----------
* **Validation-based early stopping** — fit stops when the held-out (chronological)
  validation RMSE stops improving.
* **Tuning** (:func:`tune`) explores ``max_depth``, ``min_child_weight``,
  ``learning_rate``, ``subsample`` and ``colsample_bytree`` on train→validation
  only. The **final test period is never used for tuning or early stopping.**

Artifacts (saved separately)
----------------------------
* ``models/xgboost.joblib``          — the fitted forecaster.
* ``models/xgboost_features.json``   — the ordered feature list.
* ``models/xgboost_metadata.json``   — training dates, features, target, params,
  validation score, package versions, training timestamp, git commit hash.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from ..settings import MODELS_DIR, PROJECT_ROOT
from ..features.supervised import make_horizon_samples, split_feature_groups

logger = get_logger(__name__)

TARGET = "nd_mw"
HORIZON = 96
_NON_FEATURE = {"origin_time", "target_time", "y"}

DEFAULT_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

# Parameters that may be tuned (train/validation only).
TUNABLE = ["max_depth", "min_child_weight", "learning_rate", "subsample", "colsample_bytree"]

MODEL_PATH = MODELS_DIR / "xgboost.joblib"
FEATURES_PATH = MODELS_DIR / "xgboost_features.json"
METADATA_PATH = MODELS_DIR / "xgboost_metadata.json"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without xgboost)
# ---------------------------------------------------------------------------
def feature_columns(samples: pd.DataFrame) -> list[str]:
    """Feature columns for the model = everything except identifiers and target."""
    return [c for c in samples.columns if c not in _NON_FEATURE]


def make_origin_samples(
    fm: pd.DataFrame, origin_time, horizons=range(1, HORIZON + 1)
) -> pd.DataFrame:
    """Horizon-feature rows for a single forecast origin (no ``y``)."""
    fm = fm.sort_values("timestamp_utc").reset_index(drop=True)
    origin_cols, target_cols = split_feature_groups(fm)
    matches = fm.index[fm["timestamp_utc"] == pd.Timestamp(origin_time)]
    if len(matches) == 0:
        raise KeyError(f"origin_time {origin_time} not found in feature matrix.")
    o = int(matches[0])
    n = len(fm)

    rows = []
    for h in horizons:
        tgt = o + h
        if tgt >= n:
            continue
        row = {
            "origin_time": fm.at[o, "timestamp_utc"],
            "target_time": fm.at[tgt, "timestamp_utc"],
            "forecast_horizon": h,
            "demand_origin": fm.at[o, TARGET],
        }
        for c in origin_cols:
            row[c] = fm.at[o, c]
        for c in target_cols:
            row[c] = fm.at[tgt, c]
        rows.append(row)
    return pd.DataFrame(rows)


def get_git_commit() -> str | None:
    """Current git commit hash (or None if unavailable)."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("xgboost", "numpy", "pandas", "sklearn"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[name] = "not-installed"
    return versions


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------
class XGBoostForecaster:
    def __init__(
        self,
        params: dict | None = None,
        horizons=range(1, HORIZON + 1),
        origin_step: int = 1,
        early_stopping_rounds: int = 50,
    ):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.horizons = list(horizons)
        self.origin_step = origin_step
        self.early_stopping_rounds = early_stopping_rounds
        self.model_ = None
        self.features_: list[str] = []
        self.val_rmse_: float | None = None
        self.train_start_ = None
        self.train_end_ = None

    def fit(self, train_fm: pd.DataFrame, val_fm: pd.DataFrame | None = None) -> "XGBoostForecaster":
        import xgboost as xgb

        train = make_horizon_samples(train_fm, self.horizons, origin_step=self.origin_step)
        train = train.dropna(subset=["y"])
        self.features_ = feature_columns(train)
        x_tr, y_tr = train[self.features_], train["y"]

        eval_set = None
        if val_fm is not None:
            val = make_horizon_samples(val_fm, self.horizons, origin_step=self.origin_step)
            val = val.dropna(subset=["y"])
            eval_set = [(val[self.features_], val["y"])]

        kwargs = dict(self.params, eval_metric="rmse")
        if eval_set is not None:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds
        self.model_ = xgb.XGBRegressor(**kwargs)
        self.model_.fit(x_tr, y_tr, eval_set=eval_set, verbose=False)

        self.train_start_ = train_fm["timestamp_utc"].min()
        self.train_end_ = train_fm["timestamp_utc"].max()
        if eval_set is not None:
            self.val_rmse_ = _rmse(eval_set[0][1], self.model_.predict(eval_set[0][0]))
            logger.info("XGBoost fitted | val RMSE=%.1f MW | %d features", self.val_rmse_, len(self.features_))
        return self

    def predict(self, samples: pd.DataFrame) -> np.ndarray:
        """Predict from a horizon-sample frame (must contain the fitted features)."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model_.predict(samples[self.features_])

    def forecast_origin(self, fm: pd.DataFrame, origin_time) -> pd.Series:
        """Produce a 96-step point forecast for a single origin."""
        samples = make_origin_samples(fm, origin_time, self.horizons)
        preds = self.predict(samples)
        return pd.Series(preds, index=pd.DatetimeIndex(samples["target_time"]), name="point_forecast_mw")

    # ---- persistence ------------------------------------------------------
    def metadata(self) -> dict:
        return {
            "model": "XGBoost",
            "target": TARGET,
            "training_start": str(self.train_start_),
            "training_end": str(self.train_end_),
            "n_features": len(self.features_),
            "feature_list": self.features_,
            "params": self.params,
            "best_iteration": getattr(self.model_, "best_iteration", None),
            "validation_rmse": self.val_rmse_,
            "package_versions": _package_versions(),
            "training_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": get_git_commit(),
        }

    def save(
        self,
        model_path: Path | None = None,
        features_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> tuple[Path, Path, Path]:
        import joblib

        mp = Path(model_path) if model_path else MODEL_PATH
        fp = Path(features_path) if features_path else FEATURES_PATH
        jp = Path(metadata_path) if metadata_path else METADATA_PATH
        mp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, mp)
        fp.write_text(json.dumps(self.features_, indent=2), encoding="utf-8")
        jp.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
        logger.info("Saved XGBoost -> %s (+ features %s, metadata %s)", mp, fp, jp)
        return mp, fp, jp

    @classmethod
    def load(cls, model_path: Path | None = None) -> "XGBoostForecaster":
        import joblib

        return joblib.load(Path(model_path) if model_path else MODEL_PATH)


# ---------------------------------------------------------------------------
# Tuning (train/validation only — never the test period)
# ---------------------------------------------------------------------------
def tune(
    train_fm: pd.DataFrame,
    val_fm: pd.DataFrame,
    n_iter: int = 20,
    origin_step: int = 4,
    param_distributions: dict | None = None,
    seed: int = 42,
) -> dict:
    """Randomised search over the tunable params; select by validation RMSE.

    Returns ``{"best_params", "best_val_rmse", "trials"}``. Uses ``origin_step`` to
    subsample origins for speed during search.
    """
    from sklearn.model_selection import ParameterSampler

    space = param_distributions or {
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 3, 5, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    candidates = list(ParameterSampler(space, n_iter=n_iter, random_state=seed))

    trials, best = [], None
    for cand in candidates:
        model = XGBoostForecaster(params=cand, origin_step=origin_step)
        model.fit(train_fm, val_fm)
        trials.append({"params": cand, "val_rmse": model.val_rmse_})
        if best is None or (model.val_rmse_ is not None and model.val_rmse_ < best["val_rmse"]):
            best = {"params": cand, "val_rmse": model.val_rmse_}
        logger.info("tune trial val RMSE=%.1f params=%s", model.val_rmse_ or float("nan"), cand)

    return {"best_params": best["params"], "best_val_rmse": best["val_rmse"], "trials": trials}
