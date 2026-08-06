"""Production prediction pipeline: turn a trained model into a 48h forecast.

Steps
-----
1. Load the selected model.
2. Load model metadata and feature order.
3. Retrieve recent demand.               (supplied in the context feature matrix)
4. Retrieve the latest weather forecast.  (supplied in the context feature matrix)
5. Retrieve calendar and holiday info.    (supplied in the context feature matrix)
6. Construct features for the next 96 periods.
7. Generate point forecasts.
8. Generate prediction intervals (conformal).
9. Save predictions to SQL.
10. Return a DataFrame or JSON response.

The context feature matrix (from ``build_features``) is expected to span
``[origin - history, origin + 96]``: historical rows carry observed demand, future
rows carry deterministic calendar plus the weather *forecast*.

Refusal guards (raise :class:`PredictionError`)
-----------------------------------------------
* ``missing_features``      — a required feature is absent from the built samples.
* ``feature_order_mismatch``— metadata/feature order differs from the model's.
* ``stale_data``            — most recent demand is older than ``max_data_age``.
* ``incompatible_metadata`` — metadata target/feature list is inconsistent.
* ``insufficient_history``  — fewer than ``min_history`` observations before origin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from ..models.intervals import ConformalIntervals
from ..models.xgboost_model import make_origin_samples
from ..settings import MODELS_DIR

logger = get_logger(__name__)

HORIZON = 96
MIN_HISTORY = 336  # one week of half-hours (longest lag/rolling window)
DEFAULT_MAX_DATA_AGE = pd.Timedelta(hours=2)
TARGET = "nd_mw"

_INTERVAL_RENAME = {
    "lower_80": "lower_80_mw",
    "upper_80": "upper_80_mw",
    "lower_95": "lower_95_mw",
    "upper_95": "upper_95_mw",
}
RESULT_COLUMNS = [
    "forecast_created_at",
    "forecast_for",
    "horizon_step",
    "model_name",
    "point_forecast_mw",
    "lower_80_mw",
    "upper_80_mw",
    "lower_95_mw",
    "upper_95_mw",
    "model_version",
]


class PredictionError(RuntimeError):
    """Raised when the pipeline refuses to predict. Carries a machine ``reason``."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(f"[{reason}] {message}")


def _to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class PredictionPipeline:
    def __init__(
        self,
        model,
        feature_order: list[str],
        model_name: str,
        model_version: str = "0.1.0",
        metadata: dict | None = None,
        intervals: ConformalIntervals | None = None,
        horizon: int = HORIZON,
        min_history: int = MIN_HISTORY,
        max_data_age: pd.Timedelta = DEFAULT_MAX_DATA_AGE,
    ):
        self.model = model
        self.feature_order = list(feature_order)
        self.model_name = model_name
        self.model_version = model_version
        self.metadata = metadata or {}
        self.intervals = intervals
        self.horizon = horizon
        self.min_history = min_history
        self.max_data_age = max_data_age
        self._check_metadata()

    # ---- steps 1-2: load --------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        model_path: Path | None = None,
        features_path: Path | None = None,
        metadata_path: Path | None = None,
        intervals_path: Path | None = None,
        **kwargs,
    ) -> "PredictionPipeline":
        """Load model, feature order, metadata (and optional intervals) from disk."""
        import joblib

        mp = Path(model_path) if model_path else MODELS_DIR / "xgboost.joblib"
        fp = Path(features_path) if features_path else MODELS_DIR / "xgboost_features.json"
        jp = Path(metadata_path) if metadata_path else MODELS_DIR / "xgboost_metadata.json"

        model = joblib.load(mp)
        feature_order = json.loads(fp.read_text()) if fp.exists() else list(model.features_)
        metadata = json.loads(jp.read_text()) if jp.exists() else {}
        intervals = None
        if intervals_path and Path(intervals_path).exists():
            intervals = joblib.load(intervals_path)

        return cls(
            model=model,
            feature_order=feature_order,
            model_name=metadata.get("model", "XGBoost"),
            model_version=(metadata.get("git_commit") or "0.1.0"),
            metadata=metadata,
            intervals=intervals,
            **kwargs,
        )

    # ---- guards -----------------------------------------------------------
    def _check_metadata(self) -> None:
        if self.metadata:
            if self.metadata.get("target", TARGET) != TARGET:
                raise PredictionError(
                    "incompatible_metadata",
                    f"metadata target is {self.metadata.get('target')!r}, expected {TARGET!r}",
                )
            meta_features = self.metadata.get("feature_list")
            if meta_features is not None and list(meta_features) != self.feature_order:
                raise PredictionError(
                    "incompatible_metadata", "metadata feature_list differs from features file"
                )
        model_features = getattr(self.model, "features_", None)
        if model_features is not None and list(model_features) != self.feature_order:
            raise PredictionError(
                "feature_order_mismatch", "feature order differs from the trained model"
            )

    def _check_context(self, fm: pd.DataFrame, origin: pd.Timestamp) -> None:
        hist = fm[(fm["timestamp_utc"] <= origin) & fm[TARGET].notna()]
        if len(hist) < self.min_history:
            raise PredictionError(
                "insufficient_history",
                f"need >= {self.min_history} historical observations, have {len(hist)}",
            )
        age = origin - hist["timestamp_utc"].max()
        if age > self.max_data_age:
            raise PredictionError(
                "stale_data", f"most recent demand is {age} old (limit {self.max_data_age})"
            )

    # ---- steps 6-10: predict ---------------------------------------------
    def predict(
        self,
        context_fm: pd.DataFrame,
        origin_time=None,
        engine=None,
        as_json: bool = False,
    ):
        """Run the full pipeline for one origin. Returns a DataFrame or JSON string."""
        fm = context_fm.sort_values("timestamp_utc").reset_index(drop=True)
        origin = _to_utc(origin_time) if origin_time is not None else fm[
            fm[TARGET].notna()
        ]["timestamp_utc"].max()

        self._check_context(fm, origin)  # stale_data / insufficient_history

        samples = make_origin_samples(fm, origin, range(1, self.horizon + 1))
        if len(samples) == 0:
            raise PredictionError("insufficient_history", "no future rows available to forecast")

        missing = [f for f in self.feature_order if f not in samples.columns]
        if missing:
            raise PredictionError("missing_features", f"missing required features: {missing}")

        point = np.asarray(self.model.predict(samples), dtype=float)  # step 7

        result = pd.DataFrame(
            {
                "forecast_created_at": origin,
                "forecast_for": pd.DatetimeIndex(samples["target_time"]),
                "horizon_step": samples["forecast_horizon"].to_numpy(),
                "model_name": self.model_name,
                "point_forecast_mw": point,
                "model_version": self.model_version,
            }
        )

        # step 8: intervals
        if self.intervals is not None:
            band_input = pd.DataFrame(
                {"forecast_horizon": samples["forecast_horizon"].to_numpy(), "point_forecast_mw": point}
            )
            banded = self.intervals.apply(band_input).rename(columns=_INTERVAL_RENAME)
            for col in _INTERVAL_RENAME.values():
                result[col] = banded[col].to_numpy()
        else:
            for col in _INTERVAL_RENAME.values():
                result[col] = np.nan

        result = result[RESULT_COLUMNS]

        # step 9: persist
        if engine is not None:
            from ..data.database import insert_predictions

            insert_predictions(result, engine)
            logger.info("Saved %d predictions to SQL.", len(result))

        logger.info("Forecast issued at %s for %d periods (%s).", origin, len(result), self.model_name)
        return result.to_json(orient="records", date_format="iso") if as_json else result


def build_context(processed_window: pd.DataFrame) -> pd.DataFrame:
    """Convenience: turn a processed window into the featured context matrix."""
    from ..features.build_features import build_features

    return build_features(processed_window)
