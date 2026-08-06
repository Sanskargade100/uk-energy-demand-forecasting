"""Tests for the prediction pipeline: happy path + each refusal guard (mock model)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.pipelines import prediction_pipeline as pp

FEATURES = [
    "forecast_horizon", "demand_origin", "demand_lag_1", "demand_rolling_mean_48",
    "hour", "temperature_mean", "is_weekend", "is_bank_holiday",
]


class _MockModel:
    """Stands in for a fitted XGBoostForecaster."""

    def __init__(self, features):
        self.features_ = list(features)

    def predict(self, samples):
        return np.full(len(samples), 30000.0)


def _context(n_hist=400, n_future=96, origin="2025-01-08 00:00"):
    origin_ts = pd.Timestamp(origin, tz="UTC")
    # n_hist rows before origin, the origin row, then n_future rows after it.
    idx = pd.date_range(origin_ts - pd.Timedelta(minutes=30) * n_hist,
                        periods=n_hist + n_future + 1, freq="30min", tz="UTC")
    local = idx.tz_convert("Europe/London")
    nd = 30000 + 5000 * np.sin(2 * np.pi * np.arange(len(idx)) / 48)
    nd = nd.astype(float)
    nd[idx > origin_ts] = np.nan  # future demand unknown
    return pd.DataFrame(
        {
            "timestamp_utc": idx,
            "timestamp_local": local,
            "nd_mw": nd,
            "demand_lag_1": np.roll(nd, 1),
            "demand_rolling_mean_48": 30000.0,
            "hour": local.hour,
            "temperature_mean": 8.0,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_bank_holiday": 0,
        }
    ), origin_ts


def _pipeline(intervals=None, **kw):
    return pp.PredictionPipeline(
        model=_MockModel(FEATURES), feature_order=FEATURES, model_name="XGBoost",
        metadata={"target": "nd_mw", "feature_list": FEATURES}, intervals=intervals, **kw
    )


# ---- happy path ------------------------------------------------------------
def test_predict_returns_96_row_dataframe():
    fm, origin = _context()
    out = _pipeline().predict(fm, origin_time=origin)
    assert len(out) == 96
    assert list(out.columns) == pp.RESULT_COLUMNS
    assert (out["horizon_step"] == range(1, 97)).all()
    assert (out["forecast_for"] > origin).all()
    assert (out["point_forecast_mw"] == 30000.0).all()


def test_predict_as_json():
    fm, origin = _context()
    js = _pipeline().predict(fm, origin_time=origin, as_json=True)
    parsed = json.loads(js)
    assert len(parsed) == 96 and parsed[0]["model_name"] == "XGBoost"


def test_predict_with_intervals():
    from energy_forecasting.models.intervals import ConformalIntervals

    cal = pd.DataFrame(
        {"forecast_horizon": np.tile(np.arange(1, 97), 5),
         "y_pred": 30000.0,
         "y_true": 30000.0 + np.random.default_rng(0).normal(0, 500, 96 * 5)}
    )
    ci = ConformalIntervals().fit(cal)
    fm, origin = _context()
    out = _pipeline(intervals=ci).predict(fm, origin_time=origin)
    assert out["lower_95_mw"].notna().all()
    assert (out["lower_95_mw"] <= out["lower_80_mw"]).all()
    assert (out["upper_95_mw"] >= out["upper_80_mw"]).all()


# ---- refusal guards --------------------------------------------------------
def test_reject_insufficient_history():
    fm, origin = _context(n_hist=100)  # < MIN_HISTORY (336)
    with pytest.raises(pp.PredictionError) as e:
        _pipeline().predict(fm, origin_time=origin)
    assert e.value.reason == "insufficient_history"


def test_reject_stale_data():
    fm, origin = _context()
    # Blank out the most recent day of demand -> newest actual is > 2h old.
    fm.loc[fm["timestamp_utc"] > origin - pd.Timedelta(hours=6), "nd_mw"] = np.nan
    with pytest.raises(pp.PredictionError) as e:
        _pipeline().predict(fm, origin_time=origin)
    assert e.value.reason == "stale_data"


def test_reject_missing_features():
    fm, origin = _context()
    fm = fm.drop(columns=["temperature_mean"])  # a required feature
    with pytest.raises(pp.PredictionError) as e:
        _pipeline().predict(fm, origin_time=origin)
    assert e.value.reason == "missing_features"


def test_reject_feature_order_mismatch():
    reordered = FEATURES[::-1]
    with pytest.raises(pp.PredictionError) as e:
        pp.PredictionPipeline(
            model=_MockModel(FEATURES), feature_order=reordered, model_name="XGBoost",
            metadata={"target": "nd_mw"},
        )
    assert e.value.reason == "feature_order_mismatch"


def test_reject_incompatible_metadata_target():
    with pytest.raises(pp.PredictionError) as e:
        pp.PredictionPipeline(
            model=_MockModel(FEATURES), feature_order=FEATURES, model_name="XGBoost",
            metadata={"target": "something_else", "feature_list": FEATURES},
        )
    assert e.value.reason == "incompatible_metadata"


def test_reject_incompatible_metadata_feature_list():
    with pytest.raises(pp.PredictionError) as e:
        pp.PredictionPipeline(
            model=_MockModel(FEATURES), feature_order=FEATURES, model_name="XGBoost",
            metadata={"target": "nd_mw", "feature_list": FEATURES[:-1]},
        )
    assert e.value.reason == "incompatible_metadata"
