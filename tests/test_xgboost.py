"""Tests for the XGBoost forecaster (pure helpers + guarded fit smoke)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.models import xgboost_model as xgm


def _fm(n=500):
    ts = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    local = ts.tz_convert("Europe/London")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "timestamp_local": local,
            "settlement_date": local.normalize().tz_localize(None),
            "nd_mw": 30000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 48) + rng.normal(0, 200, n),
            "demand_lag_1": np.nan,
            "demand_lag_48": np.nan,
            "demand_rolling_mean_48": np.nan,
            "hour": local.hour,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_bank_holiday": 0,
            "temperature_mean": 10 + 3 * np.sin(2 * np.pi * np.arange(n) / 336),
            "forecast_horizon_placeholder": 0,  # ensure only real horizon column is used
        }
    ).drop(columns=["forecast_horizon_placeholder"])


def test_feature_columns_excludes_identifiers_and_target():
    samples = pd.DataFrame(
        {"origin_time": [1], "target_time": [2], "y": [3], "forecast_horizon": [1],
         "demand_origin": [4], "demand_lag_1": [5], "hour": [6], "temperature_mean": [7]}
    )
    cols = xgm.feature_columns(samples)
    assert "y" not in cols and "origin_time" not in cols and "target_time" not in cols
    assert {"forecast_horizon", "demand_origin", "demand_lag_1", "hour", "temperature_mean"} <= set(cols)


def test_make_origin_samples_alignment():
    fm = _fm(200)
    origin = fm["timestamp_utc"].iloc[100]
    s = xgm.make_origin_samples(fm, origin, horizons=range(1, 5))
    assert len(s) == 4
    # target_time = origin + h*30min; demand_origin is demand at the origin.
    for _, row in s.iterrows():
        step = int((row["target_time"] - row["origin_time"]) / pd.Timedelta(minutes=30))
        assert step == row["forecast_horizon"]
    assert (s["demand_origin"] == fm.loc[fm["timestamp_utc"] == origin, "nd_mw"].iloc[0]).all()
    assert "y" not in s.columns  # prediction rows have no target


def test_get_git_commit_returns_str_or_none():
    commit = xgm.get_git_commit()
    assert commit is None or isinstance(commit, str)


def test_default_params_match_spec():
    p = xgm.DEFAULT_PARAMS
    assert p["n_estimators"] == 1000 and p["learning_rate"] == 0.03
    assert p["max_depth"] == 8 and p["min_child_weight"] == 5
    assert p["subsample"] == 0.8 and p["colsample_bytree"] == 0.8
    assert p["reg_alpha"] == 0.1 and p["reg_lambda"] == 1.0 and p["random_state"] == 42
    assert set(xgm.TUNABLE) == {
        "max_depth", "min_child_weight", "learning_rate", "subsample", "colsample_bytree"
    }


def test_fit_forecast_and_save(tmp_path):
    pytest.importorskip("xgboost")
    fm = _fm(500)
    train = fm.iloc[:400].copy()
    val = fm.iloc[400:].copy()

    model = xgm.XGBoostForecaster(
        params={"n_estimators": 50}, horizons=range(1, 13), early_stopping_rounds=10
    )
    model.fit(train, val)
    assert model.features_ and model.val_rmse_ is not None

    origin = train["timestamp_utc"].iloc[-1]
    fc = model.forecast_origin(fm, origin)
    assert len(fc) == 12
    assert (fc.index > origin).all()

    mp, fp, jp = model.save(
        tmp_path / "xgboost.joblib", tmp_path / "xgboost_features.json", tmp_path / "xgboost_metadata.json"
    )
    assert mp.exists() and fp.exists() and jp.exists()
    meta = json.loads(jp.read_text())
    for key in ["training_start", "training_end", "feature_list", "target", "params",
                "validation_rmse", "package_versions", "training_timestamp", "git_commit"]:
        assert key in meta
    assert json.loads(fp.read_text()) == model.features_
