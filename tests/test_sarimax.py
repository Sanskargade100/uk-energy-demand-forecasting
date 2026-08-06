"""Tests for the SARIMAX model — Fourier/exog helpers plus a guarded fit smoke test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.models import sarimax_model as sm


def test_time_index_steps_from_epoch():
    idx = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    steps = sm.time_index_steps(idx, epoch="2021-01-01", freq="1h")
    assert list(steps) == [0, 1, 2, 3]


def test_fourier_terms_values_and_columns():
    idx = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    f = sm.fourier_terms(idx, period_steps=4, n_harmonics=1, epoch="2021-01-01", freq="1h", name="weekly")
    assert list(f.columns) == ["weekly_sin_1", "weekly_cos_1"]
    # t=[0,1,2,3], period 4: sin -> [0,1,0,-1], cos -> [1,0,-1,0]
    assert np.allclose(f["weekly_sin_1"].to_numpy(), [0, 1, 0, -1], atol=1e-9)
    assert np.allclose(f["weekly_cos_1"].to_numpy(), [1, 0, -1, 0], atol=1e-9)


def test_fourier_phase_consistent_across_indices():
    epoch = "2021-01-01"
    a = pd.date_range("2021-01-01", periods=10, freq="1h", tz="UTC")
    b = pd.date_range("2021-01-01 05:00", periods=10, freq="1h", tz="UTC")
    fa = sm.fourier_terms(a, 24, 2, epoch, "1h", "daily")
    fb = sm.fourier_terms(b, 24, 2, epoch, "1h", "daily")
    # Overlapping timestamps must have identical Fourier values.
    common = a.intersection(b)
    assert np.allclose(fa.loc[common].to_numpy(), fb.loc[common].to_numpy())


def test_config_defaults_and_seasonal_period():
    f = sm.SARIMAXForecaster()
    assert f.config.aggregate == "hourly"
    assert f.freq == "1h"
    assert f.config.seasonal_order[-1] == 24  # daily period for hourly data
    assert f.config.exog_cols == ["temperature_mean", "is_bank_holiday", "is_weekend"]


def _synthetic(n_days=45):
    idx = pd.date_range("2024-01-01", periods=n_days * 48, freq="30min", tz="UTC")
    t = np.arange(len(idx))
    daily = 5000 * np.sin(2 * np.pi * t / 48)
    weekly = 2000 * np.sin(2 * np.pi * t / 336)
    demand = 30000 + daily + weekly + np.random.default_rng(0).normal(0, 300, len(idx))
    local = idx.tz_convert("Europe/London")
    return pd.DataFrame(
        {
            "timestamp_utc": idx,
            "nd_mw": demand,
            "temperature_mean": 10 + 5 * np.sin(2 * np.pi * t / 336),
            "is_bank_holiday": 0,
            "is_weekend": (local.dayofweek >= 5).astype(int),
        }
    )


def test_fit_predict_and_save(tmp_path):
    pytest.importorskip("statsmodels")
    df = _synthetic(45)
    train = df.iloc[:-96].copy()
    future = df.iloc[-96:].copy()

    model = sm.SARIMAXForecaster(sm.SARIMAXConfig(order=(1, 0, 1)))
    model.fit(train)
    preds = model.predict(future)

    assert len(preds) == 96
    assert {"point_forecast_mw", "lower_80_mw", "upper_80_mw", "lower_95_mw", "upper_95_mw"} <= set(
        preds.columns
    )
    # 95% band must contain the 80% band.
    assert (preds["lower_95_mw"] <= preds["lower_80_mw"]).all()
    assert (preds["upper_95_mw"] >= preds["upper_80_mw"]).all()

    mp, jp = model.save(tmp_path / "sarimax.joblib", tmp_path / "sarimax_metadata.json")
    assert mp.exists() and jp.exists()
    import json

    meta = json.loads(jp.read_text())
    assert meta["model"] == "SARIMAX" and "limitations" in meta
