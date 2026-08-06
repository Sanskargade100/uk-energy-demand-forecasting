"""Tests for the LSTM forecaster: scalers, windowing, and a guarded fit smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.models import lstm_model as lm


def _df(n=600):
    ts = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    local = ts.tz_convert("Europe/London")
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "timestamp_local": local,
            "settlement_date": local.normalize().tz_localize(None),
            "nd_mw": 30000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 48),
            "temperature_mean": 10 + 3 * np.sin(2 * np.pi * np.arange(n) / 336),
            "is_weekend": (local.dayofweek >= 5).astype(int),
        }
    )


# ---- scalers ---------------------------------------------------------------
def test_feature_scaler_fit_transform_inverse():
    X = np.arange(24, dtype=float).reshape(4, 3, 2)
    sc = lm.FeatureScaler().fit(X)
    Z = sc.transform(X)
    assert np.allclose(Z.mean(axis=(0, 1)), 0, atol=1e-9)
    assert np.allclose(sc.inverse_transform(Z), X, atol=1e-9)


def test_series_scaler_roundtrip():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    sc = lm.SeriesScaler().fit(x)
    assert np.isclose(sc.transform(x).mean(), 0.0, atol=1e-9)
    assert np.allclose(sc.inverse_transform(sc.transform(x)), x)


def test_scalers_fit_on_train_only():
    train = np.array([0.0, 10.0])
    sc = lm.SeriesScaler().fit(train)
    # Stats come from train; transforming unseen values uses train mean/std.
    assert sc.mean_ == 5.0
    z = sc.transform(np.array([100.0]))
    assert z[0] == (100.0 - 5.0) / sc.std_


# ---- feature ordering / windowing -----------------------------------------
def test_default_feature_columns_target_first():
    cols = lm.default_feature_columns(_df())
    assert cols[0] == "nd_mw"
    assert "timestamp_utc" not in cols and "settlement_date" not in cols
    assert "temperature_mean" in cols and "is_weekend" in cols


def test_window_arrays_shapes_and_alignment():
    feats = np.column_stack([np.arange(500.0), np.arange(500.0) * 2])
    tgt = np.arange(500.0)
    X, Y = lm.window_arrays(feats, tgt, input_len=336, output_len=96)
    assert X.shape == (500 - 336 - 96 + 1, 336, 2)
    assert Y.shape == (500 - 336 - 96 + 1, 96)
    assert X[0, 0, 0] == 0.0 and Y[0, 0] == 336.0 and Y[0, -1] == 431.0


# ---- guarded end-to-end ----------------------------------------------------
@pytest.mark.slow
def test_fit_predict_and_save(tmp_path):
    pytest.importorskip("tensorflow")
    df = _df(700)
    train = df.iloc[:520].copy()
    val = df.iloc[420:].copy()  # overlap for enough windows

    model = lm.LSTMForecaster(
        input_len=96, output_len=48, units=(16, 8), dense_units=16, max_epochs=2, patience=1
    )
    model.fit(train, val)
    assert model.training_seconds_ is not None
    assert "loss" in model.history_ and "val_loss" in model.history_

    preds = model.predict_window(df.tail(96))
    assert preds.shape == (48,)

    mp, sp, jp = model.save(
        tmp_path / "lstm.keras", tmp_path / "lstm_scalers.joblib", tmp_path / "lstm_metadata.json"
    )
    assert mp.exists() and sp.exists() and jp.exists()
    import json

    meta = json.loads(jp.read_text())
    assert meta["model"] == "LSTM_seq2seq"
    assert "training_seconds" in meta and "feature_list" in meta

    fig_path = model.save_history_plot(tmp_path / "curve.png")
    assert fig_path.exists()
