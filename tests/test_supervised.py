"""Tests for the supervised framings (horizon-feature and seq2seq)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.features.supervised import (
    make_horizon_samples,
    make_sequences,
    split_feature_groups,
)


def _fm(n=10):
    ts = pd.date_range("2021-06-15 00:00", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "timestamp_local": ts.tz_convert("Europe/London"),
            "settlement_date": pd.Timestamp("2021-06-15"),
            "nd_mw": np.arange(n, dtype=float),
            "demand_lag_1": np.arange(n, dtype=float) - 1,
            "demand_rolling_mean_48": np.nan,
            "hour": ts.tz_convert("Europe/London").hour,
            "is_weekend": 0,
            "temperature_mean": np.linspace(10, 20, n),
        }
    )


def test_split_feature_groups():
    origin, target = split_feature_groups(_fm())
    assert set(origin) == {"demand_lag_1", "demand_rolling_mean_48"}
    assert "hour" in target and "temperature_mean" in target and "is_weekend" in target
    assert "nd_mw" not in target and "timestamp_utc" not in target


def test_horizon_samples_alignment_and_target():
    fm = _fm(10)
    s = make_horizon_samples(fm, horizons=[1, 2, 3])
    # y is demand at target_time = origin + horizon.
    for _, row in s.iterrows():
        step = int((row["target_time"] - row["origin_time"]) / pd.Timedelta(minutes=30))
        assert step == row["forecast_horizon"]
        assert row["y"] == fm.loc[fm["timestamp_utc"] == row["target_time"], "nd_mw"].iloc[0]


def test_horizon_samples_origin_features_from_origin():
    fm = _fm(10)
    s = make_horizon_samples(fm, horizons=[3])
    # demand_origin equals demand at the origin (not the target).
    for _, row in s.iterrows():
        origin_val = fm.loc[fm["timestamp_utc"] == row["origin_time"], "nd_mw"].iloc[0]
        assert row["demand_origin"] == origin_val


def test_horizon_samples_row_counts():
    fm = _fm(10)
    s = make_horizon_samples(fm, horizons=[1, 2, 3])
    # For n=10: horizon h contributes (10 - h) rows.
    assert len(s) == (10 - 1) + (10 - 2) + (10 - 3)


def test_horizon_samples_origin_step_subsamples():
    fm = _fm(10)
    full = make_horizon_samples(fm, horizons=[1])
    every2 = make_horizon_samples(fm, horizons=[1], origin_step=2)
    assert len(every2) == len(range(0, 10 - 1, 2))
    assert len(every2) < len(full)


def test_horizon_column_present_and_correct():
    s = make_horizon_samples(_fm(10), horizons=[1, 5])
    assert set(s["forecast_horizon"].unique()) == {1, 5}


def test_horizons_larger_than_series_are_skipped():
    # With only 10 rows, a 48-step horizon cannot be formed and is dropped.
    s = make_horizon_samples(_fm(10), horizons=[1, 48])
    assert set(s["forecast_horizon"].unique()) == {1}


def test_make_sequences_shapes_and_alignment():
    values = np.arange(500.0)
    X, Y = make_sequences(values, input_len=336, output_len=96)
    assert X.shape == (500 - 336 - 96 + 1, 336)
    assert Y.shape == (500 - 336 - 96 + 1, 96)
    # First window: X = 0..335, Y = 336..431.
    assert X[0, 0] == 0.0 and X[0, -1] == 335.0
    assert Y[0, 0] == 336.0 and Y[0, -1] == 431.0


def test_make_sequences_multivariate():
    values = np.column_stack([np.arange(500.0), np.arange(500.0) * 2])
    X, Y = make_sequences(values, input_len=336, output_len=96)
    assert X.ndim == 3 and X.shape[2] == 2
    # Target comes from the first column.
    assert Y[0, 0] == 336.0
