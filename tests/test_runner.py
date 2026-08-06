"""Tests for the evaluation runner (mock forecast functions, tiny data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.evaluation import runner


def _actuals(n=500):
    idx = pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def _perfect_fn(actuals, horizon=4):
    """A forecast fn that returns the true future values (perfect model)."""
    def fn(origin):
        targets = pd.date_range(origin + pd.Timedelta(minutes=30), periods=horizon, freq="30min")
        vals = actuals.reindex(targets)
        if vals.isna().any():
            return None
        return vals
    return fn


def _biased_fn(actuals, horizon=4, bias=100.0):
    def fn(origin):
        targets = pd.date_range(origin + pd.Timedelta(minutes=30), periods=horizon, freq="30min")
        vals = actuals.reindex(targets)
        if vals.isna().any():
            return None
        return vals + bias
    return fn


def test_collect_predictions_alignment():
    act = _actuals()
    origins = act.index[100:110]
    df = runner.collect_predictions({"perfect": _perfect_fn(act)}, origins, act, horizon=4)
    assert set(df.columns) >= {"model", "origin_time", "target_time", "forecast_horizon", "y_pred", "y_true"}
    # horizon = (target - origin)/30min
    assert set(df["forecast_horizon"].unique()) == {1, 2, 3, 4}
    # perfect model: y_pred == y_true
    assert np.allclose(df["y_pred"], df["y_true"])


def test_score_models_perfect_and_biased():
    act = _actuals()
    origins = act.index[100:150]
    fns = {"perfect": _perfect_fn(act), "biased": _biased_fn(act, bias=100.0)}
    df = runner.collect_predictions(fns, origins, act, horizon=4)
    table = runner.score_models(df, training_seconds={"perfect": 0.0, "biased": 1.0})

    assert list(table.columns) == ["model", "mae", "rmse", "wape", "peak_mae", "training_seconds"]
    # perfect model has zero MAE and sorts first.
    assert table.iloc[0]["model"] == "perfect"
    assert table.iloc[0]["mae"] == 0.0
    biased = table[table["model"] == "biased"].iloc[0]
    assert np.isclose(biased["mae"], 100.0)


def test_baseline_forecast_fns_registered():
    act = _actuals(1500)
    fns = runner.baseline_forecast_fns(act, horizon=96)
    assert set(fns) == {f"baseline_{n}" for n in ["last_value", "yesterday", "last_week", "recent_4wk_avg"]}
    # a fn returns a 96-step Series once enough history exists
    out = fns["baseline_last_week"](act.index[1400])
    assert out is not None and len(out) == 96


def test_baseline_returns_none_without_history():
    act = _actuals(1500)
    fns = runner.baseline_forecast_fns(act, horizon=96, min_history=336)
    assert fns["baseline_last_week"](act.index[10]) is None  # too little history


def test_collect_predictions_empty_is_safe():
    act = _actuals()
    df = runner.collect_predictions({"none": lambda o: None}, act.index[:5], act)
    assert len(df) == 0 and "y_true" in df.columns
