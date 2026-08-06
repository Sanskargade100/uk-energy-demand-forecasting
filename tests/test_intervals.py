"""Tests for conformal prediction intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.models import intervals as iv


def test_horizon_group_label():
    assert iv.horizon_group_label(1) == "1-12"
    assert iv.horizon_group_label(12) == "1-12"
    assert iv.horizon_group_label(30) == "13-48"
    assert iv.horizon_group_label(96) == "49-96"


def _cal(residual_scale=100.0, n_per_h=200, horizons=(6, 30, 80), seed=0):
    """Calibration frame with residual size growing by horizon group."""
    rng = np.random.default_rng(seed)
    rows = []
    scale = {6: 1.0, 30: 2.0, 80: 4.0}
    for h in horizons:
        yhat = np.full(n_per_h, 30000.0)
        err = rng.normal(0, residual_scale * scale[h], n_per_h)
        rows.append(pd.DataFrame({"forecast_horizon": h, "y_pred": yhat, "y_true": yhat + err}))
    return pd.concat(rows, ignore_index=True)


def test_fit_learns_wider_quantiles_for_longer_horizons():
    model = iv.ConformalIntervals(by="group").fit(_cal())
    q_short = model._q(6, 0.95)
    q_mid = model._q(30, 0.95)
    q_long = model._q(80, 0.95)
    assert q_short < q_mid < q_long  # error grows with horizon


def test_apply_adds_ordered_bands():
    model = iv.ConformalIntervals().fit(_cal())
    preds = pd.DataFrame({"forecast_horizon": [6, 30, 80], "point_forecast_mw": [30000.0] * 3})
    out = model.apply(preds)
    for col in ["lower_80", "upper_80", "lower_95", "upper_95"]:
        assert col in out.columns
    # 95% band must contain the 80% band; bands are symmetric about the point.
    assert (out["lower_95"] <= out["lower_80"]).all()
    assert (out["upper_95"] >= out["upper_80"]).all()
    assert np.allclose(out["upper_80"] - 30000.0, 30000.0 - out["lower_80"])


def test_coverage_is_approximately_nominal():
    # Fit on calibration, test on fresh residuals from the same distribution.
    model = iv.ConformalIntervals(by="group").fit(_cal(seed=1))
    test = _cal(seed=99)
    test = test.rename(columns={"y_pred": "point_forecast_mw"})
    out = model.apply(test)
    out["y_true"] = test["y_true"].to_numpy()
    report = iv.evaluate_intervals(out)

    cov95 = report.loc[report["level"] == 0.95, "empirical_coverage"].iloc[0]
    cov80 = report.loc[report["level"] == 0.80, "empirical_coverage"].iloc[0]
    assert 0.92 <= cov95 <= 0.98
    assert 0.75 <= cov80 <= 0.85


def test_evaluate_reports_width_and_counts():
    model = iv.ConformalIntervals().fit(_cal())
    test = _cal(seed=7).rename(columns={"y_pred": "point_forecast_mw"})
    out = model.apply(test)
    out["y_true"] = test["y_true"].to_numpy()
    report = iv.evaluate_intervals(out)
    assert set(report.columns) == {
        "level", "nominal_coverage", "empirical_coverage", "avg_width", "n"
    }
    # Wider nominal level -> wider average interval.
    w80 = report.loc[report["level"] == 0.80, "avg_width"].iloc[0]
    w95 = report.loc[report["level"] == 0.95, "avg_width"].iloc[0]
    assert w95 > w80


def test_per_horizon_mode():
    model = iv.ConformalIntervals(by="horizon").fit(_cal())
    assert 6 in model.quantiles_ and 30 in model.quantiles_
