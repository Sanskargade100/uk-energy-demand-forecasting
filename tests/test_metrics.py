"""Tests for evaluation metrics and error breakdowns (known-value checks)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.evaluation import metrics as m


def test_mae_rmse_known_values():
    yt = np.array([10.0, 20.0, 30.0])
    yp = np.array([12.0, 18.0, 33.0])
    assert m.mae(yt, yp) == np.mean([2, 2, 3])
    assert np.isclose(m.rmse(yt, yp), np.sqrt(np.mean([4, 4, 9])))


def test_mape_and_wape():
    yt = np.array([100.0, 200.0])
    yp = np.array([110.0, 180.0])
    # MAPE: mean(|10/100|, |20/200|) * 100 = mean(10%, 10%) = 10
    assert np.isclose(m.mape(yt, yp), 10.0)
    # WAPE: (10 + 20) / (100 + 200) * 100 = 10
    assert np.isclose(m.wape(yt, yp), 10.0)


def test_r2_perfect_and_mean_predictor():
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(m.r2(yt, yt), 1.0)
    mean_pred = np.full_like(yt, yt.mean())
    assert abs(m.r2(yt, mean_pred)) < 1e-6  # predicting the mean -> R2 ~ 0


def test_bias_sign():
    yt = np.array([10.0, 10.0])
    assert m.bias(yt, np.array([12.0, 12.0])) == 2.0    # over-forecast -> positive
    assert m.bias(yt, np.array([8.0, 8.0])) == -2.0     # under-forecast -> negative


def test_peak_mae_focuses_on_high_demand():
    yt = np.array([10, 20, 30, 40, 50], dtype=float)
    yp = np.array([10, 20, 30, 40, 40], dtype=float)  # only the peak is wrong
    # 90th percentile of yt ~ 46 -> only the 50 row counts.
    assert m.peak_mae(yt, yp, quantile=0.9) == 10.0


def test_pinball_loss_median_equals_half_mae():
    yt = np.array([10.0, 20.0, 30.0])
    yq = np.array([12.0, 18.0, 33.0])
    # At q=0.5 pinball = 0.5 * MAE.
    assert np.isclose(m.pinball_loss(yt, yq, 0.5), 0.5 * m.mae(yt, yq))


def test_coverage():
    yt = np.array([5.0, 15.0, 25.0])
    lo = np.array([0.0, 10.0, 30.0])
    hi = np.array([10.0, 20.0, 40.0])
    # 5 in [0,10] yes; 15 in [10,20] yes; 25 in [30,40] no -> 2/3
    assert np.isclose(m.coverage(yt, lo, hi), 2 / 3)


def test_interval_metrics_keys():
    yt = np.array([10.0, 20.0])
    out = m.interval_metrics(yt, [5, 15], [15, 25], [0, 10], [20, 30])
    assert set(out) == {"coverage_80", "coverage_95"}


def _eval_df():
    t = pd.date_range("2021-06-15 00:00", periods=8, freq="3h", tz="UTC")
    return pd.DataFrame(
        {
            "target_time": t,
            "forecast_horizon": [1, 2, 3, 4, 1, 2, 3, 4],
            "y_true": [10, 20, 30, 40, 10, 20, 30, 40],
            "y_pred": [11, 19, 33, 37, 10, 20, 30, 40],
            "is_bank_holiday": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_error_by_horizon():
    out = m.error_by(_eval_df(), "forecast_horizon")
    assert set(out.columns) == {"forecast_horizon", "mae", "count"}
    # horizon 1 errors: |10-11|=1 and |10-10|=0 -> mae 0.5
    assert out.loc[out["forecast_horizon"] == 1, "mae"].iloc[0] == 0.5


def test_error_breakdowns_keys():
    out = m.error_breakdowns(_eval_df())
    assert {"by_horizon", "by_hour", "by_weekday", "by_season", "by_holiday"} <= set(out)
    assert "season" in out["by_season"].columns


def test_comparison_table_sorted_by_mae():
    rows = [
        m.comparison_row("A", [10, 20], [10, 20], training_seconds=1.0),       # mae 0
        m.comparison_row("B", [10, 20], [14, 24], training_seconds=2.0),       # mae 4
    ]
    table = m.comparison_table(rows)
    assert list(table.columns) == m.COMPARISON_COLUMNS
    assert list(table["model"]) == ["A", "B"]  # best (lowest MAE) first
