"""Numeric tests for the baseline forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.models import seasonal_naive as sn


def _history(n):
    """History whose value equals its integer position (0..n-1)."""
    idx = pd.date_range("2021-01-01 00:00", periods=n, freq="30min", tz="UTC")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def test_last_value_repeats_origin():
    h = _history(500)
    out = sn.last_value_forecast(h, horizon=96)
    assert len(out) == 96
    assert (out == 499.0).all()
    assert out.index[0] == h.index[-1] + pd.Timedelta(minutes=30)


def test_yesterday_first_day():
    h = _history(500)  # origin value = 499
    out = sn.seasonal_naive_forecast(h, horizon=96, season=48)
    # h=1 -> src = target-48 = origin-47 = value 452
    assert out.iloc[0] == 452.0
    # h=48 -> src = origin = 499
    assert out.iloc[47] == 499.0


def test_yesterday_second_day_steps_back_two_seasons():
    h = _history(500)
    out = sn.seasonal_naive_forecast(h, horizon=96, season=48)
    # h=49 -> ceil(49/48)=2 -> src = target-96 = origin-47 = 452
    assert out.iloc[48] == 452.0
    # h=96 -> src = target-96 = origin = 499
    assert out.iloc[95] == 499.0


def test_last_week():
    h = _history(1000)  # origin = 999
    out = sn.seasonal_naive_forecast(h, horizon=96, season=336)
    # h=1 -> src = target-336 = origin-335 = 664
    assert out.iloc[0] == 664.0
    # h=96 -> src = target-336 = origin-240 = 759
    assert out.iloc[95] == 759.0


def test_recent_4wk_average():
    h = _history(1500)  # origin = 1499
    out = sn.recent_period_average_forecast(h, horizon=96, season=336, n_weeks=4)
    # h=1: mean of origin+1-336k for k=1..4 = mean(1164,828,492,156) = 660
    assert out.iloc[0] == np.mean([1164, 828, 492, 156])


def test_registry_and_dispatch():
    h = _history(1500)
    assert set(sn.BASELINES) == {"last_value", "yesterday", "last_week", "recent_4wk_avg"}
    for name in sn.BASELINES:
        out = sn.forecast(name, h, horizon=96)
        assert len(out) == 96
        assert out.notna().all()


def test_forecast_targets_are_future_and_half_hourly():
    h = _history(500)
    out = sn.forecast("yesterday", h, horizon=96)
    assert (out.index > h.index[-1]).all()
    assert (out.index.to_series().diff().dropna() == pd.Timedelta(minutes=30)).all()
