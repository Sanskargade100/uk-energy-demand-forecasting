"""Tests for the reusable EDA helpers (pure pandas; no plotting backend needed)."""

from __future__ import annotations

import pandas as pd

from energy_forecasting.evaluation import plots


def _df():
    utc = pd.date_range("2021-06-14 23:00", periods=96, freq="30min", tz="UTC")
    local = utc.tz_convert("Europe/London")
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local,
            "nd_mw": [30000 + (i % 48) * 100 for i in range(96)],
            "is_weekend": (local.dayofweek >= 5).astype(int),
        }
    )


def test_add_calendar_columns():
    out = plots.add_calendar_columns(_df())
    for col in ["year", "month", "day_of_week", "hour", "minute", "time_of_day", "local_date"]:
        assert col in out.columns
    assert out["time_of_day"].between(0, 23.5).all()
    assert set(out["day_of_week"].unique()) <= set(range(7))


def test_mean_profile_by_time_of_day():
    out = plots.add_calendar_columns(_df())
    prof = plots.mean_profile(out, "time_of_day")
    assert isinstance(prof, pd.Series)
    assert len(prof) == 48  # 48 distinct half-hour slots


def test_mean_profile_two_keys_returns_frame_on_unstack():
    out = plots.add_calendar_columns(_df())
    prof = plots.mean_profile(out, ["is_weekend", "time_of_day"]).unstack(0)
    assert prof.shape[0] == 48
