"""Tests for feature engineering: calendar, cyclical, lags, rolling, temperature."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.features.calendar import add_calendar_features, add_cyclical
from energy_forecasting.features.lag_features import add_demand_lags, add_demand_rolling
from energy_forecasting.features.weather import add_temperature_features


def _frame(n=400):
    utc = pd.date_range("2021-06-14 23:00", periods=n, freq="30min", tz="UTC")
    local = utc.tz_convert("Europe/London")
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local,
            "nd_mw": np.arange(n, dtype=float),
            "temperature_mean": np.linspace(5.0, 25.0, n),
            "is_bank_holiday": 0,
        }
    )


# ---- calendar --------------------------------------------------------------
def test_calendar_columns_present():
    out = add_calendar_features(_frame())
    for col in [
        "hour", "minute", "half_hour_slot", "day_of_week", "day_of_month",
        "day_of_year", "week_of_year", "month", "quarter", "year",
        "is_weekend", "is_bank_holiday", "days_to_christmas", "days_from_christmas",
    ]:
        assert col in out.columns


def test_half_hour_slot_range_and_value():
    out = add_calendar_features(_frame())
    assert out["half_hour_slot"].between(0, 47).all()
    # A 00:00 local row -> slot 0; 00:30 -> slot 1.
    midnight = out[out["timestamp_local"].dt.strftime("%H:%M") == "00:00"].iloc[0]
    assert midnight["half_hour_slot"] == 0


def test_is_weekend_flag():
    out = add_calendar_features(_frame())
    # 2021-06-14 is a Monday; 2021-06-19/20 is the weekend.
    sat = out[out["timestamp_local"].dt.strftime("%Y-%m-%d") == "2021-06-19"]
    assert (sat["is_weekend"] == 1).all()


def test_days_to_from_christmas():
    df = pd.DataFrame(
        {"timestamp_local": pd.to_datetime(["2021-12-20", "2021-12-25", "2021-12-28"])}
    )
    out = add_calendar_features(df)
    assert out["days_to_christmas"].tolist() == [5, 0, 362]
    assert out["days_from_christmas"].tolist() == [360, 0, 3]


def test_cyclical_encoding_values():
    df = pd.DataFrame({"hour": [0, 6, 12, 18]})
    out = add_cyclical(df, "hour", 24)
    assert np.isclose(out.loc[0, "hour_sin"], 0.0) and np.isclose(out.loc[0, "hour_cos"], 1.0)
    assert np.isclose(out.loc[1, "hour_sin"], 1.0, atol=1e-9)  # 6h -> quarter turn
    assert np.isclose(out.loc[2, "hour_cos"], -1.0, atol=1e-9)  # 12h -> half turn


# ---- lags ------------------------------------------------------------------
def test_demand_lags_alignment():
    out = add_demand_lags(_frame(), column="nd_mw", lags=(1, 2, 48))
    assert np.isnan(out.loc[0, "demand_lag_1"])
    assert out.loc[1, "demand_lag_1"] == 0.0
    assert out.loc[48, "demand_lag_48"] == 0.0
    assert out.loc[100, "demand_lag_2"] == 98.0


# ---- rolling (leakage-safe) ------------------------------------------------
def test_rolling_uses_only_previous_observations():
    out = add_demand_rolling(_frame(), column="nd_mw", windows_stats={48: ("mean", "std", "min", "max")})
    # At row 48, the window is demand[0..47] (previous 48, current excluded).
    assert out.loc[48, "demand_rolling_mean_48"] == np.mean(np.arange(48))
    assert out.loc[48, "demand_rolling_min_48"] == 0.0
    assert out.loc[48, "demand_rolling_max_48"] == 47.0
    # Warm-up rows are NaN (need a full window).
    assert out["demand_rolling_mean_48"].iloc[:48].isna().all()


def test_rolling_does_not_include_current_value():
    df = _frame()
    out = add_demand_rolling(df, column="nd_mw", windows_stats={48: ("max",)})
    # Current value (demand[i]) must never equal the rolling max at that row.
    i = 100
    assert out.loc[i, "demand_rolling_max_48"] == 99.0  # demand[52..99] max = 99, not 100
    assert out.loc[i, "demand_rolling_max_48"] != df.loc[i, "nd_mw"]


# ---- temperature -----------------------------------------------------------
def test_temperature_features():
    df = pd.DataFrame({"temperature_mean": [10.0, 15.5, 20.0, 25.0]})
    out = add_temperature_features(df, lag=1, rolling_window=2)
    assert out.loc[0, "heating_degree"] == 5.5   # 15.5 - 10
    assert out.loc[1, "heating_degree"] == 0.0
    assert out.loc[3, "cooling_degree"] == 7.0   # 25 - 18
    assert out.loc[0, "cooling_degree"] == 0.0
    assert out.loc[2, "temperature_squared"] == 400.0
    assert out.loc[1, "temperature_lag_1"] == 10.0
