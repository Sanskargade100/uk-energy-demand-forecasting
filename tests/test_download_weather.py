"""Unit tests for the Open-Meteo weather downloader (no network required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.data import download_weather as wx


def _payload(times, temp, app, precip, cloud, wind):
    return {
        "hourly": {
            "time": times,
            "temperature_2m": temp,
            "apparent_temperature": app,
            "precipitation": precip,
            "cloud_cover": cloud,
            "wind_speed_10m": wind,
        }
    }


def test_parse_open_meteo_response_builds_utc_index():
    df = wx.parse_open_meteo_response(
        _payload(["2021-01-01T00:00", "2021-01-01T01:00"], [10, 11], [8, 9], [0, 1], [40, 50], [5, 6])
    )
    assert str(df.index.tz) == "UTC"
    assert list(df.columns) == wx.WEATHER_VARIABLES
    assert df.index[0] == pd.Timestamp("2021-01-01 00:00", tz="UTC")


def test_parse_missing_variable_raises():
    bad = {"hourly": {"time": ["2021-01-01T00:00"], "temperature_2m": [10]}}
    with pytest.raises(ValueError, match="missing variables"):
        wx.parse_open_meteo_response(bad)


def test_aggregate_uk_stats_two_cities():
    t = ["2021-01-01T00:00"]
    a = wx.parse_open_meteo_response(_payload(t, [10], [8], [1], [40], [5]))
    b = wx.parse_open_meteo_response(_payload(t, [20], [18], [2], [60], [15]))
    out = wx.aggregate_uk({"A": a, "B": b})

    row = out.iloc[0]
    assert row["uk_mean_temperature"] == 15
    assert row["uk_min_temperature"] == 10
    assert row["uk_max_temperature"] == 20
    assert row["uk_temperature_std"] == pytest.approx(np.std([10, 20], ddof=1))
    assert row["uk_mean_apparent_temperature"] == 13
    assert row["uk_mean_wind_speed"] == 10
    assert row["uk_total_precipitation"] == 3  # summed across cities
    assert row["uk_mean_cloud_cover"] == 50


def test_aggregate_uk_column_names():
    t = ["2021-01-01T00:00"]
    a = wx.parse_open_meteo_response(_payload(t, [10], [8], [1], [40], [5]))
    out = wx.aggregate_uk({"A": a})
    assert list(out.columns) == [
        "uk_mean_temperature",
        "uk_min_temperature",
        "uk_max_temperature",
        "uk_temperature_std",
        "uk_mean_apparent_temperature",
        "uk_mean_wind_speed",
        "uk_total_precipitation",
        "uk_mean_cloud_cover",
    ]


def test_resample_to_30min_interpolates_and_holds_precip():
    idx = pd.to_datetime(["2021-01-01T00:00", "2021-01-01T01:00"]).tz_localize("UTC")
    hourly = pd.DataFrame(
        {
            "uk_mean_temperature": [10.0, 20.0],
            "uk_total_precipitation": [2.0, 4.0],
        },
        index=idx,
    )
    out = wx.resample_to_30min(hourly)

    # Grid is 00:00, 00:30, 01:00.
    assert list(out.index) == list(
        pd.date_range("2021-01-01T00:00", "2021-01-01T01:00", freq="30min", tz="UTC")
    )
    # Continuous var interpolated at the midpoint.
    assert out.loc[pd.Timestamp("2021-01-01T00:30", tz="UTC"), "uk_mean_temperature"] == 15.0
    # Precip total held constant within the hour (step), not interpolated.
    assert out.loc[pd.Timestamp("2021-01-01T00:30", tz="UTC"), "uk_total_precipitation"] == 2.0


def test_fetch_city_rejects_bad_source():
    with pytest.raises(ValueError, match="source must be one of"):
        wx.fetch_city(51.5, -0.1, "2021-01-01", "2021-01-02", source="nope")
