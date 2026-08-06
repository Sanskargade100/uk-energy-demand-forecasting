"""Tests for cleaning and combining demand + weather + holidays."""

from __future__ import annotations

import pandas as pd

from energy_forecasting.data import clean as c
from energy_forecasting.utils.time_utils import settlement_to_utc


def _demand(day="2021-06-15", n=48):
    # Realistic timestamps: settlement period 1 == local midnight (23:00 UTC in BST).
    periods = list(range(1, n + 1))
    ts = settlement_to_utc(pd.Series([day] * n), pd.Series(periods))
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "settlement_date": pd.Timestamp(day),
            "settlement_period": periods,
            "nd_mw": [30000 + i for i in range(n)],
        }
    )


def _weather(day="2021-06-15", n=48):
    ts = settlement_to_utc(pd.Series([day] * n), pd.Series(range(1, n + 1)))
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "uk_mean_temperature": 15.0,
            "uk_min_temperature": 12.0,
            "uk_max_temperature": 18.0,
            "uk_mean_apparent_temperature": 14.0,
            "uk_mean_wind_speed": 5.0,
            "uk_total_precipitation": 0.0,
            "uk_mean_cloud_cover": 50.0,
        }
    )


def _holidays():
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2021-06-15")],
            "holiday_name": ["Test Holiday"],
            "is_england_wales_holiday": [1],
            "is_scotland_holiday": [0],
            "is_northern_ireland_holiday": [0],
            "is_any_uk_holiday": [1],
            "gb_holiday_weight": [0.916],
        }
    )


def test_clean_demand_removes_exact_duplicates_and_sorts():
    df = _demand()
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # exact dup row
    cleaned, stats = c.clean_demand(df)
    assert stats["exact_duplicates_removed"] == 1
    assert cleaned["timestamp_utc"].is_monotonic_increasing
    assert "timestamp_local" in cleaned.columns


def test_clean_demand_coerces_numeric():
    df = _demand()
    df["nd_mw"] = df["nd_mw"].astype(str)
    cleaned, _ = c.clean_demand(df)
    assert pd.api.types.is_numeric_dtype(cleaned["nd_mw"])


def test_combine_schema_and_merge():
    out, stats = c.combine(_demand(), _weather(), _holidays())
    assert list(out.columns) == c.FINAL_SCHEMA
    assert len(out) == 48
    # weather merged
    assert (out["temperature_mean"] == 15.0).all()
    # E&W holiday -> GB bank holiday flag set
    assert (out["is_bank_holiday"] == 1).all()
    # 2021-06-15 is a Tuesday
    assert (out["is_weekend"] == 0).all()
    assert stats["demand_missing_on_grid"] == 0


def test_combine_fills_gaps_and_flags_missing():
    demand = _demand().drop(index=[10, 11]).reset_index(drop=True)
    out, stats = c.combine(demand, _weather(), _holidays())
    # Grid stays complete at 48 rows; the two gaps are NaN and flagged.
    assert len(out) == 48
    assert stats["demand_missing_on_grid"] == 2
    assert out["nd_mw"].isna().sum() == 2
    assert out["nd_mw_is_missing"].sum() == 2


def test_derive_settlement_matches_source_on_normal_day():
    demand = _demand()
    out, _ = c.combine(demand)
    assert list(out["settlement_period"]) == list(range(1, 49))
    assert (out["settlement_date"] == pd.Timestamp("2021-06-15")).all()


def test_derive_settlement_autumn_day_has_50_periods():
    # 2021-10-31 spans 50 half-hours in local time.
    ts = pd.date_range("2021-10-30 23:00", "2021-11-01 00:00", freq="30min", tz="UTC")
    demand = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "settlement_date": pd.NaT,
            "settlement_period": 0,
            "nd_mw": 30000.0,
        }
    )
    out, _ = c.combine(demand)
    autumn = out[out["settlement_date"] == pd.Timestamp("2021-10-31")]
    assert autumn["settlement_period"].max() == 50
    assert autumn["settlement_period"].tolist() == list(range(1, 51))


def test_combine_without_weather_or_holidays():
    out, _ = c.combine(_demand())
    assert list(out.columns) == c.FINAL_SCHEMA
    assert out["temperature_mean"].isna().all()
    assert (out["is_bank_holiday"] == 0).all()
    assert (out["weather_is_missing"] == 1).all()
