"""Unit tests for the GOV.UK bank-holidays downloader (no network required)."""

from __future__ import annotations

import pandas as pd

from energy_forecasting.data import download_holidays as hol

# Mock payload exercising: a UK-wide day, an E&W-only day, a Scotland-only day
# ("2nd January"), and an NI-only day ("St Patrick's Day").
PAYLOAD = {
    "england-and-wales": {
        "events": [
            {"title": "New Year's Day", "date": "2021-01-01"},
            {"title": "Easter Monday", "date": "2021-04-05"},
        ]
    },
    "scotland": {
        "events": [
            {"title": "New Year's Day", "date": "2021-01-01"},
            {"title": "2nd January", "date": "2021-01-04"},
        ]
    },
    "northern-ireland": {
        "events": [
            {"title": "New Year's Day", "date": "2021-01-01"},
            {"title": "St Patrick's Day", "date": "2021-03-17"},
        ]
    },
}


def _row(df, date):
    return df[df["date"] == pd.Timestamp(date)].iloc[0]


def test_columns_and_order():
    df = hol.parse_bank_holidays(PAYLOAD)
    assert list(df.columns) == [
        "date",
        "holiday_name",
        "is_england_wales_holiday",
        "is_scotland_holiday",
        "is_northern_ireland_holiday",
        "is_any_uk_holiday",
        "gb_holiday_weight",
    ]


def test_uk_wide_day_all_flags_and_weight_one():
    df = hol.parse_bank_holidays(PAYLOAD)
    r = _row(df, "2021-01-01")
    assert r["is_england_wales_holiday"] == 1
    assert r["is_scotland_holiday"] == 1
    assert r["is_northern_ireland_holiday"] == 1
    assert r["is_any_uk_holiday"] == 1
    assert r["gb_holiday_weight"] == 1.0  # 0.916 + 0.084


def test_england_wales_only_day():
    df = hol.parse_bank_holidays(PAYLOAD)
    r = _row(df, "2021-04-05")
    assert r["is_england_wales_holiday"] == 1
    assert r["is_scotland_holiday"] == 0
    assert r["is_northern_ireland_holiday"] == 0
    assert r["gb_holiday_weight"] == hol.ENGLAND_WALES_SHARE
    assert r["holiday_name"] == "Easter Monday"


def test_scotland_only_day():
    df = hol.parse_bank_holidays(PAYLOAD)
    r = _row(df, "2021-01-04")
    assert r["is_scotland_holiday"] == 1
    assert r["is_england_wales_holiday"] == 0
    assert r["gb_holiday_weight"] == hol.SCOTLAND_SHARE
    assert r["holiday_name"] == "2nd January"


def test_northern_ireland_only_day_has_zero_gb_weight():
    df = hol.parse_bank_holidays(PAYLOAD)
    r = _row(df, "2021-03-17")
    assert r["is_northern_ireland_holiday"] == 1
    assert r["is_any_uk_holiday"] == 1
    # NI is outside GB, so it must not move GB demand weighting.
    assert r["gb_holiday_weight"] == 0.0
    assert r["holiday_name"] == "St Patrick's Day"


def test_date_range_filter():
    df = hol.parse_bank_holidays(PAYLOAD, start="2021-01-02", end="2021-03-31")
    assert list(df["date"]) == [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-03-17")]


def test_one_row_per_date_sorted():
    df = hol.parse_bank_holidays(PAYLOAD)
    assert df["date"].is_monotonic_increasing
    assert df["date"].is_unique
    assert len(df) == 4
