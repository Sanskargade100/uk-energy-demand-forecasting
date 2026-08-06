"""Tests for timestamp handling, focused on the UK clock-change days.

UK DST transitions (verified):
* Spring forward 2021-03-28 — clocks go 01:00 -> 02:00 (46 settlement periods).
* Autumn back     2021-10-31 — clocks go 02:00 -> 01:00 (50 settlement periods).
"""

from __future__ import annotations

import pandas as pd

from energy_forecasting.utils import time_utils as tu

SPRING = "2021-03-28"  # 46 periods
AUTUMN = "2021-10-31"  # 50 periods
NORMAL = "2021-06-15"  # 48 periods


# ---- expected period counts ------------------------------------------------
def test_expected_periods_normal_day():
    assert tu.expected_periods_in_day(NORMAL) == 48
    assert not tu.is_clock_change_day(NORMAL)


def test_expected_periods_spring_forward_is_46():
    assert tu.expected_periods_in_day(SPRING) == 46
    assert tu.is_clock_change_day(SPRING)


def test_expected_periods_autumn_is_50():
    assert tu.expected_periods_in_day(AUTUMN) == 50
    assert tu.is_clock_change_day(AUTUMN)


# ---- UTC construction ------------------------------------------------------
def test_utc_is_tz_aware_and_winter_offset_zero():
    out = tu.settlement_to_utc(pd.Series(["2021-01-01"]), pd.Series([1]))
    assert str(out.iloc[0].tz) == "UTC"
    assert out.iloc[0] == pd.Timestamp("2021-01-01 00:00:00", tz="UTC")


def test_summer_local_midnight_is_bst():
    # July: local midnight == 23:00 UTC the previous day (UTC+1).
    out = tu.settlement_to_utc(pd.Series([NORMAL]), pd.Series([1]))
    assert out.iloc[0] == pd.Timestamp("2021-06-14 23:00:00", tz="UTC")


def test_spring_forward_skips_the_missing_hour():
    # Periods 1..5 of the spring day. The naive formula would place period 3 at
    # 01:00 (which does not exist); the correct local times skip to 02:00.
    periods = pd.Series([1, 2, 3, 4, 5])
    local = tu.settlement_to_london(pd.Series([SPRING] * 5), periods)
    wall = local.dt.strftime("%H:%M").tolist()
    assert wall == ["00:00", "00:30", "02:00", "02:30", "03:00"]
    # UTC stays evenly spaced at 30 minutes across the gap.
    utc = tu.settlement_to_utc(pd.Series([SPRING] * 5), periods)
    assert list(utc.diff().dropna().unique()) == [pd.Timedelta(minutes=30)]


def test_autumn_repeats_the_extra_hour_in_local_time():
    # Periods 1..7. Local 01:00 and 01:30 each appear twice (BST then GMT).
    periods = pd.Series(range(1, 8))
    local = tu.settlement_to_london(pd.Series([AUTUMN] * 7), periods)
    wall = local.dt.strftime("%H:%M").tolist()
    assert wall == ["00:00", "00:30", "01:00", "01:30", "01:00", "01:30", "02:00"]
    # But the UTC instants are all distinct and 30 minutes apart.
    utc = tu.settlement_to_utc(pd.Series([AUTUMN] * 7), periods)
    assert utc.is_unique
    assert list(utc.diff().dropna().unique()) == [pd.Timedelta(minutes=30)]


# ---- duplicate / nonexistent detectors -------------------------------------
def test_find_duplicate_local_times_flags_autumn_overlap():
    periods = pd.Series(range(1, 8))
    utc = tu.settlement_to_utc(pd.Series([AUTUMN] * 7), periods)
    dups = tu.find_duplicate_local_times(utc)
    labels = sorted(dups.dt.strftime("%H:%M").unique().tolist())
    assert labels == ["01:00", "01:30"]
    assert len(dups) == 4  # two labels, each appearing twice


def test_no_duplicate_local_times_on_normal_day():
    periods = pd.Series(range(1, 49))
    utc = tu.settlement_to_utc(pd.Series([NORMAL] * 48), periods)
    assert tu.find_duplicate_local_times(utc).empty


def test_find_nonexistent_local_times_flags_spring_gap():
    # Applying the naive formula to periods 1..6 puts periods 3 and 4 at the
    # nonexistent 01:00 and 01:30 local times.
    periods = pd.Series(range(1, 7))
    missing = tu.find_nonexistent_local_times(pd.Series([SPRING] * 6), periods)
    labels = missing.dt.strftime("%H:%M").tolist()
    assert labels == ["01:00", "01:30"]


def test_no_nonexistent_local_times_on_normal_day():
    periods = pd.Series(range(1, 49))
    assert tu.find_nonexistent_local_times(pd.Series([NORMAL] * 48), periods).empty
