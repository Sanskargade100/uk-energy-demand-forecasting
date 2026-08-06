"""Timestamp handling for half-hourly settlement data.

NESO demand is delivered as ``settlement_date`` + ``settlement_period`` (a 1-based
half-hour index within the *local* London day). A normal day has 48 periods, but
UK clock-change days have **46** (spring forward, an hour is skipped) or **50**
(autumn, an hour repeats). This module builds correct, timezone-aware timestamps
and provides checks so downstream code never assumes 48 observations per day.

Design
------
The naive formula ``date + (period - 1) * 30min`` is *wrong* on clock-change days,
because the period index does not increase linearly across the spring gap. Instead
we localize only **local midnight** — which is never ambiguous or nonexistent in
the UK, since transitions happen at 01:00 — and then add *absolute* 30-minute
offsets. Converting to UTC afterwards yields the correct instant on every day,
including the 46- and 50-period ones.

Storage/modelling uses UTC (:func:`settlement_to_utc`); the London-local view
(:func:`settlement_to_london`) is available for inspection and calendar features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LONDON = "Europe/London"
UTC = "UTC"
PERIOD_MINUTES = 30
PERIODS_PER_NORMAL_DAY = 48


def _as_date_series(dates) -> pd.Series:
    return pd.to_datetime(pd.Series(np.asarray(dates))).reset_index(drop=True)


def _as_period_series(periods) -> pd.Series:
    return pd.Series(np.asarray(periods)).astype("int64").reset_index(drop=True)


def settlement_to_london(dates, periods) -> pd.Series:
    """Build ``Europe/London`` tz-aware timestamps from date + settlement period.

    Localizes local midnight (unambiguous) then adds absolute half-hour offsets,
    so spring-forward and autumn days are handled correctly.
    """
    dates = _as_date_series(dates)
    periods = _as_period_series(periods)
    midnight_local = dates.dt.tz_localize(LONDON)
    offsets = pd.to_timedelta((periods - 1) * PERIOD_MINUTES, unit="m")
    return midnight_local + offsets


def settlement_to_utc(dates, periods) -> pd.Series:
    """Build UTC tz-aware timestamps from date + settlement period (for storage)."""
    return settlement_to_london(dates, periods).dt.tz_convert(UTC)


def expected_periods_in_day(date) -> int:
    """Number of settlement periods a given local date should contain (46/48/50)."""
    day = pd.Timestamp(date).normalize()
    start = day.tz_localize(LONDON)
    end = (day + pd.Timedelta(days=1)).tz_localize(LONDON)
    return int(round((end - start) / pd.Timedelta(minutes=PERIOD_MINUTES)))


def is_clock_change_day(date) -> bool:
    """True if the local date is a UK DST transition day (46 or 50 periods)."""
    return expected_periods_in_day(date) != PERIODS_PER_NORMAL_DAY


def periods_per_day(
    df: pd.DataFrame,
    date_col: str = "settlement_date",
    period_col: str = "settlement_period",
) -> pd.Series:
    """Count distinct settlement periods present for each date (for validation)."""
    return df.groupby(date_col)[period_col].nunique()


def find_duplicate_local_times(timestamps_utc) -> pd.Series:
    """Return the London wall-clock times that occur more than once.

    On the autumn clock-change day the 01:00–01:59 local hour repeats, so the same
    wall-clock time maps to two distinct UTC instants. UTC timestamps stay unique;
    this surfaces the duplicated *local* labels as a data-quality signal.
    """
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps_utc)))
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(UTC)
    local_wall = ts.dt.tz_convert(LONDON).dt.tz_localize(None)
    return local_wall[local_wall.duplicated(keep=False)].reset_index(drop=True)


def find_nonexistent_local_times(dates, periods) -> pd.Series:
    """Return naive local times that fall in the spring-forward gap (do not exist).

    Uses the *naive* ``date + (period-1)*30min`` interpretation to flag settlement
    pairs whose local time would land in the skipped 01:00–01:59 hour — a signal
    that the naive formula (rather than :func:`settlement_to_london`) is being
    misapplied for that day.
    """
    dates = _as_date_series(dates)
    periods = _as_period_series(periods)
    naive = dates + pd.to_timedelta((periods - 1) * PERIOD_MINUTES, unit="m")
    localized = naive.dt.tz_localize(
        LONDON, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="NaT"
    )
    return naive[localized.isna() & naive.notna()].reset_index(drop=True)
