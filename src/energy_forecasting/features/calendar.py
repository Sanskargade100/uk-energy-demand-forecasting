"""Calendar and cyclical time features.

All features are derived from **local** time (``timestamp_local``, Europe/London),
because demand behaviour follows the clock people live by, not UTC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# name -> period for the cyclical (sin/cos) encodings.
CYCLICAL_PERIODS = {
    "hour": 24,
    "half_hour_slot": 48,
    "day_of_week": 7,
    "day_of_year": 365,
    "month": 12,
}


def add_cyclical(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """Add ``{column}_sin`` / ``{column}_cos`` for a cyclical integer column."""
    df = df.copy()
    radians = 2.0 * np.pi * df[column] / period
    df[f"{column}_sin"] = np.sin(radians)
    df[f"{column}_cos"] = np.cos(radians)
    return df


def _days_to_from_christmas(local_dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Days until the next Christmas and since the most recent one (0 on Dec 25)."""
    d = pd.to_datetime(local_dates).dt.normalize()
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    year = d.dt.year
    xmas = pd.to_datetime(dict(year=year, month=12, day=25))
    xmas_prev = pd.to_datetime(dict(year=year - 1, month=12, day=25))
    xmas_next = pd.to_datetime(dict(year=year + 1, month=12, day=25))

    next_xmas = xmas.where(d <= xmas, xmas_next)
    prev_xmas = xmas.where(d >= xmas, xmas_prev)
    days_to = (next_xmas - d).dt.days
    days_from = (d - prev_xmas).dt.days
    return days_to.astype(int), days_from.astype(int)


def add_calendar_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp_local"
) -> pd.DataFrame:
    """Add the full set of calendar + cyclical features.

    ``is_bank_holiday`` is taken from the processed table if present (GB-specific,
    from ``clean.py``); otherwise it defaults to 0.
    """
    df = df.copy()
    local = pd.to_datetime(df[timestamp_col])

    df["hour"] = local.dt.hour
    df["minute"] = local.dt.minute
    df["half_hour_slot"] = local.dt.hour * 2 + (local.dt.minute >= 30).astype(int)
    df["day_of_week"] = local.dt.dayofweek  # Mon=0 .. Sun=6
    df["day_of_month"] = local.dt.day
    df["day_of_year"] = local.dt.dayofyear
    df["week_of_year"] = local.dt.isocalendar().week.astype(int)
    df["month"] = local.dt.month
    df["quarter"] = local.dt.quarter
    df["year"] = local.dt.year
    df["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)

    if "is_bank_holiday" not in df.columns:
        df["is_bank_holiday"] = 0
    df["is_bank_holiday"] = df["is_bank_holiday"].astype(int)

    days_to, days_from = _days_to_from_christmas(local)
    df["days_to_christmas"] = days_to
    df["days_from_christmas"] = days_from

    for column, period in CYCLICAL_PERIODS.items():
        df = add_cyclical(df, column, period)

    return df
