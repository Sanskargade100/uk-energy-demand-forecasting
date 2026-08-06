"""Demand lag and rolling-window features.

**Leakage guard.** Rolling statistics are computed on ``demand.shift(1)`` so a row's
features only ever use *previous* observations — the current target can never enter
its own features. The frame is assumed to be sorted ascending on a gap-free
30-minute grid (as produced by ``clean.py``), so a lag of ``k`` rows equals ``k``
half-hours.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_LAGS = (1, 2, 48, 96, 336)
# window -> statistics to compute. 48 = 1 day, 336 = 1 week.
DEFAULT_ROLLING = {
    48: ("mean", "std", "min", "max"),
    336: ("mean", "std"),
}


def add_demand_lags(
    df: pd.DataFrame, column: str = "nd_mw", lags=DEFAULT_LAGS
) -> pd.DataFrame:
    """Add ``demand_lag_{k}`` columns for each lag ``k`` (in half-hour steps)."""
    df = df.copy()
    for k in lags:
        df[f"demand_lag_{k}"] = df[column].shift(k)
    return df


def add_demand_rolling(
    df: pd.DataFrame,
    column: str = "nd_mw",
    windows_stats: dict[int, tuple[str, ...]] = DEFAULT_ROLLING,
    shift: int = 1,
) -> pd.DataFrame:
    """Add rolling stats computed on the shifted series (no target leakage).

    Produces e.g. ``demand_rolling_mean_48`` … ``demand_rolling_std_336``.
    ``min_periods`` equals the window, so warm-up rows are NaN rather than being
    computed from a partial window.
    """
    df = df.copy()
    shifted = df[column].shift(shift)
    for window, stats in windows_stats.items():
        roller = shifted.rolling(window=window, min_periods=window)
        for stat in stats:
            df[f"demand_rolling_{stat}_{window}"] = getattr(roller, stat)()
    return df
