"""Baseline forecasters. A complex model is only worth it if it beats these.

Each forecaster takes a demand ``history`` (a ``pd.Series`` on a gap-free 30-minute
UTC grid, whose last element is the forecast *origin*) and returns a ``pd.Series`` of
``horizon`` predictions indexed by the future target timestamps.

Baselines
---------
1. **Last value** (persistence): ``y_hat[t+h] = y_t`` for all h.
2. **Same period yesterday**: ``y_hat[t+h] = y_{t+h-48}`` (season = 48 half-hours).
3. **Same period last week**: ``y_hat[t+h] = y_{t+h-336}`` (season = 336).
4. **Recent matching average**: mean of the same half-hour & weekday over the
   previous four weeks (``y_{t+h-336}, y_{t+h-672}, y_{t+h-1008}, y_{t+h-1344}``).

For a 96-step (48-hour) horizon with a daily season (48), the second day has no
observed value one day back, so we step back whole seasons until the source
timestamp is at or before the origin: ``src = target - ceil(h/season) * season``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

FREQ = "30min"
STEP = pd.Timedelta(minutes=30)
HORIZON = 96
DAY = 48
WEEK = 336


def _targets(last: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last + STEP, periods=horizon, freq=FREQ, name="target")


def _lookup(history: pd.Series, source_times: pd.DatetimeIndex) -> np.ndarray:
    """Values of ``history`` at the given timestamps (NaN where absent)."""
    return history.reindex(source_times).to_numpy()


def last_value_forecast(history: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Baseline 1: repeat the last observed value across the horizon."""
    targets = _targets(history.index[-1], horizon)
    return pd.Series(history.iloc[-1], index=targets, name="last_value")


def seasonal_naive_forecast(
    history: pd.Series, horizon: int = HORIZON, season: int = DAY
) -> pd.Series:
    """Baselines 2 & 3: value from the most recent matching period one season back.

    Steps back whole seasons so horizons beyond one season still resolve to an
    observed value at or before the origin.
    """
    last = history.index[-1]
    targets = _targets(last, horizon)
    h = np.arange(1, horizon + 1)
    steps_back = np.ceil(h / season).astype(int) * season
    source = targets - pd.to_timedelta(steps_back * 30, unit="m")
    values = _lookup(history, pd.DatetimeIndex(source))
    return pd.Series(values, index=targets, name=f"seasonal_naive_{season}")


def recent_period_average_forecast(
    history: pd.Series, horizon: int = HORIZON, season: int = WEEK, n_weeks: int = 4
) -> pd.Series:
    """Baseline 4: average the same half-hour & weekday over the previous n weeks."""
    last = history.index[-1]
    targets = _targets(last, horizon)
    stacked = []
    for k in range(1, n_weeks + 1):
        source = targets - pd.to_timedelta(k * season * 30, unit="m")
        stacked.append(_lookup(history, pd.DatetimeIndex(source)))
    values = np.nanmean(np.vstack(stacked), axis=0)
    return pd.Series(values, index=targets, name=f"recent_{n_weeks}wk_avg")


#: Name -> forecaster(history, horizon) for use by the evaluation loop.
BASELINES: dict[str, Callable[..., pd.Series]] = {
    "last_value": last_value_forecast,
    "yesterday": lambda history, horizon=HORIZON: seasonal_naive_forecast(
        history, horizon, season=DAY
    ),
    "last_week": lambda history, horizon=HORIZON: seasonal_naive_forecast(
        history, horizon, season=WEEK
    ),
    "recent_4wk_avg": recent_period_average_forecast,
}


def forecast(name: str, history: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Dispatch to a named baseline."""
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline {name!r}; choose from {list(BASELINES)}.")
    return BASELINES[name](history, horizon)
