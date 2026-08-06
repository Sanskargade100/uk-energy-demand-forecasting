"""Temperature-derived features.

Threshold assumptions (state clearly; tune on validation data)
-------------------------------------------------------------
* ``HDD_BASE = 15.5 °C`` — heating degree base. Below this, space heating tends to
  switch on, so demand rises as temperature falls. 15.5 °C is the long-standing UK
  degree-day convention.
* ``CDD_BASE = 18.0 °C`` — cooling degree base. Above this, cooling load begins to
  add demand. GB cooling is modest, so this term is weak but non-zero in summer.

These are modelling choices, not physical constants; both bases are parameters and
good candidates for tuning against validation performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HDD_BASE = 15.5
CDD_BASE = 18.0


def add_temperature_features(
    df: pd.DataFrame,
    temp_col: str = "temperature_mean",
    hdd_base: float = HDD_BASE,
    cdd_base: float = CDD_BASE,
    lag: int = 48,
    rolling_window: int = 48,
) -> pd.DataFrame:
    """Add heating/cooling degrees, squared temp, a lag and a rolling mean.

    Temperature is an *exogenous* input (a weather forecast is available at
    prediction time), so — unlike demand — its rolling mean may include the current
    value without leaking the target.
    """
    df = df.copy()
    temp = df[temp_col]

    df["heating_degree"] = np.maximum(hdd_base - temp, 0.0)
    df["cooling_degree"] = np.maximum(temp - cdd_base, 0.0)
    df["temperature_squared"] = temp**2
    df[f"temperature_lag_{lag}"] = temp.shift(lag)
    df[f"temperature_rolling_mean_{rolling_window}"] = temp.rolling(
        window=rolling_window, min_periods=rolling_window
    ).mean()
    return df
