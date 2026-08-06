"""Supervised framing for multi-step (96-period) forecasting.

Two framings, matching the model families:

* :func:`make_horizon_samples` — the **horizon-feature** framing for XGBoost. For
  every origin time ``o`` and horizon ``h in 1..96`` it builds one training row:
  demand-history features known *at the origin*, plus calendar/weather for the
  forecasted timestamp ``o + h``, plus ``forecast_horizon = h``; the target is
  demand at ``o + h``. One model then serves all horizons.

* :func:`make_sequences` — the **seq2seq** framing for the LSTM: an input window of
  the previous ``input_len`` periods (default 336 = 7 days) mapped to the next
  ``output_len`` periods (default 96 = 2 days).

Leakage principle
-----------------
Demand-history features come from the origin row (they only ever use demand up to
the origin). Calendar features for ``o+h`` are deterministic, and weather for
``o+h`` is assumed to come from a forecast available at the origin — so both are
legitimately known when the forecast is issued.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Columns that describe demand *history* (valid only as of the origin).
ORIGIN_PREFIXES = ("demand_lag_", "demand_rolling_")

# Columns never used as target-time features.
_NON_FEATURE = {
    "nd_mw",
    "nd_mw_is_missing",
    "weather_is_missing",
    "timestamp_utc",
    "timestamp_local",
    "settlement_date",
}


def split_feature_groups(fm: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into (origin demand-history, target-time known)."""
    origin = [c for c in fm.columns if c.startswith(ORIGIN_PREFIXES)]
    exclude = set(origin) | _NON_FEATURE
    target = [c for c in fm.columns if c not in exclude]
    return origin, target


def make_horizon_samples(
    fm: pd.DataFrame,
    horizons: Iterable[int] = range(1, 97),
    origin_step: int = 1,
    origin_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    target: str = "nd_mw",
) -> pd.DataFrame:
    """Build the long horizon-feature training table.

    Parameters
    ----------
    fm : DataFrame
        Feature matrix from ``build_features`` (sorted, gap-free 30-min grid).
    horizons : iterable of int
        Forecast steps ahead, in half-hours (default 1..96).
    origin_step : int
        Sub-sample origins (e.g. 2 = hourly origins) to control table size. The
        full cross-product of origins x horizons can be very large.

    Returns
    -------
    DataFrame with columns: ``origin_time``, ``target_time``, ``forecast_horizon``,
    the origin demand-history features, ``demand_origin`` (last known value at the
    origin), the target-time features, and ``y`` (demand at ``target_time``).
    """
    fm = fm.sort_values("timestamp_utc").reset_index(drop=True)
    if origin_cols is None or target_cols is None:
        auto_origin, auto_target = split_feature_groups(fm)
        origin_cols = origin_cols if origin_cols is not None else auto_origin
        target_cols = target_cols if target_cols is not None else auto_target

    ts = fm["timestamp_utc"].to_numpy()
    target_vals = fm[target].to_numpy()
    n = len(fm)

    blocks = []
    for h in horizons:
        if h >= n:
            continue
        origin_idx = np.arange(0, n - h, origin_step)
        tgt_idx = origin_idx + h

        block = {
            "origin_time": ts[origin_idx],
            "target_time": ts[tgt_idx],
            "forecast_horizon": h,
            "demand_origin": target_vals[origin_idx],
        }
        for c in origin_cols:
            block[c] = fm[c].to_numpy()[origin_idx]
        for c in target_cols:
            block[c] = fm[c].to_numpy()[tgt_idx]
        block["y"] = target_vals[tgt_idx]
        blocks.append(pd.DataFrame(block))

    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def make_sequences(
    values,
    input_len: int = 336,
    output_len: int = 96,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) sliding windows for seq2seq models.

    ``values`` may be 1-D (univariate) or 2-D ``(time, features)``. Returns
    ``X`` of shape ``(n_windows, input_len[, features])`` and ``Y`` of shape
    ``(n_windows, output_len)`` taken from the first column when 2-D.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    target = arr[:, 0] if arr.ndim == 2 else arr

    xs, ys = [], []
    last_start = n - input_len - output_len
    for start in range(0, last_start + 1, stride):
        xs.append(arr[start : start + input_len])
        ys.append(target[start + input_len : start + input_len + output_len])
    if not xs:
        return np.empty((0, input_len)), np.empty((0, output_len))
    return np.asarray(xs), np.asarray(ys)
