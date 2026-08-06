"""Chronological splitting and expanding-window backtesting.

**Never a random split.** Time-series models must be validated on data that comes
strictly *after* what they trained on, or performance is optimistic and leaky.

Two layers:

1. :func:`chronological_split` — the top-level train / validation / test split by
   fixed calendar dates. The **final test set is used once**, only after features
   and model settings are fixed.
2. :func:`expanding_window_folds` — rolling-origin (expanding-window) folds used
   during development: train on everything up to a month boundary, validate on the
   next month, then grow the training window and repeat.

Within a validation window, :func:`forecast_origins` chooses the origin times from
which a 96-step forecast is issued — one per day for fast development, several per
day for the final evaluation.

All intervals are **half-open** ``[start, end)`` and all timestamps are tz-aware UTC.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

UTC = "UTC"
FREQ = "30min"
HORIZON = 96  # 48 hours at 30-minute resolution

# Default top-level boundaries (from the project scope).
DEFAULT_TRAIN = ("2020-01-01", "2024-01-01")  # [train_start, train_end)
DEFAULT_VAL = ("2024-01-01", "2025-01-01")     # [val_start, val_end)
DEFAULT_TEST = ("2025-01-01", "2026-01-01")    # [test_start, test_end)


def _ts(value) -> pd.Timestamp:
    """Coerce to a tz-aware UTC Timestamp."""
    t = pd.Timestamp(value)
    return t.tz_localize(UTC) if t.tzinfo is None else t.tz_convert(UTC)


@dataclass(frozen=True)
class Split:
    """A named half-open time interval ``[start, end)``."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def slice(self, df: pd.DataFrame, time_col: str = "timestamp_utc") -> pd.DataFrame:
        ts = df[time_col]
        return df[(ts >= self.start) & (ts < self.end)]


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold with half-open train/validation intervals."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    val_start: pd.Timestamp
    val_end: pd.Timestamp    # exclusive

    def train_slice(self, df: pd.DataFrame, time_col: str = "timestamp_utc") -> pd.DataFrame:
        ts = df[time_col]
        return df[(ts >= self.train_start) & (ts < self.train_end)]

    def val_slice(self, df: pd.DataFrame, time_col: str = "timestamp_utc") -> pd.DataFrame:
        ts = df[time_col]
        return df[(ts >= self.val_start) & (ts < self.val_end)]


# ---------------------------------------------------------------------------
# Top-level split
# ---------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame,
    train: tuple[str, str] = DEFAULT_TRAIN,
    val: tuple[str, str] = DEFAULT_VAL,
    test: tuple[str, str] = DEFAULT_TEST,
    time_col: str = "timestamp_utc",
) -> dict[str, pd.DataFrame]:
    """Split ``df`` into train/validation/test by fixed date boundaries."""
    splits = {
        "train": Split("train", _ts(train[0]), _ts(train[1])),
        "val": Split("val", _ts(val[0]), _ts(val[1])),
        "test": Split("test", _ts(test[0]), _ts(test[1])),
    }
    return {name: sp.slice(df, time_col) for name, sp in splits.items()}


# ---------------------------------------------------------------------------
# Expanding-window folds
# ---------------------------------------------------------------------------
def expanding_window_folds(
    train_start: str,
    first_val_month: str,
    last_val_month: str,
    gap: pd.Timedelta = pd.Timedelta(0),
) -> list[Fold]:
    """Monthly expanding-window folds.

    For each month ``m`` from ``first_val_month`` to ``last_val_month`` (inclusive):
    train on ``[train_start, m) - gap`` and validate on that whole month ``[m, m+1)``.
    The training window grows each fold while ``train_start`` stays fixed.
    """
    ts_train_start = _ts(train_start)
    val_starts = pd.date_range(_ts(first_val_month), _ts(last_val_month), freq="MS", tz=UTC)

    folds = []
    for i, vs in enumerate(val_starts):
        ve = vs + pd.DateOffset(months=1)
        folds.append(
            Fold(
                index=i,
                train_start=ts_train_start,
                train_end=vs - gap,
                val_start=vs,
                val_end=ve,
            )
        )
    return folds


def iter_folds(
    df: pd.DataFrame, folds: list[Fold], time_col: str = "timestamp_utc"
) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(fold, train_df, val_df)`` for each fold."""
    for fold in folds:
        yield fold, fold.train_slice(df, time_col), fold.val_slice(df, time_col)


def describe_folds(folds: list[Fold]) -> pd.DataFrame:
    """A tidy summary of folds (handy for notebooks / logging)."""
    return pd.DataFrame(
        [
            {
                "fold": f.index,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "val_start": f.val_start,
                "val_end": f.val_end,
            }
            for f in folds
        ]
    )


# ---------------------------------------------------------------------------
# Forecast origins
# ---------------------------------------------------------------------------
def forecast_origins(
    start,
    end,
    per_day: int = 1,
    first_origin_hour: int = 0,
) -> pd.DatetimeIndex:
    """Origin timestamps within ``[start, end)`` from which to issue forecasts.

    ``per_day=1`` gives one origin per day (fast development); higher values space
    origins evenly across the day (e.g. ``per_day=4`` → 00:00, 06:00, 12:00, 18:00).
    """
    if per_day < 1:
        raise ValueError("per_day must be >= 1")
    start_ts, end_ts = _ts(start), _ts(end)
    days = pd.date_range(start_ts.normalize(), end_ts, freq="D", tz=UTC)
    step_hours = 24.0 / per_day

    origins = []
    for day in days:
        for k in range(per_day):
            o = day + pd.Timedelta(hours=first_origin_hour + k * step_hours)
            if start_ts <= o < end_ts:
                origins.append(o)
    return pd.DatetimeIndex(sorted(origins), name="origin")


def horizon_targets(origin, horizon: int = HORIZON, freq: str = FREQ) -> pd.DatetimeIndex:
    """The ``horizon`` target timestamps for an origin (origin+1 step … origin+H)."""
    o = _ts(origin)
    return pd.date_range(o + pd.Timedelta(freq), periods=horizon, freq=freq, name="target")
