"""Tests for chronological splitting and expanding-window backtesting."""

from __future__ import annotations

import pandas as pd

from energy_forecasting.evaluation import backtesting as bt


def _series_df(start="2020-01-01", end="2026-01-01"):
    ts = pd.date_range(start, end, freq="12h", tz="UTC", inclusive="left")
    return pd.DataFrame({"timestamp_utc": ts, "nd_mw": range(len(ts))})


# ---- chronological split ---------------------------------------------------
def test_chronological_split_partitions_without_overlap():
    df = _series_df()
    parts = bt.chronological_split(df)
    total = len(parts["train"]) + len(parts["val"]) + len(parts["test"])
    assert total == len(df)
    assert parts["train"]["timestamp_utc"].max() < parts["val"]["timestamp_utc"].min()
    assert parts["val"]["timestamp_utc"].max() < parts["test"]["timestamp_utc"].min()


def test_chronological_split_boundaries():
    df = _series_df()
    parts = bt.chronological_split(df)
    assert parts["val"]["timestamp_utc"].min() == pd.Timestamp("2024-01-01", tz="UTC")
    assert parts["test"]["timestamp_utc"].min() == pd.Timestamp("2025-01-01", tz="UTC")
    assert parts["train"]["timestamp_utc"].min() == pd.Timestamp("2020-01-01", tz="UTC")


# ---- expanding-window folds ------------------------------------------------
def test_expanding_window_folds_match_example():
    folds = bt.expanding_window_folds("2023-01-01", "2023-07-01", "2023-12-01")
    assert len(folds) == 6  # Jul..Dec
    # Fold 0: train [Jan, Jul), validate Jul.
    assert folds[0].train_start == pd.Timestamp("2023-01-01", tz="UTC")
    assert folds[0].train_end == pd.Timestamp("2023-07-01", tz="UTC")
    assert folds[0].val_start == pd.Timestamp("2023-07-01", tz="UTC")
    assert folds[0].val_end == pd.Timestamp("2023-08-01", tz="UTC")


def test_folds_expand_and_train_start_fixed():
    folds = bt.expanding_window_folds("2023-01-01", "2023-07-01", "2023-12-01")
    starts = {f.train_start for f in folds}
    assert len(starts) == 1  # train_start never moves
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)  # training window grows monotonically
    # No leakage: train_end == val_start (half-open, so train excludes val).
    for f in folds:
        assert f.train_end == f.val_start


def test_iter_folds_slices_are_disjoint_and_ordered():
    df = _series_df()
    folds = bt.expanding_window_folds("2023-01-01", "2023-07-01", "2023-08-01")
    for fold, train_df, val_df in bt.iter_folds(df, folds):
        assert train_df["timestamp_utc"].max() < val_df["timestamp_utc"].min()
        assert len(val_df) > 0


def test_gap_creates_embargo():
    folds = bt.expanding_window_folds(
        "2023-01-01", "2023-07-01", "2023-07-01", gap=pd.Timedelta(days=1)
    )
    f = folds[0]
    assert f.train_end == pd.Timestamp("2023-06-30", tz="UTC")  # 1 day before val start


# ---- forecast origins ------------------------------------------------------
def test_forecast_origins_one_per_day():
    origins = bt.forecast_origins("2024-07-01", "2024-08-01", per_day=1)
    assert len(origins) == 31  # July has 31 days
    assert (origins.hour == 0).all()


def test_forecast_origins_multiple_per_day():
    origins = bt.forecast_origins("2024-07-01", "2024-07-02", per_day=4)
    assert list(origins.hour) == [0, 6, 12, 18]


def test_forecast_origins_respects_bounds():
    origins = bt.forecast_origins("2024-07-01", "2024-07-03", per_day=1)
    assert origins.min() >= pd.Timestamp("2024-07-01", tz="UTC")
    assert origins.max() < pd.Timestamp("2024-07-03", tz="UTC")


# ---- horizon targets -------------------------------------------------------
def test_horizon_targets_96_steps():
    tgt = bt.horizon_targets("2024-07-01 00:00", horizon=96)
    assert len(tgt) == 96
    assert tgt[0] == pd.Timestamp("2024-07-01 00:30", tz="UTC")
    assert tgt[-1] == pd.Timestamp("2024-07-03 00:00", tz="UTC")  # +48h
    assert (tgt.to_series().diff().dropna() == pd.Timedelta(minutes=30)).all()
