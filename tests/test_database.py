"""Tests for the SQLite storage layer (uses a temporary DB file)."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import inspect

from energy_forecasting.data import database as db


@pytest.fixture()
def engine(tmp_path):
    return db.init_db(db_path=tmp_path / "test.db")


def _processed(n=4):
    utc = pd.date_range("2021-06-15 00:00", periods=n, freq="30min", tz="UTC")
    local = utc.tz_convert("Europe/London")
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local,
            "settlement_date": pd.Timestamp("2021-06-15"),
            "settlement_period": range(1, n + 1),
            "nd_mw": [30000.0 + i for i in range(n)],
            "temperature_mean": 15.0,
            "temperature_min": 12.0,
            "temperature_max": 18.0,
            "apparent_temperature_mean": 14.0,
            "wind_speed_mean": 5.0,
            "precipitation_total": 0.0,
            "cloud_cover_mean": 50.0,
            "is_weekend": 0,
            "is_bank_holiday": 0,
            "nd_mw_is_missing": 0,
        }
    )


def test_init_db_creates_all_tables(engine):
    tables = set(inspect(engine).get_table_names())
    assert {
        "demand_observations",
        "weather_observations",
        "calendar_features",
        "model_predictions",
        "model_metrics",
    } <= tables


def test_load_processed_populates_three_tables(engine):
    counts = db.load_processed(_processed(4), engine)
    assert counts == {
        "demand_observations": 4,
        "weather_observations": 4,
        "calendar_features": 4,
    }
    demand = db.read_table("demand_observations", engine)
    assert demand["nd_mw"].tolist() == [30000.0, 30001.0, 30002.0, 30003.0]
    cal = db.read_table("calendar_features", engine)
    assert set(cal.columns) >= {"hour", "day_of_week", "month", "is_weekend", "is_bank_holiday"}


def test_load_processed_truncates_on_reload(engine):
    db.load_processed(_processed(4), engine)
    db.load_processed(_processed(4), engine)  # truncate=True default
    assert len(db.read_table("demand_observations", engine)) == 4


def test_insert_and_read_predictions(engine):
    preds = pd.DataFrame(
        {
            "forecast_created_at": [pd.Timestamp("2025-01-01 06:00", tz="UTC")],
            "forecast_for": [pd.Timestamp("2025-01-02 06:00", tz="UTC")],
            "horizon_step": [48],
            "model_name": ["xgboost"],
            "point_forecast_mw": [32000.0],
            "lower_80_mw": [31000.0],
            "upper_80_mw": [33000.0],
            "lower_95_mw": [30500.0],
            "upper_95_mw": [33500.0],
            "model_version": ["0.1.0"],
        }
    )
    assert db.insert_predictions(preds, engine) == 1
    got = db.read_table("model_predictions", engine)
    assert got.loc[0, "model_name"] == "xgboost"
    assert got.loc[0, "point_forecast_mw"] == 32000.0
    assert got.loc[0, "lower_95_mw"] < got.loc[0, "lower_80_mw"]
    assert got.loc[0, "upper_80_mw"] < got.loc[0, "upper_95_mw"]


def test_insert_metrics(engine):
    metrics = pd.DataFrame(
        {
            "model_name": ["xgboost", "seasonal_naive"],
            "model_version": ["0.1.0", "0.1.0"],
            "split": ["test", "test"],
            "horizon_step": [None, None],
            "metric_name": ["mae", "mae"],
            "metric_value": [420.5, 900.1],
            "evaluated_at": [pd.Timestamp.utcnow(), pd.Timestamp.utcnow()],
        }
    )
    assert db.insert_metrics(metrics, engine) == 2
    got = db.read_table("model_metrics", engine)
    assert set(got["model_name"]) == {"xgboost", "seasonal_naive"}


def test_prediction_unique_constraint_blocks_exact_duplicate(engine):
    preds = pd.DataFrame(
        {
            "forecast_created_at": ["2025-01-01T06:00:00"],
            "forecast_for": ["2025-01-02T06:00:00"],
            "horizon_step": [48],
            "model_name": ["xgboost"],
            "point_forecast_mw": [32000.0],
            "lower_80_mw": [31000.0],
            "upper_80_mw": [33000.0],
            "lower_95_mw": [30500.0],
            "upper_95_mw": [33500.0],
            "model_version": ["0.1.0"],
        }
    )
    db.insert_predictions(preds, engine)
    with pytest.raises(Exception):
        db.insert_predictions(preds, engine)  # violates uq_prediction
