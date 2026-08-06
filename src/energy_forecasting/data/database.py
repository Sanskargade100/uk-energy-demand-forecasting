"""SQLite storage layer (via SQLAlchemy) for the forecasting project.

SQLite keeps the project self-contained and free; the same SQLAlchemy schema can
be pointed at PostgreSQL during deployment by changing the connection URL only.

Default database: ``data/processed/energy_forecasting.db``.

Tables
------
* ``demand_observations``  — half-hourly National Demand (target).
* ``weather_observations`` — half-hourly UK weather aggregates.
* ``calendar_features``    — per-timestamp calendar/holiday features.
* ``model_predictions``    — point forecasts with 80%/95% intervals.
* ``model_metrics``        — evaluation metrics per model/version/split.

Timestamps are stored as ISO-8601 UTC strings (SQLite has no native tz type),
which keeps them unambiguous and portable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine

from ..logging_config import get_logger
from ..settings import PROCESSED_DIR

logger = get_logger(__name__)

DEFAULT_DB_PATH = PROCESSED_DIR / "energy_forecasting.db"

metadata = MetaData()

demand_observations = Table(
    "demand_observations",
    metadata,
    Column("timestamp_utc", String, primary_key=True),
    Column("settlement_date", String, index=True),
    Column("settlement_period", Integer),
    Column("nd_mw", Float),
    Column("nd_mw_is_missing", Integer),
)

weather_observations = Table(
    "weather_observations",
    metadata,
    Column("timestamp_utc", String, primary_key=True),
    Column("temperature_mean", Float),
    Column("temperature_min", Float),
    Column("temperature_max", Float),
    Column("apparent_temperature_mean", Float),
    Column("wind_speed_mean", Float),
    Column("precipitation_total", Float),
    Column("cloud_cover_mean", Float),
)

calendar_features = Table(
    "calendar_features",
    metadata,
    Column("timestamp_utc", String, primary_key=True),
    Column("timestamp_local", String),
    Column("hour", Integer),
    Column("day_of_week", Integer),
    Column("month", Integer),
    Column("day_of_year", Integer),
    Column("is_weekend", Integer),
    Column("is_bank_holiday", Integer),
)

model_predictions = Table(
    "model_predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("forecast_created_at", String, index=True),
    Column("forecast_for", String, index=True),
    Column("horizon_step", Integer),
    Column("model_name", String, index=True),
    Column("point_forecast_mw", Float),
    Column("lower_80_mw", Float),
    Column("upper_80_mw", Float),
    Column("lower_95_mw", Float),
    Column("upper_95_mw", Float),
    Column("model_version", String),
    UniqueConstraint(
        "forecast_created_at",
        "forecast_for",
        "model_name",
        "model_version",
        "horizon_step",
        name="uq_prediction",
    ),
)

model_metrics = Table(
    "model_metrics",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model_name", String, index=True),
    Column("model_version", String),
    Column("split", String),
    Column("horizon_step", Integer, nullable=True),
    Column("metric_name", String),
    Column("metric_value", Float),
    Column("evaluated_at", String),
)

# Which columns in each writer hold timestamps (serialized to ISO strings).
_DEMAND_COLS = ["timestamp_utc", "settlement_date", "settlement_period", "nd_mw", "nd_mw_is_missing"]
_WEATHER_COLS = [
    "timestamp_utc",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "apparent_temperature_mean",
    "wind_speed_mean",
    "precipitation_total",
    "cloud_cover_mean",
]


# ---------------------------------------------------------------------------
# Engine / schema
# ---------------------------------------------------------------------------
def get_engine(db_path: str | Path | None = None) -> Engine:
    """Create a SQLAlchemy engine for a SQLite file (default project DB)."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}")


def init_db(engine: Engine | None = None, db_path: str | Path | None = None) -> Engine:
    """Create all tables if they do not exist; return the engine."""
    engine = engine or get_engine(db_path)
    metadata.create_all(engine)
    logger.info("Initialised %d tables in %s", len(metadata.tables), engine.url)
    return engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iso(series: pd.Series) -> pd.Series:
    """Serialize a datetime-like series to ISO-8601 strings (None for NaT)."""
    return series.map(lambda x: None if pd.isna(x) else pd.Timestamp(x).isoformat())


def _write(df: pd.DataFrame, table: str, engine: Engine, truncate: bool = False) -> int:
    """Append a frame to a table (optionally clearing it first). Returns row count."""
    if truncate:
        with engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {table}"))
    df.to_sql(table, engine, if_exists="append", index=False)
    return len(df)


# ---------------------------------------------------------------------------
# Loading the processed table into the observation/feature tables
# ---------------------------------------------------------------------------
def load_processed(processed: pd.DataFrame, engine: Engine, truncate: bool = True) -> dict[str, int]:
    """Split the processed 30-min table into demand/weather/calendar tables."""
    df = processed.copy()

    demand = df[[c for c in _DEMAND_COLS if c in df.columns]].copy()
    demand["timestamp_utc"] = _iso(demand["timestamp_utc"])
    if "settlement_date" in demand.columns:
        demand["settlement_date"] = _iso(demand["settlement_date"])

    weather = df[[c for c in _WEATHER_COLS if c in df.columns]].copy()
    weather["timestamp_utc"] = _iso(weather["timestamp_utc"])

    local = pd.to_datetime(df["timestamp_local"])
    calendar = pd.DataFrame(
        {
            "timestamp_utc": _iso(df["timestamp_utc"]),
            "timestamp_local": _iso(df["timestamp_local"]),
            "hour": local.dt.hour,
            "day_of_week": local.dt.dayofweek,
            "month": local.dt.month,
            "day_of_year": local.dt.dayofyear,
            "is_weekend": df.get("is_weekend"),
            "is_bank_holiday": df.get("is_bank_holiday"),
        }
    )

    counts = {
        "demand_observations": _write(demand, "demand_observations", engine, truncate),
        "weather_observations": _write(weather, "weather_observations", engine, truncate),
        "calendar_features": _write(calendar, "calendar_features", engine, truncate),
    }
    logger.info("Loaded processed data into SQL: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# Predictions / metrics
# ---------------------------------------------------------------------------
PREDICTION_COLUMNS = [
    "forecast_created_at",
    "forecast_for",
    "horizon_step",
    "model_name",
    "point_forecast_mw",
    "lower_80_mw",
    "upper_80_mw",
    "lower_95_mw",
    "upper_95_mw",
    "model_version",
]


def insert_predictions(df: pd.DataFrame, engine: Engine) -> int:
    """Append rows to ``model_predictions`` (timestamps serialized to ISO)."""
    out = df.copy()
    for col in ("forecast_created_at", "forecast_for"):
        if col in out.columns:
            out[col] = _iso(out[col])
    out = out[[c for c in PREDICTION_COLUMNS if c in out.columns]]
    return _write(out, "model_predictions", engine)


def insert_metrics(df: pd.DataFrame, engine: Engine) -> int:
    """Append rows to ``model_metrics`` (``evaluated_at`` serialized to ISO)."""
    out = df.copy()
    if "evaluated_at" in out.columns:
        out["evaluated_at"] = _iso(out["evaluated_at"])
    return _write(out, "model_metrics", engine)


def read_table(name: str, engine: Engine) -> pd.DataFrame:
    """Read a whole table back into a DataFrame."""
    return pd.read_sql_table(name, engine)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Initialise the DB and load data/processed/energy_demand_30min.parquet."""
    from ..logging_config import setup_logging

    setup_logging()
    engine = init_db()

    processed_path = PROCESSED_DIR / "energy_demand_30min.parquet"
    if not processed_path.exists():
        logger.error("No processed file at %s — run prepare_data first.", processed_path)
        return 2

    load_processed(pd.read_parquet(processed_path), engine)
    logger.info("Database ready at %s", DEFAULT_DB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
