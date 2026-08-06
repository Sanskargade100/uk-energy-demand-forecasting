"""Clean demand and combine it with weather and holidays into one processed table.

Pipeline (see :func:`clean_and_combine`)
----------------------------------------
1. Standardise column names (lowercase).
2. Convert demand to numeric.
3. Remove exact duplicate records.
4. Handle daylight-saving timestamps (UTC storage + a ``timestamp_local`` view;
   settlement date/period re-derived so 46/50-period days stay correct).
5. Sort chronologically.
6. Merge demand, weather and holidays.
7. Reindex onto a consistent 30-minute UTC grid.
8. Add missing-value indicators (demand gaps are kept as NaN, not imputed here).
9. Save to ``data/processed/energy_demand_30min.parquet`` (Parquet preserves types
   and is more storage-efficient than CSV).

Recommended schema
------------------
``timestamp_utc, timestamp_local, settlement_date, settlement_period, nd_mw,
temperature_mean, temperature_min, temperature_max, apparent_temperature_mean,
wind_speed_mean, precipitation_total, cloud_cover_mean, is_weekend,
is_bank_holiday`` (plus ``nd_mw_is_missing`` and ``weather_is_missing`` flags).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..logging_config import get_logger
from ..settings import EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR
from ..utils.time_utils import LONDON, UTC

logger = get_logger(__name__)

# interim/external -> schema column mapping for weather.
WEATHER_RENAME = {
    "uk_mean_temperature": "temperature_mean",
    "uk_min_temperature": "temperature_min",
    "uk_max_temperature": "temperature_max",
    "uk_mean_apparent_temperature": "apparent_temperature_mean",
    "uk_mean_wind_speed": "wind_speed_mean",
    "uk_total_precipitation": "precipitation_total",
    "uk_mean_cloud_cover": "cloud_cover_mean",
}
WEATHER_COLUMNS = list(WEATHER_RENAME.values())

FINAL_SCHEMA = [
    "timestamp_utc",
    "timestamp_local",
    "settlement_date",
    "settlement_period",
    "nd_mw",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "apparent_temperature_mean",
    "wind_speed_mean",
    "precipitation_total",
    "cloud_cover_mean",
    "is_weekend",
    "is_bank_holiday",
    "nd_mw_is_missing",
    "weather_is_missing",
]


# ---------------------------------------------------------------------------
# Demand cleaning (steps 1-5)
# ---------------------------------------------------------------------------
def standardize_demand_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and coerce ``nd_mw`` to numeric."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "nd_mw" in df.columns:
        df["nd_mw"] = pd.to_numeric(df["nd_mw"], errors="coerce")
    return df


def clean_demand(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardise, drop exact duplicates, ensure UTC, sort, add ``timestamp_local``."""
    df = standardize_demand_columns(df)
    n_in = len(df)

    df = df.drop_duplicates()
    exact_removed = n_in - len(df)

    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize(UTC)
    else:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert(UTC)

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(LONDON)

    stats = {"n_in": n_in, "exact_duplicates_removed": exact_removed, "n_out": len(df)}
    return df, stats


# ---------------------------------------------------------------------------
# Grid + settlement derivation (DST-correct)
# ---------------------------------------------------------------------------
def build_grid(
    demand: pd.DataFrame, start: str | None = None, end: str | None = None
) -> pd.DatetimeIndex:
    """A gap-free 30-minute UTC index spanning the data (or the given range)."""
    ts = demand["timestamp_utc"]
    lo = pd.Timestamp(start, tz=UTC) if start else ts.min()
    hi = (
        pd.Timestamp(end, tz=UTC) + pd.Timedelta(days=1)
        if end
        else ts.max() + pd.Timedelta(minutes=30)
    )
    return pd.date_range(lo, hi, freq="30min", inclusive="left")


def derive_settlement(ts_utc: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Re-derive (settlement_date, settlement_period) from UTC timestamps.

    Mirrors the forward construction in ``time_utils`` so DST days yield 46/50
    periods correctly, and so grid rows with no demand still get a valid date and
    period rather than NaN.
    """
    local = ts_utc.dt.tz_convert(LONDON)
    local_midnight = local.dt.normalize()  # tz-aware local midnight
    midnight_utc = local_midnight.dt.tz_convert(UTC)
    period = ((ts_utc - midnight_utc) / pd.Timedelta(minutes=30)).round().astype(int) + 1
    settlement_date = local_midnight.dt.tz_localize(None)
    return settlement_date, period


# ---------------------------------------------------------------------------
# Combine (steps 6-8)
# ---------------------------------------------------------------------------
def combine(
    demand: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    holidays: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge cleaned demand with weather and holidays on a 30-minute UTC grid."""
    demand, dstats = clean_demand(demand)
    grid = build_grid(demand, start, end)

    out = pd.DataFrame({"timestamp_utc": grid})
    out["timestamp_local"] = out["timestamp_utc"].dt.tz_convert(LONDON)
    out["settlement_date"], out["settlement_period"] = derive_settlement(out["timestamp_utc"])

    # Demand aligned onto the grid (missing slots -> NaN).
    nd = demand.drop_duplicates("timestamp_utc").set_index("timestamp_utc")["nd_mw"]
    out["nd_mw"] = out["timestamp_utc"].map(nd)

    # Weather aligned onto the grid.
    if weather is not None and len(weather):
        wx = weather.copy()
        wx.columns = [c.strip().lower() for c in wx.columns]
        wx = wx.rename(columns=WEATHER_RENAME)
        if wx["timestamp_utc"].dt.tz is None:
            wx["timestamp_utc"] = wx["timestamp_utc"].dt.tz_localize(UTC)
        wx = wx.drop_duplicates("timestamp_utc").set_index("timestamp_utc")
        for col in WEATHER_COLUMNS:
            out[col] = out["timestamp_utc"].map(wx[col]) if col in wx.columns else pd.NA
    else:
        for col in WEATHER_COLUMNS:
            out[col] = pd.NA

    # Calendar / holiday flags.
    out["is_weekend"] = (out["timestamp_local"].dt.dayofweek >= 5).astype(int)
    if holidays is not None and len(holidays):
        h = holidays.copy()
        # A GB bank holiday = England&Wales or Scotland (NI is outside the GB grid).
        gb = ((h["is_england_wales_holiday"] > 0) | (h["is_scotland_holiday"] > 0)).astype(int)
        gb_by_date = pd.Series(gb.values, index=pd.to_datetime(h["date"]))
        out["is_bank_holiday"] = out["settlement_date"].map(gb_by_date).fillna(0).astype(int)
    else:
        out["is_bank_holiday"] = 0

    # Missing-value indicators (do NOT impute here).
    out["nd_mw_is_missing"] = out["nd_mw"].isna().astype(int)
    out["weather_is_missing"] = out[WEATHER_COLUMNS].isna().any(axis=1).astype(int)

    out = out[FINAL_SCHEMA]

    stats = {
        **dstats,
        "grid_rows": len(out),
        "demand_missing_on_grid": int(out["nd_mw_is_missing"].sum()),
        "weather_missing_on_grid": int(out["weather_is_missing"].sum()),
        "bank_holiday_slots": int(out["is_bank_holiday"].sum()),
    }
    return out, stats


# ---------------------------------------------------------------------------
# Orchestration (step 9)
# ---------------------------------------------------------------------------
def clean_and_combine(
    start: str | None = None,
    end: str | None = None,
    save: bool = True,
    interim_dir: Path | None = None,
    external_dir: Path | None = None,
) -> pd.DataFrame:
    """Load interim demand/weather + external holidays, combine, and save."""
    idir = Path(interim_dir) if interim_dir else INTERIM_DIR
    edir = Path(external_dir) if external_dir else EXTERNAL_DIR

    demand = pd.read_parquet(idir / "demand.parquet")

    weather_path = idir / "weather_30min.parquet"
    weather = pd.read_parquet(weather_path) if weather_path.exists() else None
    if weather is None:
        logger.warning("No weather file at %s — weather columns will be empty.", weather_path)

    holidays_path = edir / "uk_bank_holidays.parquet"
    holidays = pd.read_parquet(holidays_path) if holidays_path.exists() else None
    if holidays is None:
        logger.warning("No holidays file at %s — is_bank_holiday will be 0.", holidays_path)

    combined, stats = combine(demand, weather, holidays, start=start, end=end)
    logger.info("Combined table: %s", stats)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "energy_demand_30min.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("Saved -> %s", out_path)

    return combined
