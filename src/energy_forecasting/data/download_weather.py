"""Download UK weather from Open-Meteo and build a half-hourly UK-wide feature set.

National demand is not driven by one city's weather, so we pull several
population centres and aggregate them into UK-wide statistics.

Two sources (see the ``source`` argument)
------------------------------------------
* ``"archive"`` — Open-Meteo Historical Weather API (ERA5 reanalysis / observed).
  Use for **exploratory** work. These are best-estimate *actuals* and would not all
  have been known at prediction time.
* ``"forecast"`` — Open-Meteo Historical Forecast API (archived past forecasts).
  Use for a **portfolio-quality backtest**: it reflects the weather forecast that
  was actually available operationally, avoiding leakage of future weather into
  the model.

Neither endpoint requires an API key for ordinary non-commercial use.

Output
------
UK aggregates per timestamp — ``uk_mean_temperature``, ``uk_min_temperature``,
``uk_max_temperature``, ``uk_temperature_std``, ``uk_mean_apparent_temperature``,
``uk_mean_wind_speed``, ``uk_total_precipitation``, ``uk_mean_cloud_cover`` —
resampled from hourly to 30-minute resolution and written to
``data/interim/weather_30min.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..logging_config import get_logger
from ..settings import INTERIM_DIR

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Endpoints, locations and variables
# ---------------------------------------------------------------------------
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

SOURCE_ENDPOINTS = {"archive": ARCHIVE_URL, "forecast": HISTORICAL_FORECAST_URL}

#: Population centres spread across GB (name, latitude, longitude).
LOCATIONS: list[dict] = [
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Birmingham", "lat": 52.4862, "lon": -1.8904},
    {"name": "Manchester", "lat": 53.4808, "lon": -2.2426},
    {"name": "Bristol", "lat": 51.4545, "lon": -2.5879},
    {"name": "Glasgow", "lat": 55.8642, "lon": -4.2518},
    {"name": "Edinburgh", "lat": 55.9533, "lon": -3.1883},
]

#: Hourly variables requested from Open-Meteo (their exact names).
WEATHER_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
]

PRECIP_COL = "uk_total_precipitation"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _make_session(retries: int = 4, backoff: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "uk-energy-demand-forecasting/0.1"})
    return session


def fetch_city(
    lat: float,
    lon: float,
    start: str,
    end: str,
    source: str = "archive",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch one city's hourly weather and return a parsed, UTC-indexed frame."""
    if source not in SOURCE_ENDPOINTS:
        raise ValueError(f"source must be one of {list(SOURCE_ENDPOINTS)}, got {source!r}")
    sess = session or _make_session()
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
    }
    resp = sess.get(SOURCE_ENDPOINTS[source], params=params, timeout=60)
    resp.raise_for_status()
    return parse_open_meteo_response(resp.json())


# ---------------------------------------------------------------------------
# Pure transforms (unit-tested without network)
# ---------------------------------------------------------------------------
def parse_open_meteo_response(payload: dict) -> pd.DataFrame:
    """Turn an Open-Meteo JSON payload into a UTC-indexed hourly frame."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Open-Meteo response has no 'hourly' block.")
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
    df = df.rename(columns={"time": "timestamp_utc"}).set_index("timestamp_utc")
    missing = [v for v in WEATHER_VARIABLES if v not in df.columns]
    if missing:
        raise ValueError(f"Open-Meteo response missing variables: {missing}")
    return df[WEATHER_VARIABLES].astype(float)


def aggregate_uk(city_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-city hourly weather into UK-wide statistics.

    Precipitation is summed across cities (hence *total*); the other variables are
    reduced with mean/min/max/std as named. Cities are aligned on the union of
    their timestamps.
    """
    if not city_frames:
        raise ValueError("No city frames supplied.")
    index = sorted(set().union(*(f.index for f in city_frames.values())))

    def var_matrix(var: str) -> pd.DataFrame:
        return pd.DataFrame(
            {name: f[var].reindex(index) for name, f in city_frames.items()}, index=index
        )

    temp = var_matrix("temperature_2m")
    out = pd.DataFrame(index=pd.DatetimeIndex(index, name="timestamp_utc"))
    out["uk_mean_temperature"] = temp.mean(axis=1).to_numpy()
    out["uk_min_temperature"] = temp.min(axis=1).to_numpy()
    out["uk_max_temperature"] = temp.max(axis=1).to_numpy()
    out["uk_temperature_std"] = temp.std(axis=1).to_numpy()
    out["uk_mean_apparent_temperature"] = var_matrix("apparent_temperature").mean(axis=1).to_numpy()
    out["uk_mean_wind_speed"] = var_matrix("wind_speed_10m").mean(axis=1).to_numpy()
    out[PRECIP_COL] = var_matrix("precipitation").sum(axis=1).to_numpy()
    out["uk_mean_cloud_cover"] = var_matrix("cloud_cover").mean(axis=1).to_numpy()
    return out


def resample_to_30min(hourly: pd.DataFrame) -> pd.DataFrame:
    """Upsample hourly UK aggregates to a 30-minute grid.

    Continuous variables are linearly interpolated in time; the precipitation
    *total* is held constant within its hour (step / forward-fill) so its meaning
    as an hourly accumulation is preserved.
    """
    df = hourly.sort_index()
    target = pd.date_range(df.index.min(), df.index.max(), freq="30min", tz="UTC")
    grid = df.reindex(df.index.union(target)).sort_index()

    continuous = [c for c in grid.columns if c != PRECIP_COL]
    grid[continuous] = grid[continuous].interpolate(method="time")
    if PRECIP_COL in grid.columns:
        grid[PRECIP_COL] = grid[PRECIP_COL].ffill()

    out = grid.reindex(target).ffill().bfill()
    out.index.name = "timestamp_utc"
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def download_weather(
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    source: str = "archive",
    locations: list[dict] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Download, aggregate and resample UK weather; return the 30-minute frame."""
    locs = locations if locations is not None else LOCATIONS
    session = _make_session()

    city_frames: dict[str, pd.DataFrame] = {}
    for loc in locs:
        logger.info("Fetching weather (%s) for %s ...", source, loc["name"])
        city_frames[loc["name"]] = fetch_city(
            loc["lat"], loc["lon"], start, end, source=source, session=session
        )

    uk_hourly = aggregate_uk(city_frames)
    weather_30min = resample_to_30min(uk_hourly)
    logger.info(
        "Weather assembled: %d half-hourly rows, %s -> %s",
        len(weather_30min),
        weather_30min.index.min(),
        weather_30min.index.max(),
    )

    if save:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        out_path = INTERIM_DIR / "weather_30min.parquet"
        weather_30min.reset_index().to_parquet(out_path, index=False)
        logger.info("Saved -> %s", out_path)

    return weather_30min
