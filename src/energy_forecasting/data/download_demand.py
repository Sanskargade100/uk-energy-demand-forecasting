"""Download NESO Historic Demand Data (half-hourly GB National Demand, ``ND`` in MW).

Source
------
NESO Data Portal, "Historic Demand Data" dataset
(id ``8f2fe0af-871c-488d-8bad-960426f24601``), one CSV resource per calendar year.
Used under the NESO Open Data Licence; data is fetched at run time, not vendored.

What this module does (see :func:`download_demand`)
---------------------------------------------------
1. Requests the CKAN metadata endpoint ``datapackage_show``.
2. Extracts the per-year CSV resource URLs.
3. Selects the years covering the requested date range.
4. Downloads each yearly file.
5. Saves each *original* CSV under ``data/raw/demand/``.
6. Records download time and source URL in ``data/raw/demand/_download_manifest.json``.
7. Skips files already present unless ``force=True``.

It then standardizes the raw CSVs into a single tidy frame with lowercase,
unit-suffixed column names (``nd_mw`` is the target) and a DST-correct UTC
timestamp, saved to ``data/interim/demand.parquet``.

Key facts about the source
---------------------------
* Rows are keyed by ``SETTLEMENT_DATE`` (a calendar date) and ``SETTLEMENT_PERIOD``
  (1-based half-hour index within the *local* London day: 1..48 normally, 1..46 on
  the spring clock-change day, 1..50 on the autumn one).
* ``ND`` is National Demand in MW — the target of this project.
* The exact column set grows over the years, so we only *require* the demand
  columns and keep whichever preferred extras are present.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..logging_config import get_logger
from ..settings import INTERIM_DIR, RAW_DIR, get_settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Dataset constants (verified from the NESO Data Portal, Aug 2026)
# ---------------------------------------------------------------------------
DATASET_ID = "8f2fe0af-871c-488d-8bad-960426f24601"
DATASET_SLUG = "historic-demand-data"

#: Fallback map of year -> CKAN resource id, used only if live discovery via
#: ``datapackage_show`` fails. Discovery is preferred so new years appear
#: automatically.
RESOURCE_IDS: dict[int, str] = {
    2016: "3bb75a28-ab44-4a0b-9b1c-9be9715d3c44",
    2017: "2f0f75b8-39c5-46ff-a914-ae38088ed022",
    2018: "fcb12133-0db0-4f27-a4a5-1669fd9f6d33",
    2019: "dd9de980-d724-415a-b344-d8ae11321432",
    2020: "33ba6857-2a55-479f-9308-e5c4c53d4381",
    2021: "18c69c42-f20d-46f0-84e9-e279045befc6",
    2022: "bb44a1b5-75b1-4db2-8491-257f23385006",
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
    2025: "b2bde559-3455-4021-b179-dfe60c0337b0",
    2026: "8a4a771c-3929-4e56-93ad-cdf13219dea5",
}

# Required source columns (uppercase, as delivered). ND is the target.
REQUIRED_COLUMNS = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]

# Source column -> standardized lowercase name. Extras are kept only if present.
COLUMN_RENAME: dict[str, str] = {
    "SETTLEMENT_DATE": "settlement_date",
    "SETTLEMENT_PERIOD": "settlement_period",
    "ND": "nd_mw",
    "TSD": "tsd_mw",
    "ENGLAND_WALES_DEMAND": "england_wales_demand_mw",
    "EMBEDDED_WIND_GENERATION": "embedded_wind_mw",
    "EMBEDDED_SOLAR_GENERATION": "embedded_solar_mw",
}

LONDON = "Europe/London"
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def demand_raw_dir() -> Path:
    """Directory holding the original yearly CSVs: ``data/raw/demand/``."""
    return RAW_DIR / "demand"


def manifest_path(dest_dir: Path | None = None) -> Path:
    base = Path(dest_dir) if dest_dir else demand_raw_dir()
    return base / "_download_manifest.json"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _make_session(retries: int = 4, backoff: float = 1.0) -> requests.Session:
    """A requests session with retry/backoff for transient network failures."""
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


def build_csv_url(resource_id: str, base_url: str | None = None) -> str:
    """Construct the direct CSV download URL for a resource id (fallback path)."""
    base = (base_url or get_settings().neso_base_url).rstrip("/")
    return f"{base}/dataset/{DATASET_ID}/resource/{resource_id}/download/demanddata.csv"


def _year_from_text(text: str) -> int | None:
    match = _YEAR_RE.search(text or "")
    return int(match.group(0)) if match else None


def discover_resource_urls(
    base_url: str | None = None, session: requests.Session | None = None
) -> dict[int, str]:
    """Discover year -> CSV URL from the CKAN ``datapackage_show`` endpoint.

    Falls back to URLs built from :data:`RESOURCE_IDS` on any failure so the
    pipeline still runs if the API is unreachable or its shape changes.
    """
    base = (base_url or get_settings().neso_base_url).rstrip("/")
    url = f"{base}/api/3/action/datapackage_show?id={DATASET_SLUG}"
    sess = session or _make_session()

    urls: dict[int, str] = {}
    try:
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        result = payload.get("result", payload)
        for res in result.get("resources", []):
            csv_url = res.get("path") or res.get("url")
            if not csv_url or not str(csv_url).lower().endswith(".csv"):
                continue
            year = _year_from_text(res.get("name", "")) or _year_from_text(csv_url)
            if year:
                urls[year] = csv_url
        if urls:
            logger.info("Discovered %d yearly CSV resources via datapackage_show.", len(urls))
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Resource discovery failed (%s); using built-in fallback.", exc)

    # Fill any gaps from the fallback id map.
    for year, rid in RESOURCE_IDS.items():
        urls.setdefault(year, build_csv_url(rid, base_url))
    return urls


# ---------------------------------------------------------------------------
# Download + manifest
# ---------------------------------------------------------------------------
def _read_manifest(dest_dir: Path | None = None) -> dict[str, dict]:
    path = manifest_path(dest_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Manifest was corrupt; starting a fresh one.")
    return {}


def _write_manifest(manifest: dict[str, dict], dest_dir: Path | None = None) -> None:
    path = manifest_path(dest_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def download_year_file(
    year: int,
    url: str,
    session: requests.Session,
    dest_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download one year's CSV to ``data/raw/demand/`` and record the manifest.

    Skips the network entirely if the file already exists and ``force`` is False.
    """
    out_dir = Path(dest_dir) if dest_dir else demand_raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"demanddata_{year}.csv"

    if dest.exists() and not force:
        logger.info("  %d: cached (%s) — skipping download.", year, dest.name)
        return dest

    logger.info("  %d: downloading %s", year, url)
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)

    manifest = _read_manifest(out_dir)
    manifest[str(year)] = {
        "year": year,
        "source_url": url,
        "filename": dest.name,
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_bytes": dest.stat().st_size,
    }
    _write_manifest(manifest, out_dir)
    return dest


# ---------------------------------------------------------------------------
# Parsing / standardization
# ---------------------------------------------------------------------------
def settlement_to_utc(dates: pd.Series, periods: pd.Series) -> pd.Series:
    """Convert (settlement_date, settlement_period) to a UTC timestamp.

    DST-correct: we localize only *local midnight* (never ambiguous in the UK,
    since transitions occur at 01:00), then add absolute 30-minute offsets. This
    yields the right instants across spring-forward (46-period) and autumn
    (50-period) days.
    """
    midnight_local = pd.to_datetime(dates).dt.tz_localize(LONDON)
    offsets = pd.to_timedelta((periods.astype(int) - 1) * 30, unit="m")
    return (midnight_local + offsets).dt.tz_convert("UTC")


def standardize_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to lowercase ``*_mw`` columns, add ``timestamp_utc``, type + dedup.

    Output columns: ``timestamp_utc``, ``settlement_date``, ``settlement_period``,
    ``nd_mw`` and whichever preferred extras were present.
    """
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Demand data missing required columns: {missing}")

    keep = [c for c in COLUMN_RENAME if c in df.columns]
    df = df[keep].rename(columns=COLUMN_RENAME)

    df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
    df["settlement_period"] = pd.to_numeric(df["settlement_period"], errors="coerce")
    df = df.dropna(subset=["settlement_date", "settlement_period"])
    df["settlement_period"] = df["settlement_period"].astype(int)
    df = df[(df["settlement_period"] >= 1) & (df["settlement_period"] <= 50)]

    for col in df.columns:
        if col.endswith("_mw"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.insert(0, "timestamp_utc", settlement_to_utc(df["settlement_date"], df["settlement_period"]))
    df = (
        df.sort_values("timestamp_utc")
        .drop_duplicates(subset=["settlement_date", "settlement_period"], keep="last")
        .reset_index(drop=True)
    )
    return df


def parse_demand_csv(raw: str | bytes) -> pd.DataFrame:
    """Parse a raw NESO demand CSV (bytes or text) into a standardized frame."""
    buf = io.BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else io.StringIO(raw)
    return standardize_demand(pd.read_csv(buf))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def download_demand(
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    force: bool = False,
    base_url: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Download half-hourly ND for a date range and return a tidy frame.

    Original yearly CSVs are cached under ``data/raw/demand/`` (re-downloaded only
    when ``force=True``); the standardized, concatenated result is written to
    ``data/interim/demand.parquet`` and trimmed to ``[start, end]``.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)  # inclusive end date
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)

    session = _make_session()
    urls = discover_resource_urls(base_url, session)

    frames = []
    for year in years:
        if year not in urls:
            raise KeyError(f"No NESO CSV URL known for year {year}.")
        path = download_year_file(year, urls[year], session, force=force)
        frames.append(standardize_demand(pd.read_csv(path)))

    demand = pd.concat(frames, ignore_index=True)
    demand = demand[
        (demand["timestamp_utc"] >= start_ts) & (demand["timestamp_utc"] < end_ts)
    ].reset_index(drop=True)

    logger.info(
        "Assembled demand: %d rows, %s -> %s",
        len(demand),
        demand["timestamp_utc"].min(),
        demand["timestamp_utc"].max(),
    )

    if save:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        out_path = INTERIM_DIR / "demand.parquet"
        demand.to_parquet(out_path, index=False)
        logger.info("Saved standardized demand -> %s", out_path)

    return demand
