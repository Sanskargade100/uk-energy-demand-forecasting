"""Unit tests for the NESO demand downloader (no real network required)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from energy_forecasting.data import download_demand as dl

# A tiny, well-formed sample mirroring the NESO CSV layout.
SAMPLE_CSV = (
    "SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND,TSD,ENGLAND_WALES_DEMAND,"
    "EMBEDDED_WIND_GENERATION,EMBEDDED_SOLAR_GENERATION\n"
    "2021-01-01,1,25000,26000,22000,1200,0\n"
    "2021-01-01,2,24800,25800,21800,1250,0\n"
    "2021-01-01,3,24600,25600,21600,1300,0\n"
)


# ---- fakes -----------------------------------------------------------------
class _Resp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class _Session:
    """Records GET calls and returns the sample CSV bytes."""

    def __init__(self):
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        return _Resp(SAMPLE_CSV.encode())


# ---- URL / discovery -------------------------------------------------------
def test_build_csv_url_contains_dataset_and_resource():
    url = dl.build_csv_url("abc-123", base_url="https://api.neso.energy")
    assert dl.DATASET_ID in url
    assert "abc-123" in url
    assert url.startswith("https://api.neso.energy/dataset/")


# ---- parsing / standardization ---------------------------------------------
def test_parse_renames_to_mw_lowercase():
    df = dl.parse_demand_csv(SAMPLE_CSV)
    assert list(df.columns) == [
        "timestamp_utc",
        "settlement_date",
        "settlement_period",
        "nd_mw",
        "tsd_mw",
        "england_wales_demand_mw",
        "embedded_wind_mw",
        "embedded_solar_mw",
    ]
    assert df["nd_mw"].tolist() == [25000, 24800, 24600]


def test_parse_timestamps_utc_and_spacing():
    df = dl.parse_demand_csv(SAMPLE_CSV)
    assert str(df["timestamp_utc"].iloc[0].tz) == "UTC"
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2021-01-01 00:00:00", tz="UTC")
    deltas = df["timestamp_utc"].diff().dropna().unique()
    assert list(deltas) == [pd.Timedelta(minutes=30)]


def test_parse_missing_required_column_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        dl.parse_demand_csv("SETTLEMENT_DATE,SETTLEMENT_PERIOD\n2021-01-01,1\n")


def test_parse_keeps_only_available_extras():
    minimal = "SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND\n2021-01-01,1,25000\n"
    df = dl.parse_demand_csv(minimal)
    assert list(df.columns) == ["timestamp_utc", "settlement_date", "settlement_period", "nd_mw"]


def test_settlement_to_utc_summer_bst_offset():
    out = dl.settlement_to_utc(pd.Series(pd.to_datetime(["2021-07-01"])), pd.Series([1]))
    assert out.iloc[0] == pd.Timestamp("2021-06-30 23:00:00", tz="UTC")


def test_settlement_to_utc_spring_forward_day():
    out = dl.settlement_to_utc(pd.Series(pd.to_datetime(["2021-03-28"] * 2)), pd.Series([1, 4]))
    assert out.iloc[1] == pd.Timestamp("2021-03-28 01:30:00", tz="UTC")
    assert out.iloc[1].tz_convert("Europe/London").strftime("%H:%M") == "02:30"


def test_parse_dedups_on_settlement_key_keep_last():
    dup = SAMPLE_CSV + "2021-01-01,3,99999,25600,21600,1300,0\n"
    df = dl.parse_demand_csv(dup)
    assert len(df) == 3
    assert df.loc[df["settlement_period"] == 3, "nd_mw"].iloc[0] == 99999


# ---- download caching + manifest -------------------------------------------
def test_download_year_file_saves_original_and_manifest(tmp_path):
    sess = _Session()
    dest = dl.download_year_file(2021, "https://x/demanddata_2021.csv", sess, dest_dir=tmp_path)
    assert dest.exists() and dest.name == "demanddata_2021.csv"
    assert dest.read_bytes() == SAMPLE_CSV.encode()
    assert sess.calls == 1

    manifest = json.loads((tmp_path / "_download_manifest.json").read_text())
    assert manifest["2021"]["source_url"].endswith("demanddata_2021.csv")
    assert "downloaded_at" in manifest["2021"]
    assert manifest["2021"]["n_bytes"] == len(SAMPLE_CSV.encode())


def test_download_year_file_skips_when_cached(tmp_path):
    sess = _Session()
    dl.download_year_file(2021, "https://x/demanddata_2021.csv", sess, dest_dir=tmp_path)
    # Second call must not hit the network unless force=True.
    dl.download_year_file(2021, "https://x/demanddata_2021.csv", sess, dest_dir=tmp_path)
    assert sess.calls == 1
    dl.download_year_file(
        2021, "https://x/demanddata_2021.csv", sess, dest_dir=tmp_path, force=True
    )
    assert sess.calls == 2
