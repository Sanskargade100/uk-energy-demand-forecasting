"""Tests for the raw-data validator. Each defect is injected and asserted."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from energy_forecasting.data import validate as v


def _clean_frame(n=48, start="2021-06-15 00:00"):
    """A tidy, valid one-day half-hourly demand frame."""
    ts = pd.date_range(start, periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "settlement_date": pd.Timestamp("2021-06-15"),
            "settlement_period": range(1, n + 1),
            "nd_mw": [30000 + i for i in range(n)],
        }
    )


def _check(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


def test_clean_frame_passes_all_fail_checks():
    report = v.validate_demand(_clean_frame(), start="2021-06-15", end="2021-06-15")
    assert report["ok"] is True
    assert report["n_failed"] == 0
    assert report["rows_removed"] == 0 and report["rows_changed"] == 0


def test_missing_required_column_flagged():
    df = _clean_frame().drop(columns=["nd_mw"])
    report = v.validate_demand(df)
    assert not report["ok"]
    assert _check(report, "required_columns")["missing"] == ["nd_mw"]


def test_non_numeric_demand_flagged():
    df = _clean_frame()
    df["nd_mw"] = df["nd_mw"].astype(str)
    report = v.validate_demand(df)
    assert _check(report, "demand_numeric")["passed"] is False


def test_unordered_timestamps_flagged():
    df = _clean_frame().iloc[::-1].reset_index(drop=True)
    report = v.validate_demand(df)
    c = _check(report, "timestamps_ordered")
    assert c["passed"] is False and c["n_out_of_order"] > 0


def test_negative_demand_flagged():
    df = _clean_frame()
    df.loc[3, "nd_mw"] = -5
    report = v.validate_demand(df)
    c = _check(report, "no_negative_demand")
    assert c["passed"] is False and c["n_negative"] == 1


def test_duplicate_timestamps_identified():
    df = _clean_frame()
    dup = df.iloc[[10]].copy()
    df = pd.concat([df, dup], ignore_index=True).sort_values("timestamp_utc")
    report = v.validate_demand(df)
    c = _check(report, "duplicate_timestamps")
    assert c["passed"] is False and c["n_duplicate_rows"] == 2


def test_missing_timestamps_reported():
    df = _clean_frame().drop(index=[20, 21]).reset_index(drop=True)
    report = v.validate_demand(df)
    c = _check(report, "missing_timestamps")
    assert c["passed"] is False and c["n_missing"] == 2
    assert len(c["examples"]) >= 1


def test_invalid_settlement_period_flagged():
    df = _clean_frame()
    df.loc[0, "settlement_period"] = 99
    report = v.validate_demand(df)
    c = _check(report, "settlement_periods_valid")
    assert c["passed"] is False and c["n_out_of_range"] == 1


def test_units_out_of_envelope_flagged():
    df = _clean_frame()
    df.loc[5, "nd_mw"] = 250000  # looks like a unit error
    report = v.validate_demand(df)
    c = _check(report, "units_consistent")
    assert c["passed"] is False and c["n_out_of_envelope"] == 1


def test_coverage_shortfall_flagged():
    report = v.validate_demand(_clean_frame(), start="2020-01-01", end="2025-12-31")
    c = _check(report, "covers_requested_range")
    assert c["passed"] is False


def test_validator_never_removes_rows():
    df = _clean_frame()
    df.loc[3, "nd_mw"] = -5
    df.loc[0, "settlement_period"] = 99
    before = len(df)
    report = v.validate_demand(df)
    assert report["n_rows"] == before
    assert report["rows_removed"] == 0


def test_write_report_roundtrip(tmp_path):
    report = v.validate_demand(_clean_frame(), start="2021-06-15", end="2021-06-15")
    path = v.write_report(report, tmp_path / "data_validation.json")
    loaded = json.loads(path.read_text())
    assert loaded["ok"] is True
    assert {c["name"] for c in loaded["checks"]} >= {
        "required_columns",
        "demand_numeric",
        "timestamps_ordered",
        "no_negative_demand",
        "duplicate_timestamps",
        "missing_timestamps",
        "settlement_periods_valid",
        "timezone_utc",
        "units_consistent",
    }
