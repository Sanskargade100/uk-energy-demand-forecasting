"""Validate the standardized demand data and write a JSON report.

This module *inspects* the data and records what it finds — it never silently
drops or edits rows. Every check reports counts of affected rows so cleaning
decisions are made explicitly and deliberately downstream (in ``clean.py``).

Checks performed
----------------
1. Required columns exist.
2. Demand (``nd_mw``) is numeric.
3. Timestamps are ordered (monotonically increasing).
4. No impossible negative demand.
5. Duplicate timestamps are identified.
6. Missing timestamps (gaps in the 30-minute UTC grid) are reported.
7. Settlement periods are valid (1..50 and within the expected count for each day).
8. Timezone conversion succeeded (``timestamp_utc`` is tz-aware UTC, no NaT).
9. Units are consistent (demand within a plausible GB MW range).
10. Dataset dates cover the requested range.

The result is a dict, also written to ``reports/data_validation.json``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..logging_config import get_logger
from ..settings import INTERIM_DIR, REPORTS_DIR, load_config
from ..utils.time_utils import expected_periods_in_day

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("timestamp_utc", "settlement_date", "settlement_period", "nd_mw")
TARGET = "nd_mw"
# Plausible GB National Demand envelope in MW (historic range ~17k–62k, padded).
PLAUSIBLE_MIN_MW = 1_000
PLAUSIBLE_MAX_MW = 70_000
FREQ = "30min"

# Severity levels
FAIL = "fail"
WARN = "warn"


def _example_timestamps(index: pd.DatetimeIndex, limit: int = 5) -> list[str]:
    return [t.isoformat() for t in list(index)[:limit]]


def validate_demand(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Run all checks and return a structured report. Removes/edits nothing."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, severity: str, message: str, **details: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "severity": severity,
                "message": message,
                **details,
            }
        )

    n_rows = len(df)

    # 1. Required columns ----------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    add(
        "required_columns",
        not missing_cols,
        FAIL,
        "All required columns present." if not missing_cols else f"Missing: {missing_cols}",
        required=list(REQUIRED_COLUMNS),
        missing=missing_cols,
    )
    has_target = TARGET in df.columns
    has_ts = "timestamp_utc" in df.columns
    has_period = "settlement_period" in df.columns

    # 2. Demand is numeric ---------------------------------------------------
    if has_target:
        numeric = pd.api.types.is_numeric_dtype(df[TARGET])
        n_nan = int(df[TARGET].isna().sum())
        add(
            "demand_numeric",
            numeric,
            FAIL,
            f"'{TARGET}' dtype is {df[TARGET].dtype} (numeric={numeric}); {n_nan} NaN(s).",
            dtype=str(df[TARGET].dtype),
            n_nan=n_nan,
        )

    # 3. Timestamps ordered --------------------------------------------------
    if has_ts:
        ts = df["timestamp_utc"]
        ordered = ts.is_monotonic_increasing
        n_out_of_order = int((ts.diff() < pd.Timedelta(0)).sum())
        add(
            "timestamps_ordered",
            ordered,
            FAIL,
            "Timestamps are sorted ascending." if ordered else f"{n_out_of_order} out-of-order step(s).",
            n_out_of_order=n_out_of_order,
        )

    # 4. No negative demand --------------------------------------------------
    if has_target and pd.api.types.is_numeric_dtype(df[TARGET]):
        neg = df[TARGET] < 0
        n_neg = int(neg.sum())
        add(
            "no_negative_demand",
            n_neg == 0,
            FAIL,
            "No negative demand." if n_neg == 0 else f"{n_neg} negative demand value(s).",
            n_negative=n_neg,
        )

    # 5. Duplicate timestamps ------------------------------------------------
    if has_ts:
        dup_mask = df["timestamp_utc"].duplicated(keep=False)
        n_dup = int(dup_mask.sum())
        add(
            "duplicate_timestamps",
            n_dup == 0,
            WARN,
            "No duplicate timestamps." if n_dup == 0 else f"{n_dup} row(s) share a timestamp.",
            n_duplicate_rows=n_dup,
            examples=_example_timestamps(
                pd.DatetimeIndex(df.loc[dup_mask, "timestamp_utc"].unique())
            ),
        )

    # 6. Missing timestamps (gaps in the 30-min UTC grid) --------------------
    if has_ts and n_rows:
        ts_sorted = df["timestamp_utc"].sort_values()
        full = pd.date_range(ts_sorted.iloc[0], ts_sorted.iloc[-1], freq=FREQ, tz="UTC")
        missing = full.difference(pd.DatetimeIndex(df["timestamp_utc"].unique()))
        n_missing = int(len(missing))
        add(
            "missing_timestamps",
            n_missing == 0,
            WARN,
            "No gaps in the 30-minute grid." if n_missing == 0 else f"{n_missing} missing slot(s).",
            n_missing=n_missing,
            expected_slots=int(len(full)),
            examples=_example_timestamps(missing),
        )

    # 7. Settlement periods valid -------------------------------------------
    if has_period:
        periods = pd.to_numeric(df["settlement_period"], errors="coerce")
        out_of_range = ((periods < 1) | (periods > 50) | periods.isna())
        n_range = int(out_of_range.sum())

        n_over_expected = 0
        if "settlement_date" in df.columns:
            counts = df.groupby("settlement_date")["settlement_period"].nunique()
            for date, cnt in counts.items():
                if cnt > expected_periods_in_day(date):
                    n_over_expected += int(cnt - expected_periods_in_day(date))

        add(
            "settlement_periods_valid",
            n_range == 0 and n_over_expected == 0,
            FAIL,
            f"{n_range} out-of-range period(s); {n_over_expected} beyond the day's expected count.",
            n_out_of_range=n_range,
            n_over_expected=n_over_expected,
        )

    # 8. Timezone conversion succeeded --------------------------------------
    if has_ts:
        tz = getattr(df["timestamp_utc"].dt, "tz", None)
        is_utc = tz is not None and str(tz) in ("UTC", "utc")
        n_nat = int(df["timestamp_utc"].isna().sum())
        add(
            "timezone_utc",
            is_utc and n_nat == 0,
            FAIL,
            f"tz={tz}, {n_nat} NaT timestamp(s).",
            tz=str(tz),
            n_nat=n_nat,
        )

    # 9. Units consistent (magnitude sanity) --------------------------------
    if has_target and pd.api.types.is_numeric_dtype(df[TARGET]):
        vals = df[TARGET].dropna()
        out_of_env = ((vals < PLAUSIBLE_MIN_MW) | (vals > PLAUSIBLE_MAX_MW))
        n_env = int(out_of_env.sum())
        add(
            "units_consistent",
            n_env == 0,
            WARN,
            f"{n_env} value(s) outside [{PLAUSIBLE_MIN_MW}, {PLAUSIBLE_MAX_MW}] MW.",
            n_out_of_envelope=n_env,
            observed_min=float(vals.min()) if len(vals) else None,
            observed_max=float(vals.max()) if len(vals) else None,
        )

    # 10. Coverage of requested range ---------------------------------------
    if has_ts and n_rows and (start or end):
        cov_min = df["timestamp_utc"].min()
        cov_max = df["timestamp_utc"].max()
        covers_start = start is None or cov_min <= pd.Timestamp(start, tz="UTC")
        covers_end = end is None or cov_max >= pd.Timestamp(end, tz="UTC")
        add(
            "covers_requested_range",
            covers_start and covers_end,
            WARN,
            f"Data spans {cov_min} .. {cov_max}; requested {start} .. {end}.",
            data_start=cov_min.isoformat(),
            data_end=cov_max.isoformat(),
            requested_start=start,
            requested_end=end,
        )

    n_failed = sum(1 for c in checks if not c["passed"] and c["severity"] == FAIL)
    n_warned = sum(1 for c in checks if not c["passed"] and c["severity"] == WARN)

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_rows": n_rows,
        # This validator inspects only; cleaning happens elsewhere.
        "rows_changed": 0,
        "rows_removed": 0,
        "ok": n_failed == 0,
        "n_failed": n_failed,
        "n_warnings": n_warned,
        "checks": checks,
    }


def write_report(report: dict[str, Any], path: Path | None = None) -> Path:
    """Write the validation report to JSON (default: reports/data_validation.json)."""
    out_path = Path(path) if path else REPORTS_DIR / "data_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Validation report -> %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Load interim demand, validate against the config range, write the report."""
    from ..logging_config import setup_logging

    setup_logging()
    demand_path = INTERIM_DIR / "demand.parquet"
    if not demand_path.exists():
        logger.error("No demand file at %s — run the downloader first.", demand_path)
        return 2

    df = pd.read_parquet(demand_path)
    try:
        dates = load_config("data").get("dates", {})
    except Exception:  # noqa: BLE001
        dates = {}
    report = validate_demand(df, start=dates.get("start"), end=dates.get("end"))
    write_report(report)

    logger.info(
        "Validation %s — %d failure(s), %d warning(s).",
        "OK" if report["ok"] else "FAILED",
        report["n_failed"],
        report["n_warnings"],
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
