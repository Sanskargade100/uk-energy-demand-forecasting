"""CLI: validate the raw demand, then clean and combine all sources.

Usage
-----
    python scripts/prepare_data.py                       # config date range
    python scripts/prepare_data.py --start 2020-01-01 --end 2025-12-31
    python scripts/prepare_data.py --skip-validate

Reads:  data/interim/demand.parquet, data/interim/weather_30min.parquet,
        data/external/uk_bank_holidays.parquet
Writes: reports/data_validation.json, data/processed/energy_demand_30min.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

from energy_forecasting.data.clean import clean_and_combine
from energy_forecasting.data.validate import validate_demand, write_report
from energy_forecasting.logging_config import get_logger, setup_logging
from energy_forecasting.settings import INTERIM_DIR, load_config

logger = get_logger(__name__)


def _config_range() -> tuple[str | None, str | None]:
    try:
        dates = load_config("data").get("dates", {})
        return dates.get("start"), dates.get("end")
    except Exception:  # noqa: BLE001
        return None, None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg_start, cfg_end = _config_range()
    parser = argparse.ArgumentParser(description="Validate, clean and combine data.")
    parser.add_argument("--start", default=cfg_start)
    parser.add_argument("--end", default=cfg_end)
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    if not args.skip_validate:
        demand_path = INTERIM_DIR / "demand.parquet"
        if demand_path.exists():
            report = validate_demand(
                pd.read_parquet(demand_path), start=args.start, end=args.end
            )
            write_report(report)
            logger.info(
                "Validation %s (%d failures, %d warnings).",
                "OK" if report["ok"] else "FAILED",
                report["n_failed"],
                report["n_warnings"],
            )

    combined = clean_and_combine(start=args.start, end=args.end, save=not args.no_save)
    logger.info("Processed table: %d rows, %d columns.", len(combined), combined.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
