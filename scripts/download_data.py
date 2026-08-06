"""CLI: download raw source data (currently NESO demand).

Usage
-----
    python scripts/download_data.py                     # uses configs/data.yaml range
    python scripts/download_data.py --start 2020-01-01 --end 2025-12-31
    python scripts/download_data.py --source demand

Date range precedence: CLI flags > configs/data.yaml (sources.demand.start_date
and the target's end) > built-in defaults.
"""

from __future__ import annotations

import argparse
import datetime as dt

from energy_forecasting.data.download_demand import download_demand
from energy_forecasting.logging_config import get_logger, setup_logging
from energy_forecasting.settings import load_config

logger = get_logger(__name__)


def _default_range() -> tuple[str, str]:
    """Read the demand date range from configs/data.yaml, with fallbacks."""
    start, end = "2020-01-01", dt.date.today().isoformat()
    try:
        dates = load_config("data").get("dates", {})
        start = dates.get("start", start)
        end = dates.get("end", end)
    except Exception:  # noqa: BLE001 - config is optional at this stage
        pass
    return start, end


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    start_default, end_default = _default_range()
    parser = argparse.ArgumentParser(description="Download raw source data.")
    parser.add_argument("--source", choices=["demand"], default="demand")
    parser.add_argument("--start", default=start_default, help="Inclusive ISO start date.")
    parser.add_argument("--end", default=end_default, help="Inclusive ISO end date.")
    parser.add_argument("--force", action="store_true", help="Re-download cached yearly files.")
    parser.add_argument("--no-save", action="store_true", help="Do not write parquet output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    if args.source == "demand":
        df = download_demand(
            start=args.start, end=args.end, force=args.force, save=not args.no_save
        )
        logger.info("Done: %d rows of demand from %s to %s.", len(df), args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
