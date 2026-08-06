"""CLI: download raw source data (NESO demand and Open-Meteo weather).

Usage
-----
    python scripts/download_data.py                          # demand, config range
    python scripts/download_data.py --source demand --force
    python scripts/download_data.py --source weather
    python scripts/download_data.py --source weather --weather-source forecast
    python scripts/download_data.py --source all --start 2020-01-01 --end 2025-12-31

Date range precedence: CLI flags > configs/data.yaml (dates.start/end) > defaults.
"""

from __future__ import annotations

import argparse
import datetime as dt

from energy_forecasting.data.download_demand import download_demand
from energy_forecasting.data.download_weather import download_weather
from energy_forecasting.logging_config import get_logger, setup_logging
from energy_forecasting.settings import load_config

logger = get_logger(__name__)


def _default_range() -> tuple[str, str]:
    """Read the date range from configs/data.yaml, with fallbacks."""
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
    parser.add_argument("--source", choices=["demand", "weather", "all"], default="demand")
    parser.add_argument("--start", default=start_default, help="Inclusive ISO start date.")
    parser.add_argument("--end", default=end_default, help="Inclusive ISO end date.")
    parser.add_argument("--force", action="store_true", help="Re-download cached demand files.")
    parser.add_argument(
        "--weather-source",
        choices=["archive", "forecast"],
        default="archive",
        help="archive = observed/reanalysis (exploratory); forecast = archived forecasts (backtest).",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write parquet output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    save = not args.no_save

    if args.source in ("demand", "all"):
        df = download_demand(start=args.start, end=args.end, force=args.force, save=save)
        logger.info("Demand: %d rows from %s to %s.", len(df), args.start, args.end)

    if args.source in ("weather", "all"):
        wx = download_weather(
            start=args.start, end=args.end, source=args.weather_source, save=save
        )
        logger.info("Weather (%s): %d rows.", args.weather_source, len(wx))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
