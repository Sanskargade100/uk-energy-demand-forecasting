"""Reusable helpers for exploratory analysis and evaluation figures.

Kept here (rather than inside a notebook) so the loading/aggregation logic is
importable and testable. Plotting/figure-saving imports matplotlib lazily so the
pure-pandas helpers can be used without a plotting backend installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..settings import PROCESSED_DIR, REPORTS_DIR

FIGURES_DIR = REPORTS_DIR / "figures"
PROCESSED_FILE = PROCESSED_DIR / "energy_demand_30min.parquet"


def load_processed(path: str | Path | None = None) -> pd.DataFrame:
    """Load the processed 30-minute table with timestamps parsed.

    ``timestamp_utc`` is returned tz-aware (UTC); ``timestamp_local`` tz-aware
    (Europe/London). Rows are sorted ascending by UTC time.
    """
    p = Path(path) if path else PROCESSED_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"Processed data not found at {p}. Run scripts/prepare_data.py first."
        )
    df = pd.read_parquet(p)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add local-time calendar helper columns used throughout the EDA."""
    df = df.copy()
    local = pd.to_datetime(df["timestamp_local"])
    df["year"] = local.dt.year
    df["month"] = local.dt.month
    df["day_of_week"] = local.dt.dayofweek  # Mon=0 .. Sun=6
    df["hour"] = local.dt.hour
    df["minute"] = local.dt.minute
    df["time_of_day"] = local.dt.hour + local.dt.minute / 60.0
    df["local_date"] = local.dt.normalize()
    return df


def mean_profile(df: pd.DataFrame, by, value: str = "nd_mw") -> pd.Series | pd.DataFrame:
    """Mean of ``value`` grouped by one or more columns (a demand *profile*)."""
    grouped = df.groupby(by)[value].mean()
    return grouped


def save_fig(fig, name: str, fig_dir: str | Path | None = None, dpi: int = 150) -> Path:
    """Save a matplotlib figure as a publication-quality PNG under reports/figures/."""
    out_dir = Path(fig_dir) if fig_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (name if name.endswith(".png") else f"{name}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path
