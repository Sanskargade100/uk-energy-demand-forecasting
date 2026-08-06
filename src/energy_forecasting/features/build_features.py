"""Assemble the model feature matrix from the processed 30-minute table.

Applies, in order: calendar/cyclical features, demand lags, demand rolling
statistics (leakage-safe), and temperature features. Reads lag/window settings
from ``configs/features.yaml`` when available.

Output: ``data/processed/feature_matrix.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logging_config import get_logger
from ..settings import PROCESSED_DIR, load_config
from .calendar import add_calendar_features
from .lag_features import DEFAULT_LAGS, DEFAULT_ROLLING, add_demand_lags, add_demand_rolling
from .weather import add_temperature_features

logger = get_logger(__name__)

TARGET = "nd_mw"
PROCESSED_FILE = PROCESSED_DIR / "energy_demand_30min.parquet"
FEATURES_FILE = PROCESSED_DIR / "feature_matrix.parquet"


def _rolling_from_config(cfg: dict) -> dict[int, tuple[str, ...]]:
    """Translate features.yaml rolling_windows into a window->stats mapping."""
    windows = cfg.get("rolling_windows")
    if not windows:
        return DEFAULT_ROLLING
    # Full stats for the shortest window, mean/std for longer ones (matches spec).
    stats_map: dict[int, tuple[str, ...]] = {}
    shortest = min(windows)
    for w in windows:
        stats_map[w] = ("mean", "std", "min", "max") if w == shortest else ("mean", "std")
    return stats_map


def build_features(
    df: pd.DataFrame,
    lags=DEFAULT_LAGS,
    rolling: dict[int, tuple[str, ...]] | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """Return the feature matrix for a processed 30-minute demand frame."""
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_demand_lags(df, column=TARGET, lags=lags)
    df = add_demand_rolling(df, column=TARGET, windows_stats=rolling or DEFAULT_ROLLING)
    df = add_temperature_features(df)

    if dropna:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info("Dropped %d warm-up rows with NaN features.", before - len(df))

    return df


def build_feature_matrix(save: bool = True, processed_path: Path | None = None) -> pd.DataFrame:
    """Load the processed table, build features, and (optionally) save."""
    path = Path(processed_path) if processed_path else PROCESSED_FILE
    df = pd.read_parquet(path)

    try:
        cfg = load_config("features")
    except Exception:  # noqa: BLE001
        cfg = {}
    lags = tuple(cfg.get("lags", DEFAULT_LAGS))
    rolling = _rolling_from_config(cfg)

    features = build_features(df, lags=lags, rolling=rolling)
    logger.info("Feature matrix: %d rows, %d columns.", len(features), features.shape[1])

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        features.to_parquet(FEATURES_FILE, index=False)
        logger.info("Saved -> %s", FEATURES_FILE)

    return features


def main(argv: list[str] | None = None) -> int:
    from ..logging_config import setup_logging

    setup_logging()
    if not PROCESSED_FILE.exists():
        logger.error("No processed file at %s — run prepare_data first.", PROCESSED_FILE)
        return 2
    build_feature_matrix(save=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
