"""Forecast evaluation metrics and error breakdowns.

Point metrics: MAE, RMSE, MAPE, WAPE, R^2, peak-demand MAE, forecast bias.
Interval metrics: pinball (quantile) loss and empirical coverage.
Breakdowns: error by horizon, hour, weekday, season and holiday.

Selection guidance
------------------
Do **not** rank models on R^2 alone. Operationally, **MAE** and **WAPE** (typical
error magnitude), **reliability** (interval coverage) and **inference speed** matter
more. R^2 is reported for context, not as the deciding metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (%). Rows with |y_true|<eps are ignored."""
    yt, yp = _clean(y_true, y_pred)
    keep = np.abs(yt) > _EPS
    return float(100.0 * np.mean(np.abs((yt[keep] - yp[keep]) / yt[keep])))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error (%): sum|err| / sum|y_true|."""
    yt, yp = _clean(y_true, y_pred)
    denom = np.sum(np.abs(yt))
    return float(100.0 * np.sum(np.abs(yt - yp)) / (denom + _EPS))


def r2(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - ss_res / (ss_tot + _EPS))


def bias(y_true, y_pred) -> float:
    """Mean forecast error (pred - actual). Positive = over-forecasting."""
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(yp - yt))


def peak_mae(y_true, y_pred, quantile: float = 0.9) -> float:
    """MAE restricted to the highest-demand periods (actual >= given quantile)."""
    yt, yp = _clean(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    threshold = np.quantile(yt, quantile)
    peak = yt >= threshold
    return float(np.mean(np.abs(yt[peak] - yp[peak])))


def point_metrics(y_true, y_pred) -> dict[str, float]:
    """All headline point metrics in one dict."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "peak_mae": peak_mae(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Interval metrics
# ---------------------------------------------------------------------------
def pinball_loss(y_true, y_quantile, quantile: float) -> float:
    """Pinball (quantile) loss at a given quantile level."""
    yt, yq = _clean(y_true, y_quantile)
    diff = yt - yq
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def coverage(y_true, lower, upper) -> float:
    """Fraction of actuals falling within [lower, upper]."""
    yt = np.asarray(y_true, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    mask = ~(np.isnan(yt) | np.isnan(lo) | np.isnan(hi))
    return float(np.mean((yt[mask] >= lo[mask]) & (yt[mask] <= hi[mask])))


def interval_metrics(y_true, lower_80, upper_80, lower_95, upper_95) -> dict[str, float]:
    """Empirical coverage for the 80% and 95% bands (nominal vs achieved)."""
    return {
        "coverage_80": coverage(y_true, lower_80, upper_80),
        "coverage_95": coverage(y_true, lower_95, upper_95),
    }


# ---------------------------------------------------------------------------
# Error breakdowns
# ---------------------------------------------------------------------------
def _season(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"}[int(month)]


def error_by(
    df: pd.DataFrame, by: str, y_true: str = "y_true", y_pred: str = "y_pred"
) -> pd.DataFrame:
    """MAE (and count) grouped by a column, e.g. 'forecast_horizon' or 'hour'."""
    g = df.copy()
    g["abs_err"] = (g[y_true] - g[y_pred]).abs()
    out = g.groupby(by)["abs_err"].agg(["mean", "count"]).rename(columns={"mean": "mae"})
    return out.reset_index()


def error_breakdowns(
    df: pd.DataFrame,
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    time_col: str = "target_time",
    horizon_col: str = "forecast_horizon",
) -> dict[str, pd.DataFrame]:
    """MAE by horizon, hour, weekday, season and holiday.

    ``df`` needs actual/pred columns plus a target timestamp (for hour/weekday/
    season) and optionally ``forecast_horizon`` and ``is_bank_holiday``.
    """
    g = df.copy()
    local = pd.to_datetime(g[time_col])
    if getattr(local.dt, "tz", None) is not None:
        local = local.dt.tz_convert("Europe/London")
    g["hour"] = local.dt.hour
    g["weekday"] = local.dt.dayofweek
    g["season"] = local.dt.month.map(_season)

    out: dict[str, pd.DataFrame] = {}
    if horizon_col in g.columns:
        out["by_horizon"] = error_by(g, horizon_col, y_true, y_pred)
    out["by_hour"] = error_by(g, "hour", y_true, y_pred)
    out["by_weekday"] = error_by(g, "weekday", y_true, y_pred)
    out["by_season"] = error_by(g, "season", y_true, y_pred)
    if "is_bank_holiday" in g.columns:
        out["by_holiday"] = error_by(g, "is_bank_holiday", y_true, y_pred)
    return out


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
COMPARISON_COLUMNS = ["model", "mae", "rmse", "wape", "peak_mae", "training_seconds"]


def comparison_row(
    model_name: str, y_true, y_pred, training_seconds: float | None = None
) -> dict:
    """One row of the model-comparison table."""
    m = point_metrics(y_true, y_pred)
    return {
        "model": model_name,
        "mae": m["mae"],
        "rmse": m["rmse"],
        "wape": m["wape"],
        "peak_mae": m["peak_mae"],
        "training_seconds": training_seconds,
    }


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble comparison rows into a table sorted by MAE (best first)."""
    df = pd.DataFrame(rows)
    for col in COMPARISON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COMPARISON_COLUMNS].sort_values("mae").reset_index(drop=True)
    return df
