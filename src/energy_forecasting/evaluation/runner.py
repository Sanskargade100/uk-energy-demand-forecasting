"""Backtest evaluation loop that scores every model on the same origins.

A *forecast function* maps an origin timestamp to a Series of point forecasts
indexed by target time (``fn(origin) -> pd.Series``). This module runs each
function over a set of origins, joins the actual demand, and scores the models
into a single comparison table — so baselines, SARIMAX, XGBoost and the LSTM are
all judged on identical origins and horizons.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from . import metrics
from ..models import seasonal_naive as sn

ForecastFn = Callable[[pd.Timestamp], "pd.Series | None"]


def demand_series(processed: pd.DataFrame, target: str = "nd_mw") -> pd.Series:
    """A UTC-indexed demand Series (actuals) from the processed table."""
    s = processed.set_index("timestamp_utc")[target].sort_index()
    return s


def baseline_forecast_fns(demand: pd.Series, horizon: int = 96, min_history: int = 336) -> dict[str, ForecastFn]:
    """Forecast functions for every seasonal-naive baseline."""
    fns: dict[str, ForecastFn] = {}
    for name in sn.BASELINES:
        def make(nm):
            def fn(origin):
                hist = demand[demand.index <= origin].dropna()
                if len(hist) < min_history:
                    return None
                return sn.forecast(nm, hist, horizon)
            return fn
        fns[f"baseline_{name}"] = make(name)
    return fns


def collect_predictions(
    forecast_fns: dict[str, ForecastFn],
    origins,
    actuals: pd.Series,
    horizon: int = 96,
) -> pd.DataFrame:
    """Run every forecast fn over every origin; return a tidy predictions frame.

    Columns: ``model, origin_time, target_time, forecast_horizon, y_pred, y_true``.
    Rows whose target has no actual (e.g. beyond the data) are dropped.
    """
    frames = []
    for name, fn in forecast_fns.items():
        recs = []
        for origin in origins:
            s = fn(pd.Timestamp(origin))
            if s is None or len(s) == 0:
                continue
            idx = pd.DatetimeIndex(s.index)
            step = ((idx - pd.Timestamp(origin)) / pd.Timedelta(minutes=30)).round().astype(int)
            recs.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "origin_time": pd.Timestamp(origin),
                        "target_time": idx,
                        "forecast_horizon": step.to_numpy(),
                        "y_pred": pd.to_numeric(s.to_numpy(), errors="coerce"),
                    }
                )
            )
        if recs:
            frames.append(pd.concat(recs, ignore_index=True))

    if not frames:
        return pd.DataFrame(
            columns=["model", "origin_time", "target_time", "forecast_horizon", "y_pred", "y_true"]
        )
    out = pd.concat(frames, ignore_index=True)
    out["y_true"] = out["target_time"].map(actuals)
    return out.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)


def score_models(pred_df: pd.DataFrame, training_seconds: dict | None = None) -> pd.DataFrame:
    """Comparison table (MAE/RMSE/WAPE/Peak MAE/Training time), one row per model."""
    training_seconds = training_seconds or {}
    rows = [
        metrics.comparison_row(name, g["y_true"], g["y_pred"], training_seconds.get(name))
        for name, g in pred_df.groupby("model")
    ]
    return metrics.comparison_table(rows)


def breakdowns_by_model(pred_df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """Per-model error breakdowns (by horizon/hour/weekday/season)."""
    return {name: metrics.error_breakdowns(g) for name, g in pred_df.groupby("model")}
