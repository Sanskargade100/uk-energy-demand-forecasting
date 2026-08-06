"""Conformal prediction intervals — model-independent, calibrated on validation.

Idea (split conformal)
----------------------
On a held-out validation set, compute absolute residuals ``r_i = |y_i - yhat_i|``.
The (1-alpha) empirical quantile ``q`` of those residuals gives a symmetric band
``[yhat - q, yhat + q]`` that, on exchangeable data, contains about ``1-alpha`` of
unseen actuals.

Error grows with the forecast horizon, so a single ``q`` would be too wide for
short horizons and too narrow for long ones. We therefore calibrate **separately
per horizon**, or per **horizon group** (default ``1-12``, ``13-48``, ``49-96``).

Honesty
-------
Intervals are only useful if their coverage is tested. :func:`evaluate_intervals`
reports achieved coverage and average width on held-out data, so a nominal 95%
band can be checked against the ~95% it should actually contain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LEVELS = (0.80, 0.95)
DEFAULT_GROUPS = [(1, 12), (13, 48), (49, 96)]


def horizon_group_label(horizon: int, groups=DEFAULT_GROUPS) -> str:
    """Map a horizon to its group label, e.g. 30 -> '13-48'."""
    for lo, hi in groups:
        if lo <= horizon <= hi:
            return f"{lo}-{hi}"
    return "other"


class ConformalIntervals:
    """Residual-quantile intervals, calibrated per horizon or per horizon group."""

    def __init__(self, levels=DEFAULT_LEVELS, by: str = "group", groups=DEFAULT_GROUPS):
        if by not in ("horizon", "group"):
            raise ValueError("by must be 'horizon' or 'group'")
        self.levels = tuple(levels)
        self.by = by
        self.groups = groups
        self.quantiles_: dict = {}  # key -> {level: q}

    # ---- calibration ------------------------------------------------------
    def _key(self, horizon: int):
        return horizon_group_label(horizon, self.groups) if self.by == "group" else int(horizon)

    def fit(self, cal: pd.DataFrame, y_true="y_true", y_pred="y_pred", horizon="forecast_horizon"):
        """Learn residual quantiles from a calibration (validation) frame."""
        df = cal.copy()
        df["_resid"] = (df[y_true] - df[y_pred]).abs()
        df["_key"] = df[horizon].map(self._key)

        self.quantiles_ = {}
        for key, grp in df.groupby("_key"):
            residuals = grp["_resid"].to_numpy()
            # Finite-sample conformal correction: ceil((n+1)(1-alpha))/n quantile.
            n = len(residuals)
            self.quantiles_[key] = {}
            for level in self.levels:
                rank = min(1.0, np.ceil((n + 1) * level) / n) if n > 0 else 1.0
                self.quantiles_[key][level] = float(np.quantile(residuals, rank))
        # Fallback quantiles pooled across all residuals (for unseen keys).
        self._global_ = {
            level: float(np.quantile(df["_resid"].to_numpy(), level)) for level in self.levels
        }
        return self

    def _q(self, horizon: int, level: float) -> float:
        key = self._key(horizon)
        return self.quantiles_.get(key, {}).get(level, self._global_[level])

    # ---- application ------------------------------------------------------
    def apply(self, preds: pd.DataFrame, y_pred="point_forecast_mw", horizon="forecast_horizon"):
        """Add lower/upper 80% and 95% columns to a predictions frame."""
        out = preds.copy()
        for level in self.levels:
            tag = int(round(level * 100))
            q = out[horizon].map(lambda h, lv=level: self._q(int(h), lv)).to_numpy()
            out[f"lower_{tag}"] = out[y_pred].to_numpy() - q
            out[f"upper_{tag}"] = out[y_pred].to_numpy() + q
        return out


# ---------------------------------------------------------------------------
# Evaluation (report coverage + width, never show untested intervals)
# ---------------------------------------------------------------------------
def evaluate_intervals(
    df: pd.DataFrame, levels=DEFAULT_LEVELS, y_true="y_true"
) -> pd.DataFrame:
    """Achieved coverage and mean width per nominal level."""
    rows = []
    for level in levels:
        tag = int(round(level * 100))
        lo = df[f"lower_{tag}"].to_numpy()
        hi = df[f"upper_{tag}"].to_numpy()
        yt = df[y_true].to_numpy()
        mask = ~(np.isnan(yt) | np.isnan(lo) | np.isnan(hi))
        inside = (yt[mask] >= lo[mask]) & (yt[mask] <= hi[mask])
        rows.append(
            {
                "level": level,
                "nominal_coverage": level,
                "empirical_coverage": float(np.mean(inside)),
                "avg_width": float(np.mean(hi[mask] - lo[mask])),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)
