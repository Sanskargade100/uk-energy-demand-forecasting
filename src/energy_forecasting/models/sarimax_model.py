"""SARIMAX statistical forecaster for GB demand.

Why SARIMAX (not Prophet) — it makes the time-series assumptions explicit: an
ARIMA process with a short daily seasonal term, plus exogenous regressors.

Computational strategy (half-hourly data has several seasonalities and is expensive)
------------------------------------------------------------------------------------
* **Hourly aggregation** by default (``aggregate="hourly"``): 24-period day instead
  of 48, roughly halving the series length.
* **Short training window**: fit on a recent slice (e.g. the last 60–90 days) rather
  than all six years — set via the data you pass to :meth:`fit`.
* **Fourier terms** carry the *weekly* and *annual* cycles as exogenous regressors,
  so the SARIMA seasonal order only has to model the *daily* cycle. This avoids a
  seasonal period of 336/17532, which is intractable for state-space SARIMAX.
* Exogenous drivers: temperature, a bank-holiday indicator and a weekend indicator.

These are deliberate trade-offs for tractability and are recorded in the saved
metadata (``models/sarimax_metadata.json``). SARIMAX here is a strong statistical
baseline, not the intended production model.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from ..settings import MODELS_DIR

logger = get_logger(__name__)

TARGET = "nd_mw"
DEFAULT_EXOG = ["temperature_mean", "is_bank_holiday", "is_weekend"]

# Periods (in steps) per named frequency, for seasonal order + Fourier.
_STEPS = {
    "hourly": {"freq": "1h", "day": 24, "week": 168, "year": 8766},
    "halfhourly": {"freq": "30min", "day": 48, "week": 336, "year": 17532},
}

MODEL_PATH = MODELS_DIR / "sarimax.joblib"
METADATA_PATH = MODELS_DIR / "sarimax_metadata.json"


# ---------------------------------------------------------------------------
# Fourier / exogenous helpers (pure, unit-tested)
# ---------------------------------------------------------------------------
def time_index_steps(index, epoch, freq: str) -> np.ndarray:
    """Integer step index of ``index`` relative to ``epoch`` at resolution ``freq``."""
    idx = pd.DatetimeIndex(index)
    epoch = pd.Timestamp(epoch)
    # Align tz-awareness so string epochs work against tz-aware indices.
    if idx.tz is not None and epoch.tz is None:
        epoch = epoch.tz_localize(idx.tz)
    elif idx.tz is None and epoch.tz is not None:
        idx = idx.tz_localize(epoch.tz)
    return ((idx - epoch) / pd.Timedelta(freq)).round().astype("int64").to_numpy()


def fourier_terms(
    index, period_steps: int, n_harmonics: int, epoch, freq: str, name: str
) -> pd.DataFrame:
    """Sine/cosine harmonics for a seasonal cycle of ``period_steps`` steps.

    Using ``epoch`` keeps the phase consistent between training and forecasting.
    """
    t = time_index_steps(index, epoch, freq)
    cols: dict[str, np.ndarray] = {}
    for j in range(1, n_harmonics + 1):
        ang = 2.0 * np.pi * j * t / period_steps
        cols[f"{name}_sin_{j}"] = np.sin(ang)
        cols[f"{name}_cos_{j}"] = np.cos(ang)
    return pd.DataFrame(cols, index=pd.DatetimeIndex(index))


@dataclass
class SARIMAXConfig:
    order: tuple[int, int, int] = (2, 0, 2)
    seasonal_order: tuple[int, int, int, int] | None = None  # default derived from freq
    aggregate: str = "hourly"  # "hourly" or "halfhourly"
    exog_cols: list[str] = field(default_factory=lambda: list(DEFAULT_EXOG))
    weekly_harmonics: int = 3
    annual_harmonics: int = 2


class SARIMAXForecaster:
    """SARIMAX with Fourier weekly/annual terms and exogenous drivers."""

    def __init__(self, config: SARIMAXConfig | None = None):
        self.config = config or SARIMAXConfig()
        self.steps = _STEPS[self.config.aggregate]
        self.freq = self.steps["freq"]
        if self.config.seasonal_order is None:
            self.config.seasonal_order = (1, 0, 1, self.steps["day"])
        self.result_ = None
        self.epoch_: pd.Timestamp | None = None
        self.exog_columns_: list[str] = []
        self.train_start_ = None
        self.train_end_ = None

    # ---- data prep --------------------------------------------------------
    def _as_indexed(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp_utc" in df.columns:
            df = df.set_index("timestamp_utc")
        return df.sort_index()

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._as_indexed(df)
        if self.config.aggregate == "halfhourly":
            return df
        agg = {TARGET: "mean"}
        for c in self.config.exog_cols:
            agg[c] = "max" if c.startswith("is_") else "mean"
        return df.resample(self.freq).agg(agg).dropna(subset=[TARGET])

    def _build_exog(self, index, base: pd.DataFrame) -> pd.DataFrame:
        parts = [base.reindex(index)[self.config.exog_cols]]
        parts.append(
            fourier_terms(
                index, self.steps["week"], self.config.weekly_harmonics,
                self.epoch_, self.freq, "weekly",
            )
        )
        parts.append(
            fourier_terms(
                index, self.steps["year"], self.config.annual_harmonics,
                self.epoch_, self.freq, "annual",
            )
        )
        exog = pd.concat(parts, axis=1)
        return exog

    # ---- fit / predict ----------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "SARIMAXForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        data = self._aggregate(df)
        self.epoch_ = data.index[0]
        endog = data[TARGET].astype(float)
        exog = self._build_exog(data.index, data)
        self.exog_columns_ = list(exog.columns)

        t0 = dt.datetime.now()
        self.result_ = SARIMAX(
            endog,
            exog=exog,
            order=self.config.order,
            seasonal_order=self.config.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        self.fit_seconds_ = (dt.datetime.now() - t0).total_seconds()
        self.train_start_, self.train_end_ = data.index[0], data.index[-1]
        logger.info(
            "SARIMAX fitted on %d obs (%s) in %.1fs | AIC=%.0f",
            len(endog), self.freq, self.fit_seconds_, self.result_.aic,
        )
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast for a contiguous future frame carrying the exog columns.

        Returns point forecast plus 80% and 95% intervals, indexed by timestamp.
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() before predict().")
        future = self._as_indexed(future)
        index = future.index
        exog = self._build_exog(index, future)[self.exog_columns_]

        fc = self.result_.get_forecast(steps=len(index), exog=exog)
        mean = fc.predicted_mean
        ci80 = fc.conf_int(alpha=0.20)
        ci95 = fc.conf_int(alpha=0.05)
        return pd.DataFrame(
            {
                "point_forecast_mw": mean.to_numpy(),
                "lower_80_mw": ci80.iloc[:, 0].to_numpy(),
                "upper_80_mw": ci80.iloc[:, 1].to_numpy(),
                "lower_95_mw": ci95.iloc[:, 0].to_numpy(),
                "upper_95_mw": ci95.iloc[:, 1].to_numpy(),
            },
            index=index,
        )

    # ---- persistence ------------------------------------------------------
    def metadata(self) -> dict:
        import statsmodels

        return {
            "model": "SARIMAX",
            "order": list(self.config.order),
            "seasonal_order": list(self.config.seasonal_order),
            "aggregate": self.config.aggregate,
            "freq": self.freq,
            "exog_cols": self.config.exog_cols,
            "fourier": {
                "weekly_harmonics": self.config.weekly_harmonics,
                "annual_harmonics": self.config.annual_harmonics,
                "weekly_period_steps": self.steps["week"],
                "annual_period_steps": self.steps["year"],
            },
            "exog_columns": self.exog_columns_,
            "epoch": str(self.epoch_),
            "train_start": str(self.train_start_),
            "train_end": str(self.train_end_),
            "n_obs": int(self.result_.nobs) if self.result_ is not None else 0,
            "aic": float(self.result_.aic) if self.result_ is not None else None,
            "bic": float(self.result_.bic) if self.result_ is not None else None,
            "fit_seconds": getattr(self, "fit_seconds_", None),
            "statsmodels_version": statsmodels.__version__,
            "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "limitations": (
                "Daily seasonality via SARIMA seasonal order; weekly/annual via Fourier "
                "exog. Trained on a short recent window and hourly-aggregated for "
                "tractability. Not the production model."
            ),
        }

    def save(
        self, model_path: Path | None = None, metadata_path: Path | None = None
    ) -> tuple[Path, Path]:
        """Persist the fitted forecaster (joblib) and metadata (json) separately."""
        import joblib

        mp = Path(model_path) if model_path else MODEL_PATH
        jp = Path(metadata_path) if metadata_path else METADATA_PATH
        mp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, mp)
        jp.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
        logger.info("Saved SARIMAX -> %s (+ metadata %s)", mp, jp)
        return mp, jp

    @classmethod
    def load(cls, model_path: Path | None = None) -> "SARIMAXForecaster":
        import joblib

        return joblib.load(Path(model_path) if model_path else MODEL_PATH)
