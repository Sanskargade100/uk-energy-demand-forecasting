"""Sequence-to-sequence LSTM forecaster (336 -> 96).

Seven days of half-hourly history (336 steps of features) map to the next two days
(96 steps of demand). This is the deep-learning entry in the model comparison — it
is **not assumed to beat XGBoost**. If it scores worse, that result is reported
honestly; an evenhanded comparison is more useful than a flattering one.

Discipline (all enforced here)
------------------------------
* Scalers are fit on **training data only** and saved alongside the model.
* Early stopping on validation loss (patience 10, restore best weights).
* Training and validation loss history is captured for the learning-curve plot.
* Training time is recorded in the metadata.

TensorFlow/Keras is imported lazily so the pure array/scaler logic can be used and
tested without a deep-learning backend installed.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from ..settings import MODELS_DIR, REPORTS_DIR

logger = get_logger(__name__)

TARGET = "nd_mw"
INPUT_LEN = 336
OUTPUT_LEN = 96
_EXCLUDE = {"timestamp_utc", "timestamp_local", "settlement_date"}

MODEL_PATH = MODELS_DIR / "lstm.keras"
SCALERS_PATH = MODELS_DIR / "lstm_scalers.joblib"
METADATA_PATH = MODELS_DIR / "lstm_metadata.json"


# ---------------------------------------------------------------------------
# Self-contained scalers (numpy; joblib-saveable; fit on train only)
# ---------------------------------------------------------------------------
class FeatureScaler:
    """Per-feature standardisation for a (time, features) or (n, time, features) array."""

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        A = np.asarray(X, float)
        flat = A.reshape(-1, A.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0)
        self.std_ = np.nanstd(flat, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        A = np.asarray(X, float)
        return (A - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        A = np.asarray(X, float)
        return A * self.std_ + self.mean_


class SeriesScaler:
    """Scalar standardisation for the (single-variable) target."""

    def fit(self, x) -> "SeriesScaler":
        a = np.asarray(x, float).ravel()
        self.mean_ = float(np.nanmean(a))
        self.std_ = float(np.nanstd(a)) or 1.0
        return self

    def transform(self, x) -> np.ndarray:
        return (np.asarray(x, float) - self.mean_) / self.std_

    def inverse_transform(self, x) -> np.ndarray:
        return np.asarray(x, float) * self.std_ + self.mean_


# ---------------------------------------------------------------------------
# Array preparation (pure)
# ---------------------------------------------------------------------------
def default_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns with the target first (input includes past demand)."""
    numeric = [
        c for c in df.columns if c not in _EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    return [TARGET] + [c for c in numeric if c != TARGET]


def window_arrays(
    features_2d: np.ndarray,
    target_1d: np.ndarray,
    input_len: int = INPUT_LEN,
    output_len: int = OUTPUT_LEN,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows: X = (n, input_len, features), Y = (n, output_len)."""
    feats = np.asarray(features_2d, float)
    tgt = np.asarray(target_1d, float)
    n = feats.shape[0]
    xs, ys = [], []
    for s in range(0, n - input_len - output_len + 1, stride):
        xs.append(feats[s : s + input_len])
        ys.append(tgt[s + input_len : s + input_len + output_len])
    if not xs:
        n_feat = feats.shape[1] if feats.ndim == 2 else 1
        return np.empty((0, input_len, n_feat)), np.empty((0, output_len))
    return np.asarray(xs), np.asarray(ys)


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------
class LSTMForecaster:
    def __init__(
        self,
        feature_cols: list[str] | None = None,
        input_len: int = INPUT_LEN,
        output_len: int = OUTPUT_LEN,
        units=(128, 64),
        dense_units: int = 128,
        dropout: float = 0.2,
        loss: str = "mae",  # "mae" or "huber"
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
    ):
        self.feature_cols = feature_cols
        self.input_len = input_len
        self.output_len = output_len
        self.units = units
        self.dense_units = dense_units
        self.dropout = dropout
        self.loss = loss
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

        self.feature_scaler = FeatureScaler()
        self.target_scaler = SeriesScaler()
        self.model_ = None
        self.history_: dict | None = None
        self.training_seconds_: float | None = None
        self.best_val_loss_: float | None = None
        self.train_start_ = None
        self.train_end_ = None

    def _features(self, df: pd.DataFrame) -> list[str]:
        return self.feature_cols or default_feature_columns(df)

    def _prepare(self, df: pd.DataFrame, cols: list[str], fit: bool) -> tuple[np.ndarray, np.ndarray]:
        df = df.sort_values("timestamp_utc")
        feats = df[cols].to_numpy(dtype=float)
        feats = pd.DataFrame(feats).ffill().bfill().fillna(0.0).to_numpy()  # LSTM needs no NaN
        target = df[TARGET].to_numpy(dtype=float)
        if fit:
            self.feature_scaler.fit(feats)
            self.target_scaler.fit(df[TARGET].to_numpy())
        feats_s = self.feature_scaler.transform(feats)
        target_s = self.target_scaler.transform(np.nan_to_num(target, nan=self.target_scaler.mean_))
        return window_arrays(feats_s, target_s, self.input_len, self.output_len)

    def _build(self, n_features: int):
        from tensorflow import keras

        keras.utils.set_random_seed(self.seed)
        loss = keras.losses.Huber() if self.loss == "huber" else "mae"
        model = keras.Sequential(
            [
                keras.layers.Input((self.input_len, n_features)),
                keras.layers.LSTM(self.units[0], return_sequences=True),
                keras.layers.Dropout(self.dropout),
                keras.layers.LSTM(self.units[1]),
                keras.layers.Dense(self.dense_units, activation="relu"),
                keras.layers.Dense(self.output_len),
            ]
        )
        model.compile(optimizer="adam", loss=loss)
        return model

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "LSTMForecaster":
        from tensorflow import keras

        cols = self._features(train_df)
        self.feature_cols = cols
        x_tr, y_tr = self._prepare(train_df, cols, fit=True)   # scalers fit on TRAIN only
        x_val, y_val = self._prepare(val_df, cols, fit=False)

        self.model_ = self._build(x_tr.shape[-1])
        stopper = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        t0 = dt.datetime.now()
        hist = self.model_.fit(
            x_tr, y_tr,
            validation_data=(x_val, y_val),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=[stopper],
            verbose=0,
        )
        self.training_seconds_ = (dt.datetime.now() - t0).total_seconds()
        self.history_ = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        self.best_val_loss_ = float(min(self.history_["val_loss"]))
        self.train_start_ = train_df["timestamp_utc"].min()
        self.train_end_ = train_df["timestamp_utc"].max()
        logger.info(
            "LSTM trained in %.1fs | %d epochs | best val_loss=%.4f (scaled)",
            self.training_seconds_, len(self.history_["loss"]), self.best_val_loss_,
        )
        return self

    def predict_window(self, window_df: pd.DataFrame) -> np.ndarray:
        """Forecast the next ``output_len`` demand values from the last input window."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict.")
        df = window_df.sort_values("timestamp_utc").tail(self.input_len)
        feats = df[self.feature_cols].to_numpy(dtype=float)
        feats = pd.DataFrame(feats).ffill().bfill().fillna(0.0).to_numpy()
        x = self.feature_scaler.transform(feats)[np.newaxis, :, :]
        y_scaled = self.model_.predict(x, verbose=0)[0]
        return self.target_scaler.inverse_transform(y_scaled)

    # ---- persistence ------------------------------------------------------
    def metadata(self) -> dict:
        from .xgboost_model import get_git_commit  # reuse the helper

        versions = {}
        for name in ("tensorflow", "numpy", "pandas"):
            try:
                versions[name] = __import__(name).__version__
            except Exception:  # noqa: BLE001
                versions[name] = "not-installed"
        return {
            "model": "LSTM_seq2seq",
            "target": TARGET,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "architecture": {
                "lstm_units": list(self.units),
                "dense_units": self.dense_units,
                "dropout": self.dropout,
            },
            "loss": self.loss,
            "optimizer": "adam",
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.patience,
            "epochs_ran": len(self.history_["loss"]) if self.history_ else 0,
            "feature_list": self.feature_cols,
            "training_start": str(self.train_start_),
            "training_end": str(self.train_end_),
            "training_seconds": self.training_seconds_,
            "best_val_loss_scaled": self.best_val_loss_,
            "package_versions": versions,
            "training_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": get_git_commit(),
            "note": "Deep learning is not assumed to beat XGBoost; compare honestly.",
        }

    def save(
        self,
        model_path: Path | None = None,
        scalers_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> tuple[Path, Path, Path]:
        import joblib

        mp = Path(model_path) if model_path else MODEL_PATH
        sp = Path(scalers_path) if scalers_path else SCALERS_PATH
        jp = Path(metadata_path) if metadata_path else METADATA_PATH
        mp.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save(mp)
        joblib.dump(
            {"feature_scaler": self.feature_scaler, "target_scaler": self.target_scaler,
             "feature_cols": self.feature_cols, "history": self.history_},
            sp,
        )
        jp.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
        logger.info("Saved LSTM -> %s (+ scalers %s, metadata %s)", mp, sp, jp)
        return mp, sp, jp

    def save_history_plot(self, path: Path | None = None) -> Path | None:
        """Plot training vs validation loss curves to reports/figures/."""
        if not self.history_:
            return None
        import matplotlib.pyplot as plt

        out = Path(path) if path else REPORTS_DIR / "figures" / "lstm_training_curve.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(self.history_["loss"], label="train loss")
        ax.plot(self.history_["val_loss"], label="val loss")
        ax.set_xlabel("epoch"); ax.set_ylabel(f"{self.loss} (scaled)")
        ax.set_title("LSTM training vs validation loss"); ax.legend()
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return out
