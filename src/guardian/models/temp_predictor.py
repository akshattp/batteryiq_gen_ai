"""Supervised temperature prediction model (Milestone 4).

Predicts future cell temperature T(t + H) using LightGBM regressor,
comparing against persistence (T(t+H) = T(t)) and linear regression baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from guardian.data.features import FEATURE_COLUMNS
from guardian.utils.logging import setup_logger

logger = setup_logger("guardian.models.temp_predictor")


def build_lagged_features(
    df: pd.DataFrame,
    horizon_steps: int = 10,
    lags: List[int] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build lagged feature matrix X and future target y = T(t + H) within each cycle.

    Args:
        df: Feature-augmented DataFrame.
        horizon_steps: Forecasting horizon H steps into the future.
        lags: Lag offsets (default [1, 5, 10]).

    Returns:
        Tuple of:
          - X: Feature matrix DataFrame
          - y: Target Series of T(t + H)
          - meta: Auxiliary metadata DataFrame (mode, ambient_c, current_temp, etc.)
    """
    lag_offsets = lags or [1, 5, 10]
    X_dfs = []
    y_series_list = []
    meta_dfs = []

    for _, group in df.groupby(["battery_id", "cycle"], sort=False):
        n = len(group)
        if n <= horizon_steps + max(lag_offsets):
            continue

        grp = group.copy().reset_index(drop=True)

        # Base features at t
        base_features = grp[FEATURE_COLUMNS].copy()

        # Lag features for temp, current, voltage
        lag_dict = {}
        for lag in lag_offsets:
            lag_dict[f"temp_c_lag{lag}"] = grp["temp_c"].shift(lag)
            lag_dict[f"current_a_lag{lag}"] = grp["current_a"].shift(lag)
            lag_dict[f"voltage_v_lag{lag}"] = grp["voltage_v"].shift(lag)

        lag_df = pd.DataFrame(lag_dict, index=grp.index)
        full_X = pd.concat([base_features, lag_df], axis=1)

        # Target: T(t + H)
        target_y = grp["temp_c"].shift(-horizon_steps)

        # Metadata for subgroup breakdown and persistence baseline
        aux_meta = pd.DataFrame({
            "battery_id": grp["battery_id"],
            "cycle": grp["cycle"],
            "t_s": grp["t_s"],
            "mode": grp["mode"],
            "ambient_c": grp["ambient_c"],
            "current_temp": grp["temp_c"],
        })

        # Valid rows: exclude initial lags and final horizon steps
        valid_mask = full_X.notna().all(axis=1) & target_y.notna()

        X_dfs.append(full_X[valid_mask])
        y_series_list.append(target_y[valid_mask])
        meta_dfs.append(aux_meta[valid_mask])

    if not X_dfs:
        raise ValueError("No valid rows extracted for temperature dataset.")

    X = pd.concat(X_dfs, ignore_index=True)
    y = pd.concat(y_series_list, ignore_index=True)
    meta = pd.concat(meta_dfs, ignore_index=True)

    return X, y, meta


class TemperaturePredictor:
    """LightGBM multi-step ahead battery temperature forecaster."""

    def __init__(
        self,
        horizon_steps: int = 10,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
    ) -> None:
        self.horizon_steps = horizon_steps
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        self.feature_names: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the LightGBM regressor on training data."""
        self.feature_names = list(X_train.columns)
        logger.info(f"Fitting LightGBM on {len(X_train):,} samples with {len(self.feature_names)} features...")
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecast future temperature values."""
        return self.model.predict(X)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return normalized feature importances."""
        importances = self.model.feature_importances_.astype(float)
        total = np.sum(importances)
        norm_imp = importances / total if total > 0 else importances
        return {name: round(float(imp), 4) for name, imp in zip(self.feature_names, norm_imp)}

    def save(self, model_path: str | Path) -> None:
        """Persist predictor to disk."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)
        logger.info(f"Saved temperature predictor to {model_path}.")

    @classmethod
    def load(cls, model_path: str | Path) -> TemperaturePredictor:
        """Load predictor from disk."""
        return joblib.load(model_path)


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence_pred: np.ndarray,
    meta_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute detailed evaluation metrics and subgroup breakdowns."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    persist_mae = float(mean_absolute_error(y_true, persistence_pred))
    persist_rmse = float(np.sqrt(mean_squared_error(y_true, persistence_pred)))
    persist_r2 = float(r2_score(y_true, persistence_pred))

    # Error breakdown by mode (charge vs discharge)
    by_mode = {}
    for mode in meta_df["mode"].unique():
        mask = meta_df["mode"] == mode
        if np.sum(mask) > 0:
            by_mode[str(mode)] = {
                "samples": int(np.sum(mask)),
                "mae": round(float(mean_absolute_error(y_true[mask], y_pred[mask])), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))), 4),
                "r2": round(float(r2_score(y_true[mask], y_pred[mask])), 4),
            }

    # Error breakdown by ambient temperature
    by_ambient = {}
    ambient_bins = meta_df["ambient_c"].round().unique()
    for amb in sorted(ambient_bins):
        mask = meta_df["ambient_c"].round() == amb
        if np.sum(mask) > 10:
            by_ambient[f"{int(amb)}C"] = {
                "samples": int(np.sum(mask)),
                "mae": round(float(mean_absolute_error(y_true[mask], y_pred[mask])), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))), 4),
            }

    return {
        "lgbm": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "beats_persistence": bool(mae < persist_mae),
            "improvement_pct": round(float((persist_mae - mae) / persist_mae * 100.0), 2),
        },
        "persistence": {
            "mae": round(persist_mae, 4),
            "rmse": round(persist_rmse, 4),
            "r2": round(persist_r2, 4),
        },
        "by_mode": by_mode,
        "by_ambient": by_ambient,
    }
