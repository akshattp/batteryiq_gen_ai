"""Feature engineering, sliding window generation, and leakage-safe splits (Milestone 2).

Features derived per time step:
temp_c, dT_dt (°C/min, smoothed), voltage_v, dV_dt, current_a, abs_current_a,
soc_pct, cycle, is_charge, temp_minus_ambient, temp_x_abs_current, soh_pct.

Windows:
Sliding windows of W=30 steps (stride 5) within individual battery cycles.
Scaling:
StandardScaler fit ONLY on the normal training split, saved to artifacts/models/scaler.joblib.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from guardian.utils.logging import setup_logger

logger = setup_logger("guardian.data.features")

FEATURE_COLUMNS: List[str] = [
    "temp_c",
    "dT_dt",
    "voltage_v",
    "dV_dt",
    "current_a",
    "abs_current_a",
    "soc_pct",
    "cycle",
    "is_charge",
    "temp_minus_ambient",
    "temp_x_abs_current",
    "soh_pct",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive physical and interaction features for each time step.

    Groups by (battery_id, cycle) so time derivatives do not leak across boundaries.

    Args:
        df: Cleaned telemetry DataFrame.

    Returns:
        DataFrame containing original columns plus all computed feature columns.
    """
    df = df.copy()

    # Pre-allocate feature columns
    df["abs_current_a"] = df["current_a"].abs()
    df["is_charge"] = (df["mode"].str.lower() == "charge").astype(float)
    df["temp_minus_ambient"] = df["temp_c"] - df["ambient_c"]
    df["temp_x_abs_current"] = df["temp_c"] * df["abs_current_a"]

    # Calculate cycle-level derivatives (dT_dt in °C/min and dV_dt in V/s)
    dt_list: List[float] = []
    dT_dt_list: List[float] = []
    dV_dt_list: List[float] = []

    # Vectorized derivative computation per cycle
    for _, group in df.groupby(["battery_id", "cycle"], sort=False):
        t_vals = group["t_s"].values
        temp_vals = group["temp_c"].values
        v_vals = group["voltage_v"].values

        # Delta t in seconds (bound dt minimum to avoid div by zero)
        delta_t = np.diff(t_vals, prepend=t_vals[0])
        delta_t[delta_t <= 0] = 1.0

        # Delta T in °C and Delta V in V
        delta_T = np.diff(temp_vals, prepend=temp_vals[0])
        delta_V = np.diff(v_vals, prepend=v_vals[0])

        # dT/dt in °C/min = (delta_T / delta_t) * 60.0
        raw_dT_dt = (delta_T / delta_t) * 60.0
        # Smooth dT/dt with small convolution window to suppress sensor noise
        smoothed_dT_dt = pd.Series(raw_dT_dt).rolling(window=5, min_periods=1, center=False).mean().values

        # dV_dt in V/s
        raw_dV_dt = delta_V / delta_t
        smoothed_dV_dt = pd.Series(raw_dV_dt).rolling(window=5, min_periods=1, center=False).mean().values

        dT_dt_list.extend(smoothed_dT_dt)
        dV_dt_list.extend(smoothed_dV_dt)

    df["dT_dt"] = np.round(np.array(dT_dt_list), 4)
    df["dV_dt"] = np.round(np.array(dV_dt_list), 4)

    return df


def create_splits(
    df: pd.DataFrame,
    config: Dict[str, Any],
    train_cycle_ratio: float = 0.75,
    metrics_path: str | Path = "artifacts/metrics/split.json",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create leakage-safe train/val/test splits partitioned by battery and cycle.

    - Train: First train_cycle_ratio fraction of cycles from training batteries.
    - Val: Remaining cycles from training batteries.
    - Test: All cycles from the held-out test battery.

    Args:
        df: Feature-augmented DataFrame.
        config: Configuration dictionary.
        train_cycle_ratio: Proportion of cycles to assign to train split (default 0.75).
        metrics_path: Path to write split statistics.

    Returns:
        Tuple of (train_df, val_df, test_df, split_info_dict).
    """
    data_cfg = config.get("data", {})
    train_batteries = data_cfg.get("train_batteries", ["B0005", "B0006", "B0007"])
    test_battery = data_cfg.get("test_battery", "B0018")

    test_mask = df["battery_id"] == test_battery
    test_df = df[test_mask].copy()

    train_masks = []
    val_masks = []

    for bat in train_batteries:
        bat_df = df[df["battery_id"] == bat]
        unique_cycles = sorted(bat_df["cycle"].unique())
        n_train = max(1, int(len(unique_cycles) * train_cycle_ratio))
        train_cycles = set(unique_cycles[:n_train])
        val_cycles = set(unique_cycles[n_train:])

        train_masks.append((df["battery_id"] == bat) & (df["cycle"].isin(train_cycles)))
        val_masks.append((df["battery_id"] == bat) & (df["cycle"].isin(val_cycles)))

    full_train_mask = train_masks[0]
    for m in train_masks[1:]:
        full_train_mask = full_train_mask | m

    full_val_mask = val_masks[0]
    for m in val_masks[1:]:
        full_val_mask = full_val_mask | m

    train_df = df[full_train_mask].copy()
    val_df = df[full_val_mask].copy()

    split_info = {
        "train_batteries": train_batteries,
        "test_battery": test_battery,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_cycles": train_df.groupby("battery_id")["cycle"].nunique().to_dict(),
        "val_cycles": val_df.groupby("battery_id")["cycle"].nunique().to_dict(),
        "test_cycles": test_df.groupby("battery_id")["cycle"].nunique().to_dict(),
    }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    logger.info(
        f"Splits created: Train={len(train_df):,} rows, Val={len(val_df):,} rows, Test={len(test_df):,} rows."
    )
    return train_df, val_df, test_df, split_info


def fit_scaler(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    scaler_out_path: str | Path = "artifacts/models/scaler.joblib",
) -> StandardScaler:
    """Fit a StandardScaler strictly on the normal training split and save to disk.

    Args:
        train_df: Training split DataFrame.
        feature_cols: Features to scale (defaults to FEATURE_COLUMNS).
        scaler_out_path: Filepath to persist the fit scaler.

    Returns:
        Fitted StandardScaler instance.
    """
    cols = feature_cols or FEATURE_COLUMNS
    normal_train = train_df[train_df["label"] == "normal"]
    scaler = StandardScaler()
    scaler.fit(normal_train[cols].values)

    Path(scaler_out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_out_path)
    logger.info(f"Fitted StandardScaler on {len(normal_train)} normal training rows. Saved to {scaler_out_path}.")
    return scaler


def load_scaler(scaler_path: str | Path = "artifacts/models/scaler.joblib") -> StandardScaler:
    """Load pre-fitted StandardScaler from disk."""
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found at {path}")
    return joblib.load(path)


def create_sliding_windows(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    feature_cols: Optional[List[str]] = None,
    window_size: int = 30,
    stride: int = 5,
    flatten: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate sliding feature windows within each (battery_id, cycle) group.

    Args:
        df: Input DataFrame.
        scaler: Optional fitted StandardScaler. If provided, features are scaled.
        feature_cols: List of feature names.
        window_size: Number of steps per window (W=30).
        stride: Step stride between windows (default 5).
        flatten: If True, flattens (W, F) -> (W * F).

    Returns:
        Tuple of:
          - X_windows: np.ndarray of shape (N, W * F) or (N, W, F)
          - window_metadata: List of metadata dicts per window.
    """
    cols = feature_cols or FEATURE_COLUMNS
    windows_list: List[np.ndarray] = []
    meta_list: List[Dict[str, Any]] = []

    for (bat_id, cycle_num), group in df.groupby(["battery_id", "cycle"], sort=False):
        n_steps = len(group)
        if n_steps < window_size:
            continue

        raw_features = group[cols].values
        scaled_features = scaler.transform(raw_features) if scaler is not None else raw_features
        t_vals = group["t_s"].values
        timestamps = group["timestamp"].values
        modes = group["mode"].values
        labels = group["label"].values
        fault_types = group["fault_type"].values

        for start_idx in range(0, n_steps - window_size + 1, stride):
            end_idx = start_idx + window_size
            win = scaled_features[start_idx:end_idx]

            if flatten:
                windows_list.append(win.flatten())
            else:
                windows_list.append(win)

            meta_list.append({
                "battery_id": bat_id,
                "cycle": int(cycle_num),
                "mode": str(modes[start_idx]),
                "start_t_s": float(t_vals[start_idx]),
                "end_t_s": float(t_vals[end_idx - 1]),
                "start_timestamp": str(timestamps[start_idx]),
                "end_timestamp": str(timestamps[end_idx - 1]),
                "label": str(labels[end_idx - 1]),
                "fault_type": fault_types[end_idx - 1],
            })

    if not windows_list:
        shape = (0, window_size * len(cols)) if flatten else (0, window_size, len(cols))
        return np.empty(shape, dtype=np.float32), []

    X_windows = np.array(windows_list, dtype=np.float32)
    return X_windows, meta_list
