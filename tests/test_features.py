"""tests/test_features.py - Feature engineering, windowing, and leakage prevention tests."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from guardian.utils.config import load_config
from guardian.data.features import (
    FEATURE_COLUMNS,
    compute_features,
    create_splits,
    create_sliding_windows,
    load_scaler,
)


@pytest.fixture(scope="module")
def sample_telemetry() -> pd.DataFrame:
    """Load a subset of telemetry records for fast unit testing."""
    parquet_path = Path("data/processed/features.parquet")
    if parquet_path.exists():
        # Load sample from parquet
        return pd.read_parquet(parquet_path).head(5000)
    csv_path = Path("data/processed/battery_telemetry.csv")
    assert csv_path.exists(), "Telemetry CSV not found"
    df = pd.read_csv(csv_path, nrows=5000)
    return compute_features(df)


def test_feature_columns_presence(sample_telemetry: pd.DataFrame):
    """Verify all 12 feature columns are computed and non-empty."""
    for col in FEATURE_COLUMNS:
        assert col in sample_telemetry.columns, f"Feature column '{col}' missing."
        assert not sample_telemetry[col].isna().all(), f"Feature column '{col}' is all NaN."


def test_sliding_window_shapes(sample_telemetry: pd.DataFrame):
    """Verify extracted sliding windows conform to (N, W * F) and (N, W, F)."""
    W = 30
    stride = 5
    F = len(FEATURE_COLUMNS)

    # Flattened format (for MLP VAE)
    X_flat, meta_flat = create_sliding_windows(
        sample_telemetry,
        feature_cols=FEATURE_COLUMNS,
        window_size=W,
        stride=stride,
        flatten=True,
    )
    assert len(X_flat) > 0, "No windows extracted."
    assert X_flat.ndim == 2
    assert X_flat.shape[1] == W * F
    assert len(X_flat) == len(meta_flat)

    # 3D tensor format (for Conv/Recurrent VAE / GAN)
    X_3d, meta_3d = create_sliding_windows(
        sample_telemetry,
        feature_cols=FEATURE_COLUMNS,
        window_size=W,
        stride=stride,
        flatten=False,
    )
    assert X_3d.ndim == 3
    assert X_3d.shape[1] == W
    assert X_3d.shape[2] == F
    assert len(X_3d) == len(meta_3d)


def test_scaler_fit_on_train_only_no_leakage():
    """Verify StandardScaler was fit strictly on normal training data without test/val leakage."""
    config = load_config()
    features_path = Path("data/processed/features.parquet")
    assert features_path.exists(), "features.parquet missing"
    df = pd.read_parquet(features_path)

    scaler = load_scaler("artifacts/models/scaler.joblib")
    assert scaler is not None

    train_df, val_df, test_df, _ = create_splits(df, config=config, train_cycle_ratio=0.75)
    train_normal = train_df[train_df["label"] == "normal"][FEATURE_COLUMNS]

    # Calculate empirical mean of train normal split
    expected_train_mean = train_normal.mean(axis=0).values
    actual_scaler_mean = scaler.mean_

    # Must match training distribution closely
    assert np.allclose(actual_scaler_mean, expected_train_mean, atol=1e-3), (
        "Scaler mean deviates from normal training split mean! Potential leakage."
    )

    # Test split (held-out battery B0018) mean should NOT match scaler mean identically
    test_mean = test_df[FEATURE_COLUMNS].mean(axis=0).values
    assert not np.allclose(actual_scaler_mean, test_mean, atol=1e-4), (
        "Scaler mean is identical to test set mean, indicating test leakage!"
    )


def test_split_json_metadata():
    """Verify artifacts/metrics/split.json records clean battery/cycle partitions."""
    split_json_path = Path("artifacts/metrics/split.json")
    assert split_json_path.exists(), "split.json not found"

    with open(split_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["test_battery"] == "B0018"
    assert "B0005" in data["train_batteries"]
    assert data["train_rows"] > 0
    assert data["val_rows"] > 0
    assert data["test_rows"] > 0
