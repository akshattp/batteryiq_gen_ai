"""scripts/train_vae.py - Train Beta-VAE anomaly detector and baselines (Milestone 3).

Trains Beta-VAE on normal telemetry windows only.
Fits Isolation Forest and PCA baselines on the same normal training windows.
Calibrates ECDF mapping and thresholds from normal validation windows.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from guardian.utils.config import load_config
from guardian.utils.seed import set_seed
from guardian.data.features import (
    FEATURE_COLUMNS,
    create_splits,
    create_sliding_windows,
    load_scaler,
)
from guardian.models.vae import VAEAnomalyDetector
from guardian.utils.logging import setup_logger

logger = setup_logger("scripts.train_vae")


def main() -> None:
    logger.info("Initializing Beta-VAE training pipeline...")
    config = load_config()
    set_seed(config.get("seed", 42))

    # Paths
    features_parquet = Path("data/processed/features.parquet")
    scaler_path = Path("artifacts/models/scaler.joblib")
    vae_model_path = Path("artifacts/models/vae_model.pt")
    vae_meta_path = Path("artifacts/models/vae_meta.json")
    iforest_path = Path("artifacts/models/iforest_baseline.joblib")
    pca_path = Path("artifacts/models/pca_baseline.joblib")

    if not features_parquet.exists():
        raise FileNotFoundError(f"Feature dataset not found at {features_parquet}. Run 'make data' first.")

    # Step 1: Load data and scaler
    logger.info("Loading feature dataset and scaler...")
    df = pd.read_parquet(features_parquet)
    scaler = load_scaler(scaler_path)

    # Step 2: Extract splits
    train_df, val_df, test_df, _ = create_splits(df, config=config, train_cycle_ratio=0.75)
    train_normal = train_df[train_df["label"] == "normal"]
    val_normal = val_df[val_df["label"] == "normal"]

    # Step 3: Extract sliding windows (stride=10 for efficient CPU training while preserving diversity)
    window_size = config.get("data", {}).get("window_size", 30)
    logger.info(f"Extracting sliding windows (W={window_size}, stride=10)...")
    X_train_win, _ = create_sliding_windows(
        train_normal,
        scaler=scaler,
        feature_cols=FEATURE_COLUMNS,
        window_size=window_size,
        stride=10,
        flatten=True,
    )
    X_val_win, _ = create_sliding_windows(
        val_normal,
        scaler=scaler,
        feature_cols=FEATURE_COLUMNS,
        window_size=window_size,
        stride=10,
        flatten=True,
    )

    logger.info(f"Extracted {len(X_train_win):,} training windows and {len(X_val_win):,} validation windows.")

    # Step 4: Configure and train VAE
    vae_cfg = config.get("vae", {})
    input_dim = window_size * len(FEATURE_COLUMNS)  # 30 * 12 = 360
    detector = VAEAnomalyDetector(
        input_dim=input_dim,
        hidden_dims=vae_cfg.get("hidden_dims", [128, 64]),
        latent_dim=vae_cfg.get("latent_dim", 8),
        beta=vae_cfg.get("beta", 0.5),
        lr=vae_cfg.get("lr", 0.001),
        window_size=window_size,
        num_features=len(FEATURE_COLUMNS),
    )

    logger.info("Training Beta-VAE on normal windows...")
    history = detector.fit(
        X_train=X_train_win,
        X_val=X_val_win,
        epochs=vae_cfg.get("epochs", 25),
        batch_size=vae_cfg.get("batch_size", 128),
        patience=vae_cfg.get("early_stopping_patience", 5),
    )

    detector.save(vae_model_path, vae_meta_path)

    # Step 5: Fit Isolation Forest baseline on subsample of normal train windows
    logger.info("Fitting Isolation Forest baseline...")
    if_sample_size = min(15000, len(X_train_win))
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(X_train_win), size=if_sample_size, replace=False)
    X_train_sub = X_train_win[sample_indices]

    iforest = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
    iforest.fit(X_train_sub)
    Path(iforest_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(iforest, iforest_path)

    # Step 6: Fit PCA baseline
    logger.info("Fitting PCA reconstruction baseline (n_components=8)...")
    pca = PCA(n_components=8, random_state=42)
    pca.fit(X_train_sub)
    Path(pca_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, pca_path)

    print("\n" + "=" * 60)
    print("VAE TRAINING COMPLETE")
    print(f"Normal train windows: {len(X_train_win):,}")
    print(f"Normal val windows:   {len(X_val_win):,}")
    print(f"Calibrated p95:       {detector.p95_threshold:.5f}")
    print(f"Calibrated p99.5:     {detector.p995_threshold:.5f}")
    print(f"Saved model:          {vae_model_path}")
    print(f"Saved metadata:       {vae_meta_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
