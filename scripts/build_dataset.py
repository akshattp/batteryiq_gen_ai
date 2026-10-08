"""scripts/build_dataset.py - End-to-end dataset & features builder (Milestones 1 & 2).

Orchestrates raw NASA data ingestion, schema validation, data cleaning,
physical range checking, feature engineering, and leakage-safe splits:
- data/processed/battery_telemetry.csv
- data/processed/features.parquet
- data/interim/rejected_rows.csv
- artifacts/metrics/data_quality.json
- artifacts/metrics/split.json
- artifacts/models/scaler.joblib
"""

import sys
from pathlib import Path
import pandas as pd
from guardian.utils.config import load_config
from guardian.data.ingest import ingest_all
from guardian.data.clean import clean_telemetry
from guardian.data.features import compute_features, create_splits, fit_scaler
from guardian.utils.logging import setup_logger

logger = setup_logger("scripts.build_dataset")


def main() -> None:
    logger.info("Initializing dataset & feature pipeline...")
    config = load_config()

    train_bats = config.get("data", {}).get("train_batteries", ["B0005", "B0006", "B0007"])
    test_bat = config.get("data", {}).get("test_battery", "B0018")
    target_batteries = list(train_bats)
    if test_bat and test_bat not in target_batteries:
        target_batteries.append(test_bat)

    raw_dir = config.get("paths", {}).get("data_raw", "data/raw")
    processed_csv = Path(config.get("paths", {}).get("data_processed", "data/processed")) / "battery_telemetry.csv"
    features_parquet = Path(config.get("paths", {}).get("data_processed", "data/processed")) / "features.parquet"
    rejected_csv = Path(config.get("paths", {}).get("data_interim", "data/interim")) / "rejected_rows.csv"
    metrics_json = Path(config.get("paths", {}).get("metrics_dir", "artifacts/metrics")) / "data_quality.json"
    split_json = Path(config.get("paths", {}).get("metrics_dir", "artifacts/metrics")) / "split.json"
    scaler_joblib = Path(config.get("paths", {}).get("models_dir", "artifacts/models")) / "scaler.joblib"

    # Step 1: Load or Ingest raw data
    if processed_csv.exists() and metrics_json.exists():
        logger.info(f"Existing cleaned telemetry found at {processed_csv}. Loading...")
        cleaned_df = pd.read_csv(processed_csv)
    else:
        logger.info(f"Target batteries: {target_batteries} from {raw_dir}")
        raw_df = ingest_all(raw_dir=raw_dir, target_batteries=target_batteries)
        logger.info(f"Ingested {len(raw_df)} total records from {raw_df['battery_id'].nunique()} batteries.")
        cleaned_df, report = clean_telemetry(
            raw_df,
            config=config,
            output_processed_path=processed_csv,
            rejected_path=rejected_csv,
            metrics_path=metrics_json,
        )

    # Step 2: Compute features
    logger.info(f"Computing features for {len(cleaned_df):,} telemetry rows...")
    features_df = compute_features(cleaned_df)
    features_df.to_parquet(features_parquet, index=False)
    logger.info(f"Saved feature-augmented data to {features_parquet}.")

    # Step 3: Leakage-safe train/val/test splits
    logger.info("Generating leakage-safe battery/cycle splits...")
    train_df, val_df, test_df, split_info = create_splits(
        features_df,
        config=config,
        train_cycle_ratio=0.75,
        metrics_path=split_json,
    )

    # Step 4: Fit scaler strictly on normal training split
    logger.info("Fitting StandardScaler on normal training split...")
    fit_scaler(train_df, scaler_out_path=scaler_joblib)

    print("\n" + "=" * 60)
    print("FEATURE PIPELINE SUMMARY")
    print(f"Total features:  {len(features_df.columns)} columns")
    print(f"Train split:     {len(train_df):,} rows ({split_info['train_cycles']} cycles)")
    print(f"Val split:       {len(val_df):,} rows ({split_info['val_cycles']} cycles)")
    print(f"Test split:      {len(test_df):,} rows ({split_info['test_cycles']} cycles)")
    print(f"Features saved:  {features_parquet}")
    print(f"Scaler saved:    {scaler_joblib}")
    print("=" * 60 + "\n")
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
