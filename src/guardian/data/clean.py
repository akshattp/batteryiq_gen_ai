"""Data cleaning pipeline for battery telemetry.

Handles deduplication, missing value forward-filling, physical range clipping/rejection,
and generates the data quality report artifacts/metrics/data_quality.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from guardian.data.validate import validate_ranges, REQUIRED_COLUMNS
from guardian.utils.logging import setup_logger

logger = setup_logger("guardian.data.clean")


def clean_telemetry(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_processed_path: str | Path = "data/processed/battery_telemetry.csv",
    rejected_path: str | Path = "data/interim/rejected_rows.csv",
    metrics_path: str | Path = "artifacts/metrics/data_quality.json",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean telemetry DataFrame according to Section 5.3 specifications.

    Args:
        df: Raw combined DataFrame.
        config: Configuration dictionary.
        output_processed_path: Path to write cleaned CSV.
        rejected_path: Path to write out-of-bounds rejected rows.
        metrics_path: Path to write quality metrics JSON.

    Returns:
        Tuple of (cleaned_df, quality_report_dict).
    """
    initial_rows = len(df)
    logger.info(f"Starting data cleaning on {initial_rows} initial rows...")

    # 1. Deduplicate timestamps within (battery_id, cycle, mode, t_s)
    df_dedup = df.drop_duplicates(subset=["battery_id", "cycle", "mode", "t_s"], keep="first").copy()
    dropped_dups = initial_rows - len(df_dedup)
    logger.info(f"Dropped {dropped_dups} duplicate timestamp rows.")

    # 2. Identify physical range violations
    rejected_df = validate_ranges(df_dedup, config)
    rejected_count = len(rejected_df)
    if rejected_count > 0:
        logger.warning(f"Flagged {rejected_count} out-of-range rows. Saving to {rejected_path}.")
        Path(rejected_path).parent.mkdir(parents=True, exist_ok=True)
        rejected_df.to_csv(rejected_path, index=False)
        # Drop rejected rows from cleaned dataset
        df_clean = df_dedup.drop(rejected_df.index).copy()
    else:
        df_clean = df_dedup.copy()
        # Ensure empty rejected file exists
        Path(rejected_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=df.columns).to_csv(rejected_path, index=False)

    # 3. Missing values handling (forward-fill <= 3 steps)
    numeric_cols = ["voltage_v", "current_a", "temp_c", "soc_pct", "capacity_ah", "soh_pct"]
    nan_before = int(df_clean[numeric_cols].isna().sum().sum())
    # Group by battery and cycle before ffill to prevent leakage across cycles
    df_clean[numeric_cols] = df_clean.groupby(["battery_id", "cycle"])[numeric_cols].transform(
        lambda grp: grp.ffill(limit=3)
    )
    nan_after = int(df_clean[numeric_cols].isna().sum().sum())
    filled_missing = nan_before - nan_after

    # If any remaining NaNs exist in required cols, drop those rows
    df_clean = df_clean.dropna(subset=["voltage_v", "current_a", "temp_c", "soc_pct"]).copy()

    # Ensure correct column ordering and sorting
    ordered_cols = [c for c in REQUIRED_COLUMNS if c in df_clean.columns]
    df_clean = df_clean[ordered_cols].sort_values(by=["battery_id", "cycle", "t_s"]).reset_index(drop=True)

    final_rows = len(df_clean)
    logger.info(f"Cleaning complete: {final_rows} rows retained ({initial_rows - final_rows} removed).")

    # 4. Generate quality report
    report: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "cleaned_rows": final_rows,
        "dropped_duplicates": dropped_dups,
        "rejected_range_rows": rejected_count,
        "missing_filled": filled_missing,
        "batteries": sorted(list(df_clean["battery_id"].unique())),
        "battery_counts": df_clean["battery_id"].value_counts().to_dict(),
        "cycle_counts_per_battery": df_clean.groupby("battery_id")["cycle"].nunique().to_dict(),
        "voltage_range": [float(df_clean["voltage_v"].min()), float(df_clean["voltage_v"].max())],
        "temp_range": [float(df_clean["temp_c"].min()), float(df_clean["temp_c"].max())],
        "current_range": [float(df_clean["current_a"].min()), float(df_clean["current_a"].max())],
        "soc_range": [float(df_clean["soc_pct"].min()), float(df_clean["soc_pct"].max())],
    }

    # Save outputs
    Path(output_processed_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_processed_path, index=False)
    logger.info(f"Saved processed dataset to {output_processed_path}.")

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved data quality report to {metrics_path}.")

    return df_clean, report
