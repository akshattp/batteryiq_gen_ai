"""Validation module for battery telemetry data integrity and ranges."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "battery_id",
    "cycle",
    "mode",
    "t_s",
    "timestamp",
    "voltage_v",
    "current_a",
    "temp_c",
    "ambient_c",
    "soc_pct",
    "capacity_ah",
    "soh_pct",
    "label",
    "fault_type",
]

NON_NULLABLE_COLUMNS: List[str] = [
    "battery_id",
    "cycle",
    "mode",
    "t_s",
    "timestamp",
    "voltage_v",
    "current_a",
    "temp_c",
    "ambient_c",
    "soc_pct",
    "label",
]


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate DataFrame against required telemetry columns.

    Args:
        df: DataFrame to validate.

    Returns:
        Tuple of (is_valid, missing_columns).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return (len(missing) == 0, missing)


def validate_non_nulls(df: pd.DataFrame, columns: List[str] | None = None) -> Tuple[bool, Dict[str, int]]:
    """Verify that required columns contain no null or NaN values.

    Args:
        df: DataFrame to check.
        columns: Specific columns to verify. Defaults to NON_NULLABLE_COLUMNS.

    Returns:
        Tuple of (all_clean, dict_of_nan_counts).
    """
    cols = columns or NON_NULLABLE_COLUMNS
    nan_counts: Dict[str, int] = {}
    for col in cols:
        if col in df.columns:
            cnt = int(df[col].isna().sum())
            if cnt > 0:
                nan_counts[col] = cnt
    return (len(nan_counts) == 0, nan_counts)


def validate_ranges(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Identify rows that violate physical sensor ranges from config.

    Args:
        df: Input DataFrame.
        config: Project configuration containing range limits.

    Returns:
        DataFrame containing rejected / out-of-range rows.
    """
    data_cfg = config.get("data", {})
    v_min = data_cfg.get("v_min", 2.0)
    v_max = data_cfg.get("v_max", 4.5)
    temp_min = data_cfg.get("temp_min", -20.0)
    temp_max = data_cfg.get("temp_max", 80.0)
    current_abs_max = data_cfg.get("current_abs_max", 50.0)

    cond_v = (df["voltage_v"] < v_min) | (df["voltage_v"] > v_max)
    cond_temp = (df["temp_c"] < temp_min) | (df["temp_c"] > temp_max)
    cond_curr = df["current_a"].abs() > current_abs_max

    rejected_mask = cond_v | cond_temp | cond_curr
    return df[rejected_mask].copy()
