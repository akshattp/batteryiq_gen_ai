"""tests/test_data.py - Milestone M1 acceptance test suite.

Verifies:
1. Processed dataset exists and is non-empty.
2. Schema columns match REQUIRED_COLUMNS exactly.
3. No NaN / nulls in required non-nullable columns.
4. Physical sensor ranges adhere to physical limits from config.
5. At least 3 distinct batteries are present (B0005, B0006, B0007, B0018).
6. Coulomb-counted SOC is within [0.0, 100.0].
7. Data quality metrics JSON was produced with valid numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from guardian.utils.config import load_config
from guardian.data.validate import (
    REQUIRED_COLUMNS,
    NON_NULLABLE_COLUMNS,
    validate_schema,
    validate_non_nulls,
)


@pytest.fixture(scope="module")
def processed_df() -> pd.DataFrame:
    """Load the processed battery telemetry dataset."""
    config = load_config()
    proc_path = Path(config.get("paths", {}).get("data_processed", "data/processed")) / "battery_telemetry.csv"
    assert proc_path.exists(), f"Processed dataset not found at {proc_path}. Run 'make data' first."
    df = pd.read_csv(proc_path)
    assert not df.empty, "Processed dataset is empty."
    return df


@pytest.fixture(scope="module")
def quality_metrics() -> dict:
    """Load data quality report metrics."""
    config = load_config()
    metrics_path = Path(config.get("paths", {}).get("metrics_dir", "artifacts/metrics")) / "data_quality.json"
    assert metrics_path.exists(), f"Quality report not found at {metrics_path}."
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_schema_columns_exact(processed_df: pd.DataFrame):
    """Assert all required columns are present and match REQUIRED_COLUMNS."""
    is_valid, missing = validate_schema(processed_df)
    assert is_valid, f"Missing required columns in dataset: {missing}"
    assert list(processed_df.columns) == REQUIRED_COLUMNS, (
        f"Columns mismatch.\nExpected: {REQUIRED_COLUMNS}\nGot: {list(processed_df.columns)}"
    )


def test_no_nan_in_required_columns(processed_df: pd.DataFrame):
    """Assert no null / NaN values in non-nullable columns."""
    is_clean, nan_counts = validate_non_nulls(processed_df, NON_NULLABLE_COLUMNS)
    assert is_clean, f"Found NaN values in required columns: {nan_counts}"


def test_physical_ranges_valid(processed_df: pd.DataFrame):
    """Assert voltage, temperature, and current are within physical boundaries."""
    config = load_config()
    data_cfg = config.get("data", {})
    v_min = data_cfg.get("v_min", 2.0)
    v_max = data_cfg.get("v_max", 4.5)
    temp_min = data_cfg.get("temp_min", -20.0)
    temp_max = data_cfg.get("temp_max", 80.0)
    current_abs_max = data_cfg.get("current_abs_max", 50.0)

    assert processed_df["voltage_v"].min() >= v_min, f"Voltage below min {v_min}: {processed_df['voltage_v'].min()}"
    assert processed_df["voltage_v"].max() <= v_max, f"Voltage above max {v_max}: {processed_df['voltage_v'].max()}"

    assert processed_df["temp_c"].min() >= temp_min, f"Temp below min {temp_min}: {processed_df['temp_c'].min()}"
    assert processed_df["temp_c"].max() <= temp_max, f"Temp above max {temp_max}: {processed_df['temp_c'].max()}"

    assert processed_df["current_a"].abs().max() <= current_abs_max, (
        f"Current exceeded max {current_abs_max}: {processed_df['current_a'].abs().max()}"
    )


def test_battery_count_and_diversity(processed_df: pd.DataFrame):
    """Assert at least 3 distinct batteries are present in the dataset."""
    unique_bats = processed_df["battery_id"].unique()
    assert len(unique_bats) >= 3, f"Expected >= 3 batteries, got {len(unique_bats)}: {unique_bats}"
    # Verify core expected NASA batteries
    for expected in ["B0005", "B0006", "B0007"]:
        assert expected in unique_bats, f"Expected battery {expected} missing from dataset."


def test_soc_bounds(processed_df: pd.DataFrame):
    """Assert Coulomb-counted SOC is normalized and bounded in [0.0, 100.0]."""
    min_soc = processed_df["soc_pct"].min()
    max_soc = processed_df["soc_pct"].max()
    assert min_soc >= 0.0, f"SOC below 0%: {min_soc}"
    assert max_soc <= 100.0, f"SOC above 100%: {max_soc}"


def test_quality_report_metrics(quality_metrics: dict):
    """Assert artifacts/metrics/data_quality.json has valid contents."""
    assert quality_metrics["cleaned_rows"] > 0
    assert quality_metrics["cleaned_rows"] <= quality_metrics["initial_rows"]
    assert len(quality_metrics["batteries"]) >= 3
