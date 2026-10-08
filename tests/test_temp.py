"""tests/test_temp.py - Unit and acceptance tests for temperature predictor (Milestone 4)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from guardian.models.temp_predictor import TemperaturePredictor, build_lagged_features


def test_temp_model_prediction_shape():
    """Verify temperature model inference produces expected shape and plausible values."""
    model_path = Path("artifacts/models/temp_model.joblib")
    if not model_path.exists():
        pytest.skip("Model not trained yet")

    predictor = TemperaturePredictor.load(model_path)
    assert predictor is not None

    # Synthetic test input matching feature names
    dummy_input = pd.DataFrame({col: [25.0] for col in predictor.feature_names})
    pred = predictor.predict(dummy_input)
    assert len(pred) == 1
    assert 15.0 <= pred[0] <= 60.0  # Plausible Li-ion temperature range


def test_temp_metrics_beats_persistence():
    """Acceptance criterion: LightGBM MAE must beat persistence baseline."""
    metrics_path = Path("artifacts/metrics/temp.json")
    assert metrics_path.exists(), f"Metrics JSON missing at {metrics_path}. Run 'python scripts/train_temp.py' first."

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    lgbm_mae = metrics["lgbm"]["mae"]
    persist_mae = metrics["persistence"]["mae"]

    assert lgbm_mae < persist_mae, (
        f"Acceptance failure: LightGBM MAE ({lgbm_mae:.4f}) did not beat Persistence MAE ({persist_mae:.4f})"
    )
    assert metrics["lgbm"]["beats_persistence"] is True
    assert metrics["lgbm"]["r2"] > 0.0


def test_temp_artifacts_exist():
    """Assert all required model and figure artifacts exist on disk."""
    assert Path("artifacts/models/temp_model.joblib").exists()
    assert Path("artifacts/models/temp_linear_baseline.joblib").exists()
    assert Path("artifacts/figures/temp_pred_vs_actual.png").exists()
    assert Path("artifacts/figures/temp_feature_importance.png").exists()


def test_subgroup_evaluations_present():
    """Verify error breakdown by mode and ambient temperature exist in metrics."""
    with open("artifacts/metrics/temp.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "by_mode" in metrics
    assert "charge" in metrics["by_mode"]
    assert "discharge" in metrics["by_mode"]
    assert "by_ambient" in metrics
    assert len(metrics["by_ambient"]) > 0
