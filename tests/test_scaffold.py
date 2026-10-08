"""tests/test_scaffold.py - Milestone M0 repository and utility validation tests."""

import pytest
import numpy as np
import random
from pathlib import Path

from guardian.utils.config import load_config, get_config
from guardian.utils.logging import setup_logger
from guardian.utils.seed import set_seed
from guardian.utils.schemas import (
    RiskAssessment,
    RiskFactor,
    RiskLevel,
    TelemetryLabel,
    ExplanationOutput,
)
from guardian.data.ingest import find_mat_files
from guardian.genai.explain import template_explanation
from guardian.genai.validator import validate_explanation
from guardian.genai.codeexec import verify_code_safety


def test_config_loads_properly():
    config = load_config()
    assert isinstance(config, dict)
    assert config["seed"] == 42
    assert "paths" in config
    assert "data" in config
    assert "risk" in config
    assert "vae" in config
    assert "llm" in config
    assert config["data"]["train_batteries"] == ["B0005", "B0006", "B0007"]
    assert config["data"]["test_battery"] == "B0018"


def test_seed_reproducibility():
    set_seed(42)
    val1 = random.random()
    arr1 = np.random.rand(5)

    set_seed(42)
    val2 = random.random()
    arr2 = np.random.rand(5)

    assert val1 == val2
    assert np.allclose(arr1, arr2)


def test_logger_setup():
    logger = setup_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"


def test_schemas_instantiation():
    factor = RiskFactor(
        name="Temperature",
        raw_value=45.2,
        component_score=0.8,
        weight=0.3,
        weighted_score=0.24,
    )
    assessment = RiskAssessment(
        battery_id="B0005",
        cycle=10,
        t_s=120.0,
        timestamp="2026-10-08T00:00:00",
        risk_score=62.5,
        risk_level=RiskLevel.HIGH,
        anomaly_score=0.15,
        predicted_temp_c=46.1,
        factors=[factor],
    )
    d = assessment.to_dict()
    assert d["battery_id"] == "B0005"
    assert d["risk_level"] == "HIGH"
    assert len(d["factors"]) == 1


def test_raw_dataset_discovered():
    files = find_mat_files("data/raw")
    assert len(files) >= 4, f"Expected >= 4 .mat files, found {len(files)}"
    names = [f.name for f in files]
    for expected in ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]:
        assert expected in names, f"Missing expected dataset file: {expected}"


def test_template_explanation_deterministic():
    assessment_dict = {
        "risk_level": "WARNING",
        "risk_score": 45.0,
        "factors": [{"name": "Temperature", "raw_value": 38.0, "weighted_score": 12.0}],
        "hard_overrides": [],
    }
    exp = template_explanation(assessment_dict)
    assert isinstance(exp, ExplanationOutput)
    assert "WARNING" in exp.summary
    assert len(exp.recommended_actions) > 0


def test_validator_detects_forbidden_control_verb():
    bad_output = '{"summary": "Risk is HIGH", "reasons": ["High temp"], "recommended_actions": ["Shut down the vehicle now"]}'
    violations = validate_explanation(bad_output, {"risk_level": "HIGH"})
    assert any(v.rule == "forbidden_control_verb" for v in violations)


def test_codeexec_ast_safety():
    unsafe_code_import = "import os\nresult = 1"
    is_safe, reason = verify_code_safety(unsafe_code_import)
    assert not is_safe
    assert "Import" in reason

    unsafe_code_dunder = "x = df.__dict__"
    is_safe2, reason2 = verify_code_safety(unsafe_code_dunder)
    assert not is_safe2

    safe_code = "result = df['temp_c'].max()"
    is_safe3, reason3 = verify_code_safety(safe_code)
    assert is_safe3
