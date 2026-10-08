"""Unit tests for explanation validator (Section 8.3)."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from guardian.genai.validator import Violation, validate_explanation


@pytest.fixture
def sample_assessment() -> Dict[str, Any]:
    return {
        "risk_level": "HIGH",
        "risk_score": 68.4,
        "telemetry": {
            "temp_c": 44.1,
            "dT_dt": 1.85,
            "voltage_v": 3.45,
            "current_a": 2.8,
            "anomaly_score": 0.92,
            "pred_temp_h10": 47.3,
        },
        "factors": [
            {"name": "Temperature", "raw_value": 44.1, "weighted_score": 25.1},
            {"name": "Anomaly", "raw_value": 0.92, "weighted_score": 23.0},
        ],
        "hard_overrides": [],
    }


def test_valid_explanation_passes(sample_assessment: Dict[str, Any]) -> None:
    valid_output = json.dumps({
        "summary": "Battery risk level is HIGH with risk score 68.4/100.",
        "reasons": [
            "Temperature reached 44.1 °C with high anomaly score 0.92.",
            "Forecast predicts 47.3 °C at t+10 steps.",
        ],
        "recommended_actions": [
            "Reduce battery load immediately.",
            "Inspect thermal management system.",
            "Human operator verification required before any intervention.",
        ],
        "confidence_note": "High confidence grounded in telemetry.",
    })

    violations = validate_explanation(valid_output, sample_assessment)
    assert len(violations) == 0, f"Expected 0 violations, got: {[v.to_dict() for v in violations]}"


def test_invalid_json_fails(sample_assessment: Dict[str, Any]) -> None:
    raw = "This is not JSON: Battery risk is HIGH"
    violations = validate_explanation(raw, sample_assessment)
    assert len(violations) >= 1
    assert any(v.rule == "json_syntax" for v in violations)


def test_missing_required_keys_fails(sample_assessment: Dict[str, Any]) -> None:
    incomplete = json.dumps({
        "summary": "Battery risk is HIGH",
        # missing reasons and recommended_actions
    })
    violations = validate_explanation(incomplete, sample_assessment)
    assert len(violations) >= 1
    rules = [v.rule for v in violations]
    assert "schema_missing_key" in rules


def test_forbidden_control_phrases_detected(sample_assessment: Dict[str, Any]) -> None:
    bad_output = json.dumps({
        "summary": "Battery risk level is HIGH.",
        "reasons": ["Temperature is 44.1 °C."],
        "recommended_actions": [
            "Shut down the vehicle battery immediately.",
            "Disconnect the main contactor.",
        ],
        "confidence_note": "Ground truth",
    })
    violations = validate_explanation(bad_output, sample_assessment)
    rules = [v.rule for v in violations]
    assert "forbidden_control_verb" in rules


def test_level_inconsistency_detected(sample_assessment: Dict[str, Any]) -> None:
    # Expected HIGH, but output says WARNING
    mismatched = json.dumps({
        "summary": "Battery risk level is WARNING.",
        "reasons": ["Temperature is 44.1 °C."],
        "recommended_actions": ["Monitor telemetry."],
        "confidence_note": "Ground truth",
    })
    violations = validate_explanation(mismatched, sample_assessment)
    rules = [v.rule for v in violations]
    assert "level_inconsistency" in rules


def test_contradictory_claim_detected(sample_assessment: Dict[str, Any]) -> None:
    contradictory = json.dumps({
        "summary": "Battery is in a SAFE operational state despite HIGH assessment.",
        "reasons": ["Temperature is 44.1 °C."],
        "recommended_actions": ["Monitor telemetry."],
        "confidence_note": "Ground truth",
    })
    violations = validate_explanation(contradictory, sample_assessment)
    rules = [v.rule for v in violations]
    assert "contradictory_claim" in rules


def test_numeric_hallucination_detected(sample_assessment: Dict[str, Any]) -> None:
    # 98.6 °C is an ungrounded hallucination not in sample_assessment
    hallucinated = json.dumps({
        "summary": "Battery risk level is HIGH.",
        "reasons": ["Temperature spiked to 98.6 °C with internal resistance 15.4."],
        "recommended_actions": ["Inspect cooling system."],
        "confidence_note": "Ground truth",
    })
    violations = validate_explanation(hallucinated, sample_assessment)
    rules = [v.rule for v in violations]
    assert "numeric_hallucination" in rules


def test_grounded_numbers_within_tolerance_pass(sample_assessment: Dict[str, Any]) -> None:
    # 44.1 for 44.1, 68.4 for 68.4, 47.3 for 47.3
    grounded = json.dumps({
        "summary": "Battery risk level is HIGH with score 68.4.",
        "reasons": [
            "Observed temperature of 44.1 °C (forecast 47.3 °C).",
            "Anomaly score reached 0.92.",
        ],
        "recommended_actions": ["Reduce load."],
        "confidence_note": "Reliable data.",
    })
    violations = validate_explanation(grounded, sample_assessment)
    assert not any(v.rule == "numeric_hallucination" for v in violations)
