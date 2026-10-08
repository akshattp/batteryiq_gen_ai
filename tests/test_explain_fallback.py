"""Unit tests for explanation generation, self-correction loop, and fallback (Section 8.4)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from guardian.genai.explain import generate_explanation, template_explanation
from guardian.genai.llm_client import LLMClient, LLMUnavailable, TemplateClient
from guardian.genai.validator import validate_explanation
from guardian.utils.schemas import ExplanationOutput


@pytest.fixture
def sample_assessment() -> Dict[str, Any]:
    return {
        "risk_level": "WARNING",
        "risk_score": 42.5,
        "telemetry": {
            "temp_c": 36.2,
            "dT_dt": 0.45,
            "voltage_v": 3.75,
            "current_a": 1.5,
            "anomaly_score": 0.48,
            "pred_temp_h10": 37.8,
        },
        "factors": [
            {"name": "Temperature", "raw_value": 36.2, "weighted_score": 15.0},
            {"name": "Anomaly", "raw_value": 0.48, "weighted_score": 12.0},
        ],
        "hard_overrides": [],
    }


def test_template_explanation_validity(sample_assessment: Dict[str, Any]) -> None:
    expl = template_explanation(sample_assessment)
    assert isinstance(expl, ExplanationOutput)
    assert "WARNING" in expl.summary
    assert len(expl.reasons) > 0
    assert len(expl.recommended_actions) > 0
    assert "human" in expl.recommended_actions[-1].lower()

    # Convert to json and validate
    output_json = json.dumps(expl.model_dump())
    violations = validate_explanation(output_json, sample_assessment)
    assert len(violations) == 0, f"Template output produced violations: {[v.to_dict() for v in violations]}"


@pytest.mark.parametrize("level", ["NORMAL", "WARNING", "HIGH", "CRITICAL"])
def test_template_explanation_handles_all_levels(level: str) -> None:
    assessment = {
        "risk_level": level,
        "risk_score": 80.0 if level == "CRITICAL" else (60.0 if level == "HIGH" else (40.0 if level == "WARNING" else 15.0)),
        "telemetry": {
            "temp_c": 51.0 if level == "CRITICAL" else 25.0,
            "dT_dt": 1.2 if level == "CRITICAL" else 0.1,
            "voltage_v": 3.6,
            "current_a": 2.0,
            "anomaly_score": 0.95 if level == "CRITICAL" else 0.1,
            "pred_temp_h10": 53.0 if level == "CRITICAL" else 25.5,
        },
        "factors": [
            {"name": "Temperature", "raw_value": 51.0 if level == "CRITICAL" else 25.0, "weighted_score": 30.0},
        ],
        "hard_overrides": ["Temperature >= 50.0 °C (CRITICAL)"] if level == "CRITICAL" else [],
    }
    expl = template_explanation(assessment)
    output_json = json.dumps(expl.model_dump())
    violations = validate_explanation(output_json, assessment)
    assert len(violations) == 0, f"Violations for level {level}: {[v.to_dict() for v in violations]}"


def test_generate_explanation_offline_template_client(sample_assessment: Dict[str, Any]) -> None:
    client = TemplateClient()
    expl = generate_explanation(sample_assessment, client=client)
    assert isinstance(expl, ExplanationOutput)
    assert "WARNING" in expl.summary
    assert len(expl.reasons) > 0


class MockSelfHealingLLM:
    """Mock LLM that returns a violation on first attempt, then heals on second attempt."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        self.call_count += 1
        if self.call_count == 1:
            # First attempt: invalid (contains forbidden control directive)
            return json.dumps({
                "summary": "Battery risk level is WARNING.",
                "reasons": ["Temperature is 36.2 °C."],
                "recommended_actions": ["Shut down the battery immediately."],
                "confidence_note": "Ground truth",
            })
        else:
            # Second attempt: healed compliant output
            return json.dumps({
                "summary": "Battery risk level is WARNING with risk score 42.5/100.",
                "reasons": [
                    "Temperature is 36.2 °C with anomaly score 0.48.",
                    "Forecast predicts 37.8 °C at t+10 steps.",
                ],
                "recommended_actions": [
                    "Increase telemetry monitoring cadence.",
                    "Inspect thermal cooling performance.",
                    "Human operator verification required before any intervention.",
                ],
                "confidence_note": "Corrected diagnostic summary.",
            })


def test_generate_explanation_self_correction_recovers(sample_assessment: Dict[str, Any]) -> None:
    mock_client = MockSelfHealingLLM()
    expl = generate_explanation(sample_assessment, client=mock_client, max_retries=2)
    assert mock_client.call_count == 2
    assert "WARNING" in expl.summary
    assert "shut down" not in " ".join(expl.recommended_actions).lower()
    assert expl.confidence_note == "Corrected diagnostic summary."


class MockFailingLLM:
    """Mock LLM that consistently returns invalid JSON or unrecoverable responses."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        self.call_count += 1
        return "Not valid JSON at all!"


def test_generate_explanation_exhausted_retries_falls_back(sample_assessment: Dict[str, Any]) -> None:
    failing_client = MockFailingLLM()
    expl = generate_explanation(sample_assessment, client=failing_client, max_retries=2)
    # Should attempt 1 + 2 retries = 3 calls
    assert failing_client.call_count == 3
    # Returns template fallback
    assert isinstance(expl, ExplanationOutput)
    assert "deterministic baseline template" in expl.confidence_note


class MockCrashingLLM:
    """Mock LLM that raises an exception simulating API outage."""

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        raise LLMUnavailable("API server down (HTTP 503)")


def test_generate_explanation_network_error_falls_back(sample_assessment: Dict[str, Any]) -> None:
    crashing_client = MockCrashingLLM()
    expl = generate_explanation(sample_assessment, client=crashing_client)
    # Safely falls back to template explanation
    assert isinstance(expl, ExplanationOutput)
    assert "deterministic baseline template" in expl.confidence_note
