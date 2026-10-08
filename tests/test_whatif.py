"""tests/test_whatif.py - Acceptance test for What-If Simulation & OOD Guard (M14)."""

from __future__ import annotations

import pytest

from guardian.risk.whatif import is_out_of_distribution, simulate_whatif
from guardian.utils.schemas import RiskLevel


@pytest.fixture
def base_state() -> dict:
    return {
        "battery_id": "B0005",
        "current_a": 1.5,
        "temp_c": 25.0,
        "ambient_c": 24.0,
        "voltage_v": 3.8,
        "soc_pct": 80.0,
        "dT_dt": 0.02,
    }


def test_whatif_monotonic_thermal_sanity(base_state: dict):
    """Verify monotonic sanity: higher discharge current strictly does not reduce predicted temperature."""
    currents = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    predicted_temps = []

    for curr in currents:
        mod_params = {"current_a": curr}
        assessment, _ = simulate_whatif(base_state, mod_params)
        predicted_temps.append(assessment.predicted_temp_c)

    # Monotonicity check: each subsequent temp must be >= previous
    for i in range(len(predicted_temps) - 1):
        assert predicted_temps[i + 1] >= predicted_temps[i] - 1e-5, (
            f"Thermal monotonicity violated: I={currents[i]} -> T={predicted_temps[i]:.2f}, "
            f"I={currents[i+1]} -> T={predicted_temps[i+1]:.2f}"
        )


def test_whatif_ood_badge_triggers(base_state: dict):
    """Verify out-of-distribution guardrail triggers when parameters exceed training envelope."""
    # 1. Within nominal distribution
    nominal_params = {"current_a": 1.8, "temp_c": 28.0, "ambient_c": 24.0}
    _, is_ood_nominal = simulate_whatif(base_state, nominal_params)
    assert is_ood_nominal is False, "Nominal parameters falsely flagged as out of distribution"

    # 2. Extreme current (> 3.0A)
    high_curr_params = {"current_a": 4.5}
    _, is_ood_curr = simulate_whatif(base_state, high_curr_params)
    assert is_ood_curr is True, "High current (4.5A) failed to trigger OOD guard"

    # 3. Extreme temperature (> 45.0°C)
    high_temp_params = {"temp_c": 55.0}
    _, is_ood_temp = simulate_whatif(base_state, high_temp_params)
    assert is_ood_temp is True, "High temperature (55.0°C) failed to trigger OOD guard"


def test_whatif_risk_escalation(base_state: dict):
    """Verify that simulated adverse conditions appropriately escalate risk level and score."""
    # Severe scenario: 52°C triggers CRITICAL override
    severe_params = {"temp_c": 52.0, "current_a": 3.2}
    assessment, is_ood = simulate_whatif(base_state, severe_params)

    assert assessment.risk_level == RiskLevel.CRITICAL
    assert assessment.risk_score >= 75.0
    assert len(assessment.hard_overrides) > 0
    assert is_ood is True
