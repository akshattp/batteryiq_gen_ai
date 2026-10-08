"""tests/test_risk.py - Unit and acceptance test suite for deterministic risk engine (Milestone 5).

Verifies >= 15 table-driven cases, every hard safety override, boundary conditions,
and mathematical monotonicity (raising temperature/current/anomaly never lowers risk).
"""

from __future__ import annotations

import numpy as np
import pytest

from guardian.risk.engine import assess
from guardian.utils.schemas import RiskAssessment, RiskLevel


# 18 Table-Driven Test Cases
TEST_CASES = [
    # Case 1: Ideal nominal state -> NORMAL (< 30)
    {
        "name": "ideal_nominal",
        "state": {"temp_c": 24.0, "current_a": -1.0, "voltage_v": 3.7, "dT_dt": 0.1},
        "anomaly": 0.01,
        "pred_temp": 24.5,
        "expected_level": RiskLevel.NORMAL,
        "min_score": 0.0,
        "max_score": 30.0,
        "expect_override": False,
    },
    # Case 2: Hard Override - Critical measured temp (T >= 50.0) -> CRITICAL
    {
        "name": "override_t_crit_measured",
        "state": {"temp_c": 51.5, "current_a": -1.0, "voltage_v": 3.7, "dT_dt": 0.5},
        "anomaly": 0.05,
        "pred_temp": 52.0,
        "expected_level": RiskLevel.CRITICAL,
        "min_score": 75.0,
        "max_score": 100.0,
        "expect_override": True,
    },
    # Case 3: Hard Override - Critical predicted temp (T_pred >= 50.0) even if current T is mild
    {
        "name": "override_t_crit_predicted",
        "state": {"temp_c": 32.0, "current_a": -1.5, "voltage_v": 3.6, "dT_dt": 2.0},
        "anomaly": 0.10,
        "pred_temp": 50.8,
        "expected_level": RiskLevel.CRITICAL,
        "min_score": 75.0,
        "max_score": 100.0,
        "expect_override": True,
    },
    # Case 4: Hard Override - High measured temp (T >= 42.0) -> at least HIGH
    {
        "name": "override_t_high_measured",
        "state": {"temp_c": 43.0, "current_a": -0.5, "voltage_v": 3.7, "dT_dt": 0.2},
        "anomaly": 0.02,
        "pred_temp": 43.2,
        "expected_level": RiskLevel.HIGH,
        "min_score": 55.0,
        "max_score": 100.0,
        "expect_override": True,
    },
    # Case 5: Hard Override - Severe Undervoltage (V < 2.5V) -> at least HIGH
    {
        "name": "override_voltage_undervoltage",
        "state": {"temp_c": 25.0, "current_a": -1.0, "voltage_v": 2.2, "dT_dt": 0.1},
        "anomaly": 0.02,
        "pred_temp": 25.5,
        "expected_level": RiskLevel.HIGH,
        "min_score": 55.0,
        "max_score": 100.0,
        "expect_override": True,
    },
    # Case 6: Hard Override - Severe Overvoltage (V > 4.2V) -> at least HIGH
    {
        "name": "override_voltage_overvoltage",
        "state": {"temp_c": 26.0, "current_a": 1.5, "voltage_v": 4.35, "dT_dt": 0.2},
        "anomaly": 0.03,
        "pred_temp": 26.5,
        "expected_level": RiskLevel.HIGH,
        "min_score": 55.0,
        "max_score": 100.0,
        "expect_override": True,
    },
    # Case 7: Boundary - Temp exactly at T_warn (35.0 C) -> component score = 0
    {
        "name": "boundary_t_warn",
        "state": {"temp_c": 35.0, "current_a": 0.0, "voltage_v": 3.35, "dT_dt": 0.0},
        "anomaly": 0.0,
        "pred_temp": 35.0,
        "expected_level": RiskLevel.NORMAL,
        "min_score": 0.0,
        "max_score": 5.0,
        "expect_override": False,
    },
    # Case 8: Boundary - Temp exactly at T_crit (50.0 C) -> component score = 1.0 -> CRITICAL
    {
        "name": "boundary_t_crit",
        "state": {"temp_c": 50.0, "current_a": 0.0, "voltage_v": 3.35, "dT_dt": 0.0},
        "anomaly": 0.0,
        "pred_temp": 50.0,
        "expected_level": RiskLevel.CRITICAL,
        "min_score": 75.0,
        "max_score": 100.0,
        "expect_override": True,
    },
    # Case 9: Boundary - Current rated (2.0A) -> current component ratio = 1.0
    {
        "name": "boundary_rated_current",
        "state": {"temp_c": 24.0, "current_a": -2.0, "voltage_v": 3.35, "dT_dt": 0.0},
        "anomaly": 0.0,
        "pred_temp": 24.0,
        "expected_level": RiskLevel.NORMAL,
        "min_score": 9.0,
        "max_score": 15.0,
        "expect_override": False,
    },
    # Case 10: Boundary - Voltage at midpoint (3.35V) -> voltage component = 0.0
    {
        "name": "boundary_voltage_center",
        "state": {"temp_c": 25.0, "current_a": 0.0, "voltage_v": 3.35, "dT_dt": 0.0},
        "anomaly": 0.0,
        "pred_temp": 25.0,
        "expected_level": RiskLevel.NORMAL,
        "min_score": 0.0,
        "max_score": 5.0,
        "expect_override": False,
    },
    # Case 11: High anomaly score only (0.98) -> elevated WARNING
    {
        "name": "high_anomaly_only",
        "state": {"temp_c": 24.0, "current_a": -1.0, "voltage_v": 3.7, "dT_dt": 0.1},
        "anomaly": 0.98,
        "pred_temp": 24.5,
        "expected_level": RiskLevel.WARNING,
        "min_score": 30.0,
        "max_score": 55.0,
        "expect_override": False,
    },
    # Case 12: High heating rate dT/dt only (3.5 C/min) -> WARNING/HIGH
    {
        "name": "rapid_heating_trend",
        "state": {"temp_c": 32.0, "current_a": -2.5, "voltage_v": 3.6, "dT_dt": 3.8},
        "anomaly": 0.40,
        "pred_temp": 34.0,
        "expected_level": RiskLevel.WARNING,
        "min_score": 35.0,
        "max_score": 60.0,
        "expect_override": False,
    },
    # Case 13: Moderate multi-factor stress -> WARNING level
    {
        "name": "moderate_multi_factor",
        "state": {"temp_c": 37.0, "current_a": -2.2, "voltage_v": 3.9, "dT_dt": 1.5},
        "anomaly": 0.65,
        "pred_temp": 38.5,
        "expected_level": RiskLevel.WARNING,
        "min_score": 30.0,
        "max_score": 55.0,
        "expect_override": False,
    },
    # Case 14: Cumulative multi-factor severe stress -> HIGH level (55-75 pts) without single override
    {
        "name": "severe_multi_factor",
        "state": {"temp_c": 39.0, "current_a": -2.6, "voltage_v": 4.05, "dT_dt": 2.0},
        "anomaly": 0.80,
        "pred_temp": 39.5,
        "expected_level": RiskLevel.HIGH,
        "min_score": 55.0,
        "max_score": 75.0,
        "expect_override": False,
    },
    # Case 15: Maximum combined catastrophe -> CRITICAL
    {
        "name": "maximum_catastrophe",
        "state": {"temp_c": 55.0, "current_a": -10.0, "voltage_v": 1.9, "dT_dt": 10.0},
        "anomaly": 1.0,
        "pred_temp": 60.0,
        "expected_level": RiskLevel.CRITICAL,
        "min_score": 90.0,
        "max_score": 100.0,
        "expect_override": True,
    },
]


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_risk_assessment_table_cases(case: dict):
    """Run table-driven risk engine evaluation covering overrides, levels, and bounds."""
    res = assess(
        state=case["state"],
        anomaly_score=case["anomaly"],
        pred_temp=case["pred_temp"],
    )

    assert isinstance(res, RiskAssessment)
    assert res.risk_level == case["expected_level"], (
        f"Case '{case['name']}': Expected {case['expected_level']}, got {res.risk_level} (score: {res.risk_score})"
    )
    assert case["min_score"] <= res.risk_score <= case["max_score"], (
        f"Case '{case['name']}': Score {res.risk_score} outside [{case['min_score']}, {case['max_score']}]"
    )
    if case["expect_override"]:
        assert len(res.hard_overrides) > 0, f"Case '{case['name']}': Expected hard override, but none triggered."
    else:
        assert len(res.hard_overrides) == 0, f"Case '{case['name']}': Unexpected override: {res.hard_overrides}"


def test_temperature_monotonicity():
    """Mathematical property: Raising temperature must strictly never decrease risk score."""
    temps = np.linspace(20.0, 60.0, 25)
    scores = []
    base_state = {"current_a": -1.5, "voltage_v": 3.7, "dT_dt": 0.5}

    for t in temps:
        state = dict(base_state, temp_c=t)
        res = assess(state=state, anomaly_score=0.1, pred_temp=t)
        scores.append(res.risk_score)

    for i in range(len(scores) - 1):
        assert scores[i + 1] >= scores[i], (
            f"Temperature non-monotonicity: score at {temps[i+1]}°C ({scores[i+1]}) < score at {temps[i]}°C ({scores[i]})"
        )


def test_current_monotonicity():
    """Mathematical property: Raising current magnitude must never decrease risk score."""
    currents = np.linspace(0.0, 6.0, 20)
    scores = []
    base_state = {"temp_c": 28.0, "voltage_v": 3.7, "dT_dt": 0.2}

    for curr in currents:
        state = dict(base_state, current_a=-curr)
        res = assess(state=state, anomaly_score=0.1, pred_temp=28.0)
        scores.append(res.risk_score)

    for i in range(len(scores) - 1):
        assert scores[i + 1] >= scores[i], (
            f"Current non-monotonicity: score at {currents[i+1]}A ({scores[i+1]}) < score at {currents[i]}A ({scores[i]})"
        )


def test_anomaly_monotonicity():
    """Mathematical property: Raising anomaly score must never decrease risk score."""
    anomalies = np.linspace(0.0, 1.0, 20)
    scores = []
    state = {"temp_c": 28.0, "current_a": -1.0, "voltage_v": 3.7, "dT_dt": 0.2}

    for anom in anomalies:
        res = assess(state=state, anomaly_score=anom, pred_temp=28.0)
        scores.append(res.risk_score)

    for i in range(len(scores) - 1):
        assert scores[i + 1] >= scores[i], (
            f"Anomaly non-monotonicity: score at {anomalies[i+1]} ({scores[i+1]}) < score at {anomalies[i]} ({scores[i]})"
        )


def test_contributing_factor_ranking():
    """Verify contributing factors are sorted descending by weighted score."""
    state = {"temp_c": 40.0, "current_a": -1.0, "voltage_v": 3.7, "dT_dt": 0.1}
    # Anomaly=1.0 contributes 25 points, while temp at 40 C contributes: (40-35)/(50-35) * 30 = 10 pts
    res = assess(state=state, anomaly_score=1.0, pred_temp=40.0)

    assert len(res.factors) == 5
    # Factor #1 must be Anomaly
    assert res.factors[0].name == "Anomaly"
    assert res.factors[0].weighted_score >= res.factors[1].weighted_score
    # All factors must be monotonically descending
    for i in range(len(res.factors) - 1):
        assert res.factors[i].weighted_score >= res.factors[i + 1].weighted_score
