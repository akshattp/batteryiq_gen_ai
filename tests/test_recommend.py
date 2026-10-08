"""tests/test_recommend.py - Recommendation catalog and safety bounds test suite (Milestone 6).

Verifies:
1. Recommendations strictly NEVER contain forbidden autonomous control verbs.
2. Every recommendation concludes with human-in-the-loop engineering disclaimer.
3. Relevant advisories are generated across risk levels and contributing factors.
"""

from __future__ import annotations

import itertools
import pytest

from guardian.risk.recommend import get_recommendations, FORBIDDEN_WORDS, HUMAN_IN_THE_LOOP_DISCLAIMER
from guardian.utils.schemas import Recommendation, RiskFactor, RiskLevel


FACTOR_COMBINATIONS = [
    [],
    ["Temperature"],
    ["Temp trend"],
    ["Current"],
    ["Voltage"],
    ["Anomaly"],
    ["Decoupling"],
    ["Temperature", "Temp trend"],
    ["Temperature", "Current", "Anomaly"],
    ["Voltage", "Current"],
    ["Anomaly", "Decoupling"],
]


@pytest.mark.parametrize("level", [RiskLevel.NORMAL, RiskLevel.WARNING, RiskLevel.HIGH, RiskLevel.CRITICAL])
@pytest.mark.parametrize("factors", FACTOR_COMBINATIONS)
def test_recommendations_never_contain_forbidden_control_verbs(level: RiskLevel, factors: list[str]):
    """Rigorous acceptance test: Grep all recommendations for forbidden autonomous commands."""
    rec = get_recommendations(risk_level=level, contributing_factors=factors)

    assert isinstance(rec, Recommendation)
    assert len(rec.actions) >= 2, "Expected at least 1 action plus human-in-the-loop disclaimer."

    # Grep test every action
    for action in rec.actions:
        action_lower = action.lower()
        for forbidden in FORBIDDEN_WORDS:
            assert forbidden not in action_lower, (
                f"VIOLATION: Recommendation contained forbidden control directive '{forbidden}': '{action}'"
            )


@pytest.mark.parametrize("level", [RiskLevel.NORMAL, RiskLevel.WARNING, RiskLevel.HIGH, RiskLevel.CRITICAL])
def test_every_recommendation_ends_with_human_disclaimer(level: RiskLevel):
    """Verify human-in-the-loop advisory is always the terminating action."""
    rec = get_recommendations(risk_level=level, contributing_factors=["Temperature", "Current"])
    assert rec.actions[-1] == HUMAN_IN_THE_LOOP_DISCLAIMER
    assert rec.disclaimer == HUMAN_IN_THE_LOOP_DISCLAIMER


def test_factor_specific_advisories():
    """Verify factor-specific diagnostic actions are included when factors are present."""
    rec_trend = get_recommendations(risk_level=RiskLevel.HIGH, contributing_factors=["Temp trend"])
    actions_joined = " ".join(rec_trend.actions).lower()
    assert "heating curve" in actions_joined or "ramp" in actions_joined or "runaway" in actions_joined

    rec_current = get_recommendations(risk_level=RiskLevel.HIGH, contributing_factors=["Current"])
    actions_current = " ".join(rec_current.actions).lower()
    assert "busbar" in actions_current or "resistance" in actions_current

    rec_volt = get_recommendations(risk_level=RiskLevel.HIGH, contributing_factors=["Voltage"])
    actions_volt = " ".join(rec_volt.actions).lower()
    assert "balancing" in actions_volt or "voltage" in actions_volt


def test_recommendation_accepts_riskfactor_objects():
    """Verify function accepts structured RiskFactor instances as well as string names."""
    factors = [
        RiskFactor(name="Temperature", raw_value=48.0, component_score=0.9, weight=0.3, weighted_score=27.0),
        RiskFactor(name="Anomaly", raw_value=0.99, component_score=0.99, weight=0.25, weighted_score=24.75),
    ]
    rec = get_recommendations(risk_level=RiskLevel.CRITICAL, contributing_factors=factors)
    assert len(rec.actions) > 0
    assert rec.risk_level == RiskLevel.CRITICAL
