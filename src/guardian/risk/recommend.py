"""Rule-based recommendation catalog (Milestone 6).

Maps (risk_level, contributing_factors) to structured, actionable advisories.
Strictly decision-support only: contains ZERO autonomous control directives
(never issues direct shutdown, disconnect, or firmware override commands).
Always concludes with a human-in-the-loop escalation advisory.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union
from guardian.utils.schemas import Recommendation, RiskFactor, RiskLevel

# Standard human-in-the-loop advisory suffix
HUMAN_IN_THE_LOOP_DISCLAIMER = (
    "Decision-support prototype. Not for vehicle control. Escalate to human engineer."
)

FORBIDDEN_WORDS = [
    "shut down",
    "shutdown",
    "disconnect",
    "emergency stop",
    "disable bms",
    "ignore",
    "override safety",
]


def get_recommendations(
    risk_level: RiskLevel,
    contributing_factors: Union[List[str], List[RiskFactor]],
    hard_overrides: Optional[List[str]] = None,
    event_id: Optional[str] = None,
) -> Recommendation:
    """Generate ordered, non-autonomous engineering advisories for a given risk state.

    Args:
        risk_level: Current assessed RiskLevel (NORMAL, WARNING, HIGH, CRITICAL).
        contributing_factors: List of primary risk factor names or RiskFactor objects.
        hard_overrides: Optional list of hard override strings.
        event_id: Optional associated event ID.

    Returns:
        Recommendation dataclass containing prioritized action list.
    """
    # Extract factor names
    factor_names: List[str] = []
    for f in contributing_factors:
        if isinstance(f, RiskFactor):
            factor_names.append(f.name.lower())
        else:
            factor_names.append(str(f).lower())

    actions: List[str] = []

    # Priority Action 1: Level-specific baseline
    if risk_level == RiskLevel.CRITICAL:
        actions.append(
            "Immediately reduce charge/discharge current demand to minimum idle to lower resistive heating."
        )
        actions.append(
            "Initiate active cooling system diagnostics and inspect external coolant circulation pumps."
        )
        actions.append(
            "Prepare for manual system inspection; notify facility safety officer of critical thermal state."
        )

    elif risk_level == RiskLevel.HIGH:
        actions.append(
            "Reduce pack operational power by 50% to mitigate continuous thermal build-up."
        )
        actions.append(
            "Inspect thermal management heat exchangers and check for coolant flow restrictions."
        )
        actions.append(
            "Increase telemetry sampling frequency to sub-second resolution for close observation."
        )

    elif risk_level == RiskLevel.WARNING:
        actions.append(
            "Review pack power profile and schedule an off-peak diagnostic health check."
        )
        actions.append(
            "Verify cooling fan and radiator ducting operation under current ambient load."
        )

    else:  # NORMAL
        actions.append(
            "Standard operating status confirmed. Continue routine telemetry recording."
        )

    # Priority Action 2: Factor-specific advisories
    factor_str = " ".join(factor_names)

    if "temp trend" in factor_str or "trend" in factor_str:
        actions.append(
            "Temperature ramp rate is elevated: observe heating curve and monitor for thermal runaway preconditions."
        )

    if "temperature" in factor_str and risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        actions.append(
            "Verify temperature sensor calibration against redundant pack thermocouple sensors."
        )

    if "current" in factor_str:
        actions.append(
            "High current draw detected: inspect cell busbars and cable harnesses for excessive contact resistance."
        )

    if "voltage" in factor_str:
        actions.append(
            "Cell voltage near operating boundaries: perform single-cell balancing check at next charge cycle."
        )

    if "anomaly" in factor_str:
        actions.append(
            "Beta-VAE detected abnormal multivariate trajectory: inspect recent cycling history for latent degradation."
        )

    if "decoupling" in factor_str:
        actions.append(
            "Thermal escalation without electrical load indicates potential internal short or chemical heating; inspect immediately."
        )

    # Priority Action 3: Human Escalation line
    if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        actions.append(
            "Escalate to lead battery systems engineer if parameters remain elevated beyond 3 minutes."
        )
    elif risk_level == RiskLevel.WARNING:
        actions.append(
            "Log advisory in maintenance log and notify on-duty test engineer."
        )

    # Guarantee human-in-the-loop ending
    actions.append(HUMAN_IN_THE_LOOP_DISCLAIMER)

    # Verification: Guarantee zero forbidden control commands
    for action in actions:
        action_lower = action.lower()
        for forbidden in FORBIDDEN_WORDS:
            if forbidden in action_lower:
                raise ValueError(f"Forbidden control directive detected in recommendation: '{action}'")

    return Recommendation(
        event_id=event_id,
        risk_level=risk_level,
        actions=actions,
        disclaimer=HUMAN_IN_THE_LOOP_DISCLAIMER,
    )


def recommend_actions(assessment: Any) -> List[str]:
    """Convenience helper returning ordered action strings for a given assessment."""
    if hasattr(assessment, "risk_level"):
        lvl = assessment.risk_level
        factors = getattr(assessment, "factors", [])
        overrides = getattr(assessment, "hard_overrides", [])
        rec = get_recommendations(lvl, factors, overrides)
        return rec.actions
    elif isinstance(assessment, RiskLevel):
        rec = get_recommendations(assessment, [])
        return rec.actions
    return ["Maintain telemetry monitoring.", HUMAN_IN_THE_LOOP_DISCLAIMER]

