"""Event detection and debouncing engine (Milestone 6).

Scans chronological risk assessments from a telemetry session, detecting:
1. LEVEL_TRANSITION: Risk classification level changes (e.g. NORMAL -> WARNING -> HIGH -> CRITICAL).
2. ANOMALY_CROSSING: VAE anomaly score crosses Warning (p95) or Critical (p99.5) thresholds.
3. TREND_BREACH: Rapid temperature rate-of-change (dT/dt >= dtdt_crit).
4. DECOUPLING: Thermal-current decoupling (temperature elevated while current is near zero).
5. HARD_OVERRIDE: Battery physical safety bounds breached.

Applies temporal debouncing (events.min_gap_s) while allowing immediate emission on severity escalation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from guardian.utils.config import get_config
from guardian.utils.schemas import RiskAssessment, RiskLevel, SafetyEvent


def detect_events(
    assessments: List[RiskAssessment],
    min_gap_s: float = 15.0,
    config: Optional[Dict[str, Any]] = None,
) -> List[SafetyEvent]:
    """Detect safety events from a time-series list of RiskAssessment objects.

    Args:
        assessments: Chronological list of RiskAssessment objects for a session.
        min_gap_s: Minimum seconds between repeated emissions of the same event type.
        config: Configuration dictionary.

    Returns:
        List of emitted SafetyEvent objects.
    """
    if not assessments:
        return []

    cfg = config or get_config()
    thresh_cfg = cfg.get("risk", {}).get("thresholds", {})
    t_warn = thresh_cfg.get("t_warn", 35.0)
    dtdt_crit = thresh_cfg.get("dtdt_crit", 3.0)

    events: List[SafetyEvent] = []
    # Tracking for debouncing: {event_type: (last_timestamp_s, last_severity)}
    last_emitted: Dict[str, tuple[float, RiskLevel]] = {}

    prev_level: RiskLevel = assessments[0].risk_level

    severity_order = {
        RiskLevel.NORMAL: 0,
        RiskLevel.WARNING: 1,
        RiskLevel.HIGH: 2,
        RiskLevel.CRITICAL: 3,
    }

    for idx, curr in enumerate(assessments):
        t_s = curr.t_s
        raw = curr.raw_telemetry
        factors_names = [f.name for f in curr.factors[:3]]

        # Helper to check debounce condition
        def should_emit(event_type: str, new_severity: RiskLevel) -> bool:
            if event_type not in last_emitted:
                return True
            last_t, last_sev = last_emitted[event_type]
            time_elapsed = t_s - last_t
            # Allow emission if debounce time elapsed OR if severity escalated
            is_escalation = severity_order[new_severity] > severity_order[last_sev]
            return (time_elapsed >= min_gap_s) or is_escalation

        # Check 1: Level Transition
        if curr.risk_level != prev_level:
            if curr.risk_level != RiskLevel.NORMAL:  # Don't emit event when returning to NORMAL
                if should_emit("LEVEL_TRANSITION", curr.risk_level):
                    event_id = f"{curr.battery_id}_c{curr.cycle}_{int(t_s)}_level_{curr.risk_level.value}"
                    events.append(
                        SafetyEvent(
                            event_id=event_id,
                            battery_id=curr.battery_id,
                            cycle=curr.cycle,
                            t_s=t_s,
                            timestamp=curr.timestamp,
                            event_type="LEVEL_TRANSITION",
                            severity=curr.risk_level,
                            contributing_factors=factors_names,
                            telemetry_values=raw,
                            description=f"Risk level transitioned from {prev_level.value} to {curr.risk_level.value} (Score: {curr.risk_score:.1f})",
                            risk_score=curr.risk_score,
                            assessment=curr,
                        )
                    )
                    last_emitted["LEVEL_TRANSITION"] = (t_s, curr.risk_level)
            prev_level = curr.risk_level

        # Check 2: Hard Overrides
        if curr.hard_overrides:
            if should_emit("HARD_OVERRIDE", curr.risk_level):
                event_id = f"{curr.battery_id}_c{curr.cycle}_{int(t_s)}_override"
                desc = "; ".join(curr.hard_overrides)
                events.append(
                    SafetyEvent(
                        event_id=event_id,
                        battery_id=curr.battery_id,
                        cycle=curr.cycle,
                        t_s=t_s,
                        timestamp=curr.timestamp,
                        event_type="HARD_OVERRIDE",
                        severity=curr.risk_level,
                        contributing_factors=factors_names,
                        telemetry_values=raw,
                        description=f"Hard safety override triggered: {desc}",
                        risk_score=curr.risk_score,
                        assessment=curr,
                    )
                )
                last_emitted["HARD_OVERRIDE"] = (t_s, curr.risk_level)

        # Check 3: Anomaly Score Crossing
        if curr.anomaly_score >= 0.95:
            sev = RiskLevel.CRITICAL if curr.anomaly_score >= 0.995 else RiskLevel.WARNING
            if should_emit("ANOMALY_CROSSING", sev):
                event_id = f"{curr.battery_id}_c{curr.cycle}_{int(t_s)}_anom"
                events.append(
                    SafetyEvent(
                        event_id=event_id,
                        battery_id=curr.battery_id,
                        cycle=curr.cycle,
                        t_s=t_s,
                        timestamp=curr.timestamp,
                        event_type="ANOMALY_CROSSING",
                        severity=sev,
                        contributing_factors=["Anomaly", "Reconstruction Error"],
                        telemetry_values=raw,
                        description=f"Beta-VAE anomaly score breached threshold: {curr.anomaly_score:.3f} >= 0.95",
                        risk_score=curr.risk_score,
                        assessment=curr,
                    )
                )
                last_emitted["ANOMALY_CROSSING"] = (t_s, sev)

        # Check 4: Temperature Trend Breach
        dT_dt = raw.get("dT_dt", 0.0)
        if dT_dt >= dtdt_crit:
            sev = RiskLevel.HIGH if dT_dt >= (dtdt_crit * 1.5) else RiskLevel.WARNING
            if should_emit("TREND_BREACH", sev):
                event_id = f"{curr.battery_id}_c{curr.cycle}_{int(t_s)}_trend"
                events.append(
                    SafetyEvent(
                        event_id=event_id,
                        battery_id=curr.battery_id,
                        cycle=curr.cycle,
                        t_s=t_s,
                        timestamp=curr.timestamp,
                        event_type="TREND_BREACH",
                        severity=sev,
                        contributing_factors=["Temp trend"],
                        telemetry_values=raw,
                        description=f"Rapid heating rate breach: dT/dt = {dT_dt:.2f} °C/min (limit: {dtdt_crit} °C/min)",
                        risk_score=curr.risk_score,
                        assessment=curr,
                    )
                )
                last_emitted["TREND_BREACH"] = (t_s, sev)

        # Check 5: Thermal-Current Decoupling
        temp_val = raw.get("temp_c", 0.0)
        curr_val = abs(raw.get("current_a", 0.0))
        if temp_val >= t_warn and curr_val < 0.10:
            sev = RiskLevel.HIGH
            if should_emit("DECOUPLING", sev):
                event_id = f"{curr.battery_id}_c{curr.cycle}_{int(t_s)}_decoupling"
                events.append(
                    SafetyEvent(
                        event_id=event_id,
                        battery_id=curr.battery_id,
                        cycle=curr.cycle,
                        t_s=t_s,
                        timestamp=curr.timestamp,
                        event_type="DECOUPLING",
                        severity=sev,
                        contributing_factors=["Temperature", "Decoupling"],
                        telemetry_values=raw,
                        description=f"Exothermic decoupling detected: T = {temp_val:.1f} °C while current draw is zero ({curr_val:.2f} A)",
                        risk_score=curr.risk_score,
                        assessment=curr,
                    )
                )
                last_emitted["DECOUPLING"] = (t_s, sev)

    return events
