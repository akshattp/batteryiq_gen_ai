"""tests/test_events.py - Event detection and debouncing test suite (Milestone 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guardian.data.faults import inject_thermal_ramp
from guardian.risk.engine import assess
from guardian.risk.events import detect_events
from guardian.utils.schemas import RiskAssessment, RiskLevel, SafetyEvent


def test_replayed_fault_session_emits_events():
    """Acceptance criterion: A replayed injected-fault session must produce >= 1 event."""
    n_steps = 40
    t_vals = np.linspace(0, 200, n_steps)  # 5-second steps

    # Base normal dataframe
    normal_df = pd.DataFrame({
        "battery_id": ["B0018"] * n_steps,
        "cycle": [10] * n_steps,
        "mode": ["discharge"] * n_steps,
        "t_s": t_vals,
        "timestamp": [f"2026-10-09T00:00:{int(t):02d}" for t in t_vals],
        "voltage_v": np.linspace(4.0, 3.7, n_steps),
        "current_a": np.full(n_steps, -2.0),
        "abs_current_a": np.full(n_steps, 2.0),
        "temp_c": np.linspace(25.0, 28.0, n_steps),
        "ambient_c": np.full(n_steps, 24.0),
        "soc_pct": np.linspace(100.0, 80.0, n_steps),
        "dT_dt": np.full(n_steps, 0.5),
        "label": ["normal"] * n_steps,
        "fault_type": [None] * n_steps,
    })

    # Inject severe thermal ramp into second half of session
    faulty_df = normal_df.copy()
    faulty_df.iloc[20:] = inject_thermal_ramp(normal_df.iloc[20:], severity=2.0, ramp_rate_c_per_min=8.0)

    # Generate chronological risk assessments
    assessments = []
    for idx, row in faulty_df.iterrows():
        anom = 0.05 if row["label"] == "normal" else 0.98
        pred_t = row["temp_c"] + 3.0
        ass = assess(
            state=row.to_dict(),
            anomaly_score=anom,
            pred_temp=pred_t,
        )
        assessments.append(ass)

    # Detect events with 15s debounce
    events = detect_events(assessments, min_gap_s=15.0)

    # Assert >= 1 event emitted
    assert len(events) >= 1, "Replayed fault session failed to emit any safety events!"
    # Verify event types include LEVEL_TRANSITION or TREND_BREACH
    event_types = [e.event_type for e in events]
    assert any(et in ("LEVEL_TRANSITION", "TREND_BREACH", "ANOMALY_CROSSING") for et in event_types)


def test_event_debouncing():
    """Verify that identical repeated events within min_gap_s are debounced."""
    # Create 5 assessments at 2-second intervals with identical high anomaly
    assessments = []
    for step in range(5):
        t_s = float(step * 2)  # gap = 2s, smaller than min_gap_s = 15s
        state = {
            "battery_id": "B0005",
            "cycle": 1,
            "t_s": t_s,
            "timestamp": f"2026-10-09T00:00:{int(t_s):02d}",
            "temp_c": 25.0,
            "voltage_v": 3.8,
            "current_a": -1.0,
            "dT_dt": 0.1,
        }
        ass = assess(state=state, anomaly_score=0.96, pred_temp=25.0)
        assessments.append(ass)

    events = detect_events(assessments, min_gap_s=15.0)
    anom_events = [e for e in events if e.event_type == "ANOMALY_CROSSING"]
    # Should only emit 1 event due to debouncing
    assert len(anom_events) == 1, f"Expected 1 debounced anomaly event, got {len(anom_events)}"


def test_severity_escalation_bypasses_debounce():
    """Verify that rapid escalation in severity emits immediately despite debounce gap."""
    # Step 1: Warning anomaly (0.955) at t=0
    ass1 = assess(
        state={"battery_id": "B0005", "cycle": 1, "t_s": 0.0, "temp_c": 25.0, "current_a": -1.0, "voltage_v": 3.7},
        anomaly_score=0.96,
        pred_temp=25.0,
    )
    # Step 2: Critical anomaly (0.998) at t=3s (within 15s debounce)
    ass2 = assess(
        state={"battery_id": "B0005", "cycle": 1, "t_s": 3.0, "temp_c": 51.0, "current_a": -1.0, "voltage_v": 3.7},
        anomaly_score=0.999,
        pred_temp=52.0,
    )

    events = detect_events([ass1, ass2], min_gap_s=15.0)
    # Should emit on escalation to CRITICAL
    assert len(events) >= 2
    assert any(e.severity == RiskLevel.CRITICAL for e in events)
