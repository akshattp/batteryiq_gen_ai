"""tests/test_faults.py - Unit tests for synthetic fault injection suite (Milestone 2).

Verifies that each injected fault modifies the signal in the expected physical direction
and sets appropriate anomaly metadata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guardian.data.faults import (
    FAULT_TYPES,
    inject_thermal_ramp,
    inject_current_spike,
    inject_voltage_sag,
    inject_sensor_stuck,
    inject_temp_current_decoupling,
    inject_fault,
)


@pytest.fixture
def baseline_window() -> pd.DataFrame:
    """Construct a synthetic normal 30-step telemetry window."""
    n_steps = 30
    t_s = np.linspace(0, 150, n_steps)
    return pd.DataFrame({
        "battery_id": ["B0005"] * n_steps,
        "cycle": [10] * n_steps,
        "mode": ["discharge"] * n_steps,
        "t_s": t_s,
        "timestamp": [f"2026-10-08T00:00:{int(t):02d}" for t in t_s],
        "voltage_v": np.linspace(4.1, 3.7, n_steps),
        "current_a": np.full(n_steps, -2.0),
        "abs_current_a": np.full(n_steps, 2.0),
        "temp_c": np.linspace(25.0, 28.0, n_steps),
        "ambient_c": np.full(n_steps, 24.0),
        "soc_pct": np.linspace(100.0, 80.0, n_steps),
        "dT_dt": np.full(n_steps, 1.2),
        "dV_dt": np.full(n_steps, -0.002),
        "temp_minus_ambient": np.linspace(1.0, 4.0, n_steps),
        "temp_x_abs_current": np.linspace(50.0, 56.0, n_steps),
        "label": ["normal"] * n_steps,
        "fault_type": [None] * n_steps,
    })


def test_thermal_ramp_direction(baseline_window: pd.DataFrame):
    """Verify thermal ramp increases temperature and dT/dt."""
    faulty = inject_thermal_ramp(baseline_window, severity=1.5, ramp_rate_c_per_min=5.0)

    assert faulty["label"].iloc[-1] == "injected_fault"
    assert faulty["fault_type"].iloc[-1] == "thermal_ramp"
    # Temperature at end of window must be higher than baseline
    assert faulty["temp_c"].iloc[-1] > baseline_window["temp_c"].iloc[-1]
    # dT/dt must increase
    assert faulty["dT_dt"].iloc[-1] > baseline_window["dT_dt"].iloc[-1]


def test_current_spike_direction(baseline_window: pd.DataFrame):
    """Verify current spike elevates current draw."""
    faulty = inject_current_spike(baseline_window, severity=1.0, spike_current_a=10.0)

    assert faulty["label"].iloc[-1] == "injected_fault"
    assert faulty["fault_type"].iloc[-1] == "current_spike"
    # Absolute current must be substantially higher
    assert faulty["abs_current_a"].max() > baseline_window["abs_current_a"].max()
    assert faulty["abs_current_a"].max() >= 10.0


def test_voltage_sag_direction(baseline_window: pd.DataFrame):
    """Verify voltage sag reduces voltage."""
    faulty = inject_voltage_sag(baseline_window, severity=1.0, sag_volts=0.8)

    assert faulty["label"].iloc[-1] == "injected_fault"
    assert faulty["fault_type"].iloc[-1] == "voltage_sag"
    # Minimum voltage must drop below baseline minimum
    assert faulty["voltage_v"].min() < baseline_window["voltage_v"].min()
    assert np.all(faulty["voltage_v"] <= baseline_window["voltage_v"])


def test_sensor_stuck_direction(baseline_window: pd.DataFrame):
    """Verify stuck sensor leads to constant zero temperature variance."""
    faulty = inject_sensor_stuck(baseline_window)

    assert faulty["label"].iloc[-1] == "injected_fault"
    assert faulty["fault_type"].iloc[-1] == "sensor_stuck"
    # Temperature variance across window must be zero
    assert faulty["temp_c"].std() == 0.0
    # dT/dt becomes 0
    assert (faulty["dT_dt"] == 0.0).all()


def test_temp_current_decoupling_direction(baseline_window: pd.DataFrame):
    """Verify decoupling elevates temperature while zeroing current."""
    faulty = inject_temp_current_decoupling(baseline_window, severity=1.0, temp_boost_c=8.0)

    assert faulty["label"].iloc[-1] == "injected_fault"
    assert faulty["fault_type"].iloc[-1] == "temp_current_decoupling"
    # Temperature rises
    assert faulty["temp_c"].iloc[-1] > baseline_window["temp_c"].iloc[-1]
    # Current is decoupled to 0
    assert (faulty["current_a"] == 0.0).all()
    assert (faulty["abs_current_a"] == 0.0).all()


def test_dispatcher_handles_all_fault_types(baseline_window: pd.DataFrame):
    """Verify inject_fault dispatcher works for all registered fault types."""
    for f_type in FAULT_TYPES:
        res = inject_fault(baseline_window, fault_type=f_type, severity=1.0)
        assert res["label"].iloc[0] == "injected_fault"
        assert res["fault_type"].iloc[0] == f_type
