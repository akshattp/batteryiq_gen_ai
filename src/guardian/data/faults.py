"""Synthetic fault injection suite for safety evaluation (Section 5.6).

Contains functions to inject controlled abnormalities into normal telemetry:
1. thermal_ramp: abnormal temperature ramp rate
2. current_spike: short high-current burst
3. voltage_sag: abnormal voltage drop under load
4. sensor_stuck: temperature sensor flat-lines
5. temp_current_decoupling: temperature rises while current is near zero

Injected data is marked label='injected_fault' and fault_type='<name>'.
CRITICAL: Synthetic fault data is used ONLY for evaluation and stress-testing;
it is NEVER used to train the VAE normal baseline.
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


FAULT_TYPES: List[str] = [
    "thermal_ramp",
    "current_spike",
    "voltage_sag",
    "sensor_stuck",
    "temp_current_decoupling",
]


def inject_thermal_ramp(df: pd.DataFrame, severity: float = 1.0, ramp_rate_c_per_min: float = 6.0) -> pd.DataFrame:
    """Inject abnormal temperature rise rate into telemetry.

    Args:
        df: Input normal DataFrame.
        severity: Fault severity multiplier (default 1.0).
        ramp_rate_c_per_min: Ramp rate in °C per minute.

    Returns:
        Modified DataFrame with elevated temperature and dT/dt.
    """
    faulty = df.copy()
    t_vals = faulty["t_s"].values
    t_start = t_vals[0] if len(t_vals) > 0 else 0.0

    # Additional temperature ramp: delta_T = severity * ramp_rate * (t - t0)/60
    t_delta_min = (t_vals - t_start) / 60.0
    temp_boost = severity * ramp_rate_c_per_min * t_delta_min

    faulty["temp_c"] = np.round(faulty["temp_c"] + temp_boost, 4)

    if "temp_minus_ambient" in faulty.columns:
        faulty["temp_minus_ambient"] = faulty["temp_c"] - faulty["ambient_c"]
    if "abs_current_a" in faulty.columns and "temp_x_abs_current" in faulty.columns:
        faulty["temp_x_abs_current"] = faulty["temp_c"] * faulty["abs_current_a"]
    if "dT_dt" in faulty.columns:
        faulty["dT_dt"] = np.round(faulty["dT_dt"] + (severity * ramp_rate_c_per_min), 4)

    faulty["label"] = "injected_fault"
    faulty["fault_type"] = "thermal_ramp"
    return faulty


def inject_current_spike(df: pd.DataFrame, severity: float = 1.0, spike_current_a: float = 12.0) -> pd.DataFrame:
    """Inject short high-current burst into telemetry.

    Args:
        df: Input normal DataFrame.
        severity: Fault severity multiplier.
        spike_current_a: Magnitude of added current in Amperes.

    Returns:
        Modified DataFrame with spiked current draw.
    """
    faulty = df.copy()
    n = len(faulty)
    if n == 0:
        return faulty

    # Spike occurs across middle portion or all of window
    spike_magnitude = severity * spike_current_a

    # Maintain mode sign convention
    is_discharge = (faulty["current_a"] < 0) | (faulty["mode"] == "discharge")
    curr_mod = np.where(is_discharge, faulty["current_a"] - spike_magnitude, faulty["current_a"] + spike_magnitude)

    faulty["current_a"] = np.round(curr_mod, 4)
    faulty["abs_current_a"] = np.abs(faulty["current_a"])

    if "temp_x_abs_current" in faulty.columns:
        faulty["temp_x_abs_current"] = faulty["temp_c"] * faulty["abs_current_a"]

    faulty["label"] = "injected_fault"
    faulty["fault_type"] = "current_spike"
    return faulty


def inject_voltage_sag(df: pd.DataFrame, severity: float = 1.0, sag_volts: float = 0.75) -> pd.DataFrame:
    """Inject abnormal voltage drop under load.

    Args:
        df: Input normal DataFrame.
        severity: Fault severity multiplier.
        sag_volts: Voltage decrease in Volts.

    Returns:
        Modified DataFrame with voltage sag.
    """
    faulty = df.copy()
    drop = severity * sag_volts
    faulty["voltage_v"] = np.round(np.maximum(1.8, faulty["voltage_v"] - drop), 4)

    if "dV_dt" in faulty.columns:
        # Initial drop creates a negative step
        faulty["dV_dt"] = np.round(faulty["dV_dt"] - (drop / 2.0), 4)

    faulty["label"] = "injected_fault"
    faulty["fault_type"] = "voltage_sag"
    return faulty


def inject_sensor_stuck(df: pd.DataFrame, severity: float = 1.0) -> pd.DataFrame:
    """Inject temperature sensor flat-line fault.

    Args:
        df: Input normal DataFrame.
        severity: Unused for stuck fault, retained for uniform signature.

    Returns:
        Modified DataFrame with constant flat-lined temperature and dT/dt = 0.
    """
    faulty = df.copy()
    if len(faulty) > 0:
        stuck_val = float(faulty["temp_c"].iloc[0])
        faulty["temp_c"] = stuck_val

        if "temp_minus_ambient" in faulty.columns:
            faulty["temp_minus_ambient"] = stuck_val - faulty["ambient_c"]
        if "abs_current_a" in faulty.columns and "temp_x_abs_current" in faulty.columns:
            faulty["temp_x_abs_current"] = stuck_val * faulty["abs_current_a"]
        if "dT_dt" in faulty.columns:
            faulty["dT_dt"] = 0.0

    faulty["label"] = "injected_fault"
    faulty["fault_type"] = "sensor_stuck"
    return faulty


def inject_temp_current_decoupling(
    df: pd.DataFrame, severity: float = 1.0, temp_boost_c: float = 10.0
) -> pd.DataFrame:
    """Inject thermal rise decoupled from near-zero current.

    Simulates internal cell self-heating or exothermic reaction where
    temperature escalates despite minimal external electrical load.

    Args:
        df: Input normal DataFrame.
        severity: Fault severity multiplier.
        temp_boost_c: Exothermic heating boost in °C.

    Returns:
        Modified DataFrame with high temperature and zero current.
    """
    faulty = df.copy()
    t_vals = faulty["t_s"].values
    t_start = t_vals[0] if len(t_vals) > 0 else 0.0
    t_end = t_vals[-1] if len(t_vals) > 1 else t_start + 1.0
    duration = max(1.0, t_end - t_start)

    progress = (t_vals - t_start) / duration
    thermal_rise = severity * temp_boost_c * progress

    faulty["temp_c"] = np.round(faulty["temp_c"] + thermal_rise, 4)
    # Current forced to ~0 (quiescent)
    faulty["current_a"] = 0.0
    faulty["abs_current_a"] = 0.0

    if "temp_minus_ambient" in faulty.columns:
        faulty["temp_minus_ambient"] = faulty["temp_c"] - faulty["ambient_c"]
    if "temp_x_abs_current" in faulty.columns:
        faulty["temp_x_abs_current"] = 0.0
    if "dT_dt" in faulty.columns:
        faulty["dT_dt"] = np.round(faulty["dT_dt"] + (severity * temp_boost_c * 60.0 / duration), 4)

    faulty["label"] = "injected_fault"
    faulty["fault_type"] = "temp_current_decoupling"
    return faulty


def inject_fault(df: pd.DataFrame, fault_type: str, severity: float = 1.0) -> pd.DataFrame:
    """Dispatch fault injection by type name."""
    dispatch = {
        "thermal_ramp": inject_thermal_ramp,
        "current_spike": inject_current_spike,
        "voltage_sag": inject_voltage_sag,
        "sensor_stuck": inject_sensor_stuck,
        "temp_current_decoupling": inject_temp_current_decoupling,
    }
    if fault_type not in dispatch:
        raise ValueError(f"Unknown fault type: '{fault_type}'. Available: {list(dispatch.keys())}")
    return dispatch[fault_type](df, severity=severity)
