"""Deterministic, explainable safety risk scoring engine (Milestone 5).

Pure function assess(state, anomaly, pred_temp, history, config) -> RiskAssessment.

Calculates weighted risk score (0-100) combining:
1. Temperature: max(T, T_pred) relative to T_warn and T_crit (weight 0.30)
2. Temp trend: dT/dt relative to dtdt_crit (weight 0.20)
3. Current: abs(I) relative to I_rated (weight 0.15)
4. Voltage: proximity to operating boundaries [V_min, V_max] (weight 0.10)
5. Anomaly: VAE reconstruction anomaly score (weight 0.25)

Enforces deterministic hard safety overrides:
- max(T, T_pred) >= T_crit -> CRITICAL
- max(T, T_pred) >= T_high or V outside [V_min, V_max] -> at least HIGH
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from guardian.utils.config import get_config
from guardian.utils.schemas import RiskAssessment, RiskFactor, RiskLevel


def calculate_temperature_component(
    temp_c: float,
    pred_temp_c: float,
    t_warn: float = 35.0,
    t_crit: float = 50.0,
) -> Tuple[float, float]:
    """Calculate temperature risk component in [0.0, 1.0].

    Returns (component_score, effective_temp).
    """
    t_eff = max(float(temp_c), float(pred_temp_c))
    if t_eff <= t_warn:
        return 0.0, t_eff
    if t_eff >= t_crit:
        return 1.0, t_eff
    score = (t_eff - t_warn) / (t_crit - t_warn)
    return float(np.clip(score, 0.0, 1.0)), t_eff


def calculate_trend_component(
    dT_dt: float,
    dtdt_crit: float = 3.0,
) -> float:
    """Calculate temperature rate-of-change risk component in [0.0, 1.0]."""
    rate = max(0.0, float(dT_dt))
    score = rate / dtdt_crit
    return float(np.clip(score, 0.0, 1.0))


def calculate_current_component(
    current_a: float,
    i_rated: float = 2.0,
) -> float:
    """Calculate current overload risk component in [0.0, 1.0]."""
    abs_i = abs(float(current_a))
    ratio = abs_i / i_rated
    # clip(abs(I)/I_rated, 0, 1.5) / 1.5
    score = np.clip(ratio, 0.0, 1.5) / 1.5
    return float(np.clip(score, 0.0, 1.0))


def calculate_voltage_component(
    voltage_v: float,
    v_min: float = 2.5,
    v_max: float = 4.2,
) -> float:
    """Calculate voltage risk component in [0.0, 1.0].

    1.0 if outside [v_min, v_max], else scaled by distance to nearest limit.
    """
    v = float(voltage_v)
    if v <= v_min or v >= v_max:
        return 1.0

    v_mid = (v_min + v_max) / 2.0
    half_width = (v_max - v_min) / 2.0
    dev = abs(v - v_mid) / half_width
    return float(np.clip(dev, 0.0, 1.0))


def calculate_anomaly_component(anomaly_score: float) -> float:
    """Bound anomaly score into [0.0, 1.0]."""
    return float(np.clip(float(anomaly_score), 0.0, 1.0))


def assess(
    state: Dict[str, float],
    anomaly_score: float,
    pred_temp: float,
    history: Optional[List[Dict[str, float]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> RiskAssessment:
    """Assess battery safety risk from state, anomaly detector, and temperature prediction.

    Args:
        state: Telemetry reading dict containing voltage_v, current_a, temp_c, etc.
        anomaly_score: VAE reconstruction anomaly score in [0.0, 1.0].
        pred_temp: Forecasted cell temperature at t + H steps.
        history: Optional historical steps to estimate dT/dt if absent in state.
        config: Configuration dictionary containing weights and thresholds.

    Returns:
        RiskAssessment dataclass instance.
    """
    cfg = config or get_config()
    risk_cfg = cfg.get("risk", {})
    weights_cfg = risk_cfg.get("weights", {})
    thresh_cfg = risk_cfg.get("thresholds", {})
    levels_cfg = risk_cfg.get("levels", {})

    w_temp = weights_cfg.get("temp", 0.30)
    w_trend = weights_cfg.get("temp_trend", 0.20)
    w_curr = weights_cfg.get("current", 0.15)
    w_volt = weights_cfg.get("voltage", 0.10)
    w_anom = weights_cfg.get("anomaly", 0.25)

    t_warn = thresh_cfg.get("t_warn", 35.0)
    t_high = thresh_cfg.get("t_high", 42.0)
    t_crit = thresh_cfg.get("t_crit", 50.0)
    dtdt_crit = thresh_cfg.get("dtdt_crit", 3.0)
    i_rated = thresh_cfg.get("i_rated", 2.0)
    v_min = thresh_cfg.get("v_min", 2.5)
    v_max = thresh_cfg.get("v_max", 4.2)

    norm_max = levels_cfg.get("normal_max", 30.0)
    warn_max = levels_cfg.get("warning_max", 55.0)
    high_max = levels_cfg.get("high_max", 75.0)

    # Extract state fields
    temp_c = float(state.get("temp_c", 25.0))
    current_a = float(state.get("current_a", 0.0))
    voltage_v = float(state.get("voltage_v", 3.7))
    battery_id = str(state.get("battery_id", "BATTERY"))
    cycle = int(state.get("cycle", 1))
    t_s = float(state.get("t_s", 0.0))
    timestamp = str(state.get("timestamp", ""))

    # Extract or calculate dT_dt
    if "dT_dt" in state:
        dT_dt = float(state["dT_dt"])
    elif history and len(history) > 1:
        prev = history[-1]
        dt = max(1.0, t_s - float(prev.get("t_s", t_s - 1.0)))
        dT_dt = (temp_c - float(prev.get("temp_c", temp_c))) / dt * 60.0
    else:
        dT_dt = 0.0

    # 1. Compute components (each in [0, 1])
    c_temp, t_eff = calculate_temperature_component(temp_c, pred_temp, t_warn, t_crit)
    c_trend = calculate_trend_component(dT_dt, dtdt_crit)
    c_curr = calculate_current_component(current_a, i_rated)
    c_volt = calculate_voltage_component(voltage_v, v_min, v_max)
    c_anom = calculate_anomaly_component(anomaly_score)

    factors = [
        RiskFactor("Temperature", t_eff, c_temp, w_temp, w_temp * c_temp * 100.0),
        RiskFactor("Temp trend", dT_dt, c_trend, w_trend, w_trend * c_trend * 100.0),
        RiskFactor("Current", abs(current_a), c_curr, w_curr, w_curr * c_curr * 100.0),
        RiskFactor("Voltage", voltage_v, c_volt, w_volt, w_volt * c_volt * 100.0),
        RiskFactor("Anomaly", anomaly_score, c_anom, w_anom, w_anom * c_anom * 100.0),
    ]

    # Weighted base score (0 to 100)
    base_score = sum(f.weighted_score for f in factors)
    base_score = float(np.clip(base_score, 0.0, 100.0))

    # Base level classification
    if base_score < norm_max:
        level = RiskLevel.NORMAL
    elif base_score < warn_max:
        level = RiskLevel.WARNING
    elif base_score < high_max:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.CRITICAL

    # 2. Hard overrides
    hard_overrides: List[str] = []

    # Override 1: T_eff >= T_crit -> CRITICAL
    if t_eff >= t_crit:
        level = RiskLevel.CRITICAL
        base_score = max(base_score, 75.0)
        hard_overrides.append(f"Temperature {t_eff:.1f}°C breached critical limit (>= {t_crit}°C)")

    # Override 2: T_eff >= T_high -> At least HIGH
    if t_eff >= t_high and level in (RiskLevel.NORMAL, RiskLevel.WARNING):
        level = RiskLevel.HIGH
        base_score = max(base_score, 55.0)
        hard_overrides.append(f"Temperature {t_eff:.1f}°C breached high limit (>= {t_high}°C)")

    # Override 3: Voltage outside operating bounds -> At least HIGH
    if (voltage_v < v_min or voltage_v > v_max) and level in (RiskLevel.NORMAL, RiskLevel.WARNING):
        level = RiskLevel.HIGH
        base_score = max(base_score, 55.0)
        hard_overrides.append(f"Voltage {voltage_v:.2f}V outside operating bounds [{v_min}V, {v_max}V]")

    # Sort contributing factors descending by weighted contribution
    sorted_factors = sorted(factors, key=lambda f: f.weighted_score, reverse=True)

    return RiskAssessment(
        battery_id=battery_id,
        cycle=cycle,
        t_s=t_s,
        timestamp=timestamp,
        risk_score=round(base_score, 2),
        risk_level=level,
        anomaly_score=round(float(anomaly_score), 4),
        predicted_temp_c=round(float(pred_temp), 2),
        factors=sorted_factors,
        hard_overrides=hard_overrides,
        raw_telemetry={
            "temp_c": temp_c,
            "pred_temp_c": pred_temp,
            "dT_dt": dT_dt,
            "current_a": current_a,
            "voltage_v": voltage_v,
        },
    )
