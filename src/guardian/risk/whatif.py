"""What-if simulation and out-of-distribution guard (Milestone 14).

Simulates alternative operating parameters (higher current, increased ambient temp)
and alerts when query inputs fall outside the training distribution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from guardian.risk.engine import assess
from guardian.utils.config import get_config
from guardian.utils.schemas import RiskAssessment

# Nominal training bounds derived from NASA PCoE battery dataset
TRAINING_BOUNDS = {
    "current_a": (0.0, 3.0),       # Rated 2.0A, transients up to 3.0A
    "temp_c": (18.0, 45.0),        # 18°C to 45°C
    "ambient_c": (18.0, 35.0),     # Ambient operating envelope
    "voltage_v": (2.3, 4.3),       # Normal 2.5V - 4.2V operating range
    "soc_pct": (0.0, 100.0),
}


def is_out_of_distribution(params: Dict[str, float]) -> bool:
    """Check if any operating parameter lies outside training feature envelope."""
    for key, (low, high) in TRAINING_BOUNDS.items():
        if key in params:
            val = float(params[key])
            if val < low or val > high:
                return True
    return False


def simulate_whatif(
    current_state: Dict[str, float],
    modified_params: Dict[str, float],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[RiskAssessment, bool]:
    """Run what-if analysis on modified operating conditions.

    Args:
        current_state: Baseline telemetry state.
        modified_params: User slider adjustments (e.g. current_a, temp_c, ambient_c, soc_pct).
        config: Optional project configuration.

    Returns:
        Tuple of (new_assessment, is_out_of_distribution).
    """
    cfg = config or get_config()

    sim_state = dict(current_state)
    sim_state.update(modified_params)

    # Monotonic thermal physics modeling:
    # Joule heating dT ~ I^2 * R + convective dissipation to ambient
    current = float(sim_state.get("current_a", 2.0))
    temp = float(sim_state.get("temp_c", 25.0))
    ambient = float(sim_state.get("ambient_c", 24.0))

    # Calculate simulated rate of temperature change (dT/dt)
    # Higher current monotonically increases heating rate
    base_dtdt = max(0.01, (current / 2.0) ** 1.8 * 0.15)
    ambient_offset = max(0.0, (ambient - 24.0) * 0.02)
    sim_dtdt = base_dtdt + ambient_offset
    sim_state["dT_dt"] = sim_dtdt

    # Forward horizon predicted temperature (t + 10 steps)
    pred_temp = temp + sim_dtdt * 10.0

    # Synthetic anomaly scaling if conditions diverge
    anomaly = float(np.clip(
        0.1 + max(0.0, (temp - 35.0) / 20.0) + max(0.0, (current - 2.0) * 0.25),
        0.05,
        0.98,
    ))

    assessment = assess(
        state=sim_state,
        anomaly_score=anomaly,
        pred_temp=pred_temp,
        config=cfg,
    )

    ood = is_out_of_distribution(sim_state)
    return assessment, ood
