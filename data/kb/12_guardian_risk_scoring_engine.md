# Guardian Deterministic Risk Scoring Engine Architecture

## Component Formulation and Weights
The AI Battery Safety Guardian risk assessment engine is implemented in pure deterministic Python (src/guardian/risk/engine.py). It isolates safety score computation completely from generative LLMs.
The total Risk Score (0 to 100, higher = riskier) is computed as:
RiskScore = 100 * sum(weight_i * component_i)

The 5 components are:
1. **Temperature Score (weight 0.30):** Normalized scaling of max(T_actual, T_pred) against warning (35 °C) and critical (50 °C) thresholds.
2. **Temperature Trend Score (weight 0.20):** Normalized scaling of dT/dt against critical heating rate (3.0 °C/min).
3. **Current Score (weight 0.15):** Normalized ratio of |I| against rated continuous capacity (2.0 A).
4. **Voltage Score (weight 0.10):** Distance-to-boundary score penalizing operation near or outside safe bounds [2.5 V, 4.2 V].
5. **Anomaly Score (weight 0.25):** Direct reconstruction error percentile score [0.0, 1.0] from the Beta-VAE anomaly detector.

## Risk Levels and Hard Safety Overrides
- **NORMAL (< 30.0):** Standard nominal operation.
- **WARNING (30.0 – 55.0):** Elevated telemetry parameters requiring enhanced logging.
- **HIGH (55.0 – 75.0):** Significant safety risk requiring immediate load reduction and inspection.
- **CRITICAL (>= 75.0):** Critical safety violation.

**Hard Overrides:**
- Effective Temperature >= 50 °C immediately forces level to CRITICAL.
- Effective Temperature >= 42 °C or Voltage outside [2.5, 4.2] V immediately forces level to at least HIGH.
