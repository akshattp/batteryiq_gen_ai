# State of Charge (SOC) Estimation and Coulomb Counting

## Coulomb Counting Method
State of Charge represents the available charge in a cell relative to its rated capacity. The primary time-domain calculation method is Coulomb counting (current integration):
SOC(t) = SOC(t0) - (1 / Q_usable) * integral_{t0}^t [I(tau) * dtau]
where I(tau) is discharge current (conventionally signed negative for discharge, positive for charge) and Q_usable is the cycle-specific battery capacity in Ampere-hours.

## Sources of Coulomb Counting Drift
While simple and computationally efficient, raw Coulomb counting suffers from cumulative errors:
- **Current Sensor Bias:** Small DC offsets in Hall-effect or shunt sensors integrate monotonically into large drift over prolonged operating sessions.
- **Integration Inaccuracy:** Coarse sampling intervals miss rapid transient current spikes.
- **Capacity Degradation:** Usable capacity declines as the battery ages (SOH reduction), requiring dynamic parameter updating.

## Recalibration Strategies
Drift is corrected using Open-Circuit Voltage (OCV) lookup tables when the battery rests in an unexcited equilibrium state for > 30 minutes, or by resetting to 100% when reaching CC-CV charge cutoff.
