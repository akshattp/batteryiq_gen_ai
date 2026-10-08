# Telemetry Sensor Fault Detection and Plausibility Checking

## Common Sensor Failure Modes
In battery monitoring systems, sensor instrumentation can produce erroneous readings that mimic or mask safety hazards:
1. **Stuck / Flat-lined Sensor:** A thermistor or ADC channel outputs a completely constant reading (e.g. 24.0 °C or 0.0 V) across dozens of operating minutes regardless of load variation.
2. **Erratic Noise / Spikes:** Damaged sensor wiring harnesses or EMI from switching inverters inject rapid unphysical transients (e.g. instantaneous jumps of 30 °C in 1 second).
3. **Sensor Decoupling:** Temperature readings drift upward while current is zero, or voltage remains pinned while current transitions from charge to discharge.

## Multi-Channel Plausibility Checks
- **Thermal-Electric Coupling:** In an operating cell, temperature rate of change dT/dt must correlate with |I|^2 and cooling airflow. Uncorrelated heat indicates either a sensor anomaly or an internal short.
- **Redundant Sensing Validation:** Parallel sensors across adjacent modules are compared to confirm physical spatial coherence.
