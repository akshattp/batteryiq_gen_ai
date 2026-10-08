# Guardian Event Detection, Debouncing, and Advisory Catalog

## Event Engine Specifications
The Guardian safety event monitor (src/guardian/risk/events.py) scans continuous telemetry assessments and emits discrete events:
1. **Level Transition Events:** When risk shifts across levels (e.g. NORMAL -> WARNING or WARNING -> HIGH).
2. **Anomaly Crossing Events:** When VAE anomaly score breaches the 95th or 99.5th percentile thresholds.
3. **Thermal Trend Breaches:** When heating rate dT/dt exceeds critical threshold (3.0 °C/min).
4. **Thermal Decoupling:** When temperature rises while current is near zero.
5. **Hard Overrides:** When temperature >= 50 °C or voltage exceeds [2.5 V, 4.2 V].

## Debounce Filtering with Escalation Bypass
To avoid alert flooding during persistent anomalies, events are debounced with a 15-second minimum gap (min_gap_s = 15.0).
However, if severity escalates (e.g. from WARNING to HIGH or CRITICAL), the debounce timer is bypassed immediately to ensure zero latency on safety-critical deteriorations.

## Advisory Rules and Zero-Control Mandate
Recommendations (src/guardian/risk/recommend.py) provide non-autonomous, human-in-the-loop engineering advisories (e.g. reduce load, verify thermal management, check busbar connections). All recommendations conclude with the mandatory disclaimer:
'Decision-support prototype. Not for vehicle control. Escalate to human engineer.'
