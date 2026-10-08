# Safe Operating Windows for Voltage and Temperature

## Voltage Boundaries
Lithium-ion cells operating with graphite anodes and transition metal oxide cathodes (such as 18650 NMC or LCO cells) have strictly defined safe voltage limits:
- **Upper Cutoff Voltage (4.20 V – 4.25 V):** Exceeding 4.25 V causes electrolyte oxidation, transition metal ion dissolution from the cathode, and rapid gassing.
- **Lower Cutoff Voltage (2.50 V – 2.80 V):** Discharging below 2.50 V triggers copper dissolution from the anode current collector foil into the electrolyte, creating conductive copper dendrites upon subsequent recharge that cause severe internal shorts.

## Temperature Boundaries
- **Nominal Operating Window (15 °C to 35 °C):** Optimal balance between ionic conductivity, calendar life, and low degradation.
- **Upper Warning Threshold (42.0 °C):** Accelerated SEI growth, electrolyte decomposition, and premature capacity fade. Active cooling must operate at maximum capability.
- **Upper Critical Safety Threshold (50.0 °C):** Immediate risk of localized thermal instability. Cell load must be sharply reduced and isolated.
- **Low-Temperature Plating Boundary (< 0 °C):** Charging at or below freezing severely restricts lithium ion solid diffusion into the graphite lattice, forcing metallic lithium to plate onto the anode surface, creating tree-like dendrites.
