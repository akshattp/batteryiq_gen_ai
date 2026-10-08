# Charging Safety Protocols and Fast Charging Considerations

## Constant Current - Constant Voltage (CC-CV) Protocol
Standard lithium-ion charging follows the CC-CV profile:
1. **Constant Current (CC) Phase:** Battery is charged with a controlled continuous current until the cell terminal voltage reaches the target ceiling (typically 4.20 V). During this phase, State of Charge typically rises from empty to approximately 75%–80%.
2. **Constant Voltage (CV) Phase:** Voltage is held strictly constant at 4.20 V while charging current decays exponentially toward a cutoff threshold (usually C/20 to C/50).

## DC Fast Charging (DCFC) Thermal Hazards
High-rate fast charging (>= 1.5C) introduces severe challenges:
- High Joule heating concentrated in early charging phases creates steep thermal gradients between cell cores and outer casing.
- Anode polarization can push negative electrode potential below 0 V vs. Li/Li+, leading to metallic lithium plating even at moderate ambient temperatures.
- Uneven temperature distribution accelerates localized cell aging, leading to early cell-to-cell capacity divergence within multi-cell packs.
