# Battery Management System (BMS) Architecture and Protections

## Core Functions of the BMS
The Battery Management System (BMS) serves as the primary electronic safety supervisor for lithium-ion battery packs. Its core responsibilities include:
1. **Cell Telemetry Monitoring:** High-accuracy measurement of individual cell voltages (within ±2 mV), pack current, and multiple distributed temperature sensing points.
2. **State Estimation:** Continual calculation of State of Charge (SOC), State of Health (SOH), and State of Power (SOP).
3. **Cell Balancing:** Minimizing capacity differences across series-connected cells via passive resistive bleed-off or active capacitive/inductive charge shuttling.
4. **Thermal Management Control:** Modulating cooling fans, coolant circulation pumps, or PTC heaters.

## Fault Isolation Hierarchy
When sensor readings breach safety boundaries, the BMS follows a tiered response ladder:
- **Level 1 (Advisory Warning):** Parameter approaches warning threshold (e.g. T > 38 °C); fan speed increases; derating advisory issued to motor controller.
- **Level 2 (Severe Derate):** Parameter reaches high risk (e.g. T > 45 °C or V < 2.7 V); charge/discharge current strictly derated by 50% to 80%.
- **Level 3 (Safety Cutoff):** Parameter reaches critical threshold (e.g. T >= 50 °C or V < 2.5 V); main pyrofuse or high-voltage contactors open to physically isolate the battery pack from external electrical circuits.
