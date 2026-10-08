"""scripts/generate_incidents.py - Generate synthetic technician incident notes (Milestone M8 / Exp 5).

Generates 750 diverse synthetic technician/driver incident notes across 5 categories:
- thermal_overheat
- overcurrent
- overvoltage_or_undervoltage
- sensor_fault
- normal_observation

Saves output to data/incidents/incident_reports.csv with is_synthetic=True flag.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from guardian.utils.logging import get_logger
from guardian.utils.seed import set_seed

logger = get_logger(__name__)

CATEGORIES: List[str] = [
    "thermal_overheat",
    "overcurrent",
    "overvoltage_or_undervoltage",
    "sensor_fault",
    "normal_observation",
]

BATTERIES: List[str] = ["B0005", "B0006", "B0007", "B0018"]

# Combinatorial template parts per category
TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "thermal_overheat": {
        "intros": [
            "Thermal alarm activated during test run.",
            "Technician reported excessive heat dissipation.",
            "High temperature fault logged in battery telemetry.",
            "Vehicle thermal management flagged abnormal heating.",
            "Emergency thermal inspection conducted on pack.",
        ],
        "details": [
            "Cell skin temperature climbed rapidly to {temp:.1f} °C under standard discharge.",
            "Module 2 thermistor recorded {temp:.1f} °C, triggering critical thermal warning.",
            "Surface temperature reached {temp:.1f} °C; cooling fan was operating at 100% capacity.",
            "Sustained thermal rise observed with dT/dt exceeding {rate:.1f} °C/min.",
            "Busbar thermal camera inspection revealed hot spots peaking at {temp:.1f} °C.",
        ],
        "conclusions": [
            "Thermal runaway mitigation procedure was initiated.",
            "Pack allowed to cool down to 25 °C before resuming diagnostics.",
            "Inspect auxiliary coolant loop and heat exchanger airflow.",
            "Immediate thermal inspection recommended prior to next charge.",
        ],
    },
    "overcurrent": {
        "intros": [
            "Current transducer logged abnormal surge.",
            "Overcurrent protection triggered on discharge line.",
            "Driver reported momentary power pulse during acceleration.",
            "Inverter telemetry indicates brief current breach.",
            "Bench test telemetry recorded peak current event.",
        ],
        "details": [
            "Peak current reached {curr:.1f} A against rated continuous limit of 2.0 A.",
            "Current spiked to {curr:.1f} A during sudden regenerative braking pulse.",
            "Current amplitude exceeded maximum threshold with measured value of {curr:.1f} A.",
            "Transient overcurrent of {curr:.1f} A recorded across main contactor.",
            "Discharge rate surged to {curr:.1f} A for 4 consecutive seconds.",
        ],
        "conclusions": [
            "Contactor wear inspection requested.",
            "Check current sensor calibration and shunt resistor connections.",
            "Recommend load limitation to prevent cell damage.",
            "Inverter control firmware logs flagged for review.",
        ],
    },
    "overvoltage_or_undervoltage": {
        "intros": [
            "Cell voltage bounds violation logged by BMS.",
            "Abnormal voltage behavior observed during cycling.",
            "Low voltage cutoff event recorded on test bench.",
            "Overvoltage alarm triggered during CC-CV charging phase.",
            "Severe voltage sag detected under moderate load.",
        ],
        "details": [
            "Terminal voltage dropped precipitously to {volt:.2f} V under 2 A load.",
            "Cell potential sagged to {volt:.2f} V, below minimum safe operating limit.",
            "Voltage spiked to {volt_high:.2f} V during top balancing, exceeding cutoff limit.",
            "High internal impedance caused terminal voltage to drop to {volt:.2f} V.",
            "Voltage collapsed to {volt:.2f} V at 35% estimated state of charge.",
        ],
        "conclusions": [
            "Check cell balancing circuitry and wiring harness.",
            "Inspect cell internal resistance and state of health degradation.",
            "Recalibrate BMS voltage sensing taps.",
            "Perform low-rate capacity verification cycle.",
        ],
    },
    "sensor_fault": {
        "intros": [
            "Telemetry anomaly flagged by validation monitor.",
            "Sensor telemetry mismatch observed between channels.",
            "Intermittent sensor disconnect logged on CAN bus.",
            "Diagnostic check revealed questionable sensor readings.",
            "Technician suspected instrumentation failure on cell rig.",
        ],
        "details": [
            "Temperature reading flat-lined at {sensor_temp:.1f} °C for 45 minutes during cycling.",
            "Thermistor channel reported erratic oscillations between 12 °C and 78 °C.",
            "Zero dT/dt reported while discharge current was actively flowing.",
            "Sensor signal dropped to ground potential intermittently.",
            "NTC thermistor resistance reading stuck out of plausible physical range.",
        ],
        "conclusions": [
            "Replace faulty thermistor harness and verify connector pin seating.",
            "Sensor replaced; bench recalibration confirmed normal operation.",
            "Check wiring insulation for crimp defects or chafing.",
            "Recalibrate analog-to-digital converter on telemetry acquisition board.",
        ],
    },
    "normal_observation": {
        "intros": [
            "Routine scheduled maintenance inspection completed.",
            "Standard baseline cycling log review conducted.",
            "Post-charge quality control check carried out.",
            "Periodic battery health verification completed.",
            "Nominal operational cycle documented by technician.",
        ],
        "details": [
            "All cell voltages closely balanced within 12 mV across pack.",
            "Operating temperature steady at {norm_temp:.1f} °C throughout entire charge cycle.",
            "Measured capacity matches expected nominal degradation curve at {soh:.1f}% SOH.",
            "Current profile remained within nominal operating window with clean transitions.",
            "Visual inspection of enclosure showed zero swelling, leaks, or busbar discoloration.",
        ],
        "conclusions": [
            "Battery cleared for continued service.",
            "No safety maintenance actions required.",
            "System performing as designed in nominal state.",
            "Next scheduled inspection in 50 operating cycles.",
        ],
    },
}


def generate_incident_report(category: str, battery_id: str, cycle: int, seed: int) -> str:
    """Compose a synthetic technician note from combinatorial building blocks."""
    rng = random.Random(seed)
    tmpl = TEMPLATES[category]

    intro = rng.choice(tmpl["intros"])
    detail_tmpl = rng.choice(tmpl["details"])
    conclusion = rng.choice(tmpl["conclusions"])

    # Format values
    params = {
        "temp": rng.uniform(48.5, 62.0),
        "rate": rng.uniform(3.5, 8.0),
        "curr": rng.uniform(22.0, 48.0),
        "volt": rng.uniform(1.85, 2.45),
        "volt_high": rng.uniform(4.28, 4.45),
        "sensor_temp": rng.choice([20.0, 25.0, 0.0, -5.0]),
        "norm_temp": rng.uniform(23.0, 27.5),
        "soh": rng.uniform(82.0, 96.0),
    }

    detail = detail_tmpl.format(**params)
    note = f"{intro} {detail} {conclusion}"

    # Occasional realistic typing variation / noise
    if rng.random() < 0.15:
        note = note.replace("temperature", "temp.")
    if rng.random() < 0.10:
        note = note.replace("voltage", "volt.")

    return note


def generate_dataset(n_per_category: int = 150, out_path: str = "data/incidents/incident_reports.csv") -> pd.DataFrame:
    """Generate balanced dataset across all 5 incident categories."""
    set_seed(42)
    records: List[Dict[str, Any]] = []
    report_id = 1

    for cat in CATEGORIES:
        for i in range(n_per_category):
            seed = report_id * 101 + i
            rng = random.Random(seed)
            bat = rng.choice(BATTERIES)
            cycle = rng.randint(5, 160)
            note = generate_incident_report(cat, bat, cycle, seed)

            severity = "HIGH" if cat in ("thermal_overheat", "overcurrent") else (
                "WARNING" if cat in ("overvoltage_or_undervoltage", "sensor_fault") else "NORMAL"
            )

            records.append({
                "incident_id": f"INC-{report_id:04d}",
                "timestamp": f"2026-09-{10 + (report_id % 20):02d}T{10 + (report_id % 8):02d}:{report_id % 60:02d}:00Z",
                "battery_id": bat,
                "cycle": cycle,
                "category": cat,
                "note": note,
                "text": note,
                "label": cat,
                "severity": severity,
                "is_synthetic": True,
            })
            report_id += 1

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    # Also save standard incidents.csv for colab handoff
    alt_dest = dest.parent / "incidents.csv"
    df.to_csv(alt_dest, index=False)
    logger.info(f"Generated {len(df)} synthetic incident notes. Saved to {dest} and {alt_dest}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic battery incident reports.")
    parser.add_argument("--samples-per-class", type=int, default=150, help="Samples per category (default: 150 -> 750 total)")
    parser.add_argument("--out", type=str, default="data/incidents/incidents.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_dataset(n_per_category=args.samples_per_class, out_path=args.out)
    print(f"\n[INCIDENTS GENERATED] Total records: {len(df)}")
    print(df["category"].value_counts().to_string())
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
