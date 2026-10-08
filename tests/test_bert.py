"""Unit tests for incident report classification (Milestone M8 / Exp 5)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from guardian.models.bert_incident import (
    CATEGORIES,
    IncidentClassifier,
)


@pytest.fixture
def incidents_csv() -> Path:
    csv_path = Path("data/incidents/incident_reports.csv")
    if not csv_path.exists():
        from scripts.generate_incidents import generate_dataset
        generate_dataset(n_per_category=30, out_path=str(csv_path))
    return csv_path


def test_incident_reports_dataset_schema(incidents_csv: Path) -> None:
    df = pd.read_csv(incidents_csv)
    assert len(df) >= 150
    expected_cols = {"incident_id", "timestamp", "battery_id", "cycle", "category", "note", "severity", "is_synthetic"}
    assert expected_cols.issubset(set(df.columns))

    categories_found = set(df["category"].unique())
    assert set(CATEGORIES).issubset(categories_found)
    assert bool(df["is_synthetic"].all()) is True


def test_incident_classifier_predict() -> None:
    classifier = IncidentClassifier()
    # Train lightweight baseline on small representative examples
    train_texts = [
        "Thermal alarm activated, cell skin temperature reached 58.5 °C.",
        "High temperature cutoff after rapid thermal rise.",
        "Current surged to 45.0 A during sudden acceleration.",
        "Overcurrent warning logged on discharge contactor line.",
        "Terminal voltage dropped down to 2.15 V under load.",
        "Overvoltage alert during balancing, cell reached 4.40 V.",
        "Thermistor flat-lined at 25 °C with zero dT/dt recorded.",
        "Erratic voltage oscillations due to loose sensor wiring.",
        "Routine scheduled inspection completed, all cells balanced within 12 mV.",
        "Nominal operational cycle documented with nominal thermal profile.",
    ]
    train_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    classifier.fit_baseline(train_texts, train_labels)

    # Test thermal prediction
    res_thermal = classifier.predict("Cell temperature climbed to 55 °C rapidly.")
    assert res_thermal["category"] == "thermal_overheat"
    assert res_thermal["confidence"] >= 0.2

    # Test current prediction
    res_curr = classifier.predict("Current spiked to 40 A over rated limit.")
    assert res_curr["category"] == "overcurrent"
    assert res_curr["confidence"] >= 0.2

    # Test normal observation
    res_norm = classifier.predict("Standard routine maintenance check with balanced cells.")
    assert res_norm["category"] == "normal_observation"


def test_incident_classifier_save_load() -> None:
    classifier = IncidentClassifier()
    train_texts = [
        "Thermal alarm activated: 58 °C.",
        "Current surged to 45 A.",
        "Voltage sagged to 2.1 V.",
        "Sensor flat-lined at 25 °C.",
        "Routine scheduled inspection passed.",
    ]
    train_labels = [0, 1, 2, 3, 4]
    classifier.fit_baseline(train_texts, train_labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        classifier.save(tmpdir)

        loaded = IncidentClassifier()
        loaded.load(tmpdir)

        pred1 = classifier.predict("Overcurrent fault logged.")
        pred2 = loaded.predict("Overcurrent fault logged.")

        assert pred1["category"] == pred2["category"]
        assert pytest.approx(pred1["confidence"], rel=1e-3) == pred2["confidence"]
