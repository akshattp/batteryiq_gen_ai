"""tests/test_lora.py - Acceptance test for M11 LoRA Explainer (Exp 7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from guardian.genai.llm_client import LLMClient, LocalHFClient
from guardian.models.lora_explainer import LoRAExplainer
from scripts.train_lora import train_lora


def test_lora_dataset_exists():
    """Verify train and validation jsonl files exist and have valid structure."""
    train_file = Path("data/lora/train.jsonl")
    val_file = Path("data/lora/val.jsonl")

    assert train_file.exists(), "train.jsonl not found"
    assert val_file.exists(), "val.jsonl not found"

    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = [json.loads(line) for line in f if line.strip()]

    with open(val_file, "r", encoding="utf-8") as f:
        val_lines = [json.loads(line) for line in f if line.strip()]

    assert len(train_lines) >= 300, f"Expected >= 300 train samples, got {len(train_lines)}"
    assert len(val_lines) >= 50, f"Expected >= 50 val samples, got {len(val_lines)}"

    sample = train_lines[0]
    assert "instruction" in sample
    assert "input" in sample
    assert "output" in sample

    # Output must parse as JSON with required keys
    out_obj = json.loads(sample["output"])
    assert "summary" in out_obj
    assert "reasons" in out_obj
    assert "recommended_actions" in out_obj


def test_lora_train_smoke_and_metrics():
    """Run CPU smoke training (5 steps) and verify saved adapter and metrics."""
    metrics = train_lora(
        smoke=True,
        data_dir="data/lora",
        output_dir="artifacts/models/lora_explainer",
        metrics_path="artifacts/metrics/lora.json",
    )

    adapter_dir = Path("artifacts/models/lora_explainer")
    assert adapter_dir.exists(), "Adapter directory was not created"
    assert (adapter_dir / "adapter_config.json").exists(), "adapter_config.json missing"

    metrics_file = Path("artifacts/metrics/lora.json")
    assert metrics_file.exists(), "artifacts/metrics/lora.json missing"

    assert metrics["smoke_mode"] is True
    assert "evaluation" in metrics
    eval_m = metrics["evaluation"]
    assert "mean_rouge_l" in eval_m
    assert "json_validity_rate" in eval_m
    assert "numeric_fidelity_rate" in eval_m
    assert eval_m["forbidden_violation_rate"] == 0.0


def test_lora_explainer_inference():
    """Verify LoRAExplainer loads the adapter and produces an explanation."""
    adapter_dir = "artifacts/models/lora_explainer"
    explainer = LoRAExplainer(adapter_path=adapter_dir)

    sample_assessment = {
        "battery_id": "B0005",
        "cycle": 50,
        "t_s": 500.0,
        "risk_level": "WARNING",
        "risk_score": 42.0,
        "anomaly_score": 0.45,
        "predicted_temp_c": 36.2,
        "telemetry": {
            "temp_c": 35.8,
            "dT_dt": 0.42,
            "voltage_v": 3.45,
            "current_a": 2.1,
            "soc_pct": 55.0,
            "soh_pct": 88.0,
        },
        "factors": [
            {"name": "Temperature", "raw_value": 35.8, "weighted_score": 15.0},
        ],
        "hard_overrides": [],
    }

    prompt = explainer.format_input(sample_assessment)
    output = explainer.generate(prompt, max_new_tokens=150)
    assert isinstance(output, str)
    assert len(output.strip()) > 0


def test_local_hf_client_protocol():
    """Verify LocalHFClient implements LLMClient protocol."""
    client = LocalHFClient(adapter_path="artifacts/models/lora_explainer")
    assert isinstance(client, LLMClient)
