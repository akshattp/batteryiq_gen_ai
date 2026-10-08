"""scripts/build_lora_dataset.py - Generate training and validation pairs for LoRA explainer (M11 / Exp 7).

Generates 500 (structured assessment -> validated JSON explanation) pairs,
holding out 15% for evaluation.
Outputs to:
- data/lora/train.jsonl
- data/lora/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from guardian.genai.explain import template_explanation
from guardian.utils.logging import get_logger
from guardian.utils.seed import set_seed

logger = get_logger(__name__)


def generate_synthetic_assessment_sample(seed: int) -> Dict[str, Any]:
    """Generate a realistic structured assessment covering the 4 risk tiers."""
    rng = random.Random(seed)
    tier = rng.choice(["NORMAL", "WARNING", "HIGH", "CRITICAL"])
    bat = rng.choice(["B0005", "B0006", "B0007", "B0018"])
    cycle = rng.randint(5, 160)
    t_s = round(rng.uniform(60.0, 1800.0), 1)

    if tier == "NORMAL":
        temp = round(rng.uniform(20.0, 29.0), 1)
        rate = round(rng.uniform(0.01, 0.15), 2)
        v = round(rng.uniform(3.60, 4.10), 2)
        curr = round(rng.uniform(0.8, 1.8), 2)
        anomaly = round(rng.uniform(0.01, 0.15), 3)
        score = round(rng.uniform(5.0, 28.0), 1)
        overrides = []
    elif tier == "WARNING":
        temp = round(rng.uniform(35.0, 41.5), 1)
        rate = round(rng.uniform(0.3, 0.9), 2)
        v = round(rng.uniform(3.30, 3.60), 2)
        curr = round(rng.uniform(1.8, 2.5), 2)
        anomaly = round(rng.uniform(0.35, 0.65), 3)
        score = round(rng.uniform(32.0, 54.0), 1)
        overrides = []
    elif tier == "HIGH":
        temp = round(rng.uniform(42.0, 49.5), 1)
        rate = round(rng.uniform(1.0, 2.8), 2)
        v = round(rng.uniform(2.60, 3.20), 2)
        curr = round(rng.uniform(2.6, 3.8), 2)
        anomaly = round(rng.uniform(0.70, 0.92), 3)
        score = round(rng.uniform(56.0, 74.0), 1)
        overrides = ["Temperature >= 42.0 °C (HIGH)"] if temp >= 42.0 else []
    else:  # CRITICAL
        temp = round(rng.uniform(50.0, 65.0), 1)
        rate = round(rng.uniform(3.0, 7.5), 2)
        v = round(rng.uniform(2.10, 2.48), 2)
        curr = round(rng.uniform(4.0, 6.0), 2)
        anomaly = round(rng.uniform(0.93, 0.99), 3)
        score = round(rng.uniform(78.0, 99.0), 1)
        overrides = ["Temperature >= 50.0 °C (CRITICAL)"]

    return {
        "battery_id": bat,
        "cycle": cycle,
        "t_s": t_s,
        "risk_level": tier,
        "risk_score": score,
        "anomaly_score": anomaly,
        "predicted_temp_c": round(temp + rate * 1.5, 1),
        "telemetry": {
            "temp_c": temp,
            "dT_dt": rate,
            "voltage_v": v,
            "current_a": curr,
            "soc_pct": round(rng.uniform(10.0, 90.0), 1),
            "soh_pct": round(rng.uniform(70.0, 98.0), 1),
        },
        "factors": [
            {"name": "Temperature", "raw_value": temp, "weighted_score": round(score * 0.35, 1)},
            {"name": "Anomaly", "raw_value": anomaly, "weighted_score": round(score * 0.25, 1)},
        ],
        "hard_overrides": overrides,
    }


def build_dataset(n_samples: int = 500, out_dir: str = "data/lora") -> Tuple[int, int]:
    """Generate JSONL training and validation dataset for fine-tuning."""
    set_seed(42)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for i in range(n_samples):
        assessment = generate_synthetic_assessment_sample(seed=1000 + i)
        explanation = template_explanation(assessment)

        records.append({
            "instruction": (
                "You are the AI Battery Safety Guardian diagnostic engine. "
                "Analyze the provided structured battery telemetry assessment and generate a "
                "strictly grounded, non-hallucinatory engineering explanation in valid JSON format."
            ),
            "input": json.dumps(assessment),
            "output": json.dumps(explanation.to_dict()),
        })

    # Shuffle
    random.Random(42).shuffle(records)

    # 85% train / 15% val split
    n_val = int(n_samples * 0.15)
    train_records = records[n_val:]
    val_records = records[:n_val]

    train_path = dest / "train.jsonl"
    val_path = dest / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Generated LoRA dataset: {len(train_records)} train, {len(val_records)} val -> {dest}")
    return len(train_records), len(val_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset for LoRA explainer fine-tuning.")
    parser.add_argument("--n-samples", type=int, default=500, help="Total samples to generate (default: 500)")
    parser.add_argument("--out-dir", type=str, default="data/lora", help="Output directory")
    args = parser.parse_args()

    n_train, n_val = build_dataset(n_samples=args.n_samples, out_dir=args.out_dir)
    print(f"\n[LoRA DATASET COMPLETE]")
    print(f"Train samples: {n_train} ({args.out_dir}/train.jsonl)")
    print(f"Val samples: {n_val} ({args.out_dir}/val.jsonl)")


if __name__ == "__main__":
    main()
