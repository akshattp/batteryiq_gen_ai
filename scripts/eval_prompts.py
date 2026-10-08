"""scripts/eval_prompts.py - Prompt engineering evaluation study (Experiment 1 + 6).

Evaluates prompt variants (explain_v1 zero-shot, explain_v2 few-shot, explain_v3 structured JSON)
on stored battery assessments using validator metrics (numeric fidelity, level consistency,
forbidden-content safety, schema compliance, latency, length).

Saves comparative metrics to artifacts/metrics/prompts.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from guardian.genai.explain import generate_explanation
from guardian.genai.llm_client import TemplateClient, get_llm_client
from guardian.genai.prompts import load_prompt
from guardian.genai.validator import validate_explanation
from guardian.utils.logging import get_logger

logger = get_logger(__name__)


def generate_benchmark_assessments(n: int = 30) -> List[Dict[str, Any]]:
    """Generate diverse benchmark assessment cases spanning all 4 risk levels."""
    assessments: List[Dict[str, Any]] = []

    # 1. Normal operating scenarios (9 cases)
    for i in range(9):
        temp = round(22.0 + i * 0.8, 1)
        v = round(3.85 - i * 0.04, 2)
        curr = round(1.0 + (i % 3) * 0.4, 2)
        anomaly = round(0.04 + i * 0.02, 2)
        score = round(10.0 + i * 2.0, 1)
        assessments.append({
            "battery_id": f"B000{(i % 3) + 5}",
            "cycle": 10 + i * 15,
            "t_s": 120.0 + i * 60,
            "risk_level": "NORMAL",
            "risk_score": score,
            "anomaly_score": anomaly,
            "predicted_temp_c": round(temp + 0.3, 1),
            "telemetry": {
                "temp_c": temp,
                "dT_dt": 0.05,
                "voltage_v": v,
                "current_a": curr,
                "soc_pct": round(90.0 - i * 5.0, 1),
                "soh_pct": round(98.0 - i * 0.5, 1),
            },
            "factors": [
                {"name": "Temperature", "raw_value": temp, "weighted_score": round(score * 0.35, 1)},
                {"name": "Current", "raw_value": curr, "weighted_score": round(score * 0.25, 1)},
            ],
            "hard_overrides": [],
        })

    # 2. Warning operational scenarios (7 cases)
    for i in range(7):
        temp = round(35.5 + i * 0.9, 1)
        v = round(3.50 - i * 0.08, 2)
        curr = round(2.2 + i * 0.3, 2)
        anomaly = round(0.40 + i * 0.05, 2)
        score = round(32.0 + i * 3.2, 1)
        assessments.append({
            "battery_id": f"B000{(i % 3) + 5}",
            "cycle": 60 + i * 12,
            "t_s": 300.0 + i * 40,
            "risk_level": "WARNING",
            "risk_score": score,
            "anomaly_score": anomaly,
            "predicted_temp_c": round(temp + 1.2, 1),
            "telemetry": {
                "temp_c": temp,
                "dT_dt": round(0.40 + i * 0.1, 2),
                "voltage_v": v,
                "current_a": curr,
                "soc_pct": round(60.0 - i * 4.0, 1),
                "soh_pct": round(88.0 - i * 1.0, 1),
            },
            "factors": [
                {"name": "Temperature", "raw_value": temp, "weighted_score": round(score * 0.4, 1)},
                {"name": "Temp trend", "raw_value": round(0.40 + i * 0.1, 2), "weighted_score": round(score * 0.3, 1)},
            ],
            "hard_overrides": [],
        })

    # 3. High risk scenarios (7 cases)
    for i in range(7):
        temp = round(42.5 + i * 0.9, 1)
        v = round(3.20 - i * 0.08, 2)
        curr = round(3.5 + i * 0.4, 2)
        anomaly = round(0.75 + i * 0.03, 2)
        score = round(58.0 + i * 2.5, 1)
        assessments.append({
            "battery_id": "B0018",
            "cycle": 110 + i * 8,
            "t_s": 500.0 + i * 30,
            "risk_level": "HIGH",
            "risk_score": score,
            "anomaly_score": anomaly,
            "predicted_temp_c": round(temp + 2.5, 1),
            "telemetry": {
                "temp_c": temp,
                "dT_dt": round(1.20 + i * 0.2, 2),
                "voltage_v": v,
                "current_a": curr,
                "soc_pct": round(40.0 - i * 3.0, 1),
                "soh_pct": round(76.0 - i * 1.5, 1),
            },
            "factors": [
                {"name": "Temperature", "raw_value": temp, "weighted_score": round(score * 0.45, 1)},
                {"name": "Anomaly", "raw_value": anomaly, "weighted_score": round(score * 0.35, 1)},
            ],
            "hard_overrides": ["Temperature >= 42.0 °C (HIGH)"] if temp >= 42.0 else [],
        })

    # 4. Critical safety scenarios (7 cases)
    for i in range(7):
        temp = round(50.5 + i * 1.2, 1)
        v = round(2.35 - i * 0.05, 2)
        curr = round(4.8 + i * 0.5, 2)
        anomaly = round(0.92 + i * 0.01, 2)
        score = round(78.0 + i * 3.0, 1)
        assessments.append({
            "battery_id": "B0018",
            "cycle": 140 + i * 5,
            "t_s": 750.0 + i * 20,
            "risk_level": "CRITICAL",
            "risk_score": min(score, 99.5),
            "anomaly_score": anomaly,
            "predicted_temp_c": round(temp + 3.8, 1),
            "telemetry": {
                "temp_c": temp,
                "dT_dt": round(2.50 + i * 0.4, 2),
                "voltage_v": v,
                "current_a": curr,
                "soc_pct": round(25.0 - i * 2.0, 1),
                "soh_pct": round(70.0 - i * 1.0, 1),
            },
            "factors": [
                {"name": "Temperature", "raw_value": temp, "weighted_score": 30.0},
                {"name": "Temp trend", "raw_value": round(2.50 + i * 0.4, 2), "weighted_score": 20.0},
                {"name": "Anomaly", "raw_value": anomaly, "weighted_score": 25.0},
            ],
            "hard_overrides": ["Temperature >= 50.0 °C (CRITICAL)"],
        })

    return assessments[:n]


def evaluate_prompt_variant(
    variant_name: str,
    assessments: List[Dict[str, Any]],
    client: Any,
) -> Dict[str, Any]:
    """Evaluate a specific prompt variant across benchmark assessments."""
    logger.info(f"Evaluating prompt variant: {variant_name} ({len(assessments)} samples)...")
    prompt = load_prompt(variant_name)

    total = len(assessments)
    first_pass_valid = 0
    healed = 0
    fallback = 0

    numeric_fidelity_hits = 0
    level_consistent_hits = 0
    forbidden_free_hits = 0
    schema_valid_hits = 0

    total_chars = 0
    total_latency = 0.0

    for idx, asm in enumerate(assessments, start=1):
        t0 = time.time()
        # Direct first attempt generation
        asm_json = json.dumps(asm, indent=2)
        messages = prompt.render_messages(assessment_json=asm_json)

        try:
            raw = client.generate(
                messages,
                json_schema=prompt.output_schema,
                temperature=0.2,
                max_tokens=600,
            )
            duration = time.time() - t0
        except Exception as exc:
            duration = time.time() - t0
            raw = ""

        total_latency += duration
        total_chars += len(raw)

        # Validate first pass
        violations = validate_explanation(raw, asm)

        has_schema_err = any(v.rule in ("json_syntax", "schema_missing_key", "empty_field") for v in violations)
        has_num_err = any(v.rule == "numeric_hallucination" for v in violations)
        has_level_err = any(v.rule in ("level_inconsistency", "contradictory_claim") for v in violations)
        has_verb_err = any(v.rule == "forbidden_control_verb" for v in violations)

        if not has_schema_err:
            schema_valid_hits += 1
        if not has_num_err:
            numeric_fidelity_hits += 1
        if not has_level_err:
            level_consistent_hits += 1
        if not has_verb_err:
            forbidden_free_hits += 1

        if len(violations) == 0:
            first_pass_valid += 1
        else:
            # Test self-correction loop
            expl = generate_explanation(asm, client=client, prompt_name=variant_name, max_retries=2)
            if "deterministic baseline template" in expl.confidence_note:
                fallback += 1
            else:
                healed += 1

    return {
        "variant": variant_name,
        "total_cases": total,
        "first_pass_valid_count": first_pass_valid,
        "first_pass_valid_rate": round(first_pass_valid / total, 4),
        "healed_count": healed,
        "healed_rate": round(healed / total, 4),
        "fallback_count": fallback,
        "fallback_rate": round(fallback / total, 4),
        "total_success_rate": round((first_pass_valid + healed) / total, 4),
        "numeric_fidelity_rate": round(numeric_fidelity_hits / total, 4),
        "level_consistency_rate": round(level_consistent_hits / total, 4),
        "forbidden_free_rate": round(forbidden_free_hits / total, 4),
        "schema_valid_rate": round(schema_valid_hits / total, 4),
        "avg_length_chars": round(total_chars / total, 1),
        "avg_latency_s": round(total_latency / total, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate explanation prompt variants.")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of benchmark assessments (default: 30)")
    parser.add_argument("--offline", action="store_true", help="Force TemplateClient offline execution")
    parser.add_argument("--out", type=str, default="artifacts/metrics/prompts.json", help="Output JSON path")
    args = parser.parse_args()

    client = TemplateClient() if args.offline else get_llm_client()
    logger.info(f"Using client: {client.__class__.__name__}")

    assessments = generate_benchmark_assessments(args.n_samples)
    variants = ["explain_v1", "explain_v2", "explain_v3"]
    results: List[Dict[str, Any]] = []

    for var in variants:
        res = evaluate_prompt_variant(var, assessments, client)
        results.append(res)
        logger.info(
            f"Result [{var}]: 1st-pass={res['first_pass_valid_rate']:.1%}, "
            f"Fidelity={res['numeric_fidelity_rate']:.1%}, Level={res['level_consistency_rate']:.1%}, "
            f"Forbidden-Free={res['forbidden_free_rate']:.1%}"
        )

    # Pick winner: highest total_success_rate, then numeric_fidelity_rate, then lowest avg_latency
    winner = max(
        results,
        key=lambda r: (r["total_success_rate"], r["numeric_fidelity_rate"], -r["avg_latency_s"]),
    )["variant"]

    output_payload = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "client": client.__class__.__name__,
        "n_samples": args.n_samples,
        "variants_evaluated": variants,
        "winner": winner,
        "rationale": f"Selected {winner} due to superior validation pass rate and strict numerical grounding.",
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info(f"Evaluation complete. Winner: {winner}. Saved metrics to {out_path}")
    print(f"\n[PROMPT STUDY COMPLETE] Winner: {winner}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
