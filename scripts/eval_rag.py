"""scripts/eval_rag.py - Comprehensive evaluation of RAG pipeline (Milestone M9 / Exp 4).

Evaluates retrieval quality (hit@1, hit@4) and citation fidelity across 20 benchmark questions
in tests/rag_eval.yaml, persisting metrics to artifacts/metrics/rag.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from guardian.genai.llm_client import TemplateClient, get_llm_client
from guardian.genai.rag import RAGPipeline
from guardian.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_rag_benchmark(
    eval_yaml_path: str = "tests/rag_eval.yaml",
    client: Any = None,
    top_k: int = 4,
    out_metrics_path: str = "artifacts/metrics/rag.json",
) -> Dict[str, Any]:
    """Run full evaluation suite over benchmark questions."""
    with open(eval_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    questions: List[Dict[str, Any]] = data.get("questions", [])
    logger.info(f"Loaded {len(questions)} evaluation questions from {eval_yaml_path}...")

    rag = RAGPipeline()
    if rag.index is None:
        raise RuntimeError("FAISS index not loaded. Run scripts/build_kb_index.py first.")

    hits_at_1 = 0
    hits_at_k = 0
    with_citations = 0
    valid_citations_count = 0
    total_latency = 0.0

    eval_details: List[Dict[str, Any]] = []

    for item in questions:
        qid = item["id"]
        q_text = item["question"]
        expected_doc = item["expected_doc_id"]

        t0 = time.time()
        retrieved = rag.retrieve(q_text, top_k=top_k)
        retrieval_time = time.time() - t0

        retrieved_doc_ids = [c["doc_id"] for c in retrieved]
        retrieved_chunk_ids = [c["chunk_id"] for c in retrieved]

        hit_1 = (len(retrieved_doc_ids) > 0 and retrieved_doc_ids[0] == expected_doc)
        hit_k = (expected_doc in retrieved_doc_ids)

        if hit_1:
            hits_at_1 += 1
        if hit_k:
            hits_at_k += 1

        # Generate answer with citations
        t_gen0 = time.time()
        res = rag.answer(q_text, client=client, top_k=top_k)
        latency = (time.time() - t_gen0) + retrieval_time
        total_latency += latency

        citations = res.get("citations", [])
        has_cits = len(citations) > 0
        if has_cits:
            with_citations += 1

        # Citations must come from retrieved set
        all_cits_valid = all(cid in retrieved_chunk_ids for cid in citations) if citations else False
        if all_cits_valid:
            valid_citations_count += 1

        eval_details.append({
            "id": qid,
            "question": q_text,
            "expected_doc_id": expected_doc,
            "retrieved_doc_ids": retrieved_doc_ids,
            "top_1_hit": hit_1,
            "top_k_hit": hit_k,
            "citations": citations,
            "citations_valid": all_cits_valid,
            "confidence_score": res.get("confidence_score", 0.0),
            "latency_s": round(latency, 3),
        })

    n = len(questions)
    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_questions": n,
        "top_k": top_k,
        "hit_at_1": round(hits_at_1 / n, 4),
        "hit_at_k": round(hits_at_k / n, 4),
        "target_hit_at_k": 0.80,
        "hit_at_k_passed": bool(hits_at_k / n >= 0.80),
        "citation_presence_rate": round(with_citations / n, 4),
        "citation_validity_rate": round(valid_citations_count / n, 4),
        "avg_latency_s": round(total_latency / n, 3),
        "details": eval_details,
    }

    out_file = Path(out_metrics_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation complete. Hit@{top_k}: {summary['hit_at_k']:.1%}, Citation validity: {summary['citation_validity_rate']:.1%}")
    logger.info(f"Saved RAG metrics to {out_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline.")
    parser.add_argument("--eval-yaml", type=str, default="tests/rag_eval.yaml", help="Path to evaluation questions YAML")
    parser.add_argument("--offline", action="store_true", help="Force TemplateClient offline execution")
    parser.add_argument("--out", type=str, default="artifacts/metrics/rag.json", help="Output JSON path")
    args = parser.parse_args()

    client = TemplateClient() if args.offline else get_llm_client()
    summary = evaluate_rag_benchmark(eval_yaml_path=args.eval_yaml, client=client, out_metrics_path=args.out)

    print(f"\n[RAG BENCHMARK COMPLETE]")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Hit@1: {summary['hit_at_1']:.1%}")
    print(f"Hit@4: {summary['hit_at_k']:.1%} (Target >= 80%: {'PASS' if summary['hit_at_k_passed'] else 'FAIL'})")
    print(f"Citation Validity Rate: {summary['citation_validity_rate']:.1%}")
    print(f"Average Latency: {summary['avg_latency_s']}s")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
