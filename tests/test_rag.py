"""Unit tests for RAG Knowledge Assistant pipeline (Milestone M9 / Exp 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from guardian.genai.llm_client import TemplateClient
from guardian.genai.rag import RAGPipeline


@pytest.fixture(scope="module")
def rag_pipeline() -> RAGPipeline:
    rag = RAGPipeline()
    if rag.index is None:
        from scripts.build_kb_index import build_index
        build_index()
        rag = RAGPipeline()
    return rag


def test_rag_pipeline_retrieval(rag_pipeline: RAGPipeline) -> None:
    query = "What happens during thermal runaway in lithium cells?"
    results = rag_pipeline.retrieve(query, top_k=4)

    assert len(results) > 0
    assert len(results) <= 4

    top_chunk = results[0]
    for key in ("chunk_id", "doc_id", "doc_title", "text", "score"):
        assert key in top_chunk

    assert top_chunk["score"] > 0.4
    # Expected to retrieve KB-01 for thermal runaway
    retrieved_doc_ids = [c["doc_id"] for c in results]
    assert "KB-01" in retrieved_doc_ids


def test_rag_pipeline_offline_answer_has_citations(rag_pipeline: RAGPipeline) -> None:
    query = "Why does high electrical current cause severe Joule heating?"
    client = TemplateClient()

    res = rag_pipeline.answer(query, client=client, top_k=4)

    assert res["has_citations"] is True
    assert len(res["citations"]) >= 1

    # Every citation must be a valid chunk ID in the retrieved set
    retrieved_chunk_ids = {c["chunk_id"] for c in res["retrieved_chunks"]}
    for cid in res["citations"]:
        assert cid in retrieved_chunk_ids
        assert f"[{cid}]" in res["answer"]


def test_rag_pipeline_with_telemetry_context(rag_pipeline: RAGPipeline) -> None:
    query = "Why is cell temperature rising?"
    telemetry = {
        "temp_c": 46.5,
        "voltage_v": 3.4,
        "current_a": 3.8,
        "risk_level": "HIGH",
    }
    client = TemplateClient()

    res = rag_pipeline.answer(query, telemetry_context=telemetry, client=client)

    assert "[Measured/Model]" in res["answer"]
    assert "[Knowledge Base]" in res["answer"]
    assert res["has_citations"] is True


def test_rag_pipeline_irrelevant_query_not_found(rag_pipeline: RAGPipeline) -> None:
    # Gibberish with very high similarity threshold
    query = "Quantum mechanics of interstellar dark matter neutrino oscillations"
    res = rag_pipeline.answer(query, min_similarity=0.95)

    assert "not found in battery safety knowledge base" in res["answer"].lower()
    assert len(res["citations"]) == 0


def test_rag_no_answer_without_citations(rag_pipeline: RAGPipeline) -> None:
    test_queries = [
        "What are the safe voltage bounds for lithium cells?",
        "How does Coulomb counting drift over time?",
        "What is the 15-second debounce window in the Guardian event monitor?",
    ]
    client = TemplateClient()

    for q in test_queries:
        res = rag_pipeline.answer(q, client=client)
        assert res["has_citations"] is True, f"Answer for '{q}' had no citations!"
        assert len(res["citations"]) > 0
