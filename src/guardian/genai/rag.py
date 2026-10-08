"""Retrieval-Augmented Generation (RAG) pipeline (Experiment 4).

Indexes markdown safety documentation using FAISS-cpu and MiniLM embeddings,
retrieving relevant knowledge chunks with chunk-id citations for the chat assistant.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from guardian.genai.llm_client import LLMClient, TemplateClient, get_llm_client
from guardian.genai.prompts import load_prompt
from guardian.utils.config import get_config
from guardian.utils.logging import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """FAISS-based Battery Safety Knowledge Base RAG Assistant."""

    def __init__(
        self,
        index_dir: str | Path = "artifacts/index",
        embed_model_name: str = "all-MiniLM-L6-v2",
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.config = config or get_config()
        self.embed_model_name = embed_model_name
        self.llm_client = llm_client

        self.faiss_path = self.index_dir / "kb.faiss"
        self.chunks_path = self.index_dir / "kb_chunks.json"

        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []
        self._embedder: Optional[SentenceTransformer] = None

        self._load_index()

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _load_index(self) -> None:
        if self.faiss_path.exists() and self.chunks_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} chunks from {self.index_dir}")
        else:
            logger.warning(f"FAISS index not found at {self.index_dir}. Run scripts/build_kb_index.py first.")

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant knowledge chunks with cosine similarity above threshold."""
        if self.index is None or not self.chunks:
            return []

        embedder = self._get_embedder()
        q_emb = embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        top_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_emb, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            sim = float(score)
            if sim >= min_similarity:
                chunk_data = dict(self.chunks[idx])
                chunk_data["score"] = round(sim, 4)
                results.append(chunk_data)

        return results

    def format_retrieved_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into structured context text."""
        formatted_blocks = []
        for c in chunks:
            formatted_blocks.append(
                f"[{c['chunk_id']}] Document: {c['doc_title']}\n{c['text']}"
            )
        return "\n\n".join(formatted_blocks)

    def extract_citations(self, text: str) -> List[str]:
        """Extract all [KB-xx-Cx] citation tokens from text."""
        matches = re.findall(r"\[(KB-\d+-C\d+)\]", text)
        return sorted(list(set(matches)))

    def answer(
        self,
        query: str,
        telemetry_context: Optional[Dict[str, Any]] = None,
        client: Optional[LLMClient] = None,
        prompt_name: str = "rag_answer_v1",
        top_k: int = 4,
        min_similarity: float = 0.25,
    ) -> Dict[str, Any]:
        """Generate cited answer separating [Measured/Model] from [Knowledge Base] facts."""
        chunks = self.retrieve(query, top_k=top_k, min_similarity=min_similarity)

        if not chunks:
            return {
                "answer": "Information not found in battery safety knowledge base.",
                "citations": [],
                "retrieved_chunks": [],
                "has_citations": False,
                "confidence_score": 0.0,
            }

        retrieved_ids = {c["chunk_id"] for c in chunks}
        prompt = load_prompt(prompt_name)

        telemetry_str = (
            json.dumps(telemetry_context, indent=2)
            if telemetry_context
            else "No active telemetry context provided."
        )
        chunks_str = self.format_retrieved_chunks(chunks)

        messages = prompt.render_messages(
            question=query,
            telemetry_context=telemetry_str,
            retrieved_chunks=chunks_str,
        )

        if client is None:
            client = get_llm_client()

        if isinstance(client, TemplateClient):
            # Deterministic offline cited answer fallback
            top_chunk = chunks[0]
            cid = top_chunk["chunk_id"]
            title = top_chunk["doc_title"]
            preview = top_chunk["text"][:200]

            measured_part = ""
            if telemetry_context:
                t = telemetry_context.get("temp_c", "N/A")
                v = telemetry_context.get("voltage_v", "N/A")
                level = telemetry_context.get("risk_level", "NORMAL")
                measured_part = f"[Measured/Model]: Current telemetry indicates risk level {level} (T={t} °C, V={v} V).\n\n"

            answer_text = (
                f"{measured_part}[Knowledge Base]: According to {title} [{cid}], "
                f"{preview.replace(chr(10), ' ')}...\n\n"
                "Advisory Note: Human engineer verification recommended before taking any operational actions."
            )
            return {
                "answer": answer_text,
                "citations": [cid],
                "retrieved_chunks": chunks,
                "has_citations": True,
                "confidence_score": top_chunk["score"],
            }

        # Call LLM
        try:
            raw_output = client.generate(
                messages,
                json_schema=prompt.output_schema,
                temperature=0.2,
                max_tokens=600,
            )
            data = json.loads(raw_output)
            answer_text = str(data.get("answer", ""))
            cited_ids = data.get("cited_chunk_ids", [])
            # Also extract citations from answer text
            text_citations = self.extract_citations(answer_text)
            all_citations = sorted(list(set(cited_ids + text_citations)))

            # Validate citations exist in retrieved set
            valid_citations = [cid for cid in all_citations if cid in retrieved_ids]

            return {
                "answer": answer_text,
                "citations": valid_citations,
                "retrieved_chunks": chunks,
                "has_citations": len(valid_citations) > 0,
                "confidence_score": float(data.get("confidence_score", chunks[0]["score"])),
            }
        except Exception as exc:
            logger.warning(f"RAG LLM generation failed: {exc}. Using deterministic cited fallback.")
            # Graceful fallback
            top_chunk = chunks[0]
            cid = top_chunk["chunk_id"]
            fallback_answer = (
                f"[Knowledge Base]: According to {top_chunk['doc_title']} [{cid}], {top_chunk['text'][:250]}...\n\n"
                "Advisory: Operational parameters require review by a qualified engineer."
            )
            return {
                "answer": fallback_answer,
                "citations": [cid],
                "retrieved_chunks": chunks,
                "has_citations": True,
                "confidence_score": top_chunk["score"],
            }
