"""scripts/build_kb_index.py - Build FAISS dense index over knowledge base docs (Milestone M9 / Exp 4).

Reads markdown docs in data/kb/, chunks by section/token budget, embeds using all-MiniLM-L6-v2,
builds L2-normalized FAISS IndexFlatIP (cosine similarity), and saves artifacts to:
- artifacts/index/kb.faiss
- artifacts/index/kb_chunks.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from guardian.utils.logging import get_logger

logger = get_logger(__name__)


def chunk_document(file_path: Path, max_words: int = 250, overlap_words: int = 40) -> List[Dict[str, Any]]:
    """Split a markdown document into overlapping passage chunks."""
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    doc_name = file_path.stem  # e.g. 01_thermal_runaway_fundamentals
    match = re.match(r"^(\d+)", doc_name)
    doc_num = match.group(1) if match else "00"
    doc_id = f"KB-{int(doc_num):02d}"

    # Extract title from first H1
    title = doc_name.replace("_", " ").title()
    for line in lines:
        if line.startswith("# "):
            title = line.replace("# ", "").strip()
            break

    # Split by H2 sections
    sections = [s.strip() for s in re.split(r"(?m)^(?=##\s+)", content) if s.strip()]

    chunks: List[str] = []
    for sec in sections:
        sec_words = sec.split()
        if len(sec_words) <= max_words:
            chunks.append(sec)
        else:
            # Sub-split long section with overlap
            for i in range(0, len(sec_words), max_words - overlap_words):
                chunk_words = sec_words[i : i + max_words]
                if chunk_words:
                    chunks.append(" ".join(chunk_words))

    records: List[Dict[str, Any]] = []
    for c_idx, text in enumerate(chunks, start=1):
        chunk_id = f"{doc_id}-C{c_idx}"
        records.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_title": title,
            "filename": file_path.name,
            "text": text,
            "word_count": len(text.split()),
        })

    return records


def build_index(
    kb_dir: str = "data/kb",
    index_out_dir: str = "artifacts/index",
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """Assemble all chunks, compute normalized embeddings, and build FAISS index."""
    kb_path = Path(kb_dir)
    doc_files = sorted([f for f in kb_path.glob("*.md") if f.name != "SOURCES.md"])

    logger.info(f"Discovered {len(doc_files)} markdown documents in {kb_dir}...")
    all_chunks: List[Dict[str, Any]] = []

    for f in doc_files:
        chunks = chunk_document(f)
        all_chunks.extend(chunks)

    logger.info(f"Generated {len(all_chunks)} total chunks across all documents.")

    logger.info(f"Loading embedding model: {embed_model_name}...")
    embedder = SentenceTransformer(embed_model_name)

    texts = [c["text"] for c in all_chunks]
    logger.info("Computing dense embeddings...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # Normalize vectors for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS IndexFlatIP initialized with {index.ntotal} vectors of dimension {dim}.")

    # Save to disk
    out_dir = Path(index_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = out_dir / "kb.faiss"
    chunks_path = out_dir / "kb_chunks.json"

    faiss.write_index(index, str(faiss_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    logger.info(f"Saved FAISS index to {faiss_path}")
    logger.info(f"Saved chunks metadata to {chunks_path}")

    return index, all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for knowledge base.")
    parser.add_argument("--kb-dir", type=str, default="data/kb", help="Path to knowledge base directory")
    parser.add_argument("--out-dir", type=str, default="artifacts/index", help="Output index directory")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    args = parser.parse_args()

    index, chunks = build_index(kb_dir=args.kb_dir, index_out_dir=args.out_dir, embed_model_name=args.model)
    print(f"\n[KB INDEXING COMPLETE]")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"FAISS index saved to: {args.out_dir}/kb.faiss")
    print(f"Chunks saved to: {args.out_dir}/kb_chunks.json")


if __name__ == "__main__":
    main()
