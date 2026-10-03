#!/usr/bin/env python3
"""Build and persist Chroma vector index and BM25 sparse index for ResearchMate chunks.

Features:
- Chroma vector store with FastEmbed embeddings.
- BM25Okapi sparse keyword index with gzip-compressed pickle serialization.
- Idempotent insertion for Chroma; fast single-pass build for BM25.
- Measures and logs execution times and disk sizes.

Usage:
    python scripts/build_index.py [--chunks-path data/processed/chunks.jsonl] [--limit-papers 100] [--mode all|vector|bm25]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import time
from pathlib import Path

from researchmate.chunking.chunker import Chunk
from researchmate.config import settings
from researchmate.indexing.bm25_index import BM25Index
from researchmate.indexing.embedder import FastEmbedder
from researchmate.indexing.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_dir_size(path: Path) -> int:
    """Calculate recursive directory or file size in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            with contextlib.suppress(OSError):
                total += fp.stat().st_size
    return total


def load_chunks(
    chunks_path: Path,
    limit_papers: int | None = None,
    filter_paper_ids: set[str] | None = None,
) -> list[Chunk]:
    """Load chunks from jsonl with optional paper limit or paper_id whitelist."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {chunks_path}")

    chunks: list[Chunk] = []
    seen_papers: set[str] = set()

    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pid = item.get("paper_id", "")
            if filter_paper_ids is not None and pid not in filter_paper_ids:
                continue
            if limit_papers is not None:
                if pid not in seen_papers and len(seen_papers) >= limit_papers:
                    continue
                seen_papers.add(pid)
            chunks.append(Chunk.model_validate(item))

    num_papers = len(seen_papers) if limit_papers else len({c.paper_id for c in chunks})
    log.info(
        "Loaded %d chunks across %d papers (limit=%s, filtered=%s)",
        len(chunks),
        num_papers,
        limit_papers,
        filter_paper_ids is not None,
    )
    return chunks


def build_vector_index(
    chunks: list[Chunk],
    index_dir: Path | None = None,
    batch_size: int = 128,
) -> dict[str, int | float]:
    """Embed chunks with FastEmbedder and upsert into Chroma VectorStore."""
    start_time = time.perf_counter()
    persist_dir = (index_dir or settings.index_dir) / "chroma"
    vector_store = VectorStore(persist_dir=persist_dir)
    embedder = FastEmbedder()

    initial_count = vector_store.count()
    total_chunks = len(chunks)
    log.info(
        "Building Chroma vector index. Initial count: %d, targets: %d chunks",
        initial_count,
        total_chunks,
    )

    inserted_chunks = 0
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        embed_texts = [c.embed_text for c in batch]
        embeddings = embedder.embed_documents(embed_texts, batch_size=batch_size)
        vector_store.add(batch, embeddings)
        inserted_chunks += len(batch)

        pct = (inserted_chunks / total_chunks) * 100
        elapsed = time.perf_counter() - start_time
        log.info(
            "Chroma progress: %d / %d chunks (%.1f%%) in %.1fs",
            inserted_chunks,
            total_chunks,
            pct,
            elapsed,
        )

    final_count = vector_store.count()
    duration = time.perf_counter() - start_time
    dir_size_mb = get_dir_size(persist_dir) / (1024 * 1024)

    log.info(
        "Chroma index complete! Total chunks: %d. Time: %.2fs. Disk size: %.2f MB",
        final_count,
        duration,
        dir_size_mb,
    )
    return {
        "chunks_indexed": final_count,
        "elapsed_seconds": duration,
        "size_mb": dir_size_mb,
    }


def build_bm25_index(
    chunks: list[Chunk],
    index_dir: Path | None = None,
) -> dict[str, int | float]:
    """Construct BM25 sparse index and save to disk."""
    start_time = time.perf_counter()
    bm25_path = (index_dir or settings.index_dir) / "bm25.pkl.gz"
    log.info("Building BM25 sparse index for %d chunks...", len(chunks))

    bm25_index = BM25Index(chunks)
    bm25_index.save(bm25_path)

    duration = time.perf_counter() - start_time
    size_mb = get_dir_size(bm25_path) / (1024 * 1024)

    log.info(
        "BM25 index complete! Total chunks: %d. Time: %.2fs. Disk size: %.2f MB",
        bm25_index.count(),
        duration,
        size_mb,
    )
    return {
        "chunks_indexed": bm25_index.count(),
        "elapsed_seconds": duration,
        "size_mb": size_mb,
    }


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--limit-papers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only index papers present in the evaluation set.",
    )
    parser.add_argument(
        "--eval-set-path",
        type=Path,
        default=Path("data/eval/eval_set.jsonl"),
        help="Path to evaluation set jsonl.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "vector", "bm25"],
        default="all",
        help="Which indexes to build: all, vector, or bm25.",
    )
    args = parser.parse_args()

    filter_paper_ids = None
    if args.eval_only:
        if not args.eval_set_path.exists():
            raise FileNotFoundError(f"Eval set file not found: {args.eval_set_path}")
        with args.eval_set_path.open(encoding="utf-8") as f:
            filter_paper_ids = {json.loads(line)["paper_id"] for line in f if line.strip()}
        log.info("Filtering to %d eval papers", len(filter_paper_ids))

    chunks = load_chunks(
        args.chunks_path,
        limit_papers=args.limit_papers,
        filter_paper_ids=filter_paper_ids,
    )

    if args.mode in ("all", "bm25"):
        build_bm25_index(chunks, index_dir=args.index_dir)

    if args.mode in ("all", "vector"):
        build_vector_index(chunks, index_dir=args.index_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
