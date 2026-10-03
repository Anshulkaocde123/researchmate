#!/usr/bin/env python3
"""Build and persist Chroma vector index for ResearchMate chunks.

Features:
- Idempotent insertion: re-running upserts records without duplicates.
- Batched embedding and vector store writes.
- Informative progress logging.
- Measures and logs execution time and index size on disk.

Usage:
    python scripts/build_index.py [--chunks-path data/processed/chunks.jsonl] [--limit-papers 100]
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
from researchmate.indexing.embedder import FastEmbedder
from researchmate.indexing.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_dir_size(path: Path) -> int:
    """Calculate recursive directory size in bytes."""
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            with contextlib.suppress(OSError):
                total += fp.stat().st_size
    return total


def build_vector_index(
    chunks_path: Path = Path("data/processed/chunks.jsonl"),
    index_dir: Path | None = None,
    limit_papers: int | None = None,
    batch_size: int = 128,
) -> dict[str, int | float]:
    """Load chunks, embed with FastEmbedder, and upsert into Chroma VectorStore."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {chunks_path}")

    start_time = time.perf_counter()
    persist_dir = (index_dir or settings.index_dir) / "chroma"
    vector_store = VectorStore(persist_dir=persist_dir)
    embedder = FastEmbedder()

    initial_count = vector_store.count()
    log.info("Starting index build. Current index chunk count: %d", initial_count)

    # 1. Read chunks
    chunks: list[Chunk] = []
    seen_papers: set[str] = set()

    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pid = item.get("paper_id", "")
            if limit_papers is not None:
                if pid not in seen_papers and len(seen_papers) >= limit_papers:
                    continue
                seen_papers.add(pid)
            chunks.append(Chunk.model_validate(item))

    total_chunks = len(chunks)
    log.info(
        "Loaded %d chunks across %d papers (limit=%s). Indexing in batches of %d ...",
        total_chunks,
        len(seen_papers) if limit_papers else "all",
        limit_papers,
        batch_size,
    )

    # 2. Batched Embed & Insert
    inserted_chunks = 0
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        # Embed the contextualized embed_text
        embed_texts = [c.embed_text for c in batch]
        embeddings = embedder.embed_documents(embed_texts, batch_size=batch_size)
        vector_store.add(batch, embeddings)
        inserted_chunks += len(batch)

        pct = (inserted_chunks / total_chunks) * 100
        elapsed = time.perf_counter() - start_time
        log.info(
            "Progress: %d / %d chunks (%.1f%%) in %.1fs",
            inserted_chunks,
            total_chunks,
            pct,
            elapsed,
        )

    final_count = vector_store.count()
    duration = time.perf_counter() - start_time
    dir_size_bytes = get_dir_size(persist_dir)
    dir_size_mb = dir_size_bytes / (1024 * 1024)

    log.info(
        "Done! VectorStore chunk count: %d (net added: %d). Time: %.2fs. Disk size: %.2f MB",
        final_count,
        final_count - initial_count,
        duration,
        dir_size_mb,
    )

    return {
        "chunks_indexed": final_count,
        "elapsed_seconds": duration,
        "size_mb": dir_size_mb,
    }


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--limit-papers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    build_vector_index(
        chunks_path=args.chunks_path,
        index_dir=args.index_dir,
        limit_papers=args.limit_papers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
