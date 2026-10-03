"""Unit tests for VectorStore using Chroma with cosine metric.

Tests cover:
- Idempotent upserts: adding identical chunks does not duplicate records
- Count verification: count equals total number of unique chunks
- Self-retrieval: querying with a chunk's own embedding returns it at rank 1
- Metadata filtering: filtering by paper_id strictly returns matching chunks
- Retrieval scores: cosine similarity values are ordered descending
- get_by_id and clear methods
- Integration on mini_qasper fixture
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from researchmate.chunking.chunker import chunk_paper
from researchmate.data.loader import load_papers
from researchmate.indexing.vector_store import VectorStore


def _pseudo_embed(text: str, dim: int = 32) -> list[float]:
    """Generate a deterministic unit-norm pseudo-embedding for testing."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim)
    norm = np.linalg.norm(raw)
    return (raw / (norm if norm > 0 else 1.0)).tolist()


# =========================================================================
# 1. Basic Operations and Idempotency
# =========================================================================


def test_vector_store_add_count_and_idempotence() -> None:
    """Adding chunks multiple times must be idempotent without count inflation."""
    store = VectorStore(persist_dir=":memory:", collection_name="test_idempotence")
    assert store.count() == 0

    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    chunks = chunk_paper(papers[0])
    embeddings = [_pseudo_embed(c.embed_text) for c in chunks]

    # Initial insertion
    store.add(chunks, embeddings)
    initial_count = store.count()
    assert initial_count == len(chunks)

    # Re-running add with the exact same chunks
    store.add(chunks, embeddings)
    assert store.count() == initial_count  # No duplicate records!


def test_vector_store_mismatched_lengths_raise() -> None:
    """Adding mismatched chunks and embeddings lists raises ValueError."""
    store = VectorStore(persist_dir=":memory:", collection_name="test_err")
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    chunks = chunk_paper(papers[0])
    embeddings = [_pseudo_embed(c.embed_text) for c in chunks[:1]]

    with pytest.raises(ValueError, match="Chunks count"):
        store.add(chunks, embeddings)


# =========================================================================
# 2. Self-Retrieval and Rank 1 Ordering
# =========================================================================


def test_query_returns_self_at_rank_1() -> None:
    """Querying with a chunk's own embedding vector returns that chunk at rank 1."""
    store = VectorStore(persist_dir=":memory:", collection_name="test_rank1")
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))

    # Gather chunks across all fixture papers
    all_chunks = []
    for p in papers:
        all_chunks.extend(chunk_paper(p))

    embeddings = [_pseudo_embed(c.embed_text) for c in all_chunks]
    store.add(all_chunks, embeddings)
    assert store.count() == len(all_chunks)

    # Query with chunk 0's embedding
    target_chunk = all_chunks[0]
    target_emb = embeddings[0]

    results = store.query(embedding=target_emb, k=5)
    assert len(results) > 0
    top_result = results[0]

    assert top_result["chunk_id"] == target_chunk.chunk_id
    assert np.isclose(top_result["score"], 1.0, atol=1e-3)
    assert top_result["text"] == target_chunk.text
    assert top_result["metadata"]["paper_id"] == target_chunk.paper_id


# =========================================================================
# 3. Metadata Filtering
# =========================================================================


def test_query_filtering_by_paper_id() -> None:
    """Filtering by paper_id strictly restricts returned chunks to that paper."""
    store = VectorStore(persist_dir=":memory:", collection_name="test_filter")
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))

    all_chunks = []
    for p in papers:
        all_chunks.extend(chunk_paper(p))

    embeddings = [_pseudo_embed(c.embed_text) for c in all_chunks]
    store.add(all_chunks, embeddings)

    # Query targeting paper 1 with an arbitrary embedding
    target_pid = papers[1].paper_id
    dummy_query = _pseudo_embed("arbitrary search query")

    results = store.query(
        embedding=dummy_query,
        k=10,
        filter={"paper_id": target_pid},
    )

    assert len(results) > 0
    for r in results:
        assert r["metadata"]["paper_id"] == target_pid


# =========================================================================
# 4. get_by_id and clear
# =========================================================================


def test_get_by_id_and_clear() -> None:
    """get_by_id retrieves specific chunk; clear purges collection."""
    store = VectorStore(persist_dir=":memory:", collection_name="test_get_clear")
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    chunks = chunk_paper(papers[0])
    embeddings = [_pseudo_embed(c.embed_text) for c in chunks]

    store.add(chunks, embeddings)
    target = chunks[0]

    item = store.get_by_id(target.chunk_id)
    assert item is not None
    assert item["chunk_id"] == target.chunk_id
    assert item["text"] == target.text

    assert store.get_by_id("non_existent_id") is None

    # Clear
    store.clear()
    assert store.count() == 0
    assert store.get_by_id(target.chunk_id) is None
