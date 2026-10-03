"""Unit tests for FastEmbedder, query instruction prefixes, and disk cache.

Tests use a FakeEmbeddingModel fixture to ensure NO network calls or model downloads
occur during unit tests.

Checks:
- Output shape and dimensionality
- Strict unit L2 normalization: ||v||_2 == 1.0
- Determinism across repeated calls
- Query instruction prefix: applied to queries only and exactly once
- Document texts remain unprefixed
- On-disk SQLite caching skips re-computation
- Partial cache hit resolution
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from researchmate.indexing.embedder import (
    BGE_QUERY_PREFIX,
    EmbeddingCache,
    FastEmbedder,
    l2_normalize,
)


class FakeEmbeddingModel:
    """Deterministic mock embedding model without external network dependencies."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.embedding_size = dim
        self.call_count = 0
        self.last_inputs: list[str] = []

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate deterministic pseudo-embeddings from text MD5 hashes."""
        self.call_count += len(texts)
        self.last_inputs = list(texts)
        embeddings = []
        for t in texts:
            seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            raw = rng.standard_normal(self.dim)
            embeddings.append(raw)
        return embeddings


# =========================================================================
# 1. Output Shape and Unit Norm
# =========================================================================


def test_embed_documents_shape_and_unit_norm() -> None:
    """Document embeddings have expected dimension and unit L2 norm."""
    mock = FakeEmbeddingModel(dim=32)
    embedder = FastEmbedder(
        model_name="mock-bge",
        cache_path=":memory:",
        embedding_model=mock,
    )

    docs = ["First document text.", "Second document text with more words."]
    vectors = embedder.embed_documents(docs)

    assert len(vectors) == 2
    assert len(vectors[0]) == 32
    assert len(vectors[1]) == 32

    # L2 norm check
    norm0 = float(np.linalg.norm(vectors[0]))
    norm1 = float(np.linalg.norm(vectors[1]))
    assert np.isclose(norm0, 1.0, atol=1e-5)
    assert np.isclose(norm1, 1.0, atol=1e-5)


def test_embed_query_shape_and_unit_norm() -> None:
    """Query embedding has expected dimension and unit L2 norm."""
    mock = FakeEmbeddingModel(dim=48)
    embedder = FastEmbedder(
        model_name="mock-bge",
        cache_path=":memory:",
        embedding_model=mock,
    )

    vec = embedder.embed_query("What is the main finding?")
    assert len(vec) == 48
    norm = float(np.linalg.norm(vec))
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_l2_normalize_handles_zero_vector() -> None:
    """l2_normalize safely handles zero vectors without dividing by zero."""
    zero_vec = np.zeros((1, 16), dtype=np.float32)
    normed = l2_normalize(zero_vec)
    assert normed.shape == (1, 16)
    assert not np.isnan(normed).any()


# =========================================================================
# 2. Query Prefix Behavior
# =========================================================================


def test_query_prefix_applied_to_queries_only() -> None:
    """Queries receive BGE_QUERY_PREFIX, while documents remain unprefixed."""
    mock = FakeEmbeddingModel(dim=16)
    embedder = FastEmbedder(
        model_name="mock-bge",
        cache_path=":memory:",
        embedding_model=mock,
    )

    # Embed document
    embedder.embed_documents(["Common text"])
    assert mock.last_inputs == ["Common text"]
    assert not mock.last_inputs[0].startswith(BGE_QUERY_PREFIX)

    # Embed query
    embedder.embed_query("Common text")
    assert mock.last_inputs == [f"{BGE_QUERY_PREFIX}Common text"]
    assert mock.last_inputs[0].startswith(BGE_QUERY_PREFIX)


def test_query_prefix_not_duplicated() -> None:
    """If query already starts with prefix, it is applied exactly once."""
    mock = FakeEmbeddingModel(dim=16)
    embedder = FastEmbedder(
        model_name="mock-bge",
        cache_path=":memory:",
        embedding_model=mock,
    )

    already_prefixed = f"{BGE_QUERY_PREFIX}How does it work?"
    formatted = embedder.format_query(already_prefixed)
    assert formatted == already_prefixed
    assert formatted.count(BGE_QUERY_PREFIX) == 1

    embedder.embed_query(already_prefixed)
    assert mock.last_inputs[0] == already_prefixed
    assert mock.last_inputs[0].count(BGE_QUERY_PREFIX) == 1


# =========================================================================
# 3. Determinism and Embedding Differentiation
# =========================================================================


def test_embedder_determinism() -> None:
    """Same input produces identical embeddings across repeated calls."""
    mock = FakeEmbeddingModel(dim=32)
    embedder = FastEmbedder(
        model_name="mock-bge",
        cache_path=":memory:",
        embedding_model=mock,
    )

    text = "Reproducible test passage."
    v1 = embedder.embed_documents([text])[0]
    v2 = embedder.embed_documents([text])[0]
    assert v1 == v2

    q1 = embedder.embed_query("same query")
    q2 = embedder.embed_query("same query")
    assert q1 == q2


def test_query_and_document_embeddings_differ_for_same_text() -> None:
    """Query and document embeddings of the identical text differ due to query prefix."""
    mock = FakeEmbeddingModel(dim=32)
    embedder = FastEmbedder(
        model_name="mock-bge",
        cache_path=":memory:",
        embedding_model=mock,
    )

    phrase = "Identical phrase for testing."
    doc_vec = embedder.embed_documents([phrase])[0]
    query_vec = embedder.embed_query(phrase)

    # Because query has instruction prefix, underlying input to model differs
    assert doc_vec != query_vec


# =========================================================================
# 4. On-Disk Caching Skips Recomputation
# =========================================================================


def test_caching_skips_recomputation(tmp_path: Path) -> None:
    """Subsequent calls for the same text hit cache and skip model calls."""
    db_file = tmp_path / "embed_cache.sqlite"
    mock = FakeEmbeddingModel(dim=16)
    embedder = FastEmbedder(
        model_name="mock-bge",
        cache_path=db_file,
        embedding_model=mock,
    )

    docs = ["Apple", "Banana", "Cherry"]

    # First call: computes 3 embeddings
    assert mock.call_count == 0
    res1 = embedder.embed_documents(docs)
    assert mock.call_count == 3

    # Second call: all 3 should hit cache; call_count stays 3
    res2 = embedder.embed_documents(docs)
    assert mock.call_count == 3
    assert res1 == res2

    # Query caching
    assert embedder.cache is not None
    q_res1 = embedder.embed_query("What is fruit?")
    assert mock.call_count == 4
    q_res2 = embedder.embed_query("What is fruit?")
    assert mock.call_count == 4
    assert q_res1 == q_res2


def test_partial_cache_hits(tmp_path: Path) -> None:
    """Only uncached items are passed to the model when batch is partially cached."""
    db_file = tmp_path / "partial_cache.sqlite"
    mock = FakeEmbeddingModel(dim=16)
    embedder = FastEmbedder(
        model_name="mock-bge",
        cache_path=db_file,
        embedding_model=mock,
    )

    # Prime cache with "A" and "B"
    embedder.embed_documents(["A", "B"])
    assert mock.call_count == 2

    # Query with ["A", "C", "B", "D"] -> only "C" and "D" should be encoded
    results = embedder.embed_documents(["A", "C", "B", "D"])
    assert len(results) == 4
    # Call count should increase by exactly 2 (for C and D)
    assert mock.call_count == 4


def test_cache_persistence_across_instances(tmp_path: Path) -> None:
    """A new FastEmbedder instance can load embeddings cached by a previous instance."""
    db_file = tmp_path / "persistent_cache.sqlite"
    mock1 = FakeEmbeddingModel(dim=16)
    embedder1 = FastEmbedder(
        model_name="mock-bge",
        cache_path=db_file,
        embedding_model=mock1,
    )

    vec1 = embedder1.embed_documents(["Persistent doc"])[0]
    assert mock1.call_count == 1

    # Second instance sharing the same db
    mock2 = FakeEmbeddingModel(dim=16)
    embedder2 = FastEmbedder(
        model_name="mock-bge",
        cache_path=db_file,
        embedding_model=mock2,
    )

    vec2 = embedder2.embed_documents(["Persistent doc"])[0]
    assert mock2.call_count == 0  # Re-computation skipped!
    assert vec1 == vec2


def test_embedding_cache_direct_operations() -> None:
    """EmbeddingCache get, set_batch, count, and clear methods function properly."""
    cache = EmbeddingCache(":memory:")
    assert cache.count() == 0

    cache.set_batch({"k1": [0.1, 0.2], "k2": [0.3, 0.4]})
    assert cache.count() == 2

    val1 = cache.get("k1")
    assert val1 is not None and np.isclose(val1, [0.1, 0.2]).all()

    batch_vals = cache.get_batch(["k1", "k2", "k3"])
    assert len(batch_vals) == 2
    assert "k1" in batch_vals and "k2" in batch_vals

    cache.clear()
    assert cache.count() == 0
    assert cache.get("k1") is None


def test_embedder_empty_input_and_dim() -> None:
    """Empty list returns empty embeddings; embedding_dim reflects model dim."""
    mock = FakeEmbeddingModel(dim=24)
    embedder = FastEmbedder(
        model_name="mock-bge",
        cache_path=":memory:",
        embedding_model=mock,
    )
    assert embedder.embed_documents([]) == []
    assert embedder.embedding_dim == 24
