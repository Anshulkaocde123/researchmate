"""Embedding and vector-store indexing (Chroma)."""

from researchmate.indexing.embedder import (
    BGE_QUERY_PREFIX,
    EmbeddingCache,
    FastEmbedder,
    l2_normalize,
)

__all__ = [
    "BGE_QUERY_PREFIX",
    "EmbeddingCache",
    "FastEmbedder",
    "l2_normalize",
]
