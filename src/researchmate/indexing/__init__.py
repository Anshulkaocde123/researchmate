"""Embedding and vector-store indexing (Chroma)."""

from researchmate.indexing.embedder import (
    BGE_QUERY_PREFIX,
    EmbeddingCache,
    FastEmbedder,
    l2_normalize,
)
from researchmate.indexing.vector_store import VectorStore

__all__ = [
    "BGE_QUERY_PREFIX",
    "EmbeddingCache",
    "FastEmbedder",
    "VectorStore",
    "l2_normalize",
]
