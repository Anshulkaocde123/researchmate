"""Cross-encoder reranker for ResearchMate retrieval candidates using fastembed.

Uses ONNX-backed cross-encoder (default: BAAI/bge-reranker-base).
Lazy-loaded to keep import times fast, with dependency injection support for unit testing.
"""

from __future__ import annotations

import logging
from typing import Any

from researchmate.config import settings

log = logging.getLogger(__name__)


class FastReranker:
    """Reranker wrapping FastEmbed TextCrossEncoder."""

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = model

    @property
    def model(self) -> Any:
        """Lazy load the cross-encoder model on first access."""
        if self._model is None:
            log.info("Loading FastEmbed cross-encoder: %s", self.model_name)
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
        batch_size: int = 64,
    ) -> list[dict[str, Any]]:
        """Rerank candidate chunks using the cross-encoder.

        Args:
            query: The search query string.
            chunks: List of candidate chunk dicts, each having at least 'text' or 'embed_text'.
            top_k: Number of highest-scoring chunks to return. If None, returns all.
            batch_size: Batch size for model inference.

        Returns:
            List of chunk dicts sorted by rerank score descending, with 'rerank_score'
            and updated 'score'.
        """
        if not chunks or (top_k is not None and top_k <= 0):
            return []

        # Prepare text representation for each candidate
        doc_texts: list[str] = []
        for c in chunks:
            meta = c.get("metadata", {})
            title = meta.get("title", "")
            section = meta.get("section", "")
            text = c.get("text", "")
            if title and section:
                context_text = f"[{title} | {section}]\n{text}"
            elif title:
                context_text = f"[{title}]\n{text}"
            else:
                context_text = text
            doc_texts.append(context_text)

        # Compute cross-encoder scores
        raw_scores = list(
            self.model.rerank(query=query, documents=doc_texts, batch_size=batch_size)
        )

        scored_chunks: list[dict[str, Any]] = []
        for orig, score in zip(chunks, raw_scores, strict=True):
            entry = dict(orig)
            entry["rerank_score"] = float(score)
            entry["dense_or_sparse_score"] = entry.get("score")
            entry["score"] = float(score)  # update primary score
            scored_chunks.append(entry)

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        if top_k is not None:
            return scored_chunks[:top_k]
        return scored_chunks
