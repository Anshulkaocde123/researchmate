"""Hybrid retriever combining Dense (Chroma) and Sparse (BM25) with Reciprocal Rank Fusion and Reranking."""

from __future__ import annotations

import logging
from typing import Any

from researchmate.config import settings
from researchmate.indexing.bm25_index import BM25Index
from researchmate.indexing.embedder import FastEmbedder
from researchmate.indexing.vector_store import VectorStore
from researchmate.retrieval.reranker import FastReranker

log = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k_rrf: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Merge ranked lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(doc) = sum_{system} (weight / (k_rrf + rank))
    where rank is 1-indexed.
    """
    merged: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        rrf_contrib = dense_weight / (k_rrf + rank)
        if cid not in merged:
            merged[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "metadata": item.get("metadata", {}),
                "dense_rank": rank,
                "sparse_rank": None,
                "dense_score": item.get("score"),
                "sparse_score": None,
                "rrf_score": rrf_contrib,
            }
        else:
            merged[cid]["dense_rank"] = rank
            merged[cid]["dense_score"] = item.get("score")
            merged[cid]["rrf_score"] += rrf_contrib

    for rank, item in enumerate(sparse_results, start=1):
        cid = item["chunk_id"]
        rrf_contrib = sparse_weight / (k_rrf + rank)
        if cid not in merged:
            merged[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "metadata": item.get("metadata", {}),
                "dense_rank": None,
                "sparse_rank": rank,
                "dense_score": None,
                "sparse_score": item.get("score"),
                "rrf_score": rrf_contrib,
            }
        else:
            merged[cid]["sparse_rank"] = rank
            merged[cid]["sparse_score"] = item.get("score")
            merged[cid]["rrf_score"] += rrf_contrib

    fused = list(merged.values())
    for item in fused:
        item["score"] = item["rrf_score"]

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


class HybridRetriever:
    """Hybrid Retriever coordinating Dense, Sparse, RRF, and Cross-Encoder Reranking."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: FastEmbedder | None = None,
        bm25_index: BM25Index | None = None,
        reranker: FastReranker | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or FastEmbedder()
        self.bm25_index = bm25_index
        self.reranker = reranker or FastReranker()

    def retrieve_dense(
        self,
        query: str,
        paper_id: str | None = None,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve candidates using Chroma vector similarity."""
        top_k = k or settings.top_k_dense
        query_emb = self.embedder.embed_query(query)
        filt = {"paper_id": paper_id} if paper_id else None
        return self.vector_store.query(embedding=query_emb, k=top_k, filter=filt)

    def retrieve_sparse(
        self,
        query: str,
        paper_id: str | None = None,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve candidates using BM25 sparse keyword matching."""
        if self.bm25_index is None:
            return []
        top_k = k or settings.top_k_sparse
        filt = {"paper_id": paper_id} if paper_id else None
        return self.bm25_index.query(query=query, k=top_k, filter=filt)

    def retrieve(
        self,
        query: str,
        paper_id: str | None = None,
        top_k: int | None = None,
        rerank: bool = True,
        k_dense: int | None = None,
        k_sparse: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval: Dense + Sparse -> RRF fusion -> Cross-Encoder Rerank.

        Args:
            query: Question or search query.
            paper_id: Optional paper ID filter.
            top_k: Number of final chunks to return (default: settings.top_k_rerank).
            rerank: If True, apply cross-encoder reranking. If False, return top RRF chunks.
            k_dense: Dense candidate count (default: settings.top_k_dense).
            k_sparse: Sparse candidate count (default: settings.top_k_sparse).

        Returns:
            List of ranked chunk dicts.
        """
        final_k = top_k or settings.top_k_rerank

        # 1. Fetch dense candidates
        dense_results = self.retrieve_dense(query=query, paper_id=paper_id, k=k_dense)

        # 2. Fetch sparse candidates
        sparse_results = self.retrieve_sparse(query=query, paper_id=paper_id, k=k_sparse)

        # 3. Fuse with RRF
        if dense_results and sparse_results:
            candidates = reciprocal_rank_fusion(dense_results, sparse_results)
        elif dense_results:
            candidates = dense_results
        elif sparse_results:
            candidates = sparse_results
        else:
            return []

        # 4. Cross-encoder rerank
        if rerank:
            return self.reranker.rerank(query=query, chunks=candidates, top_k=final_k)

        return candidates[:final_k]
