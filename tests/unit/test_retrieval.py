"""Unit tests for FastReranker and HybridRetriever with mock components."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from researchmate.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from researchmate.retrieval.reranker import FastReranker


class FakeCrossEncoder:
    """Mock cross encoder that scores docs by query token overlap."""

    def rerank(self, query: str, documents: list[str], **kwargs: Any) -> list[float]:
        q_tokens = set(query.lower().split())
        scores = []
        for doc in documents:
            d_tokens = set(doc.lower().split())
            overlap = len(q_tokens & d_tokens)
            scores.append(float(overlap) / (len(q_tokens) or 1.0))
        return scores


def test_fast_reranker() -> None:
    """FastReranker scores documents and returns top_k sorted descending."""
    fake_model = FakeCrossEncoder()
    reranker = FastReranker(model=fake_model)

    chunks = [
        {"chunk_id": "c1", "text": "neural computation and logic", "metadata": {"title": "P1"}},
        {"chunk_id": "c2", "text": "apples bananas fruit salad", "metadata": {"title": "P2"}},
        {
            "chunk_id": "c3",
            "text": "artificial neural networks deep learning",
            "metadata": {"title": "P3"},
        },
    ]

    res = reranker.rerank("neural networks", chunks, top_k=2)
    assert len(res) == 2
    assert res[0]["chunk_id"] == "c3"
    assert res[0]["score"] > res[1]["score"]
    assert "rerank_score" in res[0]

    # Empty inputs
    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", chunks, top_k=0) == []


def test_reciprocal_rank_fusion() -> None:
    """RRF correctly fuses dense and sparse rankings."""
    dense = [
        {"chunk_id": "A", "text": "Doc A", "score": 0.9},
        {"chunk_id": "B", "text": "Doc B", "score": 0.8},
        {"chunk_id": "C", "text": "Doc C", "score": 0.7},
    ]
    sparse = [
        {"chunk_id": "B", "text": "Doc B", "score": 10.0},
        {"chunk_id": "A", "text": "Doc A", "score": 8.0},
        {"chunk_id": "D", "text": "Doc D", "score": 5.0},
    ]

    fused = reciprocal_rank_fusion(dense, sparse, k_rrf=60)
    assert len(fused) == 4  # A, B, C, D

    # Doc A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # Doc B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.03252
    # Both A and B are top 2
    top_cids = {fused[0]["chunk_id"], fused[1]["chunk_id"]}
    assert top_cids == {"A", "B"}

    # Doc C: 1/(60+3) = 0.01587
    # Doc D: 1/(60+3) = 0.01587
    bottom_cids = {fused[2]["chunk_id"], fused[3]["chunk_id"]}
    assert bottom_cids == {"C", "D"}


def test_hybrid_retriever_mocked() -> None:
    """HybridRetriever integrates dense, sparse, RRF, and reranking."""
    mock_vs = MagicMock()
    mock_embedder = MagicMock()
    mock_bm25 = MagicMock()
    fake_model = FakeCrossEncoder()
    reranker = FastReranker(model=fake_model)

    mock_embedder.embed_query.return_value = [0.1, 0.2]
    mock_vs.query.return_value = [
        {"chunk_id": "c1", "text": "neural networks deep learning", "score": 0.9, "metadata": {}},
        {"chunk_id": "c2", "text": "computer vision images", "score": 0.8, "metadata": {}},
    ]
    mock_bm25.query.return_value = [
        {"chunk_id": "c1", "text": "neural networks deep learning", "score": 12.0, "metadata": {}},
        {"chunk_id": "c3", "text": "natural language processing", "score": 8.0, "metadata": {}},
    ]

    retriever = HybridRetriever(
        vector_store=mock_vs,
        embedder=mock_embedder,
        bm25_index=mock_bm25,
        reranker=reranker,
    )

    # 1. Test retrieve_dense
    dense = retriever.retrieve_dense("neural networks", paper_id="123")
    assert len(dense) == 2
    mock_vs.query.assert_called_with(embedding=[0.1, 0.2], k=20, filter={"paper_id": "123"})

    # 2. Test retrieve_sparse
    sparse = retriever.retrieve_sparse("neural networks", paper_id="123")
    assert len(sparse) == 2
    mock_bm25.query.assert_called_with(query="neural networks", k=20, filter={"paper_id": "123"})

    # 3. Test full hybrid retrieve with reranking
    results = retriever.retrieve("neural networks", paper_id="123", top_k=2)
    assert len(results) == 2
    # c1 has higher token overlap with "neural networks" than c2 or c3
    assert results[0]["chunk_id"] == "c1"
