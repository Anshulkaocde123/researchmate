"""Unit tests for retrieval and generation evaluation metrics."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from researchmate.evaluation.evaluator import BenchmarkEvaluator
from researchmate.evaluation.generation_metrics import citation_metrics, exact_match_score, f1_score
from researchmate.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_retrieval_metrics() -> None:
    """HitRate, Recall, MRR, and NDCG calculate accurately."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    gold = {"c2", "c4"}

    # HitRate
    assert hit_rate_at_k(retrieved, gold, k=1) == 0.0
    assert hit_rate_at_k(retrieved, gold, k=2) == 1.0
    assert hit_rate_at_k(retrieved, gold, k=5) == 1.0
    assert hit_rate_at_k(retrieved, set(), k=5) == 0.0

    # Recall
    assert recall_at_k(retrieved, gold, k=1) == 0.0
    assert recall_at_k(retrieved, gold, k=2) == 0.5  # found 1 of 2
    assert recall_at_k(retrieved, gold, k=4) == 1.0  # found 2 of 2

    # MRR
    assert mrr_at_k(retrieved, gold, k=1) == 0.0
    assert mrr_at_k(retrieved, gold, k=2) == 0.5  # first hit at rank 2
    assert mrr_at_k(retrieved, {"c1"}, k=5) == 1.0
    assert mrr_at_k(retrieved, {"c99"}, k=5) == 0.0

    # NDCG
    assert ndcg_at_k(retrieved, {"c1"}, k=1) == 1.0
    assert ndcg_at_k(retrieved, {"c99"}, k=5) == 0.0
    ndcg_val = ndcg_at_k(retrieved, gold, k=5)
    assert 0.0 < ndcg_val < 1.0


def test_generation_metrics() -> None:
    """EM and F1 handle text normalization, punctuation, and partial overlap."""
    pred = "The BERT model achieves 93.2% accuracy."
    gold = "bert model achieves 93.2 accuracy"

    assert exact_match_score(pred, gold) == 1.0
    assert f1_score(pred, gold) == 1.0

    # Partial match
    partial_pred = "BERT achieves accuracy"
    f1 = f1_score(partial_pred, gold)
    assert 0.0 < f1 < 1.0

    # Completely disjoint
    disjoint_pred = "Random text here"
    assert f1_score(disjoint_pred, gold) == 0.0
    assert exact_match_score(disjoint_pred, gold) == 0.0


def test_citation_metrics() -> None:
    """Citation precision, recall, and F1 calculations."""
    cited = ["c1", "c2"]
    gold = {"c2", "c3"}

    res = citation_metrics(cited, gold)
    assert np.isclose(res["precision"], 0.5)  # 1 of 2 cited is gold
    assert np.isclose(res["recall"], 0.5)  # 1 of 2 gold is cited
    assert np.isclose(res["f1"], 0.5)

    # Perfect citation
    perfect = citation_metrics(["c1", "c2"], {"c1", "c2"})
    assert perfect["f1"] == 1.0

    # Zero cited
    empty_cited = citation_metrics([], {"c1"})
    assert empty_cited["f1"] == 0.0


def test_benchmark_evaluator_mocked() -> None:
    """BenchmarkEvaluator runs across records and produces summary metrics."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve_sparse.return_value = [
        {"chunk_id": "c1", "text": "t1"},
        {"chunk_id": "c2", "text": "t2"},
    ]
    mock_retriever.retrieve.return_value = [
        {"chunk_id": "c2", "text": "t2"},
        {"chunk_id": "c1", "text": "t1"},
    ]

    evaluator = BenchmarkEvaluator(retriever=mock_retriever)
    eval_records = [
        {
            "question_id": "q1",
            "paper_id": "p1",
            "question": "What is the result?",
            "answer_type": "extractive",
            "gold_chunk_ids": ["c2"],
        }
    ]

    summary = evaluator.evaluate_retrieval(eval_records, ks=[1, 2], retrieval_mode="sparse")
    assert summary["total_questions_evaluated"] == 1
    assert summary["metrics"]["HitRate@1"] == 0.0
    assert summary["metrics"]["HitRate@2"] == 1.0
    assert summary["metrics"]["MRR@2"] == 0.5
