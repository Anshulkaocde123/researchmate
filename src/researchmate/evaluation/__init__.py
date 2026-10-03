"""Evaluation metrics and benchmarking suite."""

from researchmate.evaluation.evaluator import BenchmarkEvaluator
from researchmate.evaluation.generation_metrics import (
    citation_metrics,
    exact_match_score,
    f1_score,
    normalize_answer,
)
from researchmate.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

__all__ = [
    "BenchmarkEvaluator",
    "citation_metrics",
    "exact_match_score",
    "f1_score",
    "hit_rate_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "normalize_answer",
    "recall_at_k",
]
