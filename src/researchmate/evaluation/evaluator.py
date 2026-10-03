"""Benchmark evaluator for ResearchMate retrieval and RAG generation."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from researchmate.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from researchmate.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluates retrieval and generation quality against curated eval sets."""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def load_eval_data(self, eval_path: Path | str) -> list[dict[str, Any]]:
        """Load evaluation records from a JSONL file."""
        records = []
        with Path(eval_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def evaluate_retrieval(
        self,
        eval_records: list[dict[str, Any]],
        ks: list[int] | None = None,
        retrieval_mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Evaluate retrieval performance across evaluation questions.

        Args:
            eval_records: List of eval items containing 'question', 'paper_id', 'gold_chunk_ids'.
            ks: Cutoff values for metrics (default: [1, 3, 5, 10, 20]).
            retrieval_mode: 'hybrid', 'sparse', or 'dense'.

        Returns:
            Dictionary containing overall and stratified metrics.
        """
        if ks is None:
            ks = [1, 3, 5, 10, 20]

        total_evaluated = 0
        hit_rates: dict[int, list[float]] = {k: [] for k in ks}
        recalls: dict[int, list[float]] = {k: [] for k in ks}
        mrrs: dict[int, list[float]] = {k: [] for k in ks}
        ndcgs: dict[int, list[float]] = {k: [] for k in ks}

        by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for item in eval_records:
            gold_chunk_ids = item.get("all_gold_chunk_ids") or item.get("gold_chunk_ids") or []
            # Questions with no gold chunks (e.g. unanswerable or unmapped) are skipped for retrieval evaluation
            if not gold_chunk_ids:
                continue

            question = item["question"]
            paper_id = item.get("paper_id")
            ans_type = item.get("category") or item.get("answer_type", "unknown")

            if retrieval_mode == "sparse":
                results = self.retriever.retrieve_sparse(
                    query=question, paper_id=paper_id, k=max(ks)
                )
            elif retrieval_mode == "dense":
                results = self.retriever.retrieve_dense(
                    query=question, paper_id=paper_id, k=max(ks)
                )
            else:
                results = self.retriever.retrieve(
                    query=question,
                    paper_id=paper_id,
                    top_k=max(ks),
                    rerank=True,
                )

            retrieved_ids = [r["chunk_id"] for r in results]
            total_evaluated += 1

            for k in ks:
                hr = hit_rate_at_k(retrieved_ids, gold_chunk_ids, k=k)
                rec = recall_at_k(retrieved_ids, gold_chunk_ids, k=k)
                mrr = mrr_at_k(retrieved_ids, gold_chunk_ids, k=k)
                ndcg = ndcg_at_k(retrieved_ids, gold_chunk_ids, k=k)

                hit_rates[k].append(hr)
                recalls[k].append(rec)
                mrrs[k].append(mrr)
                ndcgs[k].append(ndcg)

                if k == 5:
                    by_type[ans_type]["hit_rate_5"].append(hr)
                    by_type[ans_type]["recall_5"].append(rec)
                    by_type[ans_type]["mrr_5"].append(mrr)

        summary = {
            "total_questions_evaluated": total_evaluated,
            "mode": retrieval_mode,
            "metrics": {},
            "stratified_by_type": {},
        }

        for k in ks:
            summary["metrics"][f"HitRate@{k}"] = (
                sum(hit_rates[k]) / total_evaluated if total_evaluated else 0.0
            )
            summary["metrics"][f"Recall@{k}"] = (
                sum(recalls[k]) / total_evaluated if total_evaluated else 0.0
            )
            summary["metrics"][f"MRR@{k}"] = (
                sum(mrrs[k]) / total_evaluated if total_evaluated else 0.0
            )
            summary["metrics"][f"NDCG@{k}"] = (
                sum(ndcgs[k]) / total_evaluated if total_evaluated else 0.0
            )

        for ans_type, vals in by_type.items():
            summary["stratified_by_type"][ans_type] = {
                "count": len(vals["hit_rate_5"]),
                "HitRate@5": sum(vals["hit_rate_5"]) / len(vals["hit_rate_5"]),
                "Recall@5": sum(vals["recall_5"]) / len(vals["recall_5"]),
                "MRR@5": sum(vals["mrr_5"]) / len(vals["mrr_5"]),
            }

        return summary
