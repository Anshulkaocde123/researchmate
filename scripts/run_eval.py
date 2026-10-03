#!/usr/bin/env python3
"""Run automated retrieval evaluation on the curated QASPER evaluation set.

Evaluates BM25, Chroma Dense, and Hybrid (RRF + Cross-Encoder) retrieval.
Produces reports/retrieval_eval.md with quantitative IR metrics.

Usage:
    python scripts/run_eval.py [--eval-path data/eval/eval_set.jsonl] [--limit 50]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from researchmate.config import settings
from researchmate.evaluation.evaluator import BenchmarkEvaluator
from researchmate.indexing.bm25_index import BM25Index
from researchmate.indexing.embedder import FastEmbedder
from researchmate.indexing.vector_store import VectorStore
from researchmate.retrieval.hybrid import HybridRetriever
from researchmate.retrieval.reranker import FastReranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_benchmark(
    eval_path: Path | None = None,
    report_path: Path = Path("reports/retrieval_eval.md"),
    limit: int | None = None,
) -> None:
    """Run retrieval benchmarks and write markdown report."""
    if eval_path is None:
        cand_gold = Path("data/eval/eval_with_gold_chunks.jsonl")
        eval_path = cand_gold if cand_gold.exists() else Path("data/eval/eval_set.jsonl")
    log.info("Loading indexes for evaluation...")
    bm25_path = settings.index_dir / "bm25.pkl.gz"
    bm25_index = BM25Index.load(bm25_path) if bm25_path.exists() else None

    vector_store = VectorStore()
    embedder = FastEmbedder()
    reranker = FastReranker()

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_index=bm25_index,
        reranker=reranker,
    )

    evaluator = BenchmarkEvaluator(retriever=retriever)
    eval_records = evaluator.load_eval_data(eval_path)
    if limit is not None:
        eval_records = eval_records[:limit]

    log.info("Running Sparse (BM25) evaluation across %d questions...", len(eval_records))
    sparse_summary = evaluator.evaluate_retrieval(eval_records, retrieval_mode="sparse")

    log.info("Running Hybrid (RRF + Rerank) evaluation across %d questions...", len(eval_records))
    hybrid_summary = evaluator.evaluate_retrieval(eval_records, retrieval_mode="hybrid")

    # Generate Markdown Report
    ks = [1, 3, 5, 10, 20]
    report_lines = [
        "# Retrieval Performance Evaluation Report",
        "",
        f"- **Evaluation Dataset:** `{eval_path}`",
        f"- **Questions Evaluated:** {sparse_summary['total_questions_evaluated']}",
        "- **Evaluation Scope:** Paper-scoped retrieval (strict paper_id filter)",
        "",
        "## Overall Metrics Comparison",
        "",
        "| Metric | BM25 (Sparse) | Hybrid (RRF + Rerank) |",
        "|---|:---:|:---:|",
    ]

    for k in ks:
        s_hr = sparse_summary["metrics"].get(f"HitRate@{k}", 0.0) * 100
        h_hr = hybrid_summary["metrics"].get(f"HitRate@{k}", 0.0) * 100
        report_lines.append(f"| **HitRate@{k}** | {s_hr:.1f}% | {h_hr:.1f}% |")

    for k in [5, 10]:
        s_mrr = sparse_summary["metrics"].get(f"MRR@{k}", 0.0)
        h_mrr = hybrid_summary["metrics"].get(f"MRR@{k}", 0.0)
        report_lines.append(f"| **MRR@{k}** | {s_mrr:.3f} | {h_mrr:.3f} |")

    for k in [5, 10]:
        s_ndcg = sparse_summary["metrics"].get(f"NDCG@{k}", 0.0)
        h_ndcg = hybrid_summary["metrics"].get(f"NDCG@{k}", 0.0)
        report_lines.append(f"| **NDCG@{k}** | {s_ndcg:.3f} | {h_ndcg:.3f} |")

    report_lines.extend(
        [
            "",
            "## Stratified Performance by Question Type (Hybrid @ k=5)",
            "",
            "| Question Type | Count | HitRate@5 | Recall@5 | MRR@5 |",
            "|---|:---:|:---:|:---:|:---:|",
        ]
    )

    for qtype, stats in hybrid_summary.get("stratified_by_type", {}).items():
        cnt = stats["count"]
        hr = stats["HitRate@5"] * 100
        rec = stats["Recall@5"] * 100
        mrr = stats["MRR@5"]
        report_lines.append(
            f"| {qtype.capitalize()} | {cnt} | {hr:.1f}% | {rec:.1f}% | {mrr:.3f} |"
        )

    report_lines.append("")
    report_content = "\n".join(report_lines)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    log.info("Evaluation report saved to %s", report_path)
    print("\n" + report_content)


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=Path("reports/retrieval_eval.md"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_benchmark(eval_path=args.eval_path, report_path=args.report_path, limit=args.limit)


if __name__ == "__main__":
    main()
