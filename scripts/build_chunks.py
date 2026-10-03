#!/usr/bin/env python3
"""Build chunk corpus and map gold evaluation evidence to chunks.

1. Chunks all papers into ``data/processed/chunks.jsonl``.
2. Loads ``data/eval/eval_set.jsonl``, maps all answer evidence to gold chunk IDs,
   and writes ``data/eval/eval_with_gold_chunks.jsonl``.
3. Generates ``reports/evidence_mapping.md`` documenting empirical match rates
   and ungroundable float (table/figure) evidence examples.

Usage:
    python scripts/build_chunks.py [--data-dir data/raw] [--out-chunks data/processed/chunks.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchmate.chunking.chunker import chunk_paper
from researchmate.chunking.evidence import map_evidence_to_chunks
from researchmate.data.loader import load_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_corpus_and_evidence(
    raw_dir: Path = Path("data/raw"),
    eval_set_path: Path = Path("data/eval/eval_set.jsonl"),
    chunks_out_path: Path = Path("data/processed/chunks.jsonl"),
    eval_out_path: Path = Path("data/eval/eval_with_gold_chunks.jsonl"),
    report_path: Path = Path("reports/evidence_mapping.md"),
    max_tokens: int = 400,
    overlap_paragraphs: int = 1,
) -> dict[str, Any]:
    """Execute chunking pipeline and gold evidence mapping."""
    log.info("Loading papers from %s ...", raw_dir)

    all_papers = []
    papers_by_id = {}
    for split in ("train", "validation", "test"):
        split_file = raw_dir / f"{split}.jsonl"
        if split_file.exists():
            papers, _ = load_papers(split_file)
            for p in papers:
                all_papers.append(p)
                papers_by_id[p.paper_id] = p

    log.info("Loaded %d papers total. Generating chunks ...", len(all_papers))

    chunks_out_path.parent.mkdir(parents=True, exist_ok=True)
    paper_chunks_map: dict[str, list[Any]] = {}
    total_chunks = 0

    with chunks_out_path.open("w", encoding="utf-8") as fh:
        for paper in all_papers:
            chunks = chunk_paper(
                paper=paper,
                max_tokens=max_tokens,
                overlap_paragraphs=overlap_paragraphs,
            )
            paper_chunks_map[paper.paper_id] = chunks
            total_chunks += len(chunks)
            for c in chunks:
                fh.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

    log.info("Wrote %d chunks to %s", total_chunks, chunks_out_path)

    # ── Map Evaluation Evidence ────────────────────────────────────────────
    if not eval_set_path.exists():
        raise FileNotFoundError(f"Evaluation set missing at: {eval_set_path}")

    log.info("Mapping gold evidence for %s ...", eval_set_path)
    eval_questions: list[dict[str, Any]] = [
        json.loads(line)
        for line in eval_set_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    total_evidence_strings = 0
    matched_evidence_strings = 0
    unmatched_records: list[dict[str, Any]] = []
    enriched_eval_items: list[dict[str, Any]] = []

    for item in eval_questions:
        pid = item["paper_id"]
        paper = papers_by_id.get(pid)
        chunks = paper_chunks_map.get(pid, [])
        all_item_gold_chunks: set[str] = set()

        enriched_answers = []
        for ans in item.get("answers", []):
            evidence_list = ans.get("evidence", [])
            if paper and evidence_list:
                mapping = map_evidence_to_chunks(
                    evidence_strings=evidence_list,
                    chunks=chunks,
                    paragraphs=paper.paragraphs,
                )
                gold_cids = mapping["gold_chunk_ids"]
                all_item_gold_chunks.update(gold_cids)

                for d in mapping["details"]:
                    total_evidence_strings += 1
                    if d["matched"]:
                        matched_evidence_strings += 1
                    else:
                        unmatched_records.append(
                            {
                                "paper_id": pid,
                                "question_id": item["question_id"],
                                "question": item["question"],
                                "evidence": d["evidence"],
                                "reason": d["unmatched_reason"],
                            }
                        )

                ans_copy = dict(ans)
                ans_copy["gold_chunk_ids"] = gold_cids
                enriched_answers.append(ans_copy)
            else:
                ans_copy = dict(ans)
                ans_copy["gold_chunk_ids"] = []
                enriched_answers.append(ans_copy)

        item_copy = dict(item)
        item_copy["answers"] = enriched_answers
        item_copy["all_gold_chunk_ids"] = sorted(all_item_gold_chunks)
        enriched_eval_items.append(item_copy)

    eval_out_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_out_path.open("w", encoding="utf-8") as fh:
        for item in enriched_eval_items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    match_rate = (
        (matched_evidence_strings / total_evidence_strings * 100) if total_evidence_strings else 0.0
    )
    log.info(
        "Mapped evidence: %d / %d (%.1f%% match rate)",
        matched_evidence_strings,
        total_evidence_strings,
        match_rate,
    )
    log.info("Saved enriched eval set to %s", eval_out_path)

    # ── Write Markdown Report ──────────────────────────────────────────────
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# QASPER Gold Evidence to Chunk Mapping Report")
    md.append("")
    md.append(f"> Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    md.append(f"> Corpus: {total_chunks:,} chunks generated across {len(all_papers):,} papers.")
    md.append("")
    md.append("## 1. Evidence Mapping Performance")
    md.append("")
    md.append("| Metric | Count / Value |")
    md.append("|---|---:|")
    md.append(f"| **Evaluation Questions** | {len(eval_questions):,} |")
    md.append(f"| **Total Evidence Strings** | {total_evidence_strings:,} |")
    md.append(f"| **Successfully Grounded Evidence** | {matched_evidence_strings:,} |")
    md.append(f"| **Overall Match Rate** | **{match_rate:.1f}%** |")
    md.append(f"| **Unmatched Evidence Strings** | {len(unmatched_records):,} |")
    md.append("")
    md.append("## 2. Root Cause Analysis of Unmatched Evidence")
    md.append("")
    md.append(
        "In QASPER, annotators had the option to highlight float objects (tables and figures) "
        "as evidence using the annotation tool. These appear in raw annotations prefixed with "
        "`FLOAT SELECTED: Table ...` or `FLOAT SELECTED: Figure ...`. Because ResearchMate's "
        "paragraph chunker specifically extracts textual body paragraphs, ungrounded table and "
        "figure captions remain unmatched as expected."
    )
    md.append("")
    md.append("## 3. Sample Unmatched Evidence Examples")
    md.append("")
    for i, ex in enumerate(unmatched_records[:5], 1):
        md.append(f"### Example {i}")
        md.append(f"- **Paper ID:** `{ex['paper_id']}`")
        md.append(f'- **Question:** *"{ex["question"]}"*')
        md.append(f"- **Reason:** `{ex['reason']}`")
        md.append(f"- **Evidence Text:**\n  > {ex['evidence']}")
        md.append("")

    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("Wrote evidence mapping report to %s", report_path)

    return {
        "total_chunks": total_chunks,
        "total_evidence": total_evidence_strings,
        "matched_evidence": matched_evidence_strings,
        "match_rate": match_rate,
        "unmatched_count": len(unmatched_records),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.jsonl"))
    parser.add_argument("--out-chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument(
        "--out-eval", type=Path, default=Path("data/eval/eval_with_gold_chunks.jsonl")
    )
    parser.add_argument("--report-path", type=Path, default=Path("reports/evidence_mapping.md"))
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--overlap-paragraphs", type=int, default=1)
    args = parser.parse_args()

    build_corpus_and_evidence(
        raw_dir=args.data_dir,
        eval_set_path=args.eval_set,
        chunks_out_path=args.out_chunks,
        eval_out_path=args.out_eval,
        report_path=args.report_path,
        max_tokens=args.max_tokens,
        overlap_paragraphs=args.overlap_paragraphs,
    )


if __name__ == "__main__":
    main()
