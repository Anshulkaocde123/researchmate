#!/usr/bin/env python3
"""Dataset profiling and deterministic evaluation set generator.

Analyzes train, validation, and test splits of QASPER, checks dataset integrity
with assertions (duplicate IDs, empty papers), outputs reports/data_profile.md,
and creates the stratified 200-question evaluation set at data/eval/eval_set.jsonl.

Usage:
    python scripts/profile_data.py [--data-dir data/raw] [--eval-dir data/eval] [--report-path reports/data_profile.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchmate.data.sampler import categorize_qa, sample_stratified_qa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """Fast, reproducible token count based on whitespace separation."""
    return len(text.split())


def profile_split(path: Path) -> dict[str, Any]:
    """Profile one JSONL split file and perform dataset integrity assertions."""
    if not path.exists():
        raise FileNotFoundError(f"Split file does not exist: {path}")

    paper_ids: list[str] = []
    question_ids: list[str] = []
    paras_per_paper: list[int] = []
    para_token_lengths: list[int] = []
    questions_per_paper: list[int] = []
    empty_papers: list[str] = []
    q_categories: Counter[str] = Counter()

    # Per-annotation breakdown
    total_answers = 0
    ans_unans = 0
    ans_yesno = 0
    ans_extractive = 0
    ans_freeform = 0

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            pid = paper.get("paper_id", f"line_{lineno}")
            paper_ids.append(pid)

            # Count non-empty paragraphs
            paras = []
            for sec in paper.get("full_text", []):
                for p in sec.get("paragraphs", []):
                    if isinstance(p, str) and p.strip():
                        paras.append(p)

            if len(paras) == 0:
                empty_papers.append(pid)
            else:
                for p in paras:
                    para_token_lengths.append(count_tokens(p))

            paras_per_paper.append(len(paras))

            # Questions
            qas = paper.get("qas", [])
            questions_per_paper.append(len(qas))

            for qa in qas:
                qid = qa.get("question_id", "")
                question_ids.append(qid)
                cat = categorize_qa(qa)
                q_categories[cat] += 1

                for annotator in qa.get("answers", []):
                    ans = annotator.get("answer", annotator) if isinstance(annotator, dict) else {}
                    total_answers += 1
                    if ans.get("unanswerable"):
                        ans_unans += 1
                    elif ans.get("yes_no") is not None:
                        ans_yesno += 1
                    elif ans.get("extractive_spans"):
                        ans_extractive += 1
                    elif ans.get("free_form_answer"):
                        ans_freeform += 1

    # ── Assertions ─────────────────────────────────────────────────────────
    # 1. No duplicate paper IDs within split
    dup_papers = [pid for pid, cnt in Counter(paper_ids).items() if cnt > 1]
    assert len(dup_papers) == 0, f"Duplicate paper IDs in {path.name}: {dup_papers}"

    # 2. No duplicate question IDs within split
    dup_qids = [qid for qid, cnt in Counter(question_ids).items() if cnt > 1]
    assert len(dup_qids) == 0, f"Duplicate question IDs in {path.name}: {dup_qids}"

    total_q = len(question_ids)
    return {
        "n_papers": len(paper_ids),
        "paper_ids": paper_ids,
        "question_ids": question_ids,
        "empty_papers": empty_papers,
        "n_paragraphs": len(para_token_lengths),
        "paras_per_paper": {
            "mean": statistics.mean(paras_per_paper) if paras_per_paper else 0.0,
            "median": statistics.median(paras_per_paper) if paras_per_paper else 0,
            "max": max(paras_per_paper) if paras_per_paper else 0,
            "min": min(paras_per_paper) if paras_per_paper else 0,
        },
        "para_token_lengths": {
            "mean": statistics.mean(para_token_lengths) if para_token_lengths else 0.0,
            "median": statistics.median(para_token_lengths) if para_token_lengths else 0,
            "max": max(para_token_lengths) if para_token_lengths else 0,
            "min": min(para_token_lengths) if para_token_lengths else 0,
        },
        "n_questions": total_q,
        "questions_per_paper": {
            "mean": statistics.mean(questions_per_paper) if questions_per_paper else 0.0,
            "median": statistics.median(questions_per_paper) if questions_per_paper else 0,
            "max": max(questions_per_paper) if questions_per_paper else 0,
            "min": min(questions_per_paper) if questions_per_paper else 0,
        },
        "question_categories": dict(q_categories),
        "total_answers": total_answers,
        "answer_proportions": {
            "unanswerable": (ans_unans / total_answers * 100) if total_answers else 0.0,
            "yes_no": (ans_yesno / total_answers * 100) if total_answers else 0.0,
            "extractive": (ans_extractive / total_answers * 100) if total_answers else 0.0,
            "free_form": (ans_freeform / total_answers * 100) if total_answers else 0.0,
        },
    }


def generate_markdown_report(
    profiles: dict[str, dict[str, Any]],
    eval_stats: dict[str, Any],
) -> str:
    """Format profiled metrics into a GitHub-Flavored Markdown report."""
    md = []
    md.append("# QASPER Dataset Profile & Evaluation Set Report")
    md.append("")
    md.append(f"> Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    md.append("> Scope: Full empirical inspection across train, validation, and test splits.")
    md.append("")

    # Summary table
    md.append("## 1. High-Level Split Statistics")
    md.append("")
    md.append("| Split | Papers | Paragraphs | Questions | Annotations | Empty Papers |")
    md.append("|---|---:|---:|---:|---:|---:|")
    total_papers = 0
    total_paras = 0
    total_q = 0
    total_ans = 0
    total_empty = 0

    for split in ("train", "validation", "test"):
        p = profiles[split]
        n_p = p["n_papers"]
        n_para = p["n_paragraphs"]
        n_q = p["n_questions"]
        n_ans = p["total_answers"]
        n_emp = len(p["empty_papers"])
        total_papers += n_p
        total_paras += n_para
        total_q += n_q
        total_ans += n_ans
        total_empty += n_emp
        emp_note = f"{n_emp} ({', '.join(p['empty_papers'])})" if n_emp else "0"
        md.append(f"| **{split}** | {n_p:,} | {n_para:,} | {n_q:,} | {n_ans:,} | {emp_note} |")
    md.append(
        f"| **Total** | **{total_papers:,}** | **{total_paras:,}** | **{total_q:,}** | **{total_ans:,}** | **{total_empty}** |"
    )
    md.append("")

    # Paragraphs per paper
    md.append("## 2. Paragraph Distribution per Paper")
    md.append("")
    md.append("| Split | Mean | Median | Max | Min |")
    md.append("|---|---:|---:|---:|---:|")
    for split in ("train", "validation", "test"):
        d = profiles[split]["paras_per_paper"]
        md.append(f"| {split} | {d['mean']:.1f} | {d['median']} | {d['max']} | {d['min']} |")
    md.append("")

    # Paragraph token lengths
    md.append("## 3. Paragraph Length in Tokens (whitespace words)")
    md.append("")
    md.append("| Split | Mean Tokens | Median Tokens | Max Tokens | Min Tokens |")
    md.append("|---|---:|---:|---:|---:|")
    for split in ("train", "validation", "test"):
        d = profiles[split]["para_token_lengths"]
        md.append(f"| {split} | {d['mean']:.1f} | {d['median']} | {d['max']} | {d['min']} |")
    md.append("")

    # Questions per paper
    md.append("## 4. Questions per Paper")
    md.append("")
    md.append("| Split | Mean Qs | Median Qs | Max Qs | Min Qs |")
    md.append("|---|---:|---:|---:|---:|")
    for split in ("train", "validation", "test"):
        d = profiles[split]["questions_per_paper"]
        md.append(f"| {split} | {d['mean']:.1f} | {d['median']} | {d['max']} | {d['min']} |")
    md.append("")

    # Question types breakdown
    md.append("## 5. Question Type & Answer Breakdown")
    md.append("")
    md.append(
        "### Question-Level Categories (Prioritized: Unanswerable > Yes/No > Extractive > Free-form)"
    )
    md.append("")
    md.append("| Split | Extractive | Free-form | Unanswerable | Yes/No | Total |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for split in ("train", "validation", "test"):
        c = profiles[split]["question_categories"]
        n_ext = c.get("extractive", 0)
        n_ff = c.get("free_form", 0)
        n_un = c.get("unanswerable", 0)
        n_yn = c.get("yes_no", 0)
        tot = sum(c.values())
        md.append(
            f"| {split} | {n_ext} ({n_ext / tot * 100:.1f}%) | {n_ff} ({n_ff / tot * 100:.1f}%) | "
            f"{n_un} ({n_un / tot * 100:.1f}%) | {n_yn} ({n_yn / tot * 100:.1f}%) | {tot:,} |"
        )
    md.append("")

    # Annotator-level answer proportions
    md.append("### Annotator-Level Answer Distribution")
    md.append("")
    md.append("| Split | Extractive Spans % | Free-form % | Unanswerable % | Yes/No % |")
    md.append("|---|---:|---:|---:|---:|")
    for split in ("train", "validation", "test"):
        a = profiles[split]["answer_proportions"]
        md.append(
            f"| {split} | {a['extractive']:.1f}% | {a['free_form']:.1f}% | "
            f"{a['unanswerable']:.1f}% | {a['yes_no']:.1f}% |"
        )
    md.append("")

    # Data Integrity & Assertions
    md.append("## 6. Dataset Integrity & Assertions")
    md.append("")
    md.append("- **Duplicate Paper IDs:** 0 across all splits (asserted).")
    md.append("- **Duplicate Question IDs:** 0 across all splits (asserted).")
    md.append("- **Zero-Paragraph Papers in Validation / Test:** 0 (asserted).")
    md.append(
        "- **Zero-Paragraph Papers in Train:** 1 (`1801.07804` — Vietnamese Open Information Extraction, "
        "raw source paper had empty text extraction; excluded from chunking)."
    )
    md.append("")

    # Fixed Eval Set
    md.append("## 7. Fixed Evaluation Set (Stratified Sample)")
    md.append("")
    md.append(f"- **Destination:** `{eval_stats['path']}`")
    md.append(f"- **Sample Size:** {eval_stats['sample_size']} questions")
    md.append(f"- **Random Seed:** {eval_stats['seed']} (deterministic)")
    md.append(f"- **SHA-256 Checksum:** `{eval_stats['sha256']}`")
    md.append("")
    md.append("| Stratum | Validation Population | Val Share | Sample Quota | Sample Share |")
    md.append("|---|---:|---:|---:|---:|")
    val_cats = profiles["validation"]["question_categories"]
    tot_val = sum(val_cats.values())
    for cat in ("extractive", "free_form", "unanswerable", "yes_no"):
        pop_cnt = val_cats.get(cat, 0)
        pop_pct = pop_cnt / tot_val * 100
        quota = eval_stats["quotas"].get(cat, 0)
        quota_pct = quota / eval_stats["sample_size"] * 100
        md.append(f"| **{cat}** | {pop_cnt:,} | {pop_pct:.1f}% | {quota} | {quota_pct:.1f}% |")
    md.append("")

    return "\n".join(md) + "\n"


def main() -> None:
    """Run profiling across splits, assert integrity, and generate eval set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/data_profile.md"))
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    eval_dir: Path = args.eval_dir
    report_path: Path = args.report_path
    sample_size: int = args.sample_size
    seed: int = args.seed

    log.info("Profiling QASPER splits from %s ...", data_dir)
    profiles = {}
    all_paper_ids: list[str] = []
    all_q_ids: list[str] = []

    for split in ("train", "validation", "test"):
        split_path = data_dir / f"{split}.jsonl"
        log.info("Inspecting %s ...", split_path)
        prof = profile_split(split_path)
        profiles[split] = prof
        all_paper_ids.extend(prof["paper_ids"])
        all_q_ids.extend(prof["question_ids"])

    # Global cross-split assertions
    global_dup_papers = [pid for pid, c in Counter(all_paper_ids).items() if c > 1]
    assert len(global_dup_papers) == 0, f"Duplicate paper IDs across splits: {global_dup_papers}"

    global_dup_qids = [qid for qid, c in Counter(all_q_ids).items() if c > 1]
    assert len(global_dup_qids) == 0, f"Duplicate question IDs across splits: {global_dup_qids}"

    # Validation and test must have ZERO empty papers
    assert len(profiles["validation"]["empty_papers"]) == 0, "Validation has empty papers!"
    assert len(profiles["test"]["empty_papers"]) == 0, "Test has empty papers!"

    log.info("All dataset integrity assertions passed!")

    # ── Sample Fixed Stratified Evaluation Set ─────────────────────────────
    val_path = data_dir / "validation.jsonl"
    eval_output_path = eval_dir / "eval_set.jsonl"
    manifest_path = data_dir / "MANIFEST.json"

    log.info("Sampling %d questions from %s with seed %d ...", sample_size, val_path, seed)

    # Load validation questions
    val_candidates: list[dict[str, Any]] = []
    with val_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            pid = paper["paper_id"]
            title = paper.get("title", "")
            for qa in paper.get("qas", []):
                cat = categorize_qa(qa)
                val_candidates.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "question_id": qa["question_id"],
                        "question": qa["question"],
                        "category": cat,
                        "answers": [
                            a.get("answer", a) for a in qa.get("answers", []) if isinstance(a, dict)
                        ],
                    }
                )

    sampled, quotas = sample_stratified_qa(val_candidates, n=sample_size, seed=seed)

    # Assertions on sampled eval set
    assert len(sampled) == sample_size, f"Sample size mismatch: {len(sampled)} != {sample_size}"
    sampled_qids = [q["question_id"] for q in sampled]
    assert len(set(sampled_qids)) == sample_size, "Sampled eval set contains duplicate questions!"

    eval_dir.mkdir(parents=True, exist_ok=True)
    with eval_output_path.open("w", encoding="utf-8") as fh:
        for item in sampled:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    eval_sha256 = hashlib.sha256(eval_output_path.read_bytes()).hexdigest()
    log.info("Saved eval set to %s (SHA-256: %s)", eval_output_path, eval_sha256)

    # Update manifest
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "splits" not in manifest_data:
            manifest_data["splits"] = {}
        manifest_data["splits"]["eval"] = {
            "file": eval_output_path.name,
            "rows": len(sampled),
            "size_bytes": eval_output_path.stat().st_size,
            "sha256": eval_sha256,
            "sample_seed": seed,
            "stratification_quotas": quotas,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
        log.info("Updated %s with eval split entry.", manifest_path)

    # Write data profile report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = generate_markdown_report(
        profiles=profiles,
        eval_stats={
            "path": eval_output_path,
            "sample_size": sample_size,
            "seed": seed,
            "sha256": eval_sha256,
            "quotas": quotas,
        },
    )
    report_path.write_text(report_content, encoding="utf-8")
    log.info("Wrote profile report to %s", report_path)


if __name__ == "__main__":
    main()
