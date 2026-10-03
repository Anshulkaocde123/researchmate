#!/usr/bin/env python3
"""Inspect one QASPER paper and one QA entry; print the real schema.

Run scripts/download_data.py first.

Usage
-----
    python scripts/inspect_schema.py [--data-dir data/raw] [--split train]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

SEED = 42
W = 72  # print width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _type_label(value: Any) -> str:  # noqa: ANN401
    if value is None:
        return "null"
    if isinstance(value, list):
        inner = _type_label(value[0]) if value else "?"
        return f"list[{inner}] (len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"
    return type(value).__name__


def hr(title: str = "") -> None:
    print(f"\n{'─' * W}")
    if title:
        print(f"  {title}")
        print(f"{'─' * W}")


def trunc(text: str, n: int = 300) -> str:
    return text[:n] + " …" if len(text) > n else text


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------


def print_paper(paper: dict) -> None:
    hr("PAPER — top-level keys and types")
    for key, val in paper.items():
        print(f"  {key!r:40s} {_type_label(val)}")

    hr("PAPER — abstract")
    print(trunc(paper.get("abstract", "")))

    hr("PAPER — full_text (first 3 sections, up to 2 paragraphs each)")
    for i, section in enumerate(paper.get("full_text", [])[:3]):
        name = section.get("section_name", "") or "(no section name)"
        paras = section.get("paragraphs", [])
        print(f"\n  [{i}] {name!r}  ({len(paras)} paragraphs)")
        for j, para in enumerate(paras[:2]):
            print(f"    Para {j}: {trunc(para, 200)}")
        if len(paras) > 2:
            print(f"    … {len(paras) - 2} more paragraphs")

    total_sections = len(paper.get("full_text", []))
    if total_sections > 3:
        print(f"\n  … {total_sections - 3} more sections")


def print_qa(paper: dict) -> None:
    qas = paper.get("qas", [])
    if not qas:
        print("  (no QA pairs in this paper)")
        return

    qa = qas[0]
    hr("QA ENTRY — keys and types")
    for key, val in qa.items():
        print(f"  {key!r:40s} {_type_label(val)}")

    hr("QA ENTRY — question")
    print(f"  {qa['question']!r}")

    hr("QA ENTRY — answers (all annotators, all fields)")
    for ann_idx, annotator in enumerate(qa.get("answers", [])[:3]):
        print(
            f"\n  Annotator [{ann_idx}]  (annotation_id: {annotator.get('annotation_id','?')[:12]}…)"
        )
        ans = annotator.get("answer", {})
        for field, val in ans.items():
            print(f"    {field!r:30s} {json.dumps(val)[:100]}")


def print_answer_type_counts(splits_dir: Path, splits: tuple[str, ...]) -> None:
    hr("ANSWER TYPE COUNTS (all splits, all annotators)")
    counts: dict[str, int] = {"unanswerable": 0, "yes_no": 0, "extractive": 0, "free_form": 0, "total": 0}
    for split in splits:
        path = splits_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            paper = json.loads(line)
            for qa in paper.get("qas", []):
                for annotator in qa.get("answers", []):
                    ans = annotator.get("answer", {})
                    counts["total"] += 1
                    if ans.get("unanswerable"):
                        counts["unanswerable"] += 1
                    if ans.get("yes_no") is not None:
                        counts["yes_no"] += 1
                    if ans.get("extractive_spans"):
                        counts["extractive"] += 1
                    if ans.get("free_form_answer"):
                        counts["free_form"] += 1

    total = counts["total"]
    for key, val in counts.items():
        pct = f"({100 * val // total}%)" if total and key != "total" else ""
        print(f"  {key:20s} {val:6d}  {pct}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(data_dir: Path, split: str) -> None:
    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        log.error("File not found: %s\n  Run: python scripts/download_data.py", jsonl_path)
        raise SystemExit(1)

    papers = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    log.info("Loaded %d papers from %s", len(papers), jsonl_path)

    rng = random.Random(SEED)
    paper = rng.choice(papers)

    print("\n" + "═" * W)
    print("  QASPER SCHEMA INSPECTION")
    print(f"  Split: {split} | Papers: {len(papers)}")
    print(f"  Inspecting paper_id: {paper.get('paper_id', '?')}")
    print("═" * W)

    print_paper(paper)
    print_qa(paper)
    print_answer_type_counts(data_dir, ("train", "validation", "test"))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect QASPER schema from downloaded JSONL files"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    args = parser.parse_args()
    main(args.data_dir, args.split)
