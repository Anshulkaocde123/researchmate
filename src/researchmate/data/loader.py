"""QASPER dataset loader.

Public API
----------
``load_split(split, data_dir)``
    Load a named split (train / validation / test) and return
    ``(list[Paper], list[QAExample])``.

``load_papers(path)``
    Load from any JSONL file where each line is a paper dict
    (same format as ``data/raw/<split>.jsonl``).  Used by tests.

File format (data/raw/<split>.jsonl)
-------------------------------------
Each line is a JSON object with the keys documented in docs/data_schema.md.
The ``paper_id`` field is the arXiv ID injected by ``scripts/download_data.py``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from researchmate.config import settings
from researchmate.data.models import Answer, Paper, Paragraph, QAExample

log = logging.getLogger(__name__)

_FALLBACK_SECTION = "(no section)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_raw(path: Path) -> Iterator[dict]:
    """Yield one raw dict per non-empty line in a JSONL file."""
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)


def _parse_answer(raw_ans: dict) -> Answer:
    """Parse one annotator's answer dict into an Answer model.

    Handles null / missing fields explicitly.
    """
    evidence_raw = raw_ans.get("evidence") or []
    evidence = [str(e) for e in evidence_raw if e is not None and str(e).strip()]

    return Answer(
        unanswerable=bool(raw_ans.get("unanswerable", False)),
        yes_no=raw_ans.get("yes_no"),  # bool | None
        extractive_spans=raw_ans.get("extractive_spans") or [],
        free_form=raw_ans.get("free_form_answer") or "",
        evidence=evidence,
    )


def _parse_paper(raw: dict) -> tuple[Paper, list[QAExample]]:
    """Convert one raw JSONL dict into a Paper + its QAExamples.

    Empty paragraphs are silently dropped.
    Missing or None fields fall back to safe defaults.
    """
    paper_id: str = raw.get("paper_id", "")

    # ── Full text → flat Paragraph list ───────────────────────────────────
    paragraphs: list[Paragraph] = []
    global_idx = 0
    for section in raw.get("full_text") or []:
        section_name = (section.get("section_name") or "").strip() or _FALLBACK_SECTION
        for para_text in section.get("paragraphs") or []:
            if not isinstance(para_text, str) or not para_text.strip():
                continue
            paragraphs.append(
                Paragraph(
                    paper_id=paper_id,
                    section=section_name,
                    index=global_idx,
                    text=para_text,
                )
            )
            global_idx += 1

    paper = Paper(
        paper_id=paper_id,
        title=raw.get("title") or "",
        abstract=raw.get("abstract") or "",
        paragraphs=paragraphs,
    )

    # ── QA examples ──────────────────────────────────────────────────────
    qa_examples: list[QAExample] = []
    for qa in raw.get("qas") or []:
        answers: list[Answer] = []
        for annotator in qa.get("answers") or []:
            raw_ans = annotator.get("answer")
            if not isinstance(raw_ans, dict):
                continue
            try:
                answers.append(_parse_answer(raw_ans))
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Skipping malformed answer in question %s: %s",
                    qa.get("question_id", "?"),
                    exc,
                )

        qa_examples.append(
            QAExample(
                paper_id=paper_id,
                question_id=qa.get("question_id", ""),
                question=qa.get("question", ""),
                answers=answers,
            )
        )

    return paper, qa_examples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_papers(path: Path) -> tuple[list[Paper], list[QAExample]]:
    """Load papers and QA examples from a JSONL file.

    Parameters
    ----------
    path:
        Path to a JSONL file (one paper JSON object per line).

    Returns
    -------
    (papers, qa_examples)
        ``papers`` contains one ``Paper`` per line.
        ``qa_examples`` contains one ``QAExample`` per question
        (a paper with N questions contributes N QAExamples).
    """
    papers: list[Paper] = []
    qa_examples: list[QAExample] = []

    for raw in _iter_raw(path):
        paper, qas = _parse_paper(raw)
        papers.append(paper)
        qa_examples.extend(qas)

    log.info(
        "Loaded %d papers, %d QA examples from %s",
        len(papers),
        len(qa_examples),
        path,
    )
    return papers, qa_examples


def load_split(
    split: str,
    data_dir: Path | None = None,
) -> tuple[list[Paper], list[QAExample]]:
    """Load a named QASPER split from ``data_dir``.

    Parameters
    ----------
    split:
        One of ``'train'``, ``'validation'``, ``'test'``.
    data_dir:
        Directory containing ``<split>.jsonl`` files.
        Defaults to ``settings.data_dir / 'raw'``.

    Returns
    -------
    (papers, qa_examples)
    """
    if data_dir is None:
        data_dir = settings.data_dir / "raw"
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n  Run: python scripts/download_data.py"
        )
    return load_papers(path)
