"""Evidence mapping from QASPER gold annotations to chunk IDs.

Given an evidence string from ground truth annotations:
1. Normalize text (lowercase, strip whitespace, strip punctuation).
2. Match to paper paragraphs: exact substring match first, then fuzzy
   SequenceMatcher similarity above threshold (default 0.80).
3. Map matched paragraph indices to the chunks that contain them.
4. Mark figure/table evidence (FLOAT SELECTED) that cannot be grounded
   in text paragraphs as unmatched with explicit reason.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

from researchmate.chunking.chunker import Chunk
from researchmate.data.cleaning import clean_text
from researchmate.data.models import Paragraph

log = logging.getLogger(__name__)

_RE_NON_ALPHANUM = re.compile(r"[^\w\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    """Normalize text for evidence substring and fuzzy matching."""
    if not text:
        return ""
    # Strip FLOAT SELECTED prefixes and clean common tokens
    cleaned = clean_text(text).lower()
    no_punct = _RE_NON_ALPHANUM.sub(" ", cleaned)
    return _RE_MULTI_SPACE.sub(" ", no_punct).strip()


def match_evidence_to_paragraphs(
    evidence_text: str,
    paragraphs: list[Paragraph],
    fuzzy_threshold: float = 0.80,
) -> tuple[list[int], float, str]:
    """Find paragraph indices that contain or match the evidence text.

    Args:
        evidence_text: Verbatim or cleaned evidence string from annotation.
        paragraphs: List of paper paragraphs to search.
        fuzzy_threshold: Minimum SequenceMatcher similarity ratio for fuzzy matches.

    Returns:
        A tuple of ``(matched_para_indices, best_score, match_type)``.
        ``match_type`` is one of ``'exact'``, ``'fuzzy'``, or ``'none'``.
    """
    if not evidence_text or not paragraphs:
        return [], 0.0, "none"

    norm_ev = normalize_for_matching(evidence_text)
    if not norm_ev:
        return [], 0.0, "none"

    # 1. Exact match (case-insensitive normalized substring in either direction)
    exact_matches: list[int] = []
    for p in paragraphs:
        norm_p = normalize_for_matching(p.raw_text or p.text)
        if norm_ev in norm_p or (len(norm_p) >= 20 and norm_p in norm_ev):
            exact_matches.append(p.index)

    if exact_matches:
        return exact_matches, 1.0, "exact"

    # 2. Fuzzy match across paragraphs
    best_score = 0.0
    best_para_idx: int | None = None

    for p in paragraphs:
        norm_p = normalize_for_matching(p.raw_text or p.text)
        if not norm_p:
            continue
        # Use quick ratio or find longest matching block if lengths differ wildly
        matcher = difflib.SequenceMatcher(None, norm_ev, norm_p, autojunk=False)
        match_block = matcher.find_longest_match(0, len(norm_ev), 0, len(norm_p))
        block_len = match_block.size
        # Ratio relative to evidence length
        coverage_score = block_len / len(norm_ev) if norm_ev else 0.0
        full_ratio = matcher.ratio()
        score = max(coverage_score, full_ratio)

        if score > best_score:
            best_score = score
            best_para_idx = p.index

    if best_score >= fuzzy_threshold and best_para_idx is not None:
        return [best_para_idx], best_score, "fuzzy"

    return [], best_score, "none"


def map_evidence_to_chunks(
    evidence_strings: list[str],
    chunks: list[Chunk],
    paragraphs: list[Paragraph],
    fuzzy_threshold: float = 0.80,
) -> dict[str, Any]:
    """Map a question's evidence strings to the chunks containing them.

    Args:
        evidence_strings: Ground-truth evidence strings for an answer.
        chunks: All chunks for the paper.
        paragraphs: All paragraphs for the paper.
        fuzzy_threshold: Minimum similarity threshold for fuzzy matching.

    Returns:
        Dictionary containing:
        - ``gold_chunk_ids``: Sorted unique chunk IDs containing matched evidence.
        - ``details``: Per-evidence matching breakdown.
    """
    gold_chunk_ids: set[str] = set()
    details: list[dict[str, Any]] = []

    # Map paragraph index to chunk IDs
    para_to_chunks: dict[int, list[str]] = {}
    for c in chunks:
        for idx in range(c.para_start, c.para_end + 1):
            para_to_chunks.setdefault(idx, []).append(c.chunk_id)

    for ev in evidence_strings:
        ev_clean = ev.strip()
        matched_paras, score, match_type = match_evidence_to_paragraphs(
            evidence_text=ev_clean,
            paragraphs=paragraphs,
            fuzzy_threshold=fuzzy_threshold,
        )

        matched_cids: list[str] = []
        for p_idx in matched_paras:
            for cid in para_to_chunks.get(p_idx, []):
                matched_cids.append(cid)
                gold_chunk_ids.add(cid)

        # Determine reason if unmatched
        reason = ""
        if not matched_cids:
            if (
                "FLOAT SELECTED" in ev_clean.upper()
                or "TABLE" in ev_clean.upper()
                or "FIGURE" in ev_clean.upper()
            ):
                reason = "figure_or_table_evidence"
            else:
                reason = f"no_paragraph_match (best_score={score:.2f})"

        details.append(
            {
                "evidence": ev_clean,
                "matched": len(matched_cids) > 0,
                "match_type": match_type,
                "score": score,
                "matched_paras": matched_paras,
                "matched_chunks": sorted(set(matched_cids)),
                "unmatched_reason": reason,
            }
        )

    return {
        "gold_chunk_ids": sorted(gold_chunk_ids),
        "details": details,
    }
