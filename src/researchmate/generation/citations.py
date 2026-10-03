"""Citation extraction, validation, and metadata alignment."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

# Matches variations: [Passage 1], [Passage 1, 2], [1], [Chunk 1]
_CITATION_PATTERN = re.compile(
    r"\[(?:Passage|Chunk)?\s*(\d+(?:\s*,\s*\d+)*)\]",
    re.IGNORECASE,
)


class CitationMatch(BaseModel):
    """Details for an extracted citation."""

    passage_num: int
    chunk_id: str
    paper_id: str
    section: str
    title: str
    text_snippet: str


class CitationResult(BaseModel):
    """Aggregate citation evaluation for a generated answer."""

    cited_passage_nums: list[int] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    citations: list[CitationMatch] = Field(default_factory=list)
    has_valid_citations: bool = False
    is_unanswerable: bool = False


def extract_citations(
    answer: str,
    context_chunks: list[dict[str, Any]],
) -> CitationResult:
    """Extract bracketed passage citations and map to chunk metadata.

    Args:
        answer: Generated LLM response text.
        context_chunks: Ordered list of retrieved chunks presented in the prompt.

    Returns:
        CitationResult containing structured citation details.
    """
    is_unanswerable = (
        "cannot answer" in answer.lower()
        or "not mentioned" in answer.lower()
        or "insufficient information" in answer.lower()
        or "unanswerable" in answer.lower()
    )

    extracted_nums: set[int] = set()
    for match in _CITATION_PATTERN.finditer(answer):
        raw_nums = match.group(1).split(",")
        for num_str in raw_nums:
            num_str = num_str.strip()
            if num_str.isdigit():
                num = int(num_str)
                extracted_nums.add(num)

    sorted_nums = sorted(extracted_nums)
    citations: list[CitationMatch] = []
    chunk_ids: list[str] = []

    for num in sorted_nums:
        idx = num - 1  # 1-indexed to 0-indexed
        if 0 <= idx < len(context_chunks):
            chunk = context_chunks[idx]
            cid = chunk.get("chunk_id", "")
            meta = chunk.get("metadata", {})
            citations.append(
                CitationMatch(
                    passage_num=num,
                    chunk_id=cid,
                    paper_id=meta.get("paper_id", ""),
                    section=meta.get("section", ""),
                    title=meta.get("title", ""),
                    text_snippet=chunk.get("text", "")[:200],
                )
            )
            chunk_ids.append(cid)

    has_valid = len(citations) > 0 or is_unanswerable

    return CitationResult(
        cited_passage_nums=sorted_nums,
        cited_chunk_ids=chunk_ids,
        citations=citations,
        has_valid_citations=has_valid,
        is_unanswerable=is_unanswerable,
    )
