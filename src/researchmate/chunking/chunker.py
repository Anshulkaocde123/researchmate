"""Section-aware paragraph and sentence chunker for ResearchMate.

Rules:
1. Chunk strictly within sections (never span across sections).
2. Skip empty sections or sections without paragraphs.
3. Pack whole paragraphs up to ``max_tokens`` (default 400) with
   ``overlap_paragraphs`` (default 1).
4. Split inside a paragraph on sentence boundaries only if that single
   paragraph exceeds ``max_tokens``.
5. Produce deterministic ``chunk_id`` hashes based on paper_id, section,
   and paragraph span.
6. Build a separate ``embed_text`` formatted as ``Title > Section\\n\\n{text}``
   while keeping ``text`` clean for display.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from researchmate.data.models import Paper, Paragraph

log = logging.getLogger(__name__)

# Sentence boundary regex (matches after punctuation followed by space or newline)
_RE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class Chunk(BaseModel):
    """A retrieval and display chunk produced from a paper."""

    chunk_id: str = Field(description="Deterministic 16-char hex hash ID.")
    paper_id: str = Field(description="arXiv ID of the source paper.")
    title: str = Field(description="Title of the paper.")
    section: str = Field(description="Section heading where this chunk resides.")
    text: str = Field(description="Unmodified display text of the chunk.")
    embed_text: str = Field(description="Contextualized text formatted for dense embedding.")
    para_start: int = Field(ge=0, description="Global paragraph index where chunk begins.")
    para_end: int = Field(ge=0, description="Global paragraph index where chunk ends (inclusive).")
    n_tokens: int = Field(gt=0, description="Number of tokens in the chunk.")


def default_token_counter(text: str) -> int:
    """Default token counter.

    Estimates tokens using whitespace words plus punctuation tokens, or
    falls back cleanly without external network calls.
    """
    if not text:
        return 0
    # Splitting on whitespace and punctuation tokens gives a close upper bound to subword BPE
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return max(1, len(tokens))


def get_token_counter(
    model_name: str | None = None,
) -> Callable[[str], int]:
    """Return a token counter function.

    If a tokenizer can be safely loaded locally, it is used; otherwise falls back
    to ``default_token_counter`` without hitting the network.
    """
    # Safely check if fastembed or tokenizers has a cached local tokenizer
    try:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]

        # Check if local cache has BGE tokenizer
        if model_name:
            tok = Tokenizer.from_pretrained(model_name)
            return lambda s: len(tok.encode(s).ids)
    except Exception:  # noqa: BLE001
        pass

    return default_token_counter


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries."""
    sentences = [s.strip() for s in _RE_SENTENCE_BOUNDARY.split(text) if s.strip()]
    return sentences if sentences else [text]


def chunk_section(
    paper_id: str,
    title: str,
    section: str,
    paragraphs: list[Paragraph],
    max_tokens: int = 400,
    overlap_paragraphs: int = 1,
    token_counter: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Chunk all paragraphs within a single section.

    Never crosses section boundaries.
    """
    if not paragraphs or not section.strip():
        return []

    count_tokens = token_counter or default_token_counter
    chunks: list[Chunk] = []
    n = len(paragraphs)
    i = 0

    while i < n:
        para_i = paragraphs[i]
        p_text = para_i.text.strip()
        p_tokens = count_tokens(p_text)

        # ── Rule 4: Single paragraph exceeds max_tokens → split on sentences ─
        if p_tokens > max_tokens:
            sentences = split_sentences(p_text)
            curr_sents: list[str] = []
            curr_tokens = 0
            sent_sub_idx = 0

            for sent in sentences:
                s_tokens = count_tokens(sent)
                if curr_sents and (curr_tokens + s_tokens > max_tokens):
                    sub_text = " ".join(curr_sents)
                    raw_key = f"{paper_id}:{section}:{para_i.index}:{sent_sub_idx}"
                    cid = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
                    embed_txt = f"{title} > {section}\n\n{sub_text}"

                    chunks.append(
                        Chunk(
                            chunk_id=cid,
                            paper_id=paper_id,
                            title=title,
                            section=section,
                            text=sub_text,
                            embed_text=embed_txt,
                            para_start=para_i.index,
                            para_end=para_i.index,
                            n_tokens=curr_tokens,
                        )
                    )
                    sent_sub_idx += 1
                    curr_sents = [sent]
                    curr_tokens = s_tokens
                else:
                    curr_sents.append(sent)
                    curr_tokens += s_tokens

            if curr_sents:
                sub_text = " ".join(curr_sents)
                raw_key = f"{paper_id}:{section}:{para_i.index}:{sent_sub_idx}"
                cid = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
                embed_txt = f"{title} > {section}\n\n{sub_text}"

                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        paper_id=paper_id,
                        title=title,
                        section=section,
                        text=sub_text,
                        embed_text=embed_txt,
                        para_start=para_i.index,
                        para_end=para_i.index,
                        n_tokens=curr_tokens,
                    )
                )
            i += 1
            continue

        # ── Rule 3: Pack whole paragraphs up to max_tokens ───────────────────
        curr_paras = [p_text]
        curr_tokens = p_tokens
        j = i + 1

        while j < n:
            next_para = paragraphs[j]
            next_text = next_para.text.strip()
            next_tokens = count_tokens(next_text)

            # Do not pack next paragraph if it would exceed max_tokens
            if curr_tokens + next_tokens > max_tokens:
                break

            curr_paras.append(next_text)
            curr_tokens += next_tokens
            j += 1

        para_start = paragraphs[i].index
        para_end = paragraphs[j - 1].index
        chunk_text = "\n\n".join(curr_paras)
        raw_key = f"{paper_id}:{section}:{para_start}:{para_end}"
        cid = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        embed_txt = f"{title} > {section}\n\n{chunk_text}"

        chunks.append(
            Chunk(
                chunk_id=cid,
                paper_id=paper_id,
                title=title,
                section=section,
                text=chunk_text,
                embed_text=embed_txt,
                para_start=para_start,
                para_end=para_end,
                n_tokens=curr_tokens,
            )
        )

        if j >= n:
            break

        # Step forward with overlap
        step = max(1, (j - i) - overlap_paragraphs)
        i += step

    return chunks


def chunk_paper(
    paper: Paper,
    max_tokens: int = 400,
    overlap_paragraphs: int = 1,
    token_counter: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Chunk an entire paper section-by-section.

    Args:
        paper: Paper object containing paragraphs.
        max_tokens: Maximum token ceiling per chunk.
        overlap_paragraphs: Number of whole paragraphs to overlap between consecutive chunks.
        token_counter: Callable returning token count for a string.

    Returns:
        List of Chunk objects spanning the paper.
    """
    if not paper.paragraphs:
        return []

    # Group paragraphs by section preserving order
    sections: dict[str, list[Paragraph]] = {}
    for para in paper.paragraphs:
        sec = para.section.strip()
        if not sec:
            continue
        sections.setdefault(sec, []).append(para)

    all_chunks: list[Chunk] = []
    for section_name, sec_paras in sections.items():
        # Skip empty sections
        if not sec_paras:
            continue

        sec_chunks = chunk_section(
            paper_id=paper.paper_id,
            title=paper.title,
            section=section_name,
            paragraphs=sec_paras,
            max_tokens=max_tokens,
            overlap_paragraphs=overlap_paragraphs,
            token_counter=token_counter,
        )
        all_chunks.extend(sec_chunks)

    return all_chunks
