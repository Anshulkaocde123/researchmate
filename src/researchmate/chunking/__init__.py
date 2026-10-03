"""Section-aware document chunking, token counting, and evidence mapping."""

from researchmate.chunking.chunker import (
    Chunk,
    chunk_paper,
    chunk_section,
    default_token_counter,
    get_token_counter,
    split_sentences,
)
from researchmate.chunking.evidence import (
    map_evidence_to_chunks,
    match_evidence_to_paragraphs,
    normalize_for_matching,
)

__all__ = [
    "Chunk",
    "chunk_paper",
    "chunk_section",
    "default_token_counter",
    "get_token_counter",
    "map_evidence_to_chunks",
    "match_evidence_to_paragraphs",
    "normalize_for_matching",
    "split_sentences",
]
