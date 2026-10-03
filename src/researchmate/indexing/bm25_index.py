"""Sparse BM25 index for keyword retrieval over ResearchMate chunks.

Implements the BM25Index interface:
- build: construct BM25Okapi inverted index from chunks
- query: score and rank chunks using BM25 with optional paper_id filtering
- save / load: serialize index and metadata to disk
"""

from __future__ import annotations

import gzip
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from researchmate.chunking.chunker import Chunk

log = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric word tokens."""
    return _WORD_PATTERN.findall(text.lower())


class BM25Index:
    """In-memory BM25 index with disk serialization and paper filtering."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []
        self._paper_to_indices: dict[str, list[int]] = {}

        if chunks:
            self.build(chunks)

    def build(self, chunks: list[Chunk]) -> None:
        """Construct BM25 index from a list of chunks."""
        self.chunks = []
        self._paper_to_indices = {}
        tokenized_corpus: list[list[str]] = []

        for idx, c in enumerate(chunks):
            # Tokenize the rich embed_text (contains title + section context + text)
            tokens = tokenize(c.embed_text if hasattr(c, "embed_text") and c.embed_text else c.text)
            tokenized_corpus.append(tokens)

            record = {
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "title": c.title,
                "section": c.section,
                "para_start": c.para_start,
                "para_end": c.para_end,
                "n_tokens": c.n_tokens,
                "text": c.text,
                "embed_text": c.embed_text if hasattr(c, "embed_text") else "",
            }
            self.chunks.append(record)

            pid = c.paper_id
            if pid not in self._paper_to_indices:
                self._paper_to_indices[pid] = []
            self._paper_to_indices[pid].append(idx)

        self.bm25 = BM25Okapi(tokenized_corpus)
        log.info(
            "Built BM25 index with %d chunks across %d papers",
            len(chunks),
            len(self._paper_to_indices),
        )

    def count(self) -> int:
        """Return total chunks in index."""
        return len(self.chunks)

    def query(
        self,
        query: str,
        k: int = 20,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[dict[str, Any]]:
        """Retrieve top-k chunks matching the query string.

        Args:
            query: Natural language query string.
            k: Maximum number of results to return.
            filter: Optional dict filter, e.g. {"paper_id": "2002.06424"}.

        Returns:
            List of dicts sorted by BM25 score descending:
            [{"chunk_id": str, "text": str, "score": float, "metadata": dict}]
        """
        if self.bm25 is None or not self.chunks or k <= 0:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens))

        # Check paper_id filter
        paper_id = filter.get("paper_id") if filter else None
        if paper_id is not None:
            valid_indices = self._paper_to_indices.get(paper_id, [])
            if not valid_indices:
                return []
            cand_indices = np.array(valid_indices)
            sub_scores = scores[cand_indices]
            n_select = min(k, len(cand_indices))
            if n_select == len(cand_indices):
                sorted_rel = np.argsort(sub_scores)[::-1]
            else:
                top_rel = np.argpartition(sub_scores, -n_select)[-n_select:]
                sorted_rel = top_rel[np.argsort(sub_scores[top_rel])[::-1]]
            top_indices = cand_indices[sorted_rel]
        else:
            n_select = min(k, len(scores))
            if n_select == len(scores):
                top_indices = np.argsort(scores)[::-1]
            else:
                cand_indices = np.argpartition(scores, -n_select)[-n_select:]
                top_indices = cand_indices[np.argsort(scores[cand_indices])[::-1]]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            c = self.chunks[idx]
            results.append(
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "score": score,
                    "metadata": {
                        "paper_id": c["paper_id"],
                        "title": c["title"],
                        "section": c["section"],
                        "para_start": c["para_start"],
                        "para_end": c["para_end"],
                        "n_tokens": c["n_tokens"],
                        "embed_text": c.get("embed_text", ""),
                    },
                }
            )

        return results

    def save(self, path: Path | str) -> None:
        """Serialize BM25 index and chunk records to compressed pickle."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bm25": self.bm25,
            "chunks": self.chunks,
            "paper_to_indices": self._paper_to_indices,
        }
        with gzip.open(p, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved BM25 index to %s (%d chunks)", p, len(self.chunks))

    @classmethod
    def load(cls, path: Path | str) -> BM25Index:
        """Deserialize BM25 index from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"BM25 index file not found: {p}")
        with gzip.open(p, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        instance = cls()
        instance.bm25 = data["bm25"]
        instance.chunks = data["chunks"]
        instance._paper_to_indices = data["paper_to_indices"]
        log.info("Loaded BM25 index from %s (%d chunks)", p, len(instance.chunks))
        return instance
