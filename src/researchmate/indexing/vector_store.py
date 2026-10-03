"""Persistent Chroma vector store with cosine similarity for ResearchMate chunks.

Implements the VectorStore interface:
- add: Idempotently upsert chunks with embeddings and metadata
- query: Retrieve top-k nearest chunks with optional metadata filters
- count: Total chunks indexed in the collection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb

from researchmate.chunking.chunker import Chunk
from researchmate.config import settings

log = logging.getLogger(__name__)


class VectorStore:
    """Persistent vector store wrapping ChromaDB with cosine metric."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = "chunks",
        client: Any | None = None,
    ) -> None:
        self.collection_name = collection_name

        if client is not None:
            self._client = client
        elif persist_dir == ":memory:":
            self._client = chromadb.Client()
        else:
            p_dir = Path(persist_dir) if persist_dir else (settings.index_dir / "chroma")
            p_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(p_dir))

        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Idempotently add or update chunks in the vector store.

        Args:
            chunks: List of Chunk objects containing metadata and text.
            embeddings: Parallel list of L2-normalized embedding vectors.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "paper_id": c.paper_id,
                "title": c.title,
                "section": c.section,
                "para_start": c.para_start,
                "para_end": c.para_end,
                "n_tokens": c.n_tokens,
                "embed_text": c.embed_text,
            }
            for c in chunks
        ]

        # Use upsert so re-running never duplicates records
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[dict[str, Any]]:
        """Query nearest chunks by cosine similarity.

        Args:
            embedding: Query embedding vector.
            k: Number of nearest chunks to return.
            filter: Optional Chroma metadata filter (e.g. {"paper_id": "2002.06424"}).

        Returns:
            List of result dicts sorted by score descending:
            [{"chunk_id": str, "text": str, "score": float, "metadata": dict}]
        """
        n_results = min(k, self.count())
        if n_results == 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": n_results,
        }
        if filter:
            query_kwargs["where"] = filter

        raw_results = self.collection.query(**query_kwargs)

        results: list[dict[str, Any]] = []
        ids = raw_results.get("ids", [[]])[0]
        docs = raw_results.get("documents", [[]])[0]
        metas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            # In cosine space, similarity = 1 - cosine_distance
            sim = 1.0 - float(dist) if dist is not None else 0.0
            results.append(
                {
                    "chunk_id": cid,
                    "text": doc,
                    "score": sim,
                    "metadata": meta,
                }
            )

        # Sort by similarity descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def count(self) -> int:
        """Return total number of chunks currently stored."""
        return self.collection.count()

    def get_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a single chunk by chunk_id."""
        raw = self.collection.get(ids=[chunk_id])
        ids = raw.get("ids", [])
        if not ids:
            return None
        return {
            "chunk_id": ids[0],
            "text": raw.get("documents", [""])[0],
            "metadata": raw.get("metadatas", [{}])[0],
        }

    def clear(self) -> None:
        """Delete all chunks from the collection."""
        self._client.delete_collection(name=self.collection_name)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
