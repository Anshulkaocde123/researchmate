"""Dense text embedder wrapping fastembed with query prefix and disk cache.

Features:
- ONNX CPU execution via fastembed (no torch/transformers needed).
- L2-normalized unit embeddings (||v||_2 = 1.0).
- Query instruction prefix: 'Represent this sentence for searching relevant passages: '
  applied to queries only and exactly once.
- On-disk SQLite cache keyed by SHA-256 text hash to avoid redundant computation.
- Lazy model loading so imports and cache-only operations never download weights.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from researchmate.config import settings

log = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Normalize vectors along the last axis to unit L2 norm."""
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


class EmbeddingCache:
    """Persistent SQLite cache for embedding vectors keyed by SHA-256 hash."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    hash_key TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL
                )
                """
            )

    def get(self, key: str) -> list[float] | None:
        """Retrieve a single cached embedding by key."""
        row = self._conn.execute(
            "SELECT embedding, dim FROM embedding_cache WHERE hash_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        blob, _dim = row
        arr = np.frombuffer(blob, dtype=np.float32)
        return arr.tolist()

    def get_batch(self, keys: list[str]) -> dict[str, list[float]]:
        """Retrieve multiple embeddings by keys in a single query."""
        if not keys:
            return {}
        results: dict[str, list[float]] = {}
        # Chunk in batches of 500 to avoid SQLite variable limits
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            placeholders = ",".join("?" for _ in batch)
            cursor = self._conn.execute(
                f"SELECT hash_key, embedding, dim FROM embedding_cache WHERE hash_key IN ({placeholders})",  # noqa: S608
                batch,
            )
            for h_key, blob, _dim in cursor.fetchall():
                results[h_key] = np.frombuffer(blob, dtype=np.float32).tolist()
        return results

    def set_batch(self, items: dict[str, list[float]]) -> None:
        """Insert or replace a batch of embeddings."""
        if not items:
            return
        records = [
            (h_key, np.array(vec, dtype=np.float32).tobytes(), len(vec))
            for h_key, vec in items.items()
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (hash_key, embedding, dim) VALUES (?, ?, ?)",
                records,
            )

    def count(self) -> int:
        """Return total number of cached embeddings."""
        row = self._conn.execute("SELECT count(*) FROM embedding_cache").fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        """Delete all entries from cache."""
        with self._conn:
            self._conn.execute("DELETE FROM embedding_cache")


class FastEmbedder:
    """Dense embedder using ONNX-based fastembed models with caching and prefixes."""

    def __init__(
        self,
        model_name: str | None = None,
        cache_path: Path | str | None = None,
        use_cache: bool = True,
        batch_size: int = 32,
        embedding_model: Any | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.use_cache = use_cache
        self.batch_size = batch_size
        self._custom_model = embedding_model
        self._model: Any | None = embedding_model

        if cache_path is None:
            cache_file = settings.index_dir / "embedding_cache.sqlite"
        else:
            cache_file = (
                Path(cache_path)
                if isinstance(cache_path, str) and cache_path != ":memory:"
                else cache_path
            )
        self.cache = EmbeddingCache(cache_file) if self.use_cache else None

    @property
    def model(self) -> Any:
        """Lazily initialize the fastembed TextEmbedding model on first use."""
        if self._model is None:
            from fastembed import TextEmbedding  # type: ignore[import-untyped]

            log.info("Loading fastembed model: %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _hash_key(self, text: str, is_query: bool) -> str:
        """Generate a deterministic cache key based on model, prefix state, and text."""
        raw = f"{self.model_name}:{'Q:' if is_query else 'D:'}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def format_query(self, query: str) -> str:
        """Format a query with the BGE query prefix exactly once."""
        query_clean = query.strip()
        if not query_clean.startswith(BGE_QUERY_PREFIX):
            return f"{BGE_QUERY_PREFIX}{query_clean}"
        return query_clean

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Generate normalized embeddings for a list of document passages.

        Checks the on-disk cache first; only encodes uncached texts.

        Args:
            texts: List of document strings to embed.
            batch_size: Batch size for model inference (default 32).

        Returns:
            List of L2-normalized embedding vectors (unit norm).
        """
        if not texts:
            return []

        bs = batch_size or self.batch_size
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        keys = [self._hash_key(t, is_query=False) for t in texts]

        # 1. Lookup in cache
        if self.use_cache and self.cache is not None:
            cached_map = self.cache.get_batch(keys)
            for idx, key in enumerate(keys):
                if key in cached_map:
                    results[idx] = cached_map[key]
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(texts[idx])
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = list(texts)

        # 2. Encode uncached texts
        if uncached_texts:
            new_embeddings: list[list[float]] = []
            for i in range(0, len(uncached_texts), bs):
                chunk = uncached_texts[i : i + bs]
                # Embed without query prefix
                raw_embeds = self._encode_batch(chunk)
                norm_embeds = l2_normalize(np.asarray(raw_embeds, dtype=np.float32))
                new_embeddings.extend(norm_embeds.tolist())

            # Update cache and fill results
            new_cache_items: dict[str, list[float]] = {}
            for orig_idx, emb in zip(uncached_indices, new_embeddings, strict=True):
                results[orig_idx] = emb
                key = keys[orig_idx]
                new_cache_items[key] = emb

            if self.use_cache and self.cache is not None and new_cache_items:
                self.cache.set_batch(new_cache_items)

        # Assert all results populated
        final_embeddings: list[list[float]] = [e for e in results if e is not None]
        assert len(final_embeddings) == len(texts)
        return final_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate a normalized embedding for a search query.

        Applies the BGE query prefix exactly once and checks cache.

        Args:
            query: User search query.

        Returns:
            L2-normalized embedding vector.
        """
        prefixed_query = self.format_query(query)
        key = self._hash_key(prefixed_query, is_query=True)

        if self.use_cache and self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        raw_embeds = self._encode_batch([prefixed_query])
        norm_embed = l2_normalize(np.asarray(raw_embeds, dtype=np.float32))[0]
        emb_list = norm_embed.tolist()

        if self.use_cache and self.cache is not None:
            self.cache.set_batch({key: emb_list})

        return emb_list

    def _encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Invoke underlying model.embed or fallback callable."""
        if hasattr(self.model, "embed"):
            generator: Iterable[np.ndarray] = self.model.embed(texts)
            return list(generator)
        if callable(self.model):
            return self.model(texts)
        raise TypeError(f"Incompatible embedding model type: {type(self.model)}")

    @property
    def embedding_dim(self) -> int:
        """Return embedding vector dimension."""
        if hasattr(self.model, "embedding_size"):
            size = self.model.embedding_size
            if isinstance(size, int) and size > 0:
                return size
        # Fallback probe
        probe = self.embed_query("probe")
        return len(probe)
