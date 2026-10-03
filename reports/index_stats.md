# ResearchMate Indexing Statistics

This report documents the characteristics, performance, and disk footprints of ResearchMate's dual retrieval indices (Dense Chroma and Sparse BM25).

## 1. Corpus Overview

- **Source Dataset:** QASPER v0.3 NLP papers
- **Total Papers:** 1,584
- **Total Chunks:** 34,280
- **Average Chunks per Paper:** ~21.6
- **Chunker:** Section-aware recursive paragraph chunker with sentence fallback
- **Target Chunk Size:** 512 tokens (overlap: 64 tokens)

## 2. Sparse Index (BM25Okapi)

- **Algorithm:** BM25Okapi (`rank_bm25`)
- **Tokenization:** Alphanumeric lowercase word tokenization (`\b\w+\b`)
- **Indexed Fields:** Contextualized chunk text (`[Title | Section]\n{text}`)
- **Metadata Preserved:** `chunk_id`, `paper_id`, `title`, `section`, `para_start`, `para_end`, `n_tokens`, `text`
- **Total Chunks Indexed:** 34,280 (100% of corpus)
- **Total Papers Indexed:** 1,584
- **Build Time:** 9.61 seconds (~3,567 chunks/second)
- **Serialization:** Gzip-compressed Python pickle (`indexes/bm25.pkl.gz`)
- **Disk Size:** 25.54 MB

## 3. Dense Vector Store (Chroma)

- **Embedding Model:** `BAAI/bge-base-en-v1.5` via ONNX runtime (`fastembed`)
- **Embedding Dimensions:** 768 float32
- **Distance Metric:** Cosine similarity (`hnsw:space = "cosine"`)
- **L2 Normalization:** Applied to all chunk and query vectors
- **Query Instruction Prefix:** `"Represent this sentence for searching relevant passages: "`
- **Cache Architecture:** SQLite (`indexes/embedding_cache.sqlite`) with WAL mode and SHA-256 keying
- **Storage:** Chroma persistent collection (`indexes/chroma/`)

## 4. Hybrid Retrieval Architecture

- **Fusion Scheme:** Reciprocal Rank Fusion (RRF) with $k=60$
- **Reranker:** `BAAI/bge-reranker-base` cross-encoder via ONNX (`fastembed`)
- **Candidate Pool:** Top-$k_{dense}$ (20) + Top-$k_{sparse}$ (20) $\to$ RRF merge $\to$ Top-$k_{rerank}$ (5)
- **Paper Filtering:** Strictly enforced via metadata filter `{"paper_id": ...}` for paper-scoped QA.
