"""Unit tests for BM25Index using rank_bm25.

Tests cover:
- Index construction and count
- Top-k query retrieval and descending score order
- paper_id filtering restricts results strictly to target paper
- Save and load round-trip persistence (gzip pickle)
- Edge cases: empty query, out-of-vocabulary query, empty index, k <= 0
- Verification using mini_qasper fixture
"""

from __future__ import annotations

from pathlib import Path

from researchmate.chunking.chunker import chunk_paper
from researchmate.data.loader import load_papers
from researchmate.indexing.bm25_index import BM25Index, tokenize


def test_tokenize() -> None:
    """tokenize lowercases and splits words properly."""
    text = "BERT: Pre-training of Deep Bidirectional Transformers!"
    tokens = tokenize(text)
    assert tokens == ["bert", "pre", "training", "of", "deep", "bidirectional", "transformers"]


def test_bm25_build_and_count() -> None:
    """BM25Index counts match the total input chunks."""
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    chunks = chunk_paper(papers[0])

    index = BM25Index(chunks)
    assert index.count() == len(chunks)


def test_bm25_query_and_ranking() -> None:
    """Querying returns descending scores and matches expected text."""
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    all_chunks = []
    for p in papers:
        all_chunks.extend(chunk_paper(p))

    index = BM25Index(all_chunks)

    # Use a unique query string from the first chunk
    query = all_chunks[0].text[:50]
    results = index.query(query, k=5)

    assert len(results) <= 5
    assert len(results) > 0

    # Scores should be in descending order
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    # Top result should be the target chunk or contain the query
    assert results[0]["chunk_id"] == all_chunks[0].chunk_id
    assert results[0]["metadata"]["paper_id"] == all_chunks[0].paper_id


def test_bm25_paper_filter() -> None:
    """Filtering by paper_id strictly returns chunks from that paper."""
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    all_chunks = []
    for p in papers:
        all_chunks.extend(chunk_paper(p))

    index = BM25Index(all_chunks)
    target_pid = papers[1].paper_id

    results = index.query(
        query="model training and evaluation",
        k=10,
        filter={"paper_id": target_pid},
    )

    assert len(results) > 0
    for r in results:
        assert r["metadata"]["paper_id"] == target_pid


def test_bm25_save_and_load(tmp_path: Path) -> None:
    """Saving and loading round-trip preserves index state and query results."""
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    chunks = chunk_paper(papers[0])

    index = BM25Index(chunks)
    save_path = tmp_path / "test_bm25.pkl.gz"
    index.save(save_path)
    assert save_path.exists()

    loaded = BM25Index.load(save_path)
    assert loaded.count() == index.count()

    query = chunks[0].text[:30]
    res_orig = index.query(query, k=3)
    res_loaded = loaded.query(query, k=3)

    assert [r["chunk_id"] for r in res_orig] == [r["chunk_id"] for r in res_loaded]
    assert [r["score"] for r in res_orig] == [r["score"] for r in res_loaded]


def test_bm25_edge_cases() -> None:
    """Edge cases: empty query, out of vocab, empty index, k <= 0."""
    empty_index = BM25Index()
    assert empty_index.count() == 0
    assert empty_index.query("test") == []

    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    chunks = chunk_paper(papers[0])
    index = BM25Index(chunks)

    # Empty string query
    assert index.query("") == []
    assert index.query("   ") == []

    # Non-alphanumeric query
    assert index.query("??? !!!") == []

    # k <= 0
    assert index.query("model", k=0) == []
    assert index.query("model", k=-5) == []

    # Filter by non-existent paper
    assert index.query("model", filter={"paper_id": "non_existent_paper"}) == []
