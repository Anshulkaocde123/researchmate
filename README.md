# ResearchMate

> **Citation-grounded RAG assistant over QASPER NLP papers.**

ResearchMate lets you ask natural-language questions about NLP research papers from the [QASPER](https://allenai.org/data/qasper) dataset and receive answers backed by exact paper citations.

## How it works

1. **Ingest** — Download and preprocess QASPER papers into structured chunks.
2. **Index** — Embed chunks with `BAAI/bge-base-en-v1.5` and store them in a Chroma vector store alongside a BM25 index.
3. **Retrieve** — Hybrid search (dense + sparse) followed by `BAAI/bge-reranker-base` cross-encoder reranking.
4. **Generate** — Build a citation-rich prompt and call Claude to produce a grounded answer.
5. **Evaluate** — Measure retrieval recall and answer quality against the QASPER gold annotations.
6. **UI** — Gradio web interface for interactive querying.

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Embeddings | fastembed (`BAAI/bge-base-en-v1.5` via ONNX) |
| Vector store | Chroma |
| Sparse retrieval | rank-bm25 |
| Reranker | fastembed cross-encoder (`BAAI/bge-reranker-base` via ONNX) |
| LLM | Google Gemini API (`gemini-2.5-flash`) |
| UI | Gradio |

## Quick start

```bash
# 1. Clone and create virtual environment
git clone <your-repo-url> && cd researchmate
python -m venv .venv && source .venv/bin/activate

# 2. Install with dev dependencies
pip install -e ".[dev]"

# 3. Configure secrets
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=<your-key>

# 4. Download QASPER dataset (~60 MB)
python scripts/download_data.py          # → data/raw/{train,validation,test}.jsonl
python scripts/inspect_schema.py        # inspect schema; output matches docs/data_schema.md

# 5. Run checks
make check
```

## Project layout

```
src/researchmate/
├── data/          # QASPER ingestion & preprocessing
├── chunking/      # Text chunking strategies
├── indexing/      # Chroma embedding & indexing
├── retrieval/     # Hybrid retrieval + reranker
├── generation/    # Prompt building & Claude calls
├── evaluation/    # Metrics (retrieval recall, answer quality)
└── ui/            # Gradio app
```

## Milestones

| Tag | Description |
|-----|-------------|
| `v0.1-data` | Data ingestion pipeline |
| `v0.2-retrieval` | Hybrid retrieval pipeline |
| `v0.3-rag` | RAG + Claude integration |
| `v0.4-eval` | Evaluation suite |
| `v1.0-deployed` | Gradio UI + deployment |

## License

MIT — see [LICENSE](LICENSE).
