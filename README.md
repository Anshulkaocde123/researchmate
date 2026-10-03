# ResearchMate

> **Citation-Grounded RAG Assistant over Scientific NLP Papers.**

ResearchMate enables researchers and engineers to query scientific NLP papers from the [QASPER dataset](https://allenai.org/data/qasper) with natural-language questions, returning verifiable answers backed by exact, bracketed passage citations `[Passage X]` and section references.

---

## 🏛️ System Architecture

```text
[ User Question ] + [ Target Paper ID ]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[ Dense Retrieval ]             [ Sparse Retrieval ]
FastEmbed BGE-Base-en-v1.5      BM25Okapi Token Index
Chroma Vector Store (Cosine)    Gzip Pickle Inverted Index
Top-20 Dense Candidates         Top-20 Sparse Candidates
       │                                 │
       └────────────────┬────────────────┘
                        ▼
       [ Reciprocal Rank Fusion (RRF) ]
       Score(d) = Σ 1 / (60 + Rank(d))
                        │
                        ▼
       [ Cross-Encoder Reranker ]
       FastEmbed BAAI/bge-reranker-base (ONNX)
       Top-5 Reranked Context Passages
                        │
                        ▼
       [ Citation Grounding & Prompt Builder ]
       Strict Rules: Factual Only | Cite Passages | Refuse if Unanswerable
                        │
                        ▼
       [ Google Gemini 2.5 Flash Generator ]
       Grounded Response with Verbatim [Passage X] Citations
                        │
                        ▼
       [ Citation Extraction & Validation ]
       Validate Passage Bounds • Map to Paper Sections • Detect Unanswerable
```

---

## ⚡ Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.11 / 3.12 | Standard modern scientific Python |
| **Embeddings** | FastEmbed (`BAAI/bge-base-en-v1.5`) | ONNX CPU runtime; zero PyTorch bloat; deterministic SQLite cache |
| **Vector Store** | ChromaDB (`chromadb`) | Persistent cosine vector collection with metadata filtering |
| **Sparse Retrieval** | `rank-bm25` (BM25Okapi) | Exact vocabulary matching; 34,280 chunks indexed in <10 seconds |
| **Reranking** | FastEmbed (`BAAI/bge-reranker-base`) | Cross-encoder cross-attention over `(query, passage)` pairs |
| **LLM Generation** | Google Gemini (`gemini-2.5-flash`) | Fast, cost-efficient, high-context generation via `google-genai` |
| **UI** | Gradio 6.x | Interactive web app with paper selector, evidence viewer, and stats |
| **Code Quality** | Pytest, Ruff, Tenacity, Pydantic | 93% test coverage (180+ tests), zero network calls in unit tests |

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/anshuljain/ResearchMate.git
cd ResearchMate

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dev dependencies in editable mode
pip install -e ".[dev]"
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env and set your Google Gemini API key:
# GEMINI_API_KEY="AIzaSy..."
```

*(Note: Retrieval, chunking, indexing, and unit tests work 100% locally without an API key. The API key is only needed for generative LLM answering).*

---

## 🔄 Pipeline Execution

### Step 1: Data Ingestion & Profiling
Downloads QASPER splits, validates SHA-256 hashes, normalizes LaTeX/BibTeX artifacts, and creates a stratified 200-question evaluation set.
```bash
python scripts/download_data.py
python scripts/profile_data.py
```

### Step 2: Section-Aware Chunking & Evidence Grounding
Splits paper paragraphs into overlapping chunks (target: 512 tokens) preserving section metadata and maps gold answer evidence strings to chunk IDs.
```bash
python scripts/build_chunks.py
```

### Step 3: Index Construction (Chroma + BM25)
Builds the BM25Okapi sparse index and Chroma dense vector store.
```bash
# Build BM25 sparse index (full corpus: 34,280 chunks across 1,584 papers in ~10s)
python scripts/build_index.py --mode bm25

# Build Chroma vector index for evaluation papers (or limit with --limit-papers)
python scripts/build_index.py --mode vector --limit-papers 10
```

### Step 4: Run Information Retrieval Evaluation
Evaluates retrieval accuracy (HitRate@k, Recall@k, MRR@k, NDCG@k) across gold annotations.
```bash
python scripts/run_eval.py
```

### Step 5: Launch Gradio Web UI
```bash
# Run CLI script
researchmate
# Or directly via Python
python -m researchmate.ui.app
```
Access the application in your browser at `http://localhost:7860`.

---

## 📊 Empirical Evaluation Results

Evaluated on QASPER stratified benchmark (`reports/retrieval_eval.md`):

| Metric | BM25 (Sparse) | Hybrid (BM25 + Chroma + Rerank) |
|---|:---:|:---:|
| **HitRate@1** | 32.0% | 28.0% |
| **HitRate@3** | 60.0% | 60.0% |
| **HitRate@5** | **88.0%** | **84.0%** |
| **HitRate@10** | **92.0%** | **92.0%** |
| **HitRate@20** | **96.0%** | **96.0%** |
| **MRR@10** | 0.505 | 0.494 |
| **NDCG@10** | 0.571 | **0.572** |

---

## 🧪 Testing & Code Quality

```bash
# Run all linter and formatting checks
make check

# Or directly:
ruff check src tests scripts
ruff format --check src tests scripts
pytest tests/unit/ -v --cov=researchmate
```

- **Unit tests:** 180 passed in ~4s (100% mock-isolated, zero network calls).
- **Code coverage:** **93% line coverage** across all modules.

---

## 💡 Interview Defense & Key Engineering Decisions

### 1. Why FastEmbed (ONNX) instead of PyTorch / HuggingFace Transformers?
- **Zero PyTorch Dependency:** Avoids downloading 2GB+ PyTorch binaries and CUDA dependencies.
- **CPU Optimized:** ONNX Runtime provides multi-threaded quantized execution, running seamlessly in lightweight production containers.
- **Cost & Reliability:** Eliminates GPU server requirements for embedding generation.

### 2. Why Hybrid Retrieval (BM25 + Dense + RRF) instead of Pure Vector Search?
- Scientific text is dense with exact domain terms, abbreviations, acronyms, and dataset names (e.g., *Switchboard*, *LORELEI*, *BLEU*, *F1*, *ASR*). Dense embeddings frequently suffer from vocabulary mismatch on specialized proper nouns.
- BM25 excels at exact keyword matching, while dense embeddings capture semantic similarity. Combining them via **Reciprocal Rank Fusion (RRF)** avoids score calibration issues and yields **96% HitRate@20**.

### 3. How Are Hallucinations and Mis-Citations Prevented?
- The system prompt enforces strict negative constraints: the model is prohibited from using external knowledge and must respond with an explicit unanswerability phrase if evidence is absent.
- Generated responses are parsed by `extract_citations()`, which validates that cited bracket numbers `[Passage X]` resolve to actual retrieved chunks within the paper.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
