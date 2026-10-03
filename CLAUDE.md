# ResearchMate — Project Context

> Paste this file at the start of every session to keep the assistant consistent across steps.

## Project Overview

**ResearchMate** is a citation-grounded RAG (Retrieval-Augmented Generation) assistant that answers questions over QASPER NLP papers, returning answers backed by exact paper citations.

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data validation | pydantic |
| Embeddings | fastembed (`BAAI/bge-base-en-v1.5` via ONNX) |
| Vector store | Chroma |
| Sparse retrieval | rank_bm25 |
| Reranker | fastembed cross-encoder (`BAAI/bge-reranker-base` via ONNX) |
| LLM | Google Gemini API (`gemini-2.5-flash`) |
| UI | Gradio |
| Testing | pytest |
| Linting / formatting | ruff |

## Repository Layout

```
researchmate/
├── CLAUDE.md               ← this file
├── Makefile                ← exposes `make check` (lint + tests)
├── pyproject.toml
├── src/
│   └── researchmate/       ← installable package (src layout)
│       ├── __init__.py
│       ├── data/           ← ingestion & preprocessing
│       ├── retrieval/      ← embedding, BM25, Chroma, reranker
│       ├── rag/            ← prompt building, Claude calls
│       └── ui/             ← Gradio app
├── tests/
│   ├── unit/               ← no network, no API calls; mock everything
│   └── integration/        ← skipped without ANTHROPIC_API_KEY
└── data/                   ← gitignored; raw + processed datasets
```

## Ground Rules

1. **One prompt = one commit.** Never stack two prompts into one commit.
2. **Gate before every commit:** run `make check` (lint + tests), do the Verify list, and read `git diff` yourself.
3. **Conventional commits:** `feat`, `fix`, `test`, `docs`, `chore`, `refactor`.
4. **Never commit** `data/`, `indexes/`, `.env`, or API keys.
5. **Unit tests never hit the network or the Claude API.** Mock them. Only tests marked `integration` may, and they skip without a key.
6. **Tag milestones:** `v0.1-data`, `v0.2-retrieval`, `v0.3-rag`, `v0.4-eval`, `v1.0-deployed`. Push after every phase.
7. **If a step fails,** use the Recovery prompts at the end instead of moving on.

## Coding Conventions

- **`src` layout** — the package lives under `src/researchmate/`.
- **Type hints and docstrings** on every public function, class, and method.
- **Every module has tests** — aim for ≥ 80 % line coverage.
- **No network calls in unit tests** — use `unittest.mock` or `pytest-mock`.
- **Fixed random seeds** wherever stochastic behaviour must be reproducible.
- **Use `logging`**, not `print`, for all runtime output.
- **Never hardcode secrets or absolute paths** — use environment variables and `pathlib.Path`.
- **Do not touch files outside the current step's scope.**

## Step Checklist (append as steps complete)

| Step | Tag | Status |
|------|-----|--------|
| Step 0 — CLAUDE.md | — | ✅ done |
| Step 1 — repo scaffold | — | ✅ done |
| Step 2 — data ingestion | v0.1-data | ✅ done |
| Step 3 — retrieval pipeline | v0.2-retrieval | ✅ done |
| Step 4 — RAG + Claude | v0.3-rag | ✅ done |
| Step 5 — evaluation | v0.4-eval | ✅ done |
| Step 6 — Gradio UI + deploy | v1.0-deployed | ✅ done |

## At the End of Each Step

List **files changed** and the **exact commands to verify**:

```
Files changed:
  src/researchmate/...
  tests/unit/...

Verify:
  make check
  python -m pytest tests/unit/ -v
```

## Recovery Prompts

If a step fails, open a new session, paste this file, and use one of:

- `"Step N failed at <description>. Error: <paste error>. Fix only the failing part, do not move on."`
- `"Revert Step N changes and start over from the Verify list."`
