"""Typed project-wide settings loaded from the environment or a .env file.

Usage
-----
Instantiate once and pass around, or import the module-level singleton::

    from researchmate.config import settings

    key = settings.get_gemini_api_key()   # raises ValueError if not set

Environment variables (all optional except GEMINI_API_KEY for generation)
--------------------------------------------------------------------------
GEMINI_API_KEY     – Gemini API key (required for answer generation)
DATA_DIR           – override default data directory (default: "data")
INDEX_DIR          – override default index directory (default: "indexes")
EMBEDDING_MODEL    – fastembed model name
RERANKER_MODEL     – fastembed cross-encoder model name
LLM_MODEL          – Gemini model name
CHUNK_SIZE         – tokens per chunk
CHUNK_OVERLAP      – overlap tokens between chunks
TOP_K_DENSE        – dense retrieval candidate count
TOP_K_SPARSE       – sparse (BM25) retrieval candidate count
TOP_K_RERANK       – passages kept after reranking
RANDOM_SEED        – global random seed
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """All ResearchMate configuration in one place.

    Values are resolved in priority order:
    1. Constructor keyword arguments
    2. Environment variables
    3. `.env` file (project root)
    4. Field defaults declared here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars silently
        populate_by_name=True,  # allow both alias and field name
    )

    # ------------------------------------------------------------------ Paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for raw and processed datasets.",
    )
    index_dir: Path = Field(
        default=Path("indexes"),
        description="Root directory for Chroma and BM25 indexes.",
    )

    # -------------------------------------------------------------- Model IDs
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="fastembed model used for dense embeddings.",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="fastembed cross-encoder model used for reranking.",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Google Gemini model name used for answer generation.",
    )

    # --------------------------------------------------------------- Chunking
    chunk_size: int = Field(
        default=512,
        ge=64,
        description="Target token count per text chunk.",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Token overlap between adjacent chunks to preserve context.",
    )

    # --------------------------------------------------------------- Retrieval
    top_k_dense: int = Field(
        default=20,
        ge=1,
        description="Number of candidate passages from dense (Chroma) retrieval.",
    )
    top_k_sparse: int = Field(
        default=20,
        ge=1,
        description="Number of candidate passages from sparse (BM25) retrieval.",
    )
    top_k_rerank: int = Field(
        default=5,
        ge=1,
        description="Number of passages kept after cross-encoder reranking.",
    )

    # --------------------------------------------------------- Reproducibility
    random_seed: int = Field(
        default=42,
        description="Global random seed for reproducible results.",
    )

    # --------------------------------------------------------------- API keys
    gemini_api_key: str | None = Field(
        default=None,
        description=(
            "Google Gemini API key.  Set via the GEMINI_API_KEY environment "
            "variable or .env file.  Missing at import time is fine; a clear "
            "error is raised only when the key is actually requested."
        ),
    )

    # ---------------------------------------------------------------- Methods
    def get_gemini_api_key(self) -> str:
        """Return the Gemini API key, raising a descriptive error if absent.

        Raises
        ------
        ValueError
            When ``GEMINI_API_KEY`` is not set in the environment or ``.env``.
        """
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set.\n"
                "  • Copy .env.example → .env and fill in your key, or\n"
                "  • export GEMINI_API_KEY=<your-key> in your shell.\n"
                "The key is only required for answer generation."
            )
        return self.gemini_api_key


# ---------------------------------------------------------------------------
# Module-level singleton — import and use directly in the rest of the project.
# The API key is optional at load time; missing it does NOT crash here.
# ---------------------------------------------------------------------------
settings: Settings = Settings()
