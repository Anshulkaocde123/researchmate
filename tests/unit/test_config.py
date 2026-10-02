"""Unit tests for researchmate.config and researchmate.logging_setup.

All tests are fully offline — no network calls, no .env dependency.
Each Settings() instance is created with ``_env_file=None`` to prevent the
real .env file (if present) from leaking into test assertions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from researchmate.config import Settings
from researchmate.logging_setup import configure_logging

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_settings(monkeypatch: pytest.MonkeyPatch, **extra_env: str) -> Settings:
    """Create a Settings instance isolated from the real environment.

    Clears GEMINI_API_KEY (and any extra env vars) then injects *extra_env*,
    so each test starts from a known baseline.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_embedding_model_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.embedding_model == "BAAI/bge-base-en-v1.5"

    def test_reranker_model_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.reranker_model == "BAAI/bge-reranker-base"

    def test_llm_model_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.llm_model == "gemini-2.5-flash"

    def test_chunk_size_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.chunk_size == 512

    def test_chunk_overlap_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.chunk_overlap == 64

    def test_top_k_dense_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.top_k_dense == 20

    def test_top_k_sparse_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.top_k_sparse == 20

    def test_top_k_rerank_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.top_k_rerank == 5

    def test_random_seed_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.random_seed == 42

    def test_api_key_none_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.gemini_api_key is None

    def test_paths_are_path_objects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert isinstance(s.data_dir, Path)
        assert isinstance(s.index_dir, Path)

    def test_data_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.data_dir == Path("data")

    def test_index_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch)
        assert s.index_dir == Path("indexes")


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    def test_chunk_size_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch, CHUNK_SIZE="256")
        assert s.chunk_size == 256

    def test_chunk_overlap_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch, CHUNK_OVERLAP="128")
        assert s.chunk_overlap == 128

    def test_llm_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch, LLM_MODEL="gemini-2.0-flash")
        assert s.llm_model == "gemini-2.0-flash"

    def test_top_k_rerank_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch, TOP_K_RERANK="10")
        assert s.top_k_rerank == 10

    def test_random_seed_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch, RANDOM_SEED="0")
        assert s.random_seed == 0

    def test_data_dir_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch, DATA_DIR="/tmp/mydata")
        assert s.data_dir == Path("/tmp/mydata")

    def test_embedding_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch, EMBEDDING_MODEL="BAAI/bge-small-en-v1.5")
        assert s.embedding_model == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# API key — lazy validation
# ---------------------------------------------------------------------------


class TestApiKey:
    def test_import_does_not_crash_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing the module (or instantiating Settings) must not raise."""
        # If this raises, the test fails — that's the assertion.
        s = _clean_settings(monkeypatch)
        assert s.gemini_api_key is None

    def test_missing_key_raises_on_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_gemini_api_key() must raise ValueError with a helpful message."""
        s = _clean_settings(monkeypatch)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            s.get_gemini_api_key()

    def test_error_message_contains_guidance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The error should tell the user how to fix the problem."""
        s = _clean_settings(monkeypatch)
        with pytest.raises(ValueError, match=r"\.env"):
            s.get_gemini_api_key()

    def test_key_returned_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch, GEMINI_API_KEY="test-key-abc123")
        assert s.get_gemini_api_key() == "test-key-abc123"

    def test_key_stored_on_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _clean_settings(monkeypatch, GEMINI_API_KEY="sk-test")
        assert s.gemini_api_key == "sk-test"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


class TestLoggingSetup:
    def test_sets_root_level_info(self) -> None:
        configure_logging(level=logging.INFO)
        assert logging.getLogger().level == logging.INFO

    def test_sets_root_level_debug(self) -> None:
        configure_logging(level=logging.DEBUG)
        assert logging.getLogger().level == logging.DEBUG

    def test_silences_httpx(self) -> None:
        configure_logging(level=logging.DEBUG)
        assert logging.getLogger("httpx").level == logging.WARNING

    def test_silences_chromadb(self) -> None:
        configure_logging(level=logging.DEBUG)
        assert logging.getLogger("chromadb").level == logging.WARNING

    def test_silences_urllib3(self) -> None:
        configure_logging(level=logging.INFO)
        assert logging.getLogger("urllib3").level == logging.WARNING

    def test_idempotent_repeated_calls(self) -> None:
        """Calling configure_logging twice must not raise or duplicate handlers."""
        configure_logging(level=logging.INFO)
        configure_logging(level=logging.WARNING)
        assert logging.getLogger().level == logging.WARNING
        # force=True means only one StreamHandler on root
        stream_handlers = [
            h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 1
