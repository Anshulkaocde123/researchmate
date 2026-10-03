.PHONY: check lint format test install clean

# ── Install ──────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Lint ─────────────────────────────────────────────────────────────────────
lint:
	ruff check src tests
	ruff format --check src tests

# ── Format (auto-fix) ────────────────────────────────────────────────────────
format:
	ruff check --fix src tests
	ruff format src tests

# ── Test ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/unit/ -v --cov=researchmate --cov-report=term-missing

# ── Gate (lint + unit tests) — run before every commit ───────────────────────
check: lint test

# ── Integration tests (require GEMINI_API_KEY) ───────────────────────────────
test-integration:
	pytest tests/integration/ -v -m integration

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
