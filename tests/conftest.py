"""Root pytest configuration: fixtures and automatic marker handling.

Integration-test auto-skip
--------------------------
Any test decorated with ``@pytest.mark.integration`` is **skipped automatically**
unless the ``GEMINI_API_KEY`` environment variable is set.  This keeps the
default ``make check`` / CI run fast and free of network calls while still
allowing integration tests to run in environments that have the key.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip integration tests when GEMINI_API_KEY is absent."""
    if os.getenv("GEMINI_API_KEY"):
        return  # key present → run all tests including integration

    skip = pytest.mark.skip(reason="Set GEMINI_API_KEY to run integration tests")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
