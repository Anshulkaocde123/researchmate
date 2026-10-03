"""Integration smoke test — skipped automatically when GEMINI_API_KEY is absent.

This file proves the @pytest.mark.integration + conftest auto-skip wiring works.
Real end-to-end tests will be added in later steps (generation, evaluation).
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_integration_marker_is_wired() -> None:
    """Placeholder: confirms the integration marker and conftest skip logic work.

    When GEMINI_API_KEY is absent this test is skipped (not failed).
    When the key is present it runs and passes trivially.
    """
    assert True, "integration smoke test — replace with real API call in Step 4"
