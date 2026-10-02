"""Smoke tests for the researchmate package root."""

import researchmate


def test_package_importable() -> None:
    """Package can be imported and exposes a version string."""
    assert isinstance(researchmate.__version__, str)
    assert researchmate.__version__ != ""


def test_version_format() -> None:
    """Version follows basic semver major.minor.patch format."""
    parts = researchmate.__version__.split(".")
    assert len(parts) == 3, f"Expected 3 version parts, got: {researchmate.__version__}"
    assert all(part.isdigit() for part in parts)
