"""Consistent logging configuration for ResearchMate.

Call :func:`configure_logging` once at application startup (e.g. in the
Gradio app entry point or a CLI script).  All other modules obtain their
logger via ``logging.getLogger(__name__)`` as usual.

Example
-------
::

    from researchmate.logging_setup import configure_logging
    configure_logging()  # INFO to stderr, noisy deps silenced
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

__all__ = ["configure_logging"]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Third-party loggers that are chatty at INFO level; silence them to WARNING.
_NOISY_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "chromadb",
    "urllib3",
    "huggingface_hub",
    "filelock",
    "datasets",
    "google",
)


def configure_logging(
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DATE_FORMAT,
    stream: TextIO = sys.stderr,
) -> None:
    """Configure project-wide logging with a single call.

    Parameters
    ----------
    level:
        Root logger level (default: ``logging.INFO``).
    fmt:
        Log record format string.
    datefmt:
        Date/time format string (strftime).
    stream:
        Output stream for the ``StreamHandler`` (default: ``sys.stderr``).

    Notes
    -----
    * Calling this function multiple times is safe — ``force=True`` replaces
      any existing handlers on the root logger.
    * Noisy third-party loggers listed in ``_NOISY_LOGGERS`` are clamped to
      ``WARNING`` regardless of the requested ``level``.
    """
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        stream=stream,
        force=True,  # replace existing handlers; safe to call repeatedly
    )

    # Clamp noisy third-party libraries without fully silencing them.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Logging configured: level=%s", logging.getLevelName(level))
