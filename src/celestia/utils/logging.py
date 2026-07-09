"""Structured logging for the entire pipeline.

Call :func:`setup_logging` once at the CLI entry point.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a root logger with a clean format.

    Format::

        2025-07-10 14:30:01 | INFO     | Loaded 12 books
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called more than once
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers = [handler]
