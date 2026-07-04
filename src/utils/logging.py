"""Logging utilities.

Provides a single ``get_logger()`` function so all modules emit
structured, consistent log messages.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a pre-configured logger with a simple console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
