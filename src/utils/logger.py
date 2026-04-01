"""
Rich-based coloured logger.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello world")
"""

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)

# Map of already-created loggers so we don't add duplicate handlers
_loggers: dict[str, logging.Logger] = {}

LOG_FORMAT = "%(message)s"
DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return (or create) a Rich-backed logger for the given module name."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times if the root logger already has them
    if not logger.handlers:
        handler = RichHandler(
            console=_console,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)

    # Don't propagate to root — we own the output
    logger.propagate = False

    _loggers[name] = logger
    return logger
