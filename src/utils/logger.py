"""
Rich-based coloured logger with optional file output.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello world")

Logs are written to:
  - Console  : Rich-formatted, coloured output (stderr)
  - File     : logs/ddr.log (plain-text, auto-rotated at 5 MB, keeps last 3 files)
"""

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from logging.handlers import RotatingFileHandler

_console = Console(stderr=True)

# Map of already-created loggers so we don't add duplicate handlers
_loggers: dict[str, logging.Logger] = {}

LOG_FORMAT       = "%(message)s"
FILE_LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT      = "%H:%M:%S"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Resolve log directory relative to the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_LOG_DIR      = _PROJECT_ROOT / "logs"
_LOG_FILE     = _LOG_DIR / "ddr.log"


def _ensure_log_dir() -> bool:
    """Create logs/ directory if it doesn't exist. Returns True on success."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return (or create) a logger for the given module name.

    Handlers attached:
      - RichHandler  → coloured console output (stderr)
      - RotatingFileHandler → logs/ddr.log (5 MB max, 3 backups)
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # ── Console handler ────────────────────────────────────────
        console_handler = RichHandler(
            console=_console,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # ── File handler (rotating) ────────────────────────────────
        if _ensure_log_dir():
            try:
                file_handler = RotatingFileHandler(
                    filename=str(_LOG_FILE),
                    maxBytes=5 * 1024 * 1024,   # 5 MB per file
                    backupCount=3,               # keep ddr.log, ddr.log.1, ddr.log.2, ddr.log.3
                    encoding="utf-8",
                )
                file_handler.setFormatter(
                    logging.Formatter(FILE_LOG_FORMAT, datefmt=FILE_DATE_FORMAT)
                )
                file_handler.setLevel(level)
                logger.addHandler(file_handler)
            except Exception as exc:
                # Non-fatal — if we can't open the log file, console logging still works
                logger.warning(f"Could not set up file logging to {_LOG_FILE}: {exc}")

    # Don't propagate to root — we own the output
    logger.propagate = False

    _loggers[name] = logger
    return logger
