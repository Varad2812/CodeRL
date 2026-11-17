"""Structured JSON logging via structlog."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(level: str | None = None) -> None:
    """Configure structlog to emit JSON lines to stdout."""
    global _configured
    level_name = (level or os.environ.get("CODERL_LOG_LEVEL", "INFO")).upper()
    level_no = logging.getLevelNamesMapping().get(level_name, logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str) -> Any:
    """Return a structlog logger bound with ``logger=name``."""
    if not _configured:
        configure_logging()
    return structlog.get_logger().bind(logger=name)
