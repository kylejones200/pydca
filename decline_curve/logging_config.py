"""Logging configuration for decline_curve package."""

import logging
import sys
from typing import Optional


def configure_logging(
    level: int = logging.WARNING,
    format_string: Optional[str] = None,
    stream: Optional[object] = None,
) -> None:
    """Configure logging for the decline_curve package.

    Args:
        level: Logging level (default: WARNING)
        format_string: Custom format string (default: standard format)
        stream: Output stream (default: stderr)

    Example:
        >>> from decline_curve.logging_config import configure_logging
        >>> import logging
        >>> configure_logging(level=logging.INFO)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if stream is None:
        stream = sys.stderr

    logging.basicConfig(
        level=level,
        format=format_string,
        stream=stream,
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
