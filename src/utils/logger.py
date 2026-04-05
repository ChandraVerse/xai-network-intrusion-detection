"""Centralised logging configuration for XAI-NIDS.

Provides a ``get_logger(name)`` factory that returns a consistently
formatted ``logging.Logger``.  All modules in this project use this
instead of calling ``logging.basicConfig`` directly.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Model loaded.")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Track whether the root handler has been installed already.
_ROOT_CONFIGURED = False


def _configure_root() -> None:
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _ROOT_CONFIGURED = True


def get_logger(
    name: str,
    level: int = logging.INFO,
    propagate: bool = True,
) -> logging.Logger:
    """Return a named logger with a consistent format.

    Args:
        name:      Logger name — typically ``__name__`` of the calling module.
        level:     Logging level for this specific logger (default INFO).
        propagate: Whether to propagate to the root logger (default True).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    _configure_root()
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = propagate
    return log


def set_global_level(level: int) -> None:
    """Change the log level of the root logger at runtime.

    Useful in scripts/notebooks where you want verbose DEBUG output::

        from src.utils.logger import set_global_level
        import logging
        set_global_level(logging.DEBUG)
    """
    logging.getLogger().setLevel(level)
