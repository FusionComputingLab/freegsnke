"""Unified logging infrastructure for FreeGSNKE.

Provides standard logging utilities, sensible defaults (streaming to stdout),
and helpers to configure log levels throughout the package.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Union

PACKAGE_LOGGER_NAME = "freegsnke"
DEFAULT_FORMAT = "%(levelname)s [%(name)s]: %(message)s"
SIMPLE_FORMAT = "%(message)s"

_default_handler: Optional[logging.Handler] = None


def _normalize_level(level: Union[int, str]) -> int:
    """Validate and normalize logging level to integer."""
    if isinstance(level, str):
        level_name = level.upper()
        if hasattr(logging, level_name):
            val = getattr(logging, level_name)
            if isinstance(val, int):
                return val
        raise ValueError(f"Invalid log level: {level!r}")
    if isinstance(level, int):
        return level
    raise ValueError(f"Invalid log level: {level!r}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger hierarchically scoped under freegsnke.

    Parameters
    ----------
    name : str, optional
        Name of the module or submodule. If omitted or equal to
        PACKAGE_LOGGER_NAME, returns the root package logger.

    Returns
    -------
    logging.Logger
    """
    if name is None or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def setup_logging(
    level: Union[int, str] = logging.INFO,
    stream=sys.stdout,
    fmt: Optional[str] = DEFAULT_FORMAT,
    force: bool = False,
) -> logging.Logger:
    """Configure the root freegsnke logger.

    Parameters
    ----------
    level : int or str, default logging.INFO
        Logging level (e.g. logging.DEBUG, "INFO", "WARNING").
    stream : file-like, default sys.stdout
        Stream to direct log output to.
    fmt : str, optional
        Logging format string. Defaults to DEFAULT_FORMAT.
    force : bool, default False
        If True, clears existing handlers on the package logger before
        adding the new stream handler.

    Returns
    -------
    logging.Logger
        The configured freegsnke root logger.
    """
    global _default_handler
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    norm_level = _normalize_level(level)
    logger.setLevel(norm_level)

    if force:
        logger.handlers.clear()
        _default_handler = None
    elif _default_handler is not None and _default_handler in logger.handlers:
        logger.removeHandler(_default_handler)
        _default_handler = None

    handler = logging.StreamHandler(stream)
    handler.setLevel(norm_level)
    handler.setFormatter(logging.Formatter(fmt or DEFAULT_FORMAT))
    logger.addHandler(handler)
    _default_handler = handler

    return logger


def set_log_level(level: Union[int, str]) -> None:
    """Set the logging level across the freegsnke package and its handlers.

    Parameters
    ----------
    level : int or str
        Logging level (e.g. logging.DEBUG, logging.INFO, "DEBUG", "WARNING").
    """
    norm_level = _normalize_level(level)
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(norm_level)
    for handler in logger.handlers:
        handler.setLevel(norm_level)


def enable_default_handler() -> None:
    """Ensure a default stdout handler is attached to the freegsnke logger if missing."""
    global _default_handler
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    # Remove NullHandler if present
    for nh in [h for h in logger.handlers if isinstance(h, logging.NullHandler)]:
        logger.removeHandler(nh)
    if _default_handler is None or _default_handler not in logger.handlers:
        setup_logging(level=logger.level or logging.INFO, stream=sys.stdout)


def disable_default_handler() -> None:
    """Remove default handler and attach a NullHandler (pure library behavior)."""
    global _default_handler
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if _default_handler is not None and _default_handler in logger.handlers:
        logger.removeHandler(_default_handler)
        _default_handler = None
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
