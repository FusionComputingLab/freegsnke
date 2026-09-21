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

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    if force:
        logger.handlers.clear()
        _default_handler = None

    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
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
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def enable_default_handler() -> None:
    """Ensure a default stdout handler is attached to the freegsnke logger if missing."""
    global _default_handler
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if not logger.handlers:
        setup_logging(level=logging.INFO, stream=sys.stdout)


def disable_default_handler() -> None:
    """Remove default handler and attach a NullHandler (pure library behavior)."""
    global _default_handler
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if _default_handler in logger.handlers:
        logger.removeHandler(_default_handler)
        _default_handler = None
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
