"""Unified logging infrastructure for FreeGSNKE.

Provides package-scoped standard logging utilities, a default console handler
targeting the current ``sys.stdout`` at ``INFO`` level, and helpers to
configure log levels, formatting, or pure-library behavior.

License & Attribution
---------------------
FreeGSNKE is distributed under the GNU Lesser General Public License version 3.
Copyright 2025 UKRI-STFC and UKAEA contributors.
"""

from __future__ import annotations

import logging
import sys
from typing import IO, Optional, Union

PACKAGE_LOGGER_NAME = "freegsnke"
DEFAULT_FORMAT = "%(levelname)s [%(name)s]: %(message)s"
SIMPLE_FORMAT = "%(message)s"

__all__ = [
    "DEFAULT_FORMAT",
    "PACKAGE_LOGGER_NAME",
    "SIMPLE_FORMAT",
    "disable_default_handler",
    "enable_default_handler",
    "get_logger",
    "set_log_level",
    "setup_logging",
]

_default_handler: Optional[logging.Handler] = None


class _DynamicStdoutStreamHandler(logging.StreamHandler):
    """Stream handler that dynamically resolves ``sys.stdout`` on each access.

    Resolving ``sys.stdout`` dynamically ensures that runtime stream
    redirections (such as ``contextlib.redirect_stdout``, ``pytest``'s
    ``capsys`` fixture, or notebook frontends) capture log output even when
    the handler was attached at import time.
    """

    def __init__(self) -> None:
        """Initialize the handler bound dynamically to ``sys.stdout``."""
        super().__init__()

    @property
    def stream(self) -> Optional[IO[str]]:
        """Return the current ``sys.stdout`` stream."""
        return sys.stdout

    @stream.setter
    def stream(self, _value: Optional[IO[str]]) -> None:
        """Ignore static stream assignment so ``sys.stdout`` remains dynamic."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record to ``sys.stdout`` if a stdout stream is available."""
        if self.stream is None:
            return
        super().emit(record)


def _normalize_level(level: Union[int, str]) -> int:
    """Validate and normalize a logging level to its integer value.

    Parameters
    ----------
    level : int or str
        Logging level as an integer (e.g. ``logging.DEBUG``) or case-insensitive
        level name (e.g. ``"DEBUG"``, ``"info"``, ``"WARNING"``).

    Returns
    -------
    int
        Normalized integer logging level.

    Raises
    ------
    ValueError
        If ``level`` is not a recognized logging level name or integer.
    """
    if isinstance(level, bool):
        raise ValueError(f"Invalid log level: {level!r}")
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
    """Return a logger hierarchically scoped under ``freegsnke``.

    Parameters
    ----------
    name : str, optional
        Name of the module or submodule. If omitted or equal to
        ``PACKAGE_LOGGER_NAME``, returns the root package logger.

    Returns
    -------
    logging.Logger
        Package-scoped logger instance.
    """
    if name is None or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def setup_logging(
    level: Union[int, str] = logging.INFO,
    stream: Optional[IO[str]] = None,
    fmt: Optional[str] = DEFAULT_FORMAT,
    force: bool = False,
) -> logging.Logger:
    """Configure the root ``freegsnke`` package logger.

    When a package-managed handler is attached, ``logger.propagate`` is set to
    ``False`` to avoid duplicate log emission when a host application also
    configures Python's root logger. Call :func:`disable_default_handler` to
    hand log routing over to the root logger (including ``pytest``'s ``caplog``
    fixture).

    Parameters
    ----------
    level : int or str, default=logging.INFO
        Logging level (e.g. ``logging.DEBUG``, ``"INFO"``, ``"WARNING"``).
    stream : file-like, optional
        Stream to direct log output to. When ``None`` (default), dynamically
        resolves ``sys.stdout`` on each emitted record.
    fmt : str, optional
        Logging format string. Defaults to :data:`DEFAULT_FORMAT`.
    force : bool, default=False
        If ``True``, clears all existing handlers on the package logger before
        attaching the new stream handler.

    Returns
    -------
    logging.Logger
        The configured ``freegsnke`` root logger.
    """
    global _default_handler
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    norm_level = _normalize_level(level)
    logger.setLevel(norm_level)

    if force:
        logger.handlers.clear()
        _default_handler = None
    else:
        for nh in [h for h in logger.handlers if isinstance(h, logging.NullHandler)]:
            logger.removeHandler(nh)
        if _default_handler is not None and _default_handler in logger.handlers:
            logger.removeHandler(_default_handler)
            _default_handler = None

    handler: logging.Handler
    if stream is None:
        handler = _DynamicStdoutStreamHandler()
    else:
        handler = logging.StreamHandler(stream)
    handler.setLevel(norm_level)
    handler.setFormatter(logging.Formatter(fmt or DEFAULT_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    _default_handler = handler

    return logger


def set_log_level(level: Union[int, str]) -> None:
    """Set the logging level across the ``freegsnke`` logger and its handlers.

    Parameters
    ----------
    level : int or str
        Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``, ``"DEBUG"``,
        ``"WARNING"``).
    """
    norm_level = _normalize_level(level)
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(norm_level)
    for handler in logger.handlers:
        handler.setLevel(norm_level)


def enable_default_handler() -> None:
    """Ensure the default ``sys.stdout`` handler is attached and isolated.

    Removes any ``NullHandler`` on the ``freegsnke`` logger, attaches a
    dynamic ``sys.stdout`` handler if the package-managed handler is absent,
    and sets ``logger.propagate = False`` to avoid duplicate output.
    """
    global _default_handler
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for nh in [h for h in logger.handlers if isinstance(h, logging.NullHandler)]:
        logger.removeHandler(nh)
    if _default_handler is None or _default_handler not in logger.handlers:
        target_level = logging.INFO if logger.level == logging.NOTSET else logger.level
        setup_logging(level=target_level, stream=None)
    else:
        logger.propagate = False


def disable_default_handler() -> None:
    """Switch ``freegsnke`` to pure-library logging behavior.

    Removes the package-managed default handler, attaches a
    :class:`logging.NullHandler` if needed, and sets ``logger.propagate = True``
    so host applications (e.g. ``logging.basicConfig``) and test fixtures such
    as ``pytest``'s ``caplog`` receive ``freegsnke`` log records.
    """
    global _default_handler
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if _default_handler is not None and _default_handler in logger.handlers:
        logger.removeHandler(_default_handler)
        _default_handler = None
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
    logger.propagate = True
