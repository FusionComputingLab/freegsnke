"""Unit tests for the unified FreeGSNKE logging infrastructure."""

from __future__ import annotations

import contextlib
import io
import logging

import pytest

import freegsnke
from freegsnke.logging import (
    DEFAULT_FORMAT,
    SIMPLE_FORMAT,
    disable_default_handler,
    enable_default_handler,
    get_logger,
    set_log_level,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Restore default FreeGSNKE logging configuration after each test."""
    yield
    setup_logging(level=logging.INFO, stream=None, fmt=DEFAULT_FORMAT, force=True)


def test_top_level_exports_and_default_state():
    """Verify package-level logging exports and import-time defaults."""
    assert freegsnke.get_logger is get_logger
    assert freegsnke.setup_logging is setup_logging
    assert freegsnke.set_log_level is set_log_level
    assert freegsnke.enable_default_handler is enable_default_handler
    assert freegsnke.disable_default_handler is disable_default_handler
    assert freegsnke.DEFAULT_FORMAT == DEFAULT_FORMAT
    assert freegsnke.SIMPLE_FORMAT == SIMPLE_FORMAT

    root_logger = get_logger()
    assert root_logger.name == "freegsnke"
    assert root_logger.level == logging.INFO
    assert root_logger.propagate is False


def test_get_logger_hierarchy():
    """Verify get_logger scopes names hierarchically under freegsnke."""
    assert get_logger().name == "freegsnke"
    assert get_logger("freegsnke").name == "freegsnke"
    assert get_logger("freegsnke.submodule").name == "freegsnke.submodule"
    assert get_logger("submodule").name == "freegsnke.submodule"


def test_set_log_level_validation():
    """Verify set_log_level accepts strings/ints and rejects invalid values."""
    root_logger = get_logger()

    set_log_level("DEBUG")
    assert root_logger.level == logging.DEBUG
    assert all(h.level == logging.DEBUG for h in root_logger.handlers)

    set_log_level("warning")
    assert root_logger.level == logging.WARNING

    set_log_level(logging.ERROR)
    assert root_logger.level == logging.ERROR

    with pytest.raises(ValueError, match="Invalid log level"):
        set_log_level("INVALID_LEVEL")

    with pytest.raises(ValueError, match="Invalid log level"):
        set_log_level(True)


def test_dynamic_stdout_capture(capsys):
    """Verify default stdout handler follows dynamic sys.stdout redirection."""
    logger = get_logger("freegsnke.test_dynamic")
    logger.info("captured by capsys")
    captured = capsys.readouterr()
    assert "INFO [freegsnke.test_dynamic]: captured by capsys" in captured.out
    assert captured.err == ""

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        logger.warning("captured by redirect_stdout")
    assert (
        "WARNING [freegsnke.test_dynamic]: captured by redirect_stdout"
        in buf.getvalue()
    )


def test_setup_logging_custom_stream_format_and_force():
    """Verify setup_logging configures custom streams, formats, and force."""
    stream = io.StringIO()
    setup_logging(level="DEBUG", stream=stream, fmt=SIMPLE_FORMAT)

    logger = get_logger("freegsnke.test_stream")
    logger.debug("debug message test")
    logger.info("info message test")

    output = stream.getvalue()
    assert output.splitlines() == ["debug message test", "info message test"]

    root_logger = get_logger()
    extra_handler = logging.NullHandler()
    root_logger.addHandler(extra_handler)
    assert extra_handler in root_logger.handlers

    setup_logging(level="INFO", stream=io.StringIO(), force=True)
    assert extra_handler not in root_logger.handlers
    assert len(root_logger.handlers) == 1


def test_enable_disable_default_handler_and_propagate():
    """Verify toggling default handler updates handlers and propagate flag."""
    root_logger = get_logger()

    disable_default_handler()
    assert root_logger.propagate is True
    assert any(isinstance(h, logging.NullHandler) for h in root_logger.handlers)
    # Idempotent
    handlers_after_disable = list(root_logger.handlers)
    disable_default_handler()
    assert len(root_logger.handlers) == len(handlers_after_disable)

    enable_default_handler()
    assert root_logger.propagate is False
    assert not any(isinstance(h, logging.NullHandler) for h in root_logger.handlers)
    # Idempotent
    handlers_after_enable = list(root_logger.handlers)
    enable_default_handler()
    assert len(root_logger.handlers) == len(handlers_after_enable)


def test_no_duplicate_logs_with_root_handler(capsys):
    """Verify propagate=False prevents duplicate output with root handlers."""
    root_py_logger = logging.getLogger()
    extra_stream = io.StringIO()
    extra_handler = logging.StreamHandler(extra_stream)
    root_py_logger.addHandler(extra_handler)
    try:
        logger = get_logger("freegsnke.test_isolation")
        logger.info("isolated message")
    finally:
        root_py_logger.removeHandler(extra_handler)

    captured = capsys.readouterr()
    assert captured.out.count("isolated message") == 1
    assert extra_stream.getvalue() == ""


def test_caplog_capture_after_disable_default_handler(caplog):
    """Verify caplog captures records once pure-library mode is enabled."""
    disable_default_handler()
    logger = logging.getLogger("freegsnke.GSstaticsolver")
    with caplog.at_level(logging.INFO, logger="freegsnke"):
        logger.info("Inverse solve completed successfully.")
        logger.debug("This debug message should not appear at INFO level.")

    assert "Inverse solve completed successfully." in caplog.text
    assert "This debug message should not appear at INFO level." not in caplog.text
