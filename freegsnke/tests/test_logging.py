"""
Tests for unified FreeGSNKE logging architecture and verification that all
logging is captured via the standard logging module.
"""

import io
import logging
import os
import re

import pytest

import freegsnke
from freegsnke.logging import (
    DEFAULT_FORMAT,
    disable_default_handler,
    enable_default_handler,
    get_logger,
    set_log_level,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Ensure logging state is clean before and after each test."""
    root_logger = get_logger()
    original_level = root_logger.level
    yield
    # Restore default handler and INFO level
    set_log_level(logging.INFO)
    enable_default_handler()


def test_get_logger():
    """Test get_logger returns appropriate loggers."""
    root_logger = get_logger()
    assert isinstance(root_logger, logging.Logger)
    assert root_logger.name == "freegsnke"

    child_logger = get_logger("freegsnke.submodule")
    assert isinstance(child_logger, logging.Logger)
    assert child_logger.name == "freegsnke.submodule"


def test_set_log_level():
    """Test set_log_level accepts strings and integers, and validates inputs."""
    root_logger = get_logger()

    set_log_level("DEBUG")
    assert root_logger.level == logging.DEBUG

    set_log_level("warning")
    assert root_logger.level == logging.WARNING

    set_log_level(logging.ERROR)
    assert root_logger.level == logging.ERROR

    with pytest.raises(ValueError, match="Invalid log level"):
        set_log_level("INVALID_LEVEL")


def test_setup_logging_stream():
    """Test setup_logging configures output to a custom stream and format."""
    stream = io.StringIO()
    setup_logging(level="DEBUG", stream=stream, fmt="%(levelname)s - %(message)s")

    logger = get_logger("freegsnke.test_stream")
    logger.debug("debug message test")
    logger.info("info message test")

    output = stream.getvalue()
    assert "DEBUG - debug message test" in output
    assert "INFO - info message test" in output


def test_enable_and_disable_default_handler():
    """Test enabling and disabling the default handler."""
    root_logger = get_logger()

    disable_default_handler()
    # Should not raise on multiple calls
    disable_default_handler()

    enable_default_handler()
    # Idempotent - should not add duplicate handlers
    handlers_before = list(root_logger.handlers)
    enable_default_handler()
    assert len(root_logger.handlers) == len(handlers_before)


def test_logger_hierarchy_and_propagation():
    """Test that child loggers propagate messages to root freegsnke logger."""
    stream = io.StringIO()
    setup_logging(level="INFO", stream=stream, fmt="%(name)s: %(message)s")

    child = logging.getLogger("freegsnke.circuit_eq_metal")
    child.info("testing child message propagation")

    output = stream.getvalue()
    assert "freegsnke.circuit_eq_metal: testing child message propagation" in output


def test_caplog_capture(caplog):
    """Test that pytest caplog fixture captures log messages properly."""
    logger = logging.getLogger("freegsnke.GSstaticsolver")
    with caplog.at_level(logging.INFO, logger="freegsnke"):
        logger.info("Inverse solve completed successfully.")
        logger.debug("This debug message should not appear at INFO level.")

    assert "Inverse solve completed successfully." in caplog.text
    assert "This debug message should not appear at INFO level." not in caplog.text


def test_zero_raw_print_statements_in_library():
    """
    Verification gate: Assert that no raw `print(` calls exist in any
    library modules within the freegsnke package (excluding tests).
    """
    package_dir = os.path.dirname(freegsnke.__file__)
    print_pattern = re.compile(r"\bprint\s*\(")
    violations = []

    for root, dirs, files in os.walk(package_dir):
        # Exclude tests directory
        if "tests" in dirs:
            dirs.remove("tests")
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        if print_pattern.search(line):
                            rel_path = os.path.relpath(file_path, package_dir)
                            violations.append(f"{rel_path}:{line_no}: {stripped}")

    assert not violations, (
        f"Found {len(violations)} raw print() call(s) in library modules:\n"
        + "\n".join(violations)
    )
