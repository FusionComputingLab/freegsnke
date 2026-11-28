"""Unit tests for the unified FreeGSNKE logging infrastructure."""

from __future__ import annotations

import contextlib
import io
import logging
from types import SimpleNamespace

import numpy as np
import pytest

import freegsnke
from freegsnke.GSstaticsolver import NKGSsolver
from freegsnke.logging import (
    DEFAULT_FORMAT,
    SIMPLE_FORMAT,
    disable_default_handler,
    enable_default_handler,
    get_logger,
    set_log_level,
    setup_logging,
)
from freegsnke.refine_passive import generate_refinement
from freegsnke.virtual_circuits import VirtualCircuitHandling


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


def test_refine_passive_invalid_mode_logs_and_raises(caplog):
    """Verify invalid refine_mode logs an error and raises ValueError."""
    disable_default_handler()
    r = np.array([0.0, 1.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0, 1.0])
    with caplog.at_level(logging.ERROR, logger="freegsnke"):
        with pytest.raises(ValueError, match="not recognised"):
            generate_refinement(r, z, 4, "INVALID")

    assert "refine_mode 'INVALID' not recognised" in caplog.text


def test_verbose_flag_maps_to_info_and_debug(capsys):
    """Verify verbose=True logs at INFO and verbose=False logs at DEBUG."""
    handler = VirtualCircuitHandling.__new__(VirtualCircuitHandling)
    matrix = np.eye(2)

    # At default INFO level: verbose=False is quiet, verbose=True emits at INFO.
    handler.calculate_matrix_inverse(matrix, verbose=False)
    assert capsys.readouterr().out == ""

    handler.calculate_matrix_inverse(matrix, verbose=True)
    out_verbose = capsys.readouterr().out
    assert "INFO [freegsnke.virtual_circuits]:" in out_verbose
    assert "Moore-Penrose pseudoinverse" in out_verbose

    # At DEBUG level: verbose=False becomes visible at DEBUG.
    set_log_level("DEBUG")
    handler.calculate_matrix_inverse(matrix, verbose=False)
    out_debug = capsys.readouterr().out
    assert "DEBUG [freegsnke.virtual_circuits]:" in out_debug
    assert "Moore-Penrose pseudoinverse" in out_debug


def _make_stub_static_solver(rel_residual: float):
    """Build a minimal NKGSsolver/eq/profiles stub for zero-iteration solve tests."""
    solver = NKGSsolver.__new__(NKGSsolver)
    solver.nx = 2
    solver.ny = 2
    solver.F_function = lambda psi, tok_psi, prof: np.zeros_like(psi)
    solver.relative_norm_residual = lambda res, psi: rel_residual
    solver.relative_del_residual = lambda res, psi: (rel_residual, np.zeros_like(psi))
    solver.port_critical = lambda eq, profiles: None

    tokamak = SimpleNamespace(getPsitokamak=lambda vgreen=None: np.zeros((2, 2)))
    eq = SimpleNamespace(
        plasma_psi=np.ones((2, 2)),
        tokamak=tokamak,
        _vgreen=None,
        solved=False,
        set_plasma_psi=lambda psi: None,
    )
    profiles = SimpleNamespace(jtor=np.zeros((2, 2)))
    return solver, eq, profiles


def test_suppress_flag_demotes_forward_solve_summary_to_debug(capsys):
    """Verify suppress=True demotes forward_solve summary logs to DEBUG."""
    # Converged case (rel_residual <= tolerance)
    solver, eq, profiles = _make_stub_static_solver(rel_residual=1e-9)

    solver.forward_solve(
        eq,
        profiles,
        target_relative_tolerance=1e-6,
        max_solving_iterations=1,
        verbose=False,
        suppress=False,
    )
    out_default = capsys.readouterr().out
    assert (
        "INFO [freegsnke.GSstaticsolver]: Forward static solve SUCCESS." in out_default
    )
    assert "Forward static solve starting..." not in out_default

    solver.forward_solve(
        eq,
        profiles,
        target_relative_tolerance=1e-6,
        max_solving_iterations=1,
        verbose=False,
        suppress=True,
    )
    assert capsys.readouterr().out == ""

    set_log_level("DEBUG")
    solver.forward_solve(
        eq,
        profiles,
        target_relative_tolerance=1e-6,
        max_solving_iterations=1,
        verbose=False,
        suppress=True,
    )
    out_debug = capsys.readouterr().out
    assert (
        "DEBUG [freegsnke.GSstaticsolver]: Forward static solve starting..."
        in out_debug
    )
    assert (
        "DEBUG [freegsnke.GSstaticsolver]: Forward static solve SUCCESS." in out_debug
    )

    # Non-converged case (rel_residual > tolerance, max_solving_iterations=0)
    set_log_level("INFO")
    solver_nc, eq_nc, profiles_nc = _make_stub_static_solver(rel_residual=1e-2)
    solver_nc.forward_solve(
        eq_nc,
        profiles_nc,
        target_relative_tolerance=1e-6,
        max_solving_iterations=0,
        verbose=False,
        suppress=False,
    )
    out_warn = capsys.readouterr().out
    assert (
        "WARNING [freegsnke.GSstaticsolver]: Forward static solve DID NOT CONVERGE."
        in out_warn
    )

    solver_nc.forward_solve(
        eq_nc,
        profiles_nc,
        target_relative_tolerance=1e-6,
        max_solving_iterations=0,
        verbose=False,
        suppress=True,
    )
    assert capsys.readouterr().out == ""

    set_log_level("DEBUG")
    solver_nc.forward_solve(
        eq_nc,
        profiles_nc,
        target_relative_tolerance=1e-6,
        max_solving_iterations=0,
        verbose=False,
        suppress=True,
    )
    out_nc_debug = capsys.readouterr().out
    assert (
        "DEBUG [freegsnke.GSstaticsolver]: Forward static solve DID NOT CONVERGE."
        in out_nc_debug
    )
