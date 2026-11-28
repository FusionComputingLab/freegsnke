"""Unit tests for the unified FreeGSNKE logging infrastructure."""

from __future__ import annotations

import ast
import contextlib
import io
import logging
from pathlib import Path
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


_DISALLOWED_ROOT_LOGGING_FUNCS = frozenset(
    {
        "basicConfig",
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "critical",
        "exception",
        "log",
    }
)


def _find_disallowed_stdout_or_root_logging_calls(
    source: str, rel_path: str
) -> list[str]:
    """Return formatted violations for disallowed print/stdout/root-logging calls."""
    tree = ast.parse(source, filename=rel_path)

    # Track module aliases and direct symbol imports within this file.
    module_aliases: dict[str, str] = {}
    symbol_imports: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                module_aliases[bound] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                bound = alias.asname or alias.name
                symbol_imports[bound] = (node.module, alias.name)

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        hit: str | None = None

        if isinstance(func, ast.Name):
            name = func.id
            if name in {"print", "pprint"}:
                hit = f"{name}(...)"
            elif name in symbol_imports:
                mod, sym = symbol_imports[name]
                if mod == "builtins" and sym == "print":
                    hit = f"{name}(...) [from builtins.print]"
                elif mod == "pprint" and sym == "pprint":
                    hit = f"{name}(...) [from pprint.pprint]"
                elif mod == "logging" and sym in _DISALLOWED_ROOT_LOGGING_FUNCS:
                    hit = f"{name}(...) [from logging.{sym}]"
        elif isinstance(func, ast.Attribute):
            # e.g. builtins.print(...), pprint.pprint(...), logging.info(...)
            if isinstance(func.value, ast.Name):
                base = func.value.id
                resolved_mod = module_aliases.get(base, base)
                if resolved_mod == "builtins" and func.attr == "print":
                    hit = f"{base}.print(...)"
                elif resolved_mod == "pprint" and func.attr == "pprint":
                    hit = f"{base}.pprint(...)"
                elif (
                    resolved_mod == "logging"
                    and func.attr in _DISALLOWED_ROOT_LOGGING_FUNCS
                ):
                    hit = f"{base}.{func.attr}(...)"
                elif base in symbol_imports:
                    mod, sym = symbol_imports[base]
                    if (
                        mod == "sys"
                        and sym in {"stdout", "stderr"}
                        and func.attr == "write"
                    ):
                        hit = f"{base}.write(...) [from sys.{sym}.write]"
            # e.g. sys.stdout.write(...), sys.stderr.write(...)
            elif (
                isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.attr == "write"
                and func.value.attr in {"stdout", "stderr"}
            ):
                base = func.value.value.id
                resolved_mod = module_aliases.get(base, base)
                if resolved_mod == "sys":
                    hit = f"{base}.{func.value.attr}.write(...)"

        if hit is not None:
            violations.append(f"{rel_path}:{node.lineno}: `{hit}`")

    return sorted(violations)


def test_ast_logging_gate_catches_disallowed_patterns():
    """Verify the AST checker flags disallowed calls and ignores benign code."""
    bad_source = "\n".join(
        [
            "import builtins",
            "import logging as log",
            "import pprint",
            "import sys",
            "from pprint import pprint as pp",
            "from sys import stdout",
            "print('raw print')",
            "builtins.print('builtin print')",
            "pprint.pprint({'a': 1})",
            "pp({'b': 2})",
            "sys.stdout.write('stdout write')",
            "sys.stderr.write('stderr write')",
            "stdout.write('imported stdout write')",
            "log.basicConfig(level=log.INFO)",
            "log.info('root logger call')",
        ]
    )
    violations = _find_disallowed_stdout_or_root_logging_calls(bad_source, "sample.py")
    assert len(violations) == 9

    good_source = "\n".join(
        [
            '"""Docstring mentioning print() and sys.stdout.write()."""',
            "import logging",
            "logger = logging.getLogger(__name__)",
            "# print('commented out')",
            "msg = 'print(x) and logging.basicConfig() inside string'",
            "logger.info('Valid message: %s', msg)",
            "obj.print('custom method is not builtin print')",
        ]
    )
    assert _find_disallowed_stdout_or_root_logging_calls(good_source, "good.py") == []


def test_no_disallowed_stdout_or_root_logging_calls_in_library():
    """Enforce unified logging across all FreeGSNKE library modules.

    Scans every Python module under ``freegsnke/`` (excluding ``tests/``) using
    Python's AST to ensure ``print``, ``pprint``, direct ``sys.stdout`` /
    ``sys.stderr`` writes, and root ``logging`` calls do not regress into
    library code.
    """
    package_dir = Path(freegsnke.__file__).resolve().parent
    violations: list[str] = []

    for py_file in sorted(package_dir.rglob("*.py")):
        rel_path = py_file.relative_to(package_dir)
        if "tests" in rel_path.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        violations.extend(
            _find_disallowed_stdout_or_root_logging_calls(
                source, f"freegsnke/{rel_path.as_posix()}"
            )
        )

    assert not violations, (
        f"Found {len(violations)} disallowed stdout or root-logging call(s) in "
        "FreeGSNKE library modules:\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\n\n"
        "Direct stdout writes (print, pprint, sys.stdout/stderr.write) and root "
        "logging calls (logging.basicConfig, logging.<level>) are forbidden in "
        "library code.\n\n"
        "Use a module-scoped logger instead:\n"
        "    import logging\n\n"
        "    logger = logging.getLogger(__name__)\n"
        '    logger.info("Computed equilibrium in %d iterations.", n_iter)\n\n'
        "Level guidelines:\n"
        "  - DEBUG   : detailed diagnostics (also used when verbose=False or suppress=True)\n"
        "  - INFO    : normal milestones and user-requested verbose/summary output\n"
        "  - WARNING : recoverable issues or non-convergence\n"
        "  - ERROR   : failures (use exc_info=True inside exception handlers)\n\n"
        "See freegsnke/logging.py and docs/user_guide/logging.rst for details."
    )
