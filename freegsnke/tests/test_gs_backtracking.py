"""Regression checks for bounded geometry retries without machine data."""

from types import SimpleNamespace

import numpy as np
import pytest

from freegsnke.GSstaticsolver import NKGSsolver


def fixture(evaluate):
    """Construct the minimal solver state used by geometry backtracking."""
    solver = NKGSsolver.__new__(NKGSsolver)
    solver.nx, solver.ny = 1, 2
    solver.F_function = evaluate
    solver.port_critical = lambda eq, profiles: setattr(
        eq, "profile_field", profiles.field.copy()
    )
    return solver, SimpleNamespace(plasma_psi=np.zeros((1, 2))), SimpleNamespace()


@pytest.mark.parametrize("inverse", [False, True])
def test_persistent_geometry_failure_is_bounded_and_restored(inverse):
    """Permanent geometry failure must end rather than shrink forever."""
    calls = []
    base = np.array([1.0, 2.0])

    def evaluate(plasma, vacuum, profiles):
        point = vacuum if inverse else plasma
        profiles.field = point.copy()
        calls.append(point.copy())
        profiles.xpt = []
        if not inverse and not np.array_equal(point, base):
            raise ValueError("No O-point")
        return np.ones(2)

    solver, eq, profiles = fixture(evaluate)
    with pytest.raises(RuntimeError, match="accepted equilibrium restored"):
        solver._backtrack_valid_update(
            eq,
            profiles,
            base,
            base,
            np.ones(2),
            vary_tokamak=inverse,
            require_xpoint=inverse,
            max_attempts=4,
        )
    assert len(calls) == 5  # four trials plus state restoration
    np.testing.assert_array_equal(profiles.field, base)
    np.testing.assert_array_equal(eq.plasma_psi.ravel(), base)


@pytest.mark.parametrize("error", [KeyError("bug"), KeyboardInterrupt()])
def test_unexpected_error_is_not_retried(error):
    """Programming errors and user interrupts propagate after restoration."""
    calls = []
    base = np.array([1.0, 2.0])

    def evaluate(plasma, vacuum, profiles):
        profiles.field = plasma.copy()
        calls.append(plasma.copy())
        if not np.array_equal(plasma, base):
            raise error
        return np.ones(2)

    solver, eq, profiles = fixture(evaluate)
    with pytest.raises(type(error)):
        solver._backtrack_valid_update(eq, profiles, base, base, np.ones(2))
    assert len(calls) == 2
    np.testing.assert_array_equal(profiles.field, base)


def test_valid_reduced_step_preserves_original_reduction():
    """Successful retries retain the existing factor and source evaluation."""
    base = np.array([1.0, 2.0])

    def evaluate(plasma, vacuum, profiles):
        profiles.field = plasma.copy()
        if np.max(plasma - base) > 0.6:
            raise ValueError("Invalid geometry")
        return plasma - base

    solver, eq, profiles = fixture(evaluate)
    point, residual, scale = solver._backtrack_valid_update(
        eq, profiles, base, base, np.ones(2)
    )
    assert scale == 0.75**2
    np.testing.assert_allclose(point, base + scale)
    np.testing.assert_allclose(profiles.field, point)


def test_unrepresentable_step_stops_without_retrying():
    """A step below floating-point spacing cannot be rescued by shrinking."""
    base = np.array([1.0, 2.0])
    calls = []

    def evaluate(plasma, vacuum, profiles):
        profiles.field = plasma.copy()
        calls.append(1)
        return np.ones(2)

    solver, eq, profiles = fixture(evaluate)
    with pytest.raises(RuntimeError):
        solver._backtrack_valid_update(eq, profiles, base, base, np.full(2, 1e-30))
    assert len(calls) == 1  # restoration only


def test_best_tracking_preserves_legacy_return_and_reports_actual_residual(capsys):
    """Diagnostics must not change the return path of short inner solves."""
    solver = NKGSsolver.__new__(NKGSsolver)
    solver.nx, solver.ny = 1, 2
    solver.rng = np.random.default_rng(42)
    solver.anderson_diagnostics = {"reason": "old solve"}
    nk = SimpleNamespace(dx=np.ones(2), coeffs=[1.0])
    nk.Arnoldi_iteration = lambda **kwargs: None
    solver.nksolver = nk
    eq = SimpleNamespace(
        plasma_psi=np.array([[1.0, 3.0]]),
        solved=False,
        _vgreen=None,
        tokamak=SimpleNamespace(getPsitokamak=lambda **kwargs: np.zeros((1, 2))),
    )
    profiles = SimpleNamespace(jtor=np.zeros((1, 2)))

    def evaluate(plasma, vacuum, profiles):
        profiles.field = plasma.copy()
        profiles.jtor = plasma.reshape(1, 2).copy()
        return np.array([0.0, {1.0: 1.0, 2.0: 0.2, 3.0: 0.22}[plasma[0]]])

    solver.F_function = evaluate
    solver.port_critical = lambda eq, profiles: setattr(
        eq, "profile_field", profiles.field.copy()
    )
    solver.forward_solve(
        eq,
        profiles,
        target_relative_tolerance=0.01,
        max_solving_iterations=2,
        Picard_handover=1.0,
    )
    np.testing.assert_array_equal(eq.plasma_psi, [[3.0, 5.0]])
    np.testing.assert_array_equal(eq.profile_field, [3.0, 5.0])
    np.testing.assert_array_equal(solver.best_psi, [2.0, 4.0])
    assert solver.relative_change == 0.11
    assert solver.best_relative_change == 0.1
    assert solver.anderson_diagnostics is None
    assert "Tolerance 1.10e-01" in capsys.readouterr().out
