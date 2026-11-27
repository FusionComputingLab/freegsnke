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
    eq = SimpleNamespace(plasma_psi=np.zeros((1, 2)))
    eq.set_plasma_psi = lambda field: setattr(eq, "plasma_psi", field.copy())
    return solver, eq, SimpleNamespace()


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


@pytest.mark.parametrize(
    "error",
    [KeyError("bug"), IndexError("bug"), RuntimeError("bug"), KeyboardInterrupt()],
)
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


@pytest.mark.parametrize("nonfinite_flux", [False, True])
def test_nonfinite_trials_are_rejected_and_state_restored(nonfinite_flux):
    """Neither a non-finite field nor residual may be accepted."""
    base = np.array([1.0, 2.0])

    def evaluate(plasma, vacuum, profiles):
        profiles.field = plasma.copy()
        return np.zeros(2) if np.array_equal(plasma, base) else np.full(2, np.nan)

    solver, eq, profiles = fixture(evaluate)
    update = np.full(2, np.nan) if nonfinite_flux else np.ones(2)
    with pytest.raises(RuntimeError, match="accepted equilibrium restored"):
        solver._backtrack_valid_update(eq, profiles, base, base, update, max_attempts=2)
    np.testing.assert_array_equal(eq.plasma_psi.ravel(), base)
    np.testing.assert_array_equal(profiles.field, base)


def test_inverse_reduction_retains_xpoint_and_returns_current_scale():
    """Coil-current updates use the same reduction as the trial vacuum flux."""
    base = np.array([1.0, 2.0])

    def evaluate(plasma, vacuum, profiles):
        profiles.field = vacuum.copy()
        profiles.xpt = [] if np.max(vacuum - base) > 0.6 else [[1.0, 0.0]]
        return vacuum - base

    solver, eq, profiles = fixture(evaluate)
    point, residual, scale = solver._backtrack_valid_update(
        eq, profiles, base, base, np.ones(2), vary_tokamak=True, require_xpoint=True
    )
    assert scale == 0.75**2
    assert len(profiles.xpt) == 1
    np.testing.assert_allclose(point, base + scale)
    np.testing.assert_allclose(residual, np.full(2, scale))


@pytest.mark.parametrize("rollback", [False, True])
def test_forward_reports_returned_field_without_changing_selection(capsys, rollback):
    """Diagnostics must not change the return path of short inner solves."""
    solver = NKGSsolver.__new__(NKGSsolver)
    solver.nx, solver.ny = 1, 2
    solver.rng = np.random.default_rng(42)
    nk = SimpleNamespace(dx=np.ones(2), coeffs=[1.0])
    nk.Arnoldi_iteration = lambda **kwargs: None
    solver.nksolver = nk
    eq = SimpleNamespace(
        plasma_psi=np.array([[1.0, 3.0]]),
        solved=False,
        _vgreen=None,
        tokamak=SimpleNamespace(getPsitokamak=lambda **kwargs: np.zeros((1, 2))),
    )
    eq.set_plasma_psi = lambda field: setattr(eq, "plasma_psi", field.copy())
    profiles = SimpleNamespace(jtor=np.zeros((1, 2)))

    def evaluate(plasma, vacuum, profiles):
        profiles.field = plasma.copy()
        profiles.jtor = plasma.reshape(1, 2).copy()
        if rollback:
            points = [1.0, 2.0, 3.0, 8.0 / 3.0, 11.0 / 3.0]
            values = [1.0, 0.2, 0.3, 0.1, 0.11]
            value = next(v for p, v in zip(points, values) if np.isclose(plasma[0], p))
        else:
            value = {1.0: 1.0, 2.0: 0.2, 3.0: 0.22}[plasma[0]]
        return np.array([0.0, value])

    solver.F_function = evaluate
    solver.relative_norm_residual = lambda res, psi: np.ptp(res)
    solver.port_critical = lambda eq, profiles: setattr(
        eq, "profile_field", profiles.field.copy()
    )
    solver.forward_solve(
        eq,
        profiles,
        target_relative_tolerance=0.01,
        max_solving_iterations=3 if rollback else 2,
        Picard_handover=1.0,
    )
    expected = np.array([8.0 / 3.0, 14.0 / 3.0]) if rollback else np.array([3.0, 5.0])
    np.testing.assert_allclose(eq.plasma_psi.ravel(), expected)
    np.testing.assert_allclose(eq.profile_field, expected)
    np.testing.assert_allclose(profiles.jtor.ravel(), expected)
    assert solver.relative_change == (0.05 if rollback else 0.11)
    assert (
        "Tolerance 5.00e-02" if rollback else "Tolerance 1.10e-01"
    ) in capsys.readouterr().out
