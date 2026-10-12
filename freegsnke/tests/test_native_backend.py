"""Numerical and lifecycle checks for the absorbed solver and thread backend."""

import importlib.metadata
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ellipe, ellipk
from threadpoolctl import threadpool_limits

from freegsnke import (
    build_machine,
    equilibrium,
    equilibrium_update,
    gradshafranov,
    parallel,
)
from freegsnke.gs_solver import GSDSTSolver, GSLUSolver


@pytest.fixture
def grid():
    return np.meshgrid(
        np.linspace(0.2, 2.0, 17), np.linspace(-1.0, 1.0, 25), indexing="ij"
    )


@pytest.mark.parametrize("order", [2, 4])
def test_lu_recovers_manufactured_flux_with_nonzero_boundaries(grid, order):
    R, Z = grid
    expected = np.sin(2 * R) * np.cos(Z) + 0.1 * R**2 + 0.03 * Z
    operator_type = (
        gradshafranov.GSsparse if order == 2 else gradshafranov.GSsparse4thOrder
    )
    matrix = operator_type(R[0, 0], R[-1, 0], Z[0, 0], Z[0, -1])(*R.shape)
    rhs = (matrix @ expected.ravel()).reshape(R.shape)
    solver = GSLUSolver(R, Z, order=order)
    result = solver(np.zeros_like(R), rhs)
    np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-11)
    with pytest.raises(ValueError, match="shape mismatch"):
        solver(np.zeros_like(R), rhs[:-1])


def test_dst_recovers_second_order_flux_with_nonzero_boundaries(grid):
    R, Z = grid
    expected = np.sin(2 * R) * np.cos(Z) + 0.1 * R**2 + 0.03 * Z
    matrix = gradshafranov.GSsparse(R[0, 0], R[-1, 0], Z[0, 0], Z[0, -1])(*R.shape)
    rhs = (matrix @ expected.ravel()).reshape(R.shape)
    result = GSDSTSolver(R, Z)(np.zeros_like(R), rhs)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.fixture
def native_equilibrium():
    configs = Path(__file__).resolve().parents[2] / "machine_configs" / "test"
    tokamak = build_machine.tokamak(
        active_coils_path=str(configs / "active_coils.pickle"),
        passive_coils_path=str(configs / "passive_coils.pickle"),
        limiter_path=str(configs / "limiter.pickle"),
        wall_path=str(configs / "wall.pickle"),
        magnetic_probe_path=str(configs / "magnetic_probes.pickle"),
    )
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak, Zmin=-2.2, Zmax=2.2, nx=17, ny=33
    )
    eq.tokamak_psi = eq.tokamak.calcPsiFromGreens(eq._pgreen)
    return eq


def test_auxiliary_copy_preserves_lazy_and_initialized_solver_state(
    native_equilibrium, monkeypatch
):
    eq = native_equilibrium
    original_factory = equilibrium.GSLUSolver
    calls = []

    def create_solver(*args, **kwargs):
        calls.append(True)
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(equilibrium, "GSLUSolver", create_solver)
    with warnings.catch_warnings():
        # No deprecated property needs to be accessed internally during copying.
        warnings.simplefilter("error", DeprecationWarning)
        aux = eq.create_auxiliary_equilibrium()
    assert not calls
    zero = np.zeros_like(eq.R)
    np.testing.assert_array_equal(aux.callSolver(zero, zero), zero)
    assert len(calls) == 1
    eq.callSolver(zero, zero)
    assert len(calls) == 2
    copy = eq.create_auxiliary_equilibrium()
    copy.callSolver(zero, zero)
    assert len(calls) == 2
    replacement = eq.plasma_psi + np.sin(eq.R) * np.cos(eq.Z)
    copy.set_plasma_psi(replacement)
    np.testing.assert_allclose(copy.psi_func(copy.R[:, 0], copy.Z[0, :]), replacement)
    assert not np.array_equal(eq.plasma_psi, copy.plasma_psi)


def test_compatibility_solver_reset_and_custom_copy(native_equilibrium):
    eq = native_equilibrium
    rhs = np.zeros_like(eq.R)
    rhs[0, :] = 0.2

    def custom_solver(guess, source):
        return source + 1

    with pytest.deprecated_call():
        eq.setSolver(custom_solver)
    aux = eq.create_auxiliary_equilibrium()
    np.testing.assert_array_equal(aux.callSolver(rhs, rhs), rhs + 1)
    with pytest.deprecated_call():
        aux.setSolverVcycle(nlevels=2)
    reset = aux.callSolver(rhs, rhs)
    np.testing.assert_allclose(reset[0], 0.2, atol=1e-12)
    assert not np.array_equal(reset, rhs + 1)
    np.testing.assert_array_equal(eq.callSolver(rhs, rhs), rhs + 1)


@pytest.mark.parametrize("broadcast", [False, True])
def test_optimized_greens_matches_independent_elliptic_formula(broadcast):
    Rc = np.array([0.6, 0.8, 1.3])
    Zc = np.array([-0.3, 0.0, 0.4])
    R = np.array([0.6, 1.1, 1.5])
    Z = np.array([-0.3, 0.2, -0.1])
    if broadcast:
        Rc, Zc = Rc[:, None], Zc[:, None]
        R, Z = R[None, :], Z[None, :]
    k2 = np.clip(4 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2), 1e-10, 1 - 1e-10)
    reference = (
        gradshafranov.mu0
        / (2 * np.pi)
        * np.sqrt(R * Rc)
        * ((2 - k2) * ellipk(k2) - 2 * ellipe(k2))
        / np.sqrt(k2)
    )
    output = np.empty_like(reference)
    result = gradshafranov.Greens(
        Rc, Zc, R, Z, limit_threading=True, scale_factor=2, out=output
    )
    assert result is output
    np.testing.assert_allclose(result, 2 * reference, rtol=1e-14, atol=1e-20)
    scalar = gradshafranov.Greens(0.7, 0.0, 1.0, 0.3)
    assert np.asarray(scalar).shape == ()
    assert np.isfinite(scalar)


def test_threadpool_controller_reports_native_distribution_and_restores_limits():
    original = parallel.get_num_threads()
    assert parallel.CustomThreadController.internal_api == "freegsnke.parallel"
    controller = object.__new__(parallel.CustomThreadController)
    assert controller.get_version() == importlib.metadata.version("freegsnke")
    try:
        parallel.set_num_threads(min(2, parallel.get_max_threads()))
        with threadpool_limits(limits=1):
            assert parallel.get_num_threads() == 1
        assert parallel.get_num_threads() == min(2, parallel.get_max_threads())
    finally:
        parallel.set_num_threads(original)
