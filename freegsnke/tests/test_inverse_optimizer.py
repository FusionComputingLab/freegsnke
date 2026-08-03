from types import SimpleNamespace

import numpy as np
import pytest

from freegsnke.GSstaticsolver import NKGSsolver
from freegsnke.inverse import Inverse_optimizer


def test_constraint_weights_scale_matching_matrix_and_rhs_blocks():
    """Each class weight must scale its own complete residual block only."""
    optimizer = object.__new__(Inverse_optimizer)
    optimizer.n_control_coils = 1
    optimizer.isoflux_set = [object()]
    optimizer.null_points = np.empty((2, 1))
    optimizer.psi_vals = np.empty((3, 1))
    optimizer.null_points_2nd_order = None
    optimizer.weight_isoflux = 0.2
    optimizer.weight_nulls = 3.0
    optimizer.weight_psi = 4.0
    optimizer.build_isoflux_lsq = lambda currents: (
        [np.array([[2.0]])],
        [np.array([3.0])],
        [],
    )
    optimizer.build_null_points_lsq = lambda currents: (
        np.array([[5.0]]),
        np.array([11.0]),
        [],
    )
    optimizer.build_psi_vals_lsq = lambda currents: (
        np.array([[7.0]]),
        np.array([13.0]),
        [],
    )

    optimizer.build_lsq(np.zeros(1))

    assert np.allclose(optimizer.A[:, 0], [0.4, 15.0, 28.0])
    assert np.allclose(optimizer.b, [0.6, 33.0, 52.0])


def test_isoflux_point_weights_scale_response_and_residual():
    """A pair weight applies to its Green response and present mismatch."""
    optimizer = object.__new__(Inverse_optimizer)
    optimizer.isoflux_set = [object()]
    optimizer.dG_set = [np.array([[2.0]])]
    optimizer.control_mask = np.array([True])
    optimizer.d_psi_plasma_vals_iso = [np.array([5.0])]
    optimizer.isoflux_weight = [np.array([0.25, 0.5])]
    optimizer.n_control_coils = 1
    optimizer.null_points = None
    optimizer.psi_vals = None
    optimizer.null_points_2nd_order = None
    optimizer.weight_isoflux = 0.2

    matrices, rhs, _ = optimizer.build_isoflux_lsq(np.array([3.0]))

    assert np.allclose(matrices[0], [[0.5]])
    assert np.allclose(rhs[0], [-2.75])

    optimizer.build_lsq(np.array([3.0]))
    frozen_matrix = optimizer.A.copy()
    baseline_rhs = optimizer.b.copy()
    step = 1e-4
    optimizer.build_lsq(np.array([3.0 + step]))
    finite_difference = (optimizer.b - baseline_rhs) / step

    assert np.allclose(finite_difference, -frozen_matrix[:, 0])


def test_constrained_full_jacobian_uses_unperturbed_reference_currents():
    """Current limits must be applied relative to the baseline equilibrium."""

    def make_equilibrium(currents):
        tokamak = SimpleNamespace(current_vec=np.asarray(currents, dtype=float).copy())
        tokamak.set_all_coil_currents = lambda values: setattr(
            tokamak, "current_vec", np.asarray(values, dtype=float).copy()
        )
        equilibrium = SimpleNamespace(
            tokamak=tokamak,
            plasma_psi=np.zeros((2, 2)),
            _vgreen=np.ones((2, 2, 2)),
        )
        equilibrium.create_auxiliary_equilibrium = lambda: make_equilibrium(
            equilibrium.tokamak.current_vec
        )
        return equilibrium

    baseline_currents = np.array([10.0, 20.0])
    constrain = SimpleNamespace(
        b=np.zeros(1),
        n_control_coils=2,
        control_mask=np.ones(2, dtype=bool),
        coil_current_limits=([None, None], [None, None]),
        psi_norm_limits=None,
        rebuild_full_current_vec=np.asarray,
    )

    def optimize_currents(full_currents_vec, **kwargs):
        constrain.b = np.array([np.sum(full_currents_vec)])
        return np.array([1.0, 0.0]), 0.0

    constrain.optimize_currents = optimize_currents
    captured = {}

    def optimize_currents_quadratic(eq, profiles, full_currents_vec, *args, **kwargs):
        captured["reference_currents"] = np.copy(full_currents_vec)
        return np.zeros(2), 0.0

    constrain.optimize_currents_quadratic = optimize_currents_quadratic
    solver = object.__new__(NKGSsolver)
    solver.forward_solve = lambda **kwargs: None
    solver.get_rel_delta_psit = lambda *args, **kwargs: 1.0

    solver.optimize_currents(
        eq=make_equilibrium(baseline_currents),
        profiles=object(),
        constrain=constrain,
        target_relative_tolerance=1e-6,
    )

    assert np.array_equal(captured["reference_currents"], baseline_currents)
    assert np.all(np.isfinite(solver.dbdI))
    assert np.allclose(solver.dbdI, [[1.0, 1.0]])

    solver.get_rel_delta_psit = lambda *args, **kwargs: 0.0
    with pytest.raises(ValueError, match="zero or non-finite core flux response"):
        solver.optimize_currents(
            eq=make_equilibrium(baseline_currents),
            profiles=object(),
            constrain=constrain,
            target_relative_tolerance=1e-6,
        )
