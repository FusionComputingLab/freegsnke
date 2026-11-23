"""Tests for the dedicated even-only evolutive-solver interface."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from freegsnke.nonlinear_solve import nl_solver
from freegsnke.symmetry_reduced_solve import SymmetryReducedSolver


def test_reduced_solver_enforces_even_configuration():
    profiles = object()
    equilibrium = object()
    static_solver = object()
    reflection = np.eye(2)

    with patch.object(nl_solver, "__init__", return_value=None) as initialise:
        solver = SymmetryReducedSolver(
            profiles,
            equilibrium,
            static_solver,
            passive_reflection_operator=reflection,
            full_timestep=2.5e-3,
        )

    initialise.assert_called_once_with(
        profiles=profiles,
        eq=equilibrium,
        GSStaticSolver=static_solver,
        passive_reflection_operator=reflection,
        force_up_down_symmetric=True,
        remove_odd_passive_modes=True,
        full_timestep=2.5e-3,
    )


@pytest.mark.parametrize(
    "option",
    [
        {"force_up_down_symmetric": False},
        {"remove_odd_passive_modes": False},
    ],
)
def test_reduced_solver_rejects_non_even_configuration(option):
    """Users cannot disable either invariant through inherited options."""
    with pytest.raises(ValueError):
        SymmetryReducedSolver(object(), object(), object(), **option)


def _symmetry_test_equilibrium(z_values):
    """Provide the real constructor's inputs up to its first GS solve."""
    z_values = np.asarray(z_values, dtype=float)
    nx, ny = 2, len(z_values)
    return SimpleNamespace(
        nx=nx,
        ny=ny,
        dR=1.0,
        dZ=z_values[1] - z_values[0],
        R=np.broadcast_to(np.array([[1.0], [2.0]]), (nx, ny)).copy(),
        Z=np.tile(z_values, (nx, 1)),
        _vgreen=np.arange(nx * ny, dtype=float).reshape(1, nx, ny),
        limiter_handler=SimpleNamespace(
            mask_inside_limiter=np.ones((nx, ny), dtype=bool)
        ),
        tokamak=SimpleNamespace(
            n_coils=1,
            n_active_coils=1,
            coils_dict={"active": {}},
            coil_resist=np.ones(1),
            coil_self_ind=np.eye(1),
        ),
    )


@pytest.mark.parametrize("use_subclass", [False, True])
@pytest.mark.parametrize(
    "invalid_input", ["off_center", "interior", "nonfinite", "mask"]
)
def test_invalid_symmetry_inputs_fail_before_modifying_equilibrium(
    use_subclass, invalid_input
):
    """Reject bad geometry before changing Greens or calling the static solver."""
    z_values = {
        "off_center": [0.0, 1.0, 2.0],
        "interior": [-1.0, 0.1, 1.0],
        "nonfinite": [-1.0, np.nan, 1.0],
        "mask": [-1.0, 0.0, 1.0],
    }[invalid_input]
    eq = _symmetry_test_equilibrium(z_values)
    if invalid_input == "mask":
        eq.limiter_handler.mask_inside_limiter[0, -1] = False
    greens = eq._vgreen.copy()
    resistance = eq.tokamak.coil_resist.copy()
    inductance = eq.tokamak.coil_self_ind.copy()
    static_solver = Mock()
    solver = SymmetryReducedSolver if use_subclass else nl_solver
    options = {} if use_subclass else {"force_up_down_symmetric": True}
    message = (
        "limiter mask symmetric" if invalid_input == "mask" else "Z grid symmetric"
    )
    with pytest.raises(ValueError, match=message):
        solver(object(), eq, static_solver, **options)
    static_solver.forward_solve.assert_not_called()
    np.testing.assert_array_equal(eq._vgreen, greens)
    np.testing.assert_array_equal(eq.tokamak.coil_resist, resistance)
    np.testing.assert_array_equal(eq.tokamak.coil_self_ind, inductance)


@pytest.mark.parametrize(
    "z_values", [[-1.0, 0.0, 1.0], [-1.0, -0.2, 0.2, 1.0], [-1.0, 2e-16, 1.0]]
)
def test_valid_symmetric_grid_and_mask_reach_static_solver(z_values):
    """Accept odd/even grid sizes and floating-point roundoff at the midplane."""
    eq = _symmetry_test_equilibrium(z_values)
    static_solver = Mock()
    static_solver.forward_solve.side_effect = RuntimeError("Reached static solver")
    with pytest.raises(RuntimeError, match="Reached static solver"):
        SymmetryReducedSolver(object(), eq, static_solver)
    static_solver.forward_solve.assert_called_once()


def test_unrestricted_solver_accepts_asymmetric_grid_and_mask():
    """The symmetry guards must not constrain ordinary asymmetric evolution."""
    eq = _symmetry_test_equilibrium([0.0, 1.0, 2.0])
    eq.limiter_handler.mask_inside_limiter[0, -1] = False
    greens = eq._vgreen.copy()
    static_solver = Mock()
    static_solver.forward_solve.side_effect = RuntimeError("Reached static solver")
    with pytest.raises(RuntimeError, match="Reached static solver"):
        nl_solver(object(), eq, static_solver)
    static_solver.forward_solve.assert_called_once()
    np.testing.assert_array_equal(eq._vgreen, greens)
