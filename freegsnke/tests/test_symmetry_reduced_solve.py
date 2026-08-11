"""Tests for the dedicated even-only evolutive-solver interface."""

from unittest.mock import patch

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
