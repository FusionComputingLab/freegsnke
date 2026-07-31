"""Tests for passive-mode coupling estimates."""

from types import SimpleNamespace

import numpy as np

from freegsnke.nonlinear_solve import nl_solver


def test_no_gs_coupling_norm_uses_evaluated_perturbation():
    """The coupling norm must match the returned finite-difference column."""

    solver = nl_solver.__new__(nl_solver)
    solver.Iy = np.zeros(3)
    solver.nIy = 10.0
    solver.n_coils = 2
    solver.approved_target_dIy = np.array([0.25, 0.4])
    solver.final_dI_record = np.zeros(2)
    solver.profiles2 = SimpleNamespace(
        diverted_core_mask=np.array([True, False, False])
    )

    derivatives = (
        np.array([3.0, 4.0, 0.0]),
        np.array([0.0, 0.0, 2.0]),
    )

    def prepare_column(j, _rtol, target_dIy, starting_dI, GS):
        assert GS is False
        relative_response = np.linalg.norm(derivatives[j] * starting_dI) / solver.nIy
        solver.final_dI_record[j] = starting_dI * target_dIy / relative_response
        return derivatives[j], relative_response

    solver.prepare_build_dIydI_j = prepare_column
    starting_dI = np.array([2.0, 4.0])
    core_mask = solver.profiles2.diverted_core_mask.copy()

    solver.build_dIydI_noGS(
        force_core_mask_linearization=False,
        starting_dI=starting_dI,
        core_mask=core_mask,
        verbose=False,
    )

    np.testing.assert_allclose(
        solver.ndIydI_no_GS,
        np.linalg.norm(solver.dIydI_noGS, axis=0),
    )
