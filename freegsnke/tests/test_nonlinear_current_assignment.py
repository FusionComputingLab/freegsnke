from types import SimpleNamespace

import numpy as np

from freegsnke.nonlinear_solve import nl_solver


class _TokamakWithSeparateCurrentState:
    """Minimal stand-in for Machine's cache and per-coil current storage."""

    def __init__(self, size):
        self.current_vec = np.zeros(size)
        self.coil_currents = np.zeros(size)

    def set_all_coil_currents(self, currents):
        self.current_vec = np.asarray(currents, dtype=float).copy()
        self.coil_currents = np.asarray(currents, dtype=float).copy()


def test_assign_currents_synchronises_tokamak_current_state():
    solver = object.__new__(nl_solver)
    solver.plasma_norm_factor = 1.0e5
    solver.evol_metal_curr = SimpleNamespace(
        IdtoIvessel=lambda Id: 2.0 * np.asarray(Id, dtype=float)
    )

    tokamak = _TokamakWithSeparateCurrentState(size=3)
    eq = SimpleNamespace(tokamak=tokamak, _current=0.0)
    profiles = SimpleNamespace(Ip=0.0)
    currents = np.array([1.0, -2.0, 3.0, 4.0])

    solver.assign_currents(currents, eq, profiles)

    expected_metal_currents = np.array([2.0, -4.0, 6.0])
    np.testing.assert_array_equal(tokamak.current_vec, expected_metal_currents)
    np.testing.assert_array_equal(tokamak.coil_currents, expected_metal_currents)
    assert eq._current == 4.0e5
    assert profiles.Ip == 4.0e5
