import numpy as np
import pytest

from freegsnke.control_loop.vc_provider import VCGenerator
from freegsnke.virtual_circuits import VirtualCircuitHandling


class _FakeCoil:
    def __init__(self, current):
        self.current = current


class _FakeTokamak:
    def __init__(self, currents):
        self._coils = {name: _FakeCoil(current) for name, current in currents.items()}

    def __getitem__(self, name):
        return self._coils[name]

    def set_coil_current(self, name, current):
        self._coils[name].current = current

    def getCurrents(self):
        return {name: coil.current for name, coil in self._coils.items()}


class _FakeLimiterHandler:
    @staticmethod
    def Iy_from_jtor(jtor):
        return np.asarray(jtor)


class _FakeEquilibrium:
    def __init__(self, currents):
        self.tokamak = _FakeTokamak(currents)
        self.limiter_handler = _FakeLimiterHandler()

    def create_auxiliary_equilibrium(self):
        return _FakeEquilibrium(self.tokamak.getCurrents())


class _FakeProfiles:
    def __init__(self):
        self.jtor = np.zeros(3)

    def copy(self):
        copied = _FakeProfiles()
        copied.jtor = self.jtor.copy()
        return copied


class _FakeSolver:
    def forward_solve(
        self,
        eq,
        profiles,
        target_relative_tolerance=None,
        suppress=False,
    ):
        currents = eq.tokamak.getCurrents()
        p1 = currents["P1"]
        p2 = currents["P2"]
        p3 = currents["P3"]
        profiles.jtor = np.array(
            [
                0.10 * p1 + 0.20 * p2 + 0.30 * p3,
                p1 - p3,
                p2 + 2.0 * p3,
            ]
        )


def _target_calculator(eq):
    currents = eq.tokamak.getCurrents()
    return np.array(
        [
            currents["P1"] + 2.0 * currents["P2"],
            currents["P3"] - currents["P2"],
        ]
    )


def _target_calculator_with_extra_target(eq):
    currents = eq.tokamak.getCurrents()
    return np.array(
        [
            currents["P1"] + 2.0 * currents["P2"],
            currents["P3"] - currents["P2"],
            currents["P1"],
        ]
    )


def _calculate_vc(n_vc_workers):
    eq = _FakeEquilibrium({"P1": 100.0, "P2": -80.0, "P3": 40.0})
    profiles = _FakeProfiles()
    handler = VirtualCircuitHandling()
    handler.define_solver(_FakeSolver())
    handler.calculate_VC(
        eq=eq,
        profiles=profiles,
        coils=["P1", "P2", "P3"],
        target_names=["sum", "difference"],
        target_calculator=_target_calculator,
        target_dIy=1e-3,
        starting_dI=np.array([20.0, 20.0, 20.0]),
        name="test_vc",
        n_vc_workers=n_vc_workers,
    )
    return handler.test_vc


def test_parallel_virtual_circuit_matches_serial():
    serial = _calculate_vc(n_vc_workers=1)
    parallel = _calculate_vc(n_vc_workers=2)

    np.testing.assert_allclose(parallel.shape_matrix, serial.shape_matrix)
    np.testing.assert_allclose(parallel.VCs_matrix, serial.VCs_matrix)


def test_parallel_vc_generator_matches_serial_with_reordered_targets():
    eq = _FakeEquilibrium({"P1": 100.0, "P2": -80.0, "P3": 40.0})
    profiles = _FakeProfiles()
    coils = ["P1", "P2", "P3"]
    coils_calc = ["P1", "P3"]
    targets = ["sum"]
    targets_calc = ["difference", "sum"]

    serial = VCGenerator(
        solver=_FakeSolver(),
        target_calculator=_target_calculator_with_extra_target,
        target_names=["sum", "difference", "p1"],
        targets_ctrl=targets,
        targets_calc=targets_calc,
        coils_calc=coils_calc,
        n_vc_workers=1,
    )
    parallel = VCGenerator(
        solver=_FakeSolver(),
        target_calculator=_target_calculator_with_extra_target,
        target_names=["sum", "difference", "p1"],
        targets_ctrl=targets,
        targets_calc=targets_calc,
        coils_calc=coils_calc,
        n_vc_workers=2,
    )

    serial_vc = serial.get_vc(
        targets=targets,
        targets_calc=targets_calc,
        coils=coils,
        coils_calc=coils_calc,
        input_data=serial.get_inputs_from_eq(eq, profiles),
    )
    parallel_vc = parallel.get_vc(
        targets=targets,
        targets_calc=targets_calc,
        coils=coils,
        coils_calc=coils_calc,
        input_data=parallel.get_inputs_from_eq(eq, profiles),
    )

    np.testing.assert_allclose(parallel.latest_shape_matrix, serial.latest_shape_matrix)
    np.testing.assert_allclose(parallel_vc, serial_vc)


@pytest.mark.parametrize("n_vc_workers", [0, -1, True, 1.5])
def test_virtual_circuit_worker_count_must_be_positive_integer(n_vc_workers):
    with pytest.raises(ValueError, match="n_vc_workers"):
        _calculate_vc(n_vc_workers=n_vc_workers)
