import numpy as np
import pytest

from freegsnke import (
    GSstaticsolver,
    build_machine,
    equilibrium_update,
    nonlinear_solve,
)
from freegsnke.jtor_update import ConstrainPaxisIp


FREQUENCY_ONLY_MODE_SETTINGS = {
    "max_mode_frequency": 30.0,
    "threshold_dIy_dI": 2.0,
    "min_dIy_dI": 0.0,
    "mode_removal": False,
}


@pytest.fixture(scope="module")
def initial_condition():
    tokamak = build_machine.tokamak(
        active_coils_path="./machine_configs/test/active_coils.pickle",
        passive_coils_path="./machine_configs/test/passive_coils.pickle",
        limiter_path="./machine_configs/test/limiter.pickle",
        wall_path="./machine_configs/test/wall.pickle",
        magnetic_probe_path="./machine_configs/test/magnetic_probes.pickle",
    )

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-2.2,
        Zmax=2.2,
        nx=65,
        ny=129,
    )

    profiles = ConstrainPaxisIp(
        eq,
        8.1e3,
        6.2e5,
        0.5,
        alpha_m=1.8,
        alpha_n=1.2,
    )

    currents = np.array(
        [
            40000,
            623.1330076232998,
            15761.113413087669,
            6218.6648587680265,
            10169.401670695957,
            -1913.7157252356117,
            2440.9195954337097,
            -5349.68745069716,
            -1786.696839741781,
            93.17532977532858,
            -4057.3992383452764,
            -100,
        ]
    )
    for coil, current in zip(eq.tokamak.getCurrents().keys(), currents):
        eq.tokamak.set_coil_current(coil, current)

    static_solver = GSstaticsolver.NKGSsolver(eq)
    static_solver.solve(eq, profiles, target_relative_tolerance=1e-8)

    return eq, profiles, static_solver


def frequency_only_nonlinear_solver(eq, profiles, static_solver, **linearization):
    return nonlinear_solve.nl_solver(
        profiles=profiles,
        eq=eq,
        GSStaticSolver=static_solver,
        full_timestep=3e-3,
        max_internal_timestep=3e-3,
        plasma_resistivity=5e-7,
        automatic_timestep=False,
        **FREQUENCY_ONLY_MODE_SETTINGS,
        **linearization,
    )


@pytest.fixture(scope="module")
def cached_linearization(initial_condition):
    eq, profiles, static_solver = initial_condition
    original_solver = frequency_only_nonlinear_solver(eq, profiles, static_solver)
    cache = {
        "dIydI": original_solver.dIydI_ICs.copy(),
        "dIydtheta": original_solver.dIydtheta_ICs.copy(),
    }
    return original_solver, cache


def test_cached_jacobian_can_be_reused_with_frequency_only_truncation(
    monkeypatch,
    initial_condition,
    cached_linearization,
):
    original_solver, cache = cached_linearization
    eq, profiles, static_solver = initial_condition

    def fail_if_current_jacobian_is_recomputed(*args, **kwargs):
        raise AssertionError("dIydI should be sourced from the supplied cache.")

    def fail_if_profile_jacobian_is_recomputed(*args, **kwargs):
        raise AssertionError("dIydtheta should be sourced from the supplied cache.")

    monkeypatch.setattr(
        nonlinear_solve.nl_solver,
        "build_dIydI_j",
        fail_if_current_jacobian_is_recomputed,
    )
    monkeypatch.setattr(
        nonlinear_solve.nl_solver,
        "build_dIydtheta",
        fail_if_profile_jacobian_is_recomputed,
    )

    reused_solver = frequency_only_nonlinear_solver(
        eq,
        profiles,
        static_solver,
        dIydI=cache["dIydI"],
        dIydtheta=cache["dIydtheta"],
    )

    assert reused_solver.n_metal_modes == original_solver.n_metal_modes
    assert reused_solver.plasma_domain_size == original_solver.plasma_domain_size
    assert reused_solver.dIydI.shape == (
        reused_solver.plasma_domain_size,
        reused_solver.n_metal_modes + 1,
    )
    assert reused_solver.dIydtheta.shape == (
        reused_solver.plasma_domain_size,
        reused_solver.n_profiles_parameters,
    )
    np.testing.assert_allclose(reused_solver.dIydI, cache["dIydI"])
    np.testing.assert_allclose(reused_solver.dIydtheta, cache["dIydtheta"])
