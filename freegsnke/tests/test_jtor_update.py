import os

import freegs4e
import numpy as np
import pytest

import freegsnke.jtor_update as jtor
from freegsnke import build_machine, equilibrium_update


@pytest.fixture()
def create_machine():

    # build machine
    tokamak = build_machine.tokamak(
        active_coils_path=f"./machine_configs/test/active_coils.pickle",
        passive_coils_path=f"./machine_configs/test/passive_coils.pickle",
        limiter_path=f"./machine_configs/test/limiter.pickle",
        wall_path=f"./machine_configs/test/wall.pickle",
        magnetic_probe_path=f"./machine_configs/test/magnetic_probes.pickle",
    )

    # Creates equilibrium object and initializes it with
    # a "good" solution
    # plasma_psi = np.loadtxt('plasma_psi_example.txt')
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        # domains can be changed
        Rmin=0.1,
        Rmax=2.0,  # Radial domain
        Zmin=-2.2,
        Zmax=2.2,  # Height range
        # grid resolution can be changed
        nx=65,
        ny=129,  # Number of grid points
        # psi=plasma_psi[::2,:])
    )
    return eq, tokamak


def test_profiles_PaxisIp(create_machine):
    """Tests that the profiles have the xpt, opt and jtor attributes."""
    eq, tokamak = create_machine

    profiles = jtor.ConstrainPaxisIp(
        eq,
        8.1e3,  # Plasma pressure on axis [Pascals]
        6.2e5,  # Plasma current [Amps]
        0.5,  # vacuum f = R*Bt
        alpha_m=1.8,
        alpha_n=1.2,
    )

    profiles.Jtor(eq.R, eq.Z, eq.psi())
    assert (
        hasattr(profiles, "xpt")
        and hasattr(profiles, "opt")
        and hasattr(profiles, "jtor")
    ), "The profiles object does not have the xpt, opt and jtor attributes"


def test_profiles_BetapIp(create_machine):
    """Tests that the profiles have the xpt, opt and jtor attributes."""
    eq, tokamak = create_machine

    profiles = jtor.ConstrainBetapIp(
        eq,
        8.1e3,  # Plasma pressure on axis [Pascals]
        6.2e5,  # Plasma current [Amps]
        0.5,  # vacuum f = R*Bt
    )

    profiles.Jtor(eq.R, eq.Z, eq.psi())
    assert (
        hasattr(profiles, "xpt")
        and hasattr(profiles, "opt")
        and hasattr(profiles, "jtor")
    ), "The profiles object does not have the xpt, opt and jtor attributes"


def test_profile_geometry_is_shared_and_fallback_indices_are_lazy(create_machine):
    eq, _ = create_machine
    profiles = jtor.ConstrainPaxisIp(eq, 8.1e3, 6.2e5, 0.5)
    profiles.inputs = []
    copied_profiles = jtor.Jtor_universal.copy(profiles)
    handler = eq.limiter_handler

    for name in (
        "dR_dZ",
        "R0Z0",
        "eqRidx",
        "eqZidx",
        "mask_inside_limiter",
        "mask_outside_limiter",
        "limiter_mask_out",
    ):
        assert getattr(profiles, name) is getattr(handler, name)
        assert getattr(copied_profiles, name) is getattr(handler, name)

    assert handler._idx_grid_points is None
    assert handler.idx_grid_points.shape == (eq.nx * eq.ny, 2)
    expected_indices = np.indices((eq.nx, eq.ny)).reshape(2, -1).T
    assert np.array_equal(handler.idx_grid_points, expected_indices)
    assert handler.idx_grid_points is handler._idx_grid_points
