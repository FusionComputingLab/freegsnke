import os

import freegs4e
import matplotlib.pyplot as plt
import numpy as np
import pytest

import freegsnke.jtor_update as jtor
from freegsnke import build_machine, equilibrium_update


def simple_active_coils_data():
    """Return the minimal active coil data needed to instantiate a test tokamak."""
    return {
        "P1": {
            "R": [0.5],
            "Z": [0.0],
            "dR": 0.05,
            "dZ": 0.05,
            "resistivity": 1e-8,
            "polarity": 1.0,
            "multiplier": 1.0,
        }
    }


def rectangular_limiter(z_extent):
    """Return a rectangular limiter/wall contour with the requested half-height."""
    wall_R = [0.6, 1.3, 1.3, 0.6, 0.6]
    wall_Z = [-z_extent, -z_extent, z_extent, z_extent, -z_extent]
    return [{"R": r, "Z": z} for r, z in zip(wall_R, wall_Z)]


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


def test_no_diverted_mask_falls_back_to_limiter_boundary():
    """If no X-point mask is available, the plasma boundary is limiter-defined."""
    limiter_data = rectangular_limiter(z_extent=0.45)
    tokamak = build_machine.tokamak(
        active_coils_data=simple_active_coils_data(),
        limiter_data=limiter_data,
        wall_data=limiter_data,
    )
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.55,
        Rmax=1.35,
        Zmin=-0.5,
        Zmax=0.5,
        nx=33,
        ny=33,
    )
    profiles = jtor.ConstrainPaxisIp(
        eq,
        8.1e3,
        6.2e5,
        0.5,
        alpha_m=1.8,
        alpha_n=1.2,
    )
    psi = -((eq.R - 0.95) ** 2 + eq.Z**2)

    def no_diverted_mask(R, Z, psi, psi_bndry=None, mask_outside_limiter=None):
        opt = np.array([[0.95, 0.0, np.amax(psi)]])
        xpt = np.empty((0, 3))
        return opt, xpt, None, None

    def mask_as_current(R, Z, psi, psi_axis, psi_bndry, mask):
        return mask.astype(float)

    (
        jtor_map,
        _,
        xpt,
        psi_bndry,
        diverted_core_mask,
        limiter_core_mask,
        flag_limiter,
    ) = profiles.Jtor_build(
        no_diverted_mask,
        mask_as_current,
        eq.limiter_handler.core_mask_limiter,
        eq.R,
        eq.Z,
        psi,
        None,
        profiles.mask_outside_limiter,
        profiles.limiter_mask_out,
    )

    assert len(xpt) == 0, "The test setup should not provide an X-point"
    assert (
        diverted_core_mask is None
    ), "The test setup should not provide a diverted mask"
    assert flag_limiter, "No-diverted-mask cases should be identified as limiter cases"
    assert not profiles.has_relevant_xpoint
    assert limiter_core_mask is not None, "Limiter fallback did not produce a core mask"
    assert np.sum(limiter_core_mask) > 0, "Limiter fallback produced an empty core mask"
    assert np.all(
        jtor_map == limiter_core_mask
    ), "Jtor was not built from the limiter mask"
    assert np.isclose(
        psi_bndry, np.amax(eq.limiter_handler.psi_on_limiter_boundary(psi))
    )

    profiles.jtor = jtor_map
    profiles.opt = np.array([[0.95, 0.0, np.amax(psi)]])
    profiles.xpt = xpt
    profiles.psi_bndry = psi_bndry
    profiles.flag_limiter = flag_limiter
    eq._profiles = profiles
    eq.psi_bndry = psi_bndry
    eq.has_relevant_xpoint = profiles.has_relevant_xpoint

    axis = eq.plot(show=False)
    assert axis is not None
    plt.close(axis.figure)
