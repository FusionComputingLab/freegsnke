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


def test_contour_fallback_coordinates_and_current_sign(create_machine):
    """The contour fallback is invariant under simultaneous psi and Ip reversal."""
    eq, _ = create_machine
    positive_psi = -((eq.R - 1.0) ** 2 + (eq.Z / 1.5) ** 2)
    results = []

    for current_sign in (1, -1):
        profiles = jtor.ConstrainPaxisIp(
            eq,
            8.1e3,
            current_sign * 6.2e5,
            0.5,
            alpha_m=1.8,
            alpha_n=1.2,
        )
        opt, xpt, core_mask, psi_bndry = profiles.diverted_critical(
            eq.R,
            eq.Z,
            current_sign * positive_psi,
            mask_outside_limiter=profiles.mask_outside_limiter,
            rel_tolerance_xpt=1e-4,
        )

        distances = np.linalg.norm(
            profiles.lcfs[:, np.newaxis] - profiles.lcfs[np.newaxis, :], axis=-1
        ) + 10 * np.eye(len(profiles.lcfs))
        closest_pair = distances == np.amin(distances)
        expected_xpt = np.mean(profiles.lcfs[np.any(closest_pair, axis=0)], axis=0)
        assert np.allclose(xpt[0, :2], expected_xpt)
        assert np.isclose(xpt[0, 2], psi_bndry)
        results.append((opt, xpt, core_mask, psi_bndry))

    positive, negative = results
    assert np.allclose(positive[0][0, :2], negative[0][0, :2])
    assert np.allclose(positive[1][0, :2], negative[1][0, :2])
    assert np.array_equal(positive[2], negative[2])
    assert np.isclose(positive[0][0, 2], -negative[0][0, 2])
    assert np.isclose(positive[3], -negative[3])
