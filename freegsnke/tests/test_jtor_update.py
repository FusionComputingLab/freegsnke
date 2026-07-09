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


def _minimal_eq():
    class LimiterHandler:
        mask_inside_limiter = np.ones((2, 2), dtype=bool)
        limiter_mask_out = np.zeros((2, 2), dtype=bool)

        def make_layer_mask(self, mask, layer_size=1):
            return np.zeros_like(mask, dtype=bool)

    class Eq:
        R_1D = np.array([1.0, 2.0])
        Z_1D = np.array([0.0, 1.0])
        R, Z = np.meshgrid(R_1D, Z_1D, indexing="ij")
        limiter_handler = LimiterHandler()

    return Eq()


def test_general_pprime_ffprime_linear_interpolation():
    eq = _minimal_eq()
    psi_n = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    pprime = np.array([0.0, 2.0, -1.0, 1.0, 0.0])
    ffprime = np.array([1.0, -1.0, 2.0, 0.0, 1.0])

    profiles = jtor.GeneralPprimeFFprime(
        eq,
        Ip=6.2e5,
        fvac=0.5,
        psi_n=psi_n,
        pprime_data=pprime,
        ffprime_data=ffprime,
        p_data=np.zeros_like(psi_n),
        f_data=np.ones_like(psi_n),
        interpolation="linear",
    )

    assert profiles.interpolation == "linear"
    assert np.isclose(profiles.pprime_func(0.125), 1.0)
    assert np.isclose(profiles.ffprime_func(0.125), 0.0)

    profiles.inputs = []
    profiles.L = 1.0
    copied = profiles.copy()
    assert copied.interpolation == "linear"
    assert np.isclose(copied.pprime_func(0.125), 1.0)
    assert np.isclose(copied.ffprime_func(0.125), 0.0)


def test_general_pprime_ffprime_rejects_unknown_interpolation():
    eq = _minimal_eq()

    with pytest.raises(ValueError, match="interpolation"):
        jtor.GeneralPprimeFFprime(
            eq,
            Ip=6.2e5,
            fvac=0.5,
            psi_n=np.array([0.0, 0.5, 1.0]),
            pprime_data=np.array([0.0, 1.0, 0.0]),
            ffprime_data=np.array([1.0, 0.0, 1.0]),
            interpolation="nearest",
        )
