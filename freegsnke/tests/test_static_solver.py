import os
import sys
import time
from copy import deepcopy

import freegs4e
import matplotlib.pyplot as plt
import numpy as np
import pytest
from freegs4e.critical import find_critical
from freegs4e.plotting import plotConstraints
from IPython.display import clear_output, display
from matplotlib.widgets import Slider

from freegsnke import build_machine


@pytest.fixture()
def create_machine():
    # Create the machine, which specifies coil locations
    # and equilibrium, specifying the domain to solve over
    # this has to be either
    # freegs4e.machine.MASTU(), in which case:
    # tokamak = freegs4e.machine.MASTU()
    # or
    # MASTU_coils.MASTU_wpass()
    # from freegsnke import MASTU_coils

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
    from freegsnke import equilibrium_update

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

    # Sets desired plasma properties for the 'starting equilibrium'
    # values can be changed
    from freegsnke.jtor_update import ConstrainPaxisIp

    profiles = ConstrainPaxisIp(
        eq,
        8.1e3,  # Plasma pressure on axis [Pascals]
        6.2e5,  # Plasma current [Amps]
        0.5,  # vacuum f = R*Bt
        alpha_m=1.8,
        alpha_n=1.2,
    )

    Rx = 0.6  # X-point radius
    Zx = 1.1  # X-point height
    Ra = 0.85
    Rout = 1.4  # outboard midplane radius
    Rin = 0.35  # inboard midplane radius

    # set desired null_points locations
    # this can include X-point and O-point locations
    null_points = [[Rx, Rx], [Zx, -Zx]]

    # set desired isoflux constraints with format
    # isoflux_set = [isoflux_0, isoflux_1 ... ]
    # with each isoflux_i = [R_coords, Z_coords]
    isoflux_set = np.array(
        [
            [
                [Rx, Rx, Rin, Rout, 1.3, 1.3, 0.8, 0.8],
                [Zx, -Zx, 0.0, 0.0, 2.1, -2.1, 1.62, -1.62],
            ]
        ]
    )

    # instantiate the freegsnke constrain object
    from freegsnke.inverse import Inverse_optimizer

    constrain = Inverse_optimizer(null_points=null_points, isoflux_set=isoflux_set)

    return eq, profiles, constrain


def create_test_files_static_solve(create_machine):
    """
    Saves the control currents and psi map needed for testing the static solver.
    This should not be run every test, just if there is a major change that
    changes the machine.

    Parameters
    ----------
    create_machine : pytest.fixture
        the equilibirum, profiles and constrain object to generate the test set
        from.
    """
    eq, profiles, constrain = create_machine

    from freegsnke import GSstaticsolver

    NK = GSstaticsolver.NKGSsolver(eq)

    eq.tokamak.set_coil_current("P6", 0)
    eq.tokamak["P6"].control = False
    eq.tokamak["Solenoid"].control = False
    eq.tokamak.set_coil_current("Solenoid", 15000)

    controlCurrents = np.load("./freegsnke/tests/test_controlCurrents.npy")
    eq.tokamak.setControlCurrents(controlCurrents)

    NK.forward_solve(eq, profiles, 1e-8)

    test_psi = np.load("./freegsnke/tests/test_psi.npy")


def test_static_solve(create_machine):
    """Tests the implementation of the static solver.

    Parameters
    ----------
    create_machine : pytest.fixture
        the equilibirum, profiles and constrain object to generate the test set
        from.
    """
    eq, profiles, constrain = create_machine

    from freegsnke import GSstaticsolver

    NK = GSstaticsolver.NKGSsolver(eq)

    # from freegsnke import newtonkrylov
    # NK = newtonkrylov.NewtonKrylov(eq)

    eq.tokamak.set_coil_current("P6", 0)
    eq.tokamak["P6"].control = False
    eq.tokamak["Solenoid"].control = False
    eq.tokamak.set_coil_current("Solenoid", 15000)

    # # freegs4e.solve(
    # NK.solve(
    #     eq=eq,  # The equilibrium to adjust
    #     profiles=profiles,  # The plasma profiles
    #     constrain=constrain,  # Plasma control constraints
    #     # show=False,
    #     target_relative_tolerance=1e-3,
    #     picard=False,
    # )

    controlCurrents = np.load("./freegsnke/tests/test_controlCurrents.npy")
    eq.tokamak.setControlCurrents(controlCurrents)

    NK.forward_solve(eq, profiles, 1e-8)

    test_psi = np.load("./freegsnke/tests/test_psi.npy")

    assert np.allclose(
        eq.psi(), test_psi, atol=(np.max(test_psi) - np.min(test_psi)) * 0.003
    ), "Psi map differs significantly from the test map"
