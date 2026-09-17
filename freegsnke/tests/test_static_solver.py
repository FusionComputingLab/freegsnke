from copy import deepcopy
from pathlib import Path

import freegs4e
import matplotlib.pyplot as plt
import numpy as np
import pytest
from freegs4e.critical import find_critical
from freegs4e.plotting import plotConstraints
from IPython.display import clear_output, display
from matplotlib.widgets import Slider

from freegsnke import build_machine

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = Path(__file__).resolve().parent / "baselines"
MACHINE_CONFIG_DIR = REPO_ROOT / "machine_configs" / "test"

STATIC_CURRENT_BASELINE = TEST_DATA_DIR / "test_controlCurrents.npy"
STATIC_PSI_BASELINE = TEST_DATA_DIR / "test_psi.npy"


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
        active_coils_path=str(MACHINE_CONFIG_DIR / "active_coils.pickle"),
        passive_coils_path=str(MACHINE_CONFIG_DIR / "passive_coils.pickle"),
        limiter_path=str(MACHINE_CONFIG_DIR / "limiter.pickle"),
        wall_path=str(MACHINE_CONFIG_DIR / "wall.pickle"),
        magnetic_probe_path=str(MACHINE_CONFIG_DIR / "magnetic_probes.pickle"),
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

    controlCurrents = np.load(STATIC_CURRENT_BASELINE)
    eq.tokamak.setControlCurrents(controlCurrents)

    NK.forward_solve(eq, profiles, 1e-8)

    test_psi = np.load(STATIC_PSI_BASELINE)


@pytest.mark.parametrize(
    "solver,order,error,error_msg",
    [
        ("xx", 2, ValueError, "Solver type xx"),
        ("LUsparse", 3, ValueError, "operator of order 3"),
        ("DST", 4, Warning, "ignored"),
    ],
)
def test_NKGS_invalid(create_machine, solver, order, error, error_msg):
    """Tests the implementation of the static solver.

    Parameters
    ----------
    create_machine : pytest.fixture
        the equilibirum, profiles and constrain object to generate the test set
        from.
    """
    eq, profiles, constrain = create_machine

    from freegsnke import GSstaticsolver

    catcher = pytest.warns if issubclass(error, Warning) else pytest.raises

    with catcher(error, match=error_msg):
        NK = GSstaticsolver.NKGSsolver(
            eq,
            solver_type=solver,
            gs_operator_order=order,
        )


#    if issubclass(error,Warning):
#        with pytest.warns(error,match=error_msg):
#            NK = GSstaticsolver.NKGSsolver(
#                eq,
#                solver_type=solver,
#                gs_operator_order=order,
#            )
#    else:
#        with pytest.raises(error,match=error_msg):
#            NK = GSstaticsolver.NKGSsolver(
#                eq,
#                solver_type=solver,
#                gs_operator_order=order,
#            )


@pytest.mark.parametrize(
    "cache,solver,order",
    [
        (True, "LUsparse", 4),
        (False, "DST", None),
        (True, "LUsparse", 2),
        pytest.param(True, "multigrid", 2, marks=pytest.mark.xfail),
    ],
)
def test_static_solve(create_machine, cache, solver, order):
    """Tests the implementation of the static solver.

    Parameters
    ----------
    create_machine : pytest.fixture
        the equilibirum, profiles and constrain object to generate the test set
        from.
    """
    eq, profiles, constrain = create_machine

    from freegsnke import GSstaticsolver

    NK = GSstaticsolver.NKGSsolver(
        eq,
        cache_greens=cache,
        solver_type=solver,
        gs_operator_order=order,
    )

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

    controlCurrents = np.load(STATIC_CURRENT_BASELINE)
    eq.tokamak.setControlCurrents(controlCurrents)

    NK.forward_solve(eq, profiles, 1e-8)

    test_psi = np.load(STATIC_PSI_BASELINE)

    assert np.allclose(
        eq.psi(), test_psi, atol=(np.max(test_psi) - np.min(test_psi)) * 0.003
    ), "Psi map differs significantly from the test map"
    assert np.allclose(
        eq.psi_func(eq.R, eq.Z, grid=False), eq.plasma_psi
    ), "Plasma-flux interpolator is stale after the solve"


def test_second_order_static_solve(create_machine):
    """The opt-in second-order GS operator produces a consistent equilibrium."""
    eq, profiles, _ = create_machine

    from freegsnke import GSstaticsolver

    eq.tokamak.set_coil_current("P6", 0)
    eq.tokamak["P6"].control = False
    eq.tokamak["Solenoid"].control = False
    eq.tokamak.set_coil_current("Solenoid", 15000)
    eq.tokamak.setControlCurrents(np.load(STATIC_CURRENT_BASELINE))

    solver = GSstaticsolver.NKGSsolver(eq, gs_operator_order=2)
    solver.forward_solve(eq, profiles, 1e-8, suppress=True)

    reference_psi = np.load(STATIC_PSI_BASELINE)
    tolerance = np.ptp(reference_psi) * 0.003
    assert solver.gs_operator_order == 2
    assert np.allclose(eq.psi(), reference_psi, atol=tolerance)


@pytest.mark.skip(reason="this should be covered by test_NKGS_invalid above")
def test_static_solver_rejects_invalid_operator_order(create_machine):
    """Only the two finite-difference operators supplied by FreeGS4E are valid."""
    eq, _, _ = create_machine

    from freegsnke import GSstaticsolver

    with pytest.raises(ValueError, match="order"):
        GSstaticsolver.NKGSsolver(eq, gs_operator_order=3)


@pytest.mark.skip(reason="not currently using the masked/reduced version")
def test_limiter_reduced_boundary_green_is_exact(create_machine):
    """Limiter reduction preserves the boundary flux for confined current."""
    eq, _, _ = create_machine

    from freegsnke import GSstaticsolver

    solver = GSstaticsolver.NKGSsolver(eq)
    boundary_indices = solver.bndry_indices
    full_green = freegs4e.gradshafranov.Greens(
        eq.R[np.newaxis, :, :],
        eq.Z[np.newaxis, :, :],
        eq.R[:, 0][boundary_indices[:, 0]][:, np.newaxis, np.newaxis],
        eq.Z[0, :][boundary_indices[:, 1]][:, np.newaxis, np.newaxis],
    )
    full_green[
        np.arange(len(boundary_indices)),
        boundary_indices[:, 0],
        boundary_indices[:, 1],
    ] = 0.0
    full_green *= solver.dRdZ

    jtor = np.zeros_like(eq.R)
    jtor[solver.plasma_source_mask] = np.random.default_rng(0).random(
        np.count_nonzero(solver.plasma_source_mask)
    )
    expected = np.tensordot(full_green, jtor, axes=([1, 2], [0, 1]))
    actual = solver._boundary_flux_from_jtor(jtor)

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14)
