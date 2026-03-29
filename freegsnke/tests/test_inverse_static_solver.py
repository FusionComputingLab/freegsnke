from pathlib import Path

import numpy as np
import pytest
from freegs4e.critical import find_critical

from freegsnke import GSstaticsolver, build_machine, equilibrium_update
from freegsnke.inverse import Inverse_optimizer
from freegsnke.jtor_update import ConstrainPaxisIp


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = REPO_ROOT / "freegsnke" / "tests"
MACHINE_CONFIG_DIR = REPO_ROOT / "machine_configs" / "MAST-U"

INVERSE_CURRENT_BASELINE = TEST_DATA_DIR / "test_inverse_control_currents.npy"
INVERSE_PSI_BASELINE = TEST_DATA_DIR / "test_inverse_psi.npy"


def _build_diverted_inverse_case():
    tokamak = build_machine.tokamak(
        active_coils_path=str(
            MACHINE_CONFIG_DIR / "MAST-U_like_active_coils.pickle"
        ),
        passive_coils_path=str(
            MACHINE_CONFIG_DIR / "MAST-U_like_passive_coils.pickle"
        ),
        limiter_path=str(MACHINE_CONFIG_DIR / "MAST-U_like_limiter.pickle"),
        wall_path=str(MACHINE_CONFIG_DIR / "MAST-U_like_wall.pickle"),
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
        eq=eq,
        paxis=8e3,
        Ip=6e5,
        fvac=0.5,
        alpha_m=1.8,
        alpha_n=1.2,
    )

    rx = 0.6
    zx = 1.1
    rout = 1.4
    rin = 0.34

    null_points = [[rx, rx], [zx, -zx]]
    isoflux_set = np.array(
        [
            [
                [rx, rx, rin, rout, 1.0, 1.0, 0.8, 0.8],
                [zx, -zx, 0.0, 0.0, 2.0, -2.0, 1.62, -1.62],
            ]
        ]
    )
    coil_current_limits = [
        [5e3, 9e3, 9e3, 7e3, 7e3, 5e3, 4e3, 5e3, 0.0, 0.0, None],
        [-5e3, -9e3, -9e3, -7e3, -7e3, -5e3, -4e3, -5e3, -10e3, -10e3, None],
    ]

    constrain = Inverse_optimizer(
        null_points=null_points,
        isoflux_set=isoflux_set,
        coil_current_limits=coil_current_limits,
    )
    constrain.mu_coils = 1e5

    eq.tokamak.set_coil_current("Solenoid", 5000)
    eq.tokamak["Solenoid"].control = False

    return eq, profiles, constrain, coil_current_limits


def _generate_inverse_static_baselines():
    """Helper function to generate regression baselines for the diverted inverse static solver test.
    
    This should be run once and the generated .npy files should be committed to the repository.

    Only run this if there is a major change to the inverse static solver that would cause the control currents or psi map to change significantly, otherwise the regression test should catch any unintended changes.
    """
    eq, profiles, constrain, _ = _build_diverted_inverse_case()

    GSstaticsolver.NKGSsolver(eq=eq).solve(
        eq=eq,
        profiles=profiles,
        constrain=constrain,
        target_relative_tolerance=1e-6,
        target_relative_psit_update=1e-3,
        verbose=False,
        l2_reg=np.array([1e-12] * 10 + [1e-6]),
    )

    np.save(INVERSE_CURRENT_BASELINE, np.asarray(eq.tokamak.getCurrentsVec())[:12])
    np.save(INVERSE_PSI_BASELINE, eq.psi())


@pytest.fixture()
def diverted_inverse_case():
    return _build_diverted_inverse_case()


def test_inverse_static_diverted_solve_regression(diverted_inverse_case):
    eq, profiles, constrain, coil_current_limits = diverted_inverse_case

    solver = GSstaticsolver.NKGSsolver(eq=eq)
    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=constrain,
        target_relative_tolerance=1e-6,
        target_relative_psit_update=1e-3,
        verbose=False,
        l2_reg=np.array([1e-12] * 10 + [1e-6]),
    )

    solved_currents = np.asarray(eq.tokamak.getCurrentsVec())[:12]
    reference_currents = np.load(INVERSE_CURRENT_BASELINE)
    reference_psi = np.load(INVERSE_PSI_BASELINE)

    assert np.allclose(solved_currents, reference_currents, atol=1e-9), (
        "Inverse-solve control currents differ from the regression baseline"
    )

    psi_tolerance = (np.max(reference_psi) - np.min(reference_psi)) * 0.003
    assert np.allclose(eq.psi(), reference_psi, atol=psi_tolerance), (
        "Inverse-solve psi map differs significantly from the regression baseline"
    )

    opt, xpt = find_critical(
        eq.R,
        eq.Z,
        eq.psi(),
        eq.mask_inside_limiter.astype(bool),
        None,
    )

    assert len(opt) == 1, "Expected a single magnetic axis in the diverted solution"
    assert len(xpt) >= 2, "Expected at least two X-points in the diverted solution"
    assert np.allclose(opt[0][:2], [0.951053009, 0.0], atol=5e-4), (
        "Magnetic axis location drifted from the diverted regression solution"
    )

    expected_xpts = np.array([[0.59848009, -1.09716935], [0.59848008, 1.09716927]])
    for expected_xpt in expected_xpts:
        distances = np.linalg.norm(xpt[:, :2] - expected_xpt, axis=1)
        assert np.min(distances) <= 5e-4, (
            "Primary X-point locations drifted from the diverted regression solution"
        )

    upper_limits = [None] + coil_current_limits[0]
    lower_limits = [None] + coil_current_limits[1]
    for coil_name, coil_current, upper_limit, lower_limit in zip(
        eq.tokamak.coils_dict,
        solved_currents,
        upper_limits,
        lower_limits,
    ):
        if upper_limit is None or lower_limit is None:
            continue

        assert lower_limit <= coil_current <= upper_limit, (
            f"{coil_name} current {coil_current} violates [{lower_limit}, {upper_limit}]"
        )