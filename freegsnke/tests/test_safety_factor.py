"""
Tests of the safety factor calculation (``Equilibrium.q``) with tension-spline profiles.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE. If not, see <http://www.gnu.org/licenses/>.
"""

import pickle
from pathlib import Path

import contourpy
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, RectBivariateSpline

from freegsnke import GSstaticsolver, build_machine, equilibrium_update
from freegsnke.jtor_update import TensionSpline

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTU_CONFIG_DIR = REPO_ROOT / "machine_configs" / "MAST-U"
EXAMPLE_DATA_DIR = REPO_ROOT / "examples" / "data"

PSI_N = np.linspace(0, 1, 11)
PPRIME = np.array(
    [
        208061.72441505,
        204110.95531309,
        194359.61041113,
        179811.09692964,
        161053.35395914,
        138602.79928382,
        113000.64896365,
        84893.04354249,
        55164.04720078,
        25298.53358472,
        0.0,
    ]
)
FFPRIME = np.array(
    [
        0.58961295,
        0.57841712,
        0.5507834,
        0.50955529,
        0.45639891,
        0.3927777,
        0.32022539,
        0.24057302,
        0.1563259,
        0.07169191,
        0.0,
    ]
)


def _tension_spline_profiles(eq, Ip, fvac):
    def second_derivatives(values):
        return CubicSpline(PSI_N, values, bc_type="natural")(PSI_N, 2)

    return TensionSpline(
        eq=eq,
        Ip=Ip,
        fvac=fvac,
        pp_knots=PSI_N,
        pp_values=PPRIME,
        pp_values_2=second_derivatives(PPRIME),
        pp_sigma=1.0,
        ffp_knots=PSI_N,
        ffp_values=FFPRIME,
        ffp_values_2=second_derivatives(FFPRIME),
        ffp_sigma=1.0,
    )


def _q_from_contours(eq, profiles, psinorm):
    """Independent q calculation: q = F/(2 pi) * contour integral of dl / (R |grad psi|)."""
    psi_n = (eq.psi() - profiles.psi_axis) / (profiles.psi_bndry - profiles.psi_axis)
    psi_spline = RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], eq.psi())
    contours = contourpy.contour_generator(
        eq.R, eq.Z, psi_n, line_type=contourpy.LineType.Separate
    )
    q = []
    for level in psinorm:
        line = max(contours.lines(level), key=len)
        r, z = line[:, 0], line[:, 1]
        dl = np.hypot(np.diff(np.append(r, r[0])), np.diff(np.append(z, z[0])))
        r_mid, z_mid = 0.5 * (r + np.roll(r, -1)), 0.5 * (z + np.roll(z, -1))
        grad_psi = np.hypot(
            psi_spline(r_mid, z_mid, dx=1, grid=False),
            psi_spline(r_mid, z_mid, dy=1, grid=False),
        )
        q.append(eq.fpol(level) / (2 * np.pi) * np.sum(dl / (r_mid * grad_psi)))
    return np.array(q)


def _check_q(eq, profiles):
    psinorm = np.array([0.2, 0.5, 0.8])
    q = eq.q(psinorm)
    assert q.shape == (3,)
    assert np.all(np.isfinite(q)) and np.all(np.diff(q) > 0)
    assert np.allclose(q, _q_from_contours(eq, profiles, psinorm), rtol=1e-2)

    # every form of single value gives a float equal to the array result
    for single in (0.5, np.float64(0.5), np.array(0.5), [0.5], np.array([0.5])):
        value = eq.q(single)
        assert isinstance(value, float)
        assert np.isclose(value, q[1], rtol=1e-12)

    # other input shapes return arrays of the same shape
    assert np.allclose(eq.q([0.2, 0.5, 0.8]), q, rtol=1e-12)
    grid_q = eq.q(np.array([[0.2, 0.5], [0.8, 0.5]]))
    assert grid_q.shape == (2, 2)
    assert np.allclose(grid_q.reshape(-1), q[[0, 1, 2, 1]], rtol=1e-12)


@pytest.fixture(scope="module")
def mastu_machine():
    return build_machine.tokamak(
        active_coils_path=str(MASTU_CONFIG_DIR / "MAST-U_like_active_coils.pickle"),
        passive_coils_path=str(MASTU_CONFIG_DIR / "MAST-U_like_passive_coils.pickle"),
        limiter_path=str(MASTU_CONFIG_DIR / "MAST-U_like_limiter.pickle"),
        wall_path=str(MASTU_CONFIG_DIR / "MAST-U_like_wall.pickle"),
    )


def _solve_mastu(tokamak, currents_file, Ip):
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=129
    )
    profiles = _tension_spline_profiles(eq, Ip=Ip, fvac=0.5)
    with open(EXAMPLE_DATA_DIR / currents_file, "rb") as f:
        for coil, current in pickle.load(f).items():
            eq.tokamak.set_coil_current(coil, current)
    GSstaticsolver.NKGSsolver(eq, gs_operator_order=4).solve(
        eq=eq, profiles=profiles, constrain=None, target_relative_tolerance=1e-9
    )
    return eq, profiles


def test_q_tension_spline_diverted(mastu_machine):
    eq, profiles = _solve_mastu(
        mastu_machine, "simple_diverted_currents_PaxisIp.pk", Ip=6e5
    )
    assert not profiles.flag_limiter
    _check_q(eq, profiles)


def test_q_tension_spline_limiter_with_xpoints_in_grid(mastu_machine):
    """The boundary of a limiter plasma is psi_bndry, not the flux of an X-point outside it."""
    eq, profiles = _solve_mastu(
        mastu_machine, "simple_limited_currents_PaxisIp.pk", Ip=4e5
    )
    assert profiles.flag_limiter
    assert len(profiles.xpt) > 0
    _check_q(eq, profiles)


def test_q_tension_spline_limiter_without_xpoints_in_grid():
    """No X-point in the grid: the profile still needs psi_axis for fpol (and so q)."""
    d = 0.05

    def coil(R, Z):
        return {
            "R": [R],
            "Z": [Z],
            "dR": d,
            "dZ": d,
            "resistivity": 1e-8,
            "polarity": 1.0,
            "multiplier": 1.0,
        }

    active_coils = {
        "P1": {"P11": coil(0.5, 0.8), "P12": coil(0.5, -0.8)},
        "P2": {"P21": coil(1.6, 0.6), "P22": coil(1.6, -0.6)},
        "P5": coil(1.6, 0.0),
    }
    limiter = [
        {"R": r, "Z": z}
        for r, z in zip([0.6, 1.3, 1.3, 0.6, 0.6], [-0.48, -0.48, 0.48, 0.48, -0.48])
    ]
    tokamak = build_machine.tokamak(
        active_coils_data=active_coils, limiter_data=limiter, wall_data=limiter
    )
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak, Rmin=0.55, Rmax=1.35, Zmin=-0.5, Zmax=0.5, nx=65, ny=65
    )
    for coil_name, current in {
        "P1": 193383.1596574935,
        "P2": -210499.54693910183,
        "P5": 69362.37774995311,
    }.items():
        eq.tokamak.set_coil_current(coil_name, current)
    profiles = _tension_spline_profiles(eq, Ip=0.39e6, fvac=1.0)
    GSstaticsolver.NKGSsolver(eq).solve(
        eq=eq,
        profiles=profiles,
        target_relative_tolerance=1e-8,
        max_solving_iterations=80,
    )

    assert profiles.flag_limiter and len(profiles.xpt) == 0
    assert np.isclose(profiles.psi_axis, eq.psi_axis)
    _check_q(eq, profiles)
