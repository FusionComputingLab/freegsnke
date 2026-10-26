"""
Copyright 2020-2021 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk
Copyright 2026 Nicola C. Amorisco

Derived from FreeGS through FreeGS4E.

SPDX-License-Identifier: LGPL-3.0-or-later

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.integrate import quad

from freegsnke import jtor
from freegsnke.gradshafranov import mu0


def test_psinorm_range():
    """Test that the profiles produce finite values outside core"""

    for profiles in [
        jtor.ConstrainPaxisIp(1e3, 2e5, 2.0),
        jtor.ConstrainBetapIp(1.0, 2e5, 2.0),
    ]:

        # Need to give a plasma psi
        R, Z = np.meshgrid(
            np.linspace(0.5, 1.5, 33), np.linspace(-1, 1, 33), indexing="ij"
        )
        psi = np.exp((-((R - 1.0) ** 2) - Z**2) * 3) + np.exp(
            (-((R - 1.0) ** 2) - (Z + 1) ** 2) * 3
        )

        current_density = profiles.Jtor(R, Z, psi)
        assert np.all(np.isfinite(current_density))

        assert profiles.pprime(1.0) == 0.0
        assert profiles.pprime(1.1) == 0.0
        assert np.isfinite(profiles.pprime(-0.32))

        assert profiles.ffprime(1.0) == 0.0
        assert profiles.ffprime(1.1) == 0.0
        assert np.isfinite(profiles.ffprime(-0.32))


def test_profile_jtor_stores_plotting_metadata():
    """Test that the generic Jtor wrapper records downstream metadata."""

    profiles = jtor.ConstrainBetapIp(1.0, 2e5, 2.0)
    R, Z = np.meshgrid(np.linspace(0.5, 1.5, 33), np.linspace(-1, 1, 33), indexing="ij")
    psi = np.exp((-((R - 1.0) ** 2) - Z**2) * 3) + np.exp(
        (-((R - 1.0) ** 2) - (Z + 1) ** 2) * 3
    )

    profiles.Jtor(R, Z, psi)

    assert hasattr(profiles, "opt")
    assert hasattr(profiles, "xpt")
    assert hasattr(profiles, "flag_limiter")
    assert hasattr(profiles, "psi_axis")
    assert hasattr(profiles, "psi_bndry")
    assert hasattr(profiles, "diverted_core_mask")
    assert hasattr(profiles, "limiter_core_mask")


def _nonunit_alpha_m_profile_grid():
    R, Z = np.meshgrid(
        np.linspace(0.5, 1.5, 65),
        np.linspace(-1.0, 1.0, 65),
        indexing="ij",
    )
    psi = np.exp((-((R - 1.0) ** 2) - Z**2) * 3) + np.exp(
        (-((R - 1.0) ** 2) - (Z + 1.0) ** 2) * 3
    )
    return R, Z, psi


def test_constrain_betap_ip_matches_requested_beta_for_nonunit_alpha_m():
    requested_betap = 0.2
    profiles = jtor.ConstrainBetapIp(
        requested_betap, 2e5, 2.0, alpha_m=4.0, alpha_n=1.0
    )
    R, Z, psi = _nonunit_alpha_m_profile_grid()

    current_density = profiles.Jtor(R, Z, psi)
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    psi_norm = (psi - profiles.psi_axis) / (profiles.psi_bndry - profiles.psi_axis)
    pressure = profiles.pressure(psi_norm) * profiles.limiter_core_mask
    calculated_betap = (
        (8.0 * np.pi / mu0)
        * np.sum(pressure)
        * dR
        * dZ
        / (np.sum(current_density) * dR * dZ) ** 2
    )

    assert np.isclose(calculated_betap, requested_betap, rtol=1e-10)


def test_constrain_paxis_ip_matches_requested_pressure_for_nonunit_alpha_m():
    requested_paxis = 1.2e3
    profiles = jtor.ConstrainPaxisIp(
        requested_paxis, 2e5, 2.0, alpha_m=4.0, alpha_n=1.0
    )
    R, Z, psi = _nonunit_alpha_m_profile_grid()

    profiles.Jtor(R, Z, psi)

    assert np.isclose(profiles.pressure(np.asarray(0.0)), requested_paxis)


def test_profile_vectorised_integrals_match_adaptive_reference():
    """The shared quadrature remains accurate for a non-integer profile shape."""
    profiles = jtor.ConstrainPaxisIp(
        1.2e3, 2e5, 2.0, alpha_m=0.5, alpha_n=2.0
    )
    R, Z, psi = _nonunit_alpha_m_profile_grid()
    profiles.Jtor(R, Z, psi)
    psinorm = np.array([0.01, 0.25, 0.6, 0.9])
    psi_range = profiles.psi_axis - profiles.psi_bndry

    expected_pressure = np.array(
        [quad(profiles.pprime, pn, 1.0)[0] * psi_range for pn in psinorm]
    )
    expected_fpol = np.sqrt(
        [
            2.0 * quad(profiles.ffprime, pn, 1.0)[0] * psi_range
            + profiles.fvac() ** 2
            for pn in psinorm
        ]
    )

    np.testing.assert_allclose(
        profiles.pressure(psinorm), expected_pressure, rtol=1e-5
    )
    np.testing.assert_allclose(profiles.fpol(psinorm), expected_fpol, rtol=1e-5)


def test_fiesta_topeol_is_consistent_for_nonunit_alpha_m():
    requested_current = 2e5
    profiles = jtor.Fiesta_Topeol(
        0.35, requested_current, 2.0, alpha_m=4.0, alpha_n=1.0
    )
    R, Z, psi = _nonunit_alpha_m_profile_grid()

    current_density = profiles.Jtor(R, Z, psi)
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    psi_norm = (psi - profiles.psi_axis) / (profiles.psi_bndry - profiles.psi_axis)
    reconstructed = (
        R * profiles.pprime(psi_norm) + profiles.ffprime(psi_norm) / (mu0 * R)
    ) * profiles.limiter_core_mask

    assert np.isclose(np.sum(current_density) * dR * dZ, requested_current)
    np.testing.assert_allclose(current_density, reconstructed)


def _continuity_test_profile(shape):
    profiles = jtor.ConstrainPaxisIp(1e3, 2e5, 2.0)
    profiles.mask_inside_limiter = np.ones(shape, dtype=bool)
    profiles.edge_mask = np.zeros(shape, dtype=bool)
    profiles.edge_mask[[0, -1], :] = True
    profiles.edge_mask[:, [0, -1]] = True
    profiles.diverted_core_mask = np.zeros(shape, dtype=bool)
    profiles.diverted_core_mask.flat[:16] = True
    return profiles


def test_alternative_xpoint_sets_matching_boundary(monkeypatch):
    """An accepted alternative X-point also supplies the returned boundary flux."""
    shape = (5, 5)
    R, Z = np.meshgrid(np.arange(5.0), np.arange(5.0), indexing="ij")
    psi = np.ones(shape)
    opt = np.array([[2.0, 2.0, 1.0]])
    xpt = np.array([[2.0, 1.0, 0.9], [2.0, 3.0, 0.8]])
    primary_mask = np.zeros(shape, dtype=bool)
    primary_mask[2:4, 2:4] = True
    alternative_mask = np.zeros(shape, dtype=bool)
    alternative_mask[1:4, 1:4] = True

    monkeypatch.setattr(jtor.critical, "find_critical", lambda *args: (opt, xpt))
    monkeypatch.setattr(
        jtor.critical,
        "inside_mask",
        lambda *args: (primary_mask if np.isclose(args[-1], 0.9) else alternative_mask),
    )

    profiles = _continuity_test_profile(shape)
    _, selected_xpt, mask, psi_bndry = profiles.Jtor_part1(R, Z, psi)

    assert np.array_equal(selected_xpt, xpt[1:])
    assert np.isclose(psi_bndry, selected_xpt[0, 2])
    assert np.array_equal(mask, alternative_mask)


def test_alternative_xpoint_search_advances_after_edge_mask(monkeypatch):
    """An edge-touching alternative is skipped without stalling the search."""
    shape = (5, 5)
    R, Z = np.meshgrid(np.arange(5.0), np.arange(5.0), indexing="ij")
    psi = np.ones(shape)
    opt = np.array([[2.0, 2.0, 1.0]])
    xpt = np.array([[2.0, 1.0, 0.9], [2.0, 3.0, 0.8], [3.0, 2.0, 0.7]])
    primary_mask = np.zeros(shape, dtype=bool)
    primary_mask[2:4, 2:4] = True
    edge_mask = np.ones(shape, dtype=bool)
    final_mask = np.zeros(shape, dtype=bool)
    final_mask[1:4, 1:4] = True
    boundaries = []

    def inside_mask(*args):
        psi_bndry = args[-1]
        boundaries.append(psi_bndry)
        if np.isclose(psi_bndry, 0.9):
            return primary_mask
        if np.isclose(psi_bndry, 0.8):
            return edge_mask
        return final_mask

    monkeypatch.setattr(jtor.critical, "find_critical", lambda *args: (opt, xpt))
    monkeypatch.setattr(jtor.critical, "inside_mask", inside_mask)

    profiles = _continuity_test_profile(shape)
    _, selected_xpt, mask, psi_bndry = profiles.Jtor_part1(R, Z, psi)

    assert np.allclose(boundaries, [0.9, 0.8, 0.7])
    assert np.array_equal(selected_xpt, xpt[2:])
    assert np.isclose(psi_bndry, selected_xpt[0, 2])
    assert np.array_equal(mask, final_mask)


def _tension_spline_profile():
    knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    return jtor.TensionSpline(
        Ip=2e5,
        fvac=2.0,
        pp_knots=knots,
        pp_values=np.array([1.0, 0.8, 0.5, 0.2, 0.0]),
        pp_values_2=np.array([0.0, -0.5, 0.3, -0.2, 0.0]),
        pp_sigma=2.0,
        ffp_knots=knots,
        ffp_values=np.array([0.5, 0.4, 0.3, 0.1, 0.0]),
        ffp_values_2=np.array([0.0, 0.2, -0.1, 0.1, 0.0]),
        ffp_sigma=1.5,
    )


def test_tension_spline_accepts_scalars_and_arrays():
    """Scalars and arrays of any shape give the same tension-spline values."""
    profiles = _tension_spline_profile()
    args = (
        profiles.pp_knots,
        profiles.pp_values,
        profiles.pp_values_2,
        profiles.pp_sigma,
    )
    x = np.array([0.05, 0.3, 0.61, 0.9])
    expected = profiles.tension_spline(x, *args)

    for xi, ei in zip(x, expected):
        scalar = profiles.tension_spline(xi, *args)
        assert isinstance(scalar, float)
        assert np.isclose(scalar, ei, rtol=1e-12)
        assert np.isclose(profiles.tension_spline(float(xi), *args), ei, rtol=1e-12)
        zero_d = profiles.tension_spline(np.array(xi), *args)
        assert np.isclose(zero_d, ei, rtol=1e-12)

    assert profiles.tension_spline(x[:1], *args).shape == (1,)
    grid = profiles.tension_spline(x.reshape(2, 2), *args)
    assert grid.shape == (2, 2)
    assert np.allclose(grid.reshape(-1), expected, rtol=1e-12)


def test_tension_spline_reproduces_knot_values_in_any_position():
    """Evaluating at the knots returns the knot values, wherever they sit in x."""
    profiles = _tension_spline_profile()
    knots, values = profiles.pp_knots, profiles.pp_values
    args = (knots, values, profiles.pp_values_2, profiles.pp_sigma)

    for order in (np.arange(5), np.arange(5)[::-1], np.array([2, 0, 4, 1, 3])):
        result = profiles.tension_spline(knots[order], *args)
        assert np.allclose(result, values[order], rtol=1e-12)

    # an interior knot at the start or end of x must not be double counted
    assert np.isclose(profiles.tension_spline(0.5, *args), values[2])
    assert np.isclose(
        profiles.tension_spline(np.array([0.5, 0.6]), *args)[0], values[2]
    )
    assert np.isclose(
        profiles.tension_spline(np.array([0.4, 0.5]), *args)[1], values[2]
    )


def test_tension_spline_profile_fpol_and_pressure_accept_scalars():
    """fpol and pressure integrate with quad, which evaluates the profile at scalars."""
    profiles = _tension_spline_profile()
    # normally set when the current density is first calculated
    profiles.psi_axis = 1.0
    profiles.psi_bndry = 0.0
    profiles.L = 1.0

    psinorm = np.array([0.1, 0.4, 0.8])
    for method in (profiles.fpol, profiles.pressure):
        from_array = method(psinorm)
        assert np.all(np.isfinite(from_array))
        for pn, expected in zip(psinorm, from_array):
            assert np.isclose(method(float(pn)), expected, rtol=1e-12)
