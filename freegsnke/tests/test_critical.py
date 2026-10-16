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
import pytest

from freegsnke import critical


def test_one_opoint():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.0

    # This has one O-point at (r0,z0) and no x-points
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0) ** 2) / 0.3**2)

    opoints, xpoints = critical.find_critical(r2d, z2d, psi_func(r2d, z2d))

    assert len(xpoints) == 0
    assert len(opoints) == 1
    assert np.isclose(opoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(opoints[0][1], z0, atol=1.0 / ny)


def test_one_xpoint():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.0

    # This has one X-point at (r0,z0) and no O-points
    def psi_func(R, Z):
        return (R - r0) ** 2 - (Z - z0) ** 2

    # Low-level scanning remains meaningful for vacuum fields without an axis.
    opoints, xpoints = critical.scan_for_crit(r2d, z2d, psi_func(r2d, z2d))
    assert len(opoints) == 0
    assert len(xpoints) == 1
    assert np.isclose(xpoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(xpoints[0][1], z0, atol=1.0 / ny)

    # The equilibrium-oriented wrapper requires an O-point to order and filter
    # candidate X-points relative to the magnetic axis.
    with pytest.raises(ValueError, match="No opoints found"):
        critical.find_critical(r2d, z2d, psi_func(r2d, z2d))


def test_doublet():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.1

    # This has two O-points, and one x-point at (r0, z0)
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0 - 0.3) ** 2) / 0.3**2) + np.exp(
            -((R - r0) ** 2 + (Z - z0 + 0.3) ** 2) / 0.3**2
        )

    opoints, xpoints = critical.find_critical(r2d, z2d, psi_func(r2d, z2d))

    assert len(xpoints) == 1
    assert len(opoints) == 2
    assert np.isclose(xpoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(xpoints[0][1], z0, atol=1.0 / ny)


def test_mask_zero_psi_bndry():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, nx)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.0

    # This has one O-point at (r0,z0) and no x-points
    # Range from around -0.5 to +0.5
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0) ** 2) / 0.3**2) - 0.5

    psi = psi_func(r2d, z2d)

    opoints, xpoints = critical.find_critical(r2d, z2d, psi)

    assert len(xpoints) == 0
    assert len(opoints) == 1
    assert np.isclose(opoints[0][0], r0, atol=1.0 / nx)
    assert np.isclose(opoints[0][1], z0, atol=1.0 / ny)

    mask = critical.core_mask(r2d, z2d, psi, opoints, xpoints, psi_bndry=0.0)
    inside_mask = critical.inside_mask(r2d, z2d, psi, opoints, xpoints, psi_bndry=0.0)
    inside_mask_without_guard = critical.inside_mask(
        r2d, z2d, psi, opoints, xpoints, psi_bndry=0.0, use_geom=False
    )

    # Some of the mask must equal 1, some 0
    assert np.any(np.equal(mask, 1))
    assert np.any(np.equal(mask, 0))
    assert np.array_equal(inside_mask, inside_mask_without_guard)


def test_inside_mask_single_xpoint_does_not_apply_double_null_guard():
    nx = 21
    ny = 41

    r1d = np.linspace(0.5, 1.5, nx)
    z1d = np.linspace(-1.0, 1.0, ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    psi = (r2d - 1.0) ** 2 + z2d**2
    opoints = np.array([[1.0, 0.0, 0.0]])
    xpoints = np.array([[1.0, -0.8, 0.64]])

    mask = critical.inside_mask(r2d, z2d, psi, opoints, xpoints)

    assert mask.shape == psi.shape
    assert mask.dtype == bool
    assert np.any(mask)


def test_geom_inside_mask_with_horizontally_aligned_xpoint():
    r1d = np.linspace(0.5, 1.5, 21)
    z1d = np.linspace(-1.0, 1.0, 41)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")
    opoints = np.array([[1.5, 0.0, 1.0]])
    xpoints = np.array([[1.0, 0.0, 0.0]])

    mask = critical.geom_inside_mask(r2d, z2d, opoints, xpoints)

    assert np.array_equal(mask, r2d > 1.0)
