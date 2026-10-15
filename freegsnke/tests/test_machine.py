"""
Copyright 2018-2021 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk
Copyright 2025 Timothy Nunn
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

#
# Test calculation of magnetic field due to coils
# and forces between coils

import math

import numpy as np

from freegsnke import machine

mu0 = 4e-7 * np.pi


def test_coil_axis():
    """
    Test magnetic field due to a single coil, on axis
    """

    Rcoil = 1.5
    current = 123.0
    coil = machine.Coil(Rcoil, 1.0, current=current)

    def analytic_Bz(dZ):
        return (mu0 / 2) * Rcoil**2 * current / (dZ**2 + Rcoil**2) ** 1.5

    # run the test twice: once on the original coil and once on a copy
    # to check both produce the same results
    for _ in range(2):
        # Note: Can't evaluate at R=0,
        assert math.isclose(coil.Br(0.0001, 2.0), 0.0, abs_tol=1e-8)
        assert math.isclose(coil.Bz(0.001, 2.0), analytic_Bz(1.0), abs_tol=1e-8)
        assert math.isclose(coil.Bz(0.001, -1.0), analytic_Bz(-2.0), abs_tol=1e-8)

        coil = coil.copy()


def test_coil_forces():
    """
    Test forces between two coils
    """
    Rcoil = 1.5
    current = 123.0

    coil1 = machine.Coil(Rcoil, 1.0, current=current)
    coil2 = machine.Coil(Rcoil, -1.0, current=current)

    tokamak = machine.Machine([("P1", coil1), ("P2", coil2)])

    forces = tokamak.getForces()

    assert "P1" in forces and "P2" in forces

    # Vertical force is equal and opposite
    assert math.isclose(forces["P1"][1], -forces["P2"][1])
    assert forces["P1"][1] < 0.0  # Force downward towards other coil

    # Reverse one of the currents
    coil2.current = -123.0
    # check the force reverses direction
    forces = tokamak.getForces()
    assert forces["P1"][1] > 0.0

    print(forces)


def test_coil_forces_unequal():
    """
    Test forces between two different sized coils
    """

    coil1 = machine.Coil(1.5, 1.0, current=123)
    coil2 = machine.Coil(2.5, -1.0, current=456)

    tokamak = machine.Machine([("P1", coil1), ("P2", coil2)])

    forces = tokamak.getForces()

    assert "P1" in forces and "P2" in forces

    # Vertical force is equal and opposite
    assert math.isclose(forces["P1"][1], -forces["P2"][1])
    assert forces["P1"][1] < 0.0  # Force downward towards other coil


def test_get_psi_tokamak_matches_weighted_green_sum():
    """The optimized contraction must preserve the coil-weighted flux sum."""

    tokamak = machine.Machine(
        [
            ("P1", machine.Coil(1.0, 0.5)),
            ("P2", machine.Coil(1.5, 0.0)),
            ("P3", machine.Coil(2.0, -0.5)),
        ]
    )
    tokamak.set_all_coil_currents(np.array([125.0, -40.0, 310.0]))

    vgreen = np.random.default_rng(42).normal(size=(3, 17, 25))
    expected = np.sum(vgreen * tokamak.current_vec[:, np.newaxis, np.newaxis], axis=0)

    np.testing.assert_allclose(tokamak.getPsitokamak(vgreen), expected)

    empty_vgreen = np.empty((0, 17, 25))
    np.testing.assert_array_equal(
        machine.Machine([]).getPsitokamak(empty_vgreen), np.zeros((17, 25))
    )
