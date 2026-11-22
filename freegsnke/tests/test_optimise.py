"""
Copyright 2019-2021 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

from freegsnke import optimise


class DummyCoil:
    def __init__(self):
        self.R = 1.0
        self.Z = 0.0


class DummyEq:
    def __init__(self, coil_names):
        self.tokamak = {name: DummyCoil() for name in coil_names}


def test_CoilRadius():
    control_p1 = optimise.CoilRadius("P1")
    eq = DummyEq(["P1", "P2"])

    p2_R = eq.tokamak["P2"].R
    control_p1.set(eq, 2.5)

    assert np.isclose(eq.tokamak["P1"].R, 2.5)  # The value was changed
    assert np.isclose(p2_R, eq.tokamak["P2"].R)  # Other coil not changed


def test_CoilRadiusLimits():
    control_p1 = optimise.CoilRadius("P1", minimum=0.5, maximum=2)
    eq = DummyEq(["P1", "P2"])

    control_p1.set(eq, 0.1)
    assert np.isclose(eq.tokamak["P1"].R, 0.5)  # Lower limit

    control_p1.set(eq, 4)
    assert np.isclose(eq.tokamak["P1"].R, 2.0)  # Upper limit


def test_CoilHeight():
    control_p1 = optimise.CoilHeight("P1")
    eq = DummyEq(["P1", "P2"])

    p2_Z = eq.tokamak["P2"].Z
    control_p1.set(eq, 2.5)

    assert np.isclose(eq.tokamak["P1"].Z, 2.5)  # The value was changed
    assert np.isclose(p2_Z, eq.tokamak["P2"].Z)  # Other coil not changed


def test_CoilHeightLimits():
    control_p1 = optimise.CoilHeight("P1", minimum=0.5, maximum=2)
    eq = DummyEq(["P1", "P2"])

    control_p1.set(eq, 0.1)
    assert np.isclose(eq.tokamak["P1"].Z, 0.5)  # Lower limit

    control_p1.set(eq, 4)
    assert np.isclose(eq.tokamak["P1"].Z, 2.0)  # Upper limit
