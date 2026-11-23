"""
Copyright 2019-2021 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk
Copyright 2025 Timothy Nunn

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

from freegsnke.coil import Coil
from freegsnke.machine import Circuit
from freegsnke.multi_coil import MultiCoil


def test_single():
    """
    Test that a single turn MultiCoil is the same as a Coil
    """

    mcoil = MultiCoil(1.1, 0.2, current=100.0, mirror=False)
    coil = MultiCoil(1.1, 0.2, current=100.0)

    # run the test twice: once on the original coil and once on a copy
    # to check both produce the same results
    for _ in range(2):
        assert np.isclose(coil.controlPsi(0.3, 0.1), mcoil.controlPsi(0.3, 0.1))

        assert np.isclose(coil.controlBr(0.3, 0.1), mcoil.controlBr(0.3, 0.1))

        mcoil = mcoil.copy()
        coil = coil.copy()


def test_two_turns():
    """
    MultiCoil with two turns same as circuit with two coils
    """

    mcoil = MultiCoil([1.1, 1.2], [0.2, -0.3], current=100.0, mirror=False)

    circuit = Circuit(
        [
            ("A", Coil(1.1, 0.2, current=100.0), 1.0),
            ("B", Coil(1.2, -0.3, current=100.0), 1.0),
        ]
    )

    assert np.isclose(circuit.controlPsi(0.3, 0.1), mcoil.controlPsi(0.3, 0.1))

    assert np.isclose(circuit.controlBr(0.3, 0.1), mcoil.controlBr(0.3, 0.1))

    assert np.isclose(circuit.controlBz(0.3, 0.1), mcoil.controlBz(0.3, 0.1))


def test_mirrored():
    """
    Mirrored MultiCoil the same as two coils in a circuit
    """
    mcoil = MultiCoil(1.1, 0.2, current=100.0, mirror=True)

    circuit = Circuit(
        [
            ("A", Coil(1.1, 0.2, current=100.0), 1.0),
            ("B", Coil(1.1, -0.2, current=100.0), 1.0),
        ]
    )

    # run the test twice: once on the original coil and once on a copy
    # to check both produce the same results
    for _ in range(2):
        assert np.isclose(circuit.controlPsi(0.3, 0.1), mcoil.controlPsi(0.3, 0.1))

        assert np.isclose(circuit.controlBr(0.3, 0.1), mcoil.controlBr(0.3, 0.1))

        assert np.isclose(circuit.controlBz(0.3, 0.1), mcoil.controlBz(0.3, 0.1))

        mcoil = mcoil.copy()
        circuit = circuit.copy()


def test_move_R():
    """
    Changing major radius property R changes all filament locations
    """

    dR = 0.6
    coil1 = MultiCoil([1.1, 0.2], [1.2, -0.3], current=100.0, mirror=False)
    coil2 = MultiCoil([1.1 + dR, 0.2 + dR], [1.2, -0.3], current=100.0, mirror=False)

    # Shift coil1 to same location as coil2
    coil1.R += dR

    assert np.isclose(coil1.controlPsi(0.4, 0.5), coil2.controlPsi(0.4, 0.5))

    assert np.isclose(coil1.controlBr(0.3, -0.2), coil2.controlBr(0.3, -0.2))

    assert np.isclose(coil1.controlBz(1.75, 1.2), coil2.controlBz(1.75, 1.2))


def test_move_Z():
    """
    Changing height property Z changes all filament locations
    """

    dZ = 0.4
    coil1 = MultiCoil([1.1, 0.2], [1.2, -0.3], current=100.0, mirror=False)
    coil2 = MultiCoil([1.1, 0.2], [1.2 + dZ, -0.3 + dZ], current=100.0, mirror=False)

    # Shift coil1 to same location as coil2
    coil1.Z += dZ

    assert np.isclose(coil1.controlPsi(0.4, 0.5), coil2.controlPsi(0.4, 0.5))

    assert np.isclose(coil1.controlBr(0.3, -0.2), coil2.controlBr(0.3, -0.2))

    assert np.isclose(coil1.controlBz(1.75, 1.2), coil2.controlBz(1.75, 1.2))
