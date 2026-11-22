"""
Copyright 2026 Nicola C. Amorisco

Adapted from FreeGS4E.

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

from freegsnke.bilinear_interpolation import biliint


def test_biliint_uniform_grid():
    r = np.linspace(0.5, 2.0, 17)
    z = np.linspace(-1.2, 0.8, 25)
    R, Z = np.meshgrid(r, z, indexing="ij")
    psi = 2.0 + 3.0 * R - 4.0 * Z + 5.0 * R * Z
    points = np.array(
        [
            [0.5, 2.0, 0.5, 2.0, 0.73, 1.41, 1.89],
            [-1.2, -1.2, 0.8, 0.8, -0.64, 0.17, 0.52],
        ]
    )

    expected = 2.0 + 3.0 * points[0] - 4.0 * points[1] + 5.0 * points[0] * points[1]
    assert np.allclose(biliint(R, Z, psi, points), expected)
