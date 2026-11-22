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

from freegsnke import quadrature


def test_unity():
    # Sum of weights is one i.e quadrature works out average over area

    for n in [1, 3, 6]:
        ps = quadrature.polygon_quad([(0, 0), (0, 1), (1, 1), (1, 0)], n=n)
        assert len(ps) == 2 * n  # Two triangles
        assert np.isclose(sum(weight for r, z, weight in ps), 1.0)


def test_integral():
    # Limits of integration
    x0, x1 = 0.4, 1.3
    y0, y1 = -0.2, 3.1

    def func(x, y):
        return x**2 - y**3 + x * y

    exact_integral = (
        (x1**3 - x0**3) * (y1 - y0) / 3
        - (y1**4 - y0**4) * (x1 - x0) / 4
        + (x1**2 - x0**2) * (y1**2 - y0**2) / 4
    )

    # A 1st order method can't integrate this polynomial exactly
    quad1 = quadrature.polygon_quad([(x0, y0), (x0, y1), (x1, y1), (x1, y0)], n=1)
    assert not np.isclose(
        quadrature.average(func, quad1),
        exact_integral / ((x1 - x0) * (y1 - y0)),
    )

    for n in [3, 6]:
        # Higher order methods can
        quad1 = quadrature.polygon_quad([(x0, y0), (x0, y1), (x1, y1), (x1, y0)], n=n)
        assert len(quad1) == 2 * n
        assert np.isclose(
            quadrature.average(func, quad1),
            exact_integral / ((x1 - x0) * (y1 - y0)),
        )
