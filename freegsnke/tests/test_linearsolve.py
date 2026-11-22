"""
Tests of the linear solver

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

from freegsnke import multigrid

# Laplacian in 2D


def test_direct_laplacian():
    nx = 65
    ny = 65

    Lx = 2.0
    Ly = 3.0

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    xx, yy = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    solution = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4**2)

    A = multigrid.LaplacianOp()
    rhs = A(solution, dx, dy)

    # Copy boundaries
    rhs[:, 0] = solution[:, 0]
    rhs[0, :] = solution[0, :]
    rhs[:, -1] = solution[:, -1]
    rhs[-1, :] = solution[-1, :]

    solve = multigrid.MGDirect(multigrid.LaplaceSparse(Lx, Ly)(nx, ny))
    x = solve(None, rhs)

    assert np.allclose(x, solution)


def test_multigrid_laplacian():
    nx = 65
    ny = 65

    Lx = 2.0
    Ly = 3.0

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    xx, yy = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    solution = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4**2)

    A = multigrid.LaplacianOp()
    rhs = A(solution, dx, dy)

    # Copy boundaries
    rhs[:, 0] = solution[:, 0]
    rhs[0, :] = solution[0, :]
    rhs[:, -1] = solution[:, -1]
    rhs[-1, :] = solution[-1, :]

    solve = multigrid.createVcycle(
        nx,
        ny,
        multigrid.LaplaceSparse(Lx, Ly),
        ncycle=50,
        niter=50,
        nlevels=4,
        direct=True,
    )

    xinput = np.zeros(rhs.shape)
    xinput[:, 0] = solution[:, 0]
    xinput[0, :] = solution[0, :]
    xinput[:, -1] = solution[:, -1]
    xinput[-1, :] = solution[-1, :]
    x = solve(xinput, rhs)

    assert np.allclose(x, solution, atol=1e-6)
