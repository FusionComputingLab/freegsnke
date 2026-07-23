"""
Defines a few properties of the plasma current equation (i.e. the lumped parameter model 
used as an effective circuit equation for the plasma). 

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
  
You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>. 
"""

import numpy as np
from freegs4e.gradshafranov import Greens


def Myy(plasma_pts):
    """
    Compute the mutual inductance matrix between plasma grid points.

    The matrix is constructed using the Green's function evaluated between
    all pairs of plasma points in cylindrical (R, Z) coordinates.

    Parameters
    ----------
    plasma_pts : ndarray, shape (N, 2)
        Array of plasma grid point coordinates in cylindrical geometry:
        - plasma_pts[:, 0] = R coordinates
        - plasma_pts[:, 1] = Z coordinates

    Returns
    -------
    Myy : ndarray, shape (N, N)
        Mutual inductance matrix between all plasma grid points.
        The matrix is symmetric.

    Notes
    -----
    The matrix is computed as:

        Myy_ij = 2π * G(R_i, Z_i, R_j, Z_j)

    where G is the Green's function returned by `Greens`.

    """

    greenm = Greens(
        plasma_pts[:, np.newaxis, 0],
        plasma_pts[:, np.newaxis, 1],
        plasma_pts[np.newaxis, :, 0],
        plasma_pts[np.newaxis, :, 1],
    )
    return 2 * np.pi * greenm


def grid_greens(R, Z):
    """
    Compute the Green's function matrix for a structured (R, Z) grid.

    This function evaluates the Green's function between a set of radial
    grid points and an axially discretised Z-grid, using broadcasting to
    construct all pairwise interactions efficiently.

    Parameters
    ----------
    R : ndarray, shape (N, M)
        Radial grid values. Only the first column is used (R[:, 0]).
    Z : ndarray, shape (N, M)
        Axial grid values defining a uniform Z spacing.

    Returns
    -------
    ggreens : ndarray
        Green's function evaluated on the expanded grid, scaled by 2π.
        Shape depends on broadcasting of the underlying `Greens` function.

    Notes
    -----
    - The Z-grid is assumed to be uniformly spaced in the second dimension.
    - The spacing is computed as:
          dz = Z[0, 1] - Z[0, 0]
    - The number of Z points is inferred as:
          nZ = Z.shape[1]
    - The Green's function is evaluated as:
          G(R_i, z_k, R_j, z=0)
      where the axial coordinate for the second argument is constructed as:
          z_k = dz * arange(nZ)

    Warning
    -------
    - Only `R[:, 0]` is used; any variation in R across the second dimension is ignored.
    - Assumes uniform spacing in Z; non-uniform grids will produce incorrect results.
    """

    dz = Z[0, 1] - Z[0, 0]
    nZ = np.shape(Z)[1]

    ggreens = Greens(
        R[:, 0][:, np.newaxis, np.newaxis],
        dz * np.arange(nZ)[np.newaxis, np.newaxis, :],
        R[:, 0][np.newaxis, :, np.newaxis],
        0,
    )

    return 2 * np.pi * ggreens
