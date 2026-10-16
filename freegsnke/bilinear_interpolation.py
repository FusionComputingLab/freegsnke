"""Bilinear interpolation on uniformly spaced two-dimensional grids.

Copyright 2024-2026 Nicola C. Amorisco
Copyright 2025 Timothy Nunn

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

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        return lambda f: f


@njit(cache=True, fastmath=True)
def biliint(R, Z, psi, points):
    """Simple bilinear interpolation of 2d map

    Parameters
    ----------
    R : np.array
        R coordinates on 2d grid
    Z : np.array
        Z coordinates on 2d grid
    psi : np.array
        function values on 2d grid
    points : np.array
        coordinates where the interpolation is sought
        shape (2, whatever)

    Returns
    -------
    np.array
        interpolated values, same shape as points: (1, whatever)
    """

    nx, ny = np.shape(psi)

    Rmin = R[0, 0]
    Zmin = Z[0, 0]
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    points_shape = np.shape(points)
    points = points.reshape(2, -1)
    len_points = np.shape(points)[1]
    vals = np.empty(len_points)

    for point_idx in range(len_points):
        idx_R = int(np.floor((points[0, point_idx] - Rmin) / dR))
        idx_Z = int(np.floor((points[1, point_idx] - Zmin) / dZ))
        idx_R = min(max(idx_R, 0), nx - 2)
        idx_Z = min(max(idx_Z, 0), ny - 2)

        weight_R = (points[0, point_idx] - R[idx_R, 0]) / dR
        weight_Z = (points[1, point_idx] - Z[0, idx_Z]) / dZ
        vals[point_idx] = (
            (1.0 - weight_R) * (1.0 - weight_Z) * psi[idx_R, idx_Z]
            + (1.0 - weight_R) * weight_Z * psi[idx_R, idx_Z + 1]
            + weight_R * (1.0 - weight_Z) * psi[idx_R + 1, idx_Z]
            + weight_R * weight_Z * psi[idx_R + 1, idx_Z + 1]
        )

    return vals.reshape(points_shape[1:])
