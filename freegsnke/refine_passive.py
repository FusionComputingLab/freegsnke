"""
Defines some of the functionality needed by the FreeGSNKE passive_structure object.

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

from matplotlib.path import Path
from scipy.stats.qmc import LatinHypercube

# fixing the seed for reproducibility purposes
engine = LatinHypercube(d=2, seed=42)


import numpy as np


def generate_refinement(R, Z, n_refine, refine_mode):
    if refine_mode == "G":
        return generate_refinement_G(R, Z, n_refine)
    elif refine_mode == "LH":
        return generate_refinement_LH(R, Z, n_refine)
    else:
        print("refinement mode not recognised!, please use G or LH.")


def generate_refinement_LH(R, Z, n_refine):
    """Uses a latine hypercube to fill the shape defined by the input vertices R, Z
    with exactly n_refine points.

    Parameters
    ----------
    R : array
        R coordinates of the vertices
    Z : array
        Z coordinates of the vertices
    n_refine : int
        Number of refining points generated

    Returns
    -------
    array
        refining points

    """

    area, path, vmin, vmax, dv, meanR, meanZ = find_area(R, Z, n_refine)
    Len = np.linalg.norm(dv)

    rand_fil = np.zeros((0, 2))
    it = 0
    while len(rand_fil) < n_refine and it < 100:
        vals = engine.random(n=n_refine)
        vals = vmin + (vmax - vmin) * vals
        rand_fil = np.concatenate((rand_fil, vals[path.contains_points(vals)]), axis=0)
        it += 1

    return rand_fil[:n_refine], area


def generate_refinement_G(R, Z, n_refine):
    """Generates a regular square grid refinement, so to include approximately
    n_refine points in the shape with vertices R,Z

    Parameters
    ----------
    R : array
        R coordinates of the vertices
    Z : array
        Z coordinates of the vertices
    n_refine : int
        Number of desired refining points

    Returns
    -------
    array
        refining points
    """

    area, path, vmin, vmax, dv, meanR, meanZ = find_area(R, Z, n_refine)

    dl = (area / n_refine) ** 0.5
    nx = int(dv[0] // dl)
    ny = int(dv[1] // dl)

    grid_fil = []
    while len(grid_fil) < n_refine:
        if nx > 1:
            x = np.linspace(vmin[0] * 1.00001, vmax[0] * 0.99999, nx)
        else:
            x = np.mean(R)
        if ny > 1:
            y = np.linspace(vmin[1] * 1.00001, vmax[1] * 0.99999, ny)
        else:
            y = np.mean(R)

        xv, yv = np.meshgrid(x, y)

        grid_fil = np.concatenate((xv.reshape(-1, 1), yv.reshape(-1, 1)), axis=1)
        grid_fil = grid_fil[path.contains_points(grid_fil)]

        if nx < ny:
            nx += 1
        else:
            ny += 1

    return grid_fil, area


def find_area(R, Z, n_refine):
    """Finds area inside polygon and builds the path.

    Parameters
    ----------
    R : array
        R coordinates of the vertices
    Z : array
        Z coordinates of the vertices
    n_refine : int
        Number of desired refining points
    """
    if n_refine is None:
        n_refine = 100

    verts = np.concatenate(
        (
            np.array(R)[:, np.newaxis],
            np.array(Z)[:, np.newaxis],
        ),
        axis=-1,
    )
    path = Path(verts)
    vmin = np.min(verts, axis=0)
    vmax = np.max(verts, axis=0)
    dv = vmax - vmin
    area = dv[0] * dv[1]

    accepted = 0
    mult = 10
    while accepted < 10 * n_refine and mult < 1e6:
        mult *= 10
        vals = engine.random(n=int(mult * n_refine))
        vals = vmin + (vmax - vmin) * vals
        mask = path.contains_points(vals)
        accepted = np.sum(mask)
    area *= accepted / (mult * n_refine)

    meanR, meanZ = np.mean(vals[mask], axis=0)

    return area, path, vmin, vmax, dv, meanR, meanZ
