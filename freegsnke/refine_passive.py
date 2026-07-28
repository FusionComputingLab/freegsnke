"""
Defines functionality needed for building passive conductive structures.

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
from matplotlib.path import Path
from scipy.stats.qmc import LatinHypercube

# Latin hypercube sampling engine (fixed seed for reproducibility)
engine = LatinHypercube(d=2, seed=42)


def generate_refinement(R, Z, n_refine, refine_mode):
    """
    Generate a refined set of points in (R, Z) space using a selected strategy.

    The refinement strategy is selected via `refine_mode` and delegates to
    the corresponding refinement routine.

    Parameters
    ----------
    R : ndarray
        Radial coordinates of the input grid or points.
    Z : ndarray
        Vertical coordinates of the input grid or points.
    n_refine : int
        Number of refined points (or refinement resolution parameter,
        depending on the selected method).
    refine_mode : str
        Refinement strategy selector:
        - "G"  : Gaussian-based refinement
        - "LH" : Latin Hypercube refinement

    Returns
    -------
    ndarray
        Refined (R, Z) point set produced by the selected method.
    """

    if refine_mode == "G":
        return generate_refinement_G(R, Z, n_refine)
    elif refine_mode == "LH":
        return generate_refinement_LH(R, Z, n_refine)
    else:
        print("refine_mode not recognised!, please use G or LH.")


def generate_refinement_LH(R, Z, n_refine):
    """
    Generate refinement points inside a polygon using Latin Hypercube sampling.

    This method samples candidate points using a Latin Hypercube strategy,
    maps them into the bounding box of the polygon defined by (R, Z), and
    retains only points lying inside the polygon.

    Sampling is repeated until at least `n_refine` valid interior points
    are obtained (or a maximum iteration limit is reached).

    Parameters
    ----------
    R : array
        R coordinates of the polygon vertices.
    Z : array
        Z coordinates of the polygon vertices.
    n_refine : int
        Number of interior refinement points to generate.

    Returns
    -------
    points : ndarray, shape (n_refine, 2)
        Generated refinement points inside the polygon.
    area : float
        Estimated area of the polygon region returned by `find_area`.
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
    """
    Generate a structured grid refinement inside a polygon.

    This method constructs a regular Cartesian grid over the bounding box
    of the polygon defined by (R, Z), and retains only the points that lie
    inside the polygon. The grid resolution is increased iteratively until
    at least `n_refine` interior points are obtained.

    Parameters
    ----------
    R : array
        R coordinates of the polygon vertices.
    Z : array
        Z coordinates of the polygon vertices.
    n_refine : int
        Target number of refinement points.

    Returns
    -------
    points : ndarray, shape (N, 2)
        Refinement points inside the polygon (N ≥ n_refine).
    area : float
        Estimated area of the polygon region returned by `find_area`.
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
    """
    Estimate polygon area and construct a point-in-polygon test path.

    This function computes a bounding-box-based Monte Carlo estimate of the
    area enclosed by a polygon defined by vertices (R, Z). It also constructs
    a `matplotlib.path.Path` object for point-in-polygon queries and estimates
    a centroid-like mean of accepted sample points.

    Parameters
    ----------
    R : array
        R coordinates of the polygon vertices.
    Z : array
        Z coordinates of the polygon vertices.
    n_refine : int
        Target number of refinement points used to control Monte Carlo sampling.

    Returns
    -------
    area : float
        Estimated area of the polygon.
    path : matplotlib.path.Path
        Path object for point-in-polygon evaluation.
    vmin : ndarray
        Minimum bounds of the vertex coordinates (R_min, Z_min).
    vmax : ndarray
        Maximum bounds of the vertex coordinates (R_max, Z_max).
    dv : ndarray
        Bounding box size (vmax - vmin).
    meanR : float
        Mean R coordinate of accepted Monte Carlo samples.
    meanZ : float
        Mean Z coordinate of accepted Monte Carlo samples.
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
