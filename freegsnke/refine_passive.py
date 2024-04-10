from matplotlib.path import Path
from scipy.stats.qmc import LatinHypercube

engine = LatinHypercube(d=2)

import numpy as np


def generate_refinement(R, Z, n_refine, mode):
    if mode=='G':
        return generate_refinement_G(R, Z, n_refine)
    elif mode=='LH':
        return generate_refinement_LH(R, Z, n_refine)
    else:
        print('refinement mode not recognised!, please use G or LH.')

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

    # verts = np.concatenate(
    #     (
    #         np.array(R)[:, np.newaxis],
    #         np.array(Z)[:, np.newaxis],
    #     ),
    #     axis=-1,
    # )
    # path = Path(verts)
    # vmin = np.min(verts, axis=0)
    # vmax = np.max(verts, axis=0)
    area, path, vmin, vmax, dv = find_area(R, Z, n_refine)

    rand_fil = np.zeros((0,2))
    it = 0
    while len(rand_fil)<n_refine and it<100:
        vals = engine.random(n=n_refine)
        vals = vmin + (vmax - vmin)*vals
        rand_fil = np.concatenate((rand_fil, vals[path.contains_points(vals)]), axis=0)
        it+=1

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

    # verts = np.concatenate(
    #     (
    #         np.array(R)[:, np.newaxis],
    #         np.array(Z)[:, np.newaxis],
    #     ),
    #     axis=-1,
    # )
    # path = Path(verts)
    # vmin = np.min(verts, axis=0)
    # vmax = np.max(verts, axis=0)
    # dv = vmax - vmin
    # area = dv[0]*dv[1]

    # accepted = 0
    # mult = 10
    # while accepted<10*n_refine:
    #     mult *= 10
    #     vals = engine.random(n=mult*n_refine)
    #     vals = vmin + (vmax-vmin)*vals
    #     accepted = np.sum(path.contains_points(vals))
    # area *= accepted/(mult*n_refine)

    area, path, vmin, vmax, dv = find_area(R, Z, n_refine)
    dl = (area/n_refine)**.5
    nx = dv[0]//dl
    ny = dv[1]//dl

    x = np.linspace(vmin[0], vmax[0], int(nx))
    y = np.linspace(vmin[1], vmax[1], int(ny))
    xv, yv = np.meshgrid(x, y)

    grid_fil = np.concatenate((xv.reshape(-1,1),
                               yv.reshape(-1,1)), axis=1)
    grid_fil = grid_fil[path.contains_points(grid_fil)]

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
    area = dv[0]*dv[1]

    accepted = 0
    mult = 10
    while accepted<10*n_refine and mult<1e6:
        mult *= 10
        vals = engine.random(n=mult*n_refine)
        vals = vmin + (vmax-vmin)*vals
        accepted = np.sum(path.contains_points(vals))
    area *= accepted/(mult*n_refine)

    return area, path, vmin, vmax, dv

    


    



