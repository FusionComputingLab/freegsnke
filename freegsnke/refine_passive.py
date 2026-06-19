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

from copy import deepcopy
from math import ceil

from matplotlib.path import Path
import numpy as np
from scipy.stats.qmc import LatinHypercube

# fixing the seed for reproducibility purposes
engine = LatinHypercube(d=2, seed=42)


def generate_refinement(R, Z, n_refine, refine_mode, return_weights=False):
    """
    Generate refinement points for a passive polygon.

    Parameters
    ----------
    R, Z : array-like
        Polygon vertex coordinates.
    n_refine : int
        Target number of refinement points.
    refine_mode : {"G", "LH", "GQ"}
        ``"G"`` uses a regular grid, ``"LH"`` uses Latin Hypercube sampling,
        and ``"GQ"`` uses weighted Gauss-Legendre quadrature for four-vertex
        polygons.
    return_weights : bool, optional
        If True, also return normalized area weights for the refinement points.

    Returns
    -------
    tuple
        ``(points, area)`` by default, or ``(points, area, weights)`` if
        ``return_weights=True``.
    """

    if refine_mode == "G":
        filaments, area = generate_refinement_G(R, Z, n_refine)
        weights = np.full(len(filaments), 1.0 / len(filaments))
    elif refine_mode == "LH":
        filaments, area = generate_refinement_LH(R, Z, n_refine)
        weights = np.full(len(filaments), 1.0 / len(filaments))
    elif refine_mode == "GQ":
        filaments, area, weights = generate_refinement_GQ(R, Z, n_refine)
    else:
        raise ValueError("refinement mode not recognised!, please use G, LH, or GQ.")

    if return_weights:
        return filaments, area, weights
    return filaments, area


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
            y = np.mean(Z)

        xv, yv = np.meshgrid(x, y)

        grid_fil = np.concatenate((xv.reshape(-1, 1), yv.reshape(-1, 1)), axis=1)
        grid_fil = grid_fil[path.contains_points(grid_fil)]

        if nx < ny:
            nx += 1
        else:
            ny += 1

    return grid_fil, area


def generate_refinement_GQ(R, Z, n_refine):
    """
    Generate weighted Gauss-Legendre quadrature points for a four-vertex polygon.

    The polygon is mapped from the unit square using bilinear interpolation of
    the boundary vertices. The returned weights are normalized to sum to one, so
    they can be used directly as current-distribution multipliers.

    Parameters
    ----------
    R, Z : array-like
        Four polygon vertices ordered around the boundary.
    n_refine : int
        Target number of quadrature points.

    Returns
    -------
    filaments : ndarray
        Quadrature point coordinates with shape ``(n_points, 2)``.
    area : float
        Weighted quadrature area of the polygon.
    weights : ndarray
        Normalized quadrature weights with shape ``(n_points,)``.
    """

    vertices = np.column_stack((np.asarray(R, dtype=float), np.asarray(Z, dtype=float)))
    if len(vertices) != 4:
        raise ValueError("GQ passive refinement currently requires four vertices.")

    length_u, length_v = _quadrilateral_direction_lengths(vertices)
    if length_u <= 0 or length_v <= 0:
        raise ValueError("Passive polygon edge lengths must be positive.")

    n_u = max(1, int(ceil(np.sqrt(n_refine * length_u / length_v))))
    n_v = max(1, int(ceil(n_refine / n_u)))

    nodes_u, weights_u = np.polynomial.legendre.leggauss(n_u)
    nodes_v, weights_v = np.polynomial.legendre.leggauss(n_v)
    nodes_u = 0.5 * (nodes_u + 1.0)
    nodes_v = 0.5 * (nodes_v + 1.0)
    weights_u = 0.5 * weights_u
    weights_v = 0.5 * weights_v

    filaments = []
    area_weights = []
    for u, wu in zip(nodes_u, weights_u):
        for v, wv in zip(nodes_v, weights_v):
            filaments.append(_bilinear_quad_point(vertices, u, v))
            area_weights.append(wu * wv * _bilinear_quad_jacobian(vertices, u, v))

    filaments = np.asarray(filaments)
    area_weights = np.asarray(area_weights)
    area = np.sum(area_weights)
    if area <= 0:
        raise ValueError("Passive polygon quadrature area must be positive.")

    return filaments, area, area_weights / area


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


def subdivide_passive_polygons(
    passive_coils,
    max_edge_length=None,
    max_area=None,
    name_separator="__",
    non_quadrilateral="raise",
):
    """
    Split four-vertex polygonal passive structures into smaller polygon passives.

    This is a topology refinement of the machine description: each child polygon
    becomes its own passive structure and therefore its own passive-current
    degree of freedom. Metadata used to map reconstructed currents onto passive
    structures is preserved on each child, and any ``current_multiplier`` is
    distributed by child area fraction.

    Parameters
    ----------
    passive_coils : list of dict
        Passive-structure machine description. Single-filament passives are
        copied through unchanged.
    max_edge_length : float, optional
        Maximum target child edge length in metres. The subdivision count along
        each local polygon direction is based on the average length of opposite
        parent edges.
    max_area : float, optional
        Maximum target child area in square metres. If needed, extra splits are
        added along the currently longer child direction.
    name_separator : str, optional
        Separator used between parent passive name and child index. Defaults to
        ``"__"``.
    non_quadrilateral : {"raise", "keep"}, optional
        Behaviour for polygon passives that do not have four vertices.
        ``"raise"`` keeps the default strict behaviour and raises a
        ``ValueError``. ``"keep"`` copies non-quadrilateral passives through
        unchanged, allowing mixed machines to refine quadrilateral passives
        while leaving general polygons to the standard passive refinement used
        by :class:`freegsnke.passive_structure.PassiveStructure`.

    Returns
    -------
    list of dict
        Refined passive-structure description.

    Raises
    ------
    ValueError
        If neither ``max_edge_length`` nor ``max_area`` is provided, or if a
        polygon with more or fewer than four vertices needs subdivision and
        ``non_quadrilateral="raise"``.
    """

    if max_edge_length is None and max_area is None:
        raise ValueError(
            "At least one of 'max_edge_length' or 'max_area' must be provided."
        )
    if max_edge_length is not None and max_edge_length <= 0:
        raise ValueError("'max_edge_length' must be positive.")
    if max_area is not None and max_area <= 0:
        raise ValueError("'max_area' must be positive.")
    if non_quadrilateral not in ("raise", "keep"):
        raise ValueError("'non_quadrilateral' must be either 'raise' or 'keep'.")

    refined_passives = []
    for parent_index, passive in enumerate(passive_coils):
        R = np.asarray(passive["R"], dtype=float)
        Z = np.asarray(passive["Z"], dtype=float)

        if np.size(R) <= 1:
            refined_passives.append(deepcopy(passive))
            continue
        if len(R) != 4:
            if non_quadrilateral == "keep":
                refined_passives.append(deepcopy(passive))
                continue
            raise ValueError(
                "Only four-vertex passive polygons can be subdivided. "
                f"Passive {passive.get('name', parent_index)!r} has {len(R)} vertices."
            )

        refined_passives.extend(
            subdivide_passive_polygon(
                passive,
                max_edge_length=max_edge_length,
                max_area=max_area,
                parent_index=parent_index,
                name_separator=name_separator,
            )
        )

    return refined_passives


def subdivide_passive_polygon(
    passive,
    max_edge_length=None,
    max_area=None,
    parent_index=None,
    name_separator="__",
):
    """
    Split one four-vertex passive polygon into smaller child polygons.

    Parameters are the same as :func:`subdivide_passive_polygons`, except that
    ``passive`` is a single passive-structure dictionary. The input vertex order
    is expected to be around the boundary, as produced by
    :func:`freegsnke.mastu_tools.get_element_vertices`.
    """

    R = np.asarray(passive["R"], dtype=float)
    Z = np.asarray(passive["Z"], dtype=float)
    vertices = np.column_stack((R, Z))
    if len(vertices) != 4:
        raise ValueError(
            "Only four-vertex passive polygons can be subdivided by this helper."
        )
    if max_edge_length is None and max_area is None:
        raise ValueError(
            "At least one of 'max_edge_length' or 'max_area' must be provided."
        )

    parent_area = polygon_area(R, Z)
    if parent_area <= 0:
        raise ValueError("Passive polygon area must be positive.")

    n_u, n_v = _subdivision_shape(
        vertices,
        parent_area,
        max_edge_length=max_edge_length,
        max_area=max_area,
    )

    parent_name = passive.get("name", "passive")
    child_specs = []
    for iu in range(n_u):
        u0 = iu / n_u
        u1 = (iu + 1) / n_u
        for iv in range(n_v):
            v0 = iv / n_v
            v1 = (iv + 1) / n_v
            child_vertices = np.array(
                [
                    _bilinear_quad_point(vertices, u0, v0),
                    _bilinear_quad_point(vertices, u0, v1),
                    _bilinear_quad_point(vertices, u1, v1),
                    _bilinear_quad_point(vertices, u1, v0),
                ]
            )
            child_area = polygon_area(child_vertices[:, 0], child_vertices[:, 1])
            child_specs.append((iu, iv, child_vertices, child_area))

    area_fractions = np.asarray(
        [child_area / parent_area for _, _, _, child_area in child_specs]
    )
    if len(area_fractions) > 1:
        area_fractions[-1] = 1.0 - np.sum(area_fractions[:-1])

    children = []
    child_multipliers = []
    for child_index, ((iu, iv, child_vertices, _), area_fraction) in enumerate(
        zip(child_specs, area_fractions)
    ):
        child = deepcopy(passive)
        child["R"] = child_vertices[:, 0].tolist()
        child["Z"] = child_vertices[:, 1].tolist()
        child["name"] = f"{parent_name}{name_separator}{child_index}"
        child["parent_name"] = parent_name
        child["parent_index"] = parent_index
        child["subdivision_index"] = child_index
        child["subdivision_ij"] = (iu, iv)
        child["subdivision_shape"] = (n_u, n_v)
        child["parent_area"] = parent_area
        child["area_fraction"] = area_fraction
        if "current_multiplier" in child:
            if child_index == len(child_specs) - 1:
                child["current_multiplier"] = passive["current_multiplier"] - sum(
                    child_multipliers
                )
            else:
                child["current_multiplier"] = (
                    passive["current_multiplier"] * area_fraction
                )
            child_multipliers.append(child["current_multiplier"])

        children.append(child)

    return children


def polygon_area(R, Z):
    """
    Calculate the absolute area of a polygon from its vertices.

    Parameters
    ----------
    R, Z : array-like
        Vertex coordinates ordered around the polygon boundary.

    Returns
    -------
    float
        Polygon area in square metres.
    """

    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    return 0.5 * abs(np.dot(R, np.roll(Z, -1)) - np.dot(Z, np.roll(R, -1)))


def _subdivision_shape(vertices, area, max_edge_length=None, max_area=None):
    """Choose child counts along the two local directions of a quadrilateral."""

    length_u, length_v = _quadrilateral_direction_lengths(vertices)
    if max_edge_length is None:
        n_u = 1
        n_v = 1
    else:
        n_u = max(1, int(ceil(length_u / max_edge_length)))
        n_v = max(1, int(ceil(length_v / max_edge_length)))

    if max_area is not None:
        while area / (n_u * n_v) > max_area:
            child_u = length_u / n_u
            child_v = length_v / n_v
            if child_u >= child_v:
                n_u += 1
            else:
                n_v += 1

    return n_u, n_v


def _quadrilateral_direction_lengths(vertices):
    """Return average opposite-edge lengths for a four-vertex polygon."""

    edge_lengths = np.linalg.norm(
        np.roll(vertices, -1, axis=0) - vertices,
        axis=1,
    )
    length_u = 0.5 * (edge_lengths[1] + edge_lengths[3])
    length_v = 0.5 * (edge_lengths[0] + edge_lengths[2])
    return length_u, length_v


def _bilinear_quad_point(vertices, u, v):
    """Interpolate a point inside a quadrilateral using boundary vertex order."""

    return (
        (1 - u) * (1 - v) * vertices[0]
        + (1 - u) * v * vertices[1]
        + u * v * vertices[2]
        + u * (1 - v) * vertices[3]
    )


def _bilinear_quad_jacobian(vertices, u, v):
    """Return the absolute Jacobian determinant of the bilinear quad mapping."""

    dx_du = (
        -(1 - v) * vertices[0]
        - v * vertices[1]
        + v * vertices[2]
        + (1 - v) * vertices[3]
    )
    dx_dv = (
        -(1 - u) * vertices[0]
        + (1 - u) * vertices[1]
        + u * vertices[2]
        - u * vertices[3]
    )
    return abs(dx_du[0] * dx_dv[1] - dx_du[1] * dx_dv[0])
