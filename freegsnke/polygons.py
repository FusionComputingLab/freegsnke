"""
Routines for calculating intersection or other geometric calculations
with polygons (e.g. walls, flux surface approximations)

License
-------

Copyright 2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

from __future__ import annotations


def intersect(
    r1: list[float],
    z1: list[float],
    r2: list[float],
    z2: list[float],
    closed1: bool = True,
    closed2: bool = True,
) -> bool:
    """
    Test if two polylines (or polygons) intersect.

    The polylines consist of the (r1, z1) and (r2, z2) line segments.

    Parameters
    ----------
    r1, z1 : list[float]
        R and Z coordinates of the vertices of the first polyline.
    r2, z2 : list[float]
        R and Z coordinates of the vertices of the second polyline.
    closed1 : bool, optional
        If True, the last vertex of the first polyline joins the first.
    closed2 : bool, optional
        If True, the last vertex of the second polyline joins the first.

    Returns
    -------
    bool
        True if any pair of segments intersect, False otherwise.
    """

    assert len(r1) == len(z1)
    assert len(r2) == len(z2)

    n1 = len(r1) if closed1 else len(r1) - 1
    n2 = len(r2) if closed2 else len(r2) - 1

    for i in range(n1):
        for j in range(n2):
            # Test for intersection between two line segments:
            # (r1[i],z1[i]) -- (r1[i+1],z1[i+1])
            # (r1[j],z1[j]) -- (r1[j+1],z1[j+1])
            # Note that since polynomials are closed the indices wrap around
            ip = (i + 1) % n1
            jp = (j + 1) % n2

            a = r1[ip] - r1[i]
            b = r2[jp] - r2[j]
            c = z1[ip] - z1[i]
            d = z2[jp] - z2[j]

            dr = r2[jp] - r1[i]
            dz = z2[jp] - z1[i]

            det = a * d - b * c

            if abs(det) < 1e-6:
                continue  # Almost certainly doesn't intersect

            alpha = (d * dr - b * dz) / det  # Location along line 1 [0,1]
            beta = (a * dz - c * dr) / det  # Location along line 2 [0,1]

            if (alpha > 0.0) & (alpha < 1.0) & (beta > 0.0) & (beta < 1.0):
                return True
    # Tested all combinations, none intersect
    return False


def area(polygon: list[tuple[float, float]]) -> float:
    """
    Calculate the signed area of a polygon.

    Positive for a clockwise polygon, negative for an anticlockwise one.

    Parameters
    ----------
    polygon : list[tuple[float, float]]
        Vertices ``[(r1, z1), (r2, z2), ...]``.

    Returns
    -------
    float
        Signed area of the polygon.
    """
    nvert = len(polygon)  # Number of vertices

    # Integrate using trapezium rule. The sign of (r2-r1) ensures that
    # positive and negative areas leave only the area of the polygon.
    area = 0.0
    for i in range(nvert):
        r1, z1 = polygon[i]
        r2, z2 = polygon[(i + 1) % nvert]  # Next vertex in periodic list
        area += (r2 - r1) * (z1 + z2)  # 2*area
    return 0.5 * area


def clockwise(polygon: list[tuple[float, float]]) -> bool:
    """
    Detect whether a polygon is clockwise or anti-clockwise.

    Parameters
    ----------
    polygon : list[tuple[float, float]]
        Vertices ``[(r1, z1), (r2, z2), ...]``.

    Returns
    -------
    bool
        True if the polygon is clockwise, False if anticlockwise.
    """
    # Work out the winding direction by calculating the area
    return area(polygon) > 0


def triangulate(
    polygon: list[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """
    Use the ear clipping method to turn an arbitrary polygon into triangles.

    Parameters
    ----------
    polygon : list[tuple[float, float]]
        Vertices ``[(r1, z1), (r2, z2), ...]`` (at least three).

    Returns
    -------
    list[list[tuple[float, float]]]
        Triangles, each a list of three ``(r, z)`` vertices.
    """

    if clockwise(polygon):
        # Copy input into list
        polygon = list(iter(polygon))
    else:
        polygon = list(reversed(polygon))
    # Now polygon should be clockwise

    nvert = len(polygon)  # Number of vertices
    assert nvert > 2

    triangles = []
    while nvert > 3:
        # Find an "ear"
        for i in range(nvert):
            vert = polygon[i]
            next_vert = polygon[(i + 1) % nvert]
            prev_vert = polygon[i - 1]

            # Take cross-product of edge from prev->vert and vert->next
            # to check whether the angle is > 180 degrees
            cross = (vert[1] - prev_vert[1]) * (next_vert[0] - vert[0]) - (
                vert[0] - prev_vert[0]
            ) * (next_vert[1] - vert[1])
            if cross < 0:
                continue  # Skip this vertex

            # Check these edges don't intersect with other edges
            r1 = [prev_vert[0], vert[0], next_vert[0]]
            z1 = [prev_vert[1], vert[1], next_vert[1]]

            r2 = []
            z2 = []
            if i < nvert - 1:
                r2 += [v[0] for v in polygon[(i + 1) :]]
                z2 += [v[1] for v in polygon[(i + 1) :]]
            if i > 0:
                r2 += [v[0] for v in polygon[:i]]
                z2 += [v[1] for v in polygon[:i]]

            # (r1,z1) is the line along two edges of the triangle
            # (r2,z2) is the rest of the polygon
            if intersect(r1, z1, r2, z2, closed1=False, closed2=False):
                continue  # Skip

            # Found an ear
            triangles.append([prev_vert, vert, next_vert])
            # Remove this vertex
            del polygon[i]
            nvert -= 1
            break

    # Reduced to a single triangle
    triangles.append(polygon)
    return triangles
