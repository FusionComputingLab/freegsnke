"""Machine-independent strike-point measurements and ordered wall coordinates.

``Ss`` is distance in metres along the supplied wall polyline, starting at
its first vertex and increasing in vertex order. ``Rs`` is the major radius
of that same contact, obtained by evaluating the wall at ``Ss``. Neither
coordinate depends on a requested strike target.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import shapely as sh


def cumulative_arc(points: np.ndarray) -> np.ndarray:
    """Cumulative piecewise-linear distance along an ordered contour."""

    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("wall must contain at least two finite (R, Z) points.")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0):
        raise ValueError("Consecutive wall vertices must be distinct.")
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def arc_coordinates(
    points: np.ndarray,
    wall_points: np.ndarray,
    wall_arc: np.ndarray,
) -> np.ndarray:
    """Project points onto the nearest segment of an ordered wall contour."""

    points = np.atleast_2d(np.asarray(points, dtype=float))
    starts = wall_points[:-1]
    vectors = np.diff(wall_points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    coordinates = []
    for point in points:
        fractions = np.sum((point - starts) * vectors, axis=1) / lengths**2
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = starts + fractions[:, None] * vectors
        segment = np.argmin(np.linalg.norm(projections - point, axis=1))
        coordinates.append(wall_arc[segment] + fractions[segment] * lengths[segment])
    return np.asarray(coordinates)


def point_from_arc(
    coordinate: float,
    wall_points: np.ndarray,
    wall_arc: np.ndarray,
) -> np.ndarray:
    """Interpolate one point on an ordered wall contour."""

    coordinate = float(np.clip(coordinate, wall_arc[0], wall_arc[-1]))
    return np.array(
        [np.interp(coordinate, wall_arc, wall_points[:, axis]) for axis in range(2)]
    )


def _boundary_contour_lines(R, Z, psi, psi_boundary):
    """Return piecewise-linear contours at the plasma-boundary flux."""

    figure, axes = plt.subplots()
    try:
        contour = axes.contour(R, Z, psi, levels=[psi_boundary])
        lines = []
        for line in contour.allsegs[0]:
            line = np.asarray(line, dtype=float)
            if len(line) < 2:
                continue
            # Marching squares can repeat vertices at exact grid crossings.
            # Zero-length edges trigger invalid projections in GEOS.
            keep = np.r_[True, np.any(np.diff(line, axis=0) != 0, axis=1)]
            line = line[keep]
            if len(line) >= 2:
                lines.append(line)
        return lines
    finally:
        plt.close(figure)


def _point_intersections(geometry):
    """Return point components from a Shapely intersection geometry."""

    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type in ("MultiPoint", "GeometryCollection"):
        return [
            item
            for item in geometry.geoms
            if item.geom_type == "Point" and not item.is_empty
        ]
    return []


def find_strikepoints(R, Z, psi, psi_boundary, wall):
    """
    Find the strikepoints of an equilibrium with the wall.

    Parameters
    ----------
    R : np.ndarray (nx, nz)
        2D array of major radius coordinates [m]
    Z : np.ndarray (nx, nz)
        2D array of vertical coordinates [m]
    psi : np.ndarray (nx, nz)
        2D poloidal flux map [Wb]
    psi_boundary : float
        Value of psi at the plasma boundary [Wb]
    wall : np.ndarray (N, 2)
        Wall/limiter coordinates as (R, Z) pairs [m]

    Returns
    -------
    np.ndarray or None
        Array of strikepoint coordinates, shape (M, 2), or None if no
        intersections are found.
    """

    psi_boundary_lines = _boundary_contour_lines(R, Z, psi, psi_boundary)

    # use the shapely package to find where each psi_boundary_line intersects the limiter (or not)
    strikes = []
    curve1 = sh.LineString(wall)
    for j, line in enumerate(psi_boundary_lines):
        curve2 = sh.LineString(line)

        # find the intersection points
        intersection = curve2.intersection(curve1)

        points = _point_intersections(intersection)
        if points:
            strikes.append(
                np.asarray([point.coords[0] for point in points], dtype=float)
            )

    # check how many strikepoints
    if len(strikes) == 0:
        out = None
    else:
        out = np.concatenate(strikes, axis=0)

    return out


def find_leg_strikepoint(
    R,
    Z,
    psi,
    psi_boundary,
    wall,
    xpoint,
    leg="outer",
    xpoint_tolerance=None,
    probe_distance=None,
    magnetic_axis=None,
):
    """Return the first wall contact along a separatrix leg from an X-point.

    Unlike :func:`find_strikepoints`, this routine gives the intersections a
    physical ordering. It projects the supplied X-point onto each boundary
    contour that reaches the wall, identifies the radially inner or outer
    direction immediately away from the X-point, and returns the first wall
    intersection encountered in that direction.

    Parameters
    ----------
    R, Z, psi, psi_boundary, wall
        Equilibrium grid, boundary flux and ordered wall coordinates, as for
        :func:`find_strikepoints`.
    xpoint : array-like, shape (2,)
        The X-point from which the divertor legs are to be followed.
    leg : {"inner", "outer"}
        Select the direction that initially moves to smaller or larger major
        radius, respectively.
    xpoint_tolerance : float, optional
        Maximum distance between the X-point and its discretized contour.
        The default is twice the grid-cell diagonal.
    probe_distance : float, optional
        Contour distance used to classify the local leg direction. The
        default is one grid-cell diagonal.

    magnetic_axis : array-like, shape (2,), optional
        Magnetic axis. When supplied, exclude directions initially heading
        toward the core before classifying inner/outer divertor legs.

    Returns
    -------
    np.ndarray or None
        The selected ``(R, Z)`` wall contact, or ``None`` if no contour that
        passes near the X-point also reaches the wall.
    """

    if leg not in ("inner", "outer"):
        raise ValueError("leg must be either 'inner' or 'outer'.")

    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    wall = np.asarray(wall, dtype=float)
    xpoint = np.asarray(xpoint, dtype=float).reshape(2)

    radial_grid = np.unique(R.ravel())
    vertical_grid = np.unique(Z.ravel())
    radial_step = np.median(np.diff(radial_grid))
    vertical_step = np.median(np.diff(vertical_grid))
    cell_diagonal = float(np.hypot(radial_step, vertical_step))
    if xpoint_tolerance is None:
        xpoint_tolerance = 2.0 * cell_diagonal
    if probe_distance is None:
        probe_distance = cell_diagonal

    wall_curve = sh.LineString(wall)
    xpoint_geometry = sh.Point(xpoint)
    branch_candidates = []

    for vertices in _boundary_contour_lines(R, Z, psi, psi_boundary):
        contour = sh.LineString(vertices)
        if contour.length == 0.0:
            continue
        distance_to_xpoint = contour.distance(xpoint_geometry)
        if distance_to_xpoint > xpoint_tolerance:
            continue

        intersections = _point_intersections(contour.intersection(wall_curve))
        if not intersections:
            continue

        xpoint_coordinate = contour.project(xpoint_geometry)
        for direction in (-1.0, 1.0):
            available = (
                xpoint_coordinate
                if direction < 0.0
                else contour.length - xpoint_coordinate
            )
            if available <= 0.0:
                continue

            directed_intersections = []
            for intersection in intersections:
                coordinate = contour.project(intersection)
                path_distance = direction * (coordinate - xpoint_coordinate)
                if path_distance > 1e-10:
                    directed_intersections.append((path_distance, intersection))
            if not directed_intersections:
                continue

            local_distance = min(float(probe_distance), 0.5 * available)
            probe_coordinate = xpoint_coordinate + direction * local_distance
            probe = np.asarray(contour.interpolate(probe_coordinate).coords[0])
            first_distance, first_intersection = min(
                directed_intersections, key=lambda item: item[0]
            )
            if magnetic_axis is not None:
                away_from_axis = xpoint - np.asarray(magnetic_axis, dtype=float)
                if np.dot(probe - xpoint, away_from_axis) <= 0.0:
                    continue
            branch_candidates.append(
                {
                    "radial_displacement": probe[0] - xpoint[0],
                    "distance_to_xpoint": distance_to_xpoint,
                    "path_distance": first_distance,
                    "point": np.asarray(first_intersection.coords[0], dtype=float),
                }
            )

    if not branch_candidates:
        return None

    radial_displacements = np.asarray(
        [item["radial_displacement"] for item in branch_candidates]
    )
    directions_are_resolved = np.any(radial_displacements < 0.0) and np.any(
        radial_displacements > 0.0
    )
    if directions_are_resolved and leg == "outer":
        chosen = max(
            branch_candidates,
            key=lambda item: (
                item["radial_displacement"],
                -item["distance_to_xpoint"],
            ),
        )
    elif directions_are_resolved:
        chosen = min(
            branch_candidates,
            key=lambda item: (
                item["radial_displacement"],
                item["distance_to_xpoint"],
            ),
        )
    elif leg == "outer":
        # A discretized saddle need not pass through the supplied X-point.
        # Its projection can then put both contour directions on the same
        # radial side, making the one-cell probes unable to label the legs.
        # The first wall contacts still retain the physical inboard/outboard
        # ordering, so use their major radii only for this ambiguous case.
        chosen = max(
            branch_candidates,
            key=lambda item: (
                item["point"][0],
                -item["path_distance"],
                -item["distance_to_xpoint"],
            ),
        )
    else:
        chosen = min(
            branch_candidates,
            key=lambda item: (
                item["point"][0],
                item["path_distance"],
                item["distance_to_xpoint"],
            ),
        )
    return chosen["point"]


@dataclass(frozen=True)
class StrikePoint:
    """One wall contact, in metres, shared by the Ss and Rs descriptors."""

    Ss: float
    Rs: float
    Zs: float

    @property
    def point(self):
        """Return the contact in R-Z coordinates."""
        return np.array([self.Rs, self.Zs])


def measure_strikepoint(
    equilibrium,
    *,
    wall=None,
    xpoint=None,
    leg="outer",
    xpoint_tolerance=None,
    probe_distance=None,
):
    """Measure the first wall contact along one leg of an X-point.

    Parameters
    ----------
    equilibrium : object
        Solved equilibrium with R, Z, psi(), psi_bndry and xpt.
    wall : array-like, optional
        Ordered (N, 2) wall polyline. Default: equilibrium.tokamak.wall.
        Its first vertex defines Ss=0; its vertex order defines positive Ss.
        Repeat the first vertex at the end when a closed contour is intended.
        Supplying a target section or half-wall gives a local arc convention.
    xpoint : array-like, optional
        Selected (R, Z) X-point; default: equilibrium.xpt[0, :2]. In a
        multiple-null equilibrium, supply the desired X-point explicitly.
        The equilibrium's primary magnetic axis, when present, excludes
        contour directions toward the core before selecting a divertor leg.
    leg : {"inner", "outer"}
        Direction initially moving to smaller or larger major radius.
    xpoint_tolerance, probe_distance : float, optional
        Passed to find_leg_strikepoint; defaults scale with the grid spacing.

    Returns
    -------
    StrikePoint
        Ss and Rs from one contact. A missing contact raises RuntimeError;
        no nearest-target substitute is returned.
    """
    if wall is None:
        machine_wall = equilibrium.tokamak.wall
        wall = np.column_stack((machine_wall.R, machine_wall.Z))
    wall = np.asarray(wall, dtype=float)
    wall_arc = cumulative_arc(wall)
    if xpoint is None:
        xpoints = np.asarray(equilibrium.xpt, dtype=float)
        if xpoints.size == 0:
            raise RuntimeError("A strike-point measurement requires an X-point.")
        xpoint = xpoints[0, :2]
    xpoint = np.asarray(xpoint, dtype=float)
    if xpoint.shape != (2,) or not np.all(np.isfinite(xpoint)):
        raise ValueError("xpoint must be a finite (R, Z) pair.")
    contact = find_leg_strikepoint(
        equilibrium.R,
        equilibrium.Z,
        equilibrium.psi(),
        equilibrium.psi_bndry,
        wall,
        xpoint,
        leg=leg,
        xpoint_tolerance=xpoint_tolerance,
        probe_distance=probe_distance,
        magnetic_axis=(
            np.asarray(equilibrium.opt)[0, :2]
            if np.asarray(getattr(equilibrium, "opt", [])).size
            else None
        ),
    )
    if contact is None:
        raise RuntimeError(f"The {leg} separatrix leg does not intersect the wall.")
    ss = float(arc_coordinates(contact, wall, wall_arc)[0])
    rs, zs = point_from_arc(ss, wall, wall_arc)
    return StrikePoint(Ss=ss, Rs=float(rs), Zs=float(zs))


@dataclass
class StrikePointDescriptor:
    """Callable Ss/Rs measurement for ALPD or virtual-circuit construction.

    ``names`` selects the output order. Request both coordinates with
    ("Ss", "Rs"); the contour is traced only once per call. Generally use
    one of them as an equality target, since both describe the same contact.
    ``xpoint`` may be a fixed pair or a callable selecting a point from eq.
    """

    wall: object = None
    xpoint: object = None
    leg: str = "outer"
    names: tuple = ("Ss", "Rs")

    def __post_init__(self):
        """Validate requested descriptor names."""
        if any(name not in ("Ss", "Rs") for name in self.names):
            raise ValueError("Strike-point descriptor names must be Ss or Rs.")

    def __call__(self, equilibrium):
        """Return the requested coordinates from one native measurement."""
        xpoint = self.xpoint(equilibrium) if callable(self.xpoint) else self.xpoint
        contact = measure_strikepoint(
            equilibrium, wall=self.wall, xpoint=xpoint, leg=self.leg
        )
        return np.array([getattr(contact, name) for name in self.names])
