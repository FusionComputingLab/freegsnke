"""
Plot aspects of the tokamak and equilibrium. 

Copyright 2024 Nicola C. Amorisco, Adriano Agnello, George K. Holt, Ben Dudson.

FreeGS4E is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS4E is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS4E.  If not, see <http://www.gnu.org/licenses/>.

"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import amax, amin, linspace

from . import critical


def plotCoils(coils, axis=None):
    """
    Plot the geometry of magnetic field coils.

    This function prepares a plot of the coil set used in a tokamak or other
    magnetic confinement configuration. It creates a Matplotlib axis if one
    is not provided, allowing additional geometric or field features to be
    overlaid later.

    Parameters
    ----------
    coils : object
        Coil geometry object containing spatial information about the magnetic
        coils (e.g., current centerlines or cross-sections). The exact structure
        of `coils` depends on the specific equilibrium or control framework.
    axis : matplotlib.axes.Axes, optional
        Axis object on which to plot. If `None`, a new figure and axis are created.

    Returns
    -------
    axis : matplotlib.axes.Axes
        The matplotlib axis prepared for coil plotting.

    Notes
    -----
    - This function currently initializes the plot but does not draw coil shapes.
      It can be extended to plot coil positions or outlines using data from
      `coils`.
    """

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    return axis


def plotConstraints(control, axis=None, show=True):
    """
    Plot magnetic control constraints such as X-points and isoflux surfaces.

    This function visualizes the spatial constraints used for coil current
    optimization or magnetic equilibrium control, including X-point locations
    and isoflux surface definitions. These constraints are typically used in
    inverse equilibrium solvers or feedback control algorithms.

    Parameters
    ----------
    control : object
        Control data object containing constraint information with attributes:
        - `xpoints` : list of (R, Z) tuples
            Target locations for X-point constraints.
        - `isoflux` : list of (R1, Z1, R2, Z2) tuples
            Pairs of points defining line segments that should lie on the same
            poloidal flux surface.
    axis : matplotlib.axes.Axes, optional
        Axis object on which to plot. If `None`, a new figure and axis are created.
    show : bool, default=True
        If True, call `matplotlib.pyplot.show()` before returning.

    Returns
    -------
    axis : matplotlib.axes.Axes
        The matplotlib axis containing the plotted constraints.

    Notes
    -----
    - X-point constraints are plotted as blue crosses (`"bx"`).
    - Isoflux constraints are plotted as dashed blue line segments with triangle
      markers (`":b^"`).
    - A legend is displayed automatically if `show=True`.

    Examples
    --------
        axis = plotConstraints(control)
        axis.set_title("Coil Control Constraints")
    """

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    # Locations of the X-points
    for r, z in control.xpoints:
        axis.plot(r, z, "bx")

    if control.xpoints:
        axis.plot([], [], "bx", label="X-point constraints")

    # Isoflux surfaces
    for r1, z1, r2, z2 in control.isoflux:
        axis.plot([r1, r2], [z1, z2], ":b^")

    if control.isoflux:
        axis.plot([], [], ":b^", label="Isoflux constraints")

    if show:
        plt.legend()
        plt.show()

    return axis


def plotIOConstraints(control, axis=None, show=True):
    """
    Plot magnetic control constraints for equilibrium reconstruction.

    This function visualizes the geometric constraints used in coil current
    control or equilibrium reconstruction, including magnetic null points and
    isoflux target locations. These constraints are typically used to ensure
    that a magnetic configuration satisfies specified plasma shape or field
    conditions.

    Parameters
    ----------
    control : object
        An object containing magnetic control data with attributes:
        - `null_points` : tuple of array-like
            Coordinates (R, Z) of magnetic null points, if defined.
        - `isoflux_set` : list of array-like
            A list of isoflux constraint points, where each element is a pair
            of arrays (R, Z) defining a set of target locations for equal
            poloidal flux.
    axis : matplotlib.axes.Axes, optional
        Axis object on which to plot. If `None`, a new figure and axis are created.
    show : bool, default=True
        If True, call `matplotlib.pyplot.show()` before returning.

    Returns
    -------
    axis : matplotlib.axes.Axes
        The matplotlib axis containing the plotted constraints.

    Notes
    -----
    - Null points are plotted as purple downward tick markers (`"1"`).
    - Isoflux constraints are plotted as plus markers (`"+"`) in distinct colors.
    - A legend is automatically displayed in the upper-right corner if `show=True`.

    Examples
    --------
        axis = plotIOConstraints(control)
        axis.set_title("Magnetic Control Constraints: Null Points and Isoflux Sets")
    """

    # axes
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    # locations of the null points
    if control.null_points is not None:
        axis.plot(
            control.null_points[0],
            control.null_points[1],
            "1",
            color="m",
            markersize=10,
            markeredgewidth=2.0,
        )
        axis.plot(
            [],
            [],
            "1",
            color="m",
            markersize=10,
            markeredgewidth=2.0,
            label="Null points",
        )

    # locations of the 2nd order null points
    if control.null_points_2nd_order is not None:
        axis.plot(
            control.null_points_2nd_order[0],
            control.null_points_2nd_order[1],
            "1",
            color="orange",
            markersize=10,
            markeredgewidth=2.0,
        )
        axis.plot(
            [],
            [],
            "1",
            color="orange",
            markersize=10,
            markeredgewidth=2.0,
            label="2nd order null points",
        )

    # plot isoflux constraints
    if control.isoflux_set is not None:
        color = [
            "tab:brown",
            "tab:orange",
            "tab:pink",
            "tab:gray",
            "tab:olive",
        ]
        for i, isoflux in enumerate(control.isoflux_set):
            axis.plot(
                isoflux[0],
                isoflux[1],
                "+",
                color=color[i % len(color)],
                markersize=10,
                markeredgewidth=2.0,
            )
            axis.plot(
                [],
                [],
                "+",
                color=color[i % len(color)],
                label=f"Isoflux set ({i})",
                markersize=10,
                markeredgewidth=2.0,
            )

    if show:
        plt.legend(loc="upper right")
        plt.show()

    return axis


def plotEquilibrium(
    eq,
    axis=None,
    xpoints=True,
    opoints=True,
    wall=True,
    limiter=True,
    legend=False,
    show=True,
):
    """
    Plot poloidal magnetic ψ(R, Z) flux surfaces and key geometric features of an equilibrium in
    FreeGSNKE. It can also overlay the separatrix, magnetic X- and
    O-points, the plasma-facing wall, and the limiter geometry.

    Parameters
    ----------
    eq : object
        An equilibrium object containing fields `R`, `Z`, and `psi()`, as well
        as profile data (`_profiles.opt`, `_profiles.xpt`) and tokamak geometry
        (`tokamak.wall`, `tokamak.limiter`).
    axis : matplotlib.axes.Axes, optional
        Axis object to plot on. If `None`, a new figure and axis are created.
    xpoints : bool, default=True
        If True, plot magnetic X-points (red 'x' markers).
    opoints : bool, default=True
        If True, plot magnetic O-points (green '*' markers).
    wall : bool, default=True
        If True, plot the tokamak wall outline as a solid black curve.
    limiter : bool, default=True
        If True, plot the limiter outline as a dashed black curve.
    legend : bool, default=False
        If True, display a legend describing plotted features.
    show : bool, default=True
        If True, call `matplotlib.pyplot.show()` before returning.

    Returns
    -------
    axis : matplotlib.axes.Axes
        The matplotlib axis containing the plotted equilibrium.

    Notes
    -----
    - The function assumes that the equilibrium has already been solved;
      if not, it will print a warning message.
    - The separatrix is plotted using a solid red line for diverted equilibria
      and a dashed red line for limited equilibria.

    Examples
    --------
        axis = plotEquilibrium(eq, legend=True)
        axis.set_title("Tokamak Equilibrium Flux Surfaces")
    """

    # extract data
    try:
        psi = eq.psi()
        opt = eq._profiles.opt
        xpt = eq._profiles.xpt
    except AttributeError as e:
        raise RuntimeError(
            "This equilibrium has not been solved: please solve for an equilibrium first!"
        ) from e

    # axes
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)
    axis.set_aspect("equal")
    axis.set_xlabel("Major radius [m]")
    axis.set_ylabel("Height [m]")

    # plot flux contours
    levels = linspace(amin(psi), amax(psi), 35)
    axis.contour(eq.R, eq.Z, psi, levels=levels)

    # plot separatrix (from primary X-point)
    colour = "r"
    style = "solid"
    axis.contour(eq.R, eq.Z, psi, levels=[xpt[0][2]], colors=colour, linestyles=style)
    axis.plot([], [], colour, label="Separatrix (primary X-point)", linestyle=style)

    # plot extra separatrix (LCFS) if plasma limited
    if eq._profiles.flag_limiter:
        colour = "k"
        style = "dashed"
        axis.contour(
            eq.R,
            eq.Z,
            psi,
            levels=[eq.psi_bndry],
            colors=colour,
            linestyles=style,
        )
        axis.plot(
            [],
            [],
            colour,
            label="LCFS (limited plasma)",
            linestyle=style,
        )

    # plot x point
    if xpoints:
        for r, z, _ in xpt:
            axis.plot(r, z, "rx", markersize=9)
        axis.plot(xpt[0][0], xpt[0][1], "rx", markersize=9, markeredgewidth=2.5)
        axis.plot([], [], "rx", markersize=9, label="X-points")
        axis.plot(
            [],
            [],
            "rx",
            markersize=9,
            markeredgewidth=2.5,
            label="X-point (primary)",
        )

    # plot op points
    if opoints:
        for r, z, _ in opt:
            axis.plot(r, z, "g2", markersize=9)
        axis.plot([], [], "g2", markersize=9, label="O-points")

    # plot wall
    if wall and eq.tokamak.wall and len(eq.tokamak.wall.R):
        axis.plot(
            list(eq.tokamak.wall.R) + [eq.tokamak.wall.R[0]],
            list(eq.tokamak.wall.Z) + [eq.tokamak.wall.Z[0]],
            "k",
        )

    # plot limiter
    if limiter and eq.tokamak.limiter and len(eq.tokamak.limiter.R):
        axis.plot(
            list(eq.tokamak.limiter.R) + [eq.tokamak.limiter.R[0]],
            list(eq.tokamak.limiter.Z) + [eq.tokamak.limiter.Z[0]],
            "k:",
        )

    if legend:
        axis.legend(loc="upper right")

    if show:
        plt.show()

    return axis


def make_broad_mask(mask, layer_size=1):
    """
    Expand a binary mask by adding a border layer of specified thickness.

    This function broadens a 2D boolean or binary mask by marking all points
    that lie within a specified Manhattan distance (`layer_size`) from the
    original masked region. The result is a new mask that includes both the
    original region and its expanded boundary.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean or binary array representing the region to be expanded.
    layer_size : int, optional
        Number of pixels by which to expand the mask in all directions.
        Default is 1.

    Returns
    -------
    layer_mask : np.ndarray
        Boolean array of the same shape as `mask`, where `True` values
        correspond to the original region plus an outer layer of width
        `layer_size`.

    Notes
    -----
    - The expansion is performed by convolving the mask with a square kernel
      of size `(2 * layer_size + 1)`, effectively performing a morphological
      dilation.
    - This operation is useful for defining buffer regions (e.g., just outside
      a limiter or wall in equilibrium simulations).

    Examples
    --------
        mask = np.array([[0, 1, 0],
    ...                  [0, 1, 0],
    ...                  [0, 0, 0]], dtype=bool)
        make_broad_mask(mask, layer_size=1)
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ False,  True,  False]])
    """

    nx, ny = np.shape(mask)
    layer_mask = np.zeros(np.array([nx, ny]) + 2 * np.array([layer_size, layer_size]))

    for i in np.arange(-layer_size, layer_size + 1) + layer_size:
        for j in np.arange(-layer_size, layer_size + 1) + layer_size:
            layer_mask[i : i + nx, j : j + ny] += mask
    layer_mask = layer_mask[layer_size : layer_size + nx, layer_size : layer_size + ny]
    layer_mask = (layer_mask > 0).astype(bool)
    return layer_mask


def plotProbes(
    probes, axis=None, show=True, floops=True, pickups=True, pickups_scale=0.05
):
    """
    Plot magnetic diagnostics including flux loops and pickup coils.

    This function visualises diagnostic probe locations used for equilibrium
    reconstruction, showing the spatial distribution of flux loops and pickup
    coils. Pickup coil orientations are indicated by small line segments
    scaled by a user-defined factor.

    Parameters
    ----------
    probes : object
        An object containing probe geometry data with the following attributes:
        - `floop_pos` : ndarray of shape (N, 2)
            Positions of flux loops in (R, Z) coordinates.
        - `pickup_pos` : ndarray of shape (M, 3)
            Positions of pickup coils in (R, φ, Z) coordinates (only R and Z used).
        - `pickup_or` : ndarray of shape (M, 3)
            Orientation vectors of the pickup coils.
    axis : matplotlib.axes.Axes, optional
        Axis object on which to plot. If `None`, a new figure and axis are created.
    show : bool, default=True
        If True, call `matplotlib.pyplot.show()` before returning.
    floops : bool, default=True
        If True, plot flux loop probe locations as orange diamonds.
    pickups : bool, default=True
        If True, plot pickup coil locations as brown circles and their orientations
        as short brown line segments.
    pickups_scale : float, default=0.05
        Scaling factor applied to pickup coil orientation vectors for visualization.

    Returns
    -------
    axis : matplotlib.axes.Axes
        The matplotlib axis containing the plotted probes.

    Notes
    -----
    - The function assumes `probes.floop_pos`, `probes.pickup_pos`, and
      `probes.pickup_or` are NumPy arrays with appropriate dimensions.
    - The orientation lines are plotted using (R, Z) components only.

    Examples
    --------
        axis = plotProbes(probes, pickups_scale=0.1)
        axis.set_title("Magnetic Diagnostics: Flux Loops and Pickup Coils")
    """

    # create axis if none exists
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    # locations of the flux loop probes
    if floops:
        axis.scatter(
            probes.floop_pos[:, 0],
            probes.floop_pos[:, 1],
            color="orange",
            marker="D",
            s=10,
        )

    # locations of the pickup coils + their orientation
    if pickups:
        # pickup orientation
        axis.plot(
            [
                probes.pickup_pos[:, 0],
                probes.pickup_pos[:, 0] + pickups_scale * probes.pickup_or[:, 0],
            ],
            [
                probes.pickup_pos[:, 2],
                probes.pickup_pos[:, 2] + pickups_scale * probes.pickup_or[:, 2],
            ],
            color="brown",
            markersize=1,
        )
        # pickup location
        axis.scatter(
            probes.pickup_pos[:, 0],
            probes.pickup_pos[:, 2],
            color="brown",
            marker="o",
            s=3,
        )

    if show:
        plt.legend(loc="upper right")
        plt.show()

    return axis
