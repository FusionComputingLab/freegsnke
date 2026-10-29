"""
Tracing of magnetic field lines in an equilibrium.

Copyright 2024 Nicola C. Amorisco, Adriano Agnello, George K. Holt, Ben Dudson.

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from builtins import object

import numpy as np
from scipy import interpolate
from scipy.integrate import odeint

from . import critical


class FieldTracer(object):
    """Follow magnetic field lines in an equilibrium."""

    def __init__(self, eq: object) -> None:
        """
        Initialise a FieldTracer for an equilibrium.

        The domain-checking method is the wall-based one if the equilibrium's
        tokamak has a wall, otherwise the computational domain is used.

        Parameters
        ----------
        eq : object
            Equilibrium object.
        """
        self._eq = eq

        if eq.tokamak.wall:
            # Machine has a wall, used to define edges
            self.updateEvolving = self.wallDomain
        else:
            # No wall, so just use the domain
            self.updateEvolving = self.eqDomain

    def eqDomain(
        self, R: np.ndarray, Z: np.ndarray, evolving: np.ndarray
    ) -> np.ndarray:
        """
        Zero the entries of `evolving` for points outside the computational domain.

        Parameters
        ----------
        R : np.ndarray
            Major radius of the points [m].
        Z : np.ndarray
            Height of the points [m].
        evolving : np.ndarray
            Array of the same shape as R and Z, modified in place.

        Returns
        -------
        np.ndarray
            The updated `evolving` array.
        """

        eps = 1e-2

        evolving[
            np.logical_or(
                np.logical_or((R < self._eq.Rmin + eps), (R > self._eq.Rmax - eps)),
                np.logical_or((Z < self._eq.Zmin + eps), (Z > self._eq.Zmax - eps)),
            )
        ] = 0.0
        return evolving

    def wallDomain(self, R: np.ndarray, Z: np.ndarray, evolving: np.ndarray) -> None:
        """
        Zero the entries of `evolving` for points outside the wall.

        This is done by counting intersections of the line from the domain
        centre to each point with the wall.

        Parameters
        ----------
        R : np.ndarray
            Major radius of the points [m].
        Z : np.ndarray
            Height of the points [m].
        evolving : np.ndarray
            Array of the same shape as R and Z, modified in place.
        """

        Rwall = self._eq.tokamak.wall.R
        Zwall = self._eq.tokamak.wall.Z

        # Location of the middle of the domain
        Rmid = 0.5 * (self._eq.Rmin + self._eq.Rmax)
        Zmid = 0.5 * (self._eq.Zmin + self._eq.Zmax)

        # True if a point is outside the wall
        outside = np.full(R.shape, False)

        nwall = len(Rwall)
        for wi in range(nwall):
            # Find intersection between line from (Rmid, Zmid) to (R,Z)
            # with wall segment from (Rwall[wi], Zwall[wi]) to (Rwall[wi+1], Zwall[wi+1])
            wip = (wi + 1) % nwall

            a = R - Rmid
            b = Rwall[wip] - Rwall[wi]
            c = Z - Zmid
            d = Zwall[wip] - Zwall[wi]

            dr = Rwall[wip] - Rmid
            dz = Zwall[wip] - Zmid

            det = a * d - b * c

            # Note: Here expect divide-by-zero errors occasionally
            # but the resulting values won't be used
            alpha = (d * dr - b * dz) / det  # Location along line 1 [0,1]
            beta = (a * dz - c * dr) / det  # Location along line 2 [0,1]

            # If the lines cross, change outside <-> inside
            # Note: If det is small then lines are nearly parallel
            outside ^= (
                (np.abs(det) > 1e-6)
                & (alpha > 0.0)
                & (alpha < 1.0)
                & (beta > 0.0)
                & (beta < 1.0)
            )
        evolving[outside] = 0.0

    def fieldDirection(
        self,
        pos: np.ndarray,
        toroidal_angle: float,
        evolving: np.ndarray,
        backward: bool,
    ) -> np.ndarray:
        """
        Calculate the rate of change of position with toroidal angle.

        Used as the right-hand side of the ODE integrated by ``follow``.

        Parameters
        ----------
        pos : np.ndarray
            Flattened (R, Z, length) positions of the field lines.
        toroidal_angle : float
            Toroidal angle (unused; the field is axisymmetric).
        evolving : np.ndarray
            Per-line flags, 1 for lines still inside the domain and 0 once they
            have left it; updated in place.
        backward : bool
            If True, follow the field lines in the -B direction.

        Returns
        -------
        np.ndarray
            Flattened (dR/dphi, dZ/dphi, dl/dphi) for each line.
        """
        position = pos.reshape((-1, 3))
        R = position[:, 0]
        Z = position[:, 1]
        # Length is position[:,2]

        # Calculate magnetic field components
        Br = self._eq.Br(R, Z)
        Bz = self._eq.Bz(R, Z)
        Btor = self._eq.Btor(R, Z)

        B = np.sqrt(Br**2 + Bz**2 + Btor**2)

        # Detect when the boundary has been reached
        self.updateEvolving(R, Z, evolving)

        # Common factor in all equations
        evolving_R_Bt = evolving * R / Btor

        # Rate of change of position with toroidal angle phi
        dRdphi = evolving_R_Bt * Br
        dZdphi = evolving_R_Bt * Bz
        dldphi = evolving_R_Bt * B

        if backward:
            # Reverse direction
            dRdphi *= -1.0
            dZdphi *= -1.0

        return np.column_stack((dRdphi, dZdphi, dldphi)).flatten()

    def follow(
        self,
        Rstart: np.ndarray,
        Zstart: np.ndarray,
        angles: np.ndarray,
        rtol: float | None = None,
        backward: bool = False,
    ) -> np.ndarray:
        """
        Follow magnetic field lines from (Rstart, Zstart) to given toroidal angles.

        Parameters
        ----------
        Rstart : np.ndarray
            Starting major radii [m].
        Zstart : np.ndarray
            Starting heights [m], same shape as `Rstart`.
        angles : np.ndarray
            Toroidal angles at which to output the positions [rad].
        rtol : float, optional
            Relative tolerance passed to ``scipy.integrate.odeint``.
        backward : bool, optional
            If True, the field lines are followed in the -B direction.

        Returns
        -------
        np.ndarray
            (R, Z, length) along each line, shape ``angles.shape + Rstart.shape + (3,)``.
        """

        Rstart = np.array(Rstart)
        Zstart = np.array(Zstart)

        array_shape = Rstart.shape
        assert Zstart.shape == array_shape

        evolving = np.ones(array_shape)

        # (R,Z,length) with length=0 initially
        position = np.column_stack((Rstart, Zstart, np.zeros(array_shape))).flatten()

        result = odeint(
            self.fieldDirection,
            position,
            angles,
            args=(evolving, backward),
            rtol=rtol,
        )

        return result.reshape(angles.shape + array_shape + (3,))


class LineCoordinates:
    """
    Coordinates of a field line.

    Attributes
    ----------
    R : np.ndarray
        Major radius [m].
    Z : np.ndarray
        Height [m].
    length : np.ndarray
        Field line length [m].

    All of R, Z and length are NumPy arrays of the same shape.
    """

    def __init__(self, R: np.ndarray, Z: np.ndarray, length: np.ndarray) -> None:
        """
        Store the field line coordinates.

        Parameters
        ----------
        R : np.ndarray
            Major radius [m].
        Z : np.ndarray
            Height [m].
        length : np.ndarray
            Field line length [m].
        """
        self.R = R
        self.Z = Z
        self.length = length


def traceFieldLines(
    eq: object,
    solwidth: float = 0.03,
    nlines: int = 10,
    nturns: int = 50,
    npoints: int = 200,
    axis: object | None = None,
) -> tuple[LineCoordinates, LineCoordinates]:
    """
    Trace field lines from the outboard midplane.

    Parameters
    ----------
    eq : object
        Equilibrium object.
    solwidth : float, optional
        The width of the SOL [m].
    nlines : int, optional
        Number of field lines to follow.
    nturns : int, optional
        Number of times around the tokamak to follow.
    npoints : int, optional
        Maximum number of points per line. Lines may hit a wall.
    axis : object, optional
        Matplotlib Axes. If given, field lines are plotted on it.

    Returns
    -------
    forward : LineCoordinates
        Coordinates of the field lines followed in the +B direction.
    backward : LineCoordinates
        Coordinates of the field lines followed in the -B direction.
        Both have ``R``, ``Z`` and ``length`` as 2D arrays of shape
        (npoints, nlines).

    Examples
    --------
    >>> forward, backward = traceFieldLines(eq)
    """
    ft = FieldTracer(eq)

    # Find the edge of the plasma
    psi = eq.psi()
    opoint, xpoint = critical.find_critical(eq.R, eq.Z, psi)

    r0, z0, psi_axis = opoint[0]
    psi_bndry = xpoint[0][2]

    # Find outboard midplane
    psifunc = interpolate.RectBivariateSpline(
        eq.R[:, 0], eq.Z[0, :], (psi - psi_axis) / (psi_bndry - psi_axis)
    )

    rmid, zmid = critical.find_psisurface(
        eq, psifunc, r0, z0, r0 + 10.0, z0, psival=1.0
    )

    # Starting locations, just outside the separatrix
    # Start first point a bit away from separatrix
    rstart = np.linspace(rmid + (solwidth / nlines), rmid + solwidth, nlines)
    zstart = np.full(nlines, zmid)

    angles = np.linspace(0.0, nturns * 2 * np.pi, npoints)

    # Follow field lines
    line_forward = ft.follow(rstart, zstart, angles)
    line_backward = ft.follow(rstart, zstart, angles, backward=True)

    forward = LineCoordinates(
        line_forward[:, :, 0], line_forward[:, :, 1], line_forward[:, :, 2]
    )

    backward = LineCoordinates(
        line_backward[:, :, 0], line_backward[:, :, 1], line_backward[:, :, 2]
    )

    if axis:
        # Plot field lines
        axis.plot(forward.R, forward.Z)
        axis.plot(backward.R, backward.Z)

        # Mark the starting location
        axis.plot(rstart, zstart, color="k", linewidth=2)

    return forward, backward
