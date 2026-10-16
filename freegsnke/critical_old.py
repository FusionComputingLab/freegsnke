"""
Routines to find critical points (O- and X-points)

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

from typing import Callable
from unittest import makeSuite

import numpy as np
from numpy import (
    abs,
    amax,
    arctan2,
    argmax,
    argmin,
    clip,
    cos,
    dot,
    linspace,
    pi,
    sin,
    sqrt,
    sum,
    zeros,
)
from numpy.linalg import inv
from scipy import interpolate

try:
    from numba import njit
except ImportError:

    def njit(*args: object, **kwargs: object) -> Callable[[Callable], Callable]:
        """
        Fallback no-op replacement for ``numba.njit`` when numba is unavailable.

        Parameters
        ----------
        *args : object
            Ignored positional arguments.
        **kwargs : object
            Ignored keyword arguments.

        Returns
        -------
        callable
            A decorator that returns the decorated function unchanged.
        """
        return lambda f: f


from warnings import warn


def find_critical_old(
    R: np.ndarray, Z: np.ndarray, psi: np.ndarray, discard_xpoints: bool = True
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """
    Find critical points (legacy FreeGS implementation using spline Newton iterations).

    Parameters
    ----------
    R : np.ndarray
        Major radii, shape (nr, nz).
    Z : np.ndarray
        Heights, shape (nr, nz).
    psi : np.ndarray
        Poloidal flux values, shape (nr, nz).
    discard_xpoints : bool, optional
        If True, discard X-points for which psi is not monotonic along the line
        from the primary O-point.

    Returns
    -------
    opoint : list of tuple
        O-points as (R, Z, psi) tuples. The first is the primary O-point
        (magnetic axis).
    xpoint : list of tuple
        X-points as (R, Z, psi) tuples. The first is the primary X-point
        (separatrix), ordered by closeness to the axis flux.
    """

    # Get a spline interpolation function
    f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psi)

    # Find candidate locations, based on minimising Bp^2
    Bp2 = (f(R, Z, dx=1, grid=False) ** 2 + f(R, Z, dy=1, grid=False) ** 2) / R**2

    # Get grid resolution, which determines a reasonable tolerance
    # for the Newton iteration search area
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    radius_sq = 9 * (dR**2 + dZ**2)

    # Find local minima

    J = zeros([2, 2])

    xpoint = []
    opoint = []

    nx, ny = Bp2.shape
    for i in range(2, nx - 2):
        for j in range(2, ny - 2):
            if (
                (Bp2[i, j] < Bp2[i + 1, j + 1])
                and (Bp2[i, j] < Bp2[i + 1, j])
                and (Bp2[i, j] < Bp2[i + 1, j - 1])
                and (Bp2[i, j] < Bp2[i - 1, j + 1])
                and (Bp2[i, j] < Bp2[i - 1, j])
                and (Bp2[i, j] < Bp2[i - 1, j - 1])
                and (Bp2[i, j] < Bp2[i, j + 1])
                and (Bp2[i, j] < Bp2[i, j - 1])
            ):

                # Found local minimum

                R0 = R[i, j]
                Z0 = Z[i, j]

                # Use Newton iterations to find where
                # both Br and Bz vanish
                R1 = R0
                Z1 = Z0

                count = 0
                while True:

                    Br = -f(R1, Z1, dy=1, grid=False) / R1
                    Bz = f(R1, Z1, dx=1, grid=False) / R1

                    if Br**2 + Bz**2 < 1e-6:
                        # Found a minimum. Classify as either
                        # O-point or X-point

                        dR = R[1, 0] - R[0, 0]
                        dZ = Z[0, 1] - Z[0, 0]
                        d2dr2 = (psi[i + 2, j] - 2.0 * psi[i, j] + psi[i - 2, j]) / (
                            2.0 * dR
                        ) ** 2
                        d2dz2 = (psi[i, j + 2] - 2.0 * psi[i, j] + psi[i, j - 2]) / (
                            2.0 * dZ
                        ) ** 2
                        d2drdz = (
                            (psi[i + 2, j + 2] - psi[i + 2, j - 2]) / (4.0 * dZ)
                            - (psi[i - 2, j + 2] - psi[i - 2, j - 2]) / (4.0 * dZ)
                        ) / (4.0 * dR)
                        D = d2dr2 * d2dz2 - d2drdz**2

                        if D < 0.0:
                            # Found X-point
                            xpoint.append((R1, Z1, f(R1, Z1)[0][0]))
                        else:
                            # Found O-point
                            opoint.append((R1, Z1, f(R1, Z1)[0][0]))
                        break

                    # Jacobian matrix
                    # J = ( dBr/dR, dBr/dZ )
                    #     ( dBz/dR, dBz/dZ )

                    J[0, 0] = -Br / R1 - f(R1, Z1, dy=1, dx=1)[0][0] / R1
                    J[0, 1] = -f(R1, Z1, dy=2)[0][0] / R1
                    J[1, 0] = -Bz / R1 + f(R1, Z1, dx=2) / R1
                    J[1, 1] = f(R1, Z1, dx=1, dy=1)[0][0] / R1

                    d = dot(inv(J), [Br, Bz])

                    R1 = R1 - d[0]
                    Z1 = Z1 - d[1]

                    count += 1
                    # If (R1,Z1) is too far from (R0,Z0) then discard
                    # or if we've taken too many iterations
                    if ((R1 - R0) ** 2 + (Z1 - Z0) ** 2 > radius_sq) or (count > 100):
                        # Discard this point
                        break

    # Remove duplicates
    def remove_dup(
        points: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        """
        Remove duplicate critical points (those closer than ~3e-3 in R-Z distance).

        Parameters
        ----------
        points : list of tuple
            Critical points as (R, Z, psi) tuples.

        Returns
        -------
        list of tuple
            The points with duplicates removed (first occurrence kept).
        """
        result = []
        for n, p in enumerate(points):
            dup = False
            for p2 in result:
                if (p[0] - p2[0]) ** 2 + (p[1] - p2[1]) ** 2 < 1e-5:
                    dup = True  # Duplicate
                    break
            if not dup:
                result.append(p)  # Add to the list
        return result

    xpoint = remove_dup(xpoint)
    opoint = remove_dup(opoint)

    if len(opoint) == 0:
        # Can't order primary O-point, X-point so return
        print("Warning: No O points found")
        return opoint, xpoint

    # Find primary O-point by sorting by distance from middle of domain
    Rmid = 0.5 * (R[-1, 0] + R[0, 0])
    Zmid = 0.5 * (Z[0, -1] + Z[0, 0])
    opoint.sort(key=lambda x: (x[0] - Rmid) ** 2 + (x[1] - Zmid) ** 2)

    # Draw a line from the O-point to each X-point. Psi should be
    # monotonic; discard those which are not

    if discard_xpoints:
        Ro, Zo, Po = opoint[0]  # The primary O-point
        xpt_keep = []
        for xpt in xpoint:
            Rx, Zx, Px = xpt

            rline = linspace(Ro, Rx, num=50)
            zline = linspace(Zo, Zx, num=50)

            pline = f(rline, zline, grid=False)

            if Px < Po:
                pline *= -1.0  # Reverse, so pline is maximum at X-point

            # Now check that pline is monotonic
            # Tried finding maximum (argmax) and testing
            # how far that is from the X-point. This can go
            # wrong because psi can be quite flat near the X-point
            # Instead here look for the difference in psi
            # rather than the distance in space

            maxp = amax(pline)
            if (maxp - pline[-1]) / (maxp - pline[0]) > 0.001:
                # More than 0.1% drop in psi from maximum to X-point
                # -> Discard
                continue

            ind = argmin(pline)  # Should be at O-point
            if (rline[ind] - Ro) ** 2 + (zline[ind] - Zo) ** 2 > 1e-4:
                # Too far, discard
                continue
            xpt_keep.append(xpt)
        xpoint = xpt_keep

    # Sort X-points by distance to primary O-point in psi space
    psi_axis = opoint[0][2]
    xpoint.sort(key=lambda x: (x[2] - psi_axis) ** 2)

    return opoint, xpoint


# Remove duplicates
def remove_dup(
    points: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """
    Remove duplicate critical points (those closer than ~3e-3 in R-Z distance).

    Parameters
    ----------
    points : list of tuple
        Critical points as (R, Z, psi) tuples.

    Returns
    -------
    list of tuple
        The points with duplicates removed (first occurrence kept).
    """
    result = []
    for n, p in enumerate(points):
        dup = False
        for p2 in result:
            if (p[0] - p2[0]) ** 2 + (p[1] - p2[1]) ** 2 < 1e-5:
                dup = True  # Duplicate
                break
        if not dup:
            result.append(p)  # Add to the list
    return result


def find_critical(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    mask_inside_limiter: np.ndarray | None = None,
    signIp: float | None = 1,
    discard_xpoints: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the O- and X-points of psi and select the X-points consistent with the plasma current.

    Parameters
    ----------
    R : np.ndarray
        Major radii, shape (nr, nz).
    Z : np.ndarray
        Heights, shape (nr, nz).
    psi : np.ndarray
        Poloidal flux values, shape (nr, nz).
    mask_inside_limiter : np.ndarray of bool or None, optional
        Mask of grid points inside the limiter, used to discard O-points outside it.
    signIp : float or None, optional
        Sign of the plasma current. X-points are kept only if their flux is on the
        correct side of the axis flux. If None, no such selection is made.
    discard_xpoints : bool, optional
        Unused (kept for backwards compatibility).

    Returns
    -------
    opoint : np.ndarray
        O-points as rows of (R, Z, psi), primary O-point first.
    xpoint : np.ndarray
        X-points as rows of (R, Z, psi), primary X-point first.
    """
    # if old:
    #     opoint, xpoint = find_critical_old(R,Z,psi, discard_xpoints)
    # else:
    opoint, xpoint = fastcrit(R, Z, psi, mask_inside_limiter)

    if len(xpoint) > 0 and (signIp is not None):
        # select xpoint with the correct ordering wrt Ip
        xpoint = xpoint[((xpoint[:, 2] - opoint[:1, 2]) * signIp) < 0]
    if len(xpoint) > 1:
        # check distance to opoint and in case discard xpoints on non-monotonic LOS
        closer_xpoint = np.argmin(np.linalg.norm((xpoint - opoint[:1])[:, :2], axis=-1))
        if closer_xpoint != 0:
            f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psi)
            result = False
            while result is False:
                result = discard_xpoints_f(opoint[0], xpoint[0], f)
                if result is False:
                    xpoint = xpoint[1:]
                    result = len(xpoint) < 1
            print(xpoint)
    return opoint, xpoint


# # this is 10x faster if the numba import works; otherwise, @njit is the identity and fastcrit is 3x faster anyways
@njit(fastmath=True, cache=True)
def scan_for_crit(
    R: np.ndarray, Z: np.ndarray, psi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan the grid for local minima of |grad psi|^2 and classify them as O- or X-points.

    Parameters
    ----------
    R : np.ndarray
        Major radii, shape (nr, nz).
    Z : np.ndarray
        Heights, shape (nr, nz).
    psi : np.ndarray
        Poloidal flux values, shape (nr, nz).

    Returns
    -------
    opoint : np.ndarray
        O-points as rows of (R, Z, psi), shape (n_o, 3).
    xpoint : np.ndarray
        X-points as rows of (R, Z, psi), shape (n_x, 3).
    """
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    Bp2 = np.zeros_like(psi)
    psiR = Bp2.copy()
    psiZ = Bp2.copy()
    psiR[1:-1, 1:-1] = 0.5 * (psi[2:, 1:-1] - psi[:-2, 1:-1]) / dR
    psiZ[1:-1, 1:-1] = 0.5 * (psi[1:-1, 2:] - psi[1:-1, :-2]) / dZ
    #
    #    psiR[0,:]=(psi[1,:]-psi[0,:])/dR
    #     psiR[-1,:]=(psi[-1,:]-psi[-2,:])/dR
    #     psiR[1:-1,0]=(psi[1:,0]-psi[:-1,0])/dR
    #     psiR[1:-1,-1]=(psi[1:,-1]-psi[:-1,-0])/dR
    #
    Bp2[:, :] = psiR**2 + psiZ**2  # /R[:,:]**2
    #
    xpoint = [(-999.0, -999.0, -999.0)]
    opoint = [(-999.0, -999.0, -999.0)]
    # start off by finding coarse values of Bp2 closest to 0.0 as in Ben Dudsons's routine
    for i in range(1, len(Bp2) - 1):
        for j in range(1, len(Bp2[0]) - 1):
            if (
                (Bp2[i, j] < Bp2[i + 1, j + 1])
                and (Bp2[i, j] < Bp2[i + 1, j])
                and (Bp2[i, j] < Bp2[i + 1, j - 1])
                and (Bp2[i, j] < Bp2[i - 1, j + 1])
                and (Bp2[i, j] < Bp2[i - 1, j])
                and (Bp2[i, j] < Bp2[i - 1, j - 1])
                and (Bp2[i, j] < Bp2[i, j + 1])
                and (Bp2[i, j] < Bp2[i, j - 1])
            ):
                # Found local minimum
                R0 = R[i, j]
                Z0 = Z[i, j]
                fR, fZ = psiR[i, j], psiZ[i, j]
                fRR = (psi[i + 1, j] - 2 * psi[i, j] + psi[i - 1, j]) / dR**2
                fZZ = (psi[i, j + 1] - 2 * psi[i, j] + psi[i, j - 1]) / dZ**2
                fRZ = (
                    0.5
                    * (
                        psi[i + 1, j + 1]
                        + psi[i - 1, j - 1]
                        - psi[i - 1, j + 1]
                        - psi[i + 1, j - 1]
                    )
                    / (dR * dZ)
                )
                det = fRR * fZZ - 0.25 * fRZ**2
                #
                if det != 0:
                    delta_R = -(fR * fZZ - 0.5 * fRZ * fZ) / det
                    delta_Z = -(fZ * fRR - 0.5 * fRZ * fR) / det
                if np.abs(delta_R) <= dR and np.abs(delta_Z) <= dZ:
                    est_psi = psi[i, j] + 0.5 * (
                        fR * delta_R + fZ * delta_Z
                    )  # + 0.5*(fRR*delta_R**2 + fZZ*delta_Z**2 + fRZ*delta_R*delta_Z)
                    crpoint = (R0 + delta_R, Z0 + delta_Z, est_psi)
                    if det > 0.0:
                        opoint = [crpoint] + opoint
                    else:
                        xpoint = [crpoint] + xpoint

    xpoint = np.array(xpoint)
    opoint = np.array(opoint)
    # do NOT remove the "pop" command below, the lists were initialised with (-999.,-999.) so that numba could compile
    # xpoint.pop()
    # opoint.pop()
    xpoint = xpoint[xpoint[:, 0] > -990]
    opoint = opoint[opoint[:, 0] > -990]
    return opoint, xpoint


def fastcrit(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    mask_inside_limiter: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find and order the critical points of psi using the fast grid scan.

    Parameters
    ----------
    R : np.ndarray
        Major radii, shape (nr, nz).
    Z : np.ndarray
        Heights, shape (nr, nz).
    psi : np.ndarray
        Poloidal flux values, shape (nr, nz).
    mask_inside_limiter : np.ndarray of bool or None
        Mask of grid points inside the limiter. If not None, O-points outside it
        are removed.

    Returns
    -------
    opoint : np.ndarray
        O-points as rows of (R, Z, psi), ordered by distance from the domain centre.
    xpoint : np.ndarray
        X-points as rows of (R, Z, psi), ordered by closeness of their flux to the
        axis flux.

    Raises
    ------
    ValueError
        If no O-points are found.
    """
    opoint, xpoint = scan_for_crit(R, Z, psi)

    len_opoint = len(opoint)
    if len_opoint == 0:
        # Can't order primary O-point, X-point so return
        raise ValueError("No opoints found!")
        # return opoint, xpoint
    elif mask_inside_limiter is not None:
        # remove any opoint outside the limiter
        posR = np.argmin((R[:, :1].T - opoint[:, :1]) ** 2, axis=1)
        posZ = np.argmin((Z[:1, :] - opoint[:, 1:2]) ** 2, axis=1)
        opoint = opoint[mask_inside_limiter[posR, posZ]]

    if len_opoint == 0:
        # Can't order primary O-point, X-point so return
        raise ValueError("No opoints found!")
    elif len_opoint > 1:
        # Find primary O-point by sorting by distance from middle of domain
        Rmid = 0.5 * (R[-1, 0] + R[0, 0])
        Zmid = 0.5 * (Z[0, -1] + Z[0, 0])
        opoint_ordering = np.argsort(
            (opoint[:, 0] - Rmid) ** 2 + (opoint[:, 1] - Zmid) ** 2
        )
        opoint = opoint[opoint_ordering]

    # # check that primary opoint is inside the limiter
    # result = False
    # while result is False:
    #     posR = np.argmin(np.abs(R[:,0] - opoint[:1,0]))
    #     posZ = np.argmin(np.abs(Z[0,:] - opoint[:1,1]))
    #     if mask_inside_limiter[posR, posZ] < 1:
    #         opoint = opoint[1:]
    #         if len(opoint) == 0:
    #             raise ValueError('No valid opoints found!')
    #     else:
    #         result = True
    psi_axis = opoint[0][2]
    # opoint.sort(key=lambda x: (x[0] - Rmid) ** 2 + (x[1] - Zmid) ** 2)

    # Draw a line from the O-point to each X-point. Psi should be
    # monotonic; discard those which are not

    # if discard_xpoints:
    #     f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psi)
    #     Ro, Zo, Po = opoint[0]  # The primary O-point
    #     xpt_keep = []
    #     for xpt in xpoint:
    #         Rx, Zx, Px = xpt

    #         rline = linspace(Ro, Rx, num=50)
    #         zline = linspace(Zo, Zx, num=50)

    #         pline = f(rline, zline, grid=False)

    #         if Px < Po:
    #             pline *= -1.0  # Reverse, so pline is maximum at X-point

    #         # Now check that pline is monotonic
    #         # Tried finding maximum (argmax) and testing
    #         # how far that is from the X-point. This can go
    #         # wrong because psi can be quite flat near the X-point
    #         # Instead here look for the difference in psi
    #         # rather than the distance in space

    #         maxp = amax(pline)
    #         if (maxp - pline[-1]) / (maxp - pline[0]) > 0.001:
    #             # More than 0.1% drop in psi from maximum to X-point
    #             # -> Discard
    #             continue

    #         ind = argmin(pline)  # Should be at O-point
    #         if (rline[ind] - Ro) ** 2 + (zline[ind] - Zo) ** 2 > 1e-4:
    #             # Too far, discard
    #             continue
    #         xpt_keep.append(xpt)
    #     xpoint = xpt_keep

    # Sort X-points by distance to primary O-point in psi space
    if len(xpoint) > 1:
        xpoint_ordering = np.argsort((xpoint[:, 2] - psi_axis) ** 2)
        xpoint = xpoint[xpoint_ordering]
    # xpoint.sort(key=lambda x: (x[2] - psi_axis) ** 2)
    #
    return opoint, xpoint


def discard_xpoints_f(
    opoint: np.ndarray,
    xpt: np.ndarray,
    f: Callable[..., np.ndarray],
) -> bool:
    """
    Check that psi is monotonic along the line from the O-point to an X-point.

    Parameters
    ----------
    opoint : np.ndarray
        The primary O-point as (R, Z, psi).
    xpt : np.ndarray
        An individual X-point as (R, Z, psi).
    f : callable
        Spline interpolator of psi, called as ``f(R, Z, grid=False)``.

    Returns
    -------
    bool
        True if the X-point is acceptable (monotonic), False if it should be discarded.
    """
    Ro, Zo, Po = opoint  # The primary O-point
    result = False

    # for xpt in xpoint:
    Rx, Zx, Px = xpt

    rline = linspace(Ro, Rx, num=50)
    zline = linspace(Zo, Zx, num=50)

    pline = f(rline, zline, grid=False)

    if Px < Po:
        pline *= -1.0  # Reverse, so pline is maximum at X-point

    # Now check that pline is monotonic
    # Tried finding maximum (argmax) and testing
    # how far that is from the X-point. This can go
    # wrong because psi can be quite flat near the X-point
    # Instead here look for the difference in psi
    # rather than the distance in space

    maxp = amax(pline)
    if (maxp - pline[-1]) / (maxp - pline[0]) < 0.001:
        # Less than 0.1% drop in psi from maximum to X-point

        ind = argmin(pline)  # Should be at O-point
        if (rline[ind] - Ro) ** 2 + (zline[ind] - Zo) ** 2 < 1e-4:
            # Accept
            result = True

    return result


def core_mask(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    opoint: np.ndarray,
    xpoint: np.ndarray | list = [],
    psi_bndry: float | None = None,
) -> np.ndarray:
    """
    Mark the parts of the domain which are in the core.

    Parameters
    ----------
    R : np.ndarray
        Major radius values, shape (nx, ny).
    Z : np.ndarray
        Height values, shape (nx, ny).
    psi : np.ndarray
        Poloidal flux, shape (nx, ny).
    opoint : np.ndarray
        O-points as returned by find_critical.
    xpoint : np.ndarray or list, optional
        X-points as returned by find_critical.
    psi_bndry : float or None, optional
        If not None, this is used to find the separatrix, not the X-points.

    Returns
    -------
    np.ndarray
        Array of shape (nx, ny) which is 1 inside the core, 0 outside.
    """

    mask = zeros(psi.shape)
    nx, ny = psi.shape

    # Start and end points
    Ro, Zo, psi_axis = opoint[0]
    if psi_bndry is None:
        _, _, psi_bndry = xpoint[0]

    # Normalise psi
    psin = (psi - psi_axis) / (psi_bndry - psi_axis)

    # Need some care near X-points to avoid flood filling through saddle point
    # Here we first set the x-points regions to a value, to block the flood fill
    # then later return to handle these more difficult cases
    #
    xpt_inds = []
    for rx, zx, _ in xpoint:
        # Find nearest index
        ix = argmin(abs(R[:, 0] - rx))
        jx = argmin(abs(Z[0, :] - zx))
        xpt_inds.append((ix, jx))
        # Fill this point and all around with '2'
        for i in np.clip([ix - 1, ix, ix + 1], 0, nx - 1):
            for j in np.clip([jx - 1, jx, jx + 1], 0, ny - 1):
                mask[i, j] = 2

    # Find nearest index to start
    rind = argmin(abs(R[:, 0] - Ro))
    zind = argmin(abs(Z[0, :] - Zo))

    stack = [(rind, zind)]  # List of points to inspect in future

    while stack:  # Whilst there are any points left
        i, j = stack.pop()  # Remove from list

        # Check the point to the left (i,j-1)
        if (j > 0) and (psin[i, j - 1] < 1.0) and (mask[i, j - 1] < 0.5):
            stack.append((i, j - 1))

        # Scan along a row to the right
        while True:
            mask[i, j] = 1  # Mark as in the core

            if (i < nx - 1) and (psin[i + 1, j] < 1.0) and (mask[i + 1, j] < 0.5):
                stack.append((i + 1, j))
            if (i > 0) and (psin[i - 1, j] < 1.0) and (mask[i - 1, j] < 0.5):
                stack.append((i - 1, j))

            if j == ny - 1:  # End of the row
                break
            if (psin[i, j + 1] >= 1.0) or (mask[i, j + 1] > 0.5):
                break  # Finished this row
            j += 1  # Move to next point along

    # Now return to X-point locations
    for ix, jx in xpt_inds:
        for i in np.clip([ix - 1, ix, ix + 1], 0, nx - 1):
            for j in np.clip([jx - 1, jx, jx + 1], 0, ny - 1):
                if psin[i, j] < 1.0:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0

    return mask


# def core_mask(R, Z, psi, opoint, xpoint=[], psi_bndry=None, old=False):
#     if old: return core_mask_old(R, Z, psi, opoint, xpoint, psi_bndry)
#     else: return inside_mask(R, Z, psi, opoint, xpoint, psi_bndry)


@njit(fastmath=True, cache=True)
def inside_mask(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    opoint: np.ndarray,
    xpoint: np.ndarray | list = [],
    mask_outside_limiter: np.ndarray | None = None,
    psi_bndry: float | None = None,
) -> np.ndarray:
    """
    Mark the parts of the domain which are in the core (JIT-compilable version of core_mask).

    Similar to core_mask above, except:
    (1) it's all stuff that can be JIT-compiled
    (2) some stuff is vectorised
    (3) the stack does not necessarily explore in only one direction, and it should be neat around the X-points

    Parameters
    ----------
    R : np.ndarray
        Major radius values, shape (nx, ny).
    Z : np.ndarray
        Height values, shape (nx, ny).
    psi : np.ndarray
        Poloidal flux, shape (nx, ny).
    opoint : np.ndarray
        O-points as rows of (R, Z, psi); the first is the primary O-point.
    xpoint : np.ndarray or list, optional
        X-points as rows of (R, Z, psi).
    mask_outside_limiter : np.ndarray or None, optional
        Array of shape (nx, ny), non-zero outside the limiter, used to block the fill.
    psi_bndry : float or None, optional
        If not None, this is used as the boundary flux instead of that of the first X-point.

    Returns
    -------
    np.ndarray of bool
        Mask of shape (nx, ny), True inside the core.
    """
    #
    if mask_outside_limiter is not None:
        mask = mask_outside_limiter.copy()
    else:
        mask = zeros(psi.shape)
    nx, ny = psi.shape
    #
    # Start and end points
    Ro, Zo, psi_axis = opoint[0]
    if psi_bndry is None:
        _, _, psi_bndry = xpoint[0]
    #
    # Normalise psi
    psin = (psi - psi_axis) / (psi_bndry - psi_axis)
    #
    xpt_inds = []
    for rx, zx, _ in xpoint:
        # Find nearest index
        ix = argmin(abs(R[:, 0] - rx))
        jx = argmin(abs(Z[0, :] - zx))
        xpt_inds.append((ix, jx))
        # Fill this point and all around with '2'#
        ilo, ihi = min(max(ix - 1, 0), nx - 1), max(0, min(ix + 1, nx - 1))
        jlo, jhi = min(max(jx - 1, 0), ny - 1), max(0, min(jx + 1, ny - 1))
        mask[ilo : ihi + 1, jlo : jhi + 1] = 2
    #
    # Find nearest index to start
    rind = argmin(abs(R[:, 0] - Ro))
    zind = argmin(abs(Z[0, :] - Zo))
    #
    stack = [(rind, zind)]  # List of points to inspect in future
    #
    while stack:  # Whilst there are any points left
        i, j = stack.pop()  # Remove from list
        #
        # Check the point below (i,j-1) , append to list of stuff to be checked if valid
        if (j > 0) and (psin[i, j - 1] < 1.0) and (mask[i, j - 1] < 0.5):
            stack.append((i, j - 1))
        # Check the point above
        if (j < ny - 1) and (psin[i, j + 1] < 1.0) and (mask[i, j + 1] < 0.5):
            stack.append((i, j + 1))
        #
        # Scan along a row to the right
        mask[i, j] = 1  # Mark as sampled in the core
        #
        if (i < nx - 1) and (psin[i + 1, j] < 1.0) and (mask[i + 1, j] < 0.5):
            stack.append((i + 1, j))
        if (i > 0) and (psin[i - 1, j] < 1.0) and (mask[i - 1, j] < 0.5):
            stack.append((i - 1, j))
    #
    # Now return to X-point locations
    for ix, jx in xpt_inds:
        ilo, ihi = min(max(ix - 1, 0), nx - 1), max(0, min(ix + 1, nx - 1))
        jlo, jhi = min(max(jx - 1, 0), ny - 1), max(0, min(jx + 1, ny - 1))
        mask[ilo : ihi + 1, jlo : jhi + 1] = (
            1
            * (mask[ilo : ihi + 1, jlo : jhi + 1] == 1)
            * (psin[ilo : ihi + 1, jlo : jhi + 1] < 1.0)
        )
    #

    # remove effect of mask_outside_limiter
    mask = mask == 1

    return mask


def find_psisurface(
    eq: object,
    psifunc: Callable[..., np.ndarray],
    r0: float,
    z0: float,
    r1: float,
    z1: float,
    psival: float = 1.0,
    n: int = 100,
    axis: object | None = None,
) -> tuple[float, float]:
    """
    Find where a line from inside the separatrix crosses a given psi surface.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium object (provides the domain limits Rmin, Rmax, Zmin, Zmax).
    psifunc : callable
        Function ``psifunc(r, z, grid=False)`` returning normalised psi.
    r0 : float
        R of the start location, inside the separatrix.
    z0 : float
        Z of the start location, inside the separatrix.
    r1 : float
        R of the end location, outside the separatrix (clipped to the domain).
    z1 : float
        Z of the end location, outside the separatrix (clipped to the domain).
    psival : float, optional
        Value of psi for which to find the surface.
    n : int, optional
        Number of starting points to use along the line.
    axis : matplotlib axis or None, optional
        If given, the line and the result are plotted on it.

    Returns
    -------
    r, z : float
        Coordinates of the crossing point. If `psival` has a length, the
        points along the line are returned unchanged.
    """
    # Clip (r1,z1) to be inside domain
    # Shorten the line so that the direction is unchanged
    if abs(r1 - r0) > 1e-6:
        rclip = clip(r1, eq.Rmin, eq.Rmax)
        z1 = z0 + (z1 - z0) * abs((rclip - r0) / (r1 - r0))
        r1 = rclip

    if abs(z1 - z0) > 1e-6:
        zclip = clip(z1, eq.Zmin, eq.Zmax)
        r1 = r0 + (r1 - r0) * abs((zclip - z0) / (z1 - z0))
        z1 = zclip

    r = linspace(r0, r1, n)
    z = linspace(z0, z1, n)

    if axis is not None:
        axis.plot(r, z)

    pnorm = psifunc(r, z, grid=False)

    if hasattr(psival, "__len__"):
        pass

    else:
        # Only one value
        ind = argmax(pnorm > psival)

        # Edited by Bhavin 31/07/18
        # Changed 1.0 to psival in f
        # make f gradient to psival surface
        f = (pnorm[ind] - psival) / (pnorm[ind] - pnorm[ind - 1])

        r = (1.0 - f) * r[ind] + f * r[ind - 1]
        z = (1.0 - f) * z[ind] + f * z[ind - 1]

    if axis is not None:
        axis.plot(r, z, "bo")

    return r, z


def find_separatrix(
    eq: object,
    opoint: np.ndarray | list | None = None,
    xpoint: np.ndarray | list | None = None,
    ntheta: int = 20,
    psi: np.ndarray | None = None,
    axis: object | None = None,
    psival: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """
    Find the R, Z coordinates of the separatrix for equilibrium `eq`.

    Points are equally spaced in geometric poloidal angle. If opoint, xpoint or
    psi are not given, they are calculated from eq.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium object.
    opoint : np.ndarray or list or None, optional
        O-points as (R, Z, psi) tuples/rows.
    xpoint : np.ndarray or list or None, optional
        X-points as (R, Z, psi) tuples/rows.
    ntheta : int, optional
        Number of points to find.
    psi : np.ndarray or None, optional
        Grid of psi on (R, Z).
    axis : matplotlib axis or None, optional
        Axis to plot points on.
    psival : float, optional
        Normalised psi value of the surface to find.

    Returns
    -------
    list of tuple
        One tuple (R, Z, R_X, Z_X) per poloidal angle, where R_X, Z_X are the
        coordinates of the X-point on the separatrix.
    """
    if psi is None:
        psi = eq.psi()

    if (opoint is None) or (xpoint is None):
        opoint, xpoint = find_critical(eq.R, eq.Z, psi)

    psinorm = (psi - opoint[0][2]) / (xpoint[0][2] - opoint[0][2])

    psifunc = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], psinorm)

    r0, z0 = opoint[0][0:2]

    theta_grid = linspace(0, 2 * pi, ntheta, endpoint=False)
    dtheta = theta_grid[1] - theta_grid[0]

    # Avoid putting theta grid points exactly on the X-points
    xpoint_theta = arctan2(xpoint[0][0] - r0, xpoint[0][1] - z0)
    xpoint_theta = xpoint_theta * (xpoint_theta >= 0) + (xpoint_theta + 2 * pi) * (
        xpoint_theta < 0
    )  # let's make it between 0 and 2*pi
    # How close in theta to allow theta grid points to the X-point
    TOLERANCE = 1.0e-3
    if any(abs(theta_grid - xpoint_theta) < TOLERANCE):
        warn("Theta grid too close to X-point, shifting by half-step")
        theta_grid += dtheta / 2

    isoflux = []
    for theta in theta_grid:
        r, z = find_psisurface(
            eq,
            psifunc,
            r0,
            z0,
            r0 + 10.0 * sin(theta),
            z0 + 10.0 * cos(theta),
            psival=psival,
            axis=axis,
            n=1000,
        )
        isoflux.append((r, z, xpoint[0][0], xpoint[0][1]))

    return isoflux


def find_safety(
    eq: object,
    npsi: int = 1,
    psinorm: float | np.ndarray | list | None = None,
    ntheta: int = 128,
    psi: np.ndarray | None = None,
    opoint: np.ndarray | list | None = None,
    xpoint: np.ndarray | list | None = None,
    axis: object | None = None,
) -> np.ndarray:
    """
    Find the safety factor for each value of psi.

    Calculates equally spaced flux surfaces. Points on each flux surface are
    equally spaced in poloidal angle. Performs line integral around flux
    surface to get q. If opoint, xpoint or psi are not given, they are
    calculated from eq.

    Parameters
    ----------
    eq : Equilibrium
        The equilibrium object.
    npsi : int, optional
        Number of flux surface values to find q for (overridden by the length of
        `psinorm`, or by 100 if `psinorm` is None).
    psinorm : float or array_like or None, optional
        Normalised flux surface value(s) to calculate q for. If None, 100 values
        are used.
    ntheta : int, optional
        Number of poloidal points to find it on.
    psi : np.ndarray or None, optional
        Grid of psi on (R, Z).
    opoint : np.ndarray or list or None, optional
        O-points as (R, Z, psi) tuples/rows.
    xpoint : np.ndarray or list or None, optional
        X-points as (R, Z, psi) tuples/rows.
    axis : matplotlib axis or None, optional
        Axis to plot points on.

    Returns
    -------
    np.ndarray
        Safety factor at each of the `npsi` normalised psi values.

    Raises
    ------
    ValueError
        If there is no X-point.
    """

    if psi is None:
        psi = eq.psi()

    if (opoint is None) or (xpoint is None):
        opoint, xpoint = find_critical(eq.R, eq.Z, psi)

    if (xpoint is None) or (len(xpoint) == 0):
        # No X-point
        raise ValueError("No X-point so no separatrix")
    else:
        psinormal = (psi - opoint[0][2]) / (xpoint[0][2] - opoint[0][2])

    psifunc = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], psinormal)

    r0, z0 = opoint[0][0:2]

    theta_grid = linspace(0, 2 * pi, ntheta, endpoint=False)
    dtheta = theta_grid[1] - theta_grid[0]

    # Avoid putting theta grid points exactly on the X-points
    xpoint_theta = arctan2(xpoint[0][0] - r0, xpoint[0][1] - z0)
    xpoint_theta = xpoint_theta * (xpoint_theta >= 0) + (xpoint_theta + 2 * pi) * (
        xpoint_theta < 0
    )  # let's make it between 0 and 2*pi
    # How close in theta to allow theta grid points to the X-point
    TOLERANCE = 1.0e-3

    if any(abs(theta_grid - xpoint_theta) < TOLERANCE):
        warn("Theta grid too close to X-point, shifting by half-step")
        theta_grid += dtheta / 2

    if psinorm is None:
        npsi = 100
        psirange = linspace(1.0 / (npsi + 1), 1.0, npsi, endpoint=False)
    else:
        try:
            psirange = psinorm
            npsi = len(psinorm)
        except TypeError:
            npsi = 1
            psirange = [psinorm]

    psisurf = zeros([npsi, ntheta, 2])

    # Calculate flux surface positions
    for i in range(npsi):
        psin = psirange[i]
        for j in range(ntheta):
            theta = theta_grid[j]
            r, z = find_psisurface(
                eq,
                psifunc,
                r0,
                z0,
                r0 + 8.0 * sin(theta),
                z0 + 8.0 * cos(theta),
                psival=psin,
                axis=axis,
            )
            psisurf[i, j, :] = [r, z]

    # Get variables for loop integral around flux surface
    r = psisurf[:, :, 0]
    z = psisurf[:, :, 1]
    fpol = eq.fpol(psirange[:]).reshape(npsi, 1)
    Br = eq.Br(r, z)
    Bz = eq.Bz(r, z)
    Bthe = sqrt(Br**2 + Bz**2)

    # Differentiate location w.r.t. index
    dr_di = (np.roll(r, 1, axis=1) - np.roll(r, -1, axis=1)) / 2.0
    dz_di = (np.roll(z, 1, axis=1) - np.roll(z, -1, axis=1)) / 2.0

    # Distance between points
    dl = sqrt(dr_di**2 + dz_di**2)

    # Integrand - Btor/(R*Bthe) = Fpol/(R**2*Bthe)
    qint = fpol / (r**2 * Bthe)

    # Integral
    q = sum(qint * dl, axis=1) / (2 * pi)

    return q
