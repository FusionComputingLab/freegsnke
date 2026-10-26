"""
Routines to find critical points (O- and X-points).

Copyright 2024 Nicola C. Amorisco, Adriano Agnello, George K. Holt, Ben Dudson.

Modified substantially from the original FreeGS code.

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

import warnings
from typing import Callable

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
    warnings.warn("Numba not found, using slower version")

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


from . import bilinear_interpolation

# from unittest import makeSuite


def find_critical_old(
    R: np.ndarray, Z: np.ndarray, psi: np.ndarray, discard_xpoints: bool = True
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """
    Finds the critical points in the total poloidal flux map ψ (legacy Newton-iteration version).

    Parameters
    ----------
    R : np.ndarray
        Radial positions at which flux measured, shape (nx, ny).
    Z : np.ndarray
        Vertical positions at which flux measured, shape (nx, ny).
    psi : np.ndarray
        Total poloidal flux [Webers/2pi], shape (nx, ny).
    discard_xpoints : bool, optional
        Discard X-points not on (or close to) the separatrix.

    Returns
    -------
    opoint : list of tuple
        O-point locations and flux values (R, Z, psi). The first tuple is the
        primary O-point (i.e. the magnetic axis).
    xpoint : list of tuple
        X-point locations and flux values (R, Z, psi). The first tuple is the
        primary X-point (i.e. closest in flux to the magnetic axis, usually on
        the plasma separatrix).
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


def remove_dup(
    points: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """
    Removes duplicate points in the list 'points' based on
    a squared Euclidean distance (in R and Z) below 1e-5.

    Parameters
    ----------
    points : list of tuple
        List of points, each with (R, Z, ...) as its first two entries.

    Returns
    -------
    list of tuple
        List of "unique" points (first occurrence kept).
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
    signIp: int | None = 1,
    discard_xpoints: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the critical points in the total poloidal flux map ψ.

    Parameters
    ----------
    R : np.ndarray
        Radial positions at which flux measured, shape (nx, ny).
    Z : np.ndarray
        Vertical positions at which flux measured, shape (nx, ny).
    psi : np.ndarray
        Total poloidal flux [Webers/2pi], shape (nx, ny).
    mask_inside_limiter : np.ndarray or None, optional
        Masking array, describing which (R, Z) grid points are inside the limiter.
    signIp : int or None, optional
        Sign of the plasma current (+1 or -1). If None, X-points are not
        selected by their flux ordering relative to the O-point.
    discard_xpoints : bool, optional
        Unused (X-points not on a monotonic line of sight from the O-point
        are always discarded).

    Returns
    -------
    opoint : np.ndarray
        O-point locations and flux values (R, Z, psi) as rows, shape (n_o, 3). The
        first row is the primary O-point (i.e. the magnetic axis).
    xpoint : np.ndarray
        X-point locations and flux values (R, Z, psi) as rows, shape (n_x, 3). The
        first row is the primary X-point (i.e. closest in flux to the magnetic axis,
        usually on the plasma separatrix). May be empty.
    """

    # if old:
    #     opoint, xpoint = find_critical_old(R,Z,psi, discard_xpoints)
    # else:
    opoint, xpoint = fastcrit(R, Z, psi, mask_inside_limiter)

    if len(xpoint) > 0 and (signIp is not None):
        # select xpoint with the correct ordering wrt Ip
        xpoint = xpoint[((xpoint[:, 2] - opoint[:1, 2]) * signIp) < 0]
        # also select xpoints that are not on the z=0 axis
        # xpoint = xpoint[np.abs(xpoint[:,1])>.1]
    if len(xpoint) > 0:
        # check distance to opoint and in case discard xpoints on non-monotonic LOS
        # closer_xpoint = np.argmin(np.linalg.norm((xpoint-opoint[:1])[:,:2], axis=-1))
        # if closer_xpoint != 0:
        # f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psi)
        result = False
        while result is False:
            result = discard_xpoints_f(R, Z, psi, opoint[0], xpoint[0])
            if result is False:
                xpoint = xpoint[1:]
                result = len(xpoint) < 1
            # print(xpoint)
    return opoint, xpoint


# # this is 10x faster if the numba import works; otherwise, @njit is the identity and fastcrit is 3x faster anyways
@njit(cache=True, fastmath=True)
def scan_for_crit(
    R: np.ndarray, Z: np.ndarray, psi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scans the grid for critical points of the total poloidal flux map ψ.

    Candidate points are local minima of |grad psi|^2 on the grid, refined by a
    single quadratic (Newton) step and classified as O- or X-points by the sign
    of the Hessian determinant. The points are not ordered.

    Parameters
    ----------
    R : np.ndarray
        Radial positions at which flux measured, shape (nx, ny).
    Z : np.ndarray
        Vertical positions at which flux measured, shape (nx, ny).
    psi : np.ndarray
        Total poloidal flux [Webers/2pi], shape (nx, ny).

    Returns
    -------
    opoint : np.ndarray
        O-point locations and flux values (R, Z, psi) as rows, shape (n_o, 3).
    xpoint : np.ndarray
        X-point locations and flux values (R, Z, psi) as rows, shape (n_x, 3).
    """

    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    Bp2 = np.zeros_like(psi)
    Bp2[1:-1, 1:-1] = (0.5 * (psi[2:, 1:-1] - psi[:-2, 1:-1]) / dR) ** 2 + (
        0.5 * (psi[1:-1, 2:] - psi[1:-1, :-2]) / dZ
    ) ** 2
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
                fR = 0.5 * (psi[i + 1, j] - psi[i - 1, j]) / dR
                fZ = 0.5 * (psi[i, j + 1] - psi[i, j - 1]) / dZ
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
                if np.abs(delta_R) < 1.5 * dR and np.abs(delta_Z) < 1.5 * dZ:
                    est_psi = psi[i, j] + 0.5 * (
                        fR * delta_R + fZ * delta_Z
                    )  # + 0.5*(fRR*delta_R**2 + fZZ*delta_Z**2 + fRZ*delta_R*delta_Z)
                    crpoint = (R0 + delta_R, Z0 + delta_Z, est_psi)
                    if det > 0.0:
                        # opoint = [crpoint] + opoint
                        opoint.append(crpoint)
                    else:
                        # xpoint = [crpoint] + xpoint
                        xpoint.append(crpoint)

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
    Finds and orders the critical points in the total poloidal flux map ψ.

    Parameters
    ----------
    R : np.ndarray
        Radial positions at which flux measured, shape (nx, ny).
    Z : np.ndarray
        Vertical positions at which flux measured, shape (nx, ny).
    psi : np.ndarray
        Total poloidal flux [Webers/2pi], shape (nx, ny).
    mask_inside_limiter : np.ndarray or None
        Masking array, describing which (R, Z) grid points are inside the limiter.
        If not None, O-points outside the limiter are discarded.

    Returns
    -------
    opoint : np.ndarray
        O-point locations and flux values (R, Z, psi) as rows, shape (n_o, 3), ordered
        by distance from the domain centre. The first row is the primary O-point
        (i.e. the magnetic axis).
    xpoint : np.ndarray
        X-point locations and flux values (R, Z, psi) as rows, shape (n_x, 3), ordered
        by closeness of their flux to the axis flux. The first row is the primary
        X-point.

    Raises
    ------
    ValueError
        If no (valid) O-points are found.
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

    len_opoint = len(opoint)
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
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    opoint: np.ndarray,
    xpt: np.ndarray,
) -> bool:
    """
    Checks whether an X-point is reachable from the O-point along a line of
    monotonic flux, so that X-points not on (or close to) the separatrix can be discarded.

    Parameters
    ----------
    R : np.ndarray
        Radial positions at which flux measured, shape (nx, ny).
    Z : np.ndarray
        Vertical positions at which flux measured, shape (nx, ny).
    psi : np.ndarray
        Total poloidal flux [Webers/2pi], shape (nx, ny).
    opoint : np.ndarray
        The (primary) O-point as (R, Z, psi).
    xpt : np.ndarray
        The individual X-point to test, as (R, Z, psi).

    Returns
    -------
    bool
        Returns True if the X-point is to be kept, False if discarded.
    """

    # Here opoint and xpt are individual critical points
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    Ro, Zo, Po = opoint  # The primary O-point
    result = False

    # for xpt in xpoint:
    Rx, Zx, Px = xpt

    num = int(max((np.abs(Rx - Ro)) / dR, (np.abs(Zx - Zo)) / dZ) + 2)
    # print('num ', num)

    # print('num')
    rline = linspace(Ro, Rx, num)  # (np.abs(Rx-Ro)//dR + 1))
    zline = linspace(Zo, Zx, num)  # (np.abs(Zx-Zo)//dZ + 1))

    pline = bilinear_interpolation.biliint(
        R, Z, psi, np.array([rline, zline])
    )  # , grid=False)

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
    opoint: np.ndarray | list,
    xpoint: np.ndarray | list = [],
    psi_bndry: float | None = None,
) -> np.ndarray:
    """
    Generates a masking array of computational grid points (R,Z) that
    reside within the last closed flux surface (plasma boundary).

    Parameters
    ----------
    R : np.ndarray
        Radial positions at which flux measured, shape (nx, ny).
    Z : np.ndarray
        Vertical positions at which flux measured, shape (nx, ny).
    psi : np.ndarray
        Total poloidal flux [Webers/2pi], shape (nx, ny).
    opoint : np.ndarray or list
        The O-point tuples/rows (R,Z,psi); the first is the primary O-point.
    xpoint : np.ndarray or list, optional
        The X-point tuples/rows (R,Z,psi).
    psi_bndry : float or None, optional
        Total poloidal flux on the plasma boundary [Webers/2pi]. If None, the flux
        of the first X-point is used.

    Returns
    -------
    np.ndarray
        Float array of shape (nx, ny) where 1 denotes a point inside the core and 0
        denotes a point outside.
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


@njit(fastmath=True, cache=True)
def inside_mask_(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    opoint: np.ndarray,
    xpoint: np.ndarray | list = [],
    mask_outside_limiter: np.ndarray | None = None,
    psi_bndry: float | None = None,
) -> np.ndarray:
    """
    Identifies the plasma region inside the separatrix by performing
    a flood-fill algorithm starting from the magnetic axis (O-point).

    Parameters
    ----------
    R : np.ndarray
        Radial positions at which flux measured, shape (nx, ny).
    Z : np.ndarray
        Vertical positions at which flux measured, shape (nx, ny).
    psi : np.ndarray
        Total poloidal flux [Webers/2pi], shape (nx, ny).
    opoint : np.ndarray
        O-points as rows (R,Z,psi); the first is the primary O-point.
    xpoint : np.ndarray or list, optional
        X-points as rows (R,Z,psi).
    mask_outside_limiter : np.ndarray or None, optional
        Masking array of shape (nx, ny), non-zero for (R, Z) grid points outside the limiter.
    psi_bndry : float or None, optional
        Total poloidal flux on the plasma boundary [Webers/2pi]. If None, the flux
        of the first X-point is used.

    Returns
    -------
    np.ndarray
        Boolean array of shape (nx, ny) where True denotes a point inside the core
        and False a point outside.
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


def inside_mask(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    opoint: np.ndarray,
    xpoint: np.ndarray | list = [],
    mask_outside_limiter: np.ndarray | None = None,
    psi_bndry: float | None = None,
    use_geom: bool = True,
) -> np.ndarray:
    """
    Calls inside_mask_ to find plasma region inside the separatrix (with
    the option of calling geom_inside_mask too).

    geom_inside_mask applies an additional geometrical contraint
    aimed at resolving cases of 'flooding' of the core mask through the primary Xpoint.
    It excludes regions based on perpendicular to segment
    from O-point to primary X-point.

    Parameters
    ----------
    R : np.ndarray
        Radial positions at which flux measured, shape (nx, ny).
    Z : np.ndarray
        Vertical positions at which flux measured, shape (nx, ny).
    psi : np.ndarray
        Total poloidal flux [Webers/2pi], shape (nx, ny).
    opoint : np.ndarray
        O-points as rows (R,Z,psi); the first is the primary O-point.
    xpoint : np.ndarray or list, optional
        X-points as rows (R,Z,psi).
    mask_outside_limiter : np.ndarray or None, optional
        Masking array of shape (nx, ny), non-zero for (R, Z) grid points outside the limiter.
    psi_bndry : float or None, optional
        Total poloidal flux on the plasma boundary [Webers/2pi].
    use_geom : bool, optional
        Use (refined) geom_inside_mask function in conjuction with inside_mask_
        to find the mask when at least one X-point is available (if set to True).

    Returns
    -------
    np.ndarray
        Boolean array of shape (nx, ny) where True denotes a point inside the core
        and False a point outside.
    """

    mask = inside_mask_(R, Z, psi, opoint, xpoint, mask_outside_limiter, psi_bndry)
    if use_geom and len(xpoint) > 0:
        # cure flooding
        mask = mask * geom_inside_mask(R, Z, opoint, xpoint)
        # apply geometric masking criterion to second Xpoint if close to double null
        if len(xpoint) > 1:
            if (
                np.abs((xpoint[0, 2] - xpoint[1, 2]) / (opoint[0, 2] - xpoint[0, 2]))
                < 0.1
            ):
                mask = mask * geom_inside_mask(R, Z, opoint, xpoint[1:])
    return mask


def geom_inside_mask(
    R: np.ndarray, Z: np.ndarray, opoint: np.ndarray, xpoint: np.ndarray
) -> np.ndarray:
    """
    Excludes grid regions based on a line perpendicular to the segment from
    the O-point to the primary X-point. Points on the O-point's side of that
    line are considered part of the plasma core.

    Parameters
    ----------
    R : np.ndarray
        Grid R coordinates, shape (nx, ny).
    Z : np.ndarray
        Grid Z coordinates, shape (nx, ny).
    opoint : np.ndarray
        O-points as rows; only the (R, Z) of the first row is used.
    xpoint : np.ndarray
        X-points as rows; only the (R, Z) of the first row (the primary X-point) is used.

    Returns
    -------
    geom_mask : ndarray of bool
        True where the grid point lies in the core (O-point) half-plane,
        False otherwise.
    """

    delta_R = opoint[0, 0] - xpoint[0, 0]
    delta_Z = opoint[0, 1] - xpoint[0, 1]
    geom_mask = (delta_R * (R - xpoint[0, 0]) + delta_Z * (Z - xpoint[0, 1])) > 0

    return geom_mask


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
    Determines the (R,Z) location on a specified magnetic flux surface by
    interpolating along a straight line in the poloidal plane (defined by (r0,z0) and (r1,z1)).

    Parameters
    ----------
    eq : object
        Equilibrium object (provides the domain limits Rmin, Rmax, Zmin, Zmax).
    psifunc : callable
        Callable function for the (normalised) flux map, called as
        ``psifunc(r, z, grid=False)``.
    r0 : float
        Starting R location inside the separatrix [m].
    z0 : float
        Starting Z location inside the separatrix [m].
    r1 : float
        Final R location outside the separatrix [m].
    z1 : float
        Final Z location outside the separatrix [m].
    psival : float, optional
        Value of the flux (in the units of `psifunc`) to find the surface of.
    n : int, optional
        Number of starting points to use along the line.
    axis : object or None, optional
        Matplotlib axis object, used for plotting.

    Returns
    -------
    r : float
        R coordinate of the point on the flux surface [m].
    z : float
        Z coordinate of the point on the flux surface [m].
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
    ntheta: int = 20,
    axis: object | None = None,
    psival: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """
    Determines the (R,Z) locations (at 'ntheta' points) of the last
    closed flux surface (i.e. the separatrix). The 'ntheta' points are equally spaced
    in the geometrix poloidal angle.

    Parameters
    ----------
    eq : object
        Equilibrium object.
    ntheta : int, optional
        Number of points to spawn in the geometric poloidal angle.
    axis : object or None, optional
        Matplotlib axis object, used for plotting.
    psival : float, optional
        Normalised flux value (0 at the axis, 1 at the boundary) of the surface to find.

    Returns
    -------
    list of tuple
        One tuple (R, Z, R_X, Z_X) per poloidal angle: the point on the plasma
        boundary (last closed flux surface) and the (R, Z) of the primary X-point.
    """

    psi = eq.psi()

    # if recalculate_equilibrum:
    #     opoint, xpoint = find_critical(eq.R, eq.Z, psi)
    # else:
    try:
        opoint = eq.opt
        xpoint = eq.xpt
        psi_boundary = eq.psi_bndry
    except AttributeError:
        warnings.warn(
            "The equilibrium object does not have the critical points stored. Recalculating. If a limiter equilibrium, solve first for the limiter separatrix."
        )
        opoint, xpoint = find_critical(eq.R, eq.Z, psi)
        psi_boundary = xpoint[0][2]

    psinorm = (psi - opoint[0][2]) / (psi_boundary - opoint[0][2])

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
        warnings.warn("Theta grid too close to X-point, shifting by half-step")
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
    psinorm: float | np.ndarray | list[float] | None = None,
    ntheta: int = 128,
    psi: np.ndarray | None = None,
    opoint: np.ndarray | list | None = None,
    xpoint: np.ndarray | list | None = None,
    axis: object | None = None,
) -> np.ndarray:
    """
    Determines the safety factor profile for a given equilbirium. Points on
    each flux surface are equally paced in poloidal angle. The function performs a line
    integral around each flux surface to get q (on that flux surface).

    Parameters
    ----------
    eq : object
        Equilibrium object.
    npsi : int, optional
        Number of flux surface values to find q for. Overridden by the length of
        `psinorm` if given, or set to 100 if `psinorm` is None.
    psinorm : float or np.ndarray or list of float or None, optional
        The normalised flux surface value(s) to calculate q for. If None, 100
        surfaces are used.
    ntheta : int, optional
        Number of poloidal points to find q values on.
    psi : np.ndarray or None, optional
        Total poloidal flux [Webers/2pi], shape (nx, ny). Defaults to ``eq.psi()``.
    opoint : np.ndarray or list or None, optional
        The O-point tuples/rows (R,Z,psi). Calculated if `opoint` or `xpoint` is None.
    xpoint : np.ndarray or list or None, optional
        The X-point tuples/rows (R,Z,psi). Calculated if `opoint` or `xpoint` is None.
    axis : object or None, optional
        Matplotlib axis object, used for plotting.

    Returns
    -------
    np.ndarray
        Returns the safety factor at each of the normalised psi values.
    """

    if psi is None:
        psi = eq.psi()

    if (opoint is None) or (xpoint is None):
        opoint, xpoint = find_critical(eq.R, eq.Z, psi)

    if (xpoint is None) or (len(xpoint) == 0):
        # No X-point - e.g. a limiter-bounded plasma
        psi_boundary = eq.psi_bndry
    else:
        psi_boundary = xpoint[0][2]

    psinormal = (psi - opoint[0][2]) / (psi_boundary - opoint[0][2])

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
        warnings.warn("Theta grid too close to X-point, shifting by half-step")
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
