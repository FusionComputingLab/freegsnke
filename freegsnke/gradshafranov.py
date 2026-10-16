"""
Contains various classes and functions related to the elliptic operator
of the Grad-Shafranov equation.

Copyright 2026 Ben Dudson, Tomas Rubio Cruz

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

from abc import ABC, abstractmethod
from warnings import warn

import numexpr as ne
import numpy as np
from numpy import pi
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, eye

from .parallel import threaded_clip, threaded_elliptics_ek

# magnetic permeability of free space
mu0 = 4e-7 * pi


class GSElliptic:
    """
    Class representing the elliptc operator within the Grad-Shafranov
    equation:

        Δ^* = d^2/dR^2 + d^2/dZ^2 - (1/R)*d/dR

        where:
         -  R is the radial coordinate.
         -  Z is the vertical coordinate.

    """

    def __init__(self, Rmin: float, *, dtype: type = np.float64) -> None:
        """
        Initializes the class.

        Parameters
        ----------
        Rmin : float
            Minimum major radius [m].
        dtype : numpy dtype, optional
            Data type of the output array (default: np.float64).

        """

        self.Rmin = Rmin
        self.dtype = dtype

    def __call__(self, psi: np.ndarray, dR: float, dZ: float) -> np.ndarray:
        """
        Apply the elliptic operator to the flux function such that:

            (Δ^*) * ψ.

        Computes to second-order accuracy.

        Parameters
        ----------
        psi : np.array
            The total poloidal flux at each (R,Z) grid point [Webers/2pi].
        dR : float
            Radial grid size [m].
        dZ : float
            Vertical grid size [m].

        Returns
        -------
        np.array
            The operator applied to the total poloidal flux.
        """

        # number of radial and vertical grid points
        nx = psi.shape[0]
        ny = psi.shape[1]

        # to store output
        b = np.zeros([nx, ny], dtype=self.dtype)

        # pre-compute constants
        invdR2 = 1.0 / dR**2
        invdZ2 = 1.0 / dZ**2

        # compute the operator (can be vectorised)
        for x in range(1, nx - 1):
            R = self.Rmin + dR * x  # Major radius of this point
            for y in range(1, ny - 1):
                # Loop over points in the domain
                b[x, y] = (
                    psi[x, y - 1] * invdZ2
                    + (invdR2 + 1.0 / (2.0 * R * dR)) * psi[x - 1, y]
                    - 2.0 * (invdR2 + invdZ2) * psi[x, y]
                    + (invdR2 - 1.0 / (2.0 * R * dR)) * psi[x + 1, y]
                    + psi[x, y + 1] * invdZ2
                )
        return b

    def diag(self, dR: float, dZ: float) -> float:
        """
        Computes:

            -2 * ( (1/dR^2) + (1/dZ^2) ).

        Parameters
        ----------
        dR : float
            Radial grid size [m].
        dZ : float
            Vertical grid size [m].

        Returns
        -------
        float
            The value above.
        """
        return -2.0 / dR**2 - 2.0 / dZ**2


class GSsparse:
    """
    Class representing the elliptic operator within the Grad-Shafranov
    equation:

        Δ^* = d^2/dR^2 + d^2/dZ^2 - (1/R)*d/dR

        where:
         -  R is the radial coordinate.
         -  Z is the vertical coordinate.

    This class calculates the sparse version to second-order accuracy.

    """

    order = 2

    def __init__(
        self,
        Rmin: float,
        Rmax: float,
        Zmin: float,
        Zmax: float,
        dtype: type = np.float64,
    ) -> None:
        """
        Initializes the class.

        Parameters
        ----------
        Rmin : float
            Minimum major radius [m].
        Rmax : float
            Maximum major radius [m].
        Zmin : float
            Minimum height [m].
        Zmax : float
            Maximum height [m].
        dtype : numpy dtype, optional
            Data type of the matrix entries (default: np.float64).

        """

        # set parameters
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

        self.dtype = dtype

    def __call__(self, nx: int, ny: int) -> csr_array | csr_matrix:
        """
        Equivalent to `discretize(nx, ny)` with the default (CSR) format.

        Parameters
        ----------
        nx : int
            Number of radial grid points (including boundaries).
        ny : int
            Number of vertical grid points (including boundaries).

        Returns
        -------
        scipy.sparse.csr_array or scipy.sparse.csr_matrix
            The operator matrix, in CSR format.
        """
        return self.discretize(nx, ny)

    def discretize(
        self, nx: int, ny: int, format: str = "csr"
    ) -> csr_array | csc_array | csr_matrix | csc_matrix:
        """
        Generates the sparse elliptic operator Δ^* for a given number of
        grid points. Computes to second-order accuracy.

        Parameters
        ----------
        nx : int
            Number of radial grid points (including boundaries).
        ny : int
            Number of vertical grid points (including boundaries).
        format : str, optional
            Sparse format of the returned matrix, either "csr" (default) or "csc".
            Any other value falls back to "csr" with a warning.

        Returns
        -------
        scipy.sparse matrix/array
            The operator matrix, in the requested sparse format.
        """

        # This method could be optimized by vectorizing with a COO approach
        # It may not be worth it though, considering the DST solver will always be faster

        # calculate grid spacing
        dR = (self.Rmax - self.Rmin) / (nx - 1)
        dZ = (self.Zmax - self.Zmin) / (ny - 1)

        # total number of points
        N = nx * ny

        # create a linked list sparse matrix
        A = eye(N, format="lil", dtype=self.dtype)

        # pre-compute constants
        invdR2 = 1.0 / dR**2
        invdZ2 = 1.0 / dZ**2

        # generate the operator values (can be optimsied)
        for x in range(1, nx - 1):
            R = self.Rmin + dR * x  # Major radius of this point
            for y in range(1, ny - 1):
                # Loop over points in the domain
                row = x * ny + y

                # y-1
                A[row, row - 1] = invdZ2

                # x-1
                A[row, row - ny] = invdR2 + 1.0 / (2.0 * R * dR)

                # diagonal
                A[row, row] = -2.0 * (invdR2 + invdZ2)

                # x+1
                A[row, row + ny] = invdR2 - 1.0 / (2.0 * R * dR)

                # y+1
                A[row, row + 1] = invdZ2

        # convert to Compressed Sparse Row (CSR) format

        if format == "csc":
            A = A.tocsc()
        elif format == "csr":
            A = A.tocsr()
        else:
            warn("Unrecognized format for sparse matrix, defaulting to csr")
            A = A.tocsr()

        return A


class GSsparse4thOrder(GSsparse):
    """
    Class representing the elliptic operator within the Grad-Shafranov
    equation:

        Δ^* = d^2/dR^2 + d^2/dZ^2 - (1/R)*d/dR

        where:
         -  R is the radial coordinate.
         -  Z is the vertical coordinate.

    This class calculates the sparse version to fourth-order accuracy.

    """

    order = 4

    # Coefficients for first derivatives
    # (index offset, weight)

    centred_1st = [
        (-2, 1.0 / 12),
        (-1, -8.0 / 12),
        (1, 8.0 / 12),
        (2, -1.0 / 12),
    ]

    offset_1st = [
        (-1, -3.0 / 12),
        (0, -10.0 / 12),
        (1, 18.0 / 12),
        (2, -6.0 / 12),
        (3, 1.0 / 12),
    ]

    # Coefficients for second derivatives
    # (index offset, weight)
    centred_2nd = [
        (-2, -1.0 / 12),
        (-1, 16.0 / 12),
        (0, -30.0 / 12),
        (1, 16.0 / 12),
        (2, -1.0 / 12),
    ]

    offset_2nd = [
        (-1, 10.0 / 12),
        (0, -15.0 / 12),
        (1, -4.0 / 12),
        (2, 14.0 / 12),
        (3, -6.0 / 12),
        (4, 1.0 / 12),
    ]

    def __call__(self, nx: int, ny: int) -> csr_array | csr_matrix:
        """
        Equivalent to `discretize(nx, ny)` with the default (CSR) format.

        Parameters
        ----------
        nx : int
            Number of radial grid points (including boundaries).
        ny : int
            Number of vertical grid points (including boundaries).

        Returns
        -------
        scipy.sparse.csr_array or scipy.sparse.csr_matrix
            The operator matrix, in CSR format.
        """
        return self.discretize(nx, ny)

    def discretize(
        self, nx: int, ny: int, format: str = "csr"
    ) -> csr_array | csc_array | csr_matrix | csc_matrix:
        """
        Generates the sparse elliptic operator Δ^* for a given number of
        grid points. Computes to fourth-order accuracy.

        Parameters
        ----------
        nx : int
            Number of radial grid points (including boundaries).
        ny : int
            Number of vertical grid points (including boundaries).
        format : str, optional
            Sparse format of the returned matrix, either "csr" (default) or "csc".
            Any other value falls back to "csr" with a warning.

        Returns
        -------
        scipy.sparse matrix/array
            The operator matrix, in the requested sparse format.

        Raises
        ------
        KeyError
            If an unrecognised derivative operator is encountered while
            assembling the stencil (internal error).
        """

        # calculate grid spacing
        dR = (self.Rmax - self.Rmin) / (nx - 1)
        dZ = (self.Zmax - self.Zmin) / (ny - 1)

        # total number of points, including boundaries
        N = nx * ny

        # calculate constants
        invdR2 = 1.0 / dR**2
        invdZ2 = 1.0 / dZ**2

        # The GS operator is constructed in COO format using lists of ndarrays. Each ndarray contains
        # the row/col/value of the entries for a given stencil.
        rows = []
        cols = []
        entries = []

        # set boundary entries

        for x in (0, nx - 1):
            y = np.arange(ny)

            row = x * ny + y
            entry = np.ones_like(row, dtype=self.dtype)

            rows.append(row)
            cols.append(row)  # no offset for dirichlet bc
            entries.append(entry)

        for y in (0, ny - 1):
            x = np.arange(1, nx - 1)  # x=0, x=(nx-1) were already set above

            row = x * ny + y
            entry = np.ones_like(row, dtype=self.dtype)

            rows.append(row)
            cols.append(row)  # no offset for dirichlet bc
            entries.append(entry)

        # set near-boundary entries (need offset stencils)

        # d^2 / dR^2 - (1/R) d/dR
        for x, offsign in zip(
            (1, nx - 2), (1, -1)
        ):  # offset on right (nx-2) has negative sign

            y = np.arange(1, ny - 1)
            R = self.Rmin + dR * x  # major radius of each point

            row = x * ny + y

            for offset, weight in self.offset_2nd:

                col = row + offsign * offset * ny
                entry = weight * invdR2
                entry = np.full_like(row, entry, dtype=self.dtype)

                rows.append(row)
                cols.append(col)
                entries.append(entry)

            for offset, weight in self.offset_1st:

                col = row + offsign * offset * ny
                entry = (
                    -offsign * weight / (R * dR)
                )  # sign of entry depends on direction (offsign)
                entry = np.full_like(row, entry, dtype=self.dtype)

                rows.append(row)
                cols.append(col)
                entries.append(entry)

        # d^2 / dZ^2
        for y, offsign in zip(
            (1, ny - 2), (1, -1)
        ):  # offset on top (ny-2) has negative sign

            x = np.arange(1, nx - 1)
            row = x * ny + y

            for offset, weight in self.offset_2nd:

                col = row + offsign * offset
                entry = weight * invdZ2
                entry = np.full_like(row, entry, dtype=self.dtype)

                rows.append(row)
                cols.append(col)
                entries.append(entry)

        # set internal entries (use centred stencil)

        # build the largest rectangle in domain with only centred-scheme entries
        y_vals = np.arange(2, ny - 2)
        x_vals = np.arange(2, nx - 2)
        R = self.Rmin + dR * x_vals  # major radius of each point

        # 2d grid with row no. of each (x,y)
        row_2d = x_vals[:, np.newaxis] * ny + y_vals[np.newaxis, :]

        # iterate over differential operators:
        # longitudinal --> d^2/dZ^2; radial_1 --> d^2/dR^2; radial_2 --> -(1/R) d/dR

        operators = ["longitudinal", "radial_1", "radial_2"]
        stencils = {
            "longitudinal": self.centred_2nd,
            "radial_1": self.centred_2nd,
            "radial_2": self.centred_1st,
        }
        offscales = {"longitudinal": 1, "radial_1": ny, "radial_2": ny}

        for op in operators:

            stencil = stencils[op]
            offscale = offscales[op]

            size_stencil = len(stencil)
            stencil_offsets = offscale * np.array([entry[0] for entry in stencil])
            stencil_weights = np.array([entry[1] for entry in stencil])

            # 3d grids with the row/col/weight corresponding to each x,y,offset combo
            row = np.broadcast_to(
                row_2d[:, :, np.newaxis], (*row_2d.shape, size_stencil)
            )
            col = row + stencil_offsets[np.newaxis, np.newaxis, :]
            weights = np.broadcast_to(
                stencil_weights[np.newaxis, np.newaxis, :], row.shape
            )

            entry = None

            if op == "longitudinal":
                # d^2 / dZ^2
                entry = weights * invdZ2
            elif op == "radial_1":
                # d^2 / dR^2
                entry = weights * invdR2
            elif op == "radial_2":
                # -(1/R) d/dR
                entry = -weights / (R[:, np.newaxis, np.newaxis] * dR)
            else:
                raise KeyError(op)

            rows.append(row.flatten())
            cols.append(col.flatten())
            entries.append(entry.flatten())

            if op == "longitudinal":
                # set stencil for points where a centred scheme is used longitudinally
                # but not radially
                for x in (1, nx - 2):
                    y = np.arange(2, ny - 2)

                    row = x * ny + y
                    row = np.broadcast_to(
                        row[:, np.newaxis], (*row.shape, size_stencil)
                    )
                    col = row + stencil_offsets[np.newaxis, :]
                    weights = np.broadcast_to(stencil_weights[np.newaxis, :], row.shape)

                    entry = weights * invdZ2

                    rows.append(row.flatten())
                    cols.append(col.flatten())
                    entries.append(entry.flatten())

            else:
                # set stencil for points where a centred scheme is used radially
                # but not longitudinally
                for y in (1, ny - 2):
                    x = np.arange(2, nx - 2)

                    row = x * ny + y
                    row = np.broadcast_to(
                        row[:, np.newaxis], (*row.shape, size_stencil)
                    )
                    col = row + stencil_offsets[np.newaxis, :]
                    weights = np.broadcast_to(stencil_weights[np.newaxis, :], row.shape)

                    entry = None
                    if op == "radial_1":
                        entry = weights * invdR2
                    elif op == "radial_2":
                        entry = -weights / (R[:, np.newaxis] * dR)
                    else:
                        raise KeyError(op)

                    rows.append(row.flatten())
                    cols.append(col.flatten())
                    entries.append(entry.flatten())

        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_entries = np.concatenate(entries)

        A = None

        if format == "csc":
            A = csc_array((all_entries, (all_rows, all_cols)), dtype=self.dtype)
        elif format == "csr":
            A = csr_array((all_entries, (all_rows, all_cols)), dtype=self.dtype)
        else:
            warn("Unrecognized format for sparse matrix, defaulting to csr")
            A = csr_array((all_entries, (all_rows, all_cols)), dtype=self.dtype)

        return A


def Greens(
    Rc: float | np.ndarray,
    Zc: float | np.ndarray,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    limit_threading: bool = False,
    scale_factor: float = 1,
    out: np.ndarray | None = None,
) -> float | np.ndarray:
    """
    Calculate poloidal flux at (R,Z) due to a single unit of current at
    (Rc,Zc) using Greens function for the elliptic operator above. Greens
    function is given by:

        G(R, Z; Rc, Zc) = (μ0 / (2π)) * sqrt(R * Rc) * ((2 - k^2) * K(k^2) - 2 * E(k^2)) / k

    where:
     - k^2 = 4 R Rc / ((R + Rc)^2 + (Z - Zc)^2)
     - k = sqrt(k^2)

    and K(k^2) and E(k^2) are the complete elliptic integrals of the first
    and second kind.

    This function is multithreaded using numexpr and freegsnke's custom parallel library. For
    details consult the relevant section in the README.

    Parameters
    ----------
    Rc : float or np.ndarray
        Radial position where current is located [m].
    Zc : float or np.ndarray
        Vertical position where current is located [m].
    R : float or np.ndarray
        Radial position where poloidal flux is to be calculated [m].
    Z : float or np.ndarray
        Vertical position where poloidal flux is to be calculated [m].
    limit_threading : bool, optional
        If True, forces SOME internal functions, with high threading overhead, to run single
        threaded. Multiple threads will still be used for low overhead functionalities.
        Recommended when the input arrays are small (default: False).
    scale_factor : float, optional
        Scalar factor to apply on the final result (default: 1).
    out : np.ndarray, optional
        Pre-allocated buffer for the final result.

    Returns
    -------
    float or np.ndarray
        Value of the poloidal flux at (R,Z) [Webers/2pi].
    """

    # TODO: the next version of numexpr should allow for cache_disabling, this could
    # help address memory issues.

    # calculate k^2
    k2 = ne.evaluate("4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)")

    # clip to between 0 and 1 to avoid nans e.g. when coil is on grid point
    k2 = threaded_clip(k2, 1e-10, 1.0 - 1e-10, out=k2, single_thread=limit_threading)

    # note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
    eie, eik = threaded_elliptics_ek(k2, single_thread=limit_threading)

    ct = (mu0 / (2.0 * pi)) * scale_factor

    res = ne.evaluate(
        "ct * sqrt(R * Rc) * ((2.0 - k2) * eik - 2.0 * eie) / sqrt(k2)",
        out=out,
    )

    return res


def GreensBz(
    Rc: float | np.ndarray,
    Zc: float | np.ndarray,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    eps: float = 1e-4,
) -> float | np.ndarray:
    """
    Calculate vertical magnetic field at (R,Z) due to a single unit of current at
    (Rc,Zc) using Greens function for the elliptic operator above.

        Bz(R,Z) = (1/R) d psi/dR,

    where psi is found with the Greens function finite difference.

    Parameters
    ----------
    Rc : float or np.ndarray
        Radial position where current is located [m].
    Zc : float or np.ndarray
        Vertical position where current is located [m].
    R : float or np.ndarray
        Radial position where the field is evaluated [m].
    Z : float or np.ndarray
        Vertical position where the field is evaluated [m].
    eps : float, optional
        Small step size for numerical differentiation in the radial direction [m].

    Returns
    -------
    float or np.ndarray
        Value of the vertical magnetic field at (R,Z) [T].
    """

    return (Greens(Rc, Zc, R + eps, Z) - Greens(Rc, Zc, R - eps, Z)) / (2.0 * eps * R)


def GreensBr(
    Rc: float | np.ndarray,
    Zc: float | np.ndarray,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    eps: float = 1e-4,
) -> float | np.ndarray:
    """
    Calculate radial magnetic field at (R,Z) due to a single unit of current at
    (Rc,Zc) using Greens function for the elliptic operator above.

        Br(R,Z) = -(1/R) d psi/dZ,

    where psi is found with the Greens function finite difference.

    Parameters
    ----------
    Rc : float or np.ndarray
        Radial position where current is located [m].
    Zc : float or np.ndarray
        Vertical position where current is located [m].
    R : float or np.ndarray
        Radial position where the field is evaluated [m].
    Z : float or np.ndarray
        Vertical position where the field is evaluated [m].
    eps : float, optional
        Small step size for numerical differentiation in the vertical direction [m].

    Returns
    -------
    float or np.ndarray
        Value of the radial magnetic field at (R,Z) [T].
    """

    return (Greens(Rc, Zc, R, Z - eps) - Greens(Rc, Zc, R, Z + eps)) / (2.0 * eps * R)


def GreensdBzdr(
    Rc: float | np.ndarray,
    Zc: float | np.ndarray,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    eps: float = 2e-3,
) -> float | np.ndarray:
    """
    Calculate radial derivative of vertical magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBz/dR (R,Z) = (Bz(R + eps, Z) - Bz(R - eps, Z))/ (2 * eps).

    Parameters
    ----------
    Rc : float or np.ndarray
        Radial position where current is located [m].
    Zc : float or np.ndarray
        Vertical position where current is located [m].
    R : float or np.ndarray
        Radial position where the field is evaluated [m].
    Z : float or np.ndarray
        Vertical position where the field is evaluated [m].
    eps : float, optional
        Small step size for numerical differentiation in the radial direction [m].

    Returns
    -------
    float or np.ndarray
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBz(Rc, Zc, R + eps, Z) - GreensBz(Rc, Zc, R - eps, Z)) / (2.0 * eps)


def GreensdBrdz(
    Rc: float | np.ndarray,
    Zc: float | np.ndarray,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    eps: float = 2e-3,
) -> float | np.ndarray:
    """
    Calculate vertical derivative of radial magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBr/dZ (R,Z) = (Br(R, Z + eps) - Br(R, Z - eps))/ (2 * eps).

    Parameters
    ----------
    Rc : float or np.ndarray
        Radial position where current is located [m].
    Zc : float or np.ndarray
        Vertical position where current is located [m].
    R : float or np.ndarray
        Radial position where the field is evaluated [m].
    Z : float or np.ndarray
        Vertical position where the field is evaluated [m].
    eps : float, optional
        Small step size for numerical differentiation in the vertical direction [m].

    Returns
    -------
    float or np.ndarray
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBr(Rc, Zc, R, Z + eps) - GreensBr(Rc, Zc, R, Z - eps)) / (2.0 * eps)
    # return GreensdBzdr(Rc, Zc, R, Z, eps)


def GreensdBzdz(
    Rc: float | np.ndarray,
    Zc: float | np.ndarray,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    eps: float = 2e-3,
) -> float | np.ndarray:
    """
    Calculate vertical derivative of vertical magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBz/dZ (R,Z) = (Bz(R, Z + eps) - Bz(R, Z - eps)) / (2 * eps).

    Parameters
    ----------
    Rc : float or np.ndarray
        Radial position where current is located [m].
    Zc : float or np.ndarray
        Vertical position where current is located [m].
    R : float or np.ndarray
        Radial position where the field is evaluated [m].
    Z : float or np.ndarray
        Vertical position where the field is evaluated [m].
    eps : float, optional
        Small step size for numerical differentiation in the vertical direction [m].

    Returns
    -------
    float or np.ndarray
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBz(Rc, Zc, R, Z + eps) - GreensBz(Rc, Zc, R, Z - eps)) / (2.0 * eps)


def GreensdBrdr(
    Rc: float | np.ndarray,
    Zc: float | np.ndarray,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    eps: float = 2e-3,
) -> float | np.ndarray:
    """
    Calculate radial derivative of radial magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBr/dR (R,Z) = (Br(R + eps, Z) - Br(R - eps, Z))/ (2 * eps).

    Parameters
    ----------
    Rc : float or np.ndarray
        Radial position where current is located [m].
    Zc : float or np.ndarray
        Vertical position where current is located [m].
    R : float or np.ndarray
        Radial position where the field is evaluated [m].
    Z : float or np.ndarray
        Vertical position where the field is evaluated [m].
    eps : float, optional
        Small step size for numerical differentiation in the radial direction [m].

    Returns
    -------
    float or np.ndarray
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBr(Rc, Zc, R + eps, Z) - GreensBr(Rc, Zc, R - eps, Z)) / (2.0 * eps)


def GreensdBrdz(
    Rc: float | np.ndarray,
    Zc: float | np.ndarray,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    eps: float = 2e-3,
) -> float | np.ndarray:
    """
    Calculate vertical derivative of radial magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBr/dZ (R,Z) = (Br(R, Z + eps) - Br(R, Z - eps))/ (2 * eps).

    Parameters
    ----------
    Rc : float or np.ndarray
        Radial position where current is located [m].
    Zc : float or np.ndarray
        Vertical position where current is located [m].
    R : float or np.ndarray
        Radial position where the field is evaluated [m].
    Z : float or np.ndarray
        Vertical position where the field is evaluated [m].
    eps : float, optional
        Small step size for numerical differentiation in the vertical direction [m].

    Returns
    -------
    float or np.ndarray
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBr(Rc, Zc, R, Z + eps) - GreensBr(Rc, Zc, R, Z - eps)) / (2.0 * eps)


def GreensdBzdr(
    Rc: float | np.ndarray,
    Zc: float | np.ndarray,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    eps: float = 2e-3,
) -> float | np.ndarray:
    """
    Calculate radial derivative of vertical magnetic field at (R,Z) due to a
    single unit of current at (Rc,Zc) using Greens function for the
    elliptic operator above:

        dBz/dR (R,Z) = (Bz(R + eps, Z) - Bz(R - eps, Z))/ (2 * eps).

    Parameters
    ----------
    Rc : float or np.ndarray
        Radial position where current is located [m].
    Zc : float or np.ndarray
        Vertical position where current is located [m].
    R : float or np.ndarray
        Radial position where the field is evaluated [m].
    Z : float or np.ndarray
        Vertical position where the field is evaluated [m].
    eps : float, optional
        Small step size for numerical differentiation in the radial direction [m].

    Returns
    -------
    float or np.ndarray
        Value of the derivative at (R,Z) [T/m].
    """

    return (GreensBz(Rc, Zc, R + eps, Z) - GreensBz(Rc, Zc, R - eps, Z)) / (2.0 * eps)
