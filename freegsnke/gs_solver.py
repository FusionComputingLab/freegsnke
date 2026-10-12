"""
Defines a base class for all solvers for the GS equation, as well as some
specific subclasses.

Copyright 2026 Tomas Rubio Cruz, Ubaid Qadri

This file is part of FreeGS4E.

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

import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.fft import dst
from scipy.linalg import solve_banded
from scipy.sparse.linalg import factorized

from .gradshafranov import GSsparse, GSsparse4thOrder


class GSSolver(ABC):
    """
    Parent class for all GS solvers. Enforces the definition of a solve method and its use
    when calling the solver directly, as well as ensuring that the solution from call is
    in the right shape.

    In addition to the abstract methods below, all instances should initialize `self.dimensions`,
    the dimensions of the grid where the discrete problem is solved.
    """

    # TODO: __call__ and solve receive xi as an argument, despite it not being necessary
    # for most solvers. Should consider ways to modify how this argument is handled.

    @abstractmethod
    def __init__(self, R, Z, *, order=2, dtype=np.float64):
        """
        Recommended initialization signature for subclasses.

        Parameters
        ----------
        R, Z: ndarray (nr,nz)
            grids with the radius and height of each point in the domain
        order: int
            order of the finite difference approximation to the GS operator
        dtype: dtype
            datatype to use, per numpy conventions
        """
        pass

    def __call__(self, xi, rhs, **kwargs):
        """
        Calls the solver and returns the solution in the shape of the given rhs.

        Parameters
        ----------
        xi: ndarray
            initial guess (may be unused by the solver)
        rhs: ndarray
            right-hand-side of the GS equation, of shape `self.dimensions`
        **kwargs:
            passed on to `solve`

        Returns
        -------
        ndarray
            solution psi, with the shape of rhs

        Raises
        ------
        ValueError
            If the shape of rhs does not match the grid of the solver.
        """

        if rhs.shape != self.dimensions:
            raise ValueError(
                f"shape mismatch: rhs shape {rhs.shape} does not match solver grid shape {self.dimensions}"
            )

        psi = self.solve(xi, rhs, **kwargs)
        return psi.reshape(rhs.shape)

    @abstractmethod
    def solve(self, xi, rhs, **kwargs):
        """
        Solves the GS equation for the given rhs and returns psi (possibly flattened,
        `__call__` restores the shape).
        """
        pass


class GSLUSolver(GSSolver):
    """
    Sparse-LU solver for the Grad-Shafranov equation on a rectangular (R,Z) grid with
    equally spaced points. Solves:

    (d^2/dR^2 + d^2/dZ^2 - (1/R)*d/dR) ψ = -μ_0 R J_{phi}(ψ),

    with the Dirichlet (free-)boundary conditions applied.

    Instances are called as `solver(xi, rhs)` (see `GSSolver`); the solution has the
    shape of `rhs`.

    """

    def __init__(self, R, Z, *, order=4, dtype=np.float64):
        """
        Initialize the solver with the given setup. R and Z define the grid (domain size and resolution).

        Parameters
        ----------
        R: ndarray (nr,nz)
            ndarray of the shape of the domain (nr,nz) with the radius of each point in the grid
        Z: ndarray (nr,nz)
            ndarray of the shape of the domain (nr,nz) with the height of each point in the grid
        order: int
            order of the finite difference approximation to use for the GS operator (2 or 4, default 4)
        dtype: dtype
            datatype to use, per numpy conventions

        Raises
        ------
        ValueError
            If R and Z have different shapes, or `order` is not 2 or 4.
        """

        if R.shape != Z.shape:
            raise ValueError(
                f"shape mismatch: Shapes of radial grid ({R.shape}) and longitudinal grid ({Z.shape}) do not match"
            )

        Rmin = float(R[0, 0])
        Rmax = float(R[-1, 0])
        Zmin = float(Z[0, 0])
        Zmax = float(Z[0, -1])

        operator = None

        if order == 2:
            operator = GSsparse(Rmin, Rmax, Zmin, Zmax, dtype=dtype)
        elif order == 4:
            operator = GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax, dtype=dtype)
        else:
            raise ValueError(
                f"Cannot create sparse GS operator of order {order}. Only 2 or 4 supported."
            )

        self.dtype = dtype
        self.order = order
        nx, ny = R.shape
        self.dimensions = (nx, ny)

        operator_matrix = operator.discretize(nx, ny, format="csc")
        self.solver = factorized(operator_matrix)

    def solve(self, xi, rhs):
        """
        Solves the Grad-Shafranov equation with the given rhs.

        Parameters
        ----------
        xi: ndarray
            unused, present for cross-compatibility with other solvers
        rhs: ndarray
            right-hand-side of the GS equation (f)

        Returns
        -------
        psi
            value of psi obtained for the given rhs

        """

        return self.solver(rhs.reshape(-1))


class GSDSTSolver(GSSolver):
    """
    Discrete Sine Transform (DST) based solver for the Grad-Shafranov equation on a rectangular (R,Z) grid with
    equally spaced points. Solves:

    (d^2/dR^2 + d^2/dZ^2 - (1/R)*d/dR) ψ = -μ_0 R J_{phi}(ψ),

    with the Dirichlet (free-)boundary conditions applied.

    Instances are called as `solver(xi, rhs)` (see `GSSolver`); the solution has the
    shape of `rhs`.

    """

    order = 2

    def __init__(self, R, Z, *, order=2, dtype=np.float64):
        """
        Initialize the solver with the given setup. R and Z define the grid (domain size and resolution).

        Parameters
        ----------
        R: ndarray (nr,nz)
            ndarray of the shape of the domain (nr,nz) with the radius of each point in the grid
        Z: ndarray (nr,nz)
            ndarray of the shape of the domain (nr,nz) with the height of each point in the grid
        order: int
            unused: the DST solver is always second order. A warning is issued if a different
            value is given.
        dtype: dtype
            datatype to use, per numpy conventions

        Raises
        ------
        ValueError
            If R and Z have different shapes, or the grid has fewer than 3 points in R or Z.
        """

        if R.shape != Z.shape:
            raise ValueError(
                f"shape mismatch: Shapes of radial grid ({R.shape}) and longitudinal grid ({Z.shape}) do not match"
            )
        if order != 2:
            warnings.warn(
                "DST solver only supports order 2. Provided value ignored."
            )

        self.Rmin = float(R[0, 0])
        self.Rmax = float(R[-1, 0])
        self.Zmin = float(Z[0, 0])
        self.Zmax = float(Z[0, -1])
        self.R = np.ascontiguousarray(R, dtype=dtype)
        self.Z = np.ascontiguousarray(Z, dtype=dtype)

        self.dimensions = self.R.shape
        self.dtype = dtype

        # Uniform spacings (assumed)
        self.dR = float(R[1, 0] - R[0, 0])
        self.dZ = float(Z[0, 1] - Z[0, 0])

        # Precompute batch tridiagonal diagonals and Z-eigenvalues
        self.__init_radial_operator()

    def _dst1(self, x):
        """Apply DST-I (orthonormal) along the last axis; function serves as its own inverse."""
        return dst(x, type=1, axis=-1, norm="ortho")

    def __init_radial_operator(self):
        """
        Initializes the FD operator for the radial component of the Grad-Shafranov operator
        d2/dR2 - (1/R)d/dR, in scipy banded format.
        """

        R = self.R[:, 0]  # vector of R values (not grid)
        nr, nz = self.R.shape
        Nint = nz - 2

        if Nint < 1:
            raise ValueError(
                f"Need at least 3 Z points to have an interior. Current number: {nz}"
            )
        elif nr < 3:
            raise ValueError(
                f"Need at least 3 R points for the discretization scheme. Current number: {nr}"
            )

        # Z eigenvalues mu_m > 0 for Dirichlet interior FD Laplacian

        m = np.arange(1, Nint + 1, dtype=self.dtype)  # Fourier modes

        self.mu = (2.0 / self.dZ**2) * (1.0 - np.cos(m * np.pi / (Nint + 1)))

        # R-direction FD operator for (d2/dR2 - (1/R)d/dR)

        invdR2 = 1.0 / self.dR**2

        # Diagonal (x_i)
        main = np.full(nr, -2.0 * invdR2, dtype=self.dtype)

        # Lower diagonal ( x_(i-1) )
        sub = invdR2 + 1.0 / (2.0 * R[1:] * self.dR)

        # Upper diagonal ( x_(i+1) )
        sup = invdR2 - 1.0 / (2.0 * R[:-1] * self.dR)

        # Impose Dirichlet at R boundaries by fixing rows:
        # main[0] = 1, main[-1] = 1, and ensure no coupling outside:
        main[0] = 1.0
        main[-1] = 1.0
        sub[-1] = 0.0  # nothing below last row
        sup[0] = 0.0  # nothing above first row

        # scipy-format banded matrix (template)
        ab = np.zeros((3, nr), dtype=self.dtype)
        ab[0, 1:] = sup[:]  # upper diag (length nr-1)
        ab[1, :] = main[:]  # main diag (length nr)
        ab[2, :-1] = sub[:]  # lower diag (length nr-1)

        self.ab_template = ab

    def solve(self, xi, rhs):
        """
        Solves the Grad-Shafranov equation with the given rhs.

        Parameters
        ----------
        xi: ndarray
            unused, present for cross-compatibility with other solvers
        rhs: ndarray
            right-hand-side of the GS equation (f)

        Returns
        -------
        psi
            value of psi obtained for the given rhs

        """

        # Build "g": linear function in Z that matches Z boundary values (given rhs)
        phi0 = rhs[:, 0]  # psi at Z=Zmin
        phiL = rhs[:, -1]  # psi at Z=Zmax
        zfrac = (self.Z - self.Zmin) / (self.Zmax - self.Zmin)
        g = phi0[:, None] + (phiL - phi0)[:, None] * zfrac  # shape (nr, nz)

        # Compute Δ*g with centered scheme
        g_RR = np.zeros_like(g)
        g_RR[1:-1, :] = (g[:-2, :] - 2.0 * g[1:-1, :] + g[2:, :]) / self.dR**2

        g_ZZ = np.zeros_like(g)
        g_ZZ[:, 1:-1] = (g[:, :-2] - 2.0 * g[:, 1:-1] + g[:, 2:]) / self.dZ**2

        iR = 1.0 / (2.0 * self.R * self.dR)
        iRgR = np.zeros_like(g)
        iRgR[1:-1, :] = iR[1:-1, :] * (
            g[2:, :] - g[:-2, :]
        )  # iRgR = (1/R)*∂g/∂R

        Delta_g = g_RR + g_ZZ - iRgR

        # Modified RHS for w: Δ* w = f - Δ* g, with w = 0 at Z boundaries
        F = rhs - Delta_g
        F_int = F[:, 1:-1]  # (view of) Z-interior

        # DST-I along Z (interior only). For type-1 + ortho, inverse == forward.
        F_hat = self._dst1(F_int)  # shape (nr, Nint)

        # Transform R-boundary values for w
        w_b0_hat = self._dst1((rhs[0, 1:-1] - g[0, 1:-1]))  # (Nint,)
        w_bL_hat = self._dst1((rhs[-1, 1:-1] - g[-1, 1:-1]))  # (Nint,)

        # Instead of a single RHS for the full 2D domain, we end up with a set of 1D RHS's, one for
        # each interior Z-point (each Z-mode). Insert transformed R dimension bc into the RHS
        rhs_batch = (
            F_hat.T.copy()
        )  # transpose so that rows correspond to each Z-mode
        rhs_batch[:, 0] = w_b0_hat
        rhs_batch[:, -1] = w_bL_hat

        Nint = rhs_batch.shape[0]

        # Solve banded system for each Z-mode: (D_RR - (1/R)D_R - mu_m * I) w_hat_m = rhs_m

        w_hat_modes = np.empty_like(rhs_batch)  # (Nint, nr)

        for k in range(Nint):
            mu_k = self.mu[k]
            ab = self.ab_template.copy()
            # Shift main diagonal (ab row 1) by -mu_k (Z-separation term)
            ab[1, 1:-1] -= mu_k
            # Solve A_k * x_k = rhs_batch[k]
            w_hat_modes[k] = solve_banded((1, 1), ab, rhs_batch[k])

        # Inverse DST (same as forward) to reconstruct w on Z interior
        w_hat_int = w_hat_modes.T  # (nr, Nint)
        w_int = self._dst1(w_hat_int)

        # Assemble full w with zeros at Z boundaries
        w = np.zeros_like(rhs)
        w[:, 1:-1] = w_int

        psi = g + w
        return psi.reshape(-1)
