"""
Defines class that represents the equilibrium state and its associated
quantities.

Modified substantially from the original FreeGS code.

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

import warnings

import contourpy
import matplotlib.pyplot as plt
import numpy as np
import shapely as sh
from numpy import exp, linspace, meshgrid, pi
from scipy import interpolate
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, squareform

from . import multigrid  # prevents bugs with imports elsewhere
from . import critical, machine, polygons
from .boundary import fixedBoundary, freeBoundary  # finds free-boundary
from .gradshafranov import mu0
from .gs_solver import GSLUSolver
from .plotting import plotEquilibrium


class Equilibrium:
    """
    Represents the plasma equilibrium state and its associated properties.

    """

    def __init__(
        self,
        tokamak=None,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=freeBoundary,
        psi=None,
        current=0.0,
        order=None,
    ):
        """
        Initializes a plasma equilibrium.

        Parameters
        ----------
        tokamak : machine.Machine | None
            The set of active coils, passive structures, limiter, and magnetic probes in the tokamak.
            If `None` (default), an empty tokamak (`machine.EmptyTokamak`) is used.
        Rmin : float
            Minimum major radius [m].
        Rmax : float
            Maximum major radius [m].
        Zmin : float
            Minimum height [m].
        Zmax : float
            Maximum height [m].
        nx : int
            Number of radial grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).
        ny : int
            Number of vertical grid points (must be of form 2^n + 1, n=0,1,2,3,4,5,...).
        boundary : callable
            The boundary condition function. Can be either `freeBoundary`
            or `fixedBoundary`.
        psi : np.array | None | callable
            Initial guess for plasma flux [Webers/2pi]. If `None`, default initial Gaussian-shaped
            guess used. If a callable is provided, it is called with the equilibrium to get the initial guess.
            The array (returned by the callable) must have the shape (nx, ny).
        current : float
            Plasma current [A].
        order : int | None
            Deprecated, passing a value raises a DeprecationWarning. No solver is created at
            initialization; the value (default 4) is only used as the order of the sparse-LU
            solver that is lazily created by `callSolver`.

        """

        # assign tokamak object
        if tokamak:
            self.tokamak = tokamak
        else:
            self.tokamak = machine.EmptyTokamak()

        # assign bounds of computational domain
        if Rmin > Rmax:
            raise ValueError("Rmin must be smaller than Rmax.")
        if Zmin > Zmax:
            raise ValueError("Zmin must be smaller than Zmax.")
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

        # assign number of grid points
        if nx < 0 or ny < 0:
            raise ValueError("nx and ny must be integers greater than zero.")
        self.nx = nx
        self.ny = ny

        # define 1D and 2D radial/vertical grids
        self.R_1D = linspace(Rmin, Rmax, nx)
        self.Z_1D = linspace(Zmin, Zmax, ny)
        self.R, self.Z = meshgrid(self.R_1D, self.Z_1D, indexing="ij")

        # assign grid sizes
        self.dR = self.R[1, 0] - self.R[0, 0]
        self.dZ = self.Z[0, 1] - self.Z[0, 0]

        # assign boundary function
        self._applyBoundary = boundary

        # generate Greens function mappings (used
        # in self.psi() to speed up calculations)
        self._pgreen = self.tokamak.createPsiGreens(self.R, self.Z)
        self._vgreen = self.tokamak.createPsiGreensVec(self.R, self.Z)
        # self._updatePlasmaPsi(psi)  # Needs to be after _pgreen

        # assign plasma current
        self._current = current

        # order attribute is kept for backwards compatibility only
        if order is not None:
            warnings.warn(
                "Order attribute of Equilibrium objects is deprecated. No solver is created inside Equilibrium.",
                DeprecationWarning,
            )
        else:
            order = 4

        self._order = order
        self.__solver = None

        # assign initial guess for plasma flux (if None)
        if psi is None:
            # Starting guess for psi using Gaussian-shape
            psi_guess_callable = PsiGuessGaussian()
            psi = psi_guess_callable(self)
            self.gpars = np.array(psi_guess_callable.gpars)
        elif callable(psi):
            psi = psi(self)

        if psi.shape != (nx, ny):
            raise ValueError(
                f"Shape of psi grid {psi.shape} must match grid size ({nx}, {ny})."
            )
        self._updatePlasmaPsi(psi)

    @property
    def order(self):
        warnings.warn(
            "Order attribute of Equilibrium objects is deprecated, as no solver is created",
            DeprecationWarning,
        )
        return self._order

    @order.setter
    def order(self, value):
        warnings.warn(
            "Order attribute of Equilibrium objects is deprecated, as no solver is created",
            DeprecationWarning,
        )
        self._order = value

    @property
    def _solver(self):
        warnings.warn(
            "Solver attribute of Equilibrium objects is deprecated. Equilibrium contains no linear solver.",
            DeprecationWarning,
        )
        return self.__solver

    @_solver.setter
    def _solver(self, value):
        warnings.warn(
            "Solver attribute of Equilibrium objects is deprecated. Equilibrium contains no linear solver.",
            DeprecationWarning,
        )
        self.__solver = value

    def create_psi_plasma_default(
        self, adaptive_centre=False, gpars=(0.5, 0.5, 0, 2)
    ):
        """This method is kept for backwards compatibility and will generate a Gaussian-shaped
        initial guess for the plasma flux. It is NOT called by this class anymore.
        """
        return PsiGuessGaussian(adaptive_centre, gpars)(self)

    def setSolverVcycle(
        self, nlevels=1, ncycle=1, niter=1, direct=True, order=4
    ):
        """
        Deprecated. Equilibrium objects do not contain a multigrid solver. All arguments
        are ignored and any previously set solver is discarded, so that the default
        sparse-LU solver is (lazily) used again.
        """
        self.setSolver(None)

    def setSolver(self, solver):
        """
        Deprecated. Sets the solver used by `callSolver`. The object must be callable as
        `solver(x, b)`, where x is the initial guess and b the right hand side. Passing
        None restores the default sparse-LU solver (created lazily on next use).
        """
        self._solver = solver

    def callSolver(self, psi, rhs):
        """
        Solves the GS elliptic equation for the given right hand side. The solver
        (a sparse-LU `GSLUSolver` of the order given at initialization) is created
        lazily on first use, so that constructing an Equilibrium carries no
        factorization cost.

        Parameters
        ----------
        psi : np.array
            The initial guess [Webers/2pi] (unused by the LU solver).
        rhs : np.array
            The right hand side of the GS equation.

        Returns
        -------
        np.array
            The solution of the GS equation, with the shape of `rhs`.
        """

        if self.__solver is None:
            self.__solver = GSLUSolver(self.R, self.Z, order=self._order)

        return self.__solver(psi, rhs)

    def getMachine(self):
        """
        Machine object.

        Parameters
        ----------

        Returns
        -------
        object
            Returns the tokamak object.

        """

        return self.tokamak

    def plasmaCurrent(self):
        """
        Returns the plasma current [Amps].

        Parameters
        ----------

        Returns
        -------
        float
            Returns the total plasma current [Amps].

        """

        return self._current

    def plasmaVolume(self):
        """
        Plasma volume [m^3] calculated using:

            V = 2 * pi * integral(R) dR dZ,

        where the integral is evaluated over the plasma core domain.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the total plasma volume [m^3].
        """

        # volume element
        dV = 2.0 * pi * self.R * self.dR * self.dZ

        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # integrate volume in 2D
        return np.sum(dV)

    def plasmaBr(self, R, Z):
        """
        Radial magnetic field due to plasma:

            Br_p(R, Z) = -(1 / R) * dψ_p/dZ (R, Z),

        where ψ_p is the plasma flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The radial magnetic field due to the plasma at the given (R, Z) positions [T].
        """

        return -self.psi_func(R, Z, dy=1, grid=False) / R

    def plasmaBz(self, R, Z):
        """
        Vertical magnetic field Bz due to plasma::

            Bz_p(R, Z) = (1 / R) * dψ/dR (R, Z),

        where ψ_p is the plasma flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The vertical magnetic field due to the plasma at the given (R, Z) positions [T].
        """

        return self.psi_func(R, Z, dx=1, grid=False) / R

    def Br(self, R, Z):
        """
        Total radial magnetic field due to plasma and coils:

            Br(R, Z) = -(1 / R) * dψ/dZ (R, Z),

        where ψ is the total flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The total radial magnetic field at the given (R, Z) positions [T].
        """

        return self.plasmaBr(R, Z) + self.tokamak.Br(R, Z)

    def Bz(self, R, Z):
        """
        Total vertical magnetic field due to plasma and coils:

            Bz(R, Z) = (1 / R) * dψ/dR (R, Z),

        where ψ is the total flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The total vertical magnetic field at the given (R, Z) positions [T].
        """

        return self.plasmaBz(R, Z) + self.tokamak.Bz(R, Z)

    def Btor(self, R, Z):
        """
        Total toroidal magnetic field:

            Btor(R, Z) = I(R, Z) * fpol(ψ_n) + (1 - I(R, Z)) * fvac,

        where:
            - I(R, Z) = 1 inside core plasma, 0 outside.
            - fpol = R*Bt at specified values of normalised psi ψ_n (i.e.
        inside the plasma core).
            - fvac = = R*Bt in vacuum (i.e. outside plasma).

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The toroidal magnetic field at the given (R, Z) positions [T].
        """

        # find f = R * Btor in the core
        fpol = self.fpol(self.psiNRZ(self.R, self.Z))

        try:
            # combine values in core plasma with vacuum field outside
            fpol = (
                fpol * self._profiles.limiter_core_mask
                + (1.0 - self._profiles.limiter_core_mask) * self.fvac()
            )
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # interpolate over the field
        func = interpolate.RectBivariateSpline(
            self.R_1D, self.Z_1D, fpol / self.R
        )

        # if inputs are scalars, return scalar
        if np.isscalar(R) and np.isscalar(Z):
            return func(R, Z)[0, 0]
        elif R.ndim == 1 and Z.ndim == 1:
            return func(R.flatten(), Z.flatten(), grid=False)
        elif R.ndim == 2 and Z.ndim == 2:
            return func(R, Z, grid=False)
        else:
            raise ValueError("Unexpected input shape for R and Z.")

    def Bpol(self, R, Z):
        """
        Total poloidal magnetic field due to plasma and coils:

            Bpol(R, Z) = ( Br(R, Z)^2 + Bz(R, Z)^2 )^(1/2).

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The poloidal magnetic field at the given (R, Z) positions [T].
        """

        Br = self.Br(R, Z)
        Bz = self.Bz(R, Z)

        return np.sqrt((Br * Br) + (Bz * Bz))

    def psi(self):
        """
        Total poloidal flux due to plasma and coils on the
        entire (R,Z) grid:

            ψ(R, Z) = ψ_p (R, Z) + ψ_c (R, Z),

        where:
         -  ψ_p (R, Z) is the plasma flux.
         -  ψ_c (R, Z) is the coil flux.

        Parameters
        ----------

        Returns
        -------
        np.array
            The total poloidal flux due to the plasma and the coils [Webers/2pi].
        """

        return self.plasma_psi + self.tokamak.getPsitokamak(self._vgreen)

    def psiRZ(self, R, Z):
        """
        Total poloidal flux due to plasma and coils (at chosen R and Z):

            ψ(R, Z) = ψ_p (R, Z) + ψ_c (R, Z),

        where:
         -  ψ_p (R, Z) is the plasma flux.
         -  ψ_c (R, Z) is the coil flux.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The total poloidal flux due to the plasma and the coils at chosen (R,Z) locations [Webers/2pi].
        """

        return self.psi_func(R, Z, grid=False) + self.tokamak.psi(R, Z)

    def psiNRZ(self, R, Z):
        """
        Total poloidal flux due to plasma and coils (at chosen R and Z), normalised
        with respect to the flux on the magnetic axis and last closed flux surface:

            ψ_n(R, Z) = (ψ(R, Z) - ψ_a)/(ψ_b - ψ_a),

        where:
         -  ψ_a is the total flux on the magnetic axis.
         -  ψ_b is the total flux on the last closed flux surface.

        Parameters
        ----------
        R : float or np.array
            Radial position(s) to evaluate at.
        Z : float or np.array
            Vertical position(s) to evaluate at.

        Returns
        -------
        float or np.array
            The total normalised poloidal flux due to the plasma and the coils at chosen (R,Z) locations.
        """

        return (self.psiRZ(R, Z) - self.psi_axis) / (
            self.psi_bndry - self.psi_axis
        )

    def psi_1D(self, N):
        """
        Total poloidal flux between magnetic axis and last closed flux surface (at
        equally spaced locations).

        Parameters
        ----------
        N : int
            Number of discretisation points.

        Returns
        -------
        np.array
            The total poloidal flux due to the plasma and the coils between the magnetic axis
            and last closed flux surface [Webers/2pi].
        """

        return np.linspace(self.psi_axis, self.psi_bndry, N)

    def psiN_1D(self, N):
        """
        Total normalised poloidal flux between magnetic axis and last closed flux surface.

        Parameters
        ----------
        N : int
            Number of discretisation points.

        Returns
        -------
        np.array
            The total normalised poloidal flux due to the plasma and the coils between the magnetic axis
            and last closed flux surface.
        """

        return np.linspace(0, 1, N)

    def rho_1D(
        self,
        N,
    ):
        """
        Toroidal flux ρ (at locations of normalised psi) such that:

            ρ = Φ(ψ) and

            Φ = - (1/(2 * pi)) integral(q(ψ)) dψ,

        where:
         - q is the safety factor profile as a function of ψ.

        Parameters
        ----------
        N : int
            Number of discretisation points.

        Returns
        -------
        np.array
            The toroidal flux (at locations of normalised psi).
        """

        # calculate safety factor of the (normalised) flux surfaces
        qvals = self.q(self.psiN_1D(N))

        # compute the integral (cumulatively)
        integral = (-1.0 / (2.0 * np.pi)) * cumulative_trapezoid(
            qvals, self.psi_1D(N), initial=0.0
        )
        integral[0] = 0  # hard-code for floating point issues

        # convert to a scalar if only single output
        if len(integral) == 1:
            return integral[0]
        return integral

    def rhoN_1D(self, N):
        """
        Normalised toroidal flux ρ_n (at locations of normailsed psi) such that:

        ρ_n = sqrt( ρ / ρ_b ),

        where ρ_b is the toridal flux at the last closed flux surface.

        Parameters
        ----------
        N : int
            Number of discretisation points.

        Returns
        -------
        np.array
            The normalised toroidal flux (at locations of normalised psi).
        """

        # calculate rho
        rho = self.rho_1D(N)

        return np.sqrt(rho / rho[-1])

    def fpol(self, psinorm):
        """
        Return f = R*Bt at specified values of normalised psi (i.e.
        inside the plasma core).

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            fpol at values of normalised psi [Tm].
        """

        return self._profiles.fpol(psinorm)

    def fvac(self):
        """
        Return vacuum f = R*Bt (i.e. outside the plasma core).

        Parameters
        ----------

        Returns
        -------
        float
            fvac value in vacuum [Tm].
        """

        return self._profiles.fvac()

    def q(self, psinorm):
        """
        Safety factor q at normalised flux values.

        Note that at both psinorm = 0 (magnetic axis) and psinorm = 1 (separatrix),
        calculating q is problematic (results may be inaccurate).

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between (not including) 0 and 1.

        Returns
        -------
        np.array
            Safety factor q at values of normalised psi.
        """

        # print warning if needed
        if 0 in psinorm or 1 in psinorm:
            print(
                "psinorm contains 0 or 1, which may cause numerical issues druing safety factor calculation."
            )

        # find safety factor profile
        q = critical.find_safety(self, psinorm=psinorm)

        # convert to a scalar if only single output
        if len(q) == 1:
            return q.item()

        return q

    def pprime(self, psinorm):
        """
        Return pprime (dp/dψ_n) at normalised flux values.

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between (not including) 0 and 1.

        Returns
        -------
        np.array
            pprime (dp/dψ_n) at normalised flux values [Pa/(Webers/2pi)].
        """

        return self._profiles.pprime(psinorm)

    def ffprime(self, psinorm):
        """
        Return ffprime (F * dF/dψ_n) at normalised flux values.

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between (not including) 0 and 1.

        Returns
        -------
        np.array
            ffprime (F * dF/dψ_n) at normalised flux values [T^2 m^2/(Wb/2pi)].
        """

        return self._profiles.ffprime(psinorm)

    def pressure(self, psinorm):
        """
        Return pressure (p(ψ_n)) at normalised flux values.

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between (not including) 0 and 1.

        Returns
        -------
        np.array
            pressure (p(ψ_n)) at normalised flux values [Pa].
        """

        return self._profiles.pressure(psinorm)

    def separatrix(self, ntheta=360, IQR_factor=2.0, verbose=False):
        """
        Return (ntheta x 2) array of (R,Z) points on the last closed
        flux surface (plasma boundary), equally spaced in the geometric
        poloidal angle.

        Sometimes there may be spurious points far from the core plasma (due to
        open field lines), we exclude these by finding the minimum distance between points
        and then excluding those outside of some upper and lower bounds. These bounds can be
        increased/decreased using IQR_factor.

        Parameters
        ----------
        ntheta : int
            Number of points on the boundary to return.
        IQR_factor : float
            Scaling factor used when exluding spurious points.
        verbose : bool
            Prints any outliers if True.

        Returns
        -------
        np.array
            (R,Z) points on the last closed flux surface (plasma boundary).
        """

        # initial points on core boundary
        points = np.array(critical.find_separatrix(self, ntheta=ntheta))[
            :, 0:2
        ]

        # compute pairwise distances using pdist
        dist_matrix = squareform(pdist(points))  # convert to square form
        # find minimum distances
        min_dists = np.min(
            dist_matrix + 1e10 * np.eye(len(dist_matrix)), axis=0
        )

        # calculate inter-quartile ranges and bounds
        Q1 = np.percentile(min_dists, 25)
        Q3 = np.percentile(min_dists, 75)

        lower_bound = Q1 - IQR_factor * (Q3 - Q1)
        upper_bound = Q3 + IQR_factor * (Q3 - Q1)

        # identify outliers
        outlier_indices = np.where(
            (min_dists < lower_bound) | (min_dists > upper_bound)
        )[0]
        outliers = points[outlier_indices, :]

        # verbose
        if verbose:
            print("Outlier locations:", outliers)

        # filter points
        mask = np.full(len(min_dists), True)
        mask[outlier_indices] = False
        filtered_points = points[mask]

        return filtered_points

    def separatrix_area(self):
        """
        The area of the last closed flux surface [m^2].

        Parameters
        ----------

        Returns
        -------
        float
            Area of the last closed flux surface (plasma boundary) [m^2].
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        return self._sep_area

    def separatrix_length(self):
        """
        The circumference of the last closed flux surface [m].

        Parameters
        ----------

        Returns
        -------
        float
            Circumference of the last closed flux surface (plasma boundary) [m].
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        return self._sep_length

    def getForces(self):
        """
        Calculates the forces on the coils.

        Parameters
        ----------

        Returns
        -------
        dict
            Coil lables with the associated force [N].
        """

        return self.tokamak.getForces(self)

    def printForces(self):
        """
        Prints a table of forces on coils.

        Parameters
        ----------

        Returns
        -------
        None
            Table of forces on coils.
        """

        print("Forces on coils")

        def print_forces(forces, prefix=""):
            for label, force in forces.items():
                if isinstance(force, dict):
                    print(prefix + label + " (circuit)")
                    print_forces(force, prefix=prefix + "  ")
                else:
                    print(
                        prefix
                        + label
                        + " : R = {0:.2f} kN , Z = {1:.2f} kN".format(
                            force[0] * 1e-3, force[1] * 1e-3
                        )
                    )

        print_forces(self.getForces())

    def innerOuterSeparatrix(
        self,
        Z: float = 0.0,
    ):
        """
        Locate inboard and outboard R co-ordinates of last closed flux surface
        at a given Z position.

        Parameters
        ----------
        Z : float, optional
            The Z value at which to find the radii. Defaults to
            0.0.

        Returns
        -------
        float
            The inner separatrix major radius (at chosen Z) [m].
        float
            The outer separatrix major radius (at chosen Z) [m].
        """

        # Find the closest index to requested Z
        Zindex = np.argmin(abs(self.Z[0, :] - Z))

        # Normalise psi at this Z index
        psinorm = (self.psi()[:, Zindex] - self.psi_axis) / (
            self.psi_bndry - self.psi_axis
        )

        # Start from the magnetic axis
        Rindex_axis = np.argmin(abs(self.R[:, 0] - self.Rmagnetic()))

        # Inner separatrix
        # Get the maximum index where psi > 1 in the R index range from 0 to Rindex_axis
        outside_inds = np.argwhere(psinorm[:Rindex_axis] > 1.0)

        if outside_inds.size == 0:
            R_in = self.Rmin
        else:
            Rindex_inner = np.amax(outside_inds)

            # Separatrix should now be between Rindex_inner and Rindex_inner+1
            # Linear interpolation
            R_in = (
                self.R[Rindex_inner, Zindex]
                * (1.0 - psinorm[Rindex_inner + 1])
                + self.R[Rindex_inner + 1, Zindex]
                * (psinorm[Rindex_inner] - 1.0)
            ) / (psinorm[Rindex_inner] - psinorm[Rindex_inner + 1])

        # Outer separatrix
        # Find the minimum index where psi > 1
        outside_inds = np.argwhere(psinorm[Rindex_axis:] > 1.0)

        if outside_inds.size == 0:
            R_out = self.Rmax
        else:
            Rindex_outer = np.amin(outside_inds) + Rindex_axis

            # Separatrix should now be between Rindex_outer-1 and Rindex_outer
            R_out = (
                self.R[Rindex_outer, Zindex]
                * (1.0 - psinorm[Rindex_outer - 1])
                + self.R[Rindex_outer - 1, Zindex]
                * (psinorm[Rindex_outer] - 1.0)
            ) / (psinorm[Rindex_outer] - psinorm[Rindex_outer - 1])

        return R_in, R_out

    def innerOuterSeparatrix2(
        self,
        Z=0.0,
    ):
        """
        Alternative (simpler) function to find inboard and outboard R co-ordinates of
        last closed flux surface at any given Z position (using shapely).

        Parameters
        ----------
        Z : float, optional
            The Z value at which to find the radii. Defaults to
            0.0.

        Returns
        -------
        list
            List with the two (R,Z) coordinates: (R_in, Z) and (R_out, Z) [m].
        """

        # use shapely to define polygon of core plasma
        plasma_polygon = sh.LinearRing(self.separatrix(ntheta=360))

        # use shapely to define horizonal line at Z
        Z_line = sh.LineString([[0, Z], [100, Z]])

        # find intersection
        intersection = plasma_polygon.intersection(Z_line)

        # extract coordinates
        if intersection.is_empty:
            print("No intersection found.")
            intersection_coords = []
        elif intersection.geom_type == "Point":
            intersection_coords = [intersection.coords[0]]  # Single point
        elif intersection.geom_type == "MultiPoint":
            intersection_coords = [
                pt.coords[0] for pt in intersection.geoms
            ]  # Multiple points
        else:
            raise ValueError(
                f"Unexpected intersection type: {intersection.geom_type}"
            )

        return intersection_coords

    def intersectsWall(self):
        """
        Assess whether or not the core plasma touches the vessel
        walls.

        Parameters
        ----------

        Returns
        -------
        bool
            True if they interect, False if not.
        """

        # generate list of points for each
        separatrix = self.separatrix()
        wall = self.tokamak.wall

        return polygons.intersect(
            separatrix[:, 0], separatrix[:, 1], wall.R, wall.Z
        )

    def magneticAxis(self):
        """
        Returns the location of the magnetic axis and its total poloidal flux value.

        Parameters
        ----------

        Returns
        -------
        list
            Returns [R, Z, ψ(R, Z)] at magnetic axis location  [m, m, Webers/2pi].
        """

        # find critical points
        opts, xpts = critical.find_critical(self.R, self.Z, self.psi())

        return opts[0, :]

        # # find closest O-point to the geometric axis
        # geom_axis = self.geometricAxis()[0:2]
        # o_point_ind = np.argmin(
        #     np.sum((opts[:, 0:2] - geom_axis) ** 2, axis=1)
        # )

        # return opts[o_point_ind, :]

    def Rcurrent(self):
        """
        The average major radius of the toroidal current distribution

        Parameters
        ----------

        Returns
        -------
        float
            average major radius [m].
        """

        if hasattr(self, "_profiles"):
            meanR = np.sum(self.R * self._profiles.jtor) / np.sum(
                self._profiles.jtor
            )
            return meanR
        else:
            print(
                "The equilibrium is not a GS solution. Please solve the eq. first."
            )

    def Zcurrent(self):
        """
        The average height of the toroidal current distribution

        Parameters
        ----------

        Returns
        -------
        float
            average height [m].
        """

        if hasattr(self, "_profiles"):
            meanZ = np.sum(self.Z * self._profiles.jtor) / np.sum(
                self._profiles.jtor
            )
            return meanZ
        else:
            print(
                "The equilibrium is not a GS solution. Please solve the eq. first."
            )

    def Rmagnetic(self):
        """
        The major radius of the magnetic axis.

        Parameters
        ----------

        Returns
        -------
        float
            Major radius of the magnetic axis [m].
        """

        return self.magneticAxis()[0]

    def Zmagnetic(self):
        """
        The height of the magnetic axis.

        Parameters
        ----------

        Returns
        -------
        float
            Vertical height of the magnetic axis [m].
        """

        return self.magneticAxis()[1]

    def geometricAxis(self):
        """
        The geometric axis (R_geom, Z_geom) of the plasma [m]. Calculated as
        the centre of a large number of points on the separatrix.

        Parameters
        ----------

        Returns
        -------
        np.array
            Geometric axis (R,Z) position [m].
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        return np.array(
            [
                (self._sep_Rmax + self._sep_Rmin) / 2,
                (self._sep_Zmax + self._sep_Zmin) / 2,
            ]
        )

    def Rgeometric(self):
        """
        The major radius of the geometric axis.

        Parameters
        ----------

        Returns
        -------
        float
            Major radius of the geometric axis [m].
        """

        return self.geometricAxis()[0]

    def Zgeometric(self):
        """
        The height of the geometric axis.

        Parameters
        ----------

        Returns
        -------
        float
            Vertical height of the geometric axis [m].
        """

        return self.geometricAxis()[1]

    def minorRadius(self):
        """
        Calculate the minor radius of the plasma [m].

        Parameters
        ----------

        Returns
        -------
        float
            Minor radius of the plasma [m].
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        return (self._sep_Rmax - self._sep_Rmin) / 2

    def aspectRatio(self):
        """
        Calculate the aspect ratio of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Aspect ratio of the plasma.
        """

        # check if metrics are already calculated
        # calculate separatrix metrics
        self._separatrix_metrics()

        return (self._sep_Rmax + self._sep_Rmin) / (
            self._sep_Rmax - self._sep_Rmin
        )

    def geometricElongation(self):
        """
        Calculate the geometric elongation of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Geometric elongation of the plasma.
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        return (self._sep_Zmax - self._sep_Zmin) / (
            self._sep_Rmax - self._sep_Rmin
        )

    def geometricElongation_upper(self):
        """
        Calculate the geometric elongation of the upper part of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Geometric elongation of the upper part of the plasma.
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        return (
            2
            * (self._sep_Zmax - self._sep_ZRmax)
            / (self._sep_Rmax - self._sep_Rmin)
        )

    def geometricElongation_lower(self):
        """
        Calculate the geometric elongation of the lower part of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Geometric elongation of the lower part of the plasma.
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        return (
            2
            * (self._sep_ZRmax - self._sep_Zmin)
            / (self._sep_Rmax - self._sep_Rmin)
        )

    def effectiveElongation(self):
        """
        Calculate the effective elongation of the plasma, using plasma volume.

        Parameters
        ----------

        Returns
        -------
        float
            Effective elongation of the plasma.
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        R_geom = (self._sep_Rmax + self._sep_Rmin) / 2
        R_minor = (self._sep_Rmax - self._sep_Rmin) / 2

        return self.plasmaVolume() / (2.0 * R_geom * (np.pi**2) * (R_minor**2))

    def shafranov_shift(self):
        """
        Calculate the Shafranov shift of the plasma (i.e. the shift between the magnetic
        axis and the geometrix axis).

        Parameters
        ----------

        Returns
        -------
        float
            Shafranov shift of the plasma [del_R, del_Z] [m].
        """

        mag_axis = self.magneticAxis()
        geom_axis = self.geometricAxis()

        return np.array(
            [mag_axis[0] - geom_axis[0], mag_axis[1] - geom_axis[1]]
        )

    def triangularity_upper(self):
        """
        Calculate the upper triangularity of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Upper triangularity of the plasma.
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        R_geom = (self._sep_Rmax + self._sep_Rmin) / 2
        R_minor = (self._sep_Rmax - self._sep_Rmin) / 2

        return (R_geom - self._sep_RZmax) / R_minor

    def triangularity_lower(self):
        """
        Calculate the lower triangularity of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Lower triangularity of the plasma.
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        R_geom = (self._sep_Rmax + self._sep_Rmin) / 2
        R_minor = (self._sep_Rmax - self._sep_Rmin) / 2

        return (R_geom - self._sep_RZmin) / R_minor

    def triangularity(self):
        """
        Calculate the triangularity of the plasma.

        Parameters
        ----------

        Returns
        -------
        float
            Triangularity of the plasma.
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        R_geom = (self._sep_Rmax + self._sep_Rmin) / 2
        R_minor = (self._sep_Rmax - self._sep_Rmin) / 2

        return (R_geom - 0.5 * (self._sep_RZmax + self._sep_RZmin)) / R_minor

    def squareness(self):
        """
        This function calculates the squareness of the plasma core for each of the four quadrants:
        upper outer, upper lower, lower outer, and lower inner. Method outlined in:

        An analytic functional form for characterization and generation of axisymmetric plasma
        boundaries. Plasma Physics and Controlled Fusion. 2013. T C Luce.

        Parameters
        ----------

        Returns
        -------
        float
            Upper outer squareness.
        float
            Upper inner squareness.
        float
            Lower outer squareness.
        float
            Lower inner squareness.
        """

        # calculate separatrix metrics
        self._separatrix_metrics()

        # create shapely object for plasma core
        plasma_boundary = sh.Polygon(self.separatrix())

        # same for "ideal" ellipses in each quadrant
        uo_ellipse = sh.Polygon(
            ellipse_points(
                R0=self._sep_RZmax,
                Z0=self._sep_ZRmax,
                A=self._sep_Rmax - self._sep_RZmax,
                B=self._sep_Zmax - self._sep_ZRmax,
            )
        )
        ui_ellipse = sh.Polygon(
            ellipse_points(
                R0=self._sep_RZmax,
                Z0=self._sep_ZRmin,
                A=self._sep_RZmax - self._sep_Rmin,
                B=self._sep_Zmax - self._sep_ZRmin,
            )
        )
        lo_ellipse = sh.Polygon(
            ellipse_points(
                R0=self._sep_RZmin,
                Z0=self._sep_ZRmax,
                A=self._sep_Rmax - self._sep_RZmin,
                B=self._sep_ZRmax - self._sep_Zmin,
            )
        )
        li_ellipse = sh.Polygon(
            ellipse_points(
                R0=self._sep_RZmin,
                Z0=self._sep_ZRmin,
                A=self._sep_RZmin - self._sep_Rmin,
                B=self._sep_ZRmin - self._sep_Zmin,
            )
        )

        # bounding box diagonals
        uo_diag = sh.LineString(
            [
                [self._sep_RZmax, self._sep_ZRmax],
                [self._sep_Rmax, self._sep_Zmax],
            ]
        )
        ui_diag = sh.LineString(
            [
                [self._sep_RZmax, self._sep_ZRmin],
                [self._sep_Rmin, self._sep_Zmax],
            ]
        )
        lo_diag = sh.LineString(
            [
                [self._sep_RZmin, self._sep_ZRmax],
                [self._sep_Rmax, self._sep_Zmin],
            ]
        )
        li_diag = sh.LineString(
            [
                [self._sep_RZmin, self._sep_ZRmin],
                [self._sep_Rmin, self._sep_Zmin],
            ]
        )

        # find intersecting line lengths
        uo_diag_core = uo_diag.intersection(plasma_boundary).length
        ui_diag_core = ui_diag.intersection(plasma_boundary).length
        lo_diag_core = lo_diag.intersection(plasma_boundary).length
        li_diag_core = li_diag.intersection(plasma_boundary).length

        uo_diag_ellipse = uo_diag.intersection(uo_ellipse).length
        ui_diag_ellipse = ui_diag.intersection(ui_ellipse).length
        lo_diag_ellipse = lo_diag.intersection(lo_ellipse).length
        li_diag_ellipse = li_diag.intersection(li_ellipse).length

        # calculate squarenesses
        s_uo = (uo_diag_core - uo_diag_ellipse) / (
            uo_diag.length - uo_diag_ellipse
        )
        s_ui = (ui_diag_core - ui_diag_ellipse) / (
            ui_diag.length - ui_diag_ellipse
        )
        s_lo = (lo_diag_core - lo_diag_ellipse) / (
            lo_diag.length - lo_diag_ellipse
        )
        s_li = (li_diag_core - li_diag_ellipse) / (
            li_diag.length - li_diag_ellipse
        )

        return s_uo, s_ui, s_lo, s_li

    def closest_wall_point(self):
        """
        This function calculates the (R,Z) point on the wall that is closest
        to the last closed flux surface boundary (and the corresponding distance).

        Parameters
        ----------

        Returns
        -------
        list
            Contains coordinates array and minimum distance to wall: [ (R,Z), min_distance ] [m,m,m].
        """

        # create shapely linestring objects
        plasma_boundary = sh.LineString(self.separatrix())
        wall = sh.LineString(
            np.array([self.tokamak.wall.R, self.tokamak.wall.Z]).T
        )

        # initialize variables
        closest_point = None
        min_distance = float("inf")

        # iterate over each point on the wall contour
        for point_coords in wall.coords:
            wall_point = sh.Point(point_coords)

            # calculate distance from wall point to plasma boundary
            initial_distance = plasma_boundary.distance(wall_point)

            if initial_distance < min_distance:
                # find the closest point on the plasma boundary
                candidate_closest_point = plasma_boundary.interpolate(
                    plasma_boundary.project(wall_point)
                )

                # recalculate the distance between the wall point and the closest point (due to interpolation)
                corrected_distance = wall_point.distance(
                    candidate_closest_point
                )

                if corrected_distance < min_distance:
                    min_distance = corrected_distance
                    closest_point = candidate_closest_point

        return [np.array(closest_point.coords[0]), min_distance]

    def internalInductance1(self):
        """
        Calculates plasma internal inductance according to:

            li = ( (integral() dl * integral(Bpol) dV) / (mu0 * Ip * V) )^2,

        where:
         - L = length of the separatrix (LCFS).
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element.
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - V = total plasma volume.

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Plasma internal inductance.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # first order integral
        integral = np.sum(self.Bpol(self.R, self.Z) * dV)

        return (
            (self.separatrix_length() * integral)
            / (mu0 * self.plasmaCurrent() * self.plasmaVolume())
        ) ** 2

    def internalInductance2(self):
        """
        Calculates plasma internal inductance according to:

            li = ( integral(Bpol**2) dV ) / ( ( integral(Bpol) dl / L )**2 * V),

        where:
         - L = length of the separatrix (LCFS).
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element.
         - dl = line element (along LCFS).
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - V = total plasma volume.

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Plasma internal inductance.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # integrate B_pol squared over the volume
        integral_Bpol_2 = np.sum((self.Bpol(self.R, self.Z) ** 2) * dV)

        # integrate B_pol over the LCFS

        # find arc length
        separatrix = self.separatrix()
        dl = np.sqrt(
            np.diff(separatrix[:, 0]) ** 2 + np.diff(separatrix[:, 1]) ** 2
        )  # element-wise arc length
        l = np.concatenate(([0], np.cumsum(dl)))  # cumulative arc length

        # evaluate Bp on the LCFS
        Bp = self.Bpol(separatrix[:, 0], separatrix[:, 1])

        # integrate
        integral_Bpol_lcfs = trapezoid(Bp, l)

        return integral_Bpol_2 / (
            (integral_Bpol_lcfs / self.separatrix_length()) ** 2
            * self.plasmaVolume()
        )

    def internalInductance3(self):
        """
        Calculates plasma internal inductance according to:

            li = (2 * integral(Bpol^2) dV) / (mu0^2 * Ip^2 * R_geom),

        where:
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element.
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - R_geom = radial coords of geometric axis.

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Plasma internal inductance.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # integrate B_pol squared over the volume
        integral = np.sum((self.Bpol(self.R, self.Z) ** 2) * dV)

        return (
            2
            * integral
            / ((mu0 * self.plasmaCurrent()) ** 2 * self.Rgeometric())
        )

    def internalInductance4(self):
        """
        Calculates plasma internal inductance according to:

            li = (2 * integral(Bpol^2) dV) / (mu0^2 * Ip^2 * R_mag),

        where:
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element.
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - R_mag = radial coords of magnetic axis.


        Parameters
        ----------

        Returns
        -------
        float
            Plasma internal inductance.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # integrate B_pol squared over the volume
        integral = np.sum((self.Bpol(self.R, self.Z) ** 2) * dV)

        return (
            2
            * integral
            / ((mu0 * self.plasmaCurrent()) ** 2 * self.Rmagnetic())
        )

    def internalInductance5(self):
        """
        Calculates plasma internal inductance according to:

            li = [(2 * integral(Bpol^2) dV) / (mu0^2 * Ip^2 * R_geom)] * [(1 + geom_elon^2)/(2 * eff_elon)],

        where:
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element.
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - R_geom = radial coords of geometric axis.
         - geom_elon = geometric elongation.
         - eff_elon = effective elongation.

        Parameters
        ----------

        Returns
        -------
        float
            Plasma internal inductance.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # integrate B_pol squared over the volume
        integral = np.sum((self.Bpol(self.R, self.Z) ** 2) * dV)

        return (
            (2 * integral)
            / ((mu0 * self.plasmaCurrent()) ** 2 * self.Rgeometric())
        ) * (
            (1 + self.geometricElongation() ** 2)
            / (2.0 * self.effectiveElongation())
        )

    def poloidalBeta1(self):
        """
        Calculates poloidal beta from the following definition:

            betap = (8 * pi / (mu0 * Ip^2)) * integral(p) dR dZ,

        where:
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - p = plasma pressure (at each (R,Z)).

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the poloidal beta value.
        """

        # volume elements
        dV = self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        pressure_integral = np.sum(
            self.pressure(self.psiNRZ(self.R, self.Z)) * dV
        )

        return (
            ((8.0 * pi) / mu0)
            * pressure_integral
            / (self.plasmaCurrent() ** 2)
        )

    def poloidalBeta(self):
        """Backward-compatible alias for :meth:`poloidalBeta1`."""

        return self.poloidalBeta1()

    def poloidalBeta2(self):
        """
        Calculates poloidal beta from the following definition:

            betap = 2 * mu0 * integral(p) dV / integral(Bpol^2) dV,

        where:
         - mu0 = magnetic permeability in vacuum.
         - p = plasma pressure (at each (R,Z)).
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dV = volume element

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the poloidal beta value.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        pressure_integral = np.sum(
            self.pressure(self.psiNRZ(self.R, self.Z)) * dV
        )
        field_integral_pol = np.sum((self.Bpol(self.R, self.Z) ** 2) * dV)

        return 2 * mu0 * pressure_integral / field_integral_pol

    def poloidalBeta3(self):
        """
        Calculates poloidal beta from the following definition:

            betap = 2 * mu0 * ( integral(p) dV / V)
                    / (integral(Bpol) dl / L)**2,

        where:
         - p = pressure field.
         - dV = volume element.
         - V = total plasma volume.
         - Bpol(R,Z) = poloidal magnetic field (at each (R,Z)).
         - dl = line element (along LCFS).
         - L = length of the separatrix (LCFS).

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the poloidal beta value.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        # pressure integral
        pressure_integral = np.sum(
            self.pressure(self.psiNRZ(self.R, self.Z)) * dV
        )

        # integrate B_pol over the LCFS
        # find arc length
        separatrix = self.separatrix()
        dl = np.sqrt(
            np.diff(separatrix[:, 0]) ** 2 + np.diff(separatrix[:, 1]) ** 2
        )  # element-wise arc length
        l = np.concatenate(([0], np.cumsum(dl)))  # cumulative arc length

        # evaluate Bp on the LCFS
        Bp = self.Bpol(separatrix[:, 0], separatrix[:, 1])

        # integrate
        integral_Bpol_lcfs = trapezoid(Bp, l)

        return (
            2.0
            * mu0
            * (pressure_integral / self.plasmaVolume())
            / (integral_Bpol_lcfs / self.separatrix_length()) ** 2
        )

    def poloidalBeta4(self):
        """
        Calculates poloidal beta from the following definition:

            betap = 4 * integral(p) dV / (mu0 * Ip^2 * integral(R) dV / V),

        where:
         - p = plasma pressure (at each (R,Z)).
         - dV = volume element
         - mu0 = magnetic permeability in vacuum.
         - Ip = total plasma current.
         - R = radial grid points.
         - V = total plasma volume.

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the poloidal beta value.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        pressure_integral = np.sum(
            self.pressure(self.psiNRZ(self.R, self.Z)) * dV
        )
        R_avg = np.sum(self.R * dV) / self.plasmaVolume()

        return (4 * pressure_integral) / (
            mu0 * self.plasmaCurrent() ** 2 * R_avg
        )

    def toroidalBeta1(self):
        """
        Calculates a toroidal beta from the following definition:

            betat = 2 * mu0 * integral(p) dR dZ / integral(Btor**2) dR dZ,

        where:
         - mu0 = magnetic permeability in vacuum.
         - p = plasma pressure (at each (R,Z)).
         - Btor(R,Z) = toroidal magnetic field (at an (R,Z)).

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the toroidal beta value.
        """

        # plasma pressure and toroidal fields
        pressure = self.pressure(self.psiNRZ(self.R, self.Z))
        Btor2 = self.Btor(self.R, self.Z) ** 2

        # mask with the core plasma
        try:
            pressure *= self._profiles.limiter_core_mask
            Btor2 *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        pressure_integral = np.sum(pressure * self.dR * self.dZ)
        Btor2_integral = np.sum(Btor2 * self.dR * self.dZ)

        return 2 * mu0 * pressure_integral / Btor2_integral

    def toroidalBeta2(self):
        """
        Calculates a toroidal beta from the following definition:

            betat = 2 * mu0 * integral(p) dV / integral(Btor**2) dV,

        where:
         - mu0 = magnetic permeability in vacuum.
         - p = plasma pressure (at each (R,Z)).
         - dV = volume element
         - Btor(R,Z) = toroidal magnetic field (at an (R,Z)).

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the toroidal beta value.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        pressure_integral = np.sum(
            self.pressure(self.psiNRZ(self.R, self.Z)) * dV
        )
        Btor2_int = np.sum((self.Btor(self.R, self.Z) ** 2) * dV)

        return 2 * mu0 * pressure_integral / Btor2_int

    def toroidalBeta3(self):
        """
        Calculates a toroidal beta from the following definition:

            betat = 2 * mu0 * integral(p) dV / integral(Btor**2 + Bpol**2) dV,

        where:
         - mu0 = magnetic permeability in vacuum.
         - p = plasma pressure (at each (R,Z)).
         - dV = volume element
         - Btor(R,Z) = toroidal magnetic field (at an (R,Z)).
         - Bpol(R,Z) = poloidal magnetic field (at an (R,Z)).

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the toroidal beta value.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        pressure_integral = np.sum(
            self.pressure(self.psiNRZ(self.R, self.Z)) * dV
        )
        B_int = np.sum(
            (self.Btor(self.R, self.Z) ** 2 + self.Bpol(self.R, self.Z) ** 2)
            * dV
        )

        return 2 * mu0 * pressure_integral / B_int

    def toroidalBeta4(self):
        """
        Calculates a toroidal beta from the following definition:

            betat = 2 * mu0 * integral(p) dV / Btor_vac(R_geom,Z_geom)^2 * V,

        where:
         - mu0 = magnetic permeability in vacuum.
         - p = plasma pressure (at each (R,Z)).
         - dV = volume element.
         - Btor_vac(R,Z) = vacuum toroidal magnetic field (at an (R,Z)).
         - (Rgeom,Zgeom) = geoemtric axis.
         - V = total plasma volume.

        This definition is also used in the Fiesta/Topeol equilibrium codes.

        Parameters
        ----------

        Returns
        -------
        float
            Returns the toroidal beta value.
        """

        # volume elements
        dV = 2.0 * np.pi * self.R * self.dR * self.dZ

        # mask with the core plasma
        try:
            dV *= self._profiles.limiter_core_mask
        except AttributeError as e:
            print(e)
            warnings.warn(
                "The core mask is not in place. You need to solve for the equilibrium first!"
            )
            raise e

        pressure_integral = np.sum(
            self.pressure(self.psiNRZ(self.R, self.Z)) * dV
        )
        Btor_vac = self.fvac() / self.Rgeometric()

        return (
            2 * mu0 * pressure_integral / (Btor_vac**2 * self.plasmaVolume())
        )

    def normalised_total_Beta(self):
        """
        Calculates the total beta from the following definition:

            normalised_total_Beta = ( (1 / poloidalBeta1) + (1/toroidalBeta1) )^(-1).

        Parameters
        ----------

        Returns
        -------
        float
            Returns the normalised total beta value.

        """

        return 1.0 / (
            (1.0 / self.poloidalBeta1()) + (1.0 / self.toroidalBeta1())
        )

    def strikepoints(
        self,
        quadrant="all",
        loc=None,
    ):
        """
        This function can be used to find the strikepoints of an equilibrium (i.e
        the points at which the psi_boundary contour intersect the wall.

        Parameters
        ----------
        quadrant: str
            Which strikepoints to return to the user, options are "all" (returns all points) or
            one of "lower left", "lower right", "upper left", or "upper right" quadrants (which
            returns point(s) in quadrant of poloidal plane with centre given by 'loc' (R,Z) pair).
        loc: tuple
            (R,Z) point at which to centre the quadrants choice [m].

        Returns
        -------
        np.array
            Returns an array of the (R,Z) strikepoints [m].

        """

        # find contour object for psi_boundary
        if self._profiles.flag_limiter:
            cs = plt.contour(
                self.R, self.Z, self.psi(), levels=[self._profiles.psi_bndry]
            )
        else:
            cs = plt.contour(
                self.R, self.Z, self.psi(), levels=[self._profiles.xpt[0][2]]
            )
        plt.close()  # this isn't the most elegant but we don't need the plot itself

        # for each item in the contour object there's a list of points in (r,z) (i.e. a line)
        psi_boundary_lines = []
        for i, item in enumerate(cs.allsegs[0]):
            psi_boundary_lines.append(item)

        # use the shapely package to find where each psi_boundary_line intersects the wall (or not)
        strikes = []
        wall = np.array([self.tokamak.wall.R, self.tokamak.wall.Z]).T
        curve1 = sh.LineString(wall)
        for j, line in enumerate(psi_boundary_lines):
            curve2 = sh.LineString(line)

            # find the intersection points
            intersection = curve2.intersection(curve1)

            # extract intersection points
            if intersection.geom_type == "Point":
                strikes.append(np.squeeze(np.array(intersection.xy).T))
            elif intersection.geom_type == "MultiPoint":
                strikes.append(
                    np.squeeze(
                        np.array([geom.xy for geom in intersection.geoms])
                    )
                )

        # check how many strikepoints
        if len(strikes) == 0:
            return None
        else:
            all_strikes = np.concatenate(strikes, axis=0)

        # which strikepoint(s) to return
        if quadrant == "all":
            return all_strikes
        else:
            if loc == None:
                raise ValueError(
                    f"Need to define quadrant centre point in 'loc' (R,Z)."
                )
            else:
                if quadrant == "lower left":
                    ind = np.where(
                        (all_strikes[:, 0] < loc[0])
                        & (all_strikes[:, 1] < loc[1])
                    )[0]
                elif quadrant == "lower right":
                    ind = np.where(
                        (all_strikes[:, 0] > loc[0])
                        & (all_strikes[:, 1] < loc[1])
                    )[0]
                elif quadrant == "upper left":
                    ind = np.where(
                        (all_strikes[:, 0] < loc[0])
                        & (all_strikes[:, 1] > loc[1])
                    )[0]
                elif quadrant == "upper right":
                    ind = np.where(
                        (all_strikes[:, 0] > loc[0])
                        & (all_strikes[:, 1] > loc[1])
                    )[0]
                else:
                    raise ValueError(
                        f"Unexpected quadrant: {quadrant}. Choose from 'all', "
                        f"'lower left', 'lower right', 'upper left', or 'upper right'."
                    )

        # return if quadrant value required
        if len(ind) == 0:
            return np.full(2, None)
        else:
            return np.array(
                [all_strikes[:, 0][ind], all_strikes[:, 1][ind]]
            ).T.squeeze()

    def dr_sep(
        self,
        Z_level=None,
        sign_flip=False,
    ):
        """
        Computes the radial separation (on the inboard and outboard sides) at vertical position `Z_level`
        between flux surfaces passing through the lower and upper X-points. This requires two X-points to
        be located within the vessel region.

        The inboard dr_sep is defined as the difference in R (radial position) between
        the two flux surfaces intersecting `Z_level` on the inboard side (smaller R values),
        each corresponding to one of the X-points.

        Similarly, the outboard dr_sep is the radial separation of the same flux surfaces
        on the outboard side (larger R values).

        Parameters
        ----------
        Z_level : float
            Vertical height at which to calculate dr_sep [m]. Defaults to eq.Zgeoemtric().
        sign_flip : bool
            By convention, the function assumes the first X-point corresponds to the lower
            X-point, and the second to the upper X-point. So for an upper single null plasma,
            dr_sep_out should be positive and dr_sep_in should be negative. Setting sign_flip=True
            will do the opposite and reverse the signs of the output values.

        Returns
        -------
        list of float
            A list containing the inboard and outboard dr_sep values:
            [dr_sep_in, dr_sep_out].
        """

        # check if Z position specified, else set to Zgeometric
        if Z_level is None:
            Z_level = self.Zgeometric()

        # extract required quantities
        psi_bndry = self.psi_bndry
        xpts = self.xpt

        # build polygon of the tokamak wall
        polygon = sh.Polygon(
            np.array([self.tokamak.wall.R, self.tokamak.wall.Z]).T
        )

        # find x-points inside the wall
        mask = np.array(
            [polygon.contains(sh.Point(x, y)) for x, y in xpts[:, 0:2]]
        )

        # select these x-points
        xpts_inside_wall = xpts[mask, :]

        # if less than two X-points inside wall, throw error
        if xpts_inside_wall.shape[0] < 2:
            raise ValueError(
                f"Expected at least two X-points inside wall, got {xpts_inside_wall.shape[0]}."
            )

        # get the two closest to magnetic axis
        mag_axis = self.magneticAxis()
        distances = np.linalg.norm(
            xpts_inside_wall[:, 0:2] - mag_axis[0:2], axis=1
        )
        closest_two = xpts_inside_wall[np.argsort(distances)[:2]]

        # sort those two by Z (puts lowest X-point first)
        closest_xpts_sorted = closest_two[np.argsort(closest_two[:, 1])]

        # find the flux contours for the values of psi_boundary at each X-point
        contour0 = plt.contour(
            self.R, self.Z, self.psi(), levels=[closest_xpts_sorted[0, 2]]
        )
        contour1 = plt.contour(
            self.R, self.Z, self.psi(), levels=[closest_xpts_sorted[1, 2]]
        )
        plt.close()  # this isn't the most elegant but we don't need the plot itself

        # for each item in the contour object there's a list of points in (r,z) (i.e. a line)
        psi_boundary_lines0 = []
        for i, item in enumerate(contour0.allsegs[0]):
            psi_boundary_lines0.append(item)
        psi_boundary_lines1 = []
        for i, item in enumerate(contour1.allsegs[0]):
            psi_boundary_lines1.append(item)

        # use the shapely package to find where each psi_boundary_line intersects
        # the Z = Z_level line
        R_limits = np.array([[self.Rmin, Z_level], [self.Rmax, Z_level]])
        R_line = sh.LineString(R_limits)

        # set of intersection points for the first flux surface
        R_in_out0 = []
        for j, line in enumerate(psi_boundary_lines0):
            curve = sh.LineString(line)

            # find the intersection points
            intersection = curve.intersection(R_line)

            # extract intersection points
            if intersection.geom_type == "Point":
                R_in_out0.append(np.squeeze(np.array(intersection.xy).T))
            elif intersection.geom_type == "MultiPoint":
                R_in_out0.append(
                    np.squeeze(
                        np.array([geom.xy for geom in intersection.geoms])
                    )
                )

        # set of intersection points for the second flux surface
        R_in_out1 = []
        for j, line in enumerate(psi_boundary_lines1):
            curve = sh.LineString(line)

            # find the intersection points
            intersection = curve.intersection(R_line)

            # extract intersection points
            if intersection.geom_type == "Point":
                R_in_out1.append(np.squeeze(np.array(intersection.xy).T))
            elif intersection.geom_type == "MultiPoint":
                R_in_out1.append(
                    np.squeeze(
                        np.array([geom.xy for geom in intersection.geoms])
                    )
                )

        # extract points into numpy array
        R_in_out0_stacked = np.vstack(R_in_out0)
        R_in_out1_stacked = np.vstack(R_in_out1)

        # clip rows so that we only obtain the two rows with R values closest to the magnetic axis
        R_in_out0_filtered = R_in_out0_stacked[
            np.argsort(np.abs(R_in_out0_stacked[:, 0] - mag_axis[0]))[:2]
        ]
        R_in_out1_filtered = R_in_out1_stacked[
            np.argsort(np.abs(R_in_out1_stacked[:, 0] - mag_axis[0]))[:2]
        ]

        # sort so that the lower X-point is first
        R_in_out0_sort = np.sort(R_in_out0_filtered, axis=0)
        R_in_out1_sort = np.sort(R_in_out1_filtered, axis=0)

        if R_in_out0_sort.shape == (2, 2) and R_in_out1_sort.shape == (2, 2):
            dr_sep_in = R_in_out0_sort[0, 0] - R_in_out1_sort[0, 0]
            dr_sep_out = R_in_out0_sort[1, 0] - R_in_out1_sort[1, 0]

            # flip signs if upper X-point considered first
            if sign_flip:
                dr_sep_in *= -1
                dr_sep_out *= -1
        else:
            print(
                f"Expected shape (2, 2) for both R_in_out0_sort and R_in_out1_sort, "
                f"but got {R_in_out0_sort.shape} and {R_in_out1_sort.shape} respectively. "
                f"Cannot calculate dr_sep_in and dr_sep_out for this plasma."
            )
            dr_sep_in = None
            dr_sep_out = None

        return [dr_sep_in, dr_sep_out]

    def flux_averaged_function(
        self,
        f,
        psi_n=None,
    ):
        """
        Calculates the flux averaged value of a 2D scalar field f on a (normalised) flux surface
        of psi (denotes by psi_n) within the last closed flux surface according to:

            < f >(psi_n) = integral_{psi_n} (f/B_pol) dl /  integral_{psi_n} (1/B_pol) dl,

        where:
            - f(R,Z) = 2D scalar field function (e.g. jtor).
            - psi_n = value of normalised flux at which to evaluate line integrals.
            - Bpol(R,Z) = 2D scalar poloidal magnetic field function.
            - dl = parameterised line element (distance) along the flux surface.

        This definition was taken from https://www.mdpi.com/2571-6182/7/4/45.

        Parameters
        ----------
        f(R,Z) : function
            2D scalar function that is to be flux averaged. Must be able to take in arrays of
            (R,Z) coordinates.
        psi_n : float or np.array, optional
            Range of normalised flux values to average on (each element must be 0 < psi_n(i) < 1). One
            value returned for each element in psi_n.

        Returns
        -------
        flux_averaged_quantity : float or np.array
            The flux averaged quantity evaluated at each value in psi_n.
        psi_n : float or np.array
            The values of psi_n used to calculate each value in flux_averaged_quantity (may be different
            to the input psi_n if the input array had values outside [0,1]).

        """

        # cannot solve for psi_n = 0 or 1, so clip
        if psi_n is None:
            psi_n = np.linspace(0.01, 0.99, 65)
        else:
            psi_n = np.clip(psi_n, 0.01, 0.99)

        # extract normalised psi, masking outside limiter
        masked_psi = np.ma.array(
            self.psiNRZ(R=self.R, Z=self.Z), mask=self.mask_outside_limiter
        )

        # magnetic axis location does not depend on psi_n - compute once
        mag_axis = self.magneticAxis()[0:2]

        # spline of the total (plasma + coil) psi field, reused below to get
        # cheap Br/Bz/Bpol on each flux surface via derivatives of the spline
        psi_total_func = interpolate.RectBivariateSpline(
            self.R_1D, self.Z_1D, self.psi()
        )

        # single contour generator, reused for every psi_n level instead of
        cont_gen = contourpy.contour_generator(
            x=self.R,
            y=self.Z,
            z=masked_psi,
            line_type=contourpy.LineType.Separate,
        )

        flux_averaged_quantity = np.zeros(len(psi_n))
        for i, val in enumerate(psi_n):

            # for each contour line there's a list of points in (r,z) (i.e. a curve)
            psi_boundary_lines = [
                line for line in cont_gen.lines(val) if line.shape[0] > 0
            ]

            # find the flux surface closest to the magnetic axis
            min_distances = [
                np.min(np.linalg.norm(line - mag_axis, axis=1))
                for line in psi_boundary_lines
            ]
            closest_index = np.argmin(min_distances)
            flux_surface = psi_boundary_lines[closest_index]

            # do the line integral around the flux surface
            dl = np.sqrt(
                np.diff(flux_surface[:, 0]) ** 2
                + np.diff(flux_surface[:, 1]) ** 2
            )  # element-wise arc length
            l = np.concatenate(([0], np.cumsum(dl)))  # cumulative arc length

            # evaluate 1/Bp on the flux surface, from the total-psi spline:
            #   Br(R,Z) = -(1/R) dpsi/dZ,   Bz(R,Z) = (1/R) dpsi/dR
            Rc, Zc = flux_surface[:, 0], flux_surface[:, 1]
            Br = -psi_total_func(Rc, Zc, dy=1, grid=False) / Rc
            Bz = psi_total_func(Rc, Zc, dx=1, grid=False) / Rc
            Bp_inv = 1 / np.sqrt(Br**2 + Bz**2)

            # evaluate f on the flux surface
            f_on_flux_surf = f(Rc, Zc)

            # integrate with trapezoidal rule
            integral_f_inv_Bpol = trapezoid(f_on_flux_surf * Bp_inv, l)
            integral_inv_Bpol = trapezoid(Bp_inv, l)

            flux_averaged_quantity[i] = integral_f_inv_Bpol / integral_inv_Bpol

        return flux_averaged_quantity, psi_n

    def solve(self, profiles, Jtor=None, psi=None, psi_bndry=None):
        """
        This is a legacy function that can be used to solve for the plasma equilibrium. It
        performs a linear Grad-Shafranov solve.

        Parameters
        ----------
        profiles: class
            Class containing the toroidal current density profile information.
        Jtor: np.array
            Toroidal current density field over (R,Z) [A/m^2].
        psi: np.array
            Total poloidal magnetic flux over (R,Z) [Webers/2pi].
        psi_bndry: float
            Total poloidal magnetic flux on the plasma boundary [Webers/2pi].

        Returns
        -------
        None
            Modifies the equilbirium class object in place.

        """

        # set profiles
        self._profiles = profiles

        # calculate toroidal current density if not given (with psi)
        if Jtor is None:
            if psi is None:
                psi = self.psi()
            Jtor = profiles.Jtor(self.R, self.Z, psi, psi_bndry=psi_bndry)

        # set plasma boundary
        # note that the Equilibrium is passed to the boundary function
        # since the boundary may need to run the GS solver (von Hagenow's method)
        self._applyBoundary(self, Jtor, self.plasma_psi)

        # right hand side of GS equation
        rhs = -mu0 * self.R * Jtor

        # copy boundary conditions
        rhs[0, :] = self.plasma_psi[0, :]
        rhs[:, 0] = self.plasma_psi[:, 0]
        rhs[-1, :] = self.plasma_psi[-1, :]
        rhs[:, -1] = self.plasma_psi[:, -1]

        # call elliptic solver
        plasma_psi = self.callSolver(self.plasma_psi, rhs)

        self._updatePlasmaPsi(plasma_psi)

        # update plasma current
        self._current = np.sum(Jtor) * self.dR * self.dZ

    def _updatePlasmaPsi(self, plasma_psi):
        """
        Sets the plasma psi data using spline interpoation coefficients.

        Parameters
        ----------
        plasma_psi: np.array
            Plasma poloidal magnetic flux over (R,Z) [Webers/2pi].

        Returns
        -------
        None
            Modifies plasma_psi, mask, psi_axis, and psi_bndry in the class
            object in place.

        """

        # set plasma psi
        self.plasma_psi = plasma_psi

        # update spline interpolation
        self.psi_func = interpolate.RectBivariateSpline(
            self.R[:, 0], self.Z[0, :], plasma_psi
        )

        # update locations of X-points, core mask, psi ranges.
        # Note that this may fail if there are no X-points, so it should not raise an error
        # Analyse the equilibrium, finding O- and X-points
        psi = self.psi()
        opt, xpt = critical.find_critical(self.R, self.Z, psi)
        self.psi_axis = opt[0][2]

        if len(xpt) > 0:
            self.psi_bndry = xpt[0][2]
            if not hasattr(self, "mask_outside_limiter"):
                self.mask_outside_limiter = np.zeros_like(psi, dtype=float)
            self.mask = critical.inside_mask(
                self.R, self.Z, psi, opt, xpt, self.mask_outside_limiter
            )

            # Use interpolation to find if a point is in the core.
            self.mask_func = interpolate.RectBivariateSpline(
                self.R[:, 0], self.Z[0, :], self.mask
            )
        elif self._applyBoundary is fixedBoundary:
            # No X-points, but using fixed boundary
            self.psi_bndry = psi[0, 0]  # Value of psi on the boundary
            self.mask = None  # All points are in the core region
        else:
            self.psi_bndry = None
            self.mask = None

    def plot(
        self,
        axis=None,
        xpoints=True,
        opoints=True,
        wall=True,
        limiter=True,
        legend=False,
        show=True,
    ):
        """
        Plot the magnetic equilibrium flux surfaces and key geometric features.

        This method provides a convenient interface to visualize the equilibrium
        stored in the current object. It calls the `plotEquilibrium` function to
        display magnetic flux contours, separatrix, magnetic nulls (X- and O-points),
        and the tokamak boundary geometry (wall and limiter).

        Parameters
        ----------
        axis : matplotlib.axes.Axes, optional
            Axis object on which to plot. If `None`, a new figure and axis are created.
        xpoints : bool, default=True
            If True, plot magnetic X-points as red 'x' markers.
        opoints : bool, default=True
            If True, plot magnetic O-points as green '*' markers.
        wall : bool, default=True
            If True, plot the tokamak wall outline as a solid black line.
        limiter : bool, default=True
            If True, plot the limiter outline as a dashed black line.
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
        - This method is a wrapper around `plotEquilibrium(self, ...)`.
        - The equilibrium must be solved before plotting; otherwise, a warning
        will be printed.

        Examples
        --------
            eq.solve()
            axis = eq.plot(legend=True)
            axis.set_title("Plasma Equilibrium Flux Surfaces")
        """

        return plotEquilibrium(
            self,
            axis=axis,
            xpoints=xpoints,
            opoints=opoints,
            wall=wall,
            limiter=limiter,
            legend=legend,
            show=show,
        )

    def _separatrix_metrics(self):
        """
        Function that returns min/max (R,Z) points on the last closed flux surface,
        its area, and its circumference.

        Depiction here: https://imas-data-dictionary.readthedocs.io/en/latest/_downloads/23716946c6f02da1817f55b2f453444d/DefinitionEqBoundary.svg

        Parameters
        ----------

        Returns
        -------
        None
            Modifies eq object in place.
        """

        # calculate core separatrix
        core_boundary = self.separatrix()

        # min/max values on core boundary
        Rmin = np.min(core_boundary[:, 0])
        Rmax = np.max(core_boundary[:, 0])
        Zmin = np.min(core_boundary[:, 1])
        Zmax = np.max(core_boundary[:, 1])

        # find corresponding indices for these
        ZRmin_arg = np.argmin(core_boundary[:, 0])
        ZRmax_arg = np.argmax(core_boundary[:, 0])
        RZmin_arg = np.argmin(core_boundary[:, 1])
        RZmax_arg = np.argmax(core_boundary[:, 1])

        # use indices to find corresponding coords
        ZRmin = core_boundary[ZRmin_arg, 1]
        ZRmax = core_boundary[ZRmax_arg, 1]
        RZmin = core_boundary[RZmin_arg, 0]
        RZmax = core_boundary[RZmax_arg, 0]

        # use shapely to define polygon of core plasma
        plasma_polygon = sh.Polygon(core_boundary)
        area = plasma_polygon.area
        length = plasma_polygon.length

        self._sep_Rmin = Rmin
        self._sep_Rmax = Rmax
        self._sep_Zmin = Zmin
        self._sep_Zmax = Zmax
        self._sep_ZRmin = ZRmin
        self._sep_ZRmax = ZRmax
        self._sep_RZmin = RZmin
        self._sep_RZmax = RZmax
        self._sep_area = area
        self._sep_length = length


def ellipse_points(R0, Z0, A, B, N=360):
    """
    This function generates the (R,Z) points of an ellipse:
    (x/A)**2 + (y/B)**2 = 1.

    Parameters
    ----------
    R0, Z0, A, B: floats
        Centre of ellipse (R0,Z0) and the ellipticity parameters A and B.
    N: int
        Number of points to generate on the ellipse.

    Returns
    -------
    np.ndarray
        (N x 2) numpy array of points on the ellipse.
    """

    theta = np.linspace(0, 2 * np.pi, N)
    R = R0 + A * np.cos(theta)
    Z = Z0 + B * np.sin(theta)

    return np.column_stack((R, Z))


class PsiGuessGaussian:
    def __init__(
        self,
        adaptive_centre: bool = False,
        gpars: tuple[float, ...] = (0.5, 0.5, 0, 2),
    ):
        """Class to generate a Gaussian-shaped initial guess for the plasma flux.

        The flux function ψ(R, Z) is given by:

            ψ(R, Z) = exp(C - (|x - xc|^p + |y - yc|^p))

        where (xc, yc) is the center, C is a shift, and p is a power term.
        The function ensures the flux is zero on the boundaries.

        Parameters
        ----------
        adaptive_centre : bool, optional
            If True, the plasma core position (xc, yc) is adjusted dynamically.
            If False (default), it remains fixed.
        gpars : tuple (xc, yc, C, p)
            Parameters defining the flux shape:
            - xc, yc : float
                Coordinates of the flux center.
            - C : float
                Shift parameter in the exponent.
            - p : float
                Power of the distance term.
        """
        self._adaptive_centre = adaptive_centre
        self.gpars = gpars

    def __call__(self, eq):
        """Calculate the initial Gaussian-shaped psi guess.

        Returns
        -------
        np.array
            The computed flux values on the given (R, Z) grid.
        """

        # define unit meshgrid
        xx, yy = meshgrid(
            linspace(0, 1, eq.nx), linspace(0, 1, eq.ny), indexing="ij"
        )

        # finds approximate (weighted) centre of the core plasma
        if self._adaptive_centre:
            ntot = np.sum(eq.mask_inside_limiter)
            xc = (
                np.sum(
                    eq.mask_inside_limiter
                    * linspace(0, 1, eq.nx)[:, np.newaxis]
                )
                / ntot
            )
            yc = (
                np.sum(
                    eq.mask_inside_limiter
                    * linspace(0, 1, eq.ny)[np.newaxis, :]
                )
                / ntot
            )
        # else sets centre according to gpars
        else:
            xc, yc = self.gpars[:2]

        # generate the plasma flux
        psi = exp(
            self.gpars[2]
            - (
                (np.abs(xx - xc)) ** self.gpars[3]
                + (np.abs(yy - yc)) ** self.gpars[3]
            )
        )

        # set zero flux on boundary
        psi[0, :] = 0.0
        psi[:, 0] = 0.0
        psi[-1, :] = 0.0
        psi[:, -1] = 0.0

        return psi


class PsiGuessParabolicJtor:
    def __init__(self, h, k, a, b, Ip, *, psi_scale=1.0):
        """A class to generate the initial psi guess from an elliptic description
        of the plasma boundary and the plasma current.

        We assume a plasma core with a separatrix that can be described by an ellipse. From this,
        we create an elliptic distribution function E(R, Z) such that E(h, k) = 0 where (h, k) is the
        centre of the ellipse and E(R, Z) = 1 for all points on the separatrix.

        We assume a parabolic distribution of current and calculate Jtor using equation A.7 from [1]:
            Jtor = Jhat(1-E(R,Z)^i)^j
        Jhat is a constant found to match the calculated plasma current to the desired plasma current (Ip).

        Finally, we apply a linear Grad-Shafranov step to find our estimate of the initial plasma psi.

        Parameters
        ----------
        h : float
            The radial centre of the ellipse [m].
        k : float
            The vertical centre of the ellipse [m].
        a : float
            Half the length of the radial axis of the ellipse (normally the semi-major axis).
        b : float
            Half the length of the vertical axis of the ellipse (normally the semi-minor axis).
        Ip : float
            The desired plasma current [A].
        psi_scale : float
            Multiplies the psi estimate when calling this class before returning.

        References
        ----------
        [1] Wai, J. T., et al. "Feedforward equilibrium trajectory optimization with GSPulse." arXiv preprint arXiv:2506.21760 (2025).
        """
        self._h = h
        self._k = k
        self._a = a
        self._b = b
        self._Ip = Ip
        self._psi_scale = psi_scale

    @classmethod
    def from_xpoint_and_midplane(
        cls, Rx, Zx, Ri, Ro, Zm, Ip, *, psi_scale=1.0
    ):
        """Finds the elliptic distribution function that produces a boundary
        according to the desired X-point location (Rx, Zx), inner midplane (Ri, Zm),
        and outer midplane (Ro, Zm).

        This assumes that the centroid of the ellipse (where the distribution function
        is 0) occurs radially at the average of Rx, Ri, Ro.
        """
        h = (Rx + Ri + Ro) / 3
        k = Zm
        a = (Ro - Ri) / 2
        b = Zx - Zm

        return cls(h, k, a, b, Ip, psi_scale=psi_scale)

    @classmethod
    def from_separatrix_points(cls, Ip: float, *points, psi_scale=1.0):
        """Finds the elliptic distribution function that best fits the
        provided points that describe the desired core's separatrix.

        Parameters
        ----------
        Ip : float
            The desired plasma current [A].
        points : tuple[tuple[float, float], ...]
            A list of (R, Z) locations on the separatrix.

        Notes
        -----
        This method requires at least 4 isoflux points
        """
        if len(points) < 4:
            raise ValueError(
                "At least 4 isoflux points required to approximate the ellipse shape parameters"
            )

        points_np = np.array(points)
        Rpoints = points_np[:, 0]
        Zpoints = points_np[:, 1]

        def _func(args):
            h, k, a, b = args.squeeze().tolist()

            return cls._elliptic_function(Rpoints, Zpoints, h, k, a, b) - 1.0

        result = least_squares(_func, [1.0] * 4, bounds=(0.0, np.inf))

        return cls(*result.x.tolist(), Ip, psi_scale=psi_scale)

    @staticmethod
    def _elliptic_function(R, Z, h, k, a, b):
        """Evaluates the elliptic function:
            E(R, Z) = (R-h)^2/a^2 + (Z-k)^2/b^2

        Parameters
        ----------
        h : float
            The radial centre of the ellipse.
        k : float
            The vertical centre of the ellipse.
        a : float
            Half the length of the radial axis of the ellipse (normally the semi-major axis).
        b : float
            Half the length of the vertical axis of the ellipse (normally the semi-minor axis).
        """
        return (((R - h) ** 2) / a**2) + (((Z - k) ** 2) / b**2)

    def elliptic_distribution_function(self, R, Z):
        """The elliptic distribution function (E(R, Z)) for the ellipse represented by this class.

        Parameters
        ----------
        R : float | ndarray
            The radial position(s) to evaluate the distribution function at.
        Z : float | ndarray
            The vertical position(s) to evaluate the distribution function at.
        """
        return self._elliptic_function(
            R, Z, self._h, self._k, self._a, self._b
        )

    def parabolic_jtor(self, eq, Ip, *, i=1.87, j=1.5):
        """Calculate the Jtor assuming a parabolic current distribution for the given equilibrium.

        Solves the equation:
            Jtor = Jhat*(1-E(R,Z)^i)^j
        where Jhat is calculated according to the desired Ip.

        Parameters
        ----------
        eq : Equilibrium
            The equilibrium object to calculate the initial psi for
        Ip : float
            The desired plasma current.
        """
        x = self.elliptic_distribution_function(eq.R, eq.Z)

        # x > 1 indicates this grid point is outside of the core
        # because E(R, Z) = 1 at the separatrix
        outside_core_mask = x > 1
        # Set x = 1 outside the core to ensure the plasma current
        # is only distributed within the core
        # Jtor outside plasma = (1 - 1**i)**j = 0
        x[outside_core_mask] = 1.0

        Jtor_normalised = (1 - x**i) ** j
        Jtor_integrated = (
            np.sum(Jtor_normalised[~outside_core_mask]) * eq.dR * eq.dZ
        )

        Jhat = Ip / Jtor_integrated

        return Jhat * Jtor_normalised * ~outside_core_mask

    def __call__(self, eq: Equilibrium):
        """Calculates the initial psi guess by approximating the Jtor and
        applying a linear Grad-Shafranov step.
        """
        # ensure equilibrium has attribute plasma_psi which will be overwritten later
        eq.plasma_psi = np.zeros((eq.nx, eq.ny))

        jtor_guess = self.parabolic_jtor(eq, self._Ip)
        psi = eq.callSolver(psi=eq.plasma_psi, rhs=-mu0 * eq.R * jtor_guess)

        # set zero flux on boundary
        psi[0, :] = 0.0
        psi[:, 0] = 0.0
        psi[-1, :] = 0.0
        psi[:, -1] = 0.0

        return self._psi_scale * psi


if __name__ == "__main__":

    # Test the different spline interpolation routines

    import machine
    import matplotlib.pyplot as plt
    from numpy import ravel

    tokamak = machine.TestTokamak()

    Rmin = 0.1
    Rmax = 2.0

    eq = Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax)

    import constraints

    xpoints = [(1.2, -0.8), (1.2, 0.8)]
    constraints.xpointConstrain(eq, xpoints)

    psi = eq.psi()

    tck = interpolate.bisplrep(ravel(eq.R), ravel(eq.Z), ravel(psi))
    spline = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], psi)
    f = interpolate.interp2d(eq.R[:, 0], eq.Z[0, :], psi, kind="cubic")

    plt.plot(eq.R[:, 10], psi[:, 10], "o")

    r = linspace(Rmin, Rmax, 1000)
    z = eq.Z[0, 10]
    plt.plot(r, f(r, z), label="f")

    plt.plot(r, spline(r, z), label="spline")

    plt.plot(r, interpolate.bisplev(r, z, tck), label="bisplev")

    plt.legend()
    plt.show()
