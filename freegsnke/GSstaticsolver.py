"""
Module for solving the static forward and inverse GS problems. 

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

import freegs4e
import numpy as np
from freegs4e.gradshafranov import Greens

from . import nk_solver_H as nk_solver


class NKGSsolver:
    """
    Newton–Krylov solver for the nonlinear static Grad–Shafranov equilibrium problem.

    This class solves the nonlinear free-boundary Grad–Shafranov equation:

        Δψ = − μ₀ R Jtor(ψ)

    written as a nonlinear root-finding problem:

        F(ψ) = 0

    where:
        ψ          : plasma poloidal flux
        Jtor(ψ)     : toroidal plasma current density profile
        μ₀ R        : geometric operator coefficients

    The solver supports:

        • Forward equilibrium solving (fixed coil currents)
        • Inverse magnetic configuration optimisation

    Solver structure:
        The nonlinear system is solved using:
            - Newton–Krylov nonlinear iteration
            - Finite difference Jacobian approximations

    Solution domain:
        The spatial grid is fixed at instantiation using the
        provided equilibrium object.

    Notes
    -----
    The solver assumes:
        • Fixed computational domain
        • Fixed grid spacing
        • Compatible equilibrium geometry
    """

    def __init__(
        self,
        eq,
        l2_reg=1e-6,
        collinearity_reg=1e-6,
        seed=42,
    ):
        """
        Initialise the Grad–Shafranov nonlinear solver.

        The constructor prepares all numerical operators required for
        nonlinear GS solving, including:

            • Linear GS multigrid solver
            • Green's function boundary response operator
            • Newton–Krylov nonlinear solver backend
            • Random generator for Krylov direction perturbations

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Defines the computational domain and geometry.

            Required attributes:
                eq.R : 2D radial grid
                eq.Z : 2D vertical grid

            The solver grid is fixed to this domain.

        l2_reg : float, optional (default=1e-6)
            Tikhonov regularisation coefficient applied to
            nonlinear least-squares systems.

        collinearity_reg : float, optional (default=1e-6)
            Additional regularisation penalising collinear
            search directions in Krylov space.
            Improves numerical stability of NK iteration.

        seed : int, optional (default=42)
            Random number generator seed used for:
                • Krylov perturbation generation
                • Directional exploration in nonlinear solve

        Attributes
        ----------
        self.R, self.Z : ndarray
            Computational grid coordinates.

        self.nx, self.ny : int
            Grid dimensions.

        self.dRdZ : float
            Differential area element used for integration.

        self.linear_GS_solver
            Multigrid solver for linearised GS equation.

        self.greenfunc
            Boundary response Green's function matrix.

        self.nksolver
            Newton–Krylov nonlinear solver backend.

        Notes
        -----
        The solver domain cannot be changed after construction.
        A new solver instance must be created for different grids.
        """

        # store domain grid
        self.eqR = eq.R

        R = eq.R
        Z = eq.Z

        self.R = R
        self.Z = Z

        R_1D = R[:, 0]
        Z_1D = Z[0, :]

        # number of grid points
        nx, ny = np.shape(R)
        self.nx = nx
        self.ny = ny

        # grid cell area element
        # used when integrating current density and flux sources
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        self.dRdZ = dR * dZ

        # nonlinear solver backend
        self.nksolver = nk_solver.nksolver(
            problem_dimension=self.nx * self.ny,
            l2_reg=l2_reg,
            collinearity_reg=collinearity_reg,
        )

        # linear GS solver used inside nonlinear iteration
        self.linear_GS_solver = freegs4e.multigrid.createVcycle(
            nx,
            ny,
            freegs4e.gradshafranov.GSsparse4thOrder(
                eq.R[0, 0], eq.R[-1, 0], eq.Z[0, 0], eq.Z[0, -1]
            ),
            nlevels=1,
            ncycle=1,
            niter=2,
            direct=True,
        )

        # collect boundary grid indices for Dirichlet conditions
        bndry_indices = np.concatenate(
            [
                [(x, 0) for x in range(nx)],
                [(x, ny - 1) for x in range(nx)],
                [(0, y) for y in np.arange(1, ny - 1)],
                [(nx - 1, y) for y in np.arange(1, ny - 1)],
            ]
        )
        self.bndry_indices = bndry_indices

        # Compute Green's function mapping:
        #
        #   Jtor(R',Z') → ψ_boundary(R,Z)
        greenfunc = Greens(
            R[np.newaxis, :, :],
            Z[np.newaxis, :, :],
            R_1D[bndry_indices[:, 0]][:, np.newaxis, np.newaxis],
            Z_1D[bndry_indices[:, 1]][:, np.newaxis, np.newaxis],
        )

        # remove singular self-interaction terms
        zeros = np.ones_like(greenfunc)
        zeros[
            np.arange(len(bndry_indices)), bndry_indices[:, 0], bndry_indices[:, 1]
        ] = 0
        self.greenfunc = greenfunc * zeros * self.dRdZ

        # Precompute geometric RHS coefficient
        # Comes from GS equation:
        # Δψ = - μ₀ R Jtor
        self.rhs_before_jtor = -freegs4e.gradshafranov.mu0 * eq.R

        # random generator used for NK search direction exploration
        self.rng = np.random.default_rng(seed=seed)

    def freeboundary(self, plasma_psi, tokamak_psi, profiles):
        """
        Apply free-boundary Grad–Shafranov boundary conditions and compute
        plasma current source terms.

        This routine constructs the nonlinear source term and boundary
        flux conditions required to solve the Grad–Shafranov equation:

            Δψ = − μ₀ R Jtor(ψ)

        in free-boundary form.

        The algorithm performs three main tasks:

            (1) Compute toroidal current density:
                    Jtor(ψ_total)

            (2) Compute RHS source term for linear GS solve

            (3) Compute boundary flux contributions using Green's functions

        The total flux used is:

            ψ_total = ψ_tokamak + ψ_plasma

        Parameters
        ----------
        plasma_psi : ndarray
            Flattened plasma poloidal flux vector.
            Shape = (nx * ny,).

        tokamak_psi : ndarray
            Vacuum flux contribution generated by:
                • Active coils
                • Passive structures

        profiles : FreeGSNKE profile object
            Provides plasma current model Jtor(ψ).

        Returns
        -------
        None
            Updates internal solver state:

                self.jtor
                self.rhs
                self.psi_boundary

        Notes
        -----
        • This method is called before solving the linearised GS equation.
        • Boundary flux is computed using Green's function convolution.
        """

        # ------------------------------------------------------------
        # Compute toroidal current density profile
        #
        # Jtor = Jtor(ψ_total)
        #
        # This provides the nonlinear source term for GS equation:
        #
        #     Δψ = - μ₀ R Jtor
        # ------------------------------------------------------------
        self.jtor = profiles.Jtor(
            self.R,
            self.Z,
            (tokamak_psi + plasma_psi).reshape(self.nx, self.ny),
        )

        # RHS source term for linear GS solve
        # rhs_before_jtor already contains geometric operators
        self.rhs = self.rhs_before_jtor * self.jtor

        # ------------------------------------------------------------
        # Compute boundary flux via Green's function convolution
        #
        # psi_boundary = ∫ G(R,Z; R',Z') Jtor(R',Z') dR'dZ'
        #
        # Implemented using tensor contraction:
        #
        # Contract:
        #   greenfunc axis (1,2) with jtor axis (0,1)
        #
        # Result is flattened boundary flux vector.
        # ------------------------------------------------------------
        self.psi_boundary = np.zeros_like(self.R)
        psi_bnd = np.tensordot(self.greenfunc, self.jtor, axes=([1, 2], [0, 1]))

        # ------------------------------------------------------------
        # Map flattened Green's solution back to boundary grid
        # ------------------------------------------------------------
        # Vertical boundaries
        self.psi_boundary[:, 0] = psi_bnd[: self.nx]
        self.psi_boundary[:, -1] = psi_bnd[self.nx : 2 * self.nx]
        # Horizontal boundaries
        self.psi_boundary[0, 1 : self.ny - 1] = psi_bnd[
            2 * self.nx : 2 * self.nx + self.ny - 2
        ]
        self.psi_boundary[-1, 1 : self.ny - 1] = psi_bnd[2 * self.nx + self.ny - 2 :]

        # ------------------------------------------------------------
        # Impose Dirichlet boundary conditions on RHS
        # Ensures linear solver respects boundary flux constraints
        # ------------------------------------------------------------
        self.rhs[0, :] = self.psi_boundary[0, :]
        self.rhs[:, 0] = self.psi_boundary[:, 0]
        self.rhs[-1, :] = self.psi_boundary[-1, :]
        self.rhs[:, -1] = self.psi_boundary[:, -1]

    def F_function(self, plasma_psi, tokamak_psi, profiles):
        """
        Compute the nonlinear Grad–Shafranov residual written as a root-finding problem.

        The Grad–Shafranov equation is solved in residual form:

            F(ψ_plasma) = ψ_plasma − G(ψ_boundary, Jtor(ψ_plasma))

        where:
            G(...) represents the solution of the linearised GS operator
            with given boundary conditions and toroidal current source.

        The equilibrium solution satisfies:

            F(ψ_plasma) = 0

        Physically, this measures the mismatch between:
            • The current plasma flux field
            • The flux predicted by solving the linear GS equation
              with the computed toroidal current density.

        Parameters
        ----------
        plasma_psi : ndarray
            Flattened plasma poloidal flux vector.
            Shape = (nx * ny,).

        tokamak_psi : ndarray
            Vacuum flux contribution from:
                • Active coils
                • Passive conducting structures
            Same flattened shape as plasma_psi.

        profiles : freegsnke profile object
            Plasma profile model used to compute toroidal current density:
                Jtor(ψ).

        Returns
        -------
        residual : ndarray
            Nonlinear GS residual vector.
            Root of this function corresponds to GS equilibrium.

        Notes
        -----
        This formulation is used for:
            • Newton–Krylov nonlinear solves
            • Picard fixed-point iterations
            • Residual diagnostics during equilibrium solving
        """

        # ------------------------------------------------------------
        # Solve free-boundary GS problem components:
        #
        # This step computes:
        #   - Updated boundary flux
        #   - Toroidal current density source term
        #
        # Results are stored internally in:
        #   self.psi_boundary
        #   self.rhs
        # ------------------------------------------------------------
        self.freeboundary(plasma_psi, tokamak_psi, profiles)

        # ------------------------------------------------------------
        # Solve linearised GS equation:
        #
        # linear_GS_solver solves:
        #
        #     Δψ = RHS(Jtor(ψ), boundary conditions)
        #
        # and returns predicted plasma flux field.
        #
        # The residual measures the difference between:
        #     Actual ψ_plasma
        #     Predicted GS solution
        # ------------------------------------------------------------
        residual = plasma_psi - (
            self.linear_GS_solver(self.psi_boundary, self.rhs)
        ).reshape(-1)

        return residual

    def port_critical(self, eq, profiles):
        """
        Transfer critical equilibrium and topology information from the
        plasma profile solver to the equilibrium object after solving
        the Grad–Shafranov equation.

        This function synchronises diagnostic and structural information
        between the profile model and equilibrium container. It is typically
        called after a successful GS solve.

        The following information is transferred:

            • Magnetic axis (O-point)
            • X-point locations (if present)
            • Plasma boundary and limiter flags
            • Total plasma current estimate
            • Plasma profile state snapshot
            • Tokamak vacuum flux (if available)

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Equilibrium object that will be updated in-place.

        profiles : FreeGSNKE profile object
            Profile object used during GS solution. Must contain:

                - profiles.xpt : X-point locations (if diverted plasma)
                - profiles.opt : O-point / magnetic axis data
                - profiles.psi_bndry : Boundary flux value
                - profiles.flag_limiter : Limiter configuration flag
                - profiles.jtor : Toroidal current density profile

        Returns
        -------
        None
            Updates are performed in-place on the equilibrium object.

        Notes
        -----
        This routine ensures consistency between:
            - Plasma topology diagnostics
            - Flux solution storage
            - Current profile bookkeeping

        It is primarily used internally by forward and inverse solvers.
        """

        eq.solved = True

        eq.xpt = np.copy(profiles.xpt)
        eq.opt = np.copy(profiles.opt)
        eq.psi_axis = eq.opt[0, 2]

        eq.psi_bndry = profiles.psi_bndry
        eq.flag_limiter = profiles.flag_limiter

        eq._current = np.sum(profiles.jtor) * self.dRdZ
        eq._profiles = profiles.copy()

        try:
            eq.tokamak_psi = self.tokamak_psi.reshape(self.nx, self.ny)
        except:
            pass

    def relative_norm_residual(self, res, psi):
        """
        Compute a relative residual using Euclidean (L2) norm normalisation.

        This function measures the magnitude of the nonlinear GS residual
        relative to the magnitude of the plasma flux field using:

            relative_residual =
                ||res||₂
                ---------
                ||ψ||₂

        where:
            res = nonlinear Grad–Shafranov residual vector
            ψ   = plasma poloidal flux field

        This provides a scale-invariant convergence metric for nonlinear
        equilibrium solves.

        Parameters
        ----------
        res : ndarray
            Flattened nonlinear Grad–Shafranov residual vector.

        psi : ndarray
            Flattened plasma poloidal flux field.

        Returns
        -------
        float
            Dimensionless relative L2 residual.

        Notes
        -----
        • This metric is sensitive to small ||ψ|| values.
        • Used as a primary convergence indicator during nonlinear solves.
        """
        return np.linalg.norm(res) / np.linalg.norm(psi)

    def relative_del_residual(self, res, psi):
        """
        Compute a relative residual measure based on the range (max − min)
        of the residual and flux field.

        This metric measures relative variation using amplitude range
        rather than L2 norms, and is useful for detecting large-scale
        flux structure errors.

        The relative residual is defined as:

            relative_residual =
                ( max(res) − min(res) )
                ------------------------
                ( max(ψ) − min(ψ) )

        where:
            ψ = plasma poloidal flux field
            res = nonlinear GS residual vector

        This quantity is dimensionless and provides a scale-invariant
        measure of solution error magnitude.

        Parameters
        ----------
        res : ndarray
            Flattened nonlinear Grad–Shafranov residual vector.

        psi : ndarray
            Flattened plasma poloidal flux field.

        Returns
        -------
        rel_residual : float
            Relative residual defined using amplitude range normalisation.

        del_psi : float
            Flux amplitude range:
                max(ψ) − min(ψ)

        Notes
        -----
        • Range-based metrics are less sensitive to localised spikes
          than pointwise residuals.

        • If ψ has very small dynamic range, this metric may become
          ill-conditioned.

        • Used internally as a convergence and stability diagnostic.
        """

        del_psi = np.amax(psi) - np.amin(psi)
        del_res = np.amax(res) - np.amin(res)
        return del_res / del_psi, del_psi

    def forward_solve(
        self,
        eq,
        profiles,
        target_relative_tolerance,
        max_solving_iterations=100,
        Picard_handover=0.11,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.2,
        max_n_directions=16,
        max_rel_update_size=0.2,
        clip=10,
        force_up_down_symmetric=False,
        verbose=False,
        suppress=False,
    ):
        """
        Solve the forward static Grad–Shafranov (GS) equilibrium problem.

        This method computes the plasma poloidal flux ψ_plasma that satisfies
        the nonlinear static Grad–Shafranov equation:

            F(ψ_plasma; ψ_tokamak, profiles) = 0

        where:
            ψ_tokamak  : vacuum flux generated by metal currents
            profiles   : plasma current profile specification (Jtor(ψ))

        The total flux is:

            ψ_total = ψ_tokamak + ψ_plasma

        The nonlinear system is solved using a hybrid scheme:
            • Picard iterations when far from convergence
            • Newton–Krylov (NK) iterations when close to solution

        The final solution is written to:

            eq.plasma_psi

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Provides:
                - metal currents (via eq.tokamak)
                - initial guess eq.plasma_psi
                - Green's functions eq._vgreen

        profiles : FreeGSNKE profile object
            Defines plasma current model Jtor(ψ).

        target_relative_tolerance : float
            Iterations stop when the relative nonlinear residual
            falls below this value.

        max_solving_iterations : int, optional
            Maximum allowed nonlinear iterations.

        Picard_handover : float
            Relative residual threshold above which Picard
            iteration is used instead of Newton–Krylov.

        step_size : float
            Proposed NK step magnitude scaling factor.

        scaling_with_n : float
            Additional scaling applied to NK step depending
            on iteration count: (1 + n_it)**scaling_with_n.

        target_relative_unexplained_residual : float in (0,1)
            NK internal Arnoldi iteration stops once this
            fraction of the initial residual is explained.

        max_n_directions : int
            Maximum number of Krylov directions explored.

        max_rel_update_size : float
            Maximum allowed relative update to ψ_plasma.
            Larger updates are rescaled.

        clip : float
            Maximum magnitude of update contribution from
            each explored Krylov direction.

        force_up_down_symmetric : bool
            If True, enforces up-down symmetry at each step.

        verbose : bool
            Enables detailed iteration logging.

        suppress : bool
            Suppresses all print output.

        Returns
        -------
        None

        Side Effects
        ------------
        • Updates eq.plasma_psi
        • Updates profiles via Jtor(...)
        • Updates internal convergence diagnostics

        Notes
        -----
        This solver includes:
            • Adaptive initialisation
            • Residual-based step control
            • Trust-region-style update limiting
            • Automatic fallback between Picard and NK
            • Residual collinearity detection
            • Symmetry enforcement (optional)

        The algorithm is designed for robustness in highly
        nonlinear free-boundary equilibrium problems.
        """

        # suppress overrides verbose
        if suppress:
            verbose = False

        picard_flag = 0

        # ------------------------------------------------------------
        # Initial trial plasma flux
        # Optionally enforce up-down symmetry
        # ------------------------------------------------------------
        if force_up_down_symmetric:
            trial_plasma_psi = 0.5 * (eq.plasma_psi + eq.plasma_psi[:, ::-1]).reshape(
                -1
            )
            self.shape = np.shape(eq.plasma_psi)
        else:
            trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)

        # compute vacuum (metal) flux ψ_tokamak
        self.tokamak_psi = eq.tokamak.getPsitokamak(vgreen=eq._vgreen).reshape(-1)

        # Logging setup
        log = ["-----", "Forward static solve starting..."]

        # ------------------------------------------------------------
        # Validate initial guess
        # Attempt to compute residual F(ψ)
        # If failure (e.g. no O-point/core mask), scale ψ and retry
        # ------------------------------------------------------------
        control_trial_psi = False
        n_up = 0.0 + 4 * eq.solved

        while (control_trial_psi is False) and (n_up < 10):
            try:
                res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
                control_trial_psi = True
                log.append("Initial guess for plasma_psi successful, residual found.")

            except:
                trial_plasma_psi /= 0.8
                n_up += 1
                log.append("Initial guess for plasma_psi failed, trying to scale...")

        # fallback default initialization if scaling fails
        if control_trial_psi is False:
            log.append("Default plasma_psi initialisation and adjustment invoked.")
            eq.plasma_psi = trial_plasma_psi = eq.create_psi_plasma_default(
                adaptive_centre=True
            )
            eq.adjust_psi_plasma()
            trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
            res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
            control_trial_psi = True

        # store initial toroidal current profile
        self.jtor_at_start = profiles.jtor.copy()

        # ------------------------------------------------------------
        # Initial convergence diagnostics
        # ------------------------------------------------------------
        norm_rel_change = self.relative_norm_residual(res0, trial_plasma_psi)
        rel_change, del_psi = self.relative_del_residual(res0, trial_plasma_psi)
        self.relative_change = 1.0 * rel_change
        self.norm_rel_change = [norm_rel_change]

        self.best_relative_change = 1.0 * rel_change
        self.best_psi = trial_plasma_psi

        args = [self.tokamak_psi, profiles]

        # initial Krylov search direction = residual
        starting_direction = np.copy(res0)

        log.append(f"Initial relative error = {rel_change:.2e}")
        if verbose:
            for x in log:
                print(x)

        self.initial_rel_residual = 1.0 * rel_change

        log.append("-----")

        # ------------------------------------------------------------
        # Main nonlinear solve loop
        # Hybrid Picard / Newton–Krylov
        # ------------------------------------------------------------
        iterations = 0
        while (rel_change > target_relative_tolerance) * (
            iterations < max_solving_iterations
        ):

            # --------------------------------------------------------
            # Choose solver type
            # --------------------------------------------------------
            # -------- Picard iteration --------
            if rel_change > Picard_handover:
                log.append("Picard iteration: " + str(iterations))
                # using Picard instead of NK

                if picard_flag < min(max_solving_iterations - 1, 3):
                    # make picard update to the flux up-down symmetric
                    # this combats the instability of picard iterations
                    res0_2d = res0.reshape(self.nx, self.ny)
                    res0 = 0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                    picard_flag += 1
                else:
                    # update = -1.0 * res0
                    picard_flag = 1

                # standard Picard update
                update = -1.0 * res0

            # -------- Newton–Krylov iteration --------
            else:
                log.append("-----")
                log.append("Newton-Krylov iteration: " + str(iterations))
                picard_flag = False
                self.nksolver.Arnoldi_iteration(
                    x0=trial_plasma_psi.copy(),
                    dx=starting_direction.copy(),
                    R0=res0.copy(),
                    F_function=self.F_function,
                    args=args,
                    step_size=step_size,
                    scaling_with_n=scaling_with_n,
                    target_relative_unexplained_residual=target_relative_unexplained_residual,
                    max_n_directions=max_n_directions,
                    clip=clip,
                )
                update = 1.0 * self.nksolver.dx
                log.append(
                    f"...number of Krylov vectors used =  {len(self.nksolver.coeffs)}"
                )

            # --------------------------------------------------------
            # Optional symmetry enforcement
            # --------------------------------------------------------
            if force_up_down_symmetric:
                log.append("Forcing up-dpwn symmetry of the plasma.")
                update = update.reshape(self.shape)
                update = 0.5 * (update + update[:, ::-1]).reshape(-1)

            # --------------------------------------------------------
            # Trust-region style update limiting
            # --------------------------------------------------------
            del_update = np.amax(update) - np.amin(update)
            if del_update / del_psi > max_rel_update_size:
                # Reduce the size of the update as found too large
                update *= np.abs(max_rel_update_size * del_psi / del_update)
                log.append("Update too large, resized.")

            # --------------------------------------------------------
            # Attempt update
            # If critical points disappear, shrink step
            # --------------------------------------------------------
            new_residual_flag = True
            while new_residual_flag:
                try:
                    # check update does not cause the disappearance of the Opoint
                    n_trial_plasma_psi = trial_plasma_psi + update
                    new_res0 = self.F_function(
                        n_trial_plasma_psi, self.tokamak_psi, profiles
                    )
                    new_norm_rel_change = self.relative_norm_residual(
                        new_res0, n_trial_plasma_psi
                    )
                    new_rel_change, new_del_psi = self.relative_del_residual(
                        new_res0, n_trial_plasma_psi
                    )

                    new_residual_flag = False

                except:
                    log.append(
                        "Update resizing triggered due to failure to find a critical points."
                    )
                    update *= 0.75

            # --------------------------------------------------------
            # Accept or reject update
            # --------------------------------------------------------
            if new_norm_rel_change < 1.2 * self.norm_rel_change[-1]:
                # accept update
                trial_plasma_psi = n_trial_plasma_psi.copy()

                # Detect residual collinearity
                # If residuals are nearly parallel,
                # generate random direction to escape stagnation
                try:
                    residual_collinearity = np.sum(res0 * new_res0) / (
                        np.linalg.norm(res0) * np.linalg.norm(new_res0)
                    )
                    res0 = 1.0 * new_res0
                    if (residual_collinearity > 0.9) and (picard_flag is False):
                        log.append(
                            "New starting_direction used due to collinear residuals."
                        )
                        # Generate a random Krylov vector to continue the exploration
                        # This is arbitrary and can be improved
                        starting_direction = np.sin(
                            np.linspace(0, 2 * np.pi, self.nx) * 1.5 * self.rng.random()
                        )[:, np.newaxis]
                        starting_direction = (
                            starting_direction
                            * np.sin(
                                np.linspace(0, 2 * np.pi, self.ny)
                                * 1.5
                                * self.rng.random()
                            )[np.newaxis, :]
                        )
                        starting_direction = starting_direction.reshape(-1)
                        starting_direction *= trial_plasma_psi

                    else:
                        starting_direction = np.copy(res0)
                except:
                    starting_direction = np.copy(res0)
                rel_change = 1.0 * new_rel_change
                norm_rel_change = 1.0 * new_norm_rel_change
                del_psi = 1.0 * new_del_psi

            # Reject → reduce step
            else:
                reduce_by = self.relative_change / new_rel_change
                log.append("Increase in residual, update reduction triggered.")
                # log.append(reduce_by)
                new_residual_flag = True
                while new_residual_flag:
                    try:
                        n_trial_plasma_psi = trial_plasma_psi + update * reduce_by
                        res0 = self.F_function(
                            n_trial_plasma_psi, self.tokamak_psi, profiles
                        )
                        new_residual_flag = False
                    except:
                        log.append("reduction!")
                        reduce_by *= 0.75

                starting_direction = np.copy(res0)
                trial_plasma_psi = n_trial_plasma_psi.copy()
                norm_rel_change = self.relative_norm_residual(res0, trial_plasma_psi)
                rel_change, del_psi = self.relative_del_residual(res0, trial_plasma_psi)

                # track best solution encountered
                if rel_change < self.best_relative_change:
                    self.best_relative_change = 1.0 * rel_change
                    self.best_psi = np.copy(trial_plasma_psi)

            self.relative_change = 1.0 * rel_change
            self.norm_rel_change.append(norm_rel_change)
            log.append(f"...relative error =  {rel_change:.2e}")
            log.append("-----")

            if verbose:
                for x in log:
                    print(x)

            log = []

            iterations += 1

        # ------------------------------------------------------------
        # Finalise solution : update eq with new solution (compare to best on record)
        # ------------------------------------------------------------
        if self.best_relative_change < rel_change:
            self.relative_change = 1.0 * self.best_relative_change
            trial_plasma_psi = np.copy(self.best_psi)
            profiles.Jtor(
                self.R,
                self.Z,
                (self.tokamak_psi + trial_plasma_psi).reshape(self.nx, self.ny),
            )
        eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()
        self.port_critical(eq=eq, profiles=profiles)

        # ------------------------------------------------------------
        # Print output to user
        # ------------------------------------------------------------
        if not suppress:
            if rel_change > target_relative_tolerance:
                print(
                    f"Forward static solve DID NOT CONVERGE. Tolerance {rel_change:.2e} (vs. requested {target_relative_tolerance:.2e}) reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
                )
            else:
                print(
                    f"Forward static solve SUCCESS. Tolerance {rel_change:.2e} (vs. requested {target_relative_tolerance:.2e}) reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
                )

    def get_rel_delta_psit(self, delta_current, profiles, vgreen):
        """
        Estimate the relative core-region flux perturbation induced by a
        requested coil current change.

        This function computes the magnitude of the poloidal flux perturbation
        Δψ generated by a proposed change in coil currents, and expresses it
        relative to the magnitude of the existing tokamak flux.

        The induced flux perturbation is computed using the coil Green's
        functions:

            Δψ(R, Z) = Σ_i ΔI_i G_i(R, Z)

        where:
            ΔI_i       : requested current change in coil i
            G_i(R, Z)  : Green's function of coil i
            (R, Z)     : spatial grid coordinates

        The relative change metric is then defined as:

            rel_delta = || Δψ_core ||₂
                        --------------------------
                        || ψ_tokamak ||₂ + ε

        where:
            ψ_tokamak  : current total flux field
            Δψ_core    : Δψ restricted to the plasma core
            ||·||₂     : Euclidean (L2) norm
            ε          : small regularisation constant (1e-6)

        If a diverted core mask is available, only the plasma core region
        contributes to the numerator. Otherwise, the full domain is used.

        Parameters
        ----------
        delta_current : ndarray
            One-dimensional array of requested coil current changes.
            Shape (n_coils,).

        profiles : freegsnke profile object
            Provides the plasma core mask via
            `profiles.diverted_core_mask`, if available.

        vgreen : ndarray
            Array of coil Green's functions.
            Shape (n_coils, Nx, Ny), where each slice
            `vgreen[i]` corresponds to the flux response
            of coil i per unit current.

        Returns
        -------
        rel_delta_psit : float
            Dimensionless estimate of the relative magnitude of the
            induced flux perturbation.

        Notes
        -----
        • This quantity measures how large the requested current update is
        in terms of its effect on the plasma core flux.

        • It is typically used to:
            - Control step sizes in inverse solves,
            - Enforce trust-region constraints,
            - Prevent excessively large coil updates,
            - Provide stabilisation in iterative current optimisation.

        • The denominator is computed over the full domain
        (not masked), matching the current implementation.
        """

        # ------------------------------------------------------------
        # Determine the spatial mask.
        # If a diverted core mask exists, restrict computation to the
        # plasma core. Otherwise, use the full computational domain.
        # ------------------------------------------------------------
        if hasattr(profiles, "diverted_core_mask"):
            if profiles.diverted_core_mask is not None:
                core_mask = np.copy(profiles.diverted_core_mask)
            else:
                core_mask = np.ones_like(self.eqR)
        else:
            core_mask = np.ones_like(self.eqR)

        # ------------------------------------------------------------
        # Compute the flux perturbation induced by the requested
        # coil current changes.
        #
        # Mathematical form:
        #
        #   Δψ(R,Z) = Σ_i ΔI_i * G_i(R,Z)
        #
        # where:
        #   ΔI_i        = delta_current[i]
        #   G_i(R,Z)    = vgreen[i, :, :]
        #
        # Broadcasting structure:
        #   delta_current[:, None, None]   → shape (n_coils, 1, 1)
        #   vgreen                         → shape (n_coils, Nx, Ny)
        #   core_mask[None, :, :]          → shape (1, Nx, Ny)
        #
        # After multiplication and summation over axis=0,
        # we obtain Δψ over the grid (Nx, Ny).
        # ------------------------------------------------------------
        rel_delta_psit = np.linalg.norm(
            np.sum(
                delta_current[:, np.newaxis, np.newaxis]
                * vgreen
                * core_mask[np.newaxis],
                axis=0,
            )
        )

        # ------------------------------------------------------------
        # Normalise by the magnitude of the existing tokamak flux.
        #
        # This produces a dimensionless relative change:
        #
        #   rel = || Δψ_core ||₂ / || ψ_tokamak ||₂
        #
        # A small epsilon (1e-6) prevents division by zero.
        # ------------------------------------------------------------
        rel_delta_psit /= np.linalg.norm(self.tokamak_psi) + 1e-6

        return rel_delta_psit

    def get_rel_delta_psi(self, new_psi, previous_psi, profiles):
        """
        Compute the relative change between two flux states in the plasma core.

        This function measures the relative difference between two poloidal
        flux fields, restricted to the plasma core region. It is used as a
        convergence and stability metric in both forward and inverse solves.

        The relative change is defined as:

            rel_delta = || (ψ_new − ψ_old) * M ||₂
                        ---------------------------------
                        || (ψ_new + ψ_old) * M ||₂

        where:
            ψ_new, ψ_old : flattened flux vectors
            M            : binary core-region mask
            ||·||₂       : Euclidean (L2) norm

        The mask ensures that only the plasma core region contributes
        to the norm. If no core mask is available, the full domain is used.

        This symmetric normalisation (using ψ_new + ψ_old in the denominator)
        prevents bias when one field is small and provides a scale-invariant
        measure of change.

        Parameters
        ----------
        new_psi : ndarray
            Flattened array of the updated poloidal flux values.

        previous_psi : ndarray
            Flattened array of the previous poloidal flux values.
            Must be the same shape as `new_psi`.

        profiles : freegsnke profile object
            Used to obtain the plasma core mask:
                - profiles.diverted_core_mask if available
                - otherwise the entire computational domain is used.

        Returns
        -------
        rel_delta_psit : float
            Relative L2 change between the two flux fields
            restricted to the plasma core region.

        Notes
        -----
        • If `profiles.diverted_core_mask` is None or not present,
          the relative change is computed over the full domain.

        • This quantity is dimensionless.

        • Used to:
            - Detect convergence of forward solves
            - Limit coil-induced flux changes in inverse solves
            - Control damping and adaptive tolerances
        """

        # ------------------------------------------------------------
        # Determine which spatial region to use for the norm.
        # If a diverted core mask exists, use it to restrict the
        # comparison to the plasma core region.
        # Otherwise, use the full computational domain.
        # ------------------------------------------------------------
        if hasattr(profiles, "diverted_core_mask"):
            if profiles.diverted_core_mask is not None:
                # Use provided plasma core mask
                core_mask = np.copy(profiles.diverted_core_mask)
            else:
                # No mask defined → use entire domain
                core_mask = np.ones_like(self.eqR)
        else:
            # profiles has no mask attribute → use entire domain
            core_mask = np.ones_like(self.eqR)

        # Flatten mask to match flattened psi vectors
        core_mask = core_mask.reshape(-1)

        # ------------------------------------------------------------
        # Compute masked L2 norm of the difference
        # This measures absolute change in the selected region.
        # ------------------------------------------------------------
        rel_delta_psit = np.linalg.norm((new_psi - previous_psi) * core_mask)

        # ------------------------------------------------------------
        # Normalise by symmetric magnitude measure:
        # || (ψ_new + ψ_old) * mask ||₂
        #
        # This makes the metric scale-invariant and symmetric
        # with respect to the two fields.
        # ------------------------------------------------------------
        rel_delta_psit /= np.linalg.norm((new_psi + previous_psi) * core_mask)

        return rel_delta_psit

    def optimize_currents(
        self,
        eq,
        profiles,
        constrain,
        target_relative_tolerance,
        relative_psit_size=1e-3,
        l2_reg=1e-12,
        verbose=False,
    ):
        """
        Compute coil current updates using the full (plasma-aware) Jacobian.

        This routine builds the Jacobian of the magnetic constraint residual
        vector with respect to the control coil currents:

            A = d b / d I

        where:
            b  = constraint residual vector
            I  = control coil currents

        Unlike the simplified Green's-function approach (which assumes the
        plasma response is frozen), this method recomputes the full plasma
        equilibrium for each finite-difference perturbation of the control
        currents. Therefore the Jacobian includes:

            • Direct magnetic effect of coil current changes
            • Indirect effect via plasma response (GS equation)

        The Jacobian is constructed using forward finite differences:

            A[:, i] ≈ (b(I + δI_i) − b(I)) / δI_i

        Once A is constructed, the Newton step is computed by solving the
        Tikhonov-regularised least-squares problem:

            min || A ΔI + b0 ||² + ||R ΔI||²

        where:
            b0 = current constraint residual
            R  = regularisation matrix

        If current or flux limits are active, a quadratic optimisation
        routine is used instead of the closed-form normal equations.

        Parameters
        ----------
        eq : freegsnke equilibrium object
            Contains:
                - Current coil currents
                - Current plasma_psi
            Modified only through auxiliary copies inside this routine.

        profiles : freegsnke profile object
            Provides plasma properties required to solve
            the forward Grad–Shafranov problem.

        constrain : freegsnke inverse_optimizer object
            Defines:
                - Control coils
                - Constraint residual vector b
                - Loss function
                - Optional current or flux limits

        target_relative_tolerance : float
            Forward GS tolerance used when recomputing equilibria
            for each Jacobian column.

        relative_psit_size : float, optional (default=1e-3)
            Target relative change in tokamak flux used to scale the
            finite-difference perturbations δI.
            Ensures perturbations are neither too small (noise-dominated)
            nor too large (nonlinear).

        l2_reg : float or 1D array, optional (default=1e-12)
            Tikhonov regularisation factor.
            If scalar: isotropic regularisation.
            If array: per-coil regularisation weights.

        verbose : bool, optional
            If True, prints progress during Jacobian construction.

        Returns
        -------
        Newton_delta_current : ndarray
            Optimal Newton update for control coil currents.

        loss : float
            Norm of the constraint residual after applying the
            computed Newton step (predicted reduction).

        Notes
        -----
        • Computational cost scales with number of control coils
          because one full GS solve is required per column.
        • Provides quadratic convergence when close to solution.
        • Intended to be used only when already near equilibrium.
        """

        # ------------------------------------------------------------
        # Allocate storage for full Jacobian A = db/dI
        # Rows: constraint residual components
        # Columns: control coils
        # ------------------------------------------------------------
        self.dbdI = np.zeros((np.size(constrain.b), constrain.n_control_coils))

        # dummy vector used to isolate individual coil perturbations
        self.dummy_current = np.zeros(constrain.n_control_coils)

        # dummy vector used to isolate individual coil perturbations
        full_current_vec = np.copy(eq.tokamak.current_vec)

        # ------------------------------------------------------------
        # Ensure equilibrium is fully converged before building Jacobian
        # ------------------------------------------------------------
        self.forward_solve(
            eq=eq,
            profiles=profiles,
            target_relative_tolerance=target_relative_tolerance,
            suppress=True,
        )

        # compute initial constraint residual and a first current update
        # (used only to determine appropriate finite-difference scale)
        delta_current, loss = constrain.optimize_currents(
            eq=eq,
            profiles=profiles,
            full_currents_vec=full_current_vec,
            trial_plasma_psi=eq.plasma_psi,
            l2_reg=1e-12,
        )

        # store baseline constraint residual
        b0 = np.copy(constrain.b)

        # ------------------------------------------------------------
        # scale perturbation size so induced tokamak flux change
        # is approximately relative_psit_size
        # ------------------------------------------------------------
        rel_delta_psit = self.get_rel_delta_psit(
            delta_current, profiles, eq._vgreen[constrain.control_mask]
        )
        adj_factor = min(1, relative_psit_size / rel_delta_psit)
        delta_current *= adj_factor

        # ============================================================
        # Build Jacobian via finite differences
        # ============================================================
        for i in range(constrain.n_control_coils):
            if verbose:
                print(
                    f" - calculating derivatives for coil {i + 1}/{constrain.n_control_coils}"
                )

            # construct perturbation affecting only coil i
            currents = np.copy(self.dummy_current)
            currents[i] = 1.0 * delta_current[i]

            # rebuild full current vector including uncontrolled coils
            currents = full_current_vec + constrain.rebuild_full_current_vec(currents)

            # create auxiliary equilibrium to avoid overwriting main one
            self.eq2 = eq.create_auxiliary_equilibrium()
            self.eq2.tokamak.set_all_coil_currents(currents)

            # solve forward GS for perturbed currents
            self.forward_solve(
                eq=self.eq2,
                profiles=profiles,
                target_relative_tolerance=target_relative_tolerance,
                suppress=True,
            )

            # recompute constraint residual for perturbed equilibrium
            constrain.optimize_currents(
                eq=eq,
                profiles=profiles,
                full_currents_vec=currents,
                trial_plasma_psi=self.eq2.plasma_psi,
                l2_reg=1e-12,
            )

            # finite-difference derivative column
            self.dbdI[:, i] = (constrain.b - b0) / delta_current[i]

        # ============================================================
        # Construct regularisation matrix
        # ============================================================
        if isinstance(l2_reg, float):
            reg_matrix = l2_reg * np.eye(constrain.n_control_coils)
        else:
            reg_matrix = np.diag(l2_reg)

        # ============================================================
        # Solve regularised Newton system
        # ============================================================

        # if inequality constraints are active, use quadratic solver
        if (
            constrain.coil_current_limits is not None
            or constrain.psi_norm_limits is not None
        ):
            Newton_delta_current, loss = constrain.optimize_currents_quadratic(
                eq, profiles, currents, reg_matrix, A=self.dbdI, b=-b0
            )

        # otherwise solve normal equations directly
        else:
            Newton_delta_current = np.linalg.solve(
                self.dbdI.T @ self.dbdI + reg_matrix, self.dbdI.T @ -b0
            )
            loss = np.linalg.norm(b0 + np.dot(self.dbdI, Newton_delta_current))

        return Newton_delta_current, loss

    def inverse_solve(
        self,
        eq,
        profiles,
        constrain,
        target_relative_tolerance,
        target_relative_psit_update=1e-3,
        max_solving_iterations=100,
        max_iter_per_update=5,
        Picard_handover=0.15,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.3,
        max_n_directions=16,
        clip=10,
        max_rel_update_size=0.15,
        threshold_val=0.18,
        l2_reg=1e-9,
        forward_tolerance_increase=100,
        max_rel_psit=0.02,
        damping_factor=0.995,
        use_full_Jacobian=True,
        full_jacobian_handover=[1e-5, 7e-3],
        l2_reg_fj=1e-8,
        force_up_down_symmetric=False,
        verbose=False,
        suppress=False,
    ):
        """Inverse solver for static free-boundary Grad–Shafranov equilibria.

        This routine solves a coupled inverse problem:

            • Unknowns:
                - Control coil currents
                - Plasma poloidal flux (plasma_psi)

            • Constraints:
                - Magnetic constraints provided via `constrain`
                - Grad–Shafranov equation consistency
                - Target plasma profiles (via `profiles`)

        The algorithm alternates between:
            (1) Updating coil currents to better satisfy magnetic constraints.
            (2) Performing a forward GS solve to update plasma_psi for the new currents.

        Two Jacobian models are available for the current update:
            - Green's function Jacobian (plasma fixed)
            - Full Jacobian (plasma response included)

        Convergence requires BOTH:
            • GS residual < target_relative_tolerance
            • Relative tokamak flux update < target_relative_psit_update

        The solve is therefore a nonlinear block-iterative scheme coupling:
            - Coil optimisation
            - Forward equilibrium solve

        Upon success:
            • eq.plasma_psi contains the converged plasma flux
            • eq.tokamak currents contain the converged coil currents

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Contains the tokamak geometry and coil system.
            - eq.tokamak holds coil currents.
            - eq.plasma_psi holds the plasma poloidal flux.
            This object is updated in-place.

        profiles : FreeGSNKE profile object
            Specifies the desired plasma profiles, including:
                - Total plasma current
                - Pressure profile p(psi)
                - F(psi) profile
            These define the toroidal current density Jtor(psi).

        constrain : freegsnke inverse_optimizer object
            Defines:
                - Which coils are controllable
                - The magnetic constraints to be satisfied
                - The loss function measuring constraint violation

        target_relative_tolerance : float
            Target relative tolerance for the GS residual.
            The forward GS problem must satisfy this tolerance
            for the inverse problem to be considered solved.

        target_relative_psit_update : float, optional (default=1e-3)
            Maximum allowed relative change in tokamak flux (in the plasma core)
            between successive inverse iterations.
            Acts as a stability criterion.

        max_solving_iterations : int, optional (default=100)
            Maximum number of outer inverse iterations.

        max_iter_per_update : int, optional (default=5)
            Maximum number of forward GS iterations performed after each
            coil current update.

        Picard_handover : float, optional (default=0.15)
            Relative residual threshold above which Picard iteration
            is used instead of Newton–Krylov in the forward solve.

        step_size : float, optional (default=2.5)
            Scaling of exploratory steps in the forward nonlinear solve.
            Expressed in units of the residual norm.

        scaling_with_n : float, optional (default=-1.0)
            Additional scaling factor applied to exploratory steps:
                (1 + iteration_number)**scaling_with_n

        target_relative_unexplained_residual : float, optional (default=0.3)
            Forward solver termination criterion based on how much
            of the residual can be explained by current search directions.

        max_n_directions : int, optional (default=16)
            Maximum number of search directions used in forward solve.

        clip : float, optional (default=10)
            Maximum update magnitude allowed per direction
            (in units of exploratory finite-difference step).

        max_rel_update_size : float, optional (default=0.15)
            Maximum allowed relative update to plasma_psi in forward solve.
            If exceeded, the update is rescaled.

        threshold_val : float, optional (default=0.18)
            Threshold relative GS residual below which the solver
            switches to more conservative update logic.

        l2_reg : float or 1D numpy array, optional (default=1e-9)
            L2 regularisation factor applied when using the Green’s
            function Jacobian (simplified current optimisation).
            If array, must match number of control coils.

        forward_tolerance_increase : float, optional (default=100)
            Controls how tightly the forward problem is solved
            relative to the size of the coil-induced flux change.
            Smaller flux changes trigger tighter forward tolerances.

        max_rel_psit : float, optional (default=0.02)
            Maximum relative change in tokamak flux allowed
            due to coil current updates.
            Used as a step limiter.

        damping_factor : float, optional (default=0.995)
            Exponential damping applied to allowed tokamak flux change
            across iterations:
                allowed_change *= damping_factor**iteration

        use_full_Jacobian : bool, optional (default=True)
            If True, enables switching to full Jacobian optimisation
            when sufficiently close to convergence.

        full_jacobian_handover : list[float], optional (default=[1e-5, 7e-3])
            Two thresholds:
                [GS_residual_threshold, tokamak_flux_update_threshold]
            Below these values, the solver switches to full Jacobian mode.

        l2_reg_fj : float, optional (default=1e-8)
            L2 regularisation factor applied when using the full Jacobian.

        force_up_down_symmetric : bool, optional (default=False)
            If True, enforces up–down symmetry during forward solve.

        verbose : bool, optional (default=False)
            If True, prints iteration progress and diagnostic information.

        suppress : bool, optional (default=False)
            If True, suppresses all output (overrides verbose).


        Returns
        -------
        None
            The solution is written in-place into:
                eq.plasma_psi
                eq.tokamak coil currents


        Notes
        -----
        • The method is a nonlinear block-iterative inverse solver.
        • Early iterations use a frozen-plasma approximation
          for robustness and efficiency.
        • Close to convergence, the full Jacobian is used
          for quadratic-like convergence.
        • For diverted equilibria, current updates are automatically
          resized to prevent loss of X-points or O-points.
        """

        # suppress overrides verbose output
        if suppress:
            verbose = False

        if verbose:
            print("-----")
            print("Inverse static solve starting...")

        # iteration counters and damping initialisation
        iterations = 0
        damping = 1

        # track history of relative tokamak flux updates
        self.rel_psit_updates = [max_rel_psit]
        previous_rel_delta_psit = 1

        # track constraint loss history
        self.constrain_loss = []

        # whether to enforce checks that preserve X-point / O-point topology
        check_core_mask = False

        # copy full current vector (includes controlled and uncontrolled coils)
        full_currents_vec = np.copy(eq.tokamak.current_vec)

        # compute tokamak flux contribution from coils
        self.tokamak_psi = eq.tokamak.getPsitokamak(vgreen=eq._vgreen)

        # precompute Green's functions etc. needed for current optimisation
        constrain.prepare_for_solve(eq)

        check_equilibrium = False

        # ------------------------------------------------------------
        # Compute initial GS residual for current plasma + coil state
        # ------------------------------------------------------------
        try:
            GS_residual = self.F_function(
                tokamak_psi=self.tokamak_psi.reshape(-1),
                plasma_psi=eq.plasma_psi.reshape(-1),
                profiles=profiles,
            )
            if verbose:
                print(
                    "Successfully computed GS residual using initial plasma_psi and tokamak_psi guesses."
                )
            rel_change_full, del_psi = self.relative_del_residual(
                GS_residual, eq.plasma_psi
            )
            if rel_change_full < threshold_val:
                check_equilibrium = True
            if profiles.diverted_core_mask is not None:
                check_core_mask = True
        except Exception as e:
            raise RuntimeError(
                "FAILED to compute GS residual. Try modifying initial guess for plasma_psi or change some coil currents."
            ) from e

        if verbose:
            print(f"Initial relative error = {rel_change_full:.2e}")
            print("-----")

        # ============================================================
        # Main inverse iteration loop
        # ============================================================
        while (
            (rel_change_full > target_relative_tolerance)
            + (previous_rel_delta_psit > target_relative_psit_update)
        ) * (iterations < max_solving_iterations):
            if verbose:
                print("Iteration: " + str(iterations))

            # --------------------------------------------------------
            # Parameter selection depending on proximity to solution
            # --------------------------------------------------------
            if check_equilibrium:
                # adaptive restriction on tokamak flux update
                this_max_rel_psit = np.mean(self.rel_psit_updates[-6:])
                this_max_rel_update_size = 1.0 * max_rel_update_size

                # regularisation scaling
                if type(l2_reg) == float:
                    this_l2_reg = 1.0 * l2_reg
                else:
                    this_l2_reg = np.array(l2_reg)

                # allow deeper forward solves when close to convergence
                if (previous_rel_delta_psit < target_relative_psit_update) * (
                    rel_change_full < 50 * target_relative_tolerance
                ):
                    # use more iterations if 'close to solution'
                    this_max_iter_per_update = 50
                else:
                    this_max_iter_per_update = 1.0 * max_iter_per_update
            else:
                # early phase: allow larger exploratory steps
                this_max_rel_psit = False
                this_max_rel_update_size = max(max_rel_update_size, 0.3)
                this_max_iter_per_update = 1

                # reduce regularisation in early exploratory phase
                if type(l2_reg) == float:
                    this_l2_reg = 1e-4 * l2_reg
                else:
                    this_l2_reg = 1e-4 * np.array(l2_reg)

            # --------------------------------------------------------
            # Choose Jacobian model for coil current optimisation
            # --------------------------------------------------------

            # use full derivative including plasma response
            if (
                use_full_Jacobian
                * (rel_change_full < full_jacobian_handover[0])
                * (previous_rel_delta_psit < full_jacobian_handover[1])
            ):
                if verbose:
                    print(
                        "Using full Jacobian (of constraints wrt coil currents) to optimsise currents."
                    )

                # use complete Jacobian: psi_plasma changes with the coil currents
                delta_current, loss = self.optimize_currents(
                    eq=eq,
                    profiles=profiles,
                    constrain=constrain,
                    target_relative_tolerance=target_relative_tolerance,
                    relative_psit_size=this_max_rel_psit,
                    l2_reg=l2_reg_fj,
                    verbose=verbose,
                )
            # use Green's functions (plasma frozen approximation)
            else:
                if verbose:
                    print(
                        "Using simplified Green's Jacobian (of constraints wrt coil currents) to optimise the currents."
                    )
                # use Greens as Jacobian: i.e. psi_plasma is assumed fixed
                delta_current, loss = constrain.optimize_currents(
                    eq=eq,
                    profiles=profiles,
                    full_currents_vec=full_currents_vec,
                    trial_plasma_psi=eq.plasma_psi,
                    l2_reg=this_l2_reg,
                )
            self.constrain_loss.append(loss)

            # --------------------------------------------------------
            # Estimate impact of current update on tokamak flux
            # --------------------------------------------------------
            rel_delta_psit = self.get_rel_delta_psit(
                delta_current, profiles, eq._vgreen[constrain.control_mask]
            )

            # apply damping and cap relative flux change
            if this_max_rel_psit:
                # resize update to the control currents so to limit the relative change of the tokamak flux to this_max_rel_psit
                if constrain.curr_loss < 1:
                    damping *= damping_factor
                adj_factor = damping * min(1, this_max_rel_psit / rel_delta_psit)
            else:
                adj_factor = 1.0
            delta_current *= adj_factor
            previous_rel_delta_psit = rel_delta_psit * adj_factor

            # --------------------------------------------------------
            # Optional topology preservation for diverted plasmas
            # --------------------------------------------------------
            if check_core_mask:
                # make sure that the update of the control currents does not cause a loss of the Opoint or of the Xpoints
                delta_tokamak_psi = np.sum(
                    delta_current[:, np.newaxis, np.newaxis]
                    * eq._vgreen[constrain.control_mask],
                    axis=0,
                ).reshape(-1)

                resize = True
                while resize:
                    try:
                        GS_residual = self.F_function(
                            tokamak_psi=self.tokamak_psi.reshape(-1)
                            + delta_tokamak_psi,
                            plasma_psi=eq.plasma_psi.reshape(-1),
                            profiles=profiles,
                        )
                        if len(profiles.xpt):
                            # The update is approved:
                            resize = False
                    except:
                        pass

                    if resize:
                        if verbose:
                            print("Resizing of the control current update triggered!")
                        delta_current *= 0.75
                        delta_tokamak_psi *= 0.75
                        previous_rel_delta_psit *= 0.75

            self.rel_psit_updates.append(previous_rel_delta_psit)

            # --------------------------------------------------------
            # Apply coil current update
            # --------------------------------------------------------
            full_currents_vec += constrain.rebuild_full_current_vec(delta_current)
            eq.tokamak.set_all_coil_currents(full_currents_vec)
            if verbose:
                print(
                    f"Change in coil currents (being controlled): {[f'{val:.2e}' for val in delta_current]}"
                )
                print(f"Constraint losses = {loss:.2e}")
                print(
                    f"Relative update of tokamak psi (in plasma core): {previous_rel_delta_psit:.2e}"
                )

            # --------------------------------------------------------
            # Choose tolerance for forward solve
            # --------------------------------------------------------
            if previous_rel_delta_psit < target_relative_psit_update:
                tolerance = 1.0 * target_relative_tolerance
                this_max_rel_update_size = 50
            else:
                tolerance = max(
                    min(previous_rel_delta_psit / forward_tolerance_increase, 1e-3),
                    target_relative_tolerance,
                )

            # --------------------------------------------------------
            # Forward GS solve with updated currents
            # --------------------------------------------------------
            if verbose:
                print(f"Handing off to forward solve (with updated currents).")

            self.forward_solve(
                eq,
                profiles,
                target_relative_tolerance=tolerance,
                max_solving_iterations=this_max_iter_per_update,
                Picard_handover=Picard_handover,
                step_size=step_size,
                scaling_with_n=scaling_with_n,
                target_relative_unexplained_residual=target_relative_unexplained_residual,
                max_n_directions=max_n_directions,
                clip=clip,
                verbose=False,
                max_rel_update_size=this_max_rel_update_size,
                force_up_down_symmetric=force_up_down_symmetric,
                suppress=True,
            )

            # updated GS residual
            rel_change_full = 1.0 * self.relative_change
            iterations += 1

            if verbose:
                print(f"Relative error =  {rel_change_full:.2e}")
                print("-----")

            # update equilibrium proximity flag
            if rel_change_full < threshold_val:
                check_equilibrium = True
            else:
                check_equilibrium = False

        # ============================================================
        # Final reporting
        # ============================================================
        if not suppress:
            if rel_change_full > target_relative_tolerance:
                print(
                    f"Inverse static solve DID NOT CONVERGE. Tolerance {rel_change_full:.2e} (vs. requested {target_relative_tolerance}) reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
                )
            else:
                print(
                    f"Inverse static solve SUCCESS. Tolerance {rel_change_full:.2e} (vs. requested {target_relative_tolerance}) reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
                )

    def solve(
        self,
        eq,
        profiles,
        constrain=None,
        target_relative_tolerance=1e-5,
        target_relative_psit_update=1e-3,
        max_solving_iterations=100,
        max_iter_per_update=5,
        Picard_handover=0.1,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.3,
        max_n_directions=16,
        clip=10,
        max_rel_update_size=0.15,
        l2_reg=1e-9,
        forward_tolerance_increase=100,
        max_rel_psit=0.01,
        damping_factor=0.98,
        use_full_Jacobian=True,
        full_jacobian_handover=[1e-5, 7e-3],
        l2_reg_fj=1e-8,
        force_up_down_symmetric=False,
        verbose=False,
        suppress=False,
    ):
        """
        Unified entry point for solving Grad–Shafranov problems
        (forward or inverse).

        This method dispatches to either:

            • forward_solve   — fixed coil currents, solve for plasma equilibrium
            • inverse_solve   — adjust control coil currents to satisfy magnetic constraints

        The mode is determined by the `constrain` argument:

            constrain is None      → Forward equilibrium solve
            constrain is provided  → Inverse free-boundary optimisation

        ------------------------------------------------------------------------
        FORWARD MODE (constrain=None)
        ------------------------------------------------------------------------

        Solves the nonlinear free-boundary Grad–Shafranov equation:

            F(ψ, I) = 0

        for the plasma poloidal flux ψ, with fixed coil currents I.

        The solve terminates when the relative GS residual falls below
        `target_relative_tolerance` or when `max_solving_iterations` is reached.

        The result is written in-place into:
            eq.plasma_psi


        ------------------------------------------------------------------------
        INVERSE MODE (constrain provided)
        ------------------------------------------------------------------------

        Solves the PDE-constrained optimisation problem:

            min_I  ½ || b(ψ(I), I) ||²
            subject to   F(ψ(I), I) = 0

        where:
            ψ = plasma flux
            I = control coil currents
            b = magnetic constraint residual vector

        The algorithm alternates between:
            (1) Optimising control coil currents
            (2) Performing forward GS solves to update plasma response

        Convergence requires BOTH:
            • GS residual < target_relative_tolerance
            • Relative tokamak flux update < target_relative_psit_update

        Upon convergence:
            eq.plasma_psi contains the final plasma solution
            eq.tokamak contains the optimised coil currents


        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Contains:
                - Tokamak geometry and coils
                - Current coil currents
                - Current plasma flux (initial guess)
            Modified in-place.

        profiles : FreeGSNKE profile object
            Defines plasma profiles used to compute toroidal current density:
                - Pressure profile p(ψ)
                - F(ψ) profile
                - Total plasma current

        constrain : freegsnke inverse_optimizer object or None, optional
            If None:
                Forward equilibrium solve is performed.
            If provided:
                Specifies control coils and magnetic constraints
                for inverse optimisation.

        target_relative_tolerance : float, optional (default=1e-5)
            Target relative GS residual tolerance for forward solves.
            In inverse mode, this is the required final equilibrium accuracy.

        target_relative_psit_update : float, optional (default=1e-3)
            (Inverse mode only)
            Maximum allowed relative change in tokamak flux between
            successive current updates.

        max_solving_iterations : int, optional (default=100)
            Maximum number of outer iterations.
            In forward mode: maximum GS iterations.
            In inverse mode: maximum inverse iterations.

        max_iter_per_update : int, optional (default=5)
            (Inverse mode only)
            Maximum forward GS iterations performed after each
            coil current update.

        Picard_handover : float, optional (default=0.1)
            Relative GS residual threshold above which Picard iteration
            is used instead of Newton–Krylov in forward solves.

        step_size : float, optional (default=2.5)
            Scaling factor for nonlinear update steps in forward solve.
            Expressed in units of residual norm.

        scaling_with_n : float, optional (default=-1.0)
            Additional scaling of nonlinear steps as:
                (1 + iteration_number)**scaling_with_n

        target_relative_unexplained_residual : float, optional (default=0.3)
            Forward solver termination criterion based on how much
            of the residual is explained by accumulated search directions.

        max_n_directions : int, optional (default=16)
            Maximum number of search directions in forward solve.

        clip : float, optional (default=10)
            Maximum update magnitude per search direction
            in forward solve.

        max_rel_update_size : float, optional (default=0.15)
            Maximum allowed relative change in plasma_psi per forward iteration.
            Updates exceeding this are rescaled.

        l2_reg : float or 1D ndarray, optional (default=1e-9)
            (Inverse mode only)
            Tikhonov regularisation applied when using
            Green’s-function Jacobians for coil optimisation.

        forward_tolerance_increase : float, optional (default=100)
            (Inverse mode only)
            Controls how tightly forward problems are solved relative to
            the magnitude of coil-induced flux changes.

        max_rel_psit : float, optional (default=0.01)
            (Inverse mode only)
            Maximum allowed relative tokamak flux change due to
            a single coil current update.

        damping_factor : float, optional (default=0.98)
            (Inverse mode only)
            Exponential damping applied to the allowed tokamak flux change
            across iterations to encourage convergence.

        use_full_Jacobian : bool, optional (default=True)
            (Inverse mode only)
            If True, switches to full plasma-aware Jacobian when sufficiently
            close to convergence.

        full_jacobian_handover : list[float], optional (default=[1e-5, 7e-3])
            (Inverse mode only)
            Thresholds [GS_residual, tokamak_flux_update] below which
            full Jacobian optimisation is activated.

        l2_reg_fj : float, optional (default=1e-8)
            (Inverse mode only)
            Regularisation factor applied when full Jacobian is used.

        force_up_down_symmetric : bool, optional (default=False)
            If True, enforces up–down symmetry during forward solves.

        verbose : bool, optional (default=False)
            If True, prints iteration progress information.

        suppress : bool, optional (default=False)
            If True, suppresses all printed output.


        Returns
        -------
        None
            The solution is written in-place into:
                eq.plasma_psi
                eq.tokamak coil currents (inverse mode only)


        Notes
        -----
        • This is the recommended high-level API for solving equilibria.
        • Forward mode solves a nonlinear free-boundary PDE.
        • Inverse mode solves a nonlinear PDE-constrained least-squares problem.
        • Internally dispatches to `forward_solve` or `inverse_solve`.
        """

        # ensure vectorised currents are in place in tokamak object
        eq.tokamak.getCurrentsVec()
        eq._separatrix_data_flag = False

        # ============================================================
        # Forward GS solve
        # ============================================================
        if constrain is None:
            self.forward_solve(
                eq=eq,
                profiles=profiles,
                target_relative_tolerance=target_relative_tolerance,
                max_solving_iterations=max_solving_iterations,
                Picard_handover=Picard_handover,
                step_size=step_size,
                scaling_with_n=scaling_with_n,
                target_relative_unexplained_residual=target_relative_unexplained_residual,
                max_n_directions=max_n_directions,
                clip=clip,
                verbose=verbose,
                max_rel_update_size=max_rel_update_size,
                force_up_down_symmetric=force_up_down_symmetric,
                suppress=suppress,
            )
        # ============================================================
        # Inverse GS solve
        # ============================================================
        else:
            self.inverse_solve(
                eq=eq,
                profiles=profiles,
                constrain=constrain,
                target_relative_tolerance=target_relative_tolerance,
                target_relative_psit_update=target_relative_psit_update,
                max_solving_iterations=max_solving_iterations,
                max_iter_per_update=max_iter_per_update,
                Picard_handover=Picard_handover,
                step_size=step_size,
                scaling_with_n=scaling_with_n,
                target_relative_unexplained_residual=target_relative_unexplained_residual,
                max_n_directions=max_n_directions,
                clip=clip,
                max_rel_update_size=max_rel_update_size,
                l2_reg=l2_reg,
                forward_tolerance_increase=forward_tolerance_increase,
                # forward_tolerance_increase_factor=forward_tolerance_increase_factor,
                max_rel_psit=max_rel_psit,
                damping_factor=damping_factor,
                full_jacobian_handover=full_jacobian_handover,
                use_full_Jacobian=use_full_Jacobian,
                l2_reg_fj=l2_reg_fj,
                force_up_down_symmetric=force_up_down_symmetric,
                verbose=verbose,
                suppress=suppress,
            )
