"""
Implements the optimiser for the inverse Grad-Shafranov problem.

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

import cvxpy
import numpy as np
from scipy import interpolate


class Inverse_optimizer:
    """This class implements a gradient based optimiser for the coil currents,
    used to perform (static) inverse GS solves.
    """

    def __init__(
        self,
        isoflux_set=None,
        null_points=None,
        psi_vals=None,
        coil_current_limits=None,
        psi_norm_limits=None,
    ):
        """
        Initialise magnetic constraint definitions for inverse equilibrium optimisation.

        This object stores magnetic configuration constraints that are enforced
        during inverse Grad–Shafranov optimisation.

        Constraints supported include:

            • Isoflux constraints
            • Magnetic axis (O-point) and X-point constraints
            • Direct flux value constraints
            • Coil current bound constraints
            • Normalised flux inequality constraints

        These constraints are used to construct objective and penalty terms
        during nonlinear current optimisation.

        Parameters
        ----------
        isoflux_set : list or ndarray, optional
            Collection of isoflux constraint objects.

            Each isoflux constraint is specified as:
                [Rcoords, Zcoords]

            where:
                Rcoords : 1D array of radial coordinates
                Zcoords : 1D array of vertical coordinates

            All specified points within each set are required to share
            the same poloidal flux value.

        null_points : list or ndarray, optional
            Magnetic null point constraints.

            Structure:
                [Rcoords, Zcoords]

            Specifies coordinates of:
                • X-points
                • O-points (magnetic axes)

        psi_vals : list or ndarray, optional
            Direct flux value constraints.

            Structure:
                [Rcoords, Zcoords, psi_values]

            where:
                Rcoords, Zcoords, psi_values must have identical shapes.

            Used to enforce ψ(R,Z) = ψ_target at specified locations.

        coil_current_limits : list, optional
            Hard inequality bounds on coil currents.

            Structure:
                [upper_limits, lower_limits]

            Each entry is a list with length equal to the number of
            controllable coils.

            Example:
                [
                    [Imax1, Imax2, ...],
                    [Imin1, Imin2, ...]
                ]

            Use None to indicate no bound.

        psi_norm_limits : list or ndarray, optional
            Normalised flux inequality constraints.

            Structure:
                [Rcoord, Zcoord, normalised_psi_value, constraint_sign]

            Constraint form:

                If constraint_sign = 1:
                    ψ_norm ≥ ψ_target

                If constraint_sign ≠ 1:
                    ψ_norm ≤ ψ_target

            Normalised flux is defined:
                ψ_norm = (ψ − ψ_axis) / (ψ_boundary − ψ_axis)

        """

        # ------------------------------------------------------------
        # Isoflux constraint processing
        # ------------------------------------------------------------
        self.isoflux_set = isoflux_set

        if isoflux_set is not None:

            # Test if structure is already nested numeric arrays
            # If indexing succeeds, we assume isoflux_set contains
            # structured coordinate arrays.
            try:
                type(self.isoflux_set[0][0][0])
                self.isoflux_set = []
                for isoflux in isoflux_set:
                    self.isoflux_set.append(np.array(isoflux))
            # rebuild as list of numpy arrays for numerical stability
            except:
                self.isoflux_set = np.array(self.isoflux_set)[np.newaxis]

            # number of isoflux points per constraint set
            self.isoflux_set_n = [len(isoflux[0]) for isoflux in self.isoflux_set]

        # ------------------------------------------------------------
        # Null point constraints (X-points, O-points)
        # ------------------------------------------------------------
        self.null_points = null_points
        if self.null_points is not None:
            self.null_points = np.array(self.null_points)

        # ------------------------------------------------------------
        # Direct flux value constraints
        # These impose ψ(R,Z) = ψ_target at specified locations
        # ------------------------------------------------------------
        self.psi_vals = psi_vals
        if self.psi_vals is not None:

            # Flag indicating that constraint is not defined on full grid
            self.full_grid = False
            self.psi_vals = np.array(self.psi_vals)

            # Reshape to:
            #   [Rcoords, Zcoords, psi_values]
            self.psi_vals = self.psi_vals.reshape((3, -1))

            # Remove arbitrary vertical offset
            # This improves numerical conditioning because GS equations
            # are invariant under constant vertical flux shifts.
            self.psi_vals[2] -= np.mean(self.psi_vals[2])

            # Store magnitude scale of flux constraints
            # Used for normalisation in optimisation loss
            self.norm_psi_vals = np.linalg.norm(self.psi_vals[2])

        # ------------------------------------------------------------
        # Coil current bounds and penalty regularisation weights
        # ------------------------------------------------------------
        self.coil_current_limits = coil_current_limits
        self.mu_coils = 1e5

        # ------------------------------------------------------------
        # Normalised psi bounds and penalty regularisation weights
        # ------------------------------------------------------------
        self.psi_norm_limits = (
            None if psi_norm_limits is None else np.array(psi_norm_limits)
        )
        self.mu_psi_norm = 1e6

    def prepare_for_solve(self, eq):
        """
        Prepare constraint solver quantities after object instantiation.

        This method must be called before performing optimisation or
        nonlinear solve steps. It builds:

            • Control coil selection masks
            • Green's function response operators

        These quantities are required for computing:
            - Constraint residuals
            - Gradient/Jacobian approximations
            - Loss function evaluations during optimisation

        Parameters
        ----------
        eq : freegsnke equilibrium object
            Equilibrium object providing:

                • Coil geometry
                • Coil current state
                • Magnetic Green's function operators

        Returns
        -------
        None
            Updates internal constraint solver state.
        """

        self.build_control_coils(eq)
        self.build_greens(eq)

    def source_domain_properties(self, eq):
        """
        Source and cache computational domain geometry from the equilibrium object.

        This method extracts and stores the spatial grid coordinates used for
        solving the Grad–Shafranov equation.

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Equilibrium object providing computational domain information:

                • eq.R : 2D radial grid coordinates
                • eq.Z : 2D vertical grid coordinates

        Returns
        -------
        None
            Stores domain geometry internally.

        """
        self.eqR = eq.R
        self.eqZ = eq.Z

    def build_control_coils(self, eq):
        """
        Identify and cache coil systems available for active control.

        This method constructs internal indexing and masking structures
        that separate:

            • Control coils (actively optimised)
            • Passive coils (fixed currents or structures)

        These mappings are required for:

            - Current optimisation
            - Jacobian construction
            - Constraint enforcement
            - Vectorised current updates

        Parameters
        ----------
        eq : freegsnke equilibrium object
            Provides tokamak coil system information.

        Expected coil attributes:
            eq.tokamak.coils : list of coil objects
            Each coil must have:
                - coil.control : bool indicating controllability

        Returns
        -------
        None
            Updates solver internal coil control state.

        """

        # extract coils that are actively controllable
        # each coil is stored as (label, coil_object)
        self.control_coils = [
            (label, coil) for label, coil in eq.tokamak.coils if coil.control
        ]

        # Boolean mask indicating which coils are controllable
        self.control_mask = np.array(
            [coil.control for label, coil in eq.tokamak.coils]
        ).astype(bool)

        # complement mask (passive / fixed coils)
        self.no_control_mask = np.logical_not(self.control_mask)

        # number of actively optimised coils
        self.n_control_coils = np.sum(self.control_mask)

        # preserve physical coil ordering from equilibrium object
        self.coil_order = eq.tokamak.coil_order

        # total number of coils in system
        self.n_coils = len(eq.tokamak.coils)

        # dummy vector used for building perturbation current vectors (for Jacobians)
        self.full_current_dummy = np.zeros(self.n_coils)

        # cache domain geometry (PDE solver grid)
        self.source_domain_properties(eq)

    def build_control_currents(self, eq):
        """
        Extract coil current values for controllable coils from the equilibrium object.

        This method builds a reduced current vector containing only the
        currents of actively optimised control coils.

        Parameters
        ----------
        eq : freegsnke equilibrium object
            Equilibrium object providing current coil state via:

                eq.tokamak.getCurrentsVec()

        Returns
        -------
        None
            Stores result in:
                self.control_currents

        """

        self.control_currents = eq.tokamak.getCurrentsVec(coils=self.control_coils)

    def build_control_currents_Vec(self, full_currents_vec):
        """
        Extract controllable coil currents from a full coil current vector.

        This function performs projection:

            I_control = M_control I_full

        where M_control is a boolean masking operator selecting
        controllable coils.

        Parameters
        ----------
        full_currents_vec : ndarray
            Full vector of coil currents.

            Example:
                eq.tokamak.getCurrentsVec()

        Returns
        -------
        None
            Stores result in:
                self.control_currents
        """

        self.control_currents = full_currents_vec[self.control_mask]

    def build_full_current_vec(self, eq):
        """
        Build full coil current vector from equilibrium object.

        This retrieves all coil currents, including:

            • Control coils
            • Passive structure coils

        Parameters
        ----------
        eq : freegsnke equilibrium object
            Provides coil current state via:

                eq.tokamak.getCurrentsVec()

        Returns
        -------
        None
            Stores result in:
                self.full_currents_vec
        """
        self.full_currents_vec = eq.tokamak.getCurrentsVec()

    def rebuild_full_current_vec(self, control_currents, filling=0):
        """
        Reconstruct a full coil current vector from control coil values.

        This performs the inverse mapping:

            I_full = P I_control + I_background

        where:
            P = projection operator embedding control currents
                into full coil system ordering.

        Non-controlled coils are assigned the value specified by:
            filling

        Parameters
        ----------
        control_currents : ndarray
            Current values for actively controlled coils.

        filling : float, optional
            Default value used to fill non-controlled coil entries.

        Returns
        -------
        full_current_vec : ndarray
            Complete coil current vector in system ordering.
        """

        full_current_vec = filling * np.ones_like(self.full_current_dummy)
        for i, current in enumerate(control_currents):
            full_current_vec[self.coil_order[self.control_coils[i][0]]] = current
        return full_current_vec

    def build_greens(self, eq):
        """
        Construct and cache magnetic Green's function response operators.

        This method precomputes magnetic response matrices used for:

            • Flux matching constraints
            • Null point field constraints
            • Isoflux surface optimisation
            • Flux inequality constraints

        Green's functions represent linear mappings between:

            Coil current perturbations →
            Magnetic field / flux perturbations

        These operators are used during:
            - Constraint optimisation
            - Jacobian approximation
            - Inverse equilibrium solving

        Parameters
        ----------
        eq : freegsnke equilibrium object
            Provides access to tokamak geometry and magnetic response
            calculation routines.

        Returns
        -------
        None
            Stores Green's function operators internally.

        """

        # ------------------------------------------------------------
        # Isoflux constraint Green's functions
        #
        # For each isoflux contour:
        #
        #   G = ψ response matrix
        #
        # Compute pairwise flux differences:
        #
        #   dG_ij = G_i - G_j
        #
        # Used to enforce equal-flux constraints across contours.
        # ------------------------------------------------------------
        if self.isoflux_set is not None:

            self.dG_set = []
            self.mask_set = []

            for i, isoflux in enumerate(self.isoflux_set):

                # compute flux response to unit coil currents
                G = eq.tokamak.createPsiGreensVec(R=isoflux[0], Z=isoflux[1])

                # Upper triangular mask selects unique pairwise constraints
                # Avoids redundant symmetric entries.
                mask = np.triu(
                    np.ones((self.isoflux_set_n[i], self.isoflux_set_n[i])), k=1
                ).astype(bool)
                self.mask_set.append(mask)

                # Ccmpute pairwise flux differences
                dG = G[:, :, np.newaxis] - G[:, np.newaxis, :]

                # flatten masked pairwise responses
                self.dG_set.append(dG[:, mask])

        # ------------------------------------------------------------
        # Magnetic null point Green's functions
        #
        # Used to constrain:
        #   • X-point locations
        #   • O-point locations
        # ------------------------------------------------------------
        if self.null_points is not None:

            # compute radial magnetic field response to unit coil currents
            self.Gbr = eq.tokamak.createBrGreensVec(
                R=self.null_points[0], Z=self.null_points[1]
            )
            # compute vertical magnetic field response to unit coil currents
            self.Gbz = eq.tokamak.createBzGreensVec(
                R=self.null_points[0], Z=self.null_points[1]
            )

        # ------------------------------------------------------------
        # Flux value constraint Green's functions
        # ------------------------------------------------------------

        if self.psi_vals is not None:

            # detect if psi constraints are defined on full grid
            if (
                self.psi_vals[0].shape == eq.R_1D.shape
                and self.psi_vals[1].shape == eq.Z_1D.shape
                and np.all(self.psi_vals[0] == eq.R_1D)
                and np.all(self.psi_vals[1] == eq.Z_1D)
            ):
                self.full_grid = True
                self.G = np.copy(eq._vgreen).reshape((self.n_coils, -1))

            # sparse or pointwise constraint case
            else:
                self.G = eq.tokamak.createPsiGreensVec(
                    R=self.psi_vals[0], Z=self.psi_vals[1]
                )

        # ------------------------------------------------------------
        # Normalised flux inequality constraint Green's functions
        # ------------------------------------------------------------
        if self.psi_norm_limits is not None:

            # compute flux response to unit coil currents
            self.G_psi_norm = eq.tokamak.createPsiGreensVec(
                R=self.psi_norm_limits[:, 0], Z=self.psi_norm_limits[:, 1]
            )

    def build_plasma_vals(self, trial_plasma_psi):
        """
        Compute and cache plasma-dependent magnetic quantities from a candidate flux solution.

        This method evaluates the plasma flux field and derives quantities required for
        constraint optimisation and nonlinear solve diagnostics, including:

            • Magnetic field components at null points
            • Isoflux surface flux differences
            • Pointwise flux constraint values

        The plasma flux field is interpolated using bicubic spline interpolation.

        Parameters
        ----------
        trial_plasma_psi : ndarray
            Candidate plasma poloidal flux solution.

            Must have shape compatible with:
                len(self.eqR) × len(self.eqZ)

        Returns
        -------
        None
            Updates internal solver state variables:

                self.brp
                self.bzp
                self.d_psi_plasma_vals_iso
                self.psi_plasma_vals

        """

        # interpolate current plasma flux map
        psi_func = interpolate.RectBivariateSpline(
            self.eqR[:, 0], self.eqZ[0, :], trial_plasma_psi
        )

        # ------------------------------------------------------------
        # Magnetic field evaluation at null points
        #
        # GS magnetic field relations:
        #
        #   B_R = - (1/R) ∂ψ/∂Z
        #   B_Z =   (1/R) ∂ψ/∂R
        #
        # dx=1 requests derivative with respect to R
        # dy=1 requests derivative with respect to Z
        # ------------------------------------------------------------
        if self.null_points is not None:
            # radial magnetic field component
            self.brp = (
                -psi_func(self.null_points[0], self.null_points[1], dy=1, grid=False)
                / self.null_points[0]
            )
            # vertical magnetic field component
            self.bzp = (
                psi_func(self.null_points[0], self.null_points[1], dx=1, grid=False)
                / self.null_points[0]
            )

        # ------------------------------------------------------------
        # Isoflux contour constraint evaluation
        #
        # For each contour:
        #
        #   Δψ_ij = ψ_i - ψ_j
        #
        # Used to enforce equal-flux topology constraints.
        # ------------------------------------------------------------
        if self.isoflux_set is not None:
            # loop over each set of isoflux constraints
            self.d_psi_plasma_vals_iso = []
            for i, isoflux in enumerate(self.isoflux_set):

                # evaluate flux along contour coordinates
                plasma_vals = psi_func(isoflux[0], isoflux[1], grid=False)

                # compute pairwise flux differences
                d_plasma_vals = plasma_vals[:, np.newaxis] - plasma_vals[np.newaxis, :]

                # apply triangular mask to remove redundant symmetric pairs
                self.d_psi_plasma_vals_iso.append(d_plasma_vals[self.mask_set[i]])

        # ------------------------------------------------------------
        # Direct flux value constraints
        #
        # These constraints enforce:
        #
        #       ψ(R_i, Z_i) ≈ ψ_target_i
        #
        # at specified spatial locations.
        # ------------------------------------------------------------
        if self.psi_vals is not None:
            # evaluate flux on each point
            if self.full_grid:
                self.psi_plasma_vals = trial_plasma_psi.reshape(-1)
            else:
                self.psi_plasma_vals = psi_func(
                    self.psi_vals[0], self.psi_vals[1], grid=False
                )

    def build_isoflux_lsq(self, full_currents_vec):
        """
        Construct linear least-squares system for enforcing isoflux magnetic constraints.

        This method assembles the linear optimisation problem:

            A I_control ≈ b

        where:

            A = Green's function response matrix
            I_control = vector of controllable coil currents
            b = plasma + vacuum flux vector

        The least-squares system is derived from:

            ψ_total = ψ_tokamak + ψ_plasma

        and enforcing:

            ψ_i − ψ_j ≈ constant
            (equal-flux constraints along magnetic surfaces)

        Parameters
        ----------
        full_currents_vec : ndarray
            Full vector of all coil currents.

            Example:
                eq.tokamak.getCurrentsVec()

        Returns
        -------
        A : list of ndarray
            List of arrays for each isoflux constraint set.

        b : list of ndarray
            Right-hand side residual vectors.

        loss : list of float
            Constraint violation magnitudes (L2 norms).

        Mathematical formulation
        ------------------------
        For each isoflux constraint set:

            A_i = (dG_i)ᵀ P_control

            b_i = − ( Σ_k dG_i I_k + Δψ_plasma )

        where:
            dG_i = pairwise Green's function flux differences
            P_control = coil control projection operator
        """

        loss = []
        A = []
        b = []

        # loop over each isoflux set
        for i, isoflux in enumerate(self.isoflux_set):

            # pairwise Greens' flux differences (only in control coils)
            A.append(self.dG_set[i][self.control_mask].T)

            # tokamak flux contribution
            b_val = np.sum(self.dG_set[i] * full_currents_vec[:, np.newaxis], axis=0)

            # add the plasma flux contribution
            b_val += self.d_psi_plasma_vals_iso[i]

            # total
            b.append(-b_val)

            # constraint violation magnitude
            loss.append(np.linalg.norm(b_val))

        return A, b, loss

    def build_null_points_lsq(self, full_currents_vec):
        """
        Construct a least-squares system enforcing magnetic null-point constraints.

        This method builds the linear optimisation system:

            A I_control ≈ b

        for magnetic field cancellation at prescribed null points.

        Null point constraints enforce:

            B_R(R_i, Z_i) ≈ 0
            B_Z(R_i, Z_i) ≈ 0

        where:
            B_R = radial magnetic field component
            B_Z = vertical magnetic field component

        These fields are expressed using Green's function response matrices:

            B(R,Z) = Σ_k G_k(R,Z) I_k + B_plasma

        Parameters
        ----------
        full_currents_vec : ndarray
            Full vector of coil currents.

            Example:
                eq.tokamak.getCurrentsVec()

        Returns
        -------
        A : ndarray
            Combined Jacobian matrix for null-point field constraints.

        b : ndarray
            Residual field mismatch vector (negated for optimisation solving).

        loss : list of float
            Constraint violation magnitudes:

                [ ||B_R||₂ , ||B_Z||₂ ]

        Notes
        -----
        This formulation is used to stabilise:

            • Magnetic axis positioning
            • X-point control
            • Divertor topology shaping

        Mathematical formulation
        ------------------------
        The optimisation problem solved is:

            min_I || G I + B_plasma ||²

        where:
            G = magnetic Green's response operator
            I = coil current vector
        """

        # radial field constraint
        A_r = self.Gbr[self.control_mask].T
        b_r = np.sum(self.Gbr * full_currents_vec[:, np.newaxis], axis=0)
        b_r += self.brp
        loss = [np.linalg.norm(b_r)]

        # vertical field constraint
        A_z = self.Gbz[self.control_mask].T
        b_z = np.sum(self.Gbz * full_currents_vec[:, np.newaxis], axis=0)
        b_z += self.bzp
        loss.append(np.linalg.norm(b_z))

        # stack contraints
        A = np.concatenate((A_r, A_z), axis=0)
        b = -np.concatenate((b_r, b_z), axis=0)

        return A, b, loss

    def build_psi_vals_lsq(self, full_currents_vec):
        """
        Construct a least-squares system enforcing direct flux value constraints.

        This method builds the optimisation system:

            A I_control ≈ b

        where:

            ψ_model(R_i, Z_i) = ψ_target(R_i, Z_i)

        is enforced by matching magnetic flux values at specified locations.

        The optimisation residual is defined as:

            b = ψ_tokamak(I) + ψ_plasma
                - ψ_target
                - ⟨b⟩

        Mean flux removal is applied to remove arbitrary vertical flux offsets,
        since the Grad–Shafranov equation is invariant under constant flux shifts.

        Parameters
        ----------
        full_currents_vec : ndarray
            Full coil current vector.

            Example:
                eq.tokamak.getCurrentsVec()

        Returns
        -------
        A : ndarray
            Jacobian matrix mapping coil current perturbations → flux changes.

        b : ndarray
            Flux mismatch residual vector.

        normalised_loss : list of float
            Normalised constraint violation magnitude.

        Notes
        -----
        This constraint formulation is commonly used for:

            • Magnetic axis pinning
            • Boundary flux matching
            • Profile shape control

        Mathematical formulation
        ------------------------
        Solve:

            min_I || G I + ψ_plasma − ψ_target ||²
        """

        # flux response wrt coil currents
        A = self.G[self.control_mask].T

        # tokamak coil flux
        b = np.sum(self.G * full_currents_vec[:, np.newaxis], axis=0)
        # add plasma flux
        b += self.psi_plasma_vals

        # subtract mean value as Gs invariant to constant flux shifts
        b -= np.mean(b)
        b -= self.psi_vals[2]
        b *= -1

        # normalised loss
        normalised_loss = np.linalg.norm(b) / self.norm_psi_vals

        return A, b, [normalised_loss]

    def build_lsq(self, full_currents_vec):
        """
        Assemble the global least-squares optimisation system combining all
        active magnetic and control constraints.

        This method aggregates all available constraint types into a single
        linear optimisation problem of the form:

            A I_control ≈ b

        where:

            A = stacked Jacobian / response matrices
            b = stacked residual constraint vectors

        The constraint types that can be combined include:

            • Isoflux surface constraints
            • Magnetic null-point field constraints
            • Direct flux value constraints
            • Direct coil current constraints

        Parameters
        ----------
        full_currents_vec : ndarray
            Full vector of coil currents.

            Example:
                eq.tokamak.getCurrentsVec()

        Returns
        -------
        None
            Stores assembled optimisation system internally:

                self.A
                self.b
                self.loss

        Notes
        -----
        This formulation solves the global constrained optimisation problem:

            min_I || A I − b ||²

        where each constraint type contributes a block to the system.

        Constraint dimensional bookkeeping is stored as:

            self.isoflux_dim
            self.nullp_dim
            self.psiv_dim
            self.curr_dim
        """

        # storage
        loss = 0
        A = np.empty(shape=(0, self.n_control_coils))
        b = np.empty(shape=0)
        loss = []

        # isfolux constrains
        if self.isoflux_set is not None:
            A_i, b_i, l = self.build_isoflux_lsq(full_currents_vec)
            A = np.concatenate(A_i, axis=0)
            b = np.concatenate(b_i, axis=0)
            self.isoflux_dim = len(b)
            loss = loss + l

        # null point constraints
        if self.null_points is not None:
            A_np, b_np, l = self.build_null_points_lsq(full_currents_vec)
            A = np.concatenate((A, A_np), axis=0)
            b = np.concatenate((b, b_np), axis=0)
            self.nullp_dim = len(b)
            loss = loss + l

        # direct flux value constraints
        if self.psi_vals is not None:
            A_pv, b_pv, l = self.build_psi_vals_lsq(full_currents_vec)
            A = np.concatenate((A, A_pv), axis=0)
            b = np.concatenate((b, b_pv), axis=0)
            self.psiv_dim = len(b)
            loss = loss + l

        # assemble the full system
        self.A = np.copy(A)
        self.b = np.copy(b)
        self.loss = np.array(loss)

    def optimize_currents(
        self, eq, profiles, full_currents_vec, trial_plasma_psi, l2_reg
    ):
        """
        Solve the constrained least-squares optimisation problem for coil current updates.

        This method computes optimal coil current corrections by solving:

            min_I || A I − b ||² + λ || I ||²

        where:

            A = combined constraint Jacobian matrix
            b = combined constraint residual vector
            λ = Tikhonov (L2) regularisation parameter

        The optimisation accounts for:

            • Magnetic topology constraints
            • Flux surface matching
            • Null point field cancellation
            • Hardware current constraints

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Equilibrium providing geometry and tokamak configuration.

        profiles : FreeGSNKE profile object
            Plasma profile properties used for constraint evaluation.

        full_currents_vec : ndarray
            Full vector of all coil currents.

            Example:
                eq.tokamak.getCurrentsVec()

        trial_plasma_psi : ndarray
            Candidate plasma flux solution.

            Must have same shape as:
                eq.R

        l2_reg : float or ndarray
            Tikhonov regularisation parameter.

            If float:
                λ I² penalty is applied uniformly.

            If array:
                Allows coil-wise regularisation weighting.

        Returns
        -------
        delta_current : ndarray
            Optimal coil current update vector.

        loss : float
            Residual loss norm.
        """
        # prepare the plasma-related values
        self.build_plasma_vals(trial_plasma_psi=trial_plasma_psi)

        # build the matrices that define the optimization
        self.build_lsq(full_currents_vec)

        # build Tikhonov matrix
        if isinstance(l2_reg, float):
            reg_matrix = l2_reg * np.eye(self.n_control_coils)
        else:
            if len(l2_reg) != self.n_control_coils:
                raise ValueError(
                    f"Expected l2_reg to have length equal to number of coils being controlled ({self.n_control_coils}), but got {len(l2_reg)}."
                )
            reg_matrix = np.diag(l2_reg)

        # ------------------------------------------------------------
        # solve least-squares optimisation problem
        #
        # If inequality constraints (for coil limits or normalised psi bounds) are present:
        #     Use quadratic programming solver.
        #
        # Otherwise:
        #     Solve normal equations directly.
        # ------------------------------------------------------------
        if self.coil_current_limits is not None or self.psi_norm_limits is not None:
            delta_current, loss = self.optimize_currents_quadratic(
                eq, profiles, full_currents_vec, reg_matrix
            )
        else:
            # TODO: should we just use the quadratic solver all the time, regardless
            # of whether coil limits are specified?
            delta_current = np.linalg.solve(
                self.A.T @ self.A + reg_matrix, self.A.T @ self.b
            )
            loss = np.linalg.norm(self.loss)

        return delta_current, loss

    def optimize_currents_quadratic(
        self,
        eq,
        profiles,
        full_currents_vec,
        reg_matrix,
        *,
        mu_coils=None,
        mu_psi_norm=None,
        A=None,
        b=None,
    ):
        """
        Solve the regularised constrained least-squares problem using convex optimisation.

        This method computes coil current updates by solving the quadratic program:

            minimise_ΔI

                || A ΔI − b ||²
              + ΔIᵀ R ΔI
              + penalty(slack variables)

        subject to optional inequality constraints:

            • Coil current upper/lower bounds
            • Normalised flux (ψ_norm) inequality constraints

        Slack variables are introduced to allow temporary constraint violation,
        penalised in the objective function. This improves optimisation robustness
        and prevents infeasibility during nonlinear solve iterations.

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Equilibrium object used to evaluate plasma quantities and ψ_norm.

        profiles : FreeGSNKE profile object
            Provides boundary flux (psi_bndry) and related quantities.

        full_currents_vec : ndarray
            Full vector of coil currents.

        reg_matrix : ndarray
            Regularisation matrix (n_control_coils × n_control_coils),
            typically diagonal, encoding Tikhonov L2 regularisation.

        mu_coils : float | None
            Scaling factor for coil current limit slack penalties.
            If None, defaults to self.mu_coils (default ~1e5).

        mu_psi_norm : float | None
            Scaling factor for ψ_norm slack penalties.
            If None, defaults to self.mu_psi_norm.

        A : ndarray | None
            Sensitivity matrix. If None, uses self.A.

        b : ndarray | None
            Target residual vector. If None, uses self.b.

        Returns
        -------
        delta : ndarray
            Optimal coil current update vector ΔI.

        loss : float
            Combined loss including:
                • Magnetic constraint residual norm
                • Slack variable penalties

        Notes
        -----
        The optimisation problem solved is a convex quadratic program (QP).

        Base objective:

            min_ΔI  ||A ΔI − b||² + ΔIᵀ R ΔI

        With optional inequality constraints:

            I_min ≤ I_current + ΔI ≤ I_max
            ψ_norm constraints (≥ or ≤ type)

        Slack variables s ≥ 0 allow:

            constraint_violation ≤ s

        with large quadratic penalties to discourage violation.
        """

        # use stored least-squares system unless inputs are provided
        A = self.A if A is None else A
        b = self.b if b is None else b

        # optimsiation variable: coil currents
        delta = cvxpy.Variable(self.n_control_coils)
        slack_variables = []
        constraints = []

        # Setup the coil limits slack variables and constraints
        # Slack variables allow the coil limit to be violated by the solver
        # however it is penalised in the objective function; this is useful
        # to allow coil limits be violated 'on the path' to a solution.
        if self.coil_current_limits is not None:
            coil_limits_upper_slack = cvxpy.Variable(self.n_control_coils, nonneg=True)
            coil_limits_lower_slack = cvxpy.Variable(self.n_control_coils, nonneg=True)

            # Scale slack penalty relative to curvature of LS problem.
            # Using max diagonal of AᵀA gives magnitude comparable to system Hessian.
            coil_limit_slack_scale = (mu_coils or self.mu_coils) * np.diag(
                A.T @ A
            ).max()
            coil_upper_limits, coil_lower_limits = self.coil_current_limits

            # upper bound constraints
            for coil_index, ul in enumerate(coil_upper_limits):
                if ul is not None:
                    constraints.append(
                        (full_currents_vec[self.control_mask][coil_index] / 1000)
                        + (delta[coil_index] / 1000)
                        <= (ul / 1000) + coil_limits_upper_slack[coil_index]
                    )

            # lower bound constraints
            for coil_index, ll in enumerate(coil_lower_limits):
                if ll is not None:
                    constraints.append(
                        (full_currents_vec[self.control_mask][coil_index] / 1000)
                        + (delta[coil_index] / 1000)
                        >= (ll / 1000) - coil_limits_lower_slack[coil_index]
                    )

            # penalise slack variables heavily to discourage violations
            slack_variables.append(coil_limit_slack_scale * coil_limits_upper_slack)
            slack_variables.append(coil_limit_slack_scale * coil_limits_lower_slack)

        # normalised flux cosntraints
        if self.psi_norm_limits is not None:

            # ensure eq object is up-to-date
            eq._updatePlasmaPsi(eq.plasma_psi)
            eq.psi_bndry = profiles.psi_bndry

            psi_norm_slack = cvxpy.Variable(self.psi_norm_limits.shape[0], nonneg=True)

            psi_norm_slack_scale = (mu_psi_norm or self.mu_psi_norm) * np.diag(
                A.T @ A
            ).max()

            # sensitivity of ψ_norm w.r.t coil currents
            psi_norm_A = self.G_psi_norm[self.control_mask].T

            # chain rule: G_psi_norm is derivative wrt ψ, not ψ_norm
            psi_norm_A /= eq.psi_bndry - eq.psi_axis

            # residual between target ψ_norm and current ψ_norm
            psi_norm_b = self.psi_norm_limits[:, 2] - eq.psiNRZ(
                self.psi_norm_limits[:, 0], self.psi_norm_limits[:, 1]
            )

            for psin_limit_idx in range(self.psi_norm_limits.shape[0]):
                psin_con_sign = self.psi_norm_limits[psin_limit_idx, 3]
                lhs = psi_norm_A[psin_limit_idx, :] @ delta

                # enforce lower bound
                if psin_con_sign == 1:
                    constraints.append(
                        lhs
                        >= psi_norm_b[psin_limit_idx] - psi_norm_slack[psin_limit_idx]
                    )
                # enforce upper bound
                elif psin_con_sign == -1:
                    constraints.append(
                        lhs
                        <= psi_norm_b[psin_limit_idx] + psi_norm_slack[psin_limit_idx]
                    )
                else:
                    raise ValueError(
                        f"Unexpected psi norm constraint sign {psin_con_sign}. Expected 1 or -1."
                    )

            # penalise ψ_norm slack
            slack_variables.append(psi_norm_slack_scale * psi_norm_slack)

        # minimise the objectives (the least squares objective + regularisation + slack variables)
        minimisation_expression = cvxpy.sum_squares(A @ delta - b) + cvxpy.quad_form(
            delta, reg_matrix
        )
        for expr in slack_variables:
            minimisation_expression += cvxpy.sum_squares(expr)

        # solve problem
        problem = cvxpy.Problem(
            cvxpy.Minimize(minimisation_expression), constraints or None
        )
        problem.solve(solver=cvxpy.CLARABEL, tol_infeas_abs=1e-12, tol_infeas_rel=1e-10)

        # combine magnetic residual loss with slack penalties
        slack_loss = sum([i.value.sum() for i in slack_variables])

        return (
            delta.value,
            np.linalg.norm(self.loss) + slack_loss,
        )

    def optimize_currents_grad(
        self,
        full_currents_vec,
        trial_plasma_psi,
        isoflux_weight=1.0,
        null_points_weight=1.0,
        psi_vals_weight=1.0,
    ):
        """
        Compute the gradient of the magnetic least-squares objective
        with respect to control coil currents.

        This method evaluates the gradient of the unconstrained objective:

            J(ΔI) = ½ || A ΔI − b ||²

        evaluated at ΔI = 0, i.e.

            ∇J = Aᵀ b

        Optional weighting factors allow different constraint classes
        (isoflux, null points, direct flux values) to be scaled prior
        to gradient evaluation.

        Parameters
        ----------
        full_currents_vec : ndarray
            Full vector of all coil current values.

            Example:
                eq.tokamak.getCurrentsVec()

        trial_plasma_psi : ndarray
            Plasma flux contribution. Must have same shape as eq.R.

        isoflux_weight : float, optional
            Weight applied to isoflux constraint residuals.

        null_points_weight : float, optional
            Weight applied to null-point field constraint residuals.

        psi_vals_weight : float, optional
            Weight applied to direct flux value constraint residuals.

        Returns
        -------
        grad : ndarray
            Gradient of the weighted least-squares objective with respect
            to control coil current updates.

        loss : float
            Combined magnetic constraint loss (unweighted).

        """

        # prepare the plasma-related values
        self.build_plasma_vals(trial_plasma_psi=trial_plasma_psi)

        # build the matrices that define the optimization
        self.build_lsq(full_currents_vec)

        # weight the different terms in the loss
        b_weighted = np.copy(self.b)
        idx = 0
        if self.isoflux_set is not None:
            b_weighted[idx : idx + self.isoflux_dim] *= isoflux_weight
            idx += self.isoflux_dim
        if self.null_points is not None:
            b_weighted[idx : idx + self.nullp_dim] *= null_points_weight
            idx += self.nullp_dim
        if self.psi_vals is not None:
            b_weighted[idx : idx + self.psiv_dim] *= psi_vals_weight
            idx += self.psiv_dim

        grad = np.dot(self.A.T, b_weighted)

        return grad, np.linalg.norm(self.loss)

    def plot(self, axis=None, show=True):
        """
        Visualise the active coil control constraints.

        This method provides a graphical representation of the magnetic
        and operational constraints currently configured in the object.

        Parameters
        ----------
        axis : matplotlib.axes.Axes | None, optional
            Axis object on which to draw the plot. If None, a new figure
            and axis are created internally.

        show : bool, optional
            If True, calls matplotlib.pyplot.show() before returning.

        Returns
        -------
        matplotlib.axes.Axes
            Axis containing the generated plot.

        Notes
        -----
        This is a thin wrapper around:

            freegs4e.plotting.plotIOConstraints

        and exists primarily for convenience and API consistency.
        """
        from freegs4e.plotting import plotIOConstraints

        return plotIOConstraints(self, axis=axis, show=show)

    def prepare_plasma_psi(self, trial_plasma_psi):
        """
        Preprocess plasma flux values for normalisation and constraint evaluation.

        This method computes shifted plasma flux extrema used for
        normalised flux calculations and ψ_norm-based constraints.

        Parameters
        ----------
        trial_plasma_psi : ndarray
            Plasma flux array.
        """

        self.min_psi = np.amin(trial_plasma_psi)
        self.psi0 = np.amax(trial_plasma_psi)
        self.min_psi -= 0.001 * (self.psi0 - self.min_psi)
        self.psi0 -= self.min_psi

    def prepare_plasma_vals_for_plasma(self, trial_plasma_psi):
        """
        Precompute plasma-dependent quantities required for plasma optimisation.

        This method evaluates the plasma flux at constraint locations and
        constructs pairwise flux-difference terms used in isoflux constraints.

        In addition to raw flux differences, a nonlinear transformed version
        of the normalised flux is computed:

            ψ̂ = (ψ − ψ_min) / ψ0
            f(ψ̂) = ψ̂ log(ψ̂)

        which is then used to build weighted pairwise differences.

        Parameters
        ----------
        trial_plasma_psi : ndarray
            Plasma flux array defined on the equilibrium grid (eqR, eqZ).

        Notes
        -----
        For each isoflux constraint set:

            1. Interpolate ψ at constraint coordinates.
            2. Construct pairwise differences:
                ψ_i − ψ_j
            3. Construct transformed differences:
                ψ0 [ f(ψ̂_i) − f(ψ̂_j) ]

        These quantities are cached for use in plasma optimisation routines.

        The logarithmic transform enhances sensitivity near ψ̂ → 0 and
        improves conditioning when enforcing plasma-based constraints.
        """

        # prepare plasma flux guess
        self.prepare_plasma_psi(trial_plasma_psi=trial_plasma_psi)

        # interpolate
        psi_func = interpolate.RectBivariateSpline(
            self.eqR[:, 0], self.eqZ[0, :], trial_plasma_psi
        )

        # prepare isoflux constraints (contributions from plasma flux)
        if self.isoflux_set is not None:
            self.d_psi_plasma_vals_iso = []
            self.d_psi_for_plasma_iso = []

            for i, isoflux in enumerate(self.isoflux_set):

                # interpolate
                plasma_vals = psi_func(isoflux[0], isoflux[1], grid=False)
                # pairwise differences
                d_plasma_vals = plasma_vals[:, np.newaxis] - plasma_vals[np.newaxis, :]
                self.d_psi_plasma_vals_iso.append(d_plasma_vals[self.mask_set[i]])

                # normalised flux
                hat_plasma_vals = (plasma_vals - self.min_psi) / self.psi0

                # log to avoid errors near zero(?)
                hat_plasma_vals *= np.log(hat_plasma_vals)

                # pairwise diffs
                d_hat_plasma_vals = (
                    hat_plasma_vals[:, np.newaxis] - hat_plasma_vals[np.newaxis, :]
                )

                # rescale
                self.d_psi_for_plasma_iso.append(
                    self.psi0 * d_hat_plasma_vals[self.mask_set[i]]
                )

    def prepare_for_plasma_optimization(self, eq):
        """
        Prepare geometry- and Green's-function-dependent quantities
        required for plasma optimisation.

        This method ensures that:

            • Source-domain geometric properties are updated
            • Magnetic Green's operators are constructed

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Provides geometry, coil configuration, and grid data.

        Notes
        -----
        This is typically called prior to plasma-only optimisation steps
        to ensure response operators and geometric mappings are consistent
        with the current equilibrium configuration.
        """
        self.source_domain_properties(eq)
        self.build_greens(eq=eq)

    def build_plasma_isoflux_lsq(self, full_currents_vec, trial_plasma_psi):
        """
        Assemble the least-squares system for plasma-only isoflux optimisation.

        This method constructs a linearised least-squares problem of the form:

            A_plasma x ≈ b_plasma

        where x ∈ ℝ² represents plasma transformation parameters
        (e.g. normalisation and nonlinear shaping terms).

        The residual enforces pairwise isoflux constraints:

            Δψ_total = Δψ_coils + Δψ_plasma ≈ 0

        using both raw flux differences and nonlinear transformed
        (ψ̂ log ψ̂) contributions.

        Parameters
        ----------
        full_currents_vec : ndarray
            Full vector of coil currents.

        trial_plasma_psi : ndarray
            Plasma flux array defined on the equilibrium grid.

        Notes
        -----
        The constructed system solves for two plasma parameters:

            x[0] → linear normalisation scaling
            x[1] → nonlinear shaping contribution

        Results are stored internally:

            self.A_plasma
            self.b_plasma
            self.loss_plasma
        """

        # pre-compute pairwise flux differences
        self.prepare_plasma_vals_for_plasma(trial_plasma_psi)

        loss = []
        A = []
        b = []

        # loop over each constraint
        for i, isoflux in enumerate(self.isoflux_set):
            # tokamak flux difference
            b_val = np.sum(self.dG_set[i] * full_currents_vec[:, np.newaxis], axis=0)
            # add plasma contribution
            b_val += self.d_psi_plasma_vals_iso[i]
            b.append(-b_val)
            # residual
            loss.append(np.linalg.norm(b_val))

            # build the jacobian
            Amat = np.zeros((len(b_val), 2))
            # gradient with respect to the normalization of psi
            Amat[:, 0] = self.d_psi_plasma_vals_iso[i]
            # gradient with respect to the exponent of psi
            Amat[:, 1] = self.d_psi_for_plasma_iso[i]
            A.append(Amat)

        # stack everything
        self.A_plasma = np.concatenate(A, axis=0)
        self.b_plasma = np.concatenate(b, axis=0)
        self.loss_plasma = np.linalg.norm(loss)

    def optimize_plasma_psi(self, full_currents_vec, trial_plasma_psi, l2_reg):
        """
        Solve the regularised least-squares problem for plasma parameters.

        This solves:

            min_x || A_plasma x − b_plasma ||² + xᵀ R x

        where:

            x ∈ ℝ²
            R = Tikhonov regularisation matrix

        Parameters
        ----------
        full_currents_vec : ndarray
            Full vector of coil currents.

        trial_plasma_psi : ndarray
            Plasma flux array defined on the equilibrium grid.

        l2_reg : float or ndarray (length 2)
            Tikhonov regularisation strength.

            If float:
                Uniform regularisation applied.

            If array:
                Diagonal regularisation weights.

        Returns
        -------
        delta_current : ndarray
            Optimal update vector for plasma parameters (length 2).

        loss_plasma : float
            Isoflux constraint violation norm (pre-update).

        Notes
        -----
        The solution is computed via normal equations:

            x = (AᵀA + R)⁻¹ Aᵀ b

        Since the system dimension is small (2×2), direct inversion
        is computationally inexpensive.
        """

        # assemble least-squares system
        self.build_plasma_isoflux_lsq(full_currents_vec, trial_plasma_psi)

        # assemble regularisatio matrix
        if type(l2_reg) == float:
            reg_matrix = l2_reg * np.eye(2)
        else:
            reg_matrix = np.diag(l2_reg)

        # --------------------------------------------------------------
        # Solve regularised normal equations:
        #
        #   (AᵀA + R) x = Aᵀ b
        # --------------------------------------------------------------
        lhs = self.A_plasma.T @ self.A_plasma + reg_matrix
        rhs = self.A_plasma.T @ self.b_plasma
        delta_current = np.linalg.solve(lhs, rhs)

        return delta_current, self.loss_plasma
