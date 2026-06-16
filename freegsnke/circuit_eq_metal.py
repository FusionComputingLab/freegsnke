"""
Defines the metal_currents object, which handles the circuit equations of 
all metal structures in the tokamak - both active PF coils and passive structures.

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

import numpy as np
from freegs4e.gradshafranov import Greens, GreensBr, GreensBz

from .implicit_euler import implicit_euler_solver
from .normal_modes import mode_decomposition


class metal_currents:
    """
    Time evolution model for electrical currents in conducting structures
    within a tokamak (active coils, passive vessel, and optional plasma coupling).

    This class solves the dynamical circuit equations for metallic components
    using an implicit-Euler time integrator. It can operate in multiple modes:

    - Coil-only (vacuum vessel response)
    - Vessel eigenmode representation of passive structures
    - Fully coupled plasma–metal system (optional)

    The system evolves currents in a reduced basis determined by:
    - physical active coil currents
    - vessel eigenmodes (if enabled)
    - optional plasma current coupling

    Key features:
    - Construction of resistance and inductance operators
    - Optional diagonalisation of passive vessel dynamics
    - Optional coupling to plasma current evolution
    - Support for multi-step implicit time integration
    """

    def __init__(
        self,
        eq,
        flag_vessel_eig,
        flag_plasma,
        max_mode_frequency,
        max_internal_timestep,
        full_timestep,
        plasma_pts=None,
        coil_resist=None,
        coil_self_ind=None,
        verbose=True,
    ):
        """
        Initialise the dynamical evolution model for metallic currents.

        This class solves the time evolution of currents in conducting structures
        (coils, vessel, and passive structures). It can also be used independently
        to solve vacuum vessel circuit equations in the absence of plasma.

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Initial equilibrium used to define grid geometry, machine configuration,
            and linearisation for the evolution solver. Changing machine geometry
            requires re-instantiation.
        flag_vessel_eig : bool
            If True, include vessel eigenmodes in the circuit model.
        flag_plasma : bool
            If True, include plasma coupling in the circuit equations.
            In this case, `plasma_pts` must be provided.
        max_mode_frequency : float
            Maximum vessel eigenmode frequency to include (s⁻¹).
        max_internal_timestep : float
            Internal timestep used by the implicit Euler solver.
        full_timestep : float
            Physical timestep for advancing the system. May be split into
            multiple internal steps if larger than `max_internal_timestep`.
        plasma_pts : array-like, optional
            Grid points included in plasma evolution coupling (typically points
            inside the limiter).
        coil_resist : ndarray, optional
            Resistances of all conducting elements (coils + passive structures).
            If None, default machine values are used.
        coil_self_ind : ndarray, optional
            Mutual inductance matrix for all conducting elements.
            If None, default machine values are used.
        verbose : bool
            If True, enable diagnostic output during setup.
        """

        self.n_coils = eq.tokamak.n_coils
        self.n_active_coils = eq.tokamak.n_active_coils
        self.verbose = verbose

        # prepare resistance data
        if coil_resist is not None:
            if len(coil_resist) != self.n_coils:
                raise ValueError(
                    f"Resistance vector provided (size: {len(coil_resist)}) is not compatible with machine description (size: {self.n_coils})."
                )
            self.coil_resist = coil_resist
        else:
            self.coil_resist = eq.tokamak.coil_resist

        self.Rm1 = 1.0 / self.coil_resist
        # self.R = np.copy(self.coil_resist)
        self.active_coil_resistances = np.copy(self.coil_resist[: self.n_active_coils])

        # prepare inductance data
        if coil_self_ind is not None:
            if np.size(coil_self_ind) != self.n_coils**2:
                raise ValueError(
                    f"Mutual inductance matrix provided (size: {np.size(coil_self_ind)}) is not compatible with machine description (size: {self.n_coils**2})."
                )
            self.coil_self_ind = coil_self_ind
        else:
            self.coil_self_ind = eq.tokamak.coil_self_ind

        self.build_rm1l()

        self.flag_vessel_eig = flag_vessel_eig
        self.flag_plasma = flag_plasma

        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep

        if flag_vessel_eig:
            # builds mode decomposition
            self.normal_modes = mode_decomposition(
                coil_resist=self.coil_resist,
                coil_self_ind=self.coil_self_ind,
                n_coils=self.n_coils,
                n_active_coils=self.n_active_coils,
            )
            self.max_mode_frequency = max_mode_frequency
            self.initialize_for_eig(selected_modes_mask=False)

        else:
            self.max_mode_frequency = 0
            self.initialize_for_no_eig()

        if flag_plasma:
            self.plasma_pts = plasma_pts
            self.Mey_matrix = self.Mey(eq)

        # Dummy voltage vector
        self.empty_U = np.zeros(self.n_coils)

    def build_rm1l(
        self,
    ):
        """
        Construct the R⁻¹L coupling matrix for the circuit model.

        This method builds the product of the inverse resistance matrix with the
        self-inductance (mutual inductance) matrix for all conducting elements
        (active coils and passive structures).

        The resulting matrix is used in the formulation of the linear circuit
        evolution equations.

        Returns
        -------
        None
            The result is stored in `self.rm1l_non_symm`.
        """

        self.rm1l_non_symm = np.diag(self.coil_resist**-1.0) @ self.coil_self_ind

    def make_selected_mode_mask(self, mode_coupling_masks, verbose):
        """
        Build selection mask for vessel normal modes used in circuit equations.

        This method selects which passive structure eigenmodes are included in the
        reduced-order circuit model based on a frequency cutoff and optional
        plasma-coupling criteria.

        Active coil variables are always included, while passive vessel modes are
        filtered according to:
        - maximum mode frequency (`self.max_mode_frequency`)
        - optional coupling thresholds provided in `mode_coupling_masks`

        Parameters
        ----------
        mode_coupling_masks : tuple of ndarray or None
            Optional pair of boolean masks used to:
            - reintroduce strongly coupled modes
            - remove weakly coupled modes
        verbose : bool
            If True, print diagnostic information about mode selection.

        Returns
        -------
        None
            Updates internal attributes:
            - `self.selected_modes_mask`
            - `self.n_independent_vars`
        """
        # this is for passives alone
        selected_modes_mask = self.normal_modes.w_passive < self.max_mode_frequency
        freq_only_number = np.sum(selected_modes_mask)

        # selected_modes_mask = [True,...,True, False,...,False]
        # this includes the actives too
        self.selected_modes_mask = np.concatenate(
            (np.ones(self.n_active_coils).astype(bool), selected_modes_mask)
        )
        if verbose:
            print(f"   Active coils")
            print(
                f"      total selected = {self.n_active_coils} (out of {self.n_active_coils})"
            )
            print(f"   Passive structures")
            print(
                f"      {freq_only_number} selected with characteristic timescales larger than 'max_mode_frequency'"
            )

        if mode_coupling_masks is not None:
            # reintroduce modes that couple strongly
            self.selected_modes_mask = (
                self.selected_modes_mask + mode_coupling_masks[0]
            ).astype(bool)
            freq_and_thresh_number = np.sum(self.selected_modes_mask)
            if verbose:
                print(
                    f"      {freq_and_thresh_number - (freq_only_number + self.n_active_coils)} recovered that couple with the plasma more than 'threshold_dIy_dI'"
                )

            # exclude modes that do not couple enough
            self.selected_modes_mask = (
                self.selected_modes_mask * mode_coupling_masks[1]
            ).astype(bool)
            final_number = np.sum(self.selected_modes_mask)
            if verbose:
                print(
                    f"      {freq_and_thresh_number - final_number} removed that couple with the plasma less than 'min_dIy_dI'"
                )
                print(
                    f"      total selected = {final_number - self.n_active_coils} (out of {self.n_coils - self.n_active_coils})"
                )
                print(
                    f"   Total number of modes = {final_number} ({self.n_active_coils} active coils + {final_number - self.n_active_coils} passive structures)"
                )
                print(
                    f"      (Note: some additional modes may be removed after Jacobian calculation)"
                )

        self.n_independent_vars = np.sum(self.selected_modes_mask)

    def initialize_for_eig(
        self, selected_modes_mask=None, mode_coupling_masks=None, verbose=True
    ):
        """
        Initialise the metal current system in eigenmode representation.

        This method prepares the reduced-order circuit model when vessel
        eigenmodes are included. It constructs the mode transformation matrices,
        applies optional mode selection/reduction, and builds the system matrices
        required for the implicit time integration solver.

        The system is transformed between physical currents and eigenmodes via:
        - P: transformation from eigenmodes to physical currents
        - Pm1: inverse transformation (restricted to selected modes)

        Parameters
        ----------
        selected_modes_mask : ndarray of bool, optional
            Explicit mask selecting which modes to retain. If None, the mask is
            constructed using `mode_coupling_masks`. If False, all modes are used.
        mode_coupling_masks : tuple of ndarray of bool, optional
            Pair of masks used to:
            - include strongly coupled modes
            - exclude weakly coupled modes
            Only used when `selected_modes_mask is None`.
        verbose : bool
            If True, print diagnostic information about mode reduction.

        Returns
        -------
        None
            Updates internal solver state including:
            - mode selection masks
            - transformation matrices (P, Pm1)
            - system matrix (Lambdam1)
            - time integrator solver
            - forcing term selection
        """

        if selected_modes_mask is None:
            # this is the case when mode_coupling_masks are used to build self.selected_modes_mask
            self.make_selected_mode_mask(mode_coupling_masks, verbose)
            # Pmatrix is the full matrix that changes the basis in the current space
            # from the normal modes Id (for diagonal) to the metal currents I:
            # I = Pmatrix Id
            # And also
            # Id = Pmatrixm1 I
            # Therefore, taking the truncation into account:
            self.P = self.normal_modes.Pmatrix[:, self.selected_modes_mask]
            self.Pm1 = self.normal_modes.Pmatrix_inverse[self.selected_modes_mask]
        elif selected_modes_mask is False:
            # this is to include ALL modes
            self.selected_modes_mask = np.ones(self.n_coils).astype(bool)
            self.n_independent_vars = np.sum(self.selected_modes_mask)
            self.P = self.normal_modes.Pmatrix[:, self.selected_modes_mask]
            self.Pm1 = self.normal_modes.Pmatrix_inverse[self.selected_modes_mask]
        else:
            # this is the case used by nonlinear_solver.remove_modes
            self.selected_modes_mask_partial = selected_modes_mask
            print(f"Further mode reduction:")
            print(
                f"   {len(selected_modes_mask) - np.sum(selected_modes_mask)} previously included modes couple with the plasma less than 'min_dIy_dI' (following Jacobian calculation)"
            )

            self.n_independent_vars = np.sum(self.selected_modes_mask_partial)
            print(
                f"   Final number of modes = {self.n_independent_vars} ({self.n_active_coils} active coils + {self.n_independent_vars - self.n_active_coils} passive structures)"
            )

            self.P = self.P[:, self.selected_modes_mask_partial]
            self.Pm1 = self.Pm1[self.selected_modes_mask_partial]

        # this is not needed any longer and now incorrect, the eigenvectors in P are independent but NOT orthogonal
        # self.Pm1 = (self.P).T

        # Note Lambda is not actually diagonal because the passive structures has been
        # diagonalised separately from the active coils. The modes of used for the passive structures
        # diagonalise the isolated dynamics of the walls.
        # Equation is Lambda**(-1)Iddot + I = F
        self.Lambdam1 = self.Pm1 @ (self.rm1l_non_symm @ self.P)
        # self.RP = np.diag(self.coil_resist) @ self.P
        # self.RP_inv = np.linalg.solve(self.RP.T @ self.RP, self.RP.T)
        # self.Lambdam1 = (self.RP_inv @ self.coil_self_ind) @ self.P

        self.solver = implicit_euler_solver(
            Mmatrix=self.Lambdam1,
            Rmatrix=np.eye(self.n_independent_vars),
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.full_timestep,
        )

        if self.flag_plasma:
            self.forcing_term = self.forcing_term_eig_plasma
        else:
            self.forcing_term = self.forcing_term_eig_no_plasma

    def reset_active_coil_resistances(self, active_coil_resistances):
        """
        Update the resistances of the active coils and rebuild derived system matrices.

        This method replaces the active-coil portion of the full resistance vector
        while keeping passive/vessel resistances unchanged. It then updates all
        dependent operators used in the circuit evolution model.

        The update triggers a rebuild of:
        - the full resistance vector
        - the inverse resistance matrix
        - the resistance–inductance coupling operator
        - the reduced system matrix used in vessel mode dynamics

        Parameters
        ----------
        active_coil_resistances : ndarray
            Updated resistances for the active coils only (length = n_active_coils).
        """
        self.coil_resist = np.concatenate(
            (active_coil_resistances, self.coil_resist[self.n_active_coils :])
        )
        self.active_coil_resistances = np.copy(self.coil_resist[: self.n_active_coils])
        self.Rm1 = 1 / self.coil_resist
        self.build_rm1l()
        self.Lambdam1 = self.Pm1 @ (self.rm1l_non_symm @ self.P)

    def initialize_for_no_eig(self):
        """
        Initialise the metal current system without eigenmode decomposition.

        This method constructs and solves the full circuit equations in the
        physical current basis, without projecting onto vessel eigenmodes.

        The governing equation is:
        Mmatrix · dI/dt + Rmatrix · I = F

        where:
        - Mmatrix is the full mutual inductance matrix
        - Rmatrix is the diagonal resistance matrix

        Returns
        -------
        None
            Updates internal solver and forcing term configuration.
        """

        # Equation is Mmatrix Idot + Rmatrix I = F
        self.solver = implicit_euler_solver(
            Mmatrix=self.coil_self_ind,
            Rmatrix=np.diag(self.coil_resist),
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.full_timestep,
        )

        if self.flag_plasma:
            self.forcing_term = self.forcing_term_no_eig_plasma
        else:
            self.forcing_term = self.forcing_term_no_eig_no_plasma

    def reset_timesteps(self, max_internal_timestep, full_timestep):
        """
        Update solver time-stepping parameters.

        This method resets both the internal solver timestep and the external
        evolution timestep used to advance the circuit equations. If the full
        timestep exceeds the internal timestep, multiple substeps are performed
        automatically by the solver.

        Parameters
        ----------
        max_internal_timestep : float
            Maximum timestep used internally by the implicit Euler solver.
        full_timestep : float
            External timestep used to advance the system in time.
        """
        self.solver.set_timesteps(
            full_timestep=full_timestep, max_internal_timestep=max_internal_timestep
        )

    def forcing_term_eig_plasma(self, active_voltage_vec, Iydot):
        """
        Compute forcing term in eigenmode basis including plasma coupling.

        This method constructs the effective right-hand side of the circuit
        equations in eigenmode coordinates when plasma dynamics are included.

        The forcing is built from:
        - applied coil voltages
        - inductive coupling to plasma current evolution

        Parameters
        ----------
        active_voltage_vec : ndarray
            Voltages applied to the active coils.
        Iydot : ndarray
            Time derivative of plasma current degrees of freedom.

        Returns
        -------
        all_Us : ndarray
            Forcing term expressed in the eigenmode basis.
        """
        all_Us = np.zeros_like(self.empty_U)
        all_Us[: self.n_active_coils] = active_voltage_vec
        all_Us -= self.Mey @ Iydot
        all_Us = np.dot(self.Pm1, self.Rm1 * all_Us)
        return all_Us

    def forcing_term_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        """
        Compute forcing term in eigenmode basis without plasma coupling.

        This method constructs the right-hand side of the circuit equations in
        eigenmode coordinates when plasma effects are neglected. Only coil
        voltages are included.

        Parameters
        ----------
        active_voltage_vec : ndarray
            Voltages applied to the active coils.
        Iydot : ndarray, optional
            Unused placeholder for interface compatibility.

        Returns
        -------
        all_Us : ndarray
            Forcing term expressed in the eigenmode basis.
        """
        all_Us = self.empty_U.copy()
        all_Us[: self.n_active_coils] = active_voltage_vec
        all_Us = np.dot(self.Pm1, self.Rm1 * all_Us)
        return all_Us

    def forcing_term_no_eig_plasma(self, active_voltage_vec, Iydot):
        """
        Compute forcing term in coil basis including plasma coupling.

        This method builds the right-hand side of the circuit equations in the
        physical (non-eigenmode) coil basis, including inductive coupling to the
        plasma evolution.

        Parameters
        ----------
        active_voltage_vec : ndarray
            Voltages applied to the active coils.
        Iydot : ndarray
            Time derivative of plasma current degrees of freedom.

        Returns
        -------
        all_Us : ndarray
            Forcing term in the physical coil basis.
        """
        all_Us = self.empty_U.copy()
        all_Us[: self.n_active_coils] = active_voltage_vec
        all_Us -= np.dot(self.Mey, Iydot)
        return all_Us

    def forcing_term_no_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        """
        Compute forcing term in coil basis without plasma coupling.

        This method constructs the right-hand side of the circuit equations in the
        physical (non-eigenmode) coil basis when plasma effects are neglected.
        Only applied coil voltages are included.

        Parameters
        ----------
        active_voltage_vec : ndarray
            Voltages applied to the active coils.
        Iydot : ndarray, optional
            Unused placeholder for interface consistency.

        Returns
        -------
        all_Us : ndarray
            Forcing term in the physical coil basis.
        """
        all_Us = self.empty_U.copy()
        all_Us[: self.n_active_coils] = active_voltage_vec
        return all_Us

    def IvesseltoId(self, Ivessel):
        """
        Given the vector of currents in the metals, this returns Id,
        the vector of currents in the eigenmodes basis.

        Parameters
        ----------
        Ivessel : np.ndarray
            Vessel currents (all metals).

        Returns
        -------
        Id : np.ndarray
            Currents in the eigenmode basis.
        """

        return self.Pm1 @ Ivessel

    def IdtoIvessel(self, Id):
        """
        Given the vector of currents in the eigenmode basis, this returns Ivessel,
        the vector of currents in all the metals.

        Parameters
        ----------
        Id : np.ndarray
            Currents in the eigenmode basis.

        Returns
        -------
        Ivessel : np.ndarray
            Vessel currents (all metals).
        """

        return self.P @ Id

    def stepper(self, It, active_voltage_vec, Iydot=0):
        """Steps the circuit equation forward in time.

        Parameters
        ----------
        It : np.ndarray
            Currents at time t.
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray or float, optional
            Vector of rate of change of plasma currents. Defaults to 0.

        Returns
        -------
        It : np.ndarray
            Currents at time t+dt.
        """
        forcing = self.forcing_term(active_voltage_vec, Iydot)
        It = self.solver.full_stepper(It, forcing)
        return It

    def Mey(
        self,
        eq,
    ):
        """
        Calculates the matrix of mutual inductance values between plasma grid points
        included in the dynamics calculations and all vessel coils.

        Parameters
        -------
        eq : class
            FreeGSNKE equilibrium Object

        Returns
        -------
        Mey : np.ndarray
            Array of mutual inductances between plasma grid points and all vessel coils
        """
        coils_dict = eq.tokamak.coils_dict
        mey = np.zeros((eq.tokamak.n_coils, len(self.plasma_pts)))
        for j, labelj in enumerate(eq.tokamak.coils_list):
            greenm = Greens(
                self.plasma_pts[:, 0, np.newaxis],
                self.plasma_pts[:, 1, np.newaxis],
                coils_dict[labelj]["coords"][0][np.newaxis, :],
                coils_dict[labelj]["coords"][1][np.newaxis, :],
            )
            greenm *= coils_dict[labelj]["polarity"][np.newaxis, :]
            greenm *= coils_dict[labelj]["multiplier"][np.newaxis, :]
            mey[j] = np.sum(greenm, axis=-1)
        return 2 * np.pi * mey
