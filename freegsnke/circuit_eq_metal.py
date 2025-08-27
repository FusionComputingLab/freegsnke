"""
Defines the metal_currents Object, which handles the circuit equations of 
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
        """Sets up framework to solve the dynamical evolution of all metal currents.
        Can be used by itself to solve metal circuit equations for vacuum shots,
        i.e. when in the absence of the plasma.

        Parameters
        ----------

        eq : FreeGSNKE equilibrium Object
            Initial equilibrium. This is used to set the domain/grid properties
            as well as the machine properties.
            Furthermore, eq will be used to set up the linearization used by the linear evolutive solver.
            It can be changed later by initializing a new set of initial conditions.
            Note however that, to change either the machine or limiter properties
            it will be necessary to instantiate a new nl_solver object.
        flag_vessel_eig : bool
            Flag re whether vessel eigenmodes are used or not.
        flag_plasma : bool
            Whether to include plasma in circuit equation. If True, plasma_pts
            must be provided.
        plasma_pts : freegsnke.limiter_handler.plasma_pts
            Domain points in the domain that are included in the evolutive calculations.
            A typical choice would be all domain points inside the limiter. Defaults to None.
        max_mode_frequency : float
            Maximum frequency of vessel eigenmodes to include in circuit equation.
            Defaults to 1. Unit is s^-1.
        max_internal_timestep : float
            Maximum value of the 'internal' timestep for implicit euler solver. Defaults to .0001.
            The 'internal' timestep is the one actually used by the solver.
        full_timestep : float
            Timestep by which the equations are advanced. If full_timestep>max_internal_timestep
            multiple 'internal' steps are executed. Defaults to .0001.
        coil_resist : np.array
            1d array of resistance values for all conducting elements in the machine,
            including both active coils and passive structures.
            Defaults to None, meaning the values calculated by default in tokamak will be sourced and used.
        coil_self_ind : np.array
            2d matrix of mutual inductances between all pairs of machine conducting elements,
            including both active coils and passive structures
            Defaults to None, meaning the values calculated by default in tokamak will be sourced and used.
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
        # active + passive
        self.rm1l_non_symm = np.diag(self.coil_resist**-1.0) @ self.coil_self_ind

    def make_selected_mode_mask(self, mode_coupling_masks, verbose):
        """Creates a mask for the vessel normal modes to include in the circuit
        equations, based on the maximum frequency of the selected modes.
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
        """Initializes the metal_currents object for the case where vessel
        eigenmodes are used.

        Parameters
        ----------
        mode_coupling_metric_masks : (np.ndarray of booles, np.ndarray of booles)
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
        self.coil_resist = np.concatenate(
            (active_coil_resistances, self.coil_resist[self.n_active_coils :])
        )
        self.active_coil_resistances = np.copy(self.coil_resist[: self.n_active_coils])
        self.Rm1 = 1 / self.coil_resist
        self.build_rm1l()
        self.Lambdam1 = self.Pm1 @ (self.rm1l_non_symm @ self.P)

    def initialize_for_no_eig(self):
        """Initializes the metal currents object for the case where vessel
        eigenmodes are not used."""

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
        """Resets the timesteps

        Parameters
        ----------
        max_internal_timestep : float
            Maximum value of the 'internal' timestep for implicit euler solver.
            The 'internal' timestep is the one actually used by the solver.
        full_timestep : float
            Timestep by which the equations are advanced. If full_timestep>max_internal_timestep
            multiple 'internal' steps are executed.
        """
        self.solver.set_timesteps(
            full_timestep=full_timestep, max_internal_timestep=max_internal_timestep
        )

    # def reset_mode(
    #     self,
    #     flag_vessel_eig,
    #     flag_plasma,
    #     plasma_pts=None,
    #     max_mode_frequency=1,
    #     max_internal_timestep=0.0001,
    #     full_timestep=0.0001,
    # ):
    #     """Resets init inputs.

    #     flag_vessel_eig : bool
    #         Flag re whether vessel eigenmodes are used or not.
    #     flag_plasma : bool
    #         Whether to include plasma in circuit equation. If True, plasma_pts
    #         must be provided.
    #     plasma_pts : freegsnke.limiter_handler.plasma_pts
    #         Domain points in the domain that are included in the evolutive calculations.
    #         A typical choice would be all domain points inside the limiter. Defaults to None.
    #     max_mode_frequency : float
    #         Maximum frequency of vessel eigenmodes to include in circuit equation.
    #         Defaults to 1. Unit is s^-1.
    #     max_internal_timestep : float
    #         Maximum value of the 'internal' timestep for implicit euler solver. Defaults to .0001.
    #         The 'internal' timestep is the one actually used by the solver.
    #     full_timestep : float
    #         Timestep by which the equations are advanced. If full_timestep>max_internal_timestep
    #         multiple 'internal' steps are executed. Defaults to .0001.
    #     """
    #     control = self.max_internal_timestep != max_internal_timestep
    #     self.max_internal_timestep = max_internal_timestep

    #     control += self.full_timestep != full_timestep
    #     self.full_timestep = full_timestep

    #     control += flag_plasma != self.flag_plasma
    #     self.flag_plasma = flag_plasma

    #     if control * flag_plasma:
    #         self.plasma_pts = plasma_pts
    #         self.Mey_matrix = self.Mey(eq)

    #     control += flag_vessel_eig != self.flag_vessel_eig
    #     self.flag_vessel_eig = flag_vessel_eig

    #     if flag_vessel_eig:
    #         control += max_mode_frequency != self.max_mode_frequency
    #         self.max_mode_frequency = max_mode_frequency
    #     if control * flag_vessel_eig:
    #         self.initialize_for_eig(self.selected_modes_mask)
    #     else:
    #         self.initialize_for_no_eig()

    def forcing_term_eig_plasma(self, active_voltage_vec, Iydot):
        """Right-hand-side of circuit equation in eigenmode basis with plasma.

        Parameters
        ----------
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray
            Vector of rate of change of plasma currents.

        Returns
        -------
        all_Us : np.ndarray
            Effective voltages in eigenmode basis.
        """
        all_Us = np.zeros_like(self.empty_U)
        all_Us[: self.n_active_coils] = active_voltage_vec
        all_Us -= self.Mey @ Iydot
        all_Us = np.dot(self.Pm1, self.Rm1 * all_Us)
        return all_Us

    def forcing_term_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        """Right-hand-side of circuit equation in eigenmode basis without
        plasma.

        Parameters
        ----------
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray, optional
            This is not used.

        Returns
        -------
        all_Us : np.ndarray
            Effective voltages in eigenmode basis.
        """
        all_Us = self.empty_U.copy()
        all_Us[: self.n_active_coils] = active_voltage_vec
        all_Us = np.dot(self.Pm1, self.Rm1 * all_Us)
        return all_Us

    def forcing_term_no_eig_plasma(self, active_voltage_vec, Iydot):
        """Right-hand-side of circuit equation in normal mode basis with plasma.

        Parameters
        ----------
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray
            Vector of rate of change of plasma currents.

        Returns
        -------
        all_Us : np.ndarray
            Effective voltages in metals basis.
        """
        all_Us = self.empty_U.copy()
        all_Us[: self.n_active_coils] = active_voltage_vec
        all_Us -= np.dot(self.Mey, Iydot)
        return all_Us

    def forcing_term_no_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        """Right-hand-side of circuit equation in normal mode basis without
        plasma.

        Parameters
        ----------
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray, optional
            This is not used.

        Returns
        -------
        all_Us : np.ndarray
            Effective voltages in metals basis.
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

    # def Mey_f(
    #     self,
    #     eq,
    #     green_f
    #     ):
    #     """Calculates values of the function green_f for the matrix of
    #     plasma_pts x all vessel coils. For clarity, the function Mey is Mey_f(green_f = Greens)

    #     Parameters
    #     -------
    #     eq : class
    #         FreeGSNKE equilibrium Object
    #     green_f : function
    #         with same structure as Greens, i.e. Greens(R1,Z1, R2,Z2)

    #     Returns
    #     -------
    #     Mey : np.ndarray
    #         Array of 'inductance values' between plasma grid points and all vessel coils
    #     """
    #     coils_dict = eq.tokamak.coils_dict
    #     mey = np.zeros((eq.tokamak.n_coils, len(self.plasma_pts)))
    #     for j, labelj in enumerate(eq.tokamak.coils_list):
    #         greenm = green_f(
    #             coils_dict[labelj]["coords"][0][np.newaxis, :],
    #             coils_dict[labelj]["coords"][1][np.newaxis, :],
    #             self.plasma_pts[:, 0, np.newaxis],
    #             self.plasma_pts[:, 1, np.newaxis],
    #         )
    #         greenm *= coils_dict[labelj]["polarity"][np.newaxis, :]
    #         greenm *= coils_dict[labelj]["multiplier"][np.newaxis, :]
    #         mey[j] = np.sum(greenm, axis=-1)
    #     return 2 * np.pi * mey
