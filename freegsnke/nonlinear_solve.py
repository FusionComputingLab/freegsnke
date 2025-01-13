"""
Implements the core non-linear solver for the evolutive GS problem. Also handles the linearised evolution capabilites.

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

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from . import nk_solver_H as nk_solver
from .circuit_eq_metal import metal_currents
from .circuit_eq_plasma import plasma_current
from .GSstaticsolver import NKGSsolver
from .linear_solve import linear_solver
from .simplified_solve import simplified_solver_J1


class nl_solver:
    """Handles all time-evolution capabilites.
    Includes interface to use both:
    - stepper of the linearised problem
    - stepper for the full non-linear problem
    """

    def __init__(
        self,
        profiles,
        eq,
        max_mode_frequency=10**2.5,
        full_timestep=0.0001,
        max_internal_timestep=0.0001,
        plasma_resistivity=1e-6,
        plasma_norm_factor=1000,
        nbroad=1,
        blend_hatJ=0,
        dIydI=None,
        target_dIy=1e-3,
        automatic_timestep=False,
        mode_removal=True,
        linearize=True,
        min_dIy_dI=0.1,
        verbose=False,
        custom_coil_resist=None,
        custom_self_ind=None,
    ):
        """Initializes the time-evolution Object.

        Parameters
        ----------
        profiles : FreeGSNKE profile Object
            Profile function of the initial equilibrium.
            This will be used to set up the linearization used by the linear evolutive solver.
            It can be changed later by initializing a new set of initial conditions.
            Note however that, to change either the machine or limiter properties
            it will be necessary to instantiate a new nl_solver object.
        eq : FreeGSNKE equilibrium Object
            Initial equilibrium. This is used to set the domain/grid properties
            as well as the machine properties.
            Furthermore, eq will be used to set up the linearization used by the linear evolutive solver.
            It can be changed later by initializing a new set of initial conditions.
            Note however that, to change either the machine or limiter properties
            it will be necessary to instantiate a new nl_solver object.
        max_mode_frequency : float
            Threshold value used to include/exclude vessel normal modes.
            Only modes with smaller characteristic frequencies (larger timescales) are retained.
            If None, max_mode_frequency is set based on the input timestep: max_mode_frequency = 1/(5*full_timestep)
        full_timestep : float, optional, by default .0001
            The stepper advances the dynamics by a time interval dt=full_timestep.
            Applies to both linear and non-linear stepper.
            A GS equilibrium is calculated every full_timestep.
            Note that this input is overridden by 'automatic_timestep' if the latter is not set to False.
        max_internal_timestep : float, optional, by default .0001
            Each time advancement of one full_timestep is divided in several sub-steps,
            with size of, at most, max_internal_timestep.
            Such sub_step is used to advance the circuit equations
            (under the assumption of constant applied voltage during the full_timestep).
            Note that this input is overridden by 'automatic_timestep' if the latter is not set to False.
        plasma_resistivity : float, optional, by default 1e-6
            Resistivity of the plasma. Plasma resistance values for each of the domain grid points are
            2*np.pi*plasma_resistivity*eq.R/(dR*dZ)
            where dR*dZ is the area of the domain element.
        plasma_norm_factor : float, optional, by default 1000
            The plasma current is re-normalised by this factor,
            to bring to a value more akin to those of the metal currents
        nbroad : int, optional, by default 1
            enables the use of a smoothing (with a square smoothing filter of nbroad grid points per side)
            when building the plasma current distribution used to contract the plasma circuit equations
        blend_hatJ : float, optional, by default 0
            optional coefficient which enables use a blended version of the normalised plasma current distribution
            when contracting the plasma lumped circuit eq. from the left. The blend combines the
            current distribution at time t with (a guess for) the one at time t+dt.
        dIydI : np.array of size (np.sum(plasma_domain_mask), n_metal_modes+1), optional
            dIydI_(i,j) = d(Iy_i)/d(I_j)
            This is the jacobian of the plasma current distribution Iy with respect to all
            independent metal currents (both active and vessel modes) and to the total plasma current
            This is provided if known, otherwise calculated here at the linearization eq
        automatic_timestep : (float, float) or False, optional, by default False
            If not False, this overrides inputs full_timestep and max_internal_timestep:
            the timescales of the linearised problem are used to set the size of the timestep.
            The input eq and profile are used to calculate the fastest growthrate, t_growthrate, henceforth,
            full_timestep = automatic_timestep[0]*t_growthrate
            max_internal_timestep = automatic_timestep[1]*full_timestep
        mode_removal : bool, optional, by default True
            It True, vessel normal modes that have little influence on the plasma are dropped.
            Criterion is based on norm of d(Iy)/dI of the mode.
        linearize : bool, optional, by default True
            Whether to set up the linearization of the evolutive problem
        min_dIy_dI : float, optional, by default 1
            Threshold value to include/drop vessel normal modes.
            Modes with norm(d(Iy)/dI)<min_dIy_dI are dropped.
        custom_coil_resist : np.array
            1d array of resistance values for all machine conducting elements,
            including both active coils and passive structures
            If None, the values calculated in machine_config will be sourced and used
        custom_self_ind : np.array
            2d matrix of mutual inductances between all pairs of machine conducting elements,
            including both active coils and passive structures
            If None, the values calculated in machine_config will be sourced and used
        """

        self.nx = np.shape(eq.R)[0]
        self.ny = np.shape(eq.R)[1]
        self.nxny = self.nx * self.ny
        self.eqR = eq.R
        self.eqZ = eq.Z

        # area factor for Iy
        dR = eq.R[1, 0] - eq.R[0, 0]
        dZ = eq.Z[0, 1] - eq.Z[0, 0]
        self.dRdZ = dR * dZ

        # store number of coils and their names/order
        self.n_active_coils = eq.tokamak.n_active_coils
        self.n_coils = eq.tokamak.n_coils
        self.coils_order = list(eq.tokamak.coils_dict.keys())

        # instantiating static GS solver on eq's domain
        self.NK = NKGSsolver(eq)

        # setting up reduced domain for plasma circuit eq.:
        self.limiter_handler = profiles.limiter_handler
        self.plasma_domain_mask = self.limiter_handler.mask_inside_limiter
        self.plasma_domain_size = np.sum(self.plasma_domain_mask)

        # Extract relevant information on the type of profile function used and on the actual value of associated parameters
        self.get_profiles_values(profiles)

        self.plasma_norm_factor = plasma_norm_factor
        self.dt_step = full_timestep
        self.max_internal_timestep = max_internal_timestep
        self.set_plasma_resistivity(plasma_resistivity)

        if max_mode_frequency is None:
            self.max_mode_frequency = 1 / (5 * full_timestep)
            print(
                "Value of max_mode_frequency has not been provided. Set to",
                self.max_mode_frequency,
                "based on value of full_timestep as provided.",
            )
        else:
            self.max_mode_frequency = max_mode_frequency

        # handles the metal circuit eq, mode properties, and performs the vessel mode decomposition
        self.evol_metal_curr = metal_currents(
            flag_vessel_eig=1,
            flag_plasma=1,
            plasma_pts=self.limiter_handler.plasma_pts,
            max_mode_frequency=self.max_mode_frequency,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.dt_step,
            coil_resist=custom_coil_resist,
            coil_self_ind=custom_self_ind,
        )

        # this is the number of independent normal mode currents being used
        self.n_metal_modes = self.evol_metal_curr.n_independent_vars
        self.arange_currents = np.arange(self.n_metal_modes + 1)

        self.evol_plasma_curr = plasma_current(
            plasma_pts=self.limiter_handler.plasma_pts,
            Rm1=np.diag(self.evol_metal_curr.Rm1),
            P=self.evol_metal_curr.P,
            plasma_resistance_1d=self.plasma_resistance_1d,
            Mye=self.evol_metal_curr.Mey_matrix.T,
        )

        # This solves the system of circuit eqs based on an assumption
        # for the direction of the plasma current distribution at time t+dt
        self.simplified_solver_J1 = simplified_solver_J1(
            Lambdam1=self.evol_metal_curr.Lambdam1,
            Pm1=self.evol_metal_curr.Pm1,
            Rm1=np.diag(self.evol_metal_curr.Rm1),
            Mey=self.evol_metal_curr.Mey_matrix,
            Myy=self.evol_plasma_curr.Myy_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            full_timestep=self.dt_step,
        )

        # self.vessel_currents_vec is the vector of tokamak coil currents (the metal values, not the normal modes)
        # initial self.vessel_currents_vec values are taken from eq.tokamak
        # does not include plasma current
        vessel_currents_vec = np.zeros(self.n_coils)
        eq_currents = eq.tokamak.getCurrents()
        for i, labeli in enumerate(self.coils_order):
            vessel_currents_vec[i] = eq_currents[labeli]
        self.vessel_currents_vec = vessel_currents_vec.copy()

        # self.currents_vec is the vector of current values in which the dynamics is actually solved for
        # it includes: active coils, vessel normal modes, total plasma current
        # the total plasma current is divided by plasma_norm_factor to improve homogeneity of current values
        self.extensive_currents_dim = self.n_metal_modes + 1
        self.currents_vec = np.zeros(self.extensive_currents_dim)
        self.circuit_eq_residual = np.zeros(self.extensive_currents_dim)

        # Handles the linearised dynamic problem
        self.linearised_sol = linear_solver(
            Lambdam1=self.evol_metal_curr.Lambdam1,
            Pm1=self.evol_metal_curr.Pm1,
            Rm1=np.diag(self.evol_metal_curr.Rm1),
            Mey=self.evol_metal_curr.Mey_matrix,
            Myy=self.evol_plasma_curr.Myy_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.dt_step,
        )

        # sets up NK solver on the full grid, to be used when solving the
        # circuits equations as a problem in the plasma flux
        self.psi_nk_solver = nk_solver.nksolver(self.nxny, verbose=True)

        # sets up NK solver for the currents
        self.currents_nk_solver = nk_solver.nksolver(
            self.extensive_currents_dim, verbose=True
        )

        # counter for the step advancement of the dynamics
        self.step_no = 0

        # this is the filter used to broaden the normalised plasma current distribution
        # used to contract the system of plasma circuit equations
        self.ones_to_broaden = np.ones((nbroad, nbroad))
        # use convolution if nbroad>1
        if nbroad > 1:
            self.make_broad_hatIy = lambda x: self.make_broad_hatIy_conv(
                x, blend=blend_hatJ
            )
        else:
            self.make_broad_hatIy = lambda x: self.make_broad_hatIy_noconv(
                x, blend=blend_hatJ
            )

        # self.dIydI is the Jacobian of the plasma current distribution
        # with respect to the independent currents (as in self.currents_vec)
        self.dIydI_ICs = dIydI
        self.dIydI = dIydI

        # initialize and set up the linearization
        # input value for dIydI is used when available
        if automatic_timestep == False:
            automatic_timestep_flag = False
        else:
            if len(automatic_timestep) != 2:
                raise ValueError(
                    "The automatic_timestep input has not been set to False, but an appropriate input has not been provided. Please revise."
                )
            automatic_timestep_flag = True
        if automatic_timestep_flag + mode_removal + linearize:
            # builds the linearization and sets everything up for the stepper
            self.initialize_from_ICs(
                eq,
                profiles,
                rtol_NK=1e-7,
                dIydI=dIydI,
                target_dIy=target_dIy,
                verbose=verbose,
            )

        # remove passive normal modes that do not affect the plasma
        if mode_removal:
            # currently, axes of dIydI are ordered by their characteristic timescale after the active coils, up to threshold max_mode_frequency
            self.selected_modes_mask = np.linalg.norm(self.dIydI, axis=0) > min_dIy_dI
            self.selected_modes_mask = np.concatenate(
                (
                    np.ones(self.n_active_coils),
                    self.selected_modes_mask[self.n_active_coils : -1],
                    np.ones(1),
                )
            ).astype(bool)
            self.dIydI = self.dIydI[:, self.selected_modes_mask]
            self.dIydI_ICs = np.copy(self.dIydI)

            # rebuild mask of selected modes with respect to list of all modes, to be used by evolve_metal_currents
            self.selected_modes_mask = np.concatenate(
                (
                    self.selected_modes_mask[:-1],
                    np.zeros(self.n_coils - self.n_metal_modes),
                )
            ).astype(bool)
            self.remove_modes(self.selected_modes_mask)

        # check if input equilibrium and associated linearization have an instability, and its timescale
        if automatic_timestep_flag + mode_removal + linearize:
            self.linearised_sol.calculate_linear_growth_rate()
            if len(self.linearised_sol.growth_rates):
                print(
                    "The linear growth rate of this equilibrium corresponds to a characteristic timescale of",
                    self.linearised_sol.instability_timescale,
                    "s.",
                )
            else:
                print(
                    "No unstable modes found.",
                    "Either plasma is stable or, more likely, it is Alfven unstable (i.e. not enough stabilization from any metal structures).",
                )
                print(
                    "Try adding more passive modes by increasing the input values of max_mode_frequency and/or by reducing min_dIy_dI.",
                )

        # if automatic_timestep, reset the timestep accordingly,
        # note that this requires having found an instability
        if automatic_timestep_flag is False:
            print(
                "The solver's timestep has been set to",
                self.dt_step,
                "as explicitly requested. Please compare this with the linear growth rate and reset if necessary.",
            )
        else:
            if len(self.linearised_sol.growth_rates):
                dt_step = abs(
                    self.linearised_sol.instability_timescale[0] * automatic_timestep[0]
                )
                self.reset_timestep(
                    full_timestep=dt_step,
                    max_internal_timestep=dt_step / automatic_timestep[1],
                )
                print(
                    "The solver's timestep has been reset at",
                    self.dt_step,
                    "using the calculated linear growth rate and the provided values for the input automatic_timestep.",
                )
            else:
                print(
                    "No unstable modes found. It is impossible to automatically set timestep!, please do so manually."
                )

        # text for verbose mode
        self.text_nk_cycle = "This is NK cycle no {nkcycle}."
        self.text_psi_0 = "NK on psi has been skipped {skippedno} times. The residual on psi is {psi_res:.8f}."
        self.text_psi_1 = "The coefficients applied to psi are"

    def remove_modes(self, selected_modes_mask):
        """It actions the removal of the unselected normal modes.
        Given a setup with a set of normal modes and a resulting size of the vector self.currents_vec,
        modes that are not selected in the input mask are removed and the circuit equations updated accordingly.
        The dimensionality of the vector self.currents_vec is reduced.

        Parameters
        ----------
        selected_modes_mask : np.array of bool values,
            shape(selected_modes_mask) = shape(self.currents_vec) at the time of calling the function
            indexes corresponding to True are kept, indexes corresponding to False are dropped
        """
        print(
            "Mode removal is ON: the input 'min_dIy_dI' corresponds to keeping",
            np.sum(selected_modes_mask),
            "out of the original",
            self.n_metal_modes,
            "metal modes.",
        )
        self.evol_metal_curr.initialize_for_eig(selected_modes_mask)
        self.n_metal_modes = self.evol_metal_curr.n_independent_vars
        self.extensive_currents_dim = self.n_metal_modes + 1
        self.arange_currents = np.arange(self.n_metal_modes + 1)
        self.currents_vec = np.zeros(self.extensive_currents_dim)
        self.circuit_eq_residual = np.zeros(self.extensive_currents_dim)
        self.currents_nk_solver = nk_solver.nksolver(self.extensive_currents_dim)

        self.evol_plasma_curr.reset_modes(P=self.evol_metal_curr.P)

        self.simplified_solver_J1 = simplified_solver_J1(
            Lambdam1=self.evol_metal_curr.Lambdam1,
            Pm1=self.evol_metal_curr.Pm1,
            Rm1=np.diag(self.evol_metal_curr.Rm1),
            Mey=self.evol_metal_curr.Mey_matrix,
            Myy=self.evol_plasma_curr.Myy_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            full_timestep=self.dt_step,
        )

        self.linearised_sol = linear_solver(
            Lambdam1=self.evol_metal_curr.Lambdam1,
            Pm1=self.evol_metal_curr.Pm1,
            Rm1=np.diag(self.evol_metal_curr.Rm1),
            Mey=self.evol_metal_curr.Mey_matrix,
            Myy=self.evol_plasma_curr.Myy_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.dt_step,
        )

        self.linearised_sol.set_linearization_point(
            dIydI=self.dIydI,
            hatIy0=self.broad_hatIy,
        )

    def set_linear_solution(self, active_voltage_vec, d_profile_pars_dt=None):
        """Uses the solver of the linearised problem to set up an initial guess for the nonlinear solver
        for the currents at time t+dt. Uses self.currents_vec as I(t).
        Solves GS at time t+dt using the currents derived from the linearised dynamics.

        Parameters
        ----------
        active_voltage_vec : np.array
            Vector of external voltage applied to the active coils during the timestep.
        d_profile_pars_dt : None
            Does not currently use d_profile_pars_dt
            The evolution of the profile coefficient is not accounted by the linearised dynamical evolution.
        """

        self.trial_currents = self.linearised_sol.stepper(
            It=self.currents_vec,
            active_voltage_vec=active_voltage_vec,
            d_profile_pars_dt=d_profile_pars_dt,
        )
        self.assign_currents_solve_GS(self.trial_currents, self.rtol_NK)
        self.trial_plasma_psi = np.copy(self.eq2.plasma_psi)

    # not used atm, used when building the Jacobian of the plasma current distribution with respect to the coefficients of the profile object
    def prepare_build_dIydpars(self, profiles, rtol_NK, target_dIy, starting_dpars):
        """Prepares to compute the term d(Iy)/d(alpha_m, alpha_n, profifile_par)
        where profile_par = paxis or betap.
        It infers the value of delta(indep_variable) corresponding to a change delta(I_y)
        with norm(delta(I_y))=target_dIy.

        Parameters
        ----------
        profiles : FreeGS4E profile object
            The profile object of the initial condition equilibrium, i.e. the linearization point.
        rtol_NK : float
            Relative tolerance to be used in the static GS problems.
        target_dIy : float
            Target value for the norm of delta(I_y), on which th finite difference derivative is calculated.
        starting_dpars : tuple (d_alpha_m, d_alpha_n, relative_d_profile_par)
            Initial value to be used as delta(indep_variable) to infer the slope of norm(delta(I_y))/delta(indep_variable).
            Note that the first two values in the tuple are absolute deltas,
            while the third value is relative, d_profile_par = relative_d_profile_par * profile_par
        """

        current_ = np.copy(self.currents_vec)

        # vary alpha_m
        self.check_and_change_profiles(
            profile_coefficients=(
                profiles.alpha_m + starting_dpars[0],
                profiles.alpha_n,
            )
        )
        self.assign_currents_solve_GS(current_, rtol_NK)
        dIy_0 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        rel_ndIy_0 = np.linalg.norm(dIy_0) / self.nIy
        self.final_dpars_record[0] = starting_dpars[0] * target_dIy / rel_ndIy_0

        # vary alpha_n
        self.check_and_change_profiles(
            profile_coefficients=(
                profiles.alpha_m,
                profiles.alpha_n + starting_dpars[1],
            )
        )
        self.assign_currents_solve_GS(current_, rtol_NK)
        dIy_0 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        rel_ndIy_0 = np.linalg.norm(dIy_0) / self.nIy
        self.final_dpars_record[1] = starting_dpars[1] * target_dIy / rel_ndIy_0

        # vary paxis or betap
        self.check_and_change_profiles(
            profile_coefficients=(profiles.alpha_m, profiles.alpha_n),
            profile_parameter=(1 + starting_dpars[2]) * profiles.profile_parameter,
        )
        self.assign_currents_solve_GS(current_, rtol_NK)
        dIy_0 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        rel_ndIy_0 = np.linalg.norm(dIy_0) / self.nIy
        self.final_dpars_record[2] = (
            starting_dpars[2] * profiles.profile_parameter * target_dIy / rel_ndIy_0
        )

    # not used atm, builds the Jacobian of the plasma current distribution with respect to the coefficients of the profile object
    # can only handle ConstrainPaxisIp and ConstrainBetapIp profile families.
    def build_dIydIpars(self, profiles, rtol_NK, verbose=False):
        """Compute the matrix d(Iy)/d(alpha_m, alpha_n, profifile_par) as a finite difference derivative,
        using the value of delta(indep_viriable) inferred earlier by self.prepare_build_dIypars.

        Parameters
        ----------
        profiles : FreeGS4E profile object
            The profile object of the initial condition equilibrium, i.e. the linearization point.
        rtol_NK : float
            Relative tolerance to be used in the static GS problems.

        """

        current_ = np.copy(self.currents_vec)

        # vary alpha_m
        self.check_and_change_profiles(
            profile_coefficients=(
                profiles.alpha_m + self.final_dpars_record[0],
                profiles.alpha_n,
            ),
            profile_parameter=profiles.profile_parameter,
        )
        self.assign_currents_solve_GS(current_, rtol_NK)
        dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        self.dIydpars[:, 0] = dIy_1 / self.final_dpars_record[0]
        if verbose:
            print(
                "alpha_m gradient calculated on the finite difference: delta_alpha_m =",
                self.final_dpars_record[0],
                ", norm(deltaIy) =",
                np.linalg.norm(dIy_1),
            )

        # vary alpha_n
        self.check_and_change_profiles(
            profile_coefficients=(
                profiles.alpha_m,
                profiles.alpha_n + self.final_dpars_record[1],
            )
        )
        self.assign_currents_solve_GS(current_, rtol_NK)
        dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        self.dIydpars[:, 1] = dIy_1 / self.final_dpars_record[1]
        if verbose:
            print(
                "alpha_n gradient calculated on the finite difference: delta_alpha_n =",
                self.final_dpars_record[1],
                ", norm(deltaIy) =",
                np.linalg.norm(dIy_1),
            )

        # vary paxis or betap
        self.check_and_change_profiles(
            profile_coefficients=(profiles.alpha_m, profiles.alpha_n),
            profile_parameter=profiles.profile_parameter + self.final_dpars_record[2],
        )
        self.assign_currents_solve_GS(current_, rtol_NK)
        dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        self.dIydpars[:, 2] = dIy_1 / self.final_dpars_record[2]
        if verbose:
            print(
                "profile_par gradient calculated on the finite difference: delta_profile_par =",
                self.final_dpars_record[2],
                ", norm(deltaIy) =",
                np.linalg.norm(dIy_1),
            )

    def prepare_build_dIydI_j(
        self, j, rtol_NK, target_dIy, starting_dI, min_curr=1e-4, max_curr=300
    ):
        """Prepares to compute the term d(Iy)/dI_j of the Jacobian by
        inferring the value of delta(I_j) corresponding to a change delta(I_y)
        with norm(delta(I_y))=target_dIy.

        Parameters
        ----------
        j : int
            Index identifying the current to be varied. Indexes as in self.currents_vec.
        rtol_NK : float
            Relative tolerance to be used in the static GS problems.
        target_dIy : float
            Target value for the norm of delta(I_y), on which th finite difference derivative is calculated.
        starting_dI : float
            Initial value to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        min_curr : float, optional, by default 1e-4
            If inferred current value is below min_curr, clip to min_curr.
        max_curr : int, optional, by default 300
            If inferred current value is above min_curr, clip to max_curr.
        """
        current_ = np.copy(self.currents_vec)
        current_[j] += starting_dI
        self.assign_currents_solve_GS(current_, rtol_NK)

        dIy_0 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy

        rel_ndIy_0 = np.linalg.norm(dIy_0) / self.nIy
        final_dI = starting_dI * target_dIy / rel_ndIy_0
        final_dI = np.clip(final_dI, min_curr, max_curr)
        self.final_dI_record[j] = final_dI

    def build_dIydI_j(self, j, rtol_NK, verbose=False):
        """Computes the term d(Iy)/dI_j of the Jacobian as a finite difference derivative,
        using the value of delta(I_j) inferred earlier by self.prepare_build_dIydI_j.

        Parameters
        ----------
        j : int
            Index identifying the current to be varied. Indexes as in self.currents_vec.
        rtol_NK : float
            Relative tolerance to be used in the static GS problems.

        Returns
        -------
        dIydIj : np.array finite difference derivative d(Iy)/dI_j.
            This is a 1d vector including all grid points in reduced domain, as from plasma_domain_mask.
        """

        final_dI = 1.0 * self.final_dI_record[j]
        self.current_at_last_linearization[j] = self.currents_vec[j]

        current_ = np.copy(self.currents_vec)
        current_[j] += final_dI
        self.assign_currents_solve_GS(current_, rtol_NK)

        dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        dIydIj = dIy_1 / final_dI

        if verbose:
            print(
                "dimension",
                j,
                "in the vector of metal currents,",
                "gradient calculated on the finite difference: norm(deltaI) = ",
                final_dI,
                ", norm(deltaIy) =",
                np.linalg.norm(dIy_1),
            )

        return dIydIj

    def build_linearization(
        self,
        eq,
        profile,
        dIydI=None,
        rtol_NK=1e-8,
        target_dIy=1e-3,
        starting_dI=None,
        verbose=False,
    ):
        """Builds the Jacobian d(Iy)/dI to set up the solver of the linearised problem.

        Parameters
        ----------
        eq : FreeGS4E equilibrium Object
            Equilibrium around which to linearise.
        profile : FreeGS4E profile Object
            Profile properties of the equilibrium around which to linearise.
        dIydI : np.array
            input Jacobian, enter where available, otherwise this will be calculated here
        rtol_NK : float
            Relative tolerance to be used in the static GS problems.
        target_dIy : float, by default 0.001.
            Target relative value for the norm of delta(I_y), on which the finite difference derivative is calculated.
        starting_dI : float, by default .5.
            Initial value to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        """

        if (dIydI is None) and (self.dIydI is None):
            self.NK.forward_solve(eq, profile, target_relative_tolerance=rtol_NK)
            self.build_current_vec(eq, profile)
            self.Iy = self.limiter_handler.Iy_from_jtor(profile.jtor).copy()
            self.nIy = np.linalg.norm(self.Iy)

        self.R0 = np.sum(eq.R * profile.jtor) / np.sum(profile.jtor)
        self.Z0 = np.sum(eq.Z * profile.jtor) / np.sum(profile.jtor)
        self.dRZdI = np.zeros((2, self.n_metal_modes + 1))

        if starting_dI is None:
            starting_dI = np.abs(self.currents_vec.copy()) * target_dIy
            starting_dI[:-1] = np.where(starting_dI[:-1] > 50, starting_dI[:-1], 50)

        # build/update dIydI
        if dIydI is None:
            if self.dIydI_ICs is None:
                print(
                    "Linearising with respect to the currents - this may take a minute or two."
                )
                self.dIydI = np.zeros((self.plasma_domain_size, self.n_metal_modes + 1))
                self.ddIyddI = np.zeros(self.n_metal_modes + 1)
                self.final_dI_record = np.zeros(self.n_metal_modes + 1)

                for j in self.arange_currents:
                    if verbose:
                        print("direction", j, "initial current shift", starting_dI[j])
                    self.prepare_build_dIydI_j(j, rtol_NK, target_dIy, starting_dI[j])

                for j in self.arange_currents:
                    self.dIydI[:, j] = self.build_dIydI_j(j, rtol_NK, verbose)
                    R0 = np.sum(eq.R * self.profiles2.jtor) / np.sum(
                        self.profiles2.jtor
                    )
                    Z0 = np.sum(eq.Z * self.profiles2.jtor) / np.sum(
                        self.profiles2.jtor
                    )
                    self.dRZdI[0, j] = (R0 - self.R0) / self.final_dI_record[j]
                    self.dRZdI[1, j] = (Z0 - self.Z0) / self.final_dI_record[j]

                self.dIydI_ICs = np.copy(self.dIydI)
            else:
                self.dIydI = np.copy(self.dIydI_ICs)
        else:
            self.dIydI = dIydI
            self.dIydI_ICs = np.copy(self.dIydI)

    def set_plasma_resistivity(self, plasma_resistivity):
        """Function to set the resistivity of the plasma.
        self.plasma_resistance_1d is the diagonal of the matrix R_yy, the plasma resistance matrix.
        Note that it only spans the grid points in the reduced domain, as from plasma_domain_mask.

        Parameters
        ----------
        plasma_resistivity : float
            Resistivity of the plasma. Plasma resistance values for each of the domain grid points are
            2*np.pi*plasma_resistivity*eq.R/(dR*dZ)
            where dR*dZ is the area of the domain element.
        """
        self.plasma_resistivity = plasma_resistivity
        plasma_resistance_matrix = (
            self.eqR * (2 * np.pi / self.dRdZ) * self.plasma_resistivity
        )
        self.plasma_resistance_1d = plasma_resistance_matrix[self.plasma_domain_mask]

    def reset_plasma_resistivity(self, plasma_resistivity):
        """Function to reset the resistivity of the plasma.

        Parameters
        ----------
        plasma_resistivity : float
            Resistivity of the plasma. Plasma resistance values for each of the domain grid points are
            2*np.pi*plasma_resistivity*eq.R/(dR*dZ)
            where dR*dZ is the area of the domain element.
        """

        self.plasma_resistivity = plasma_resistivity
        plasma_resistance_matrix = (
            self.eqR * (2 * np.pi / self.dRdZ) * self.plasma_resistivity
        )
        self.plasma_resistance_1d = plasma_resistance_matrix[self.plasma_domain_mask]

        self.linearised_sol.reset_plasma_resistivity(self.plasma_resistance_1d)
        self.simplified_solver_J1.reset_plasma_resistivity(self.plasma_resistance_1d)
        self.evol_plasma_curr.Ryy = self.plasma_resistance_1d

    def check_and_change_plasma_resistivity(
        self, plasma_resistivity, relative_threshold_difference=0.01
    ):
        """Checks if the plasma resistivity is different and resets it.

        Parameters
        ----------
        plasma_resistivity : float
            Resistivity of the plasma. Plasma resistance values for each of the domain grid points are
            2*np.pi*plasma_resistivity*eq.R/(dR*dZ)
            where dR*dZ is the area of the domain element.
        """

        if plasma_resistivity is not None:
            # check how different
            check = (
                np.abs(plasma_resistivity - self.plasma_resistivity)
                / self.plasma_resistivity
            ) > relative_threshold_difference
            if check:
                self.reset_plasma_resistivity(plasma_resistivity=plasma_resistivity)

    def calc_lumped_plasma_resistance(self, norm_red_Iy0, norm_red_Iy1):
        """Uses the plasma resistance matrix R_yy to calculate the lumped plasma resistance,
        by contracting this with the vectors norm_red_Iy0, norm_red_Iy0.
        These should be normalised plasma current distribution vectors.

        Parameters
        ----------
        norm_red_Iy0 : np.array
            Normalised plasma current distribution. This vector should sum to 1.
        norm_red_Iy1 : np.array
            Normalised plasma current distribution. This vector should sum to 1.

        Returns
        -------
        float
            Lumped resistance of the plasma.
        """
        lumped_plasma_resistance = np.sum(
            self.plasma_resistance_1d * norm_red_Iy0 * norm_red_Iy1
        )
        return lumped_plasma_resistance

    def reset_timestep(self, full_timestep, max_internal_timestep):
        """Allows for a reset of the timesteps.

        Parameters
        ----------
        full_timestep : float
            The stepper advances the dynamics by a time interval dt=full_timestep.
            Applies to both linear and non-linear stepper.
            A GS equilibrium is calculated every full_timestep.
        max_internal_timestep : float
            Each time advancement of one full_timestep is divided in several sub-steps,
            with size of, at most, max_internal_timestep.
            Such sub_step is used to advance the circuit equations
        """
        self.dt_step = full_timestep
        self.max_internal_timestep = max_internal_timestep

        self.evol_metal_curr.reset_mode(
            flag_vessel_eig=1,
            flag_plasma=1,
            plasma_pts=self.limiter_handler.plasma_pts,
            max_mode_frequency=self.max_mode_frequency,
            max_internal_timestep=max_internal_timestep,
            full_timestep=full_timestep,
        )

        self.simplified_solver_J1.reset_timesteps(
            full_timestep=full_timestep, max_internal_timestep=full_timestep
        )

        self.linearised_sol.reset_timesteps(
            full_timestep=full_timestep, max_internal_timestep=max_internal_timestep
        )

    def get_profiles_values(self, profiles):
        """Extracts profile properties.

        Parameters
        ----------
        profiles : FreeGS4E profile Object
            Profile function of the initial equilibrium.
        """
        self.fvac = profiles.fvac

        self.profile_type = type(profiles).__name__

        # note the parameters here are the same that should be provided
        # to the stepper if these are time evolving
        if self.profile_type == "ConstrainPaxisIp":
            self.profile_parameters = {
                "paxis": profiles.paxis,
                "alpha_m": profiles.alpha_m,
                "alpha_n": profiles.alpha_n,
            }
        elif self.profile_type == "ConstrainBetapIp":
            self.profile_parameters = {
                "betap": profiles.betap,
                "alpha_m": profiles.alpha_m,
                "alpha_n": profiles.alpha_n,
            }
        elif self.profile_type == "Fiesta_Topeol":
            self.profile_parameters = {
                "beta0": profiles.beta0,
                "alpha_m": profiles.alpha_m,
                "alpha_n": profiles.alpha_n,
            }
        elif self.profile_type == "Lao85":
            self.profile_parameters = {"alpha": profiles.alpha, "beta": profiles.beta}

    def get_vessel_currents(self, eq):
        """Uses the input equilibrium to extract values for all metal currents,
        including active coils and vessel passive structures.
        These are stored in self.vessel_currents_vec

        Parameters
        ----------
        eq : FreeGSNKE equilibrium Object
            Initial equilibrium. eq.tokamak is used to extract current values.
        """
        eq_currents = eq.tokamak.getCurrents()
        for i, labeli in enumerate(self.coils_order):
            self.vessel_currents_vec[i] = eq_currents[labeli]

    def build_current_vec(self, eq, profile):
        """Builds the vector of currents in which the dynamics is actually solved, self.currents_vec
        This contains, in the order:
        (active coil currents, selected vessel normal modes currents, total plasma current/plasma_norm_factor)

        Parameters
        ----------
        profile : FreeGSNKE profile Object
            Profile function of the initial equilibrium. Used to extract the value of the total plasma current.
        eq : FreeGSNKE equilibrium Object
            Initial equilibrium. Used to extract the value of all metal currents.
        """
        # gets metal currents, note these are before mode truncation!
        self.get_vessel_currents(eq)

        # transforms in normal modes (including truncation)
        self.currents_vec[: self.n_metal_modes] = self.evol_metal_curr.IvesseltoId(
            Ivessel=self.vessel_currents_vec
        )

        # extracts total plasma current value
        self.currents_vec[-1] = profile.Ip / self.plasma_norm_factor

        # this is currents_vec(t-dt):
        self.currents_vec_m1 = np.copy(self.currents_vec)

    def initialize_from_ICs(
        self,
        eq,
        profile,
        rtol_NK=1e-7,
        dIydI=None,
        target_dIy=1e-3,
        verbose=False,
    ):
        """Uses the input equilibrium and profile as initial conditions and prepares to solve for their dynamics.
        If needed, sets the the linearised solver by calculating the Jacobian dIy/dI.

        Parameters
        ----------
        eq : FreeGS4E equilibrium Object
            Initial equilibrium. This assigns all initial metal currents.
        profiles : FreeGS4E profile Object
            Profile function of the initial equilibrium. This assigns the initial total plasma current.
        rtol_NK : float
            Relative tolerance to be used in the static GS problems in the initialization
            and when calculating the Jacobian dIy/dI to set up the linearised problem.
            This does not set the tolerance of the static GS solves used by the dynamic solver, which is set through the stepper itself.
        dIydI : np.array of size (np.sum(plasma_domain_mask), n_metal_modes+1), optional
            dIydI_(i,j) = d(Iy_i)/d(I_j)
            This is the jacobian of the plasma current distribution with respect to all
            independent metal currents (both active and vessel modes) and to the total plasma current
            If not provided, this is calculated based on the properties of the provided equilibrium.
        target_dIy : float
            Target value used to define the size of the finite difference step used
            when calculating the Jacobian dIydI
        verbose : bool
            Whether to allow progress printouts
        """

        self.step_counter = 0
        self.currents_guess = False
        self.rtol_NK = rtol_NK

        # get profile parametrization
        # this is not currently used, as the linearised evolution
        # does not currently account for the evolving profile coefficients
        self.get_profiles_values(profile)

        # set internal copy of the equilibrium and profile
        self.eq1 = deepcopy(eq)
        self.profiles1 = deepcopy(profile)
        # The pair self.eq1 and self.profiles1 is the pair that is advanced at each timestep.
        # Their properties evolve according to the dynamics.
        # Note that the input eq and profile are NOT modified by the evolution object.

        # this builds the vector of extensive current values 'currents_vec'
        # comprising (active coil currents, vessel normal modes, plasma current)
        # this vector is evolved in place when the stepper is called
        self.build_current_vec(self.eq1, self.profiles1)
        self.current_at_last_linearization = np.copy(self.currents_vec)

        # ensure internal equilibrium is a GS solution
        self.assign_currents(self.currents_vec, profile=self.profiles1, eq=self.eq1)
        self.NK.forward_solve(
            self.eq1, self.profiles1, target_relative_tolerance=rtol_NK
        )

        # self.eq2 and self.profiles2 are used as auxiliary objects when solving for the dynamics
        # they should not be used to extract properties of the evolving equilibrium
        self.eq2 = deepcopy(self.eq1)
        self.profiles2 = deepcopy(self.profiles1)

        # self.Iy is the istantaneous 1d vector representing the plasma current distribution
        # on the reduced plasma domain, as from plasma_domain_mask
        # self.Iy is updated every timestep
        self.Iy = self.limiter_handler.Iy_from_jtor(self.profiles1.jtor)
        # self.hatIy is the normalised plasma current distribution. This vector sums to 1.
        self.hatIy = self.limiter_handler.normalize_sum(self.Iy)
        # self.hatIy1 is the normalised plasma current distribution at time t+dt
        self.hatIy1 = np.copy(self.hatIy)
        self.make_broad_hatIy(self.hatIy1)

        self.time = 0
        self.step_no = -1

        # build the linearization if not provided
        self.build_linearization(
            self.eq1,
            self.profiles1,
            dIydI,
            rtol_NK,
            target_dIy=target_dIy,
            verbose=verbose,
        )

        # transfer linearization to linear solver
        self.linearised_sol.set_linearization_point(
            dIydI=self.dIydI_ICs,
            hatIy0=self.broad_hatIy,
        )

    def step_complete_assign(self, working_relative_tol_GS, from_linear=False):
        """This function completes the advancement by dt.
        The time-evolved currents as obtained by the stepper (self.trial_currents) are recorded
        in self.currents_vec and assigned to the equilibrium self.eq1.
        The time-evolved equilibrium properties (i.e. the plasma flux self.trial_plasma_psi and resulting current distribution)
        are recorded in self.eq1 and self.profiles1.


        Parameters
        ----------
        working_relative_tol_GS : float
            The relative tolerance of the GS solver used to solve the dynamics
            is set to a fraction of the change in the plasma flux associated to the timestep itself.
            The fraction is set through working_relative_tol_GS.
        from_linear : bool, optional, by default False
            If the stepper is only solving the linearised problem, use from_linear=True.
            This acellerates calculations by reducing the number of static GS solve calls.
        """

        self.time += self.dt_step
        self.step_no += 1

        self.currents_vec_m1 = np.copy(self.currents_vec)
        self.Iy_m1 = np.copy(self.Iy)

        plasma_psi_step = self.eq2.plasma_psi - self.eq1.plasma_psi
        self.d_plasma_psi_step = np.amax(plasma_psi_step) - np.amin(plasma_psi_step)

        self.currents_vec = np.copy(self.trial_currents)
        self.assign_currents(self.currents_vec, self.eq1, self.profiles1)

        if from_linear:
            self.profiles1 = deepcopy(self.profiles2)
            self.eq1 = deepcopy(self.eq2)
        else:
            self.eq1.plasma_psi = np.copy(self.trial_plasma_psi)
            self.profiles1.Ip = self.trial_currents[-1] * self.plasma_norm_factor
            self.tokamak_psi = self.eq1.tokamak.calcPsiFromGreens(
                pgreen=self.eq1._pgreen
            )
            self.profiles1.Jtor(
                self.eqR, self.eqZ, self.tokamak_psi + self.trial_plasma_psi
            )
            self.NK.port_critical(self.eq1, self.profiles1)

        self.Iy = self.limiter_handler.Iy_from_jtor(self.profiles1.jtor)
        self.hatIy = self.limiter_handler.normalize_sum(self.Iy)

        self.rtol_NK = working_relative_tol_GS * self.d_plasma_psi_step

    def assign_currents(self, currents_vec, eq, profile):
        """Assigns current values as in input currents_vec to eq.tokamak and plasma.
        The input eq and profile are modified accordingly.
        The format of the input currents aligns with self.currents_vec:
        (active coil currents, selected vessel normal modes currents, total plasma current/plasma_norm_factor)

        Parameters
        ----------
        currents_vec : np.array
            Input current values to be assigned.
        eq : FreeGSNKE equilibrium Object
            Equilibrium object to be modified.
        profiles : FreeGSNKE profile Object
            Profile object to be modified.
        """

        # assign plasma current to equilibrium
        eq._current = self.plasma_norm_factor * currents_vec[-1]
        profile.Ip = self.plasma_norm_factor * currents_vec[-1]

        # calculate vessel currents from normal modes and assign
        self.vessel_currents_vec = self.evol_metal_curr.IdtoIvessel(
            Id=currents_vec[:-1]
        )
        for i, labeli in enumerate(self.coils_order):
            eq.tokamak[labeli].current = self.vessel_currents_vec[i]

    def assign_currents_solve_GS(self, currents_vec, rtol_NK):
        """Assigns current values as in input currents_vec to private self.eq2 and self.profiles2.
        Static GS problem is accordingly solved, which finds the associated plasma flux and current distribution.

        Parameters
        ----------
        currents_vec : np.array
            Input current values to be assigned. Format as in self.assign_currents.
        rtol_NK : float
            Relative tolerance to be used in the static GS problem.
        """
        self.assign_currents(currents_vec, profile=self.profiles2, eq=self.eq2)
        self.NK.forward_solve(
            self.eq2, self.profiles2, target_relative_tolerance=rtol_NK
        )

    def make_broad_hatIy_conv(self, hatIy1, blend=0):
        """Averages the normalised plasma current distributions at time t and
        (a guess for the one at) at time t+dt to better contract the system of
        plasma circuit eqs. Applies some 'smoothing' though convolution, when
        setting is nbroad>1.

        Parameters
        ----------
        hatIy1 : np.array
            Guess for the normalised plasma current distributions at time t+dt.
            Should be a vector that sums to 1. Reduced plasma domain only.
        blend : float between 0 and 1
            Option to combine the normalised plasma current distributions at time t
            with (a guess for) the one at time t+dt before contraction of the plasma
            lumped circuit eq.
        """
        self.broad_hatIy = self.limiter_handler.rebuild_map2d(
            hatIy1 + blend * self.hatIy
        )
        self.broad_hatIy = convolve2d(
            self.broad_hatIy, self.ones_to_broaden, mode="same"
        )
        self.broad_hatIy = self.limiter_handler.hat_Iy_from_jtor(self.broad_hatIy)

    def make_broad_hatIy_noconv(self, hatIy1, blend=0):
        """Averages the normalised plasma current distributions at time t and
        (a guess for the one at) at time t+dt to better contract the system of
        plasma circuit eqs. Does not apply convolution: nbroad==1.

        Parameters
        ----------
        hatIy1 : np.array
            Guess for the normalised plasma current distributions at time t+dt.
            Should be a vector that sums to 1. Reduced plasma domain only.
        """
        self.broad_hatIy = self.limiter_handler.rebuild_map2d(
            hatIy1 + blend * self.hatIy
        )
        self.broad_hatIy = self.limiter_handler.hat_Iy_from_jtor(self.broad_hatIy)

    def currents_from_hatIy(self, hatIy1, active_voltage_vec):
        """Uses a guess for the normalised plasma current distribution at time t+dt
        to obtain all current values at time t+dt, through the 'simplified' circuit equations.

        Parameters
        ----------
        hatIy1 : np.array
            Guess for the normalised plasma current distribution at time t+dt.
            Should be a vector that sums to 1. Reduced plasma domain only.
        active_voltage_vec : np.array
            Vector of active voltages for the active coils, applied between t and t+dt.

        Returns
        -------
        np.array
            Current values at time t+dt. Same format as self.currents_vec.
        """
        self.make_broad_hatIy(hatIy1)
        current_from_hatIy = self.simplified_solver_J1.stepper(
            It=self.currents_vec,
            hatIy_left=self.broad_hatIy,
            hatIy_0=self.hatIy,
            hatIy_1=hatIy1,
            active_voltage_vec=active_voltage_vec,
        )
        return current_from_hatIy

    def hatIy1_iterative_cycle(self, hatIy1, active_voltage_vec, rtol_NK):
        """Uses a guess for the normalised plasma current distribution at time t+dt
        to obtain all current values at time t+dt through the circuit equations.
        Static GS is then solved for the same currents, which results in calculating
        the 'iterated' plasma flux and plasma current distribution at time t+dt
        (stored in the private self.eq2 and self.profiles2).

        Parameters
        ----------
        hatIy1 : np.array
            Guess for the normalised plasma current distribution at time t+dt.
            Should be a vector that sums to 1. Reduced plasma domain only.
        active_voltage_vec : np.array
            Vector of active voltages for the active coils, applied between t and t+dt.
        rtol_NK : float
            Relative tolerance to be used in the static GS problem.
        """
        current_from_hatIy = self.currents_from_hatIy(hatIy1, active_voltage_vec)
        self.assign_currents_solve_GS(currents_vec=current_from_hatIy, rtol_NK=rtol_NK)

    def calculate_hatIy(self, trial_currents, plasma_psi):
        """Finds the normalised plasma current distribution corresponding
        to the combination of the input current values and plasma flux.
        Note that this does not assume that current values and plasma flux
        together are a solution of GS.

        Parameters
        ----------
        trial_currents : np.array
            Vector of current values. Same format as self.currents_vec.
        plasma_psi : np.array
            Plasma flux values on full domain of shape (eq.nx, eq.ny), 2d.

        Returns
        -------
        np.array
            Normalised plasma current distribution. 1d vector on the reduced plasma domain.
        """
        self.assign_currents(trial_currents, profile=self.profiles2, eq=self.eq2)
        self.tokamak_psi = self.eq2.tokamak.calcPsiFromGreens(pgreen=self.eq2._pgreen)
        jtor_ = self.profiles2.Jtor(self.eqR, self.eqZ, self.tokamak_psi + plasma_psi)
        hat_Iy1 = self.limiter_handler.hat_Iy_from_jtor(jtor_)
        return hat_Iy1

    def calculate_hatIy_GS(self, trial_currents, rtol_NK, record_for_updates=False):
        """Finds the normalised plasma current distribution corresponding
        to the combination of the input current values by solving the static GS problem.

        Parameters
        ----------
        trial_currents : np.array
            Vector of current values. Same format as self.currents_vec.
        rtol_NK : float
            Relative tolerance to be used in the static GS problem.

        Returns
        -------
        np.array
            Normalised plasma current distribution. 1d vector on the reduced plasma domain.
        """
        self.assign_currents_solve_GS(
            trial_currents, rtol_NK=rtol_NK, record_for_updates=record_for_updates
        )
        hatIy1 = self.limiter_handler.hat_Iy_from_jtor(self.profiles2.jtor)
        return hatIy1

    def F_function_curr(self, trial_currents, active_voltage_vec):
        """Full non-linear system of circuit eqs written as root problem
        in the vector of current values at time t+dt.
        Note that the plasma flux at time t+dt is taken to be self.trial_plasma_psi.
        Iteration consists of:
        [trial_currents, plasma_flux] -> hatIy1, through calculating plasma distribution
        hatIy1 -> iterated_currents, through 'simplified' circuit eqs
        Residual: iterated_currents - trial_currents
        Residual is zero if the pair [trial_currents, plasma_flux] solve the full non-linear problem.

        Parameters
        ----------
        trial_currents : np.array
            Vector of current values. Same format as self.currents_vec.
        active_voltage_vec : np.array
            Vector of active voltages for the active coils, applied between t and t+dt.

        Returns
        -------
        np.array
            Residual in current values. Same format as self.currents_vec.
        """
        self.hatIy1_last = self.calculate_hatIy(trial_currents, self.trial_plasma_psi)
        iterated_currs = self.currents_from_hatIy(self.hatIy1_last, active_voltage_vec)
        current_res = iterated_currs - trial_currents
        return current_res

    def F_function_curr_GS(self, trial_currents, active_voltage_vec, rtol_NK):
        """Full non-linear system of circuit eqs written as root problem
        in the vector of current values at time t+dt.
        Note that, differently from self.F_function_curr, here the plasma flux
        is not imposed, but self-consistently solved for based on the input trial_currents.
        Iteration consists of:
        trial_currents -> plasma flux, through static GS
        [trial_currents, plasma_flux] -> hatIy1, through calculating plasma distribution
        hatIy1 -> iterated_currents, through 'simplified' circuit eqs
        Residual: iterated_currents - trial_currents
        Residual is zero if trial_currents solve the full non-linear problem.

        Parameters
        ----------
        trial_currents : np.array
            Vector of current values. Same format as self.currents_vec.
        active_voltage_vec : np.array
            Vector of active voltages for the active coils, applied between t and t+dt.
        rtol_NK : float
            Relative tolerance to be used in the static GS problem.

        Returns
        -------
        np.array
            Residual in current values. Same format as self.currents_vec.
        """
        self.hatIy1_last = self.calculate_hatIy_GS(
            trial_currents, rtol_NK=rtol_NK, record_for_updates=False
        )
        iterated_currs = self.currents_from_hatIy(self.hatIy1_last, active_voltage_vec)
        current_res = iterated_currs - trial_currents
        return current_res

    def F_function_psi(self, trial_plasma_psi, active_voltage_vec, rtol_NK):
        """Full non-linear system of circuit eqs written as root problem
        in the plasma flux. Note that the flux associated to the metal currents
        is sourced from self.tokamak_psi.
        Iteration consists of:
        [trial_plasma_psi, tokamak_psi] -> hatIy1, by calculating Jtor
        hatIy1 -> currents(t+dt), through 'simplified' circuit eq
        currents(t+dt) -> iterated_plasma_flux, through static GS
        Residual: iterated_plasma_flux - trial_plasma_psi
        Residual is zero if the pair [trial_plasma_psi, tokamak_psi] solve the full non-linear problem.

        Parameters
        ----------
        trial_plasma_psi : np.array
            Plasma flux values in 1d vector covering full domain of size eq.nx*eq.ny.
        active_voltage_vec : np.array
            Vector of active voltages for the active coils, applied between t and t+dt.
        rtol_NK : float
            Relative tolerance to be used in the static GS problem.

        Returns
        -------
        np.array
            Residual in plasma flux, 1d.
        """
        jtor_ = self.profiles2.Jtor(
            self.eqR,
            self.eqZ,
            (self.tokamak_psi + trial_plasma_psi).reshape(self.nx, self.ny),
        )
        hatIy1 = self.limiter_handler.hat_Iy_from_jtor(jtor_)
        self.hatIy1_iterative_cycle(
            hatIy1=hatIy1, active_voltage_vec=active_voltage_vec, rtol_NK=rtol_NK
        )
        psi_residual = self.eq2.plasma_psi.reshape(-1) - trial_plasma_psi
        return psi_residual

    def calculate_rel_tolerance_currents(self, current_residual, curr_eps):
        """Calculates how the current_residual in input compares to the step in the currents themselves,
        i.e. to the difference currents(t+dt) - currents(t-dt)
        This relative residual is used to quantify the relative convergence of the stepper.
        It accesses self.trial_currents and self.currents_vec_m1.

        Parameters
        ----------
        current_residual : np.array
            Residual in current values. Same format as self.currents_vec.
        curr_eps : float
            Min value of the current step. Avoids divergence when dividing by the step in the currents.

        Returns
        -------
        np.array
            Relative current residual. Same format as self.currents_vec.
        """
        curr_step = abs(self.trial_currents - self.currents_vec_m1)
        self.curr_step = np.where(curr_step > curr_eps, curr_step, curr_eps)
        rel_curr_res = abs(current_residual / self.curr_step)
        return rel_curr_res

    def calculate_rel_tolerance_GS(self, trial_plasma_psi, a_res_GS=None):
        """Calculates how the residual in the plasma flux due to the static GS problem
        compares to the change in the plasma flux itself due to the dynamics,
        i.e. to the difference psi(t+dt) - psi(t)
        The relative residual is used to quantify the relative convergence of the stepper.
        It accesses self.trial_plasma_psi, self.eq1.plasma_psi, self.tokamak_psi

        Parameters
        ----------
        trial_plasma_psi : ndarray
            psi(t+dt)
        a_res_GS : ndarray
            The residual of the static GS problem at t+dt

        Returns
        -------
        float
            Relative plasma flux residual.
        """
        plasma_psi_step = trial_plasma_psi - self.eq1.plasma_psi
        self.d_plasma_psi_step = np.amax(plasma_psi_step) - np.amin(plasma_psi_step)

        if a_res_GS is None:
            a_res_GS = self.NK.F_function(
                trial_plasma_psi.reshape(-1),
                self.tokamak_psi.reshape(-1),
                self.profiles2,
            )
        a_res_GS = np.amax(abs(a_res_GS))

        r_res_GS = a_res_GS / self.d_plasma_psi_step
        return r_res_GS

    def check_and_change_profiles(self, profile_parameters=None):
        """Checks if new input parameters are provided for the profiles at t+dt.
        If so, it actions the necessary changes.

        Parameters
        ----------
        profile_parameters : None or dictionary
            Set to None when the profile parameter are left unchanged.
            Dictionary otherwise.
            See 'get_profiles_values' for dictionary structure.
        """
        self.profile_change_flag = 0

        if profile_parameters is not None:
            for par in profile_parameters:
                setattr(self.profiles1, par, profile_parameters[par])
                setattr(self.profiles2, par, profile_parameters[par])
                if self.profile_type == "Lao85":
                    self.profiles1.initialize_profile()
                    self.profiles2.initialize_profile()
            self.profile_change_flag = 1

    def nlstepper(
        self,
        active_voltage_vec,
        profile_parameters=None,
        plasma_resistivity=None,
        target_relative_tol_currents=0.005,
        target_relative_tol_GS=0.005,
        working_relative_tol_GS=0.001,
        target_relative_unexplained_residual=0.5,
        max_n_directions=3,
        step_size_psi=2.0,
        step_size_curr=0.8,
        scaling_with_n=0,
        blend_GS=0.5,
        curr_eps=1e-5,
        max_no_NK_psi=5.0,
        clip=5,
        verbose=0,
        linear_only=False,
    ):
        """This is the main stepper function.
        If linear_only = True, this advances the linearised dynamic problem.
        If linear_only = False, a solution of the full non-linear problem is seeked using
        a combination of NK methods.
        When a solution has been found, time is advanced by self.dt_step,
        the new values of all extensive currents are recorded in self.currents_vec
        and new equilibrium and profile properties in self.eq1 and self.profiles1.

        The solver's algorithm proceeds like below:
        1) solve linearised problem to obtain an initial guess of the currents and solve
        the associated static GS problem, assign such trial_plasma_psi and trial_currents
        (including the resulting tokamak_psi);
        2) if pair [trial_plasma_psi, tokamak_psi] fails static GS tolerance check,
        update trial_plasma_psi bringing it closer to the actual GS solution;
        3) at fixed trial_currents (and associated tokamak_psi) update trial_plasma_psi
        using NK solver for the associated root problem;
        4) at fixed trial_plasma_psi, update trial_currents (and associated tokamak_psi)
        using NK solver for the associated root problem;
        5) if convergence on the current residuals is not reached or static GS tolerance check
        fails, restart from point 2;
        6) the pair [trial_currents, trial_plasma_psi] solves the nonlinear dynamic problem,
        assign values to self.currents_vec, self.eq1 and self.profiles1.


        Parameters
        ----------
        active_voltage_vec : np.array
            Vector of active voltages for the active coils, applied between t and t+dt.
        profile_parameters : Dictionary
            Set to None when the profile parameters are left unchanged.
            Otherwise, dictionary containing the relevant profile parameters
            for the profile object on which the evolution is calculated
            See 'get_profiles_values' for dictionary structure.
        target_relative_tol_currents : float, optional, by default .005
            Relative tolerance in the currents required for convergence of the dynamic problem.
            This is calculated with respect to the change in the currents themselves
            due to the dynamical evolution: residual/(currents(t+dt) - currents(t-dt))
        target_relative_tol_GS : float, optional, by default .005
            Relative tolerance in the plasma flux for the static GS problem required for convergence.
            This is calculated with respect to the change in the flux itself
            due to the dynamical evolution: residual/delta(psi(t+dt) - psi(t))
        working_relative_tol_GS : float, optional, by default .001
            Tolerance used when solving all static GS problems while executing the step,
            also expressed in relative terms as target_relative_tol_GS.
            Note this value needs to be smaller than target_relative_tol_GS to allow for convergence.
        target_relative_unexplained_residual : float, optional, by default .5
            Used in the NK solvers. Inclusion of additional Krylov basis vectors is
            stopped if the fraction of the residual (linearly) canceled is > 1-target_relative_unexplained_residual.
        max_n_directions : int, optional, by default 3
            Used in the NK solvers. Inclusion of additional Krylov basis vectors is
            stopped if max_n_directions have already been included.
        step_size_psi : float, optional, by default 2.
            Used by the NK solver applied to the root problem in the plasma flux.
            l2 norm of proposed step for the finite difference calculation, in units of the residual.
        step_size_curr : float, optional, by default .8
            Used by the NK solver applied to the root problem in the currents.
            l2 norm of proposed step for the finite difference calculation, in units of the residual.
        scaling_with_n : int, optional, by default 0
            Used in the NK solvers. Allows to further scale dx candidate steps by factor
            (1 + self.n_it)**scaling_with_n
        max_no_NK_psi : float, optional, by default 5.
            Execution of NK update on psi for the dynamic problem is triggered when
            relative_psi_residual > max_no_NK_psi * target_relative_tol_GS
            where the psi residual is calculated as for target_relative_tol_GS
        blend_GS : float, optional, by default .5
            Blend coefficient used in trial_plasma_psi updates at step 2 of the algorithm above.
            Should be between 0 and 1.
        curr_eps : float, optional, by default 1e-5
            Regulariser used in calculating the relative convergence on the currents.
            Avoids divergence when dividing by the advancement in the currents.
            Min value of the current per step.
        clip : float, optional, by default 5
            Used in the NK solvers. Maximum step size for each accepted basis vector, in units
            of the exploratory step.
        verbose : int, optional, by default 0
            Printouts of convergence process.
            Use 1 for printouts with details on each NK cycle.
            Use 2 for printouts with deeper intermediate details.
        linear_only : bool, optional, by default False
            If linear_only = True the solution of the linearised problem is accepted.
            If linear_only = False, the convergence criteria are used and a solution of
            the full nonlinear problem is seeked.
        """

        # check if profile parameters are being evolved
        # and action the change where necessary
        self.check_and_change_profiles(
            profile_parameters=profile_parameters,
        )

        # check if plasma resistivity is being evolved
        # and action the change where necessary
        self.check_and_change_plasma_resistivity(
            plasma_resistivity,
        )

        # solves the linearised problem for the currents and assigns
        # results in preparation for the nonlinear calculations
        # Solution and GS equilibrium are assigned to self.trial_currents and self.trial_plasma_psi
        self.set_linear_solution(active_voltage_vec)

        if linear_only:
            # assign currents and plasma flux to self.currents_vec, self.eq1 and self.profiles1 and complete step
            self.step_complete_assign(working_relative_tol_GS, from_linear=True)

        else:
            # seek solution of the full nonlinear problem

            # this assigns to self.eq2 and self.profiles2
            # also records self.tokamak_psi corresponding to self.trial_currents in 2d
            res_curr = self.F_function_curr(
                self.trial_currents, active_voltage_vec
            ).copy()

            # uses self.trial_currents and self.currents_vec_m1 to relate res_curr above to step advancement in the currents
            rel_curr_res = self.calculate_rel_tolerance_currents(
                res_curr, curr_eps=curr_eps
            ).copy()
            control = np.any(rel_curr_res > target_relative_tol_currents)

            # pair self.trial_currents and self.trial_plasma_psi are a GS solution
            r_res_GS = self.calculate_rel_tolerance_GS(self.trial_plasma_psi).copy()
            control_GS = 0

            args_nk = [active_voltage_vec, self.rtol_NK]

            if verbose:
                print("starting numerical solve:")
                print(
                    "max(residual on current eqs) =",
                    np.amax(rel_curr_res),
                    "mean(residual on current eqs) =",
                    np.mean(rel_curr_res),
                )
            log = []

            # counter for number of solution cycles
            n_it = 0

            while control:
                if verbose:
                    for _ in log:
                        print(_)

                log = [self.text_nk_cycle.format(nkcycle=n_it)]

                # update plasma flux if trial_currents and plasma_flux exceedingly far from GS solution
                if control_GS:
                    self.NK.forward_solve(self.eq2, self.profiles2, self.rtol_NK)
                    self.trial_plasma_psi *= 1 - blend_GS
                    self.trial_plasma_psi += blend_GS * self.eq2.plasma_psi

                # prepare for NK algorithms: 1d vectors needed for independent variable
                self.trial_plasma_psi = self.trial_plasma_psi.reshape(-1)
                self.tokamak_psi = self.tokamak_psi.reshape(-1)

                # calculate initial residual for the root dynamic problem in psi
                res_psi = self.F_function_psi(
                    trial_plasma_psi=self.trial_plasma_psi,
                    active_voltage_vec=active_voltage_vec,
                    rtol_NK=self.rtol_NK,
                ).copy()
                del_res_psi = np.amax(res_psi) - np.amin(res_psi)
                relative_psi_res = del_res_psi / self.d_plasma_psi_step
                log.append(["relative_psi_res", relative_psi_res])
                control_NK_psi = (
                    relative_psi_res > target_relative_tol_GS * max_no_NK_psi
                )

                if control_NK_psi:
                    # NK algorithm to solve the root dynamic problem in psi
                    self.psi_nk_solver.Arnoldi_iteration(
                        x0=self.trial_plasma_psi.copy(),
                        dx=res_psi.copy(),
                        R0=res_psi.copy(),
                        F_function=self.F_function_psi,
                        args=args_nk,
                        step_size=step_size_psi,
                        scaling_with_n=scaling_with_n,
                        target_relative_unexplained_residual=target_relative_unexplained_residual,
                        max_n_directions=max_n_directions,
                        clip=clip,
                    )

                    # update trial_plasma_psi according to NK solution
                    self.trial_plasma_psi += self.psi_nk_solver.dx
                    log.append([self.text_psi_1, self.psi_nk_solver.coeffs])

                # prepare for NK solver on the currents, 2d plasma flux needed
                self.trial_plasma_psi = self.trial_plasma_psi.reshape(self.nx, self.ny)

                # calculates initial residual for the root dynamic problem in the currents
                # uses the just updated self.trial_plasma_psi
                res_curr = self.F_function_curr(
                    self.trial_currents, active_voltage_vec
                ).copy()

                # NK algorithm to solve the root problem in the currents
                self.currents_nk_solver.Arnoldi_iteration(
                    x0=self.trial_currents,
                    dx=res_curr.copy(),
                    R0=res_curr,
                    F_function=self.F_function_curr,
                    args=[active_voltage_vec],
                    step_size=step_size_curr,
                    scaling_with_n=scaling_with_n,
                    target_relative_unexplained_residual=target_relative_unexplained_residual,
                    max_n_directions=max_n_directions,
                    clip=clip,
                )
                # update trial_currents according to NK solution
                self.trial_currents += self.currents_nk_solver.dx

                # check convergence properties of the pair [trial_currents, trial_plasma_psi]:
                # relative convergence on the currents:
                res_curr = self.F_function_curr(
                    self.trial_currents, active_voltage_vec
                ).copy()
                rel_curr_res = self.calculate_rel_tolerance_currents(
                    res_curr, curr_eps=curr_eps
                )
                control = np.any(rel_curr_res > target_relative_tol_currents)
                # relative convergence on the GS problem
                r_res_GS = self.calculate_rel_tolerance_GS(self.trial_plasma_psi).copy()
                control_GS = r_res_GS > target_relative_tol_GS
                control += control_GS

                log.append(
                    [
                        "The coeffs applied to the current vec = ",
                        self.currents_nk_solver.coeffs,
                    ]
                )
                log.append(
                    [
                        "The final residual on the current (relative): max =",
                        np.amax(rel_curr_res),
                        "mean =",
                        np.mean(rel_curr_res),
                    ]
                )

                log.append(["Residuals on static GS eq (relative): ", r_res_GS])

                # one full cycle completed
                n_it += 1

            # convergence checks succeeded, complete step
            self.step_complete_assign(working_relative_tol_GS)
