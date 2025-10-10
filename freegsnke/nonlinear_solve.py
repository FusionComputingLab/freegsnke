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

import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from freegs4e import bilinear_interpolation
from freegs4e.gradshafranov import GreensBr, GreensdBrdz
from scipy.signal import convolve2d

from . import nk_solver_H as nk_solver
from .circuit_eq_metal import metal_currents
from .GSstaticsolver import NKGSsolver
from .linear_solve import linear_solver
from .Myy_builder import Myy_handler
from .simplified_solve import simplified_solver_J1


class nl_solver:
    """
    Nonlinear solver for time-evolution of plasma equilibria and circuit dynamics.

    This class provides an interface to evolve both the linearised and nonlinear
    dynamic problems. It sets up plasma and metal circuit equations, handles vessel
    mode decomposition and selection, and enables coupled nonlinear simulation of
    plasma and machine dynamics.

    Main features
    -------------
    - Linear and nonlinear timestepping of equilibrium dynamics
    - Automatic timestep control based on growth rates
    - Passive vessel mode decomposition and mode selection
    - Coupling to FreeGSNKE profiles and equilibria
    - Support for regularization in nonlinear solves
    - Interfaces to Newton–Krylov solvers for plasma flux and circuit equations
    """

    def __init__(
        self,
        profiles,
        eq,
        GSStaticSolver,
        custom_coil_resist=None,
        custom_self_ind=None,
        full_timestep=0.0001,
        max_internal_timestep=0.0001,
        automatic_timestep=False,
        plasma_resistivity=1e-6,
        plasma_norm_factor=1e3,
        blend_hatJ=0,
        max_mode_frequency=10**2.0,
        fix_n_vessel_modes=-1,
        threshold_dIy_dI=0.025,
        min_dIy_dI=0.01,
        mode_removal=True,
        linearize=True,
        dIydI=None,
        dIydtheta=None,
        target_relative_tolerance_linearization=1e-8,
        target_dIy=1e-3,
        force_core_mask_linearization=False,
        l2_reg=1e-6,
        collinearity_reg=1e-6,
        verbose=False,
    ):
        """
        Initialize the nonlinear solver.

        Sets up the equilibrium, profiles, circuit equations, vessel modes,
        Jacobians, and linearization for the coupled plasma–machine system.

        Parameters
        ----------
        profiles : FreeGSNKE profiles
            Initial plasma profiles used to set up the linearisation.
        eq : FreeGSNKE equilibrium
            Equilibrium object providing grid, machine geometry, and limiter info.
        GSStaticSolver : FreeGSNKE static solver
            Static Grad–Shafranov solver for equilibrium solves.
        custom_coil_resist : ndarray, optional
            Resistances for all coils (active + passive). Defaults to tokamak values.
        custom_self_ind : ndarray, optional
            Mutual inductance matrix for coils. Defaults to tokamak values.
        full_timestep : float, default=1e-4
            Time increment for full steps (dt).
        max_internal_timestep : float, default=1e-4
            Maximum sub-step size for advancing circuit equations.
        automatic_timestep : tuple(float, float) or False, default=False
            If set, determines timestep size from growth rates.
        plasma_resistivity : float, default=1e-6
            Plasma resistivity value.
        plasma_norm_factor : float, default=1e3
            Normalization factor for plasma current.
        blend_hatJ : float, default=0
            Coefficient for blending plasma current distributions at t and t+dt.
        max_mode_frequency : float, optional
            Threshold frequency for retaining vessel modes.
        fix_n_vessel_modes : int, default=-1
            Fix number of passive vessel modes; -1 = auto-selection.
        threshold_dIy_dI : float, default=0.025
            Relative coupling threshold for including vessel modes (must be a
            number in [0,1]).
        min_dIy_dI : float, default=0.01
            Minimum coupling threshold for excluding vessel modes (must be a
            number in [0,1]).
        mode_removal : bool, default=True
            If True, remove weakly coupled vessel modes after Jacobian calculation.
        linearize : bool, default=True
            Whether to set up the linearised problem.
        dIydI : ndarray, optional
            Plasma current Jacobian wrt coil and plasma currents.
        dIydtheta : ndarray, optional
            Plasma current Jacobian wrt profile parameters.
        target_relative_tolerance_linearization : float, default=1e-8
            Relative tolerance for GS solve during Jacobian computation.
        target_dIy : float, default=1e-3
            Target perturbation size for Jacobian finite differences.
        force_core_mask_linearization : bool, default=False
            Enforce same core mask during finite-difference Jacobian evaluation.
        l2_reg : float, default=1e-6
            L2 Tikhonov regularization for nonlinear solver.
        collinearity_reg : float, default=1e-6
            Additional penalty for collinear terms in nonlinear solver.
        verbose : bool, default=False
            Print diagnostic output during initialization.
        """
        print("-----")

        # grid parameters
        self.nx = eq.nx
        self.ny = eq.ny
        self.nxny = self.nx * self.ny
        self.eqR = eq.R
        self.eqZ = eq.Z

        # area factor for Iy
        dR = eq.dR
        dZ = eq.dZ
        self.dRdZ = dR * dZ

        # store number of coils and their names/order
        self.n_active_coils = eq.tokamak.n_active_coils
        self.n_coils = eq.tokamak.n_coils
        self.n_passive_coils = eq.tokamak.n_coils - eq.tokamak.n_active_coils
        self.coils_order = list(eq.tokamak.coils_dict.keys())
        self.currents_vec = np.zeros(self.n_coils + 1)

        # setting up reduced domain for plasma circuit eq.:
        self.limiter_handler = eq.limiter_handler
        self.plasma_domain_size = np.sum(self.limiter_handler.mask_inside_limiter)

        # check threshold values
        if fix_n_vessel_modes < 0:
            if min_dIy_dI > threshold_dIy_dI:
                raise ValueError(
                    "Inputs require that 'min_dIy_dI' <= 'threshold_dIy_dI', please adjust parameters."
                )

        # check number of passives
        if self.n_passive_coils < fix_n_vessel_modes:
            print(
                f"'fix_n_vessel_modes' ({fix_n_vessel_modes}) exceeds number of passive strucutres ({self.n_passive_coils}), setting 'fix_n_vessel_modes' to {self.n_passive_coils} "
            )
            fix_n_vessel_modes = self.n_passive_coils

        # check input eq and profiles are a GS solution
        print("Checking that the provided 'eq' and 'profiles' are a GS solution...")

        # storing the static solver
        self.NK = GSStaticSolver
        self.NK.forward_solve(
            eq,
            profiles,
            target_relative_tolerance=target_relative_tolerance_linearization,
            verbose=False,
        )
        print("-----")

        # set internal copy of the equilibrium and profile
        self.eq1 = eq.create_auxiliary_equilibrium()
        self.profiles1 = profiles.copy()
        self.eq2 = eq.create_auxiliary_equilibrium()
        self.profiles2 = profiles.copy()
        self.Iy = self.limiter_handler.Iy_from_jtor(profiles.jtor).copy()
        self.nIy = np.linalg.norm(self.Iy)

        # instantiate the Myy_handler object
        self.handleMyy = Myy_handler(eq.limiter_handler)

        # Extract relevant information on the type of profiles function used and on the actual value of associated parameters
        self.get_profiles_values(profiles)

        self.plasma_norm_factor = plasma_norm_factor
        self.dt_step = full_timestep
        self.max_internal_timestep = max_internal_timestep
        self.set_plasma_resistivity(plasma_resistivity)
        self.target_dIy = target_dIy

        # prepare for mode selection
        if max_mode_frequency is None:
            self.max_mode_frequency = 1 / (5 * full_timestep)
            print(
                "Value of 'max_mode_frequency' has not been provided. Set to",
                self.max_mode_frequency,
                "based on value of 'full_timestep' as provided.",
            )
        else:
            self.max_mode_frequency = max_mode_frequency

        print("Instantiating nonlinear solver objects...")

        # handles the metal circuit eq, mode properties, and performs the vessel mode decomposition
        self.evol_metal_curr = metal_currents(
            eq=eq,
            flag_vessel_eig=1,
            flag_plasma=1,
            plasma_pts=self.limiter_handler.plasma_pts,
            max_mode_frequency=self.max_mode_frequency,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.dt_step,
            coil_resist=custom_coil_resist,
            coil_self_ind=custom_self_ind,
        )
        self.n_metal_modes = self.evol_metal_curr.n_independent_vars

        # prepare the vectorised green functions of the vessel modes
        self.vessel_modes_greens = (
            self.evol_metal_curr.normal_modes.normal_modes_greens(eq._vgreen)
        )
        # build full vector of vessel mode currents
        self.build_current_vec(eq, profiles)

        # prepare initial current shifts for the linearization using a
        # vanilla metric for the coupling between modes and plasma
        mode_coupling_metric = np.linalg.norm(
            self.vessel_modes_greens * profiles.jtor,
            axis=(1, 2),
        )
        mode_coupling_metric /= np.linalg.norm(
            eq.tokamak.getPsitokamak(vgreen=eq._vgreen) * profiles.jtor
        )
        self.mode_coupling_metric = mode_coupling_metric
        self.starting_dI = target_dIy / mode_coupling_metric
        self.final_dI_record = np.zeros_like(self.starting_dI)
        self.approved_target_dIy = target_dIy * np.ones_like(self.starting_dI)
        print("done.")
        print("-----")

        print("Identifying mode selection criteria...")
        if self.n_passive_coils > 0:
            # prepare ndIydI_no_GS for mode selection
            self.build_dIydI_noGS(
                force_core_mask_linearization,
                self.starting_dI,
                profiles.diverted_core_mask,
                verbose,
            )

            # select modes according to the provided thresholds:
            # include all modes that couple more than the threshold_dIy_dI
            # with respect to the strongest coupling vessel mode
            ordered_ndIydI_no_GS = np.sort(self.ndIydI_no_GS[self.n_active_coils :])
            strongest_coupling_vessel_mode = ordered_ndIydI_no_GS[-1]

            if fix_n_vessel_modes >= 0:
                # select modes based on ndIydI_no_GS up to fix_n_modes exactly
                print(
                    f"      'fix_n_vessel_modes' option selected --> passive structure modes that couple most to the strongest passive structure mode are being selected."
                )

                if fix_n_vessel_modes > 0:
                    threshold_value = ordered_ndIydI_no_GS[-fix_n_vessel_modes]
                else:  # zero modes to be selected
                    threshold_value = (
                        strongest_coupling_vessel_mode * 1.1
                    )  # scale up so no modes are selected

                mode_coupling_mask_include = np.concatenate(
                    (
                        [True] * self.n_active_coils,
                        self.ndIydI_no_GS[self.n_active_coils :] >= threshold_value,
                    )
                )
                mode_coupling_mask_exclude = np.concatenate(
                    (
                        [True] * self.n_active_coils,
                        self.ndIydI_no_GS[self.n_active_coils :] >= threshold_value,
                    )
                )
                # the number of modes is being fixed:
                mode_removal = False
            else:
                print(
                    f"      'threshold_dIy_dI', 'min_dIy_dI', and 'max_mode_frequency' options selected --> passive structure modes are selected according to these thresholds."
                )
                # select modes based on ndIydI_no_GS using values of threshold_dIy_dI
                mode_coupling_mask_include = np.concatenate(
                    (
                        [True] * self.n_active_coils,
                        self.ndIydI_no_GS[self.n_active_coils :]
                        >= threshold_dIy_dI * strongest_coupling_vessel_mode,
                    )
                )
                # exclude all modes that couple less than min_dIy_dI
                mode_coupling_mask_exclude = np.concatenate(
                    (
                        [True] * self.n_active_coils,
                        self.ndIydI_no_GS[self.n_active_coils :]
                        >= min_dIy_dI * strongest_coupling_vessel_mode,
                    )
                )
        else:
            print("      no passive modes present!")

            # only active coils selected
            mode_coupling_mask_include = [True] * self.n_active_coils

            # exclude all modes that couple less than min_dIy_dI
            mode_coupling_mask_exclude = mode_coupling_mask_include

            # set mode removal to false
            mode_removal = False

        print("-----")

        print(f"Initial mode selection:")
        # enact the mode selection
        mode_coupling_masks = (
            mode_coupling_mask_include,
            mode_coupling_mask_exclude,
        )

        self.evol_metal_curr.initialize_for_eig(
            selected_modes_mask=None,
            mode_coupling_masks=mode_coupling_masks,
            verbose=(fix_n_vessel_modes < 0),
        )

        if fix_n_vessel_modes >= 0:
            print(f"   Active coils")
            print(
                f"      total selected = {self.n_active_coils} (out of {self.n_active_coils})"
            )
            print(f"   Passive structures")
            print(f"      {fix_n_vessel_modes} selected using 'fix_n_vessel_modes'")
            print(
                f"   Total number of modes = {self.evol_metal_curr.n_independent_vars} ({self.n_active_coils} active coils + {fix_n_vessel_modes} passive structures)"
            )
            print(
                f"      (Note: some additional modes may be removed after Jacobian calculation if 'mode_removal=True')"
            )
        print("-----")

        # this is the number of independent normal mode currents being used
        self.n_metal_modes = self.evol_metal_curr.n_independent_vars
        self.arange_currents = np.arange(self.n_metal_modes + 1)
        # re-build vector of vessel mode currents after mode selection
        self.build_current_vec(eq, profiles)

        # select modes accordingly
        self.starting_dI = self.starting_dI[self.evol_metal_curr.selected_modes_mask]
        self.approved_target_dIy = self.approved_target_dIy[
            self.evol_metal_curr.selected_modes_mask
        ]
        # add the plasma
        self.starting_dI = np.concatenate(
            (self.starting_dI, [target_dIy * profiles.Ip / plasma_norm_factor])
        )
        self.approved_target_dIy = np.concatenate(
            (self.approved_target_dIy, [target_dIy])
        )

        # starting dtheta values for Jacobian calculation
        self.approved_target_dtheta = target_dIy * np.ones(self.n_profiles_parameters)
        self.starting_dtheta = np.zeros(self.n_profiles_parameters)
        self.final_dtheta_record = np.zeros(self.n_profiles_parameters)

        # for setting profile parameters starting shifts (for Jacobians)
        if self.profiles_param is not None:
            # alpha_m
            self.starting_dtheta[0] = max(profiles.alpha_m * 1e-2, 1e-2)
            # alpha_n
            self.starting_dtheta[1] = max(profiles.alpha_n * 1e-2, 1e-2)
            # paxis/betap/Beta0
            if self.profiles_param == "paxis":
                self.starting_dtheta[2] = max(
                    getattr(profiles, self.profiles_param) * 1e-2, 1e1
                )
            elif self.profiles_param == "betap":
                self.starting_dtheta[2] = max(
                    getattr(profiles, self.profiles_param) * 1e-2, 1e-4
                )
            elif self.profiles_param == "Beta0":
                self.starting_dtheta[2] = max(
                    getattr(profiles, self.profiles_param) * 1e-2, 1e-4
                )

        else:  # lao
            # alpha coeffs
            self.starting_dtheta[0 : self.n_profiles_parameters_alpha] = (
                self.profiles_parameters_vec[self.profiles_alpha_indices] * 1e-3
            )
            self.starting_dtheta[0 : self.n_profiles_parameters_alpha][
                self.starting_dtheta[0 : self.n_profiles_parameters_alpha] == 0
            ] = 1e-3

            # beta coeffs
            self.starting_dtheta[self.n_profiles_parameters_alpha :] = (
                self.profiles_parameters_vec[self.profiles_beta_indices] * 1e-3
            )
            self.starting_dtheta[self.n_profiles_parameters_alpha :][
                self.starting_dtheta[self.n_profiles_parameters_alpha :] == 0
            ] = 1e-3

        # This solves the system of circuit eqs based on an assumption
        # for the direction of the plasma current distribution at time t+dt
        self.simplified_solver_J1 = simplified_solver_J1(
            coil_numbers=(self.n_active_coils, self.n_coils),
            Lambdam1=self.evol_metal_curr.Lambdam1,
            P=self.evol_metal_curr.P,
            Pm1=self.evol_metal_curr.Pm1,
            Rm1=np.diag(self.evol_metal_curr.Rm1),
            Mey=self.evol_metal_curr.Mey_matrix,
            # limiter_handler=self.limiter_handler,
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
            coil_numbers=(self.n_active_coils, self.n_coils),
            Lambdam1=self.evol_metal_curr.Lambdam1,
            P=self.evol_metal_curr.P,
            Pm1=self.evol_metal_curr.Pm1,
            Rm1=np.diag(self.evol_metal_curr.Rm1),
            Mey=self.evol_metal_curr.Mey_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.dt_step,
        )

        # sets up NK solver on the full grid, to be used when solving the
        # circuits equations as a problem in the plasma flux
        self.psi_nk_solver = nk_solver.nksolver(
            self.nxny, l2_reg=l2_reg, collinearity_reg=collinearity_reg
        )

        # sets up NK solver for the currents
        self.currents_nk_solver = nk_solver.nksolver(
            self.extensive_currents_dim,
            l2_reg=l2_reg,
            collinearity_reg=collinearity_reg,
        )

        # counter for the step advancement of the dynamics
        self.step_no = 0

        # set default blend for contracting the plasma lumped eq
        self.make_blended_hatIy = lambda x: self.make_blended_hatIy_(
            x, blend=blend_hatJ
        )

        # self.dIydI is the Jacobian of the plasma current distribution
        # with respect to the independent currents (as in self.currents_vec)
        self.dIydI_ICs = dIydI
        self.dIydI = dIydI

        # self.dIydtheta is the Jacobian of the plasma current distribution
        # with respect to the plasma current density profile parameters
        self.dIydtheta_ICs = dIydtheta
        self.dIydtheta = dIydtheta

        # initialize and set up the linearization
        # input value for dIydI is used when available
        if automatic_timestep == False:
            automatic_timestep_flag = False
        else:
            if len(automatic_timestep) != 2:
                raise ValueError(
                    "The input for 'automatic_timestep' should be of the form (float, float). Please revise."
                )
            automatic_timestep_flag = True

        if automatic_timestep_flag + mode_removal + linearize:
            # builds the linearization and sets everything up for the stepper
            self.initialize_from_ICs(
                eq,
                profiles,
                target_relative_tolerance_linearization=target_relative_tolerance_linearization,
                dIydI=dIydI,
                dIydtheta=dIydtheta,
                verbose=verbose,
                force_core_mask_linearization=force_core_mask_linearization,
            )
            print("-----")

        # remove passive normal modes that have norm(dIydI) < min_dIy_dI*strongest mode
        if mode_removal:
            # selected based on full calculation of the coupling
            ndIydI = np.linalg.norm(self.dIydI, axis=0)
            selected_modes_mask = ndIydI > min_dIy_dI * max(
                ndIydI[self.n_active_coils : -1]
            )
            # force that active coils and plasma are kept
            actives_and_plasma_mask = (
                [True] * self.n_active_coils
                + [False] * (self.n_metal_modes - self.n_active_coils)
                + [True]
            )
            self.retained_modes_mask = (
                selected_modes_mask + np.array(actives_and_plasma_mask)
            ).astype(bool)

            # apply mask to dIydI, dRZdI and final_dI_record
            self.dIydI = self.dIydI[:, self.retained_modes_mask]
            self.dIydI_ICs = np.copy(self.dIydI)
            self.dRZdI = self.dRZdI[:, self.retained_modes_mask]
            self.final_dI_record = self.final_dI_record[self.retained_modes_mask]

            self.remove_modes(eq, self.retained_modes_mask[:-1])

            print(
                f"   Re-sizing the Jacobian matrix to account for any removed modes (if required)."
            )
            print("-----")

        # check if input equilibrium and associated linearization have an instability, and its timescale
        if automatic_timestep_flag + mode_removal + linearize:
            print("Stability paramters:")
            self.linearised_sol.calculate_linear_growth_rate()
            self.linearised_sol.calculate_stability_margin()
            self.calculate_Leuer_parameter()

            if len(self.linearised_sol.growth_rates):
                self.unstable_mode_deformations()
                # deformable plasma metrics
                print(f"   Deformable plasma metrics:")
                print(f"      Growth rate = {self.linearised_sol.growth_rates} [1/s]")
                print(
                    f"      Instability timescale = {self.linearised_sol.instability_timescale} [s]"
                )
                print(
                    f"      Inductive stability margin = {self.linearised_sol.stability_margin}"
                )

                # rigid plasma metrics
                print(f"   Rigid plasma metrics:")
                print(
                    f"      Leuer parameter (ratio of stabilsing to de-stabilising force gradients):"
                )
                print(
                    f"          between all metals and all metals = {self.Leuer_metals_stab_over_metals_destab}"
                )
                print(
                    f"          between all metals and active metals = {self.Leuer_metals_stab_over_active_destab}"
                )
                print(
                    f"          between passive metals and active metals = {self.Leuer_passive_stab_over_active_destab}"
                )

            else:
                print(
                    f"      No unstable modes found: either plasma stable, or more likely, it is Alfven unstable (i.e. needs more stabilisation from coils and passives)."
                )
                if fix_n_vessel_modes >= 0:
                    print(
                        f"      Try adding more passive modes (by increasing 'fix_n_vessel_modes')."
                    )
                else:
                    print(
                        f"      Try adding more passive modes (by increasing 'max_mode_frequency' and/or 'threshold_dIy_dI' and/or reducing 'min_dIy_dI'."
                    )
        print("-----")

        # if automatic_timestep, reset the timestep accordingly,
        # note that this requires having found an instability
        print("Evolutive solver timestep:")
        if automatic_timestep_flag is False:
            print(
                f"      Solver timestep 'dt_step' has been set to {self.dt_step} as requested."
            )
            print(
                f"      Ensure it is smaller than the growth rate else you may find numerical instability in any subsequent evoltuive simulations!"
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
                    f"      Solver timestep 'dt_step' has been reset to {self.dt_step} using the growth rate and scaling factors in 'automatic_timestep'."
                )
            else:
                print(
                    f"      Given no unstable modes found, it is impossible to automatically set the timestep! Please do so manually."
                )

        print("-----")

        # text for verbose mode
        self.text_nk_cycle = "This is NK cycle no {nkcycle}."
        self.text_psi_0 = "NK on psi has been skipped {skippedno} times. The residual on psi is {psi_res:.8f}."
        self.text_psi_1 = "The coefficients applied to psi are"

    def build_dIydI_noGS(
        self, force_core_mask_linearization, starting_dI, core_mask, verbose
    ):
        """
        Compute a first estimate of the Jacobian norm dIy/dI without solving GS.

        This routine evaluates the plasma current response to perturbations in each
        coil or mode current using only the modified tokamak Green’s functions
        (no Grad–Shafranov solves). The resulting Jacobian norm is used for an
        initial sifting of passive vessel modes before a full linearisation.

        If `force_core_mask_linearization` is True, the perturbation size for each
        mode is adjusted to ensure that the diverted core mask of the perturbed
        equilibrium matches the reference core mask. In that case,
        `self.starting_dI` and `self.approved_target_dIy` are updated accordingly.

        Parameters
        ----------
        force_core_mask_linearization : bool
            Whether to enforce identical core masks between perturbed and reference
            equilibria when computing finite-difference derivatives. If True,
            perturbation amplitudes are reduced until the core mask matches.
        starting_dI : ndarray
            Initial perturbation amplitudes for each independent coil/mode current.
            This array is updated in place depending on the masking strategy.
        core_mask : ndarray of bool
            Boolean mask indicating the core region of the reference equilibrium.
        verbose : bool
            If True, print diagnostic information about mode perturbations and
            accepted perturbation sizes.

        Updates
        -------
        self.dIydI_noGS : ndarray, shape (n_plasma_points, n_coils)
            Approximate Jacobian of plasma current distribution wrt coil currents,
            computed without GS solves.
        self.ndIydI_no_GS : ndarray, shape (n_coils,)
            Norm of dIy/dI for each coil/mode, used in mode selection.
        self.rel_ndIy : ndarray, shape (n_coils,)
            Relative plasma current perturbation norms for each mode.
        self.starting_dI : ndarray
            Adjusted perturbation amplitudes after enforcing masking strategy.
        self.approved_target_dIy : ndarray
            Updated target norms for plasma current perturbations (if core mask
            enforcement applied).
        """

        self.dIydI_noGS = np.zeros((len(self.Iy), self.n_coils))
        self.ndIydI_no_GS = np.zeros(self.n_coils)
        self.rel_ndIy = np.zeros(self.n_coils)

        for j in range(self.n_coils):
            dIydInoGS, rel_ndIy = self.prepare_build_dIydI_j(
                j, None, self.approved_target_dIy[j], starting_dI[j], GS=False
            )
            core_check = (
                np.sum(
                    np.abs(
                        core_mask.astype(float)
                        - self.profiles2.diverted_core_mask.astype(float)
                    )
                )
                == 0
            )
            if force_core_mask_linearization:
                while core_check == False:
                    starting_dI[j] /= 1.5
                    dIydInoGS, rel_ndIy = self.prepare_build_dIydI_j(
                        j, None, self.approved_target_dIy[j], starting_dI[j], GS=False
                    )
                    core_check = (
                        np.sum(
                            np.abs(
                                core_mask.astype(float)
                                - self.profiles2.diverted_core_mask.astype(float)
                            )
                        )
                        == 0
                    )
                self.approved_target_dIy[j] = rel_ndIy

            else:
                starting_dI[j] = 1.0 * self.final_dI_record[j]

            self.dIydI_noGS[:, j] = dIydInoGS
            self.rel_ndIy[j] = rel_ndIy
            # self.final_dI_record[j] = starting_dI[j] * self.accepted_target_dIy[j] / rel_ndIy
            self.ndIydI_no_GS[j] = rel_ndIy * self.nIy / starting_dI[j]
        self.starting_dI = 1.0 * starting_dI

    def set_solvers(
        self,
    ):
        """
        Initialize and configure the time-integration solvers.

        Creates solver instances for both the simplified nonlinear system
        (`simplified_solver_J1`) and the linearised system (`linear_solver`).
        Both solvers are constructed using machine inductance/resistance matrices
        and plasma parameters, and are configured to advance currents consistently
        with the timestep settings.

        After creation, the linearised solver is set to operate around the current
        linearisation point, using the Jacobians and reference plasma state.

        Updates
        -------
        self.simplified_solver_J1 : simplified_solver_J1
            Nonlinear solver instance for reduced-order plasma–circuit dynamics.
        self.linearised_sol : linear_solver
            Linear solver instance for coupled plasma–circuit dynamics.
        """

        self.simplified_solver_J1 = simplified_solver_J1(
            coil_numbers=(self.n_active_coils, self.n_coils),
            Lambdam1=self.evol_metal_curr.Lambdam1,
            P=self.evol_metal_curr.P,
            Pm1=self.evol_metal_curr.Pm1,
            Rm1=np.diag(self.evol_metal_curr.Rm1),
            Mey=self.evol_metal_curr.Mey_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            full_timestep=self.dt_step,
        )

        self.linearised_sol = linear_solver(
            coil_numbers=(self.n_active_coils, self.n_coils),
            Lambdam1=self.evol_metal_curr.Lambdam1,
            P=self.evol_metal_curr.P,
            Pm1=self.evol_metal_curr.Pm1,
            Rm1=np.diag(self.evol_metal_curr.Rm1),
            Mey=self.evol_metal_curr.Mey_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.dt_step,
        )

        self.linearised_sol.set_linearization_point(
            dIydI=self.dIydI,
            dIydtheta=self.dIydtheta,
            hatIy0=self.blended_hatIy,
            Myy_hatIy0=self.Myy_hatIy0,
        )

    def remove_modes(self, eq, selected_modes_mask):
        """
        Remove unselected normal modes and update circuit equations.

        Given an equilibrium and a mask over the current mode set, this method
        reduces the dimensionality of the system by discarding modes marked
        as inactive in `selected_modes_mask`. The circuit equations, current
        vector, and nonlinear solver are reinitialized consistently with the
        reduced system size.

        Parameters
        ----------
        eq : FreeGSNKE equilibrium
            Equilibrium object containing plasma state information.
        selected_modes_mask : ndarray of bool, shape (n_modes,)
            Boolean mask indicating which modes to keep. Entries corresponding
            to `True` are retained, and those corresponding to `False` are removed.
            Must have the same shape as `self.currents_vec` at the time of call.

        Updates
        -------
        self.n_metal_modes : int
            Number of remaining independent modes after reduction.
        self.extensive_currents_dim : int
            Dimension of the reduced current vector (n_metal_modes + 1).
        self.arange_currents : ndarray
            Index array for the reduced system, ranging from 0 to
            `self.extensive_currents_dim - 1`.
        self.currents_vec : ndarray
            Zero-initialized current vector of reduced dimensionality.
        self.circuit_eq_residual : ndarray
            Residual vector for the reduced circuit equations.
        self.currents_nk_solver : nk_solver.nksolver
            Nonlinear solver instance reinitialized for the reduced system size.
        self.simplified_solver_J1 : simplified_solver_J1
            Rebuilt solver instance consistent with reduced system.
        self.linearised_sol : linear_solver
            Rebuilt linearised solver instance consistent with reduced system.
        """

        self.evol_metal_curr.initialize_for_eig(selected_modes_mask)
        self.n_metal_modes = self.evol_metal_curr.n_independent_vars
        self.extensive_currents_dim = self.n_metal_modes + 1
        self.arange_currents = np.arange(self.n_metal_modes + 1)
        self.currents_vec = np.zeros(self.extensive_currents_dim)
        self.circuit_eq_residual = np.zeros(self.extensive_currents_dim)
        self.currents_nk_solver = nk_solver.nksolver(self.extensive_currents_dim)

        self.build_current_vec(self.eq1, self.profiles1)

        self.set_solvers()

    def set_linear_solution(
        self,
        active_voltage_vec,
        dtheta_dt,
    ):
        """
        Compute an initial nonlinear solve guess using the linearised dynamics.

        Advances the system one timestep using the linearised solver to generate
        a trial current vector at t + Δt, starting from `self.currents_vec` at time t.
        This trial solution is then used as the initial guess for the nonlinear solver,
        which updates the Grad–Shafranov (GS) equilibrium at t + Δt.

        Parameters
        ----------
        active_voltage_vec : ndarray
            External voltage applied to the active coils during the timestep.
        dtheta_dt : ndarray
            Time derivatives of the plasma current density profile parameters.

        Updates
        -------
        self.trial_currents : ndarray
            Trial coil/mode currents at t + Δt obtained from the linearised solver.
        self.eq2.plasma_psi : ndarray
            Plasma poloidal flux after solving GS with the trial currents.
        self.trial_plasma_psi : ndarray
            Copy of the GS solution for the plasma flux surface configuration
            corresponding to the trial currents.
        """

        self.trial_currents = self.linearised_sol.stepper(
            It=self.currents_vec,
            active_voltage_vec=active_voltage_vec,
            dtheta_dt=dtheta_dt,
        )
        self.assign_currents_solve_GS(self.trial_currents, self.rtol_NK)
        self.trial_plasma_psi = np.copy(self.eq2.plasma_psi)

    def prepare_build_dIydtheta(
        self,
        profiles,
        rtol_NK,
        target_dIy,
        starting_dtheta,
        verbose=False,
    ):
        """
        Prepare finite-difference evaluation of d(Iy)/dθ,
        where θ parameterises the plasma current density profile.

        Perturbs the profile parameters by small trial shifts `starting_dtheta`,
        solves the Grad–Shafranov equilibrium for each perturbed profile, and measures
        the corresponding change in Iy (poloidal current distribution).
        The trial perturbations are then rescaled so that the induced change in Iy
        has prescribed norm `target_dIy`. This sets up consistent perturbation sizes
        for later derivative calculations.

        Parameters
        ----------
        profiles : FreeGS4E profiles object
            Profiles object of the linearisation-point equilibrium.
        rtol_NK : float
            Relative tolerance for the Newton–Krylov GS solves.
        target_dIy : ndarray
            Desired norms of ΔIy for each perturbed direction.
        starting_dtheta : ndarray
            Initial perturbations of the profile parameters, used to infer the
            scaling between Δθ and ΔIy.
        verbose : bool, optional
            If True, print intermediate diagnostic information (default: False).

        Returns
        -------
        dIy_0 : ndarray, shape (len(Iy), n_profiles_parameters)
            Raw changes in Iy from the initial perturbations.
        rel_ndIy_0 : ndarray, shape (n_profiles_parameters,)
            Relative norms of ΔIy per unit perturbation, normalised by `self.nIy`.

        Updates
        -------
        self.final_dtheta_record : ndarray
            Adjusted perturbations Δθ that will yield ΔIy with the target norms.
        """

        current_ = np.copy(self.currents_vec)

        # storage
        dIy_0 = np.zeros((len(self.Iy), self.n_profiles_parameters))
        rel_ndIy_0 = np.zeros(self.n_profiles_parameters)

        # carry out the initial perturbations
        if self.profiles_param is not None:

            # vary alpha_m
            self.check_and_change_profiles(
                profiles_parameters={
                    "alpha_m": profiles.alpha_m + starting_dtheta[0],
                    "alpha_n": profiles.alpha_n,
                    self.profiles_param: getattr(profiles, self.profiles_param),
                }
            )

            # reset plasma flux map to original
            self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
            self.assign_currents_solve_GS(current_, rtol_NK)
            dIy_0[:, 0] = (
                self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
            )
            rel_ndIy_0[0] = np.linalg.norm(dIy_0[:, 0]) / self.nIy
            self.final_dtheta_record[0] = (
                starting_dtheta[0] * target_dIy[0] / rel_ndIy_0[0]
            )

            if verbose:
                print("")
                print("Profile parameter: alpha_m:")
                print(f"  Initial delta parameter = {starting_dtheta[0]}")
                print(f"  Initial relative Iy change = {rel_ndIy_0[0]}")
                print(f"  Final delta parameter = {self.final_dtheta_record[0]}")

            # vary alpha_n
            self.check_and_change_profiles(
                profiles_parameters={
                    "alpha_m": profiles.alpha_m,
                    "alpha_n": profiles.alpha_n + starting_dtheta[1],
                    self.profiles_param: getattr(profiles, self.profiles_param),
                }
            )

            # reset plasma flux map to original
            self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
            self.assign_currents_solve_GS(current_, rtol_NK)
            dIy_0[:, 1] = (
                self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
            )
            rel_ndIy_0[1] = np.linalg.norm(dIy_0[:, 1]) / self.nIy
            self.final_dtheta_record[1] = (
                starting_dtheta[1] * target_dIy[1] / rel_ndIy_0[1]
            )

            if verbose:
                print("")
                print("Profile parameter: alpha_n:")
                print(f"  Initial delta parameter = {starting_dtheta[1]}")
                print(f"  Initial relative Iy change = {rel_ndIy_0[1]}")
                print(f"  Final delta parameter = {self.final_dtheta_record[1]}")

            # vary paxis, betap or Beta0
            self.check_and_change_profiles(
                profiles_parameters={
                    "alpha_m": profiles.alpha_m,
                    "alpha_n": profiles.alpha_n,
                    self.profiles_param: getattr(profiles, self.profiles_param)
                    + starting_dtheta[2],
                }
            )

            # reset plasma flux map to original
            self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
            self.assign_currents_solve_GS(current_, rtol_NK)
            dIy_0[:, 2] = (
                self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
            )
            rel_ndIy_0[2] = np.linalg.norm(dIy_0[:, 2]) / self.nIy
            self.final_dtheta_record[2] = (
                starting_dtheta[2] * target_dIy[2] / rel_ndIy_0[2]
            )

            if verbose:
                print("")
                print(f"Profile parameter: {self.profiles_param}:")
                print(f"  Initial delta parameter = {starting_dtheta[2]}")
                print(f"  Initial relative Iy change = {rel_ndIy_0[2]}")
                print(f"  Final delta parameter = {self.final_dtheta_record[2]}")

            # reset profiles in profiles1 and profiles2 objects
            self.check_and_change_profiles(
                profiles_parameters={
                    "alpha_m": profiles.alpha_m,
                    "alpha_n": profiles.alpha_n,
                    self.profiles_param: getattr(profiles, self.profiles_param),
                }
            )

        else:  # this is particular to the Lao profile coefficients (which there may be few or many of)

            # for each alpha coefficient
            alpha_base = profiles.alpha.copy()
            for i in range(0, self.n_profiles_parameters_alpha):
                alpha_shift = alpha_base.copy()
                alpha_shift[i] += starting_dtheta[i]  # perturb the term
                if profiles.alpha_logic:
                    alpha_shift[-1] -= starting_dtheta[
                        i
                    ]  # final alpha term needs perturbing too

                self.check_and_change_profiles(
                    profiles_parameters={
                        "alpha": alpha_shift,
                        "beta": profiles.beta,
                    }
                )

                # reset plasma flux map to original
                self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
                self.assign_currents_solve_GS(current_, rtol_NK)
                dIy_0[:, i] = (
                    self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
                )
                rel_ndIy_0[i] = np.linalg.norm(dIy_0[:, i]) / self.nIy
                self.final_dtheta_record[i] = (
                    starting_dtheta[i] * target_dIy[i] / rel_ndIy_0[i]
                )

                if verbose:
                    print("")
                    print(f"Profile parameter: alpha_{i}:")
                    print(f"  Initial delta parameter = {starting_dtheta[i]}")
                    print(f"  Initial relative Iy change = {rel_ndIy_0[i]}")
                    print(f"  Final delta parameter = {self.final_dtheta_record[i]}")

            # for each beta coefficient
            beta_base = profiles.beta.copy()
            for i in range(0, self.n_profiles_parameters_beta):
                beta_shift = beta_base.copy()
                beta_shift[i] += starting_dtheta[
                    i + self.n_profiles_parameters_alpha
                ]  # perturb the term required
                if profiles.beta_logic:
                    beta_shift[-1] -= starting_dtheta[
                        i + self.n_profiles_parameters_alpha
                    ]  # final beta term needs perturbing too
                self.check_and_change_profiles(
                    profiles_parameters={
                        "alpha": profiles.alpha,
                        "beta": beta_shift,
                    }
                )

                # reset plasma flux map to original
                self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
                self.assign_currents_solve_GS(current_, rtol_NK)
                dIy_0[:, i + self.n_profiles_parameters_alpha] = (
                    self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
                )
                rel_ndIy_0[i + self.n_profiles_parameters_alpha] = (
                    np.linalg.norm(dIy_0[:, i + self.n_profiles_parameters_alpha])
                    / self.nIy
                )
                self.final_dtheta_record[i + self.n_profiles_parameters_alpha] = (
                    starting_dtheta[i + self.n_profiles_parameters_alpha]
                    * target_dIy[i]
                    / rel_ndIy_0[i + self.n_profiles_parameters_alpha]
                )

                if verbose:
                    print("")
                    print(f"Profile parameter: beta_{i}:")
                    print(
                        f"  Initial delta parameter = {starting_dtheta[i+self.n_profiles_parameters_alpha]}"
                    )
                    print(
                        f"  Initial relative Iy change = {rel_ndIy_0[i+self.n_profiles_parameters_alpha]}"
                    )
                    print(
                        f"  Final delta parameter = {self.final_dtheta_record[i+self.n_profiles_parameters_alpha]}"
                    )

                # reset profiles in profiles1 and profiles2 objects
                self.check_and_change_profiles(
                    profiles_parameters={
                        "alpha": profiles.alpha,
                        "beta": profiles.beta,
                    }
                )

        return dIy_0 / starting_dtheta, rel_ndIy_0

    def build_dIydtheta(self, profiles, rtol_NK, verbose=False):
        """
        Compute the finite-difference Jacobian d(Iy)/dθ using pre-scaled perturbations.

        This function calculates the derivative of the poloidal current vector Iy with
        respect to the plasma profile parameters θ. Perturbations `Δθ` are taken from
        `self.final_dtheta_record`, which are inferred previously by
        `prepare_build_dIydtheta` to produce controlled changes in Iy. For each perturbed
        profile, the Grad–Shafranov equilibrium is solved and the resulting Iy change
        is measured. The final Jacobian is the normalized change per unit Δθ.

        Parameters
        ----------
        profiles : FreeGS4E profiles object
            Profiles object of the linearization-point equilibrium.
        rtol_NK : float
            Relative tolerance for the Newton–Krylov Grad–Shafranov solver.
        verbose : bool, optional
            If True, print intermediate diagnostic information (default: False).

        Returns
        -------
        dIydtheta : ndarray, shape (len(Iy), n_profiles_parameters)
            Finite-difference derivative of Iy with respect to each profile parameter.
        rel_ndIy : ndarray, shape (n_profiles_parameters,)
            Relative norm of ΔIy induced by each perturbation, normalized by self.nIy.

        Notes
        -----
        - This function modifies the plasma profiles temporarily during the finite-difference
          calculation, then resets them to the original state.
        - Supports both the standard (alpha_m, alpha_n, paxis/betap/Beta0) parameters
          and Lao-style profile coefficients (multiple alpha and beta terms).
        """

        # the final perturbations to calc the Jacobian with
        final_theta = 1.0 * self.final_dtheta_record

        current_ = np.copy(self.currents_vec)

        # storage
        dIydtheta = np.zeros((len(self.Iy), self.n_profiles_parameters))
        rel_ndIy = np.zeros(self.n_profiles_parameters)

        # carry out the initial perturbations
        if self.profiles_param is not None:

            # vary alpha_m
            self.check_and_change_profiles(
                profiles_parameters={
                    "alpha_m": profiles.alpha_m + final_theta[0],
                    "alpha_n": profiles.alpha_n,
                    self.profiles_param: getattr(profiles, self.profiles_param),
                }
            )
            # reset plasma flux map to original
            self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
            self.assign_currents_solve_GS(current_, rtol_NK)
            dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
            rel_ndIy[0] = np.linalg.norm(dIy_1) / self.nIy
            dIydtheta[:, 0] = dIy_1 / final_theta[0]
            if verbose:
                print("")
                print(f"Profile parameter: alpha_m:")
                print(f"  Final relative Iy change = {rel_ndIy[0]}")
                print(
                    f"  Initial vs. Final GS residual: {self.NK.initial_rel_residual} vs. {self.NK.relative_change}"
                )

            # vary alpha_n
            self.check_and_change_profiles(
                profiles_parameters={
                    "alpha_m": profiles.alpha_m,
                    "alpha_n": profiles.alpha_n + final_theta[1],
                    self.profiles_param: getattr(profiles, self.profiles_param),
                }
            )
            # reset plasma flux map to original
            self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
            self.assign_currents_solve_GS(current_, rtol_NK)
            dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
            rel_ndIy[1] = np.linalg.norm(dIy_1) / self.nIy
            dIydtheta[:, 1] = dIy_1 / final_theta[1]
            if verbose:
                print("")
                print(f"Profile parameter: alpha_n:")
                print(f"  Final relative Iy change = {rel_ndIy[1]}")
                print(
                    f"  Initial vs. Final GS residual: {self.NK.initial_rel_residual} vs. {self.NK.relative_change}"
                )

            # vary paxis, betap or Beta0
            self.check_and_change_profiles(
                profiles_parameters={
                    "alpha_m": profiles.alpha_m,
                    "alpha_n": profiles.alpha_n,
                    self.profiles_param: getattr(profiles, self.profiles_param)
                    + final_theta[2],
                }
            )
            # reset plasma flux map to original
            self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
            self.assign_currents_solve_GS(current_, rtol_NK)
            dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
            rel_ndIy[2] = np.linalg.norm(dIy_1) / self.nIy
            dIydtheta[:, 2] = dIy_1 / final_theta[2]
            if verbose:
                print("")
                print(f"Profile parameter: {self.profiles_param}:")
                print(f"  Final relative Iy change = {rel_ndIy[2]}")
                print(
                    f"  Initial vs. Final GS residual: {self.NK.initial_rel_residual} vs. {self.NK.relative_change}"
                )

            # reset profiles in profiles1 and profiles2 objects
            self.check_and_change_profiles(
                profiles_parameters={
                    "alpha_m": profiles.alpha_m,
                    "alpha_n": profiles.alpha_n,
                    self.profiles_param: getattr(profiles, self.profiles_param),
                }
            )

        else:  # this is particular to the Lao profile coefficients (which there may be few or many of)

            # for each alpha coefficient

            alpha_base = profiles.alpha.copy()
            for i in range(0, self.n_profiles_parameters_alpha):
                alpha_shift = alpha_base.copy()
                alpha_shift[i] += final_theta[i]  # perturb the term
                if profiles.alpha_logic:
                    alpha_shift[-1] -= final_theta[
                        i
                    ]  # final alpha term needs perturbing too

                self.check_and_change_profiles(
                    profiles_parameters={
                        "alpha": alpha_shift,
                        "beta": profiles.beta,
                    }
                )
                # reset plasma flux map to original
                self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
                self.assign_currents_solve_GS(current_, rtol_NK)
                dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
                rel_ndIy[i] = np.linalg.norm(dIy_1) / self.nIy
                dIydtheta[:, i] = dIy_1 / final_theta[i]
                if verbose:
                    print("")
                    print(f"Profile parameter: alpha_{i}:")
                    print(f"  Final relative Iy change = {rel_ndIy[i]}")
                    print(
                        f"  Initial vs. Final GS residual: {self.NK.initial_rel_residual} vs. {self.NK.relative_change}"
                    )

            # for each beta coefficient
            beta_base = profiles.beta.copy()
            for i in range(0, self.n_profiles_parameters_beta):
                beta_shift = beta_base.copy()
                beta_shift[i] += final_theta[
                    i + self.n_profiles_parameters_alpha
                ]  # perturb the term required
                if profiles.beta_logic:
                    beta_shift[-1] -= final_theta[
                        i + self.n_profiles_parameters_alpha
                    ]  # final beta term needs perturbing too
                self.check_and_change_profiles(
                    profiles_parameters={
                        "alpha": profiles.alpha,
                        "beta": beta_shift,
                    }
                )
                # reset plasma flux map to original
                self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
                self.assign_currents_solve_GS(current_, rtol_NK)
                dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
                rel_ndIy[i + self.n_profiles_parameters_alpha] = (
                    np.linalg.norm(dIy_1) / self.nIy
                )
                dIydtheta[:, i + self.n_profiles_parameters_alpha] = (
                    dIy_1 / final_theta[i + self.n_profiles_parameters_alpha]
                )

                if verbose:
                    print("")
                    print(f"Profile parameter: beta_{i}:")
                    print(
                        f"  Final relative Iy change = {rel_ndIy[i+self.n_profiles_parameters_alpha]}"
                    )
                    print(
                        f"  Initial vs. Final GS residual: {self.NK.initial_rel_residual} vs. {self.NK.relative_change}"
                    )

            # reset profiles in profiles1 and profiles2 objects
            self.check_and_change_profiles(
                profiles_parameters={
                    "alpha": profiles.alpha,
                    "beta": profiles.beta,
                }
            )

        return dIydtheta, rel_ndIy

    def prepare_build_dIydI_j(
        self,
        j,
        rtol_NK,
        target_dIy,
        starting_dI,
        GS=True,  # , min_curr=1e-4, max_curr=300
    ):
        """
        Prepare the finite-difference derivative d(Iy)/dI_j by estimating the perturbation ΔI_j.

        This function determines the size of a current perturbation ΔI_j that produces
        a target change in the poloidal current vector Iy with norm ||ΔIy|| = target_dIy.
        Optionally, the full Grad–Shafranov (GS) problem can be solved to update the equilibrium,
        or a simplified approach using the modified tokamak flux can be used.

        Parameters
        ----------
        j : int
            Index of the coil current to be varied, corresponding to self.currents_vec.
        rtol_NK : float
            Relative tolerance for the static Grad–Shafranov solver.
        target_dIy : float
            Target norm of the induced change in Iy used to scale the perturbation.
        starting_dI : float
            Initial guess for the coil current perturbation ΔI_j.
        GS : bool, optional
            If True, solve the full Grad–Shafranov problem; if False, use modified tokamak flux
            without solving GS (default: True).

        Returns
        -------
        dIy_scaled : ndarray
            Initial finite-difference estimate of ΔIy / ΔI_j.
        rel_ndIy : float
            Relative norm of the induced change in Iy, normalized by self.nIy.
        """

        current_ = np.copy(self.currents_vec)
        current_[j] += starting_dI

        # reset the auxiliary equilibrium
        self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
        if GS:
            # solve
            self.assign_currents_solve_GS(current_, rtol_NK)
        else:
            # just use modified tokamak_psi
            self.assign_currents(current_, self.eq2, self.profiles2)
            self.profiles2.Jtor(
                self.eqR,
                self.eqZ,
                self.eq2.plasma_psi + self.eq2.tokamak.getPsitokamak(self.eq1._vgreen),
            )

        dIy_0 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy

        rel_ndIy_0 = np.linalg.norm(dIy_0) / self.nIy
        final_dI = starting_dI * target_dIy / rel_ndIy_0
        # final_dI = np.clip(final_dI, min_curr, max_curr)
        self.final_dI_record[j] = final_dI
        return dIy_0 / starting_dI, rel_ndIy_0

    def build_dIydI_j(self, j, rtol_NK):
        """
        Compute the finite-difference derivative d(Iy)/dI_j using the prepared perturbation.

        Uses the perturbation ΔI_j inferred by `prepare_build_dIydI_j` to compute the
        actual finite-difference derivative of the poloidal current vector Iy with respect
        to coil current I_j. Solves the Grad–Shafranov equilibrium at the perturbed current.

        Parameters
        ----------
        j : int
            Index of the coil current to be varied, corresponding to self.currents_vec.
        rtol_NK : float
            Relative tolerance for the static Grad–Shafranov solver.

        Returns
        -------
        dIydIj : ndarray
            Finite-difference derivative d(Iy)/dI_j, a 1D array over all grid points
            in the reduced plasma domain.
        rel_ndIy : float
            Relative norm of the induced change in Iy, normalized by self.nIy.
        """

        final_dI = 1.0 * self.final_dI_record[j]
        self.current_at_last_linearization[j] = self.currents_vec[j]

        current_ = np.copy(self.currents_vec)
        current_[j] += final_dI

        # reset the auxiliary equilibrium
        self.eq2.plasma_psi = np.copy(self.eq1.plasma_psi)
        # solve
        self.assign_currents_solve_GS(current_, rtol_NK)

        dIy_1 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        dIydIj = dIy_1 / final_dI

        rel_ndIy = np.linalg.norm(dIy_1) / self.nIy

        return dIydIj, rel_ndIy

    def build_linearization(
        self,
        eq,
        profiles,
        dIydI,
        dIydtheta,
        target_relative_tolerance_linearization,
        force_core_mask_linearization,
        verbose,
    ):
        """
        Builds the Jacobians d(Iy)/dI and d(Iy)/dtheta for linearizing the plasma-current
        response around a given equilibrium. These Jacobians are used to set up the
        solver for the linearised problem, providing initial slopes for both coil currents
        and plasma profile parameters.

        This function optionally computes dIydI and dIydtheta if they are not provided.
        Perturbations are applied to coil currents and profile parameters, and the corresponding
        changes in the poloidal current vector Iy are used to form finite-difference derivatives.
        The perturbations are adjusted to ensure stability and, if requested, to maintain the
        plasma core mask.

        Parameters
        ----------
        eq : FreeGS4E equilibrium object
            Equilibrium around which to linearize.
        profiles : FreeGS4E profiles object
            Plasma profiles associated with the equilibrium.
        dIydI : np.ndarray or None
            Optional input Jacobian of plasma current density with respect to metal currents.
            If None, it will be computed internally.
        dIydtheta : np.ndarray or None
            Optional input Jacobian of plasma current density with respect to plasma profile parameters.
            If None, it will be computed internally.
        target_relative_tolerance_linearization : float
            Relative tolerance used in the static Grad–Shafranov solver during finite-difference calculations.
        force_core_mask_linearization : bool
            If True, adjusts the perturbations so that the perturbed plasma core mask
            remains identical to the original mask.
        verbose : bool
            If True, prints intermediate results including initial and final perturbations,
            relative changes in Iy, and Grad–Shafranov residuals.

        Notes
        -----
        - This function updates self.dIydI and self.dIydtheta (or copies from initial conditions
        if already computed) and stores final perturbation magnitudes in self.final_dI_record
        and self.final_dtheta_record.
        - The core mask consistency is checked when force_core_mask_linearization is True.
        - The function also updates the derivatives of coil positions with respect to currents
        in self.dRZdI.
        """

        # if (dIydI is None) and (self.dIydI is None):
        self.build_current_vec(eq, profiles)
        self.Iy = self.limiter_handler.Iy_from_jtor(profiles.jtor).copy()
        self.nIy = np.linalg.norm(self.Iy)

        self.R0 = eq.Rcurrent()
        self.Z0 = eq.Zcurrent()
        self.dRZdI = np.zeros((2, self.n_metal_modes + 1))

        # compose the vector of initial delta_currents to be used for the finite difference calculation
        # this uses the variation to Jtor caused by the coil's contribution to the flux, ignoring the response of the plasma

        # build/update dIydI
        # dIydI = 1
        if dIydI is None:
            if self.dIydI_ICs is None:
                print(
                    f"Building the {self.plasma_domain_size} x {self.n_metal_modes + 1} Jacobian (dIy/dI)",
                    "of plasma current density (inside the LCFS)",
                    "with respect to all metal currents and the total plasma current.",
                )

                self.dIydI = np.zeros((self.plasma_domain_size, self.n_metal_modes + 1))
                self.psideltaI = np.zeros((self.n_metal_modes + 1, self.nx, self.ny))
                self.ddIyddI = np.zeros(self.n_metal_modes + 1)
                self.final_dI_record = np.zeros(self.n_metal_modes + 1)

                for j in self.arange_currents:

                    this_target_dIy = 1.0 * self.approved_target_dIy[j]
                    dIydIj, ndIy = self.prepare_build_dIydI_j(
                        j,
                        target_relative_tolerance_linearization,
                        this_target_dIy,
                        self.starting_dI[j],
                        GS=True,
                    )
                    core_check = (
                        np.sum(
                            np.abs(
                                self.profiles1.diverted_core_mask.astype(float)
                                - self.profiles2.diverted_core_mask.astype(float)
                            )
                        )
                        == 0
                    )
                    if force_core_mask_linearization:
                        while core_check == False:
                            self.starting_dI[j] /= 1.5
                            this_target_dIy /= 1.5
                            dIydIj, ndIy = self.prepare_build_dIydI_j(
                                j,
                                target_relative_tolerance_linearization,
                                this_target_dIy,
                                self.starting_dI[j],
                            )
                            core_check = (
                                np.sum(
                                    np.abs(
                                        self.profiles1.diverted_core_mask.astype(float)
                                        - self.profiles2.diverted_core_mask.astype(
                                            float
                                        )
                                    )
                                )
                                == 0
                            )

                    if (
                        np.abs(np.log10(self.final_dI_record[j] / self.starting_dI[j]))
                        > 0.5
                    ):
                        dIydIj, rel_ndIy = self.build_dIydI_j(
                            j,
                            target_relative_tolerance_linearization,
                        )
                        core_check = (
                            np.sum(
                                np.abs(
                                    self.profiles1.diverted_core_mask.astype(float)
                                    - self.profiles2.diverted_core_mask.astype(float)
                                )
                            )
                            == 0
                        )
                        if force_core_mask_linearization:
                            while core_check == False:
                                self.final_dI_record[j] /= 1.2
                                dIydIj, rel_ndIy = self.build_dIydI_j(
                                    j,
                                    target_relative_tolerance_linearization,
                                )
                                core_check = (
                                    np.sum(
                                        np.abs(
                                            self.profiles1.diverted_core_mask.astype(
                                                float
                                            )
                                            - self.profiles2.diverted_core_mask.astype(
                                                float
                                            )
                                        )
                                    )
                                    == 0
                                )
                    else:
                        self.final_dI_record[j] = 1.0 * self.starting_dI[j]

                    if verbose:
                        print("")
                        print(f"Mode: {j}")
                        print(f"  Initial delta_current = {self.starting_dI[j]}")
                        print(f"  Initial relative Iy change = {ndIy}")
                        print(f"  Final delta_current = {self.final_dI_record[j]}")
                        print("")
                        print(f"  Final relative Iy change = {rel_ndIy}")
                        print(
                            f"  Initial vs. Final GS residual: {self.NK.initial_rel_residual} vs. {self.NK.relative_change}"
                        )

                    self.dIydI[:, j] = np.copy(dIydIj)
                    self.psideltaI[j] = np.copy(self.eq2.psi())
                    R0 = self.eq2.Rcurrent()
                    Z0 = self.eq2.Zcurrent()
                    self.dRZdI[0, j] = (R0 - self.R0) / self.final_dI_record[j]
                    self.dRZdI[1, j] = (Z0 - self.Z0) / self.final_dI_record[j]

                self.dIydI_ICs = np.copy(self.dIydI)
            else:
                self.dIydI = np.copy(self.dIydI_ICs)
        else:
            self.dIydI = dIydI
            self.dIydI_ICs = np.copy(self.dIydI)

        # compose the vector of initial delta_theta (profile parameters) to be used for the finite difference calculation
        # - this uses the variation to Jtor caused by the parameter's contribution to the change in poloidal flux, ignoring the response of the plasma

        # build/update dIydtheta
        if dIydtheta is None and force_core_mask_linearization is False:
            if self.dIydtheta_ICs is None:
                print(
                    f"Building the {self.plasma_domain_size} x {self.n_profiles_parameters} Jacobian (dIy/dtheta)",
                    "of plasma current density (inside the LCFS)",
                    "with respect to all plasma current density profile parameters within Jtor.",
                )

                self.dIydtheta = np.zeros(
                    (self.plasma_domain_size, self.n_profiles_parameters)
                )

                profiles_copy = profiles.copy()

                # prepare to build the Jacobian by finding appropriate step size
                dIydtheta, ndIy = self.prepare_build_dIydtheta(
                    profiles=profiles_copy,
                    rtol_NK=target_relative_tolerance_linearization,
                    target_dIy=self.approved_target_dtheta,
                    starting_dtheta=self.starting_dtheta,
                    verbose=verbose,
                )

                if (
                    np.abs(np.log10(self.final_dtheta_record / self.starting_dtheta))
                    > 0.5
                ).any():

                    dIydtheta, rel_ndIy = self.build_dIydtheta(
                        profiles=profiles_copy,
                        rtol_NK=target_relative_tolerance_linearization,
                        verbose=verbose,
                    )

                else:
                    self.final_dtheta_record = 1.0 * self.starting_dtheta

                self.dIydtheta = np.copy(dIydtheta)
                self.dIydtheta_ICs = np.copy(self.dIydtheta)
            else:
                self.dIydtheta = np.copy(self.dIydtheta_ICs)
        else:
            self.dIydtheta = dIydtheta
            self.dIydtheta_ICs = np.copy(self.dIydtheta)

    def set_plasma_resistivity(self, plasma_resistivity):
        """
        Set the resistivity of the plasma and update the corresponding diagonal
        plasma resistance vector used in circuit calculations.

        The vector `self.plasma_resistance_1d` represents the diagonal of the plasma
        resistance matrix R_yy, restricted to the reduced domain defined by
        `plasma_domain_mask` (grid points inside the limiter).

        Parameters
        ----------
        plasma_resistivity : float
            Scalar value of the plasma resistivity [Ohm·m]. The resistance at each
            grid point in the reduced domain is computed as

                R_yy = 2 * π * plasma_resistivity * R / (dR * dZ)

            where `R` is the major radius of the grid point and `dR*dZ` is the
            area of the corresponding domain element.
        """

        self.plasma_resistivity = plasma_resistivity
        plasma_resistance_matrix = (
            self.eqR * (2 * np.pi / self.dRdZ) * self.plasma_resistivity
        )
        self.plasma_resistance_1d = plasma_resistance_matrix[
            self.limiter_handler.mask_inside_limiter
        ]

    def reset_plasma_resistivity(self, plasma_resistivity):
        """
        Reset the plasma resistivity and update all relevant solver objects.

        This updates `self.plasma_resistance_1d`, the diagonal of the plasma
        resistance matrix restricted to the reduced domain (inside the limiter),
        and also propagates the updated resistivity to the linearized and simplified solvers.

        Parameters
        ----------
        plasma_resistivity : float
            Scalar value of the plasma resistivity [Ohm·m]. The resistance at each
            grid point in the reduced domain is computed as

                R_yy = 2 * π * plasma_resistivity * R / (dR * dZ)

            where `R` is the major radius of the grid point and `dR*dZ` is the
            area of the corresponding domain element.
        """

        self.plasma_resistivity = plasma_resistivity
        plasma_resistance_matrix = (
            self.eqR * (2 * np.pi / self.dRdZ) * self.plasma_resistivity
        )
        self.plasma_resistance_1d = plasma_resistance_matrix[
            self.limiter_handler.mask_inside_limiter
        ]

        self.linearised_sol.reset_plasma_resistivity(self.plasma_resistance_1d)
        self.simplified_solver_J1.reset_plasma_resistivity(self.plasma_resistance_1d)

    def check_and_change_plasma_resistivity(
        self, plasma_resistivity, relative_threshold_difference=0.01
    ):
        """
        Check if the plasma resistivity differs from the current value and update it if necessary.

        Compares the new `plasma_resistivity` with the current `self.plasma_resistivity`.
        If the relative difference exceeds `relative_threshold_difference`, the resistivity
        is reset using `reset_plasma_resistivity`, which also updates the associated solver objects.

        Parameters
        ----------
        plasma_resistivity : float
            New scalar value of the plasma resistivity [Ohm·m]. The resistance at each
            grid point in the reduced domain is computed as

                R_yy = 2 * π * plasma_resistivity * R / (dR * dZ)

            where `R` is the major radius of the grid point and `dR*dZ` is the
            area of the corresponding domain element.

        relative_threshold_difference : float, optional
            Relative threshold for updating the resistivity. Default is 0.01 (1%).
            If the relative change in resistivity exceeds this value, the resistivity is reset.
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
        """
        Compute the lumped plasma resistance using the plasma resistance matrix R_yy.

        The lumped resistance is calculated by contracting the 1D plasma resistance
        vector (`self.plasma_resistance_1d`) with two normalized plasma current
        distribution vectors. This provides an effective scalar resistance for
        the given current profiles.

        Parameters
        ----------
        norm_red_Iy0 : np.array
            Normalized plasma current distribution vector at time t_0. The entries
            should be normalized so that their sum equals 1.
        norm_red_Iy1 : np.array
            Normalized plasma current distribution vector at time t_1. The entries
            should be normalized so that their sum equals 1.

        Returns
        -------
        float
            Lumped plasma resistance corresponding to the given current distributions.
        """
        lumped_plasma_resistance = np.sum(
            self.plasma_resistance_1d * norm_red_Iy0 * norm_red_Iy1
        )
        return lumped_plasma_resistance

    def reset_timestep(self, full_timestep, max_internal_timestep):
        """
        Reset the timestep parameters for the simulation.

        Updates the full timestep used to advance the dynamics and the maximum
        allowed internal sub-timestep used for circuit equation integration.
        Resets the corresponding timesteps in all relevant solver objects.

        Parameters
        ----------
        full_timestep : float
            The time interval dt over which the stepper advances the full system dynamics.
            Applies to both linear and nonlinear steppers. A Grad-Shafranov (GS)
            equilibrium is recalculated at every full_timestep.
        max_internal_timestep : float
            Maximum size of sub-steps when advancing a single full_timestep.
            These sub-steps are used to integrate the circuit equations.
        """
        self.dt_step = full_timestep
        self.max_internal_timestep = max_internal_timestep

        self.evol_metal_curr.reset_timesteps(
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
        """
        Extracts and stores relevant properties from a FreeGS4E profiles object.

        Sets internal attributes describing the profile type, number of independent
        parameters, and their values, which are used in linearisation and stepper
        calculations.

        Parameters
        ----------
        profiles : FreeGS4E profiles object
            The profiles object of the initial equilibrium. This provides the parameters
            defining the plasma current density profile, e.g., alpha_m, alpha_n, paxis,
            betap, Beta0, or Lao coefficients.
        """
        self.fvac = profiles.fvac

        self.profiles_type = type(profiles).__name__

        # note the parameters here are the same that should be provided
        # to the stepper if these are time evolving
        if self.profiles_type == "ConstrainPaxisIp":
            self.n_profiles_parameters = 3
            self.profiles_parameters = {
                "alpha_m": profiles.alpha_m,
                "alpha_n": profiles.alpha_n,
                "paxis": profiles.paxis,
            }
            self.profiles_param = "paxis"
            self.profiles_parameters_vec = np.array(
                [profiles.alpha_m, profiles.alpha_n, profiles.paxis]
            )
        elif self.profiles_type == "ConstrainBetapIp":
            self.n_profiles_parameters = 3
            self.profiles_parameters = {
                "alpha_m": profiles.alpha_m,
                "alpha_n": profiles.alpha_n,
                "betap": profiles.betap,
            }
            self.profiles_param = "betap"
            self.profiles_parameters_vec = np.array(
                [profiles.alpha_m, profiles.alpha_n, profiles.betap]
            )
        elif self.profiles_type == "Fiesta_Topeol":
            self.n_profiles_parameters = 3
            self.profiles_parameters = {
                "alpha_m": profiles.alpha_m,
                "alpha_n": profiles.alpha_n,
                "Beta0": profiles.Beta0,
            }
            self.profiles_param = "Beta0"
            self.profiles_parameters_vec = np.array(
                [profiles.alpha_m, profiles.alpha_n, profiles.Beta0]
            )
        elif self.profiles_type == "Lao85":
            self.n_profiles_parameters_alpha = len(profiles.alpha)
            self.n_profiles_parameters_beta = len(profiles.beta)
            if profiles.alpha_logic:
                self.n_profiles_parameters_alpha -= 1
            if profiles.beta_logic:
                self.n_profiles_parameters_beta -= 1
            self.n_profiles_parameters = (
                self.n_profiles_parameters_alpha + self.n_profiles_parameters_beta
            )

            self.profiles_alpha_indices = slice(0, self.n_profiles_parameters_alpha)
            alpha_shift = 0
            if profiles.alpha_logic:
                alpha_shift += 1

            self.profiles_beta_indices = slice(
                self.n_profiles_parameters_alpha + alpha_shift,
                self.n_profiles_parameters_alpha
                + alpha_shift
                + self.n_profiles_parameters_beta,
            )

            self.profiles_parameters = {"alpha": profiles.alpha, "beta": profiles.beta}
            self.profiles_param = None
            self.profiles_parameters_vec = np.concatenate(
                (profiles.alpha, profiles.beta)
            )

    def get_vessel_currents(self, eq):
        """
        Extracts and stores all metal currents from a given equilibrium.

        Retrieves currents for both active coils and passive vessel structures from
        the input equilibrium's tokamak object. The values are stored in
        `self.vessel_currents_vec`.

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            The equilibrium from which to extract metal currents. Uses `eq.tokamak`
            to access current values.
        """
        self.vessel_currents_vec = eq.tokamak.getCurrentsVec()

    def build_current_vec(self, eq, profiles):
        """
        Constructs the vector of currents for the dynamics solver.

        The vector `self.currents_vec` includes, in order:
        - Active coil currents
        - Selected vessel normal mode currents
        - Total plasma current normalized by `plasma_norm_factor`

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            The equilibrium used to extract all metal currents before any mode truncation.
        profiles : FreeGS4E profiles object
            Profiles of the initial equilibrium. Used to extract the total plasma current.
        """

        # gets metal currents, note these are before mode truncation!
        self.get_vessel_currents(eq)

        # transforms in normal modes (including truncation)
        self.currents_vec[: self.n_metal_modes] = self.evol_metal_curr.IvesseltoId(
            Ivessel=self.vessel_currents_vec
        )

        # extracts total plasma current value
        self.currents_vec[-1] = profiles.Ip / self.plasma_norm_factor

        # this is currents_vec(t-dt):
        self.currents_vec_m1 = np.copy(self.currents_vec)

    def initialize_from_ICs(
        self,
        eq,
        profiles,
        target_relative_tolerance_linearization=1e-7,
        dIydI=None,
        dIydtheta=None,
        force_core_mask_linearization=False,
        verbose=False,
    ):
        """
        Initialize the dynamics solver from a given equilibrium and plasma profiles.

        This method prepares all internal structures required for advancing the
        coupled plasma–vessel dynamics, including:

        - Building the vector of currents (`self.currents_vec`) comprising
        active coil currents, selected vessel normal modes, and normalized plasma current.
        - Setting up auxiliary equilibrium (`eq2`) and profile (`profiles2`) objects for intermediate calculations.
        - Computing the linearized Jacobians d(Iy)/dI and d(Iy)/dtheta if not provided,
        and transferring them to the linear solver.

        Parameters
        ----------
        eq : FreeGS4E equilibrium object
            The initial equilibrium, used to assign all metal currents.
        profiles : FreeGS4E profiles object
            Profiles of the initial equilibrium, used to assign the total plasma current.
        target_relative_tolerance_linearization : float, default=1e-7
            Relative tolerance for static Grad–Shafranov solves during initialization
            and Jacobian computation. This does not affect the solver's runtime tolerance.
        dIydI : np.array, optional
            Jacobian of plasma current distribution with respect to metal currents and total plasma current.
            Shape: (np.sum(plasma_domain_mask), n_metal_modes+1).
            If not provided, it is computed from the given equilibrium.
        dIydtheta : np.array, optional
            Jacobian of plasma current distribution with respect to plasma profile parameters.
            If not provided, it is computed from the given equilibrium.
        force_core_mask_linearization : bool, default=False
            If True, enforces that the perturbed plasma core mask remains identical
            when computing d(Iy)/dI.
        verbose : bool, default=False
            Enables progress printouts during initialization.

        Side Effects
        ------------
        - Sets up `self.eq1`, `self.profiles1`, `self.eq2`, `self.profiles2`.
        - Initializes `self.currents_vec`, `self.Iy`, `self.hatIy`, `self.blended_hatIy`.
        - Builds the linearization and transfers it to the linear solver (`self.linearised_sol`).
        - Updates the Myy matrix through `self.handleMyy`.
        """

        self.step_counter = 0
        self.currents_guess = False
        self.rtol_NK = target_relative_tolerance_linearization

        # get profiles parametrization
        # this is not currently used, as the linearised evolution
        # does not currently account for the evolving profiles coefficients
        self.get_profiles_values(profiles)

        # set internal copy of the equilibrium and profile
        # note that at this stage, the equilibrium may have vessel currents.
        # These can not be reproduced exactly if modes are truncated.
        self.eq1 = eq.create_auxiliary_equilibrium()
        self.profiles1 = profiles.copy()
        # The pair self.eq1 and self.profiles1 is the pair that is advanced at each timestep.
        # Their properties evolve according to the dynamics.
        # Note that the input eq and profiles are NOT modified by the evolution object.

        # this builds the vector of extensive current values 'currents_vec'
        # comprising (active coil currents, vessel normal modes, plasma current)
        # this vector is evolved in place when the stepper is called
        self.build_current_vec(self.eq1, self.profiles1)
        self.current_at_last_linearization = np.copy(self.currents_vec)

        # This has truncated the vessel currents, which needs to be mirrored in the equilibrium
        # First, the modified currents are assigned
        self.assign_currents(self.currents_vec, profiles=self.profiles1, eq=self.eq1)
        # Then the equilibrium is solved again to ensure it is the solution relevant to the truncated currents,
        # and not to the full set of vessel currents!
        self.NK.forward_solve(
            self.eq1,
            self.profiles1,
            target_relative_tolerance=target_relative_tolerance_linearization,
            suppress=True,
        )

        # self.eq2 and self.profiles2 are used as auxiliary objects when solving for the dynamics
        # They are used for all intermediate calculations, so
        # they should not be used to extract properties of the evolving equilibrium
        self.eq2 = self.eq1.create_auxiliary_equilibrium()
        self.profiles2 = self.profiles1.copy()

        # self.Iy is the istantaneous 1d vector representing the plasma current distribution
        # on the reduced plasma domain, as from plasma_domain_mask
        # self.Iy is updated every timestep
        self.Iy = self.limiter_handler.Iy_from_jtor(self.profiles1.jtor)
        # self.hatIy is the normalised plasma current distribution. This vector sums to 1.
        self.hatIy = self.limiter_handler.normalize_sum(self.Iy)
        # self.hatIy1 is the normalised plasma current distribution at time t+dt
        self.hatIy1 = np.copy(self.hatIy)
        self.make_blended_hatIy(self.hatIy1)

        self.time = 0
        self.step_no = -1

        # build the linearization if not provided
        self.build_linearization(
            self.eq1,
            self.profiles1,
            dIydI=dIydI,
            dIydtheta=dIydtheta,
            target_relative_tolerance_linearization=target_relative_tolerance_linearization,
            force_core_mask_linearization=force_core_mask_linearization,
            verbose=verbose,
        )

        # set Myy matrix in place throught the handling object
        self.handleMyy.force_build_Myy(self.hatIy)

        # transfer linearization to linear solver:
        self.Myy_hatIy0 = self.handleMyy.dot(self.hatIy)
        self.linearised_sol.set_linearization_point(
            dIydI=self.dIydI_ICs,
            dIydtheta=self.dIydtheta_ICs,
            hatIy0=self.blended_hatIy,
            Myy_hatIy0=self.Myy_hatIy0,
        )

    def step_complete_assign(self, working_relative_tol_GS, from_linear=False):
        """
        Finalize the timestep advancement and update the equilibrium and current state.

        This function completes the evolution over a timestep `dt_step` by:
        - Recording the time-evolved currents (`self.trial_currents`) in `self.currents_vec`.
        - Updating the equilibrium (`self.eq1`) and plasma profiles (`self.profiles1`) with
        the corresponding plasma flux (`self.trial_plasma_psi`) and derived current distribution.
        - Updating the normalized plasma current distribution (`self.hatIy`).

        Parameters
        ----------
        working_relative_tol_GS : float
            Fractional tolerance for the Grad–Shafranov solver during this timestep.
            The effective GS tolerance is set relative to the maximum change in plasma flux.
        from_linear : bool, default=False
            If True, only the linearised solution is applied. Reduces the number of full GS solves
            by copying auxiliary equilibrium and profiles to the main state.

        Side Effects
        ------------
        - Advances `self.time` and increments `self.step_no`.
        - Updates `self.currents_vec_m1` and `self.Iy_m1` to store previous step values.
        - Updates `self.currents_vec`, `self.Iy`, `self.hatIy`, and `self.rtol_NK`.
        - Modifies `self.eq1` and `self.profiles1` to reflect the timestep-evolved state.
        """

        self.time += self.dt_step
        self.step_no += 1

        self.currents_vec_m1 = np.copy(self.currents_vec)
        self.Iy_m1 = np.copy(self.Iy)

        plasma_psi_step = self.eq2.plasma_psi - self.eq1.plasma_psi
        self.d_plasma_psi_step = np.amax(plasma_psi_step) - np.amin(plasma_psi_step)

        self.currents_vec = np.copy(self.trial_currents)
        self.assign_currents(self.currents_vec, self.eq1, self.profiles1)
        self.eq1.tokamak.set_all_coil_currents(self.vessel_currents_vec)
        self.eq2.tokamak.set_all_coil_currents(self.vessel_currents_vec)

        if from_linear:
            self.profiles1 = self.profiles2.copy()
            self.eq1 = self.eq2.create_auxiliary_equilibrium()
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

    def assign_currents(self, currents_vec, eq, profiles):
        """
        Assigns the input currents to the equilibrium and plasma profiles.

        The function updates:
        - The total plasma current in `eq` and `profiles`.
        - The metal (vessel + active coil) currents in the tokamak, converting
            from normal mode representation to physical coil currents.

        Parameters
        ----------
        currents_vec : np.array
            Vector of current values to assign. Format:
                (active coil currents, vessel normal mode currents, total plasma current / plasma_norm_factor)
        eq : FreeGSNKE equilibrium Object
            Equilibrium object to be modified. Its `_current` attribute and `tokamak.current_vec` are updated.
        profiles : FreeGSNKE profiles Object
            Profiles object to be modified. Its total plasma current `Ip` is updated.

        Side Effects
        ------------
        - Updates `self.vessel_currents_vec` with the reconstructed physical currents.
        - Modifies the input `eq` and `profiles` in-place.
        """

        # assign plasma current to equilibrium
        eq._current = self.plasma_norm_factor * currents_vec[-1]
        profiles.Ip = self.plasma_norm_factor * currents_vec[-1]

        # calculate vessel currents from normal modes and assign
        self.vessel_currents_vec = self.evol_metal_curr.IdtoIvessel(
            Id=currents_vec[:-1]
        )
        eq.tokamak.current_vec = self.vessel_currents_vec.copy()

    def assign_currents_solve_GS(self, currents_vec, rtol_NK):
        """
        Assigns the input currents to the auxiliary equilibrium (`self.eq2`) and profiles (`self.profiles2`),
        then solves the static Grad-Shafranov (GS) problem to find the resulting plasma flux and current distribution.

        Parameters
        ----------
        currents_vec : np.array
            Vector of current values to assign. Format matches `self.assign_currents`:
                (active coil currents, vessel normal mode currents, total plasma current / plasma_norm_factor)
        rtol_NK : float
            Relative tolerance to be used in the static GS solver.

        Side Effects
        ------------
        - Updates `self.eq2._current` and `self.profiles2.Ip`.
        - Updates `self.eq2.tokamak.current_vec` with the reconstructed physical currents.
        - Solves the GS equation, updating `self.eq2.plasma_psi` and `self.profiles2.jtor`.
        """
        self.assign_currents(currents_vec, profiles=self.profiles2, eq=self.eq2)
        self.NK.forward_solve(
            self.eq2,
            self.profiles2,
            target_relative_tolerance=rtol_NK,
            suppress=True,
        )

    def make_blended_hatIy_(self, hatIy1, blend):
        """
        Produces a weighted average of the current plasma distribution at time t
        (`self.hatIy`) and a guess for the distribution at time t+dt (`hatIy1`).

        Parameters
        ----------
        hatIy1 : np.array
            Guess for the normalized plasma current distribution at time t+dt.
            Must sum to 1. Only covers the reduced plasma domain.
        blend : float
            Weighting factor between 0 and 1.
            - blend=0 → only uses hatIy1
            - blend=1 → only uses hatIy (current time)

        Side Effects
        ------------
        - Sets `self.blended_hatIy` to the weighted combination:
            blended_hatIy = (1 - blend) * hatIy1 + blend * self.hatIy
        """

        self.blended_hatIy = (1 - blend) * hatIy1 + blend * self.hatIy

    def currents_from_hatIy(self, hatIy1, active_voltage_vec):
        """
        Computes the full set of currents at time t+dt from a guess of the normalized plasma current distribution,
        using the simplified circuit solver.

        Parameters
        ----------
        hatIy1 : np.array
            Guess for the normalized plasma current distribution at time t+dt (sum=1, reduced plasma domain).
        active_voltage_vec : np.array
            Voltages applied to the active coils between t and t+dt.

        Returns
        -------
        np.array
            Full current vector at t+dt, matching the format of self.currents_vec.

        Workflow
        --------
        1. Computes a blended plasma distribution using `make_blended_hatIy`.
        2. Computes `Myy_hatIy_left = self.handleMyy.dot(blended_hatIy)`.
        3. Calls `self.simplified_solver_J1.stepper` to solve for all currents, given the blended distribution
        and applied coil voltages.
        """
        self.make_blended_hatIy(hatIy1)
        Myy_hatIy_left = self.handleMyy.dot(self.blended_hatIy)
        current_from_hatIy = self.simplified_solver_J1.stepper(
            It=self.currents_vec,
            hatIy_left=self.blended_hatIy,
            hatIy_0=self.hatIy,
            hatIy_1=hatIy1,
            active_voltage_vec=active_voltage_vec,
            Myy_hatIy_left=Myy_hatIy_left,
        )
        return current_from_hatIy

    def hatIy1_iterative_cycle(self, hatIy1, active_voltage_vec, rtol_NK):
        """
        Performs one iteration of the cycle:
        1. Uses a guessed plasma current distribution at t+dt (`hatIy1`) to compute all currents.
        2. Solves the static Grad-Shafranov (GS) problem to find the resulting plasma flux and updated plasma current distribution.

        Parameters
        ----------
        hatIy1 : np.array
            Guess for the normalized plasma current distribution at t+dt.
            Must sum to 1 (reduced plasma domain only).
        active_voltage_vec : np.array
            Voltages applied to the active coils between t and t+dt.
        rtol_NK : float
            Relative tolerance for the static GS solver.
        """
        current_from_hatIy = self.currents_from_hatIy(hatIy1, active_voltage_vec)
        self.assign_currents_solve_GS(currents_vec=current_from_hatIy, rtol_NK=rtol_NK)

    def calculate_hatIy(self, trial_currents, plasma_psi):
        """
        Computes the normalized plasma current distribution (hatIy) corresponding
        to a given set of currents and plasma flux.

        Parameters
        ----------
        trial_currents : np.array
            Full vector of currents (active coils, vessel modes, plasma current).
        plasma_psi : np.array
            Plasma flux on the full grid (2D array).

        Returns
        -------
        np.array
            Normalized plasma current distribution on the reduced domain.
        """
        self.assign_currents(trial_currents, profiles=self.profiles2, eq=self.eq2)
        self.tokamak_psi = self.eq2.tokamak.getPsitokamak(vgreen=self.eq2._vgreen)
        jtor_ = self.profiles2.Jtor(self.eqR, self.eqZ, self.tokamak_psi + plasma_psi)
        hat_Iy1 = self.limiter_handler.hat_Iy_from_jtor(jtor_)
        return hat_Iy1

    def calculate_hatIy_GS(self, trial_currents, rtol_NK, record_for_updates=False):
        """
        Computes the normalized plasma current distribution (hatIy) corresponding
        to a given set of currents by **solving the static Grad-Shafranov problem**.

        Parameters
        ----------
        trial_currents : np.array
            Full vector of currents (active coils, vessel modes, plasma current).
        rtol_NK : float
            Relative tolerance for the static GS solver.
        record_for_updates : bool, optional
            If True, records auxiliary updates (not always needed).

        Returns
        -------
        np.array
            Normalized plasma current distribution on the reduced plasma domain.
        """
        self.assign_currents_solve_GS(
            trial_currents, rtol_NK=rtol_NK, record_for_updates=record_for_updates
        )
        hatIy1 = self.limiter_handler.hat_Iy_from_jtor(self.profiles2.jtor)
        return hatIy1

    def F_function_curr(self, trial_currents, active_voltage_vec):
        """
        Evaluates the residual of the full non-linear plasma + circuit system
        for a given guess of currents at time t+dt.

        This function casts the coupled plasma and circuit dynamics as a root-finding
        problem in the space of current values. The residual is defined as the difference
        between the currents predicted by the simplified circuit equations (given a
        plasma current distribution) and the input trial currents. A zero residual
        indicates that the trial currents and the corresponding plasma flux
        form a self-consistent solution of the full non-linear system.

        The evaluation proceeds in two steps:
        1. Compute the normalized plasma current distribution `hatIy1` corresponding
        to the input `trial_currents` and the current plasma flux `self.trial_plasma_psi`.
        2. Compute the iterated currents using the simplified circuit equations
        and the plasma distribution `hatIy1`. The residual is then:
        `residual = iterated_currents - trial_currents`.

        Parameters
        ----------
        trial_currents : np.ndarray
            Vector of current values at time t+dt. The format matches `self.currents_vec`,
            typically including active coil currents, vessel mode currents, and the total
            plasma current (normalized by `plasma_norm_factor`).

        active_voltage_vec : np.ndarray
            Vector of voltages applied to the active coils over the timestep.

        Returns
        -------
        np.ndarray
            Residual vector of currents, same format as `self.currents_vec`.
            Zero residual indicates a self-consistent solution of the plasma + circuit system
            for the timestep.
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
        """
        Evaluates the residual of the full non-linear plasma + circuit system
        for a given guess of currents at time t+dt, solving the plasma flux
        self-consistently via the static Grad-Shafranov (GS) problem.

        Unlike `F_function_curr`, the plasma flux is not imposed but computed
        from the trial currents. The residual is defined as the difference
        between the currents predicted by the simplified circuit equations
        (given the resulting plasma distribution) and the input trial currents.
        A zero residual indicates that the trial currents form a self-consistent
        solution of the full non-linear system, including the plasma response.

        Iteration steps:
        1. Compute the plasma flux corresponding to `trial_currents` by solving the static GS problem.
        2. Compute the normalized plasma current distribution `hatIy1` from
        the combination of trial currents and computed plasma flux.
        3. Compute iterated currents from the simplified circuit equations using `hatIy1`.
        4. Residual: `residual = iterated_currents - trial_currents`.

        Parameters
        ----------
        trial_currents : np.ndarray
            Vector of current values at time t+dt. Format matches `self.currents_vec`,
            typically including active coil currents, vessel mode currents, and total
            plasma current (normalized by `plasma_norm_factor`).

        active_voltage_vec : np.ndarray
            Vector of voltages applied to the active coils over the timestep.

        rtol_NK : float
            Relative tolerance to be used in the static GS solver for the plasma flux.

        Returns
        -------
        np.ndarray
            Residual vector of currents, same format as `self.currents_vec`.
            Zero residual indicates a self-consistent solution of the plasma + circuit system.
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
        """
        Computes the relative residual of the current update compared to the
        actual step taken in the currents. This quantifies the convergence
        of the timestepper by comparing the residual to the magnitude of the
        current change.

        The relative residual is defined as:

            relative_residual_i = |current_residual_i / max(|ΔI_i|, curr_eps)|

        where ΔI_i = trial_currents_i - currents_vec_m1_i, and `curr_eps` prevents
        division by very small steps.

        Parameters
        ----------
        current_residual : np.ndarray
            Residual of the current values at the current timestep. Same format
            as `self.currents_vec`.

        curr_eps : float
            Minimum allowable step size in the currents. Used to avoid
            artificially large relative residuals when the step is very small.

        Returns
        -------
        np.ndarray
            Relative residual of the current update. Same format as `self.currents_vec`.
            Values close to 0 indicate good convergence of the timestep.
        """
        curr_step = abs(self.trial_currents - self.currents_vec_m1)
        self.curr_step = np.where(curr_step > curr_eps, curr_step, curr_eps)
        rel_curr_res = abs(current_residual / self.curr_step)
        return rel_curr_res

    def calculate_rel_tolerance_GS(self, trial_plasma_psi, a_res_GS=None):
        """
        Computes the relative residual of the plasma flux for the static Grad-Shafranov (GS)
        problem, comparing the GS residual to the actual change in plasma flux due to dynamics.
        This metric quantifies the convergence of the timestepper for the plasma flux update.

        The relative residual is defined as:

            r_res_GS = max(|GS_residual|) / max(|Δpsi|)

        where Δpsi = psi(t+dt) - psi(t). If the GS residual `a_res_GS` is not provided,
        it is computed internally using the static GS solver.

        Parameters
        ----------
        trial_plasma_psi : np.ndarray
            Plasma flux at the current timestep, psi(t+dt), shape (nx, ny).

        a_res_GS : np.ndarray, optional
            Residual of the static GS problem at t+dt. If None, it will be calculated
            internally. Shape should match the flattened plasma flux.

        Returns
        -------
        float
            Relative plasma flux residual. Values close to 0 indicate good convergence
            of the plasma flux update.
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

    def check_and_change_profiles(self, profiles_parameters=None):
        """
        Updates the plasma current profile parameters at time t+dt if new values are provided.

        This method checks whether a dictionary of new profile parameters is supplied.
        If so, it updates both the evolving equilibrium profiles (`self.profiles1`)
        and the auxiliary profiles (`self.profiles2`) accordingly.
        For profiles of type "Lao85", the internal profile initialization routine is called
        after updating the parameters. A flag is set to indicate that a change occurred.

        Parameters
        ----------
        profiles_parameters : dict or None, optional
            Dictionary of profile parameters to update. Keys and values should match the
            attributes of the profile object (see `get_profiles_values` for structure).
            If None, no changes are made and the profiles remain unchanged.

        Notes
        -----
        - Sets `self.profiles_change_flag = 1` if parameters are updated, otherwise 0.
        - Both the main (`profiles1`) and auxiliary (`profiles2`) profiles are updated
        to ensure consistency during timestep calculations.
        """
        self.profiles_change_flag = 0

        if profiles_parameters is not None:
            for par in profiles_parameters:
                setattr(self.profiles1, par, profiles_parameters[par])
                setattr(self.profiles2, par, profiles_parameters[par])
            if self.profiles_type == "Lao85":
                self.profiles1.initialize_profile()
                self.profiles2.initialize_profile()
            self.profiles_change_flag = 1

    def check_and_change_active_coil_resistances(self, active_coil_resistances):
        """
        Checks if new active coil resistances are provided and updates them if needed.

        This method compares the input array of active coil resistances with the current
        resistances stored in `self.evol_metal_curr`. If the input is different, it resets
        the resistances, updates the solvers accordingly, and prints the new coil resistances.

        Parameters
        ----------
        active_coil_resistances : np.array or None
            Array of new resistances for the active coils. If None, no changes are made.

        Notes
        -----
        - If the input resistances are identical to the current ones, the method does nothing.
        - When resistances are updated, `self.set_solvers()` is called to ensure the
        solvers reflect the new electrical properties.
        - The updated resistances are printed for confirmation.
        """

        if active_coil_resistances is None:
            return
        else:
            if np.array_equal(
                active_coil_resistances, self.evol_metal_curr.active_coil_resistances
            ):
                return
            else:
                self.evol_metal_curr.reset_active_coil_resistances(
                    active_coil_resistances
                )
                self.set_solvers()
                print(self.evol_metal_curr.coil_resist)

    def nlstepper(
        self,
        active_voltage_vec,
        profiles_parameters=None,
        plasma_resistivity=None,
        target_relative_tol_currents=0.005,
        target_relative_tol_GS=0.003,
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
        max_solving_iterations=50,
        custom_active_coil_resistances=None,
    ):
        """
        Advance the system by one timestep using a nonlinear Newton-Krylov (NK) stepper.

        If ``linear_only=True``, only the linearised dynamic problem is advanced.
        Otherwise, a full nonlinear solution is sought using an iterative NK-based algorithm.
        On convergence, the timestep is advanced by ``self.dt_step`` and the updated
        currents, equilibrium, and profile objects are assigned to ``self.currents_vec``,
        ``self.eq1``, and ``self.profiles1``.

        Algorithm overview
        ------------------
        The solver proceeds as follows:

        1. Solve the linearised problem to obtain an initial guess for the currents and
        solve the associated static Grad–Shafranov (GS) problem, yielding
        ``trial_plasma_psi`` and ``trial_currents`` (including ``tokamak_psi``).
        2. If the pair [``trial_plasma_psi``, ``tokamak_psi``] fails the GS tolerance check,
        update ``trial_plasma_psi`` toward the GS solution.
        3. At fixed currents, update ``trial_plasma_psi`` via NK iterations on the
        root problem in plasma flux.
        4. At fixed plasma flux, update currents via NK iterations on the root problem
        in currents.
        5. If either the current residuals or the GS tolerance check fail, return to step 2.
        6. On convergence, record the solution into ``self.currents_vec``, ``self.eq1``,
        and ``self.profiles1``.

        Parameters
        ----------
        active_voltage_vec : np.ndarray
            Vector of applied voltages on the active coils between ``t`` and ``t+dt``.
        profiles_parameters : dict or None, optional
            If None, profile parameters remain unchanged. Otherwise, dictionary specifying
            updated parameters for the profiles object. See ``get_profiles_values`` for
            dictionary structure. This enables time-dependent profile parameters.
        plasma_resistivity : float or array-like, optional
            Updated plasma resistivity. If None, resistivity is left unchanged. Enables time-
            dependent resistivity.
        target_relative_tol_currents : float, optional, default=0.005
            Required relative tolerance on currents for convergence of the dynamic problem.
            Computed as ``residual / (currents(t+dt) - currents(t))``.
        target_relative_tol_GS : float, optional, default=0.003
            Required relative tolerance on plasma flux for convergence of the static GS problem.
            Computed as ``residual / Δψ`` where Δψ is the flux change between timesteps.
        working_relative_tol_GS : float, optional, default=0.001
            Tolerance used when solving intermediate GS problems during the step.
            Must be stricter than ``target_relative_tol_GS``.
        target_relative_unexplained_residual : float, optional, default=0.5
            NK solver stopping criterion: inclusion of additional Krylov basis vectors
            stops once more than ``1 - target_relative_unexplained_residual`` of the residual
            is canceled.
        max_n_directions : int, optional, default=3
            Maximum number of Krylov basis vectors used in NK updates.
        step_size_psi : float, optional, default=2.0
            Step size for finite difference calculations in the NK solver applied to ψ,
            measured in units of the residual norm.
        step_size_curr : float, optional, default=0.8
            Step size for finite difference calculations in the NK solver applied to currents,
            measured in units of the residual norm.
        scaling_with_n : int, optional, default=0
            Exponent controlling step scaling in NK updates:
            candidate step is scaled by ``(1 + n_iterations)**scaling_with_n``.
        blend_GS : float, optional, default=0.5
            Blending coefficient used when updating ``trial_plasma_psi`` toward the GS solution.
            Must be in [0, 1].
        curr_eps : float, optional, default=1e-5
            Regularisation parameter for relative current convergence checks,
            preventing division by small current changes.
        max_no_NK_psi : float, optional, default=5.0
            Threshold for triggering NK updates on ψ. Activated if
            ``relative_psi_residual > max_no_NK_psi * target_relative_tol_GS``.
        clip : float, optional, default=5
            Maximum allowed step size for each accepted Krylov basis vector, in units
            of the exploratory step.
        verbose : int, optional, default=0
            Verbosity level.
            * 0: silent
            * 1: report convergence progress per NK cycle
            * 2: include detailed intermediate output
        linear_only : bool, optional, default=False
            If True, only the linearised solution is used (skipping nonlinear solves).
        max_solving_iterations : int, optional, default=50
            Maximum number of nonlinear NK cycles before the solve is terminated.
        custom_active_coil_resistances : array-like or None, optional
            If provided, overrides default active coil resistances with those specifed.
            Enables time-dependent coil resistances (can be used for switching coils "on"
            and "off").

        Notes
        -----
        On convergence, the method updates internal state:
        - ``self.currents_vec`` stores the evolved currents.
        - ``self.eq1`` stores the new Grad–Shafranov equilibrium.
        - ``self.profiles1`` stores the updated profile object.

        Raises
        ------
        RuntimeError
            If the nonlinear solve does not converge within ``max_solving_iterations``.
        """

        # retrieve the old profile parameter values
        self.get_profiles_values(self.profiles1)
        old_params = self.profiles_parameters_vec

        # check if profiles parameters are being evolved
        # and action the change where necessary
        self.check_and_change_profiles(
            profiles_parameters=profiles_parameters,
        )

        self.check_and_change_active_coil_resistances(
            active_coil_resistances=custom_active_coil_resistances
        )
        # retrieve the new profile parameter values (if present)
        self.get_profiles_values(self.profiles1)
        new_params = self.profiles_parameters_vec

        # calculate change in profiles across timestep: (profiles(t+dt)-profiles(t))/dt
        # should be zero if no change
        if self.profiles_type == "Lao85":
            old_alphas = old_params[self.profiles_alpha_indices]
            old_betas = old_params[self.profiles_beta_indices]
            new_alphas = new_params[self.profiles_alpha_indices]
            new_betas = new_params[self.profiles_beta_indices]
            dtheta_dt = (
                np.concatenate((new_alphas, new_betas))
                - np.concatenate((old_alphas, old_betas))
            ) / self.dt_step
        else:
            dtheta_dt = (new_params - old_params) / self.dt_step

        # check if plasma resistivity is being evolved
        # and action the change where necessary
        self.check_and_change_plasma_resistivity(
            plasma_resistivity,
        )

        # solves the linearised problem for the currents and assigns
        # results in preparation for the nonlinear calculations
        # Solution and GS equilibrium are assigned to self.trial_currents and self.trial_plasma_psi
        self.set_linear_solution(
            active_voltage_vec=active_voltage_vec,
            dtheta_dt=dtheta_dt,
        )

        # check Matrix is still applicable
        myy_flag = self.handleMyy.check_Myy(self.hatIy)

        if linear_only:
            # assign currents and plasma flux to self.currents_vec, self.eq1 and self.profiles1 and complete step
            self.step_complete_assign(working_relative_tol_GS, from_linear=True)
            if myy_flag:
                print(
                    "The plasma used for calculating the adopted linearization and the plasma in this evolution have departed by more than",
                    self.handleMyy.tolerance,
                    "domain pixels. The linearization may not be accurate.",
                )

        else:
            # seek solution of the full nonlinear problem

            if myy_flag:
                if verbose:
                    print("The Myy matrix is being recalculated.")
                # recalculate Myy
                self.handleMyy.force_build_Myy(self.hatIy)

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
            iterations = 0

            while control and (iterations < max_solving_iterations):
                if verbose:
                    for _ in log:
                        print(_)

                log = [self.text_nk_cycle.format(nkcycle=iterations)]

                # update plasma flux if trial_currents and plasma_flux exceedingly far from GS solution
                if control_GS:
                    self.NK.forward_solve(
                        self.eq2, self.profiles2, self.rtol_NK, suppress=True
                    )
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
                        # clip_quantiles=clip_quantiles,
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
                    # clip_quantiles=clip_quantiles,
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
                iterations += 1

            # convergence checks succeeded, complete step
            self.step_complete_assign(working_relative_tol_GS)

            # if max_iterations exceeded, print warning
            if iterations >= max_solving_iterations:
                print(f"Forward evolutive solve DID NOT CONVERGE.")
                self.converged = False
            else:
                print(f"Forward evolutive solve SUCCESS.")
                self.converged = True
            print(
                f"   Last max. relative currents change: {np.max(rel_curr_res):.2e} (vs. requested {target_relative_tol_currents:.2e})."
            )
            print(
                f"   Last max. relative flux change: {np.max(r_res_GS):.2e} (vs. requested {target_relative_tol_GS:.2e})."
            )
            print(
                f"   Iterations taken: {int(iterations)}/{int(max_solving_iterations)}."
            )

    def unstable_mode_deformations(self, starting_dI=50, rtol_NK=1e-7, target_dIy=2e-3):
        """
        Applies the first unstable mode to evaluate plasma centroid deformations and
        the corresponding plasma current distribution response.

        This method calculates the derivatives of the current-averaged plasma coordinates
        (R, Z) with respect to the magnitude of the current in the unstable mode (Im),
        i.e., dR/dIm and dZ/dIm. It also records the plasma current distribution after
        applying Im and constructs a "rigidly displaced" version of the original current
        map for comparison.

        Parameters
        ----------
        starting_dI : float, optional
            Initial perturbation amplitude in the unstable mode used to estimate the slope
            of ||delta(Iy)|| / delta(Im). Default is 50.
        rtol_NK : float, optional
            Relative tolerance to be used in the static Grad-Shafranov (GS) problem.
            Default is 1e-7.
        target_dIy : float, optional
            Target norm of the plasma current change vector (delta(Iy)) used to scale
            the mode perturbation. Default is 2e-3.

        Attributes Updated
        ------------------
        dRZd_unstable_mode : np.ndarray
            Array [dR/dIm, dZ/dIm] representing the sensitivity of the plasma centroid
            to the applied unstable mode current.
        deformable_vs_rigid_jtor : tuple of np.ndarray
            Tuple containing:
            - The plasma current distribution with the unstable mode applied.
            - The "rigidly displaced" plasma current distribution obtained by shifting
            the original distribution to match the new centroid (R, Z) positions.
        """

        # apply self.linearised_sol.unstable_modes[:,0] shift to the currents
        # so that the Iy vector changes by a target_dIy relative change
        current_ = np.copy(self.currents_vec)

        current_[:-1] += starting_dI * np.real(self.linearised_sol.unstable_modes[:, 0])
        self.assign_currents_solve_GS(current_, rtol_NK)

        dIy_0 = self.limiter_handler.Iy_from_jtor(self.profiles2.jtor) - self.Iy

        rel_ndIy_0 = np.linalg.norm(dIy_0) / self.nIy
        final_dI = starting_dI * target_dIy / rel_ndIy_0

        current_ = np.copy(self.currents_vec)
        current_[:-1] += final_dI * np.real(self.linearised_sol.unstable_modes[:, 0])
        self.assign_currents_solve_GS(current_, rtol_NK)

        # calculcate resulting positions
        R0n = self.eq2.Rcurrent()
        Z0n = self.eq2.Zcurrent()

        self.dRZd_unstable_mode = np.array([R0n - self.R0, Z0n - self.Z0]) / final_dI

        # build vector of coordinates as needed by bilint
        # to 'shift' the original jtor so to match R0n and Z0n
        grid = np.concatenate(
            (
                self.eqR[:, :, np.newaxis] - R0n + self.R0,
                self.eqZ[:, :, np.newaxis] - Z0n + self.Z0,
            ),
            axis=-1,
        )

        # shift the original current
        shifted_current = bilinear_interpolation.biliint(
            self.eqR, self.eqZ, self.eq1._profiles.jtor, grid.reshape(-1, 2).T
        )

        self.deformable_vs_rigid_jtor = (
            self.eq2._profiles.jtor,
            shifted_current.reshape(self.nx, self.ny),
        )

    def calculate_Leuer_parameter(self):
        """
        Calculates the Leuer stability parameter for the plasma, which quantifies
        the passive vertical stability provided by surrounding metals and active coils.

        The Leuer parameter, as defined in Leuer (1989, "Passive Vertical Stability in the Next Generation Tokamaks",
        Eq. 6), is the ratio of stabilising force gradients induced by metals to
        the de-stabilising force gradients caused by the plasma and coil currents.

        The calculation involves:
        - Mutual inductance derivatives between coils and plasma points (first and second z-derivatives).
        - Mutual inductances between metal coils themselves.
        - Plasma current distribution and coil currents.
        - Stabilising forces (from active and passive coils) and de-stabilising forces.
        - Computation of Leuer parameters in different configurations.

        Attributes Updated
        ------------------
        actives_stab_force : float
            Stabilising force due to active coils.
        passives_stab_force : float
            Stabilising force due to passive coils.
        all_coils_stab_force : float
            Stabilising force due to all metal coils (active + passive).

        actives_destab_force : float
            De-stabilising force due to active coils.
        passives_destab_force : float
            De-stabilising force due to passive coils.
        all_coils_destab_force : float
            De-stabilising force due to all metal coils.

        Leuer_passive_stab_over_active_destab : float
            Ratio of passive stabilising force to active coil de-stabilising force.
        Leuer_metals_stab_over_active_destab : float
            Ratio of total metal stabilising force to active coil de-stabilising force.
        Leuer_metals_stab_over_metals_destab : float
            Ratio of total metal stabilising force to total metal de-stabilising force.

        References
        ----------
        Leuer, J. A., "Passive Vertical Stability in the Next Generation Tokamaks," 1989,
        10.13182/FST89-A39747.
        """

        # calculate z derivative of mutual inductance matrix (between coils and plasma points)
        M_prime_all_coils_plasma = self.M_coils_plasma(eq=self.eq1, greens=GreensBr)
        M_prime_actives_plasma = M_prime_all_coils_plasma[0 : self.n_active_coils, :]
        M_prime_passives_plasma = M_prime_all_coils_plasma[self.n_active_coils :, :]

        # extract mutual inductances between metals
        M_coils_coils = self.evol_metal_curr.coil_self_ind
        M_actives_active = M_coils_coils[
            0 : self.n_active_coils, 0 : self.n_active_coils
        ]
        M_passives_passives = M_coils_coils[
            self.n_active_coils :, self.n_active_coils :
        ]

        # calculate second z derivative of mutual inductance matrix (between coils and plasma points)
        M_prime_prime_all_coils_plasma = self.M_coils_plasma(
            eq=self.eq1, greens=GreensdBrdz
        )
        M_prime_prime_actives_plasma = M_prime_prime_all_coils_plasma[
            0 : self.n_active_coils, :
        ]
        M_prime_prime_passives_plasma = M_prime_prime_all_coils_plasma[
            self.n_active_coils :, :
        ]

        # extract plasma current density vector and currents in the metals
        I_all_coils = self.eq1.tokamak.getCurrentsVec()
        I_actives = I_all_coils[0 : self.n_active_coils]
        I_passives = I_all_coils[self.n_active_coils :]

        # calculate stabilising force gradients created by actives, passives, and all metals
        self.actives_stab_force = (
            self.Iy @ M_prime_actives_plasma.T
        ) @ np.linalg.solve(M_actives_active, M_prime_actives_plasma @ self.Iy)
        self.passives_stab_force = (
            self.Iy @ M_prime_passives_plasma.T
        ) @ np.linalg.solve(M_passives_passives, M_prime_passives_plasma @ self.Iy)
        self.all_coils_stab_force = (
            self.Iy @ M_prime_all_coils_plasma.T
        ) @ np.linalg.solve(M_coils_coils, M_prime_all_coils_plasma @ self.Iy)

        # calculate de-stabilising force gradients created by actives, passives, and all metals
        self.actives_destab_force = self.Iy @ (
            M_prime_prime_actives_plasma.T @ I_actives
        )
        self.passives_destab_force = self.Iy @ (
            M_prime_prime_passives_plasma.T @ I_passives
        )
        self.all_coils_destab_force = self.Iy @ (
            M_prime_prime_all_coils_plasma.T @ I_all_coils
        )

        # use these to return the Leuer parameters in different cases
        self.Leuer_passive_stab_over_active_destab = (
            self.passives_stab_force / self.actives_destab_force
        )
        self.Leuer_metals_stab_over_active_destab = (
            self.all_coils_stab_force / self.actives_destab_force
        )
        self.Leuer_metals_stab_over_metals_destab = (
            self.all_coils_stab_force / self.all_coils_destab_force
        )

    def M_coils_plasma(
        self,
        eq,
        greens,
    ):
        """
        Calculates the mutual inductance matrix between all tokamak coils and plasma grid points.

        The matrix represents either the first or second derivative of the mutual inductance
        with respect to the vertical coordinate z, depending on the Greens function provided.
        This is used in stability and force calculations involving coil-plasma interactions.

        Parameters
        ----------
        eq : FreeGSNKE equilibrium Object
            The equilibrium containing plasma and tokamak coil information.
        greens : function
            The Greens function used for the calculation:
            - "GreensBr" for the first z-derivative of the magnetic field.
            - "GreensdBrdz" for the second z-derivative of the magnetic field.

        Returns
        -------
        M : np.ndarray, shape (n_coils, n_plasma_pts)
            Mutual inductance matrix between each coil (rows) and plasma grid point (columns),
            including coil polarity, multipliers, and R-coordinate weighting. The returned matrix
            is multiplied by -2π as per the standard formulation.

        Notes
        -----
        - `plasma_pts` are taken from `eq.limiter_handler.plasma_pts`, i.e., the reduced plasma domain.
        - Coil contributions are summed over all filaments in each coil.
        """

        # plasma grid points (inside limiter)
        plasma_pts = eq.limiter_handler.plasma_pts

        # create mutual inductance matrix
        M = np.zeros((len(eq.tokamak.coils_list), len(plasma_pts)))
        for j, labelj in enumerate(eq.tokamak.coils_list):
            greenm = greens(
                plasma_pts[:, 0, np.newaxis],
                plasma_pts[:, 1, np.newaxis],
                eq.tokamak.coils_dict[labelj]["coords"][0][np.newaxis, :],
                eq.tokamak.coils_dict[labelj]["coords"][1][np.newaxis, :],
            )
            greenm *= eq.tokamak.coils_dict[labelj]["polarity"][
                np.newaxis, :
            ]  # mulitply by polarity of filaments
            greenm *= eq.tokamak.coils_dict[labelj]["multiplier"][
                np.newaxis, :
            ]  # mulitply by mulitplier of filaments
            greenm *= eq.tokamak.coils_dict[labelj]["coords"][0][
                np.newaxis, :
            ]  # multiply by R co-ords
            M[j] = np.sum(greenm, axis=-1)  # sum over filaments

        return -2 * np.pi * M
