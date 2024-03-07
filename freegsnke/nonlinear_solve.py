from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from . import machine_config
from . import nk_solver as nk_solver
from . import plasma_grids
from .circuit_eq_metal import metal_currents
from .circuit_eq_plasma import plasma_current
from .GSstaticsolverH import NKGSsolver
from .linear_solve import linear_solver
from .machine_config import coils_order
from .simplified_solve import simplified_solver_J1

# from . import extrapolate


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
        plasma_domain_mask=None,
        nbroad=1,
        blend_hatJ=0,
        dIydI=None,
        dIydpars=None,
        automatic_timestep=False,
        mode_removal=True,
        min_dIy_dI=1,
        verbose=False,
    ):
        """Initializes the time-evolution Object.

        Parameters
        ----------
        profiles : FreeGS profile Object
            Profile function of the initial equilibrium.
            This will be used to set up the linearization used by the linear solver.
            It can be changed later by initializing a new set of initial conditions.
        eq : FreeGS equilibrium Object
            Initial equilibrium. This is used to set the domain/grid properties
            as well as the tokamak/machine properties.
            Furthermore, eq will be used to set up the linearization used by the linear solver.
            It can be changed later by initializing a new set of initial conditions.
            Note however that, to change either the domain or tokamak properties
            it will be necessary to instantiate a new evolution object.
        max_mode_frequency : float
            Threshold value used to include/exclude vessel normal modes.
            Only modes with smaller characteristic frequencies (larger timescales) are retained.
            If None, max_mode_frequency is set based on the input timestep: max_mode_frequency = 1/(5*full_timestep)
        full_timestep : float, optional, by default .0001
            The stepper advances the dynamics by a time interval dt=full_timestep.
            Applies to both linear and non-linear stepper.
            A GS equilibrium is calculated every full_timestep.
            Note that this input is overridden by automatic_timestep if the latter is not False.
        max_internal_timestep : float, optional, by default .0001
            Each time advancement of one full_timestep is divided in several sub-steps,
            with size of, at most, max_internal_timestep.
            Such sub_step is used to advance the circuit equations
            (under the assumption of constant applied voltage and linearly evolving plasma).
            Note that this input is overridden by automatic_timestep if the latter is not False.
        plasma_resistivity : float, optional, by default 1e-6
            Resistivity of the plasma. Plasma resistance values for each of the domain grid points are
            2*np.pi*plasma_resistivity*eq.R/(dR*dZ)
            where dR*dZ is the area of the domain element.
        plasma_norm_factor : float, optional, by default 1000
            The lumped circuit eq relating to the plasma is divided by this factor,
            to make differences with metal current values less extreme.
        plasma_domain_mask : np.array of size (eq.nx,eq.ny)
            mask of grid domain points to be included in the current circuit equation.
            This reduces the dimensionality of associated matrices.
            If None, the mask defaults to the one based on the limiter associated to the input profiles.
            Only specify if a different mask is required.
        nbroad : int, optional, by default 3
            pixel size (as number of grid points) of (square) smoothing filter applied to
            the instantaneous plasma current distribution, before contracting the plasma circuit equations
        blend_hatJ : float, optional, by default 0
            optional coefficient to use a blended version of the normalised plasma current distribution
            when contracting the plasma lumped circuit eq. from the left. The blend combines the
            current distribution at time t with (a guess for) the one at time t+dt.
        dIydI : np.array of size (np.sum(plasma_domain_mask), n_metal_modes+1), optional
            dIydI_(i,j) = d(Iy_i)/d(I_j)
            This is the jacobian of the plasma current distribution with respect to all
            independent metal currents (both active and vessel modes) and to the total plasma current
        automatic_timestep : (float, float) or False, optional, by default False
            If not False, this overrides inputs full_timestep and max_internal_timestep:
            the timescales of the linearised problem are used to set the size of the timestep.
            The input eq and profile are used to calculate the fastest growthrate, t_growthrate, henceforth,
            full_timestep = automatic_timestep[0]*t_growthrate
            max_internal_timestep = automatic_timestep[1]*full_timestep
        mode_removal : bool, optional, by default True
            It True, vessel normal modes that have little influence on the plasma are dropped.
            Criterion is based on norm of d(Iy)/dI of the mode.
        min_dIy_dI : float, optional, by default 1
            Threshold value to include/drop vessel normal modes.
            Modes with norm(d(Iy)/dI)<min_dIy_dI are dropped.
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

        # instantiating static GS solver (Newton-Krylov) on eq's domain
        self.NK = NKGSsolver(eq)

        # setting up domain for plasma circuit eq.:
        if plasma_domain_mask is None:
            plasma_domain_mask = profiles.mask_inside_limiter
            self.plasma_grids = profiles.plasma_grids
            self.plasma_domain_mask = self.plasma_grids.plasma_domain_mask
        else:
            if plasma_domain_mask != profiles.mask_inside_limiter:
                print(
                    "A plasma_domain_mask different from the profiles.mask_inside_limiter has been provided."
                )
                self.plasma_grids = plasma_grids.Grids(eq, plasma_domain_mask)
                self.plasma_domain_mask = plasma_domain_mask
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

        # handles the metal circuit eq, mode properties, and can calculate residual of metal circuit eq
        self.evol_metal_curr = metal_currents(
            flag_vessel_eig=1,
            flag_plasma=1,
            plasma_grids=self.plasma_grids,
            max_mode_frequency=self.max_mode_frequency,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.dt_step,
        )
        # this is the number of independent normal mode currents being used
        self.n_metal_modes = self.evol_metal_curr.n_independent_vars
        self.arange_currents = np.arange(self.n_metal_modes + 1)

        # to calculate residual of plasma contracted circuit eq
        self.evol_plasma_curr = plasma_current(
            plasma_grids=self.plasma_grids,
            Rm12=np.diag(self.evol_metal_curr.Rm12),
            V=self.evol_metal_curr.V,
            plasma_resistance_1d=self.plasma_resistance_1d,
            Mye=self.evol_metal_curr.Mey_matrix.T,
        )

        # This solves the system of circuit eqs based on an assumption
        # for the direction of the plasma current distribution at time t+dt
        # Note that this does not use sub-time-stepping, i.e. max_internal_timestep = full_timestep
        self.simplified_solver_J1 = simplified_solver_J1(
            Lambdam1=self.evol_metal_curr.Lambdam1,
            Vm1Rm12=np.matmul(
                self.evol_metal_curr.Vm1, np.diag(self.evol_metal_curr.Rm12)
            ),
            Mey=self.evol_metal_curr.Mey_matrix,
            Myy=self.evol_plasma_curr.Myy_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            max_internal_timestep=self.dt_step,
            full_timestep=self.dt_step,
        )

        # self.vessel_currents_vec is the vector of tokamak coil currents (not normal modes)
        # initial self.vessel_currents_vec values are taken from eq.tokamak
        # does not include plasma current
        vessel_currents_vec = np.zeros(machine_config.n_coils)
        eq_currents = eq.tokamak.getCurrents()
        for i, labeli in enumerate(coils_order):
            vessel_currents_vec[i] = eq_currents[labeli]
        self.vessel_currents_vec = 1.0 * vessel_currents_vec

        # self.currents_vec is the vector of current values in which the dynamics is actually solved for
        # it includes: active coils, vessel normal modes, total plasma current
        # total plasma current is divided by plasma_norm_factor to improve homogeneity of float values
        self.extensive_currents_dim = self.n_metal_modes + 1
        self.currents_vec = np.zeros(self.extensive_currents_dim)
        self.circuit_eq_residual = np.zeros(self.extensive_currents_dim)

        # self.linearised_sol handles the linearised dynamic problem
        self.linearised_sol = linear_solver(
            Lambdam1=self.evol_metal_curr.Lambdam1,
            Vm1Rm12=np.matmul(
                self.evol_metal_curr.Vm1, np.diag(self.evol_metal_curr.Rm12)
            ),
            Mey=self.evol_metal_curr.Mey_matrix,
            Myy=self.evol_plasma_curr.Myy_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.dt_step,
        )

        # set up NK solver on the full grid, to be used when solving for the plasma flux
        self.psi_nk_solver = nk_solver.nksolver(self.nxny, verbose=True)

        # set up NK solver for the currents
        self.currents_nk_solver = nk_solver.nksolver(
            self.extensive_currents_dim, verbose=True
        )  # , verbose=True)

        # set up unique NK solver for the full vector of unknowns
        self.full_nk_solver = nk_solver.nksolver(
            self.extensive_currents_dim + self.nxny, verbose=True
        )

        # step advancement of the dynamics
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

        # self.dIydpars is the Jacobian of the plasma current distribution
        # with respect to the independent profile parameters (alpha_m, alpha_n, paxis OR betap)
        self.dIydpars_ICs = dIydpars
        self.dIydpars = dIydpars

        self.get_number_of_profile_pars(profiles)
        self.get_number_of_independent_pars()

        # initialize and set up the linearization
        # input value for dIydI is used when available
        # no noise is added to normal modes
        if automatic_timestep or mode_removal:
            self.initialize_from_ICs(
                eq, profiles, rtol_NK=1e-9, noise_level=0, dIydI=dIydI, verbose=verbose
            )

        # remove passive normal modes that do not affect the plasma
        if mode_removal:
            self.selected_modes_mask = np.linalg.norm(self.dIydI, axis=0) > min_dIy_dI
            self.selected_modes_mask = np.concatenate(
                (
                    np.ones(machine_config.n_active_coils),
                    self.selected_modes_mask[machine_config.n_active_coils : -1],
                    np.ones(1),
                )
            ).astype(bool)
            self.dIydI = self.dIydI[:, self.selected_modes_mask]
            self.updated_dIydI = np.copy(self.dIydI)
            self.ddIyddI = self.ddIyddI[self.selected_modes_mask]
            self.selected_modes_mask = np.concatenate(
                (
                    self.selected_modes_mask[:-1],
                    np.zeros(machine_config.n_coils - self.n_metal_modes),
                )
            ).astype(bool)
            self.remove_modes(self.selected_modes_mask)

        # check if input equilibrium and associated linearization have an instability, and its timescale
        self.linearised_sol.calculate_linear_growth_rate()
        if len(self.linearised_sol.growth_rates):
            print(
                "This equilibrium has a linear growth rate of 1/",
                abs(self.linearised_sol.growth_rates[0]),
                "s",
            )
        else:
            print(
                "No unstable modes found.",
                "Either plasma is stable or, more likely, it is Alfven unstable (i.e. not enough stabilization from any metal structures).",
                "Try adding more passive modes by resetting the input values of max_mode_frequency and/or min_dIy_dI.",
            )

        # if automatic_timestep, reset the timestep accordingly,
        # note that this requires having found an instability
        if automatic_timestep is None or automatic_timestep is False:
            print(
                "The solver's timestep was set at",
                self.dt_step,
                " as explicitly requested. Please compare this with the linear growth rate above and, if necessary, reset.",
            )
        else:
            if len(self.linearised_sol.growth_rates):
                dt_step = abs(
                    self.linearised_sol.growth_rates[0] * automatic_timestep[0]
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

        # prepare regularization matrices
        self.reg_matrix = (
            np.eye(self.plasma_domain_size)
            + self.plasma_grids.build_linear_regularization()
        )  # + self.plasma_grids.build_quadratic_regularization()
        # self.reg0 = np.eye(self.plasma_domain_size)
        # self.reg1 = self.plasma_grids.build_linear_regularization()
        # self.reg2 = self.plasma_grids.build_quadratic_regularization()

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
            "Mode removal is ON: the input min_dIy_dI corresponds to keeping",
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
        self.full_nk_solver = nk_solver.nksolver(
            self.extensive_currents_dim + self.nxny
        )

        self.evol_plasma_curr.reset_modes(V=self.evol_metal_curr.V)

        self.simplified_solver_J1 = simplified_solver_J1(
            Lambdam1=self.evol_metal_curr.Lambdam1,
            Vm1Rm12=np.matmul(
                self.evol_metal_curr.Vm1, np.diag(self.evol_metal_curr.Rm12)
            ),
            Mey=self.evol_metal_curr.Mey_matrix,
            Myy=self.evol_plasma_curr.Myy_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            # this is used with no internal step subdivision, to help nonlinear convergence
            max_internal_timestep=self.dt_step,
            full_timestep=self.dt_step,
        )

        self.linearised_sol = linear_solver(
            Lambdam1=self.evol_metal_curr.Lambdam1,
            Vm1Rm12=self.simplified_solver_J1.Vm1Rm12,
            Mey=self.evol_metal_curr.Mey_matrix,
            Myy=self.evol_plasma_curr.Myy_matrix,
            plasma_norm_factor=self.plasma_norm_factor,
            plasma_resistance_1d=self.plasma_resistance_1d,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.dt_step,
        )

        self.linearised_sol.set_linearization_point(
            dIydI=self.dIydI, dIydpars=self.dIydpars, hatIy0=self.broad_hatIy
        )

        self.get_number_of_independent_pars()

    def set_linear_solution(self, active_voltage_vec, d_profile_pars_dt=None):
        """Uses the solver of the linearised problem to set up an initial guess
        for the currents at time t+dt. Uses self.currents_vec as I(t).
        Solves GS at time t+dt for the corresponding guessed currents.

        Parameters
        ----------
        active_voltage_vec : np.array
            Vector of external voltage applied to the active coils during the timestep.
        """

        self.trial_currents = self.linearised_sol.stepper(
            It=self.currents_vec,
            active_voltage_vec=active_voltage_vec,
            d_profile_pars_dt=d_profile_pars_dt,
        )
        self.assign_currents_solve_GS(self.trial_currents, self.rtol_NK)
        self.trial_plasma_psi = np.copy(self.eq2.plasma_psi)

    def prepare_build_dIydpars(self, profiles, rtol_NK, target_dIy, starting_dpars):
        """Prepares to compute the term d(Iy)/d(alpha_m, alpha_n, profifile_par)
        where profile_par = paxis or betap.
        It infers the value of delta(indep_variable) corresponding to a change delta(I_y)
        with norm(delta(I_y))=target_dIy.

        Parameters
        ----------
        profiles : FreeGS profile object
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
        dIy_0 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        self.final_dpars_record[0] = (
            starting_dpars[0] * target_dIy / np.linalg.norm(dIy_0)
        )

        # vary alpha_n
        self.check_and_change_profiles(
            profile_coefficients=(
                profiles.alpha_m,
                profiles.alpha_n + starting_dpars[1],
            )
        )
        self.assign_currents_solve_GS(current_, rtol_NK)
        dIy_0 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        self.final_dpars_record[1] = (
            starting_dpars[1] * target_dIy / np.linalg.norm(dIy_0)
        )

        # vary paxis or betap
        self.check_and_change_profiles(
            profile_coefficients=(profiles.alpha_m, profiles.alpha_n),
            profile_parameter=(1 + starting_dpars[2]) * profiles.profile_parameter,
        )
        self.assign_currents_solve_GS(current_, rtol_NK)
        dIy_0 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        self.final_dpars_record[2] = (
            starting_dpars[2]
            * profiles.profile_parameter
            * target_dIy
            / np.linalg.norm(dIy_0)
        )

    def build_dIydIpars(self, profiles, rtol_NK, verbose=False):
        """Compute the matrix d(Iy)/d(alpha_m, alpha_n, profifile_par) as a finite difference derivative,
        using the value of delta(indep_viriable) inferred earlier by self.prepare_build_dIypars.

        Parameters
        ----------
        profiles : FreeGS profile object
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
        dIy_1 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
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
        dIy_1 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
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
        dIy_1 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        self.dIydpars[:, 2] = dIy_1 / self.final_dpars_record[2]
        if verbose:
            print(
                "profile_par gradient calculated on the finite difference: delta_profile_par =",
                self.final_dpars_record[2],
                ", norm(deltaIy) =",
                np.linalg.norm(dIy_1),
            )

    def prepare_build_dIydI_j(
        self, j, rtol_NK, target_dIy, starting_dI, min_curr=1e-4, max_curr=10
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
        max_curr : int, optional, by default 10
            If inferred current value is above min_curr, clip to max_curr.
        """
        current_ = np.copy(self.currents_vec)
        current_[j] += starting_dI
        self.assign_currents_solve_GS(current_, rtol_NK)

        dIy_0 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        final_dI = starting_dI * target_dIy / np.linalg.norm(dIy_0)
        final_dI = np.clip(final_dI, min_curr, max_curr)
        self.final_dI_record[j] = final_dI

    def build_dIydI_j(self, j, rtol_NK, verbose=False):
        """Compute the term d(Iy)/dI_j of the Jacobian as a finite difference derivative,
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

        final_dI = self.final_dI_record[j]
        self.current_at_last_linearization[j] = self.currents_vec[j]

        current_ = np.copy(self.currents_vec)
        current_[j] += final_dI
        self.assign_currents_solve_GS(current_, rtol_NK)

        dIy_1 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        dIydIj = dIy_1 / final_dI

        # noise = noise_level*np.random.random(self.n_metal_modes+1)
        # current_[j] += final_dI
        # self.assign_currents_solve_GS(current_, rtol_NK)
        # dIydIj_2 = (self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - dIy_1 - self.Iy)/final_dI

        # self.ddIyddI[j] = np.linalg.norm(dIydIj_2 - dIydIj)/final_dI

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
        dIydpars=None,
        rtol_NK=1e-8,
        target_dIy=10.0,
        starting_dI=0.5,
        starting_dpars=(0.0002, 0.0002, 0.005),
        verbose=False,
    ):
        """Builds the Jacobian d(Iy)/dI to set up the solver of the linearised problem.

        Parameters
        ----------
        eq : FreeGS equilibrium Object
            Equilibrium around which to linearise.
        profile : FreeGS profile Object
            Profile properties of the equilibrium around which to linearise.
        dIydI : np.array
            input Jacobian, enter where available, otherwise this will be calculated here
        dIydpars : np.array
            input Jacobian, enter where available, otherwise this will be calculated here
        rtol_NK : float
            Relative tolerance to be used in the static GS problems.
        target_dIy : float, by default 10.
            Target value for the norm of delta(I_y), on which th finite difference derivative is calculated.
        starting_dI : float, by default .5.
            Initial value to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        starting_dpars : tuple (d_alpha_m, d_alpha_n, relative_d_profile_par)
            Initial value to be used as delta(indep_variable) to infer the slope of norm(delta(I_y))/delta(indep_variable).
            Note that the first two values in the tuple are absolute deltas,
            while the third value is relative, d_profile_par = relative_d_profile_par * profile_par
        """

        if ((dIydI is None) and (self.dIydI is None)) or (
            (dIydpars is None) and (self.dIydpars is None)
        ):
            self.NK.forward_solve(eq, profile, target_relative_tolerance=rtol_NK)
            self.build_current_vec(eq, profile)
            self.Iy = self.plasma_grids.Iy_from_jtor(profile.jtor).copy()

        # build/update dIydI
        if dIydI is None:
            if self.dIydI_ICs is None:
                print(
                    "I'm building the linearization wrt the currents. This may take a minute or two."
                )
                self.dIydI = np.zeros((self.plasma_domain_size, self.n_metal_modes + 1))
                self.ddIyddI = np.zeros(self.n_metal_modes + 1)
                self.final_dI_record = np.zeros(self.n_metal_modes + 1)

                for j in self.arange_currents:
                    self.prepare_build_dIydI_j(j, rtol_NK, target_dIy, starting_dI)

                for j in self.arange_currents:
                    self.dIydI[:, j] = self.build_dIydI_j(j, rtol_NK, verbose)
                # self.norm_updated_dIydI = np.linalg.norm(self.updated_dIydI)

        else:
            self.dIydI = dIydI
        self.dIydI_ICs = np.copy(self.dIydI)

        # build/update dIydpars
        # Note this assumes 3 free profile parameters at the moment!
        if dIydpars is None:
            if self.dIydpars_ICs is None:
                print("I'm building the linearization wrt the profile parameters.")
                self.dIydpars = np.zeros((self.plasma_domain_size, 3))
                self.final_dpars_record = np.zeros(3)

                self.prepare_build_dIydpars(
                    profile, rtol_NK, target_dIy, starting_dpars
                )
                self.build_dIydIpars(profile, rtol_NK, verbose)

        else:
            self.dIydpars = dIydpars
        self.dIydpars_ICs = np.copy(self.dIydpars)

    def set_plasma_resistivity(self, plasma_resistivity):
        """Function to set the resistivity of the plasma.
        self.plasma_resistance_1d is the diagonal of the matrix R_yy, the plasma resistance matrix.
        Note that it only spans the grid points in the reduced domain, as from plasma_domain_mask.
        Changes to the plasma sensitivity require changes to all objects that require a plasma_resistance_1d input!

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
        """Allows for a resets the timesteps.

        Parameters
        ----------
        full_timestep : float
            The stepper advances the dynamics by a time interval dt=full_timestep.
            Applies to both linear and non-linear stepper.
            A GS equilibrium is calculated every full_timestep.
            Note that this input is overridden by automatic_timestep if the latter is not False.
        max_internal_timestep : float
            Each time advancement of one full_timestep is divided in several sub-steps,
            with size of, at most, max_internal_timestep.
            Such sub_step is used to advance the circuit equations
            (under the assumption of constant applied voltage and linearly evolving plasma).
        """
        self.dt_step = full_timestep
        self.max_internal_timestep = max_internal_timestep

        self.evol_metal_curr.reset_mode(
            flag_vessel_eig=1,
            flag_plasma=1,
            plasma_grids=self.plasma_grids,
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
        profiles : FreeGS profile Object
            Profile function of the initial equilibrium.
            Can handle both freeGS profile types ConstrainPaxisIp and ConstrainBetapIp.
        """
        self.fvac = profiles.fvac
        self.alpha_m = profiles.alpha_m
        self.alpha_n = profiles.alpha_n
        if hasattr(profiles, "paxis"):
            self.profile_parameter = profiles.paxis
        else:
            self.profile_parameter = profiles.betap

    def get_vessel_currents(self, eq):
        """Uses the input equilibrium to extract values for all metal currents,
        including active coils and vessel passive structures.
        These are stored in self.vessel_currents_vec

        Parameters
        ----------
        eq : FreeGS equilibrium Object
            Initial equilibrium. eq.tokamak is used to extract current values.
        """
        eq_currents = eq.tokamak.getCurrents()
        for i, labeli in enumerate(coils_order):
            self.vessel_currents_vec[i] = eq_currents[labeli]

    def build_current_vec(self, eq, profile):
        """Builds the vector of currents in which the dynamics is solved, self.currents_vec
        This contains, in the order:
        (active coil currents, selected vessel normal modes currents, total plasma current/plasma_norm_factor)

        Parameters
        ----------
        profile : FreeGS profile Object
            Profile function of the initial equilibrium. Used to extract the value of the total plasma current.
        eq : FreeGS equilibrium Object
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

        self.currents_vec_m1 = np.copy(self.currents_vec)

    def assign_vessel_noise(self, eq, noise_level, noise_vec=None):
        """Adds noise to vessel passive currents and assigns to input equilibrium.
        Noise corresponds to gaussian noise with std=noise_level in the vessel normal modes.

        Parameters
        ----------
        eq : FreeGS equilibrium Object
            Equilibrium to which noise is added and assigned.
        noise_level : float
            Standard deviation of the noise in terms of vessel normal modes' currents.
        noise_vec : np.array, optional, by default None
            Vector of noise values to be added. If available, this is used instead of a random realization.
        """

        # Extracts metal currents and builds self.vessel_currents_vec
        self.get_vessel_currents(eq)

        # generate random noise on vessel normal modes only
        if noise_vec is None:
            noise_vec = np.random.randn(
                self.n_metal_modes - machine_config.n_active_coils
            )
            noise_vec *= noise_level
            noise_vec = np.concatenate(
                (np.zeros(machine_config.n_active_coils), noise_vec)
            )
            self.noise_vec = noise_vec

        # calculate vessel noise from noise_vec and assign
        self.vessel_currents_vec += self.evol_metal_curr.IdtoIvessel(Id=noise_vec)
        for i, labeli in enumerate(coils_order[machine_config.n_active_coils :]):
            # runs on passive only
            eq.tokamak[labeli].current = self.vessel_currents_vec[
                i + machine_config.n_active_coils
            ]

    def initialize_from_ICs(
        self,
        eq,
        profile,
        rtol_NK=1e-8,
        noise_level=0.0,
        noise_vec=None,
        dIydI=None,
        dIydpars=None,
        update_linearization=False,
        update_n_steps=16,
        threshold_svd=0.1,
        max_dIy_update=0.01,
        max_updates=6,
        verbose=False,
    ):
        """Uses the input equilibrium as initial conditions and prepares to solve for its dynamics.
        If needed, sets the the linearised solver by calculating the Jacobian dIy/dI.
        It also defines whether the linearization is updated during the evolution and the properties of such update.
        At the moment the linearization update needs work, keep update_linearization=False.

        Parameters
        ----------
        eq : FreeGS equilibrium Object
            Initial equilibrium. This assigns all initial metal currents.
        profiles : FreeGS profile Object
            Profile function of the initial equilibrium. This assigns the initial total plasma current.
            Note that this assigns the profile properties,
            for instance p_on_axis and profile coefficients (alpha_m, alpha_n).
            In current implementation these are kept constant during time evolution.
        rtol_NK : float
            Relative tolerance to be used in the static GS problems in the initialization.
            This includes when calculating the Jacobian dIy/dI to set up the linearised problem.
            Note that the tolerance used in static GS solves used by the dynamics is set through the stepper itself.
        noise_level : float, optional, by default .001
            If noise_level > 0, gaussian noise with std = noise_level is added to the
            passive structure currents.
        noise_vec : noise_vec : np.array, optional, by default None
            If not None, noise_vec is added to the metal current values extracted from eq.
            Structure of noise_vec is (active coils, selected passive normal modes).
        dIydI : np.array of size (np.sum(plasma_domain_mask), n_metal_modes+1), optional
            dIydI_(i,j) = d(Iy_i)/d(I_j)
            This is the jacobian of the plasma current distribution with respect to all
            independent metal currents (both active and vessel modes) and to the total plasma current
            If not provided, this is calculated based on the properties of the provided equilibrium.

            NOT USED AT THE MOMENT
        update_linearization : bool, optional, by default False
            Whether the linearization is updated as the dynamical evolution departs from the initial equilibrium.
        update_n_steps : int, optional, by default 16
            Use a number update_n_steps of previous evolution to inform the linearization update.
        threshold_svd : float, optional, by default .1
            Used in the linearization update. Threshold to avoid spurious numerical effects in matrix inversion.
        max_dIy_update : float, optional
            Used in the linearization update. Relative threshold value in norm(dIy/dI).
            The update is rejected if its norm is larger than max_dIy_update*norm(dIy/dI)
        max_updates : int, optional
            Maximum number of columns of the Jacobian d(Iy_i)/d(I_j) that are recalculated
            per timestep of the dynamical evolution.
        """

        self.step_counter = 0
        self.currents_guess = False
        self.rtol_NK = rtol_NK
        self.update_n_steps = update_n_steps
        self.update_linearization = update_linearization
        self.threshold_svd = threshold_svd
        self.max_dIy_update = max_dIy_update
        self.max_updates = max_updates

        # get profile parametrization
        self.get_profiles_values(profile)

        # set internal copy of the equilibrium
        # self.eq1 and self.profiles1 are advanced each timestep.
        # Their properties evolve according to the dynamics.
        # Note that the input eq and profile are NOT modified by the evolution object.
        self.eq1 = deepcopy(eq)
        self.profiles1 = deepcopy(profile)

        # Perturb passive structures when desired
        if (noise_level > 0) or (noise_vec is not None):
            self.assign_vessel_noise(self.eq1, noise_level, noise_vec)

        # ensure input equilibrium is a GS solution
        self.NK.forward_solve(
            self.eq1, self.profiles1, target_relative_tolerance=rtol_NK
        )

        # self.Iy is the istantaneous 1d vector representing the plasma current distribution
        # on the reduced plasma domain, as from plasma_domain_mask
        # self.Iy is updated every timestep
        self.Iy = self.plasma_grids.Iy_from_jtor(self.profiles1.jtor)
        # self.hatIy is the normalised version. The vector sums to 1.
        self.hatIy = self.plasma_grids.normalize_sum(self.Iy)
        # self.hatIy1 is the normalised plasma current distribution at time t+dt
        self.hatIy1 = np.copy(self.hatIy)
        self.make_broad_hatIy(self.hatIy1)
        # self.broad_hatIy is convolved with a broading filter.
        # self.broad_hatIy is used to contract the system of plasma circuit eqs.
        # self.broad_hatIy = convolve2d(self.profiles1.jtor, self.ones_to_broaden, mode='same')
        # self.broad_hatIy = self.plasma_grids.hat_Iy_from_jtor(self.broad_hatIy)

        # set an additional internal copy of the equilibrium
        # self.eq2 and self.profiles2 are used when solving for the dynamics
        # they should not be used to extract properties of the evolving equilibrium
        # as these may not be accurate
        self.eq2 = deepcopy(self.eq1)
        self.profiles2 = deepcopy(self.profiles1)

        # extract all initial current values currents
        self.build_current_vec(self.eq1, self.profiles1)
        self.current_at_last_linearization = np.copy(self.currents_vec)

        self.time = 0
        self.step_no = -1

        # build the linearization if not provided
        self.build_linearization(
            eq,
            profile,
            dIydI=dIydI,
            dIydpars=dIydpars,
            rtol_NK=rtol_NK,
            target_dIy=10.0,
            starting_dI=0.5,
            starting_dpars=(0.0008, 0.0008, 0.002),
            verbose=verbose,
        )

        # transfer linearization to linear solver
        self.linearised_sol.set_linearization_point(
            dIydI=self.dIydI_ICs, dIydpars=self.dIydpars_ICs, hatIy0=self.broad_hatIy
        )

        self.reset_records_for_linearization_update()

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

        # self.eq1.plasma_psi = np.copy(self.trial_plasma_psi)
        # self.profiles1.Ip = self.trial_currents[-1]*self.plasma_norm_factor
        if from_linear:
            self.profiles1 = deepcopy(self.profiles2)
            self.eq1 = deepcopy(self.eq2)
            # self.profiles1.jtor = np.copy(self.profiles2.jtor)
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

        self.Iy = self.plasma_grids.Iy_from_jtor(self.profiles1.jtor)
        self.hatIy = self.plasma_grids.normalize_sum(self.Iy)
        # self.broad_hatIy = convolve2d(self.profiles1.jtor, self.ones_to_broaden, mode='same')
        # self.broad_hatIy = self.plasma_grids.hat_Iy_from_jtor(self.broad_hatIy)

        self.rtol_NK = working_relative_tol_GS * self.d_plasma_psi_step

        if self.update_linearization:
            self.check_linearization_and_update(
                max_dIy_update=self.max_dIy_update, max_updates=self.max_updates
            )

    def assign_currents(self, currents_vec, eq, profile):
        """Assigns current values as in input currents_vec to tokamak and plasma.
        The input eq and profile are modified accordingly.
        The format of the input currents aligns with self.currents_vec:
        (active coil currents, selected vessel normal modes currents, total plasma current/plasma_norm_factor)

        Parameters
        ----------
        currents_vec : np.array
            Input current values to be assigned.
        eq : FreeGS equilibrium Object
            Equilibrium object to be modified.
        profiles : FreeGS profile Object
            Profile object to be modified.
        """

        # assign plasma current to equilibrium
        eq._current = self.plasma_norm_factor * currents_vec[-1]
        profile.Ip = self.plasma_norm_factor * currents_vec[-1]

        # calculate vessel currents from normal modes and assign
        self.vessel_currents_vec = self.evol_metal_curr.IdtoIvessel(
            Id=currents_vec[:-1]
        )
        for i, labeli in enumerate(coils_order):
            eq.tokamak[labeli].current = self.vessel_currents_vec[i]

    def assign_currents_solve_GS(self, currents_vec, rtol_NK, record_for_updates=False):
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
        if record_for_updates:
            self.record_for_update(currents_vec, self.profiles2)

    def record_for_update(self, currents_vec, profiles):
        """Appends new GS solution to record of independent variables
        and record of reduced Iy distributions.

        Parameters
        ----------
        currents_vec : np.array
            vector of all extensive currents (normal modes + plasma current)
        profiles : freeGS profile obj
            profile used to build the equilibrium to be recorded
        """
        self.record_currents_pars = np.vstack(
            (
                self.record_currents_pars,
                self.build_current_pars_vec(currents_vec, profiles),
            )
        )
        self.record_Iys = np.vstack(
            (self.record_Iys, self.plasma_grids.Iy_from_jtor(profiles.jtor))
        )

    def get_number_of_independent_pars(
        self,
    ):
        """Queries the profile function and the metal modes
        to establish the number of independent variables to
        the GS equilibrium.
        """
        self.number_of_independent_pars = self.n_metal_modes + 1 + self.n_profile_pars

    def get_number_of_profile_pars(self, profiles):
        """Queries the profile function to establish the number of independent parameters."""
        self.n_profile_pars = np.size(profiles.get_pars())

    def reset_records_for_linearization_update(
        self,
    ):
        """Resets the recod vectors used for building the update to the linearization matrices."""
        self.record_currents_pars = np.array([], dtype=np.float32).reshape(
            0, self.number_of_independent_pars
        )
        self.record_Iys = np.array([], dtype=np.float32).reshape(
            0, self.plasma_domain_size
        )

    def build_current_pars_vec(self, currents_vec, profiles):
        """Builds vector with full list of independent variables,
        used in the linearization update.

        Parameters
        ----------
        currents_vec : np.array
            Input current values to be assigned. Format as in self.assign_currents.
        profiles : FreeGS profile Object
            profile from which to fetch the profile parameters

        Returns
        -------
        np.array
            Input current values and profile parameter values in a single vector.
        """
        currents_pars_vec = np.concatenate((currents_vec, profiles.get_pars()))
        return currents_pars_vec

    def run_linearization_update(self, min_records=160, lambda_reg=10, threshold=0.08):
        """Updates the linearization

        Parameters
        ----------
        min_records: int
            minimum number of GS solution required to trigger an update
        lambda_reg : float, optional
            regularization coefficient, by default 1e-3
        """
        if np.shape(self.record_Iys)[0] > min_records:
            print("Im updating the linearization!")
            self.linearised_sol.find_linearization_update(
                self.record_currents_pars,
                self.record_Iys,
                lambda_reg * self.reg_matrix,
                threshold,
            )
            self.linearised_sol.apply_linearization_update()
            self.dIydI = self.linearised_sol.dIydI.copy()
            self.dIydpars = self.linearised_sol.dIydpars.copy()

            self.make_broad_hatIy(self.hatIy)
            self.linearised_sol.set_linearization_point(
                self.linearised_sol.dIydI,
                self.linearised_sol.dIydpars,
                self.broad_hatIy,
            )

            self.reset_records_for_linearization_update()

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
        self.broad_hatIy = self.plasma_grids.rebuild_map2d(hatIy1 + blend * self.hatIy)
        self.broad_hatIy = convolve2d(
            self.broad_hatIy, self.ones_to_broaden, mode="same"
        )
        self.broad_hatIy = self.plasma_grids.hat_Iy_from_jtor(self.broad_hatIy)

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
        self.broad_hatIy = self.plasma_grids.rebuild_map2d(hatIy1 + blend * self.hatIy)
        # self.broad_hatIy = convolve2d(self.broad_hatIy, self.ones_to_broaden, mode='same')
        self.broad_hatIy = self.plasma_grids.hat_Iy_from_jtor(self.broad_hatIy)

    def circ_eq_residual_f(self, trial_currents, active_voltage_vec):
        """Uses record of state at time t, the provided current values and a record
        of the plasma current distribution (as obtained while solving the dynamics)
        to quantify the residual in the system of circuit equations.
        self.Id_dot and self.Iy_dot are calculated in self.assign_currents_solve_GS

        Parameters
        ----------
        trial_currents : np.array
            Currents at time t+dt. Same format as self.currents_vec.
        active_voltage_vec : np.array
            Vector of active voltages for the active coils, applied between t and t+dt.
        """
        self.forcing_term = self.evol_metal_curr.forcing_term_eig_plasma(
            active_voltage_vec=active_voltage_vec, Iydot=self.Iy_dot
        )

        self.circuit_eq_residual[:-1] = self.evol_metal_curr.current_residual(
            Itpdt=trial_currents[:-1], Iddot=self.Id_dot, forcing_term=self.forcing_term
        )

        self.circuit_eq_residual[-1] = (
            self.evol_plasma_curr.current_residual(
                red_Iy0=self.Iy,
                red_Iy1=self.trial_Iy1,
                red_Iydot=self.Iy_dot,
                Iddot=self.Id_dot,
            )
            / self.plasma_norm_factor
        )

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
        (stored in the private self.eq2 and self.profile2).

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
        hat_Iy1 = self.plasma_grids.hat_Iy_from_jtor(jtor_)
        return hat_Iy1

    def F_function_curr(self, trial_currents, active_voltage_vec):
        """Full non-linear system of circuit eqs written as root problem
        in the vector of current values at time t+dt.
        Note that the plasma flux at time t+dt is taken to be self.trial_plasma_psi.
        Iteration consists of:
        [trial_currents, plasma_flux] -> hatIy1, through calculating plasma distribution
        hatIy1 -> iterated_currents, through 'simplified' circuit eqs
        Residual: iterated_currents - trial_currents
        is zero if [trial_currents, plasma_flux] solve the full non-linear problem.

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
        hatIy1 = self.calculate_hatIy(trial_currents, self.trial_plasma_psi)
        iterated_currs = self.currents_from_hatIy(hatIy1, active_voltage_vec)
        current_res = iterated_currs - trial_currents
        return current_res

    def calculate_hatIy_GS(self, trial_currents, rtol_NK):
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
        self.assign_currents_solve_GS(trial_currents, rtol_NK=rtol_NK)
        hatIy1 = self.plasma_grids.hat_Iy_from_jtor(self.profiles2.jtor)
        return hatIy1

    # WORKING ON IT
    # def F_function_0(self, trial_sol, active_voltage_vec):

    #     trial_currents = trial_sol[:self.extensive_currents_dim]
    #     trial_plasma_psi = trial_sol[self.extensive_currents_dim:]

    #     trial_hatIy1 = self.calculate_hatIy(trial_currents, trial_plasma_psi.reshape(self.nx, self.ny))
    #     self.make_broad_hatIy(trial_hatIy1)

    #     ceq_residuals = self.simplified_solver_J1.ceq_residuals(I_0=self.currents_vec,
    #                                                             I_1=trial_currents,
    #                                                             hatIy_left=self.broad_hatIy,
    #                                                             hatIy_0=self.hatIy,
    #                                                             hatIy_1=trial_hatIy1,
    #                                                             active_voltage_vec=active_voltage_vec)

    #     GS_psi_residuals = self.NK.F_function(trial_plasma_psi,
    #                                           self.tokamak_psi.reshape(-1),
    #                                           self.profiles2)

    #     full_residual = np.concatenate((ceq_residuals, GS_psi_residuals))

    #     return full_residual

    # def F_function_1(self, trial_sol, active_voltage_vec):

    #     trial_currents = trial_sol[:self.extensive_currents_dim]*self.current_norm
    #     trial_plasma_psi = trial_sol[self.extensive_currents_dim:]*self.psi_norm
    #     self.trial_plasma_psi = np.copy(trial_plasma_psi).reshape(self.nx, self.ny)

    #     # trial_hatIy1 = self.calculate_hatIy(trial_currents, trial_plasma_psi.reshape(self.nx, self.ny))
    #     # self.make_broad_hatIy(trial_hatIy1)

    #     # ceq_residuals = self.simplified_solver_J1.ceq_residuals(I_0=self.currents_vec,
    #     #                                                         I_1=trial_currents,
    #     #                                                         hatIy_left=self.broad_hatIy,
    #     #                                                         hatIy_0=self.hatIy,
    #     #                                                         hatIy_1=trial_hatIy1,
    #     #                                                         active_voltage_vec=active_voltage_vec)
    #     ceq_residuals = self.F_function_curr(trial_currents, active_voltage_vec)

    #     GS_psi_residuals = self.NK.F_function(trial_plasma_psi,
    #                                           self.tokamak_psi.reshape(-1),
    #                                           self.profiles2)

    #     full_residual = np.concatenate((ceq_residuals, GS_psi_residuals))

    #     return full_residual

    # def F_function_2(self, trial_sol, active_voltage_vec, curr_eps):

    #     trial_currents = trial_sol[:self.extensive_currents_dim]*self.current_norm
    #     trial_plasma_psi = trial_sol[self.extensive_currents_dim:]*self.psi_norm
    #     self.trial_plasma_psi = np.copy(trial_plasma_psi).reshape(self.nx, self.ny)

    #     curr_step = abs(trial_currents - self.currents_vec_m1)
    #     self.curr_step = np.where(curr_step>curr_eps, curr_step, curr_eps)
    #     ceq_residuals = self.F_function_curr(trial_currents, active_voltage_vec)/self.curr_step

    #     plasma_psi_step = trial_plasma_psi - self.eq1.plasma_psi.reshape(-1)
    #     self.d_plasma_psi_step = np.amax(plasma_psi_step) - np.amin(plasma_psi_step)
    #     GS_psi_residuals = self.NK.F_function(trial_plasma_psi,
    #                                           self.tokamak_psi.reshape(-1),
    #                                           self.profiles2)/self.d_plasma_psi_step

    #     full_residual = np.concatenate((ceq_residuals, GS_psi_residuals))

    #     return full_residual

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
        is zero if trial_currents solve the full non-linear problem.

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
        hatIy1 = self.calculate_hatIy_GS(trial_currents, rtol_NK=rtol_NK)
        iterated_currs = self.currents_from_hatIy(hatIy1, active_voltage_vec)
        current_res = iterated_currs - trial_currents
        return current_res

    def F_function_ceq_GS(self, trial_currents, active_voltage_vec, rtol_NK):
        """Full non-linear system of circuit eqs written as root problem
        in the vector of current values at time t+dt.
        Note that, differently from self.F_function_curr, here the plasma flux
        is not imposed, but self-consistently solved for based on the input trial_currents.
        Iteration consists of:
        trial_currents -> plasma flux, through static GS
        [trial_currents, plasma_flux] -> hatIy1, through calculating plasma distribution
        hatIy1 -> iterated_currents, through 'simplified' circuit eqs
        Residual: iterated_currents - trial_currents
        is zero if trial_currents solve the full non-linear problem.

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
        hatIy1 = self.calculate_hatIy_GS(trial_currents, rtol_NK=rtol_NK)
        self.make_broad_hatIy(hatIy1)
        # print('self.currents_vec', self.currents_vec)
        # print('trial_currents', trial_currents)
        # print('hatIy_1', np.sum(hatIy1**2))
        # print('hatIy_0', np.sum(self.hatIy**2))
        # print('active_voltage_vec',active_voltage_vec)
        ceq_residuals = self.simplified_solver_J1.ceq_residuals(
            I_0=self.currents_vec,
            I_1=trial_currents,
            hatIy_left=self.broad_hatIy,
            hatIy_0=self.hatIy,
            hatIy_1=hatIy1,
            active_voltage_vec=active_voltage_vec,
        )
        return ceq_residuals

    def F_function_psi(self, trial_plasma_psi, active_voltage_vec, rtol_NK):
        """Full non-linear system of circuit eqs written as root problem
        in the plasma flux. Note that the flux associated to the metal currents
        is fixed externally through self.tokamak_psi.
        Iteration consists of:
        [trial_plasma_psi, tokamak_psi] -> hatIy1, by calculating Jtor
        hatIy1 -> currents(t+dt), through 'simplified' circuit eq
        currents(t+dt) -> iterated_plasma_flux, through static GS
        Residual: iterated_plasma_flux - trial_plasma_psi
        is zero if [trial_plasma_psi, tokamak_psi] solve the full non-linear problem.

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
        hatIy1 = self.plasma_grids.hat_Iy_from_jtor(jtor_)
        self.hatIy1_iterative_cycle(
            hatIy1=hatIy1, active_voltage_vec=active_voltage_vec, rtol_NK=rtol_NK
        )
        psi_residual = self.eq2.plasma_psi.reshape(-1) - trial_plasma_psi
        return psi_residual

    def F_function_psi_GS(self, trial_plasma_psi, active_voltage_vec, rtol_NK):
        """Full non-linear system of circuit eqs written as root problem
        in the plasma flux. Note that the flux associated to the metal currents
        is fixed externally through self.tokamak_psi.
        Iteration consists of:
        [trial_plasma_psi, tokamak_psi] -> hatIy1, by calculating Jtor
        hatIy1 -> currents(t+dt), through 'simplified' circuit eq
        currents(t+dt) -> iterated_plasma_flux, through static GS
        Residual: iterated_plasma_flux - trial_plasma_psi
        is zero if [trial_plasma_psi, tokamak_psi] solve the full non-linear problem.

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
        hatIy1 = self.plasma_grids.hat_Iy_from_jtor(jtor_)
        self.hatIy1_iterative_cycle(
            hatIy1=hatIy1, active_voltage_vec=active_voltage_vec, rtol_NK=rtol_NK
        )
        psi_residual = self.psi_gs_alpha[0] * (
            self.eq2.plasma_psi.reshape(-1) - trial_plasma_psi
        )

        psi_residual += self.psi_gs_alpha[1] * self.NK.F_function(
            trial_plasma_psi.reshape(-1), self.tokamak_psi.reshape(-1), self.profiles2
        )

        return psi_residual

    def calculate_rel_tolerance_currents(self, current_residual, curr_eps):
        """Calculates how the current_residual in input compares to the step
        in the currents themselves.
        The relative residual is used to quantify the relative convergence of the stepper.
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

    def calculate_rel_tolerance_GS(self, trial_plasma_psi):
        """Calculates how the residual in the plasma flux due to the static GS problem
        compares to the change in the plasma flux itself due to the dynamics.
        The relative residual is used to quantify the relative convergence of the stepper.
        It accesses self.trial_plasma_psi, self.eq1.plasma_psi, self.tokamak_psi

        Returns
        -------
        float
            Relative plasma flux residual.
        """
        plasma_psi_step = trial_plasma_psi - self.eq1.plasma_psi
        self.d_plasma_psi_step = np.amax(plasma_psi_step) - np.amin(plasma_psi_step)

        a_res_GS = np.amax(
            abs(
                self.NK.F_function(
                    trial_plasma_psi.reshape(-1),
                    self.tokamak_psi.reshape(-1),
                    self.profiles2,
                )
            )
        )

        r_res_GS = a_res_GS / self.d_plasma_psi_step
        return r_res_GS

    def calculate_GS_rel_tolerance(self, trial_plasma_psi, a_res_GS):
        """Calculates how the residual in the plasma flux due to the static GS problem
        compares to the change in the plasma flux itself due to the dynamics.
        The relative residual is used to quantify the relative convergence of the stepper.
        It accesses self.trial_plasma_psi, self.eq1.plasma_psi, self.tokamak_psi

        Returns
        -------
        float
            Relative plasma flux residual.
        """
        plasma_psi_step = trial_plasma_psi - self.eq1.plasma_psi
        self.d_plasma_psi_step = np.amax(plasma_psi_step) - np.amin(plasma_psi_step)

        a_res_GS = np.amax(abs(a_res_GS))

        r_res_GS = a_res_GS / self.d_plasma_psi_step
        return r_res_GS

    def check_and_change_profiles(
        self, profile_parameter=None, profile_coefficients=None
    ):
        """Checks if new input parameters are different from those presently in place.
        If so, it actions the necessary changes.

        Parameters
        ----------
        profile_parameter : None or float for new paxis or betap
            Set to None when the profile parameter (paxis or betap) is left unchanged
            with respect to the previous timestep. Set here desired value otherwise.
        profile_coefficients : None or tuple (alpha_m, alpha_n)
            Set to None when the profile coefficients alpha_m and alpha_n are left unchanged
            with respect to the previous timestep. Set here desired values otherwise.
        """
        self.profile_change_flag = 0
        self.d_profile_pars = np.zeros(3)
        if profile_parameter is not None:
            if profile_parameter != self.profiles1.profile_parameter:
                self.profile_change_flag += 1
                self.d_profile_pars[2] = (
                    profile_parameter - self.profiles1.profile_parameter
                )
                self.profiles1.assign_profile_parameter(profile_parameter)
                self.profiles2.assign_profile_parameter(profile_parameter)
        if profile_coefficients is not None:
            if (
                profile_coefficients[0] != self.profiles1.alpha_m
                or profile_coefficients[1] != self.profiles1.alpha_n
            ):
                self.profile_change_flag += 1
                self.d_profile_pars[0] = (
                    profile_coefficients[0] - self.profiles1.alpha_m
                )
                self.d_profile_pars[1] = (
                    profile_coefficients[1] - self.profiles1.alpha_n
                )
                self.profiles1.alpha_m = profile_coefficients[0]
                self.profiles1.alpha_n = profile_coefficients[1]
                self.profiles2.alpha_m = profile_coefficients[0]
                self.profiles2.alpha_n = profile_coefficients[1]
        # print(self.profiles2.alpha_m, self.profiles2.alpha_n, self.profiles2.profile_parameter)

    def nlstepper(
        self,
        active_voltage_vec,
        profile_parameter=None,
        profile_coefficients=None,
        target_relative_tol_currents=0.005,
        target_relative_tol_GS=0.005,
        working_relative_tol_GS=0.001,
        target_relative_unexplained_residual=0.5,
        max_n_directions=3,
        max_Arnoldi_iterations=4,
        max_collinearity=0.3,
        step_size_psi=2.0,
        step_size_curr=0.8,
        scaling_with_n=0,
        relative_tol_for_nk_psi=0.002,
        blend_GS=0.5,
        blend_psi=1,
        curr_eps=1e-5,
        max_no_NK_psi=1.0,
        clip=5,
        threshold=1.5,
        clip_hard=1.5,
        verbose=0,
        linear_only=False,
    ):
        """The main stepper function.
        If linear_only = True, this advances the linearised problem.
        If linear_only = False, a solution of the full non-linear problem is seeked using
        a combination of NK methods.
        When a solution has been found, time is advanced by self.dt_step,
        currents are recorded in self.currents_vec and profile properties
        in self.eq1 and self.profiles1.
        The solver's algorithm proceeds like below:
        1) solve linearised problem for initial guess of the currents and solve associated GS,
        assign trial_plasma_psi and trial_currents (and consequent tokamak_psi);
        2) if pair [trial_plasma_psi, tokamak_psi] fails static GS check (control_GS),
        update trial_plasma_psi using GS solution;
        3) at fixed trial_currents (and consequent tokamak_psi) update trial_plasma_psi
        using NK solver for the associated root problem;
        4) at fixed trial_plasma_psi, update trial_currents (and consequent tokamak_psi)
        using NK solver for the associated root problem;
        5) if convergence on the current residuals is not achieved or static GS check
        fails, restart from point 2;
        6) the pair [trial_currents, trial_plasma_psi] solves the nonlinear dynamic problem,
        assign values to self.currents_vec, self.eq1 and self.profiles1.


        Parameters
        ----------
        active_voltage_vec : np.array
            Vector of active voltages for the active coils, applied between t and t+dt.
        profile_parameter : None or float for new paxis or betap
            Set to None when the profile parameter (paxis or betap) is left unchanged
            with respect to the previous timestep. Set here desired value otherwise.
        profile_coefficients : None or tuple (alpha_m, alpha_n)
            Set to None when the profile coefficients alpha_m and alpha_n are left unchanged
            with respect to the previous timestep. Set here desired values otherwise.
        target_relative_tol_currents : float, optional, by default .01
            Relative tolerance in the currents required for convergence.
        target_relative_tol_GS : float, optional, by default .01
            Relative tolerance in the plasma flux required for convergence.
        working_relative_tol_GS : float, optional, by default .002
            Tolerance used when solving all static GS problems, expressed in
            terms of the change in the plasma flux due to 1 timestep of evolution.
        target_relative_unexplained_residual : float, optional, by default .5
            Used in the NK solvers. Inclusion of additional basis vectors is
            stopped if the fraction of unexplained_residual is < target_relative_unexplained_residual.
        max_n_directions : int, optional, by default 3
            Used in the NK solvers. Inclusion of additional basis vectors is
            stopped if max_n_directions have already been included.
        max_Arnoldi_iterations : int, optional, by default 4
            Used in the NK solvers. Inclusion of additional basis vectors is
            stopped if max_n_directions have already been considered for inclusion,
            though not necessarily included.
        max_collinearity : float, optional, by default .3
            Used in the NK solvers. The basis vector being considered is rejected
            if scalar product with any of the previously included is larger than max_collinearity.
        step_size_psi : float, optional, by default 2.
            Used by the NK solver applied to the root problem in the plasma flux.
            l2 norm of proposed step.
        step_size_curr : float, optional, by default .8
            Used by the NK solver applied to the root problem in the currents.
            l2 norm of proposed step.
        scaling_with_n : int, optional, by default 0
            Used in the NK solvers. Allows to further scale dx candidate steps by factor
            (1 + self.n_it)**scaling_with_n
        relative_tol_for_nk_psi : float, optional, by default .002
            NK solver for the root problem in the plasma flux is not used if
            the associated residual is < self.rtol_NK/relative_tol_for_nk_psi
        max_no_NK_psi : float, optional, by default 1.
            Maximum number of consecutive times the NK solver for the root problem in the plasma flux
            can be shortcutted.
        blend_GS : float, optional, by default .5
            Blend coefficient used in trial_plasma_psi updates at step 2 of the algorithm above.
            Should be between 0 and 1.
        blend_psi : float, optional, by default 1.
            Blend coefficient used in trial_plasma_psi updates at step 3 of the algorithm above.
            Should be between 0 and 1.
        curr_eps : float, optional, by default 1e-5
            Used in calculating the relative convergence on the currents. Min value of the current
            step. Avoids divergence when dividing by the step in the currents.
        clip : float, optional, by default 5
            Used in the NK solvers. Maximum step size for each accepted basis vector, in units
            of the exploratory step.
        threshold : float, optional, by default 1.5
            Used in the NK solvers to catch cases of untreated (partial) collinearity.
            If relative_unexplained_residual>threshold, clip_hard is applied instead of clip.
        clip_hard : float, optional, by default 1.5
             Used in the NK solvers. Maximum step size for each accepted basis vector, in units
            of the exploratory step, for cases of partial collinearity.
        verbose : int, optional, by default T
            Printouts of convergence process.
            Use 1 for printouts with details on each NK cycle.
            Use 2 for printouts with deeper intermediate details.
        linear_only : bool, optional, by default False
            If linear_only = True the solution of the linearised problem is accepted.
            If linear_only = False, the convergence criteria are used and a solution of
            the full nonlinear problem is seeked.

        Returns
        -------
        int
            Number of grid points NOT in the reduced plasma domain that have some plasma in them (Jtor>0).
            Depending on the definition of the reduced plasma domain through plasma_domain_mask,
            this may mean the plasma contacted the wall. This will stop the dynamics.
        """

        # check if profile parameter (betap or paxis) is being altered
        # and action the change where necessary
        self.check_and_change_profiles(
            profile_parameter=profile_parameter,
            profile_coefficients=profile_coefficients,
        )

        # solves the linearised problem for the currents.
        # needs to use the time derivativive of the profile parameters, if they have been changed
        if self.profile_change_flag:
            self.d_profile_pars_dt = self.d_profile_pars / self.dt_step
        else:
            self.d_profile_pars_dt = None
        self.set_linear_solution(active_voltage_vec, self.d_profile_pars_dt)
        # Solution and GS equilibrium are assigned to self.trial_currents and self.trial_plasma_psi

        if linear_only:
            # assign currents and plasma flux to self.currents_vec, self.eq1 and self.profile1 and complete step
            self.step_complete_assign(working_relative_tol_GS, from_linear=True)

        else:
            # seek solution of the full nonlinear problem

            # this assigns to self.eq2 and self.profiles2
            # also records self.tokamak_psi corresponding to self.trial_currents in 2d
            res_curr = self.F_function_curr(
                self.trial_currents, active_voltage_vec
            ).copy()

            # uses self.trial_currents and self.currents_vec_m1 to relate res_curr above to step in the currents
            rel_curr_res = 1.0 * self.calculate_rel_tolerance_currents(
                res_curr, curr_eps=curr_eps
            )
            control = np.any(rel_curr_res > target_relative_tol_currents)

            # pair self.trial_currents and self.trial_plasma_psi are a GS solution
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
                # print(self.F_function_ceq_GS(self.trial_currents, *args_nk))
            log = []

            # counter for instances in which the NK solver in psi has been shortcutted
            n_no_NK_psi = 0

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
                    self.record_for_update(self.trial_currents, self.profiles2)

                # prepare for NK algorithms: 1d vectors needed for independent variable
                self.trial_plasma_psi = self.trial_plasma_psi.reshape(-1)
                self.tokamak_psi = self.tokamak_psi.reshape(-1)

                # calculate initial residual for the root problem in psi
                res_psi = (
                    1.0
                    * self.F_function_psi(
                        trial_plasma_psi=self.trial_plasma_psi,
                        active_voltage_vec=active_voltage_vec,
                        rtol_NK=self.rtol_NK,
                    ).copy()
                )
                del_res_psi = np.amax(res_psi) - np.amin(res_psi)

                if (del_res_psi > self.rtol_NK / relative_tol_for_nk_psi) + (
                    n_no_NK_psi > max_no_NK_psi
                ):
                    n_no_NK_psi = 0
                    # NK algorithm to solve the root problem in psi
                    self.psi_nk_solver.Arnoldi_iteration(
                        x0=self.trial_plasma_psi,  # trial_current expansion point
                        dx=res_psi.copy(),  # first vector for current basis
                        R0=res_psi,  # circuit eq. residual at trial_current expansion point: F_function(trial_current)
                        F_function=self.F_function_psi,
                        args=args_nk,
                        step_size=step_size_psi,
                        scaling_with_n=scaling_with_n,
                        target_relative_unexplained_residual=target_relative_unexplained_residual,  # add basis vector
                        max_n_directions=max_n_directions,  # max number of basis vectors (must be less than number of modes + 1)
                        max_Arnoldi_iterations=max_Arnoldi_iterations,
                        max_collinearity=max_collinearity,
                        clip=clip,
                        threshold=threshold,
                        clip_hard=clip_hard,
                    )

                    # update trial_plasma_psi according to NK solution
                    self.trial_plasma_psi += self.psi_nk_solver.dx * blend_psi
                    psi_text = [[self.text_psi_1, self.psi_nk_solver.coeffs]]

                else:
                    # NK algorithm has been shortcutted, keep count
                    n_no_NK_psi += 1
                    psi_text = [
                        self.text_psi_0.format(
                            skippedno=n_no_NK_psi, psi_res=del_res_psi
                        )
                    ]

                # prepare for NK solver on the currents, 2d plasma flux needed
                self.trial_plasma_psi = self.trial_plasma_psi.reshape(self.nx, self.ny)

                # calculates initial residual for the root problem in the currents
                # assumes the just updated self.trial_plasma_psi
                res_curr = self.F_function_curr(
                    self.trial_currents, active_voltage_vec
                ).copy()
                rel_curr_res = abs(res_curr / self.curr_step)
                interm_text = [
                    "The intermediate residuals on the current: max =",
                    np.amax(rel_curr_res),
                    "mean =",
                    np.mean(rel_curr_res),
                ]

                if verbose - 1:
                    log.append(psi_text)
                    log.append(interm_text)

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
                    max_Arnoldi_iterations=max_Arnoldi_iterations,
                    max_collinearity=max_collinearity,
                    clip=clip,
                    threshold=threshold,
                    clip_hard=clip_hard,
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
                r_res_GS = 1.0 * self.calculate_rel_tolerance_GS(self.trial_plasma_psi)
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
                self.ceq_res = 1.0 * self.F_function_ceq_GS(
                    self.trial_currents, *args_nk
                )
                log.append(
                    [
                        "The final residual on the current (relative): max =",
                        np.amax(self.ceq_res),
                        "mean =",
                        np.mean(self.ceq_res),
                    ]
                )
                log.append(["Residuals on GS eq (relative): ", r_res_GS])

                # one full cycle completed
                n_it += 1

            # convergence checks succeeded, complete step
            self.step_complete_assign(working_relative_tol_GS)

        # # check plasma is still fully contained in the plasma reduced domain
        # flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)

        # return flag

    # WORKING ON IT
    # def nlstepper_currents_only(self,
    #                      F_function,
    #                      active_voltage_vec,
    #                      profile_parameter=None,
    #                      profile_coefficients=None,
    #                      target_relative_tol_currents=.01,
    #                      working_relative_tol_GS=.002,
    #                      target_relative_unexplained_residual=.5,
    #                      max_n_directions=3,
    #                      max_Arnoldi_iterations=4,
    #                      max_collinearity=.3,
    #                      step_size=.8,
    #                      scaling_with_n=0,
    #                      curr_eps=1e-5,
    #                      clip=5,
    #                      threshold=1.5,
    #                      clip_hard=1.5,
    #                      verbose=0,
    #                      linear_only=False):

    #     # check if profile parameter (betap or paxis) is being altered
    #     # and action the change where necessary
    #     self.check_and_change_profiles(profile_parameter=profile_parameter,
    #                                    profile_coefficients=profile_coefficients)

    #     # solves the linearised problem for the currents.
    #     # needs to use the time derivativive of the profile parameters, if they have been changed
    #     if self.profile_change_flag:
    #         self.d_profile_pars_dt = self.d_profile_pars/self.dt_step
    #     else:
    #         self.d_profile_pars_dt = None
    #     self.set_linear_solution(active_voltage_vec, self.d_profile_pars_dt)
    #     # Solution and GS equilibrium are assigned to self.trial_currents and self.trial_plasma_psi

    #     args_nk = [active_voltage_vec, self.rtol_NK]

    #     if linear_only:
    #         # assign currents and plasma flux to self.currents_vec, self.eq1 and self.profile1 and complete step
    #         self.step_complete_assign(working_relative_tol_GS, from_linear=True)

    #     else:
    #         # seek solution of the full nonlinear problem

    #         # this assigns to self.eq2 and self.profiles2
    #         # also records self.tokamak_psi corresponding to self.trial_currents in 2d
    #         # res_curr = 1.0*self.F_function_ceq_GS(self.trial_currents, *args_nk)
    #         res_curr = F_function(self.trial_currents, *args_nk).copy()

    #         # uses self.trial_currents and self.currents_vec_m1 to relate res_curr above to step in the currents
    #         rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #         control = np.any(rel_curr_res > target_relative_tol_currents)

    #         if verbose:
    #             print('starting numerical solve:')
    #             print('max(residual on current eqs) =', np.amax(rel_curr_res), 'mean(residual on current eqs) =', np.mean(rel_curr_res))
    #             # print('res_curr', res_curr)
    #         log = []

    #         # counter for number of solution cycles
    #         n_it = 0

    #         while control:

    #             if verbose:
    #                 for _ in log:
    #                     print(_)

    #             log = [self.text_nk_cycle.format(nkcycle = n_it)]

    #             self.currents_nk_solver.Arnoldi_iteration(  x0=self.trial_currents, #trial_current expansion point
    #                                                         dx=res_curr.copy(), #first vector for current basis
    #                                                         R0=res_curr.copy(), #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                                         F_function=F_function,
    #                                                         args=args_nk,
    #                                                         step_size=step_size,
    #                                                         scaling_with_n=scaling_with_n,
    #                                                         target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector
    #                                                         max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                                         max_Arnoldi_iterations=max_Arnoldi_iterations,
    #                                                         max_collinearity=max_collinearity,
    #                                                         clip=clip,
    #                                                         threshold=threshold,
    #                                                         clip_hard=clip_hard)

    #             self.trial_currents += self.currents_nk_solver.dx

    #             res_curr = F_function(self.trial_currents, *args_nk).copy()
    #             rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #             control = np.any(rel_curr_res > target_relative_tol_currents)

    #             log.append(['The coeffs applied to the current vec = ', self.currents_nk_solver.coeffs])
    #             log.append(['The final residual on the current (relative): max =', np.amax(rel_curr_res), 'mean =', np.mean(rel_curr_res)])

    #             n_it += 1

    #         self.time += self.dt_step

    #         self.step_complete_assign(working_relative_tol_GS)

    # def nlstepper_currents_psiplasma(self,
    #                                 F_function,
    #                                 active_voltage_vec,
    #                                 profile_parameter=None,
    #                                 profile_coefficients=None,
    #                                 target_relative_tol_currents=.01,
    #                                 target_relative_tol_GS=.01,
    #                                 working_relative_tol_GS=.002,
    #                                 target_relative_unexplained_residual=.5,
    #                                 max_n_directions=3,
    #                                 max_Arnoldi_iterations=4,
    #                                 max_collinearity=.3,
    #                                 step_size=.8,
    #                                 scaling_with_n=0,
    #                                 curr_eps=1e-5,
    #                                 clip=5,
    #                                 threshold=1.2,
    #                                 clip_hard=.5,
    #                                 verbose=0,
    #                                 linear_only=False):

    #     # check if profile parameter (betap or paxis) is being altered
    #     # and action the change where necessary
    #     self.check_and_change_profiles(profile_parameter=profile_parameter,
    #                                    profile_coefficients=profile_coefficients)

    #     # solves the linearised problem for the currents.
    #     # needs to use the time derivativive of the profile parameters, if they have been changed
    #     if self.profile_change_flag:
    #         self.d_profile_pars_dt = self.d_profile_pars/self.dt_step
    #     else:
    #         self.d_profile_pars_dt = None
    #     self.set_linear_solution(active_voltage_vec, self.d_profile_pars_dt)
    #     # Solution and GS equilibrium are assigned to self.trial_currents and self.trial_plasma_psi

    #     # args_nk = [active_voltage_vec, self.rtol_NK]

    #     if linear_only:
    #         # assign currents and plasma flux to self.currents_vec, self.eq1 and self.profile1 and complete step
    #         self.step_complete_assign(working_relative_tol_GS, from_linear=True)

    #     else:
    #         # seek solution of the full nonlinear problem

    #         # self.current_norm = np.mean(np.abs(self.currents_vec))
    #         # self.current_norm = np.where(np.abs(self.currents_vec)>current_norm, np.abs(self.currents_vec), current_norm)
    #         self.psi_norm = np.mean(np.abs(self.eq1.plasma_psi))
    #         self.trial_curr_plasmapsi = np.concatenate((self.trial_currents/self.current_norm,
    #                                                     self.trial_plasma_psi.reshape(-1)/self.psi_norm))

    #         # this assigns to self.eq2 and self.profiles2
    #         # also records self.tokamak_psi corresponding to self.trial_currents in 2d
    #         # res_curr = 1.0*self.F_function_ceq_GS(self.trial_currents, *args_nk)
    #         all_res = F_function(self.trial_curr_plasmapsi, active_voltage_vec, curr_eps).copy()

    #         # uses self.trial_currents and self.currents_vec_m1 to relate res_curr above to step in the currents
    #         # rel_curr_res = self.calculate_rel_tolerance_currents(all_res[:self.extensive_currents_dim], curr_eps=curr_eps)
    #         # r_res_GS = self.calculate_GS_rel_tolerance(self.trial_plasma_psi, all_res[self.extensive_currents_dim:])
    #         rel_curr_res = all_res[:self.extensive_currents_dim].copy()
    #         r_res_GS = np.amax(abs(all_res[self.extensive_currents_dim:]))
    #         control = np.any(rel_curr_res > target_relative_tol_currents)
    #         control += (r_res_GS > target_relative_tol_GS)

    #         if verbose:
    #             print('starting numerical solve:')
    #             print('max(relative residual on current eqs) =', np.amax(rel_curr_res), 'mean(relative residual on current eqs) =', np.mean(rel_curr_res))
    #             print('max(relative residual on GS eqs) =', r_res_GS)
    #         log = []

    #         # counter for number of solution cycles
    #         n_it = 0

    #         while control:

    #             if verbose:
    #                 for _ in log:
    #                     print(_)

    #             log = [self.text_nk_cycle.format(nkcycle = n_it)]

    #             self.full_nk_solver.Arnoldi_iteration(  x0=self.trial_curr_plasmapsi, #trial_current expansion point
    #                                                     dx=all_res.copy(), #first vector for current basis
    #                                                     R0=all_res.copy(), #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                                     F_function=F_function,
    #                                                     args=[active_voltage_vec, curr_eps],
    #                                                     step_size=step_size,
    #                                                     scaling_with_n=scaling_with_n,
    #                                                     target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector
    #                                                     max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                                     max_Arnoldi_iterations=max_Arnoldi_iterations,
    #                                                     max_collinearity=max_collinearity,
    #                                                     clip=clip,
    #                                                     threshold=threshold,
    #                                                     clip_hard=clip_hard)

    #             self.trial_curr_plasmapsi += self.full_nk_solver.dx
    #             self.trial_currents = self.trial_curr_plasmapsi[:self.extensive_currents_dim]*self.current_norm
    #             self.trial_plasma_psi = self.trial_curr_plasmapsi[self.extensive_currents_dim:].reshape(self.nx,self.ny)*self.psi_norm

    #             all_res = F_function(self.trial_curr_plasmapsi, active_voltage_vec, curr_eps).copy()
    #             # rel_curr_res = self.calculate_rel_tolerance_currents(all_res[:self.extensive_currents_dim], curr_eps=curr_eps)
    #             # r_res_GS = self.calculate_GS_rel_tolerance(self.trial_plasma_psi, all_res[self.extensive_currents_dim:])
    #             rel_curr_res = all_res[:self.extensive_currents_dim].copy()
    #             r_res_GS = np.amax(abs(all_res[self.extensive_currents_dim:]))
    #             control = np.any(rel_curr_res > target_relative_tol_currents)
    #             control += (r_res_GS > target_relative_tol_GS)

    #             log.append(['The coeffs applied to the full vec = ', self.full_nk_solver.coeffs])
    #             log.append(['The final residual on the current (relative): max =', np.amax(rel_curr_res), 'mean =', np.mean(rel_curr_res)])
    #             log.append(['The final residual on GS (relative): max =', r_res_GS])

    #             n_it += 1

    #         self.time += self.dt_step

    #         self.step_complete_assign(working_relative_tol_GS)

    # def nlstepper_ceq_GS(self,
    #                     #  F_function,
    #                      active_voltage_vec,
    #                      profile_parameter=None,
    #                      profile_coefficients=None,
    #                      target_relative_tol_currents=.01,
    #                      working_relative_tol_GS=.002,
    #                      target_relative_unexplained_residual=.5,
    #                      max_n_directions=3,
    #                      max_Arnoldi_iterations=4,
    #                      max_collinearity=.3,
    #                      step_size=.8,
    #                      scaling_with_n=0,
    #                      curr_eps=1e-5,
    #                      clip=5,
    #                      threshold=1.5,
    #                      clip_hard=1.5,
    #                      verbose=0,
    #                      linear_only=False):

    #     # check if profile parameter (betap or paxis) is being altered
    #     # and action the change where necessary
    #     self.check_and_change_profiles(profile_parameter=profile_parameter,
    #                                    profile_coefficients=profile_coefficients)

    #     # solves the linearised problem for the currents.
    #     # needs to use the time derivativive of the profile parameters, if they have been changed
    #     if self.profile_change_flag:
    #         self.d_profile_pars_dt = self.d_profile_pars/self.dt_step
    #     else:
    #         self.d_profile_pars_dt = None
    #     self.set_linear_solution(active_voltage_vec, self.d_profile_pars_dt)
    #     # Solution and GS equilibrium are assigned to self.trial_currents and self.trial_plasma_psi

    #     args_nk = [active_voltage_vec, self.rtol_NK]

    #     if linear_only:
    #         # assign currents and plasma flux to self.currents_vec, self.eq1 and self.profile1 and complete step
    #         self.step_complete_assign(working_relative_tol_GS, from_linear=True)

    #     else:
    #         # seek solution of the full nonlinear problem

    #         # this assigns to self.eq2 and self.profiles2
    #         # also records self.tokamak_psi corresponding to self.trial_currents in 2d
    #         # res_curr = 1.0*self.F_function_ceq_GS(self.trial_currents, *args_nk)
    #         res_curr = 1.0*self.F_function_ceq_GS(self.trial_currents, *args_nk)

    #         # uses self.trial_currents and self.currents_vec_m1 to relate res_curr above to step in the currents
    #         rel_curr_res = 1.0*self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #         control = np.any(rel_curr_res > target_relative_tol_currents)

    #         if verbose:
    #             print('starting numerical solve:')
    #             print('max(residual on current eqs) =', np.amax(rel_curr_res), 'mean(residual on current eqs) =', np.mean(rel_curr_res))
    #             # print('res_curr', res_curr)
    #         log = []

    #         # counter for number of solution cycles
    #         n_it = 0

    #         while control:

    #             if verbose:
    #                 for _ in log:
    #                     print(_)

    #             log = [self.text_nk_cycle.format(nkcycle = n_it)]

    #             self.currents_nk_solver.Arnoldi_iteration(  x0=self.trial_currents, #trial_current expansion point
    #                                                         dx=res_curr, #first vector for current basis
    #                                                         R0=res_curr, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                                         F_function=self.F_function_ceq_GS,
    #                                                         args=args_nk,
    #                                                         step_size=step_size,
    #                                                         scaling_with_n=scaling_with_n,
    #                                                         target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector
    #                                                         max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                                         max_Arnoldi_iterations=max_Arnoldi_iterations,
    #                                                         max_collinearity=max_collinearity,
    #                                                         clip=clip,
    #                                                         threshold=threshold,
    #                                                         clip_hard=clip_hard)

    #             self.trial_currents += self.currents_nk_solver.dx

    #             res_curr = 1.0*self.F_function_ceq_GS(self.trial_currents, *args_nk)
    #             rel_curr_res = 1.0*self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #             control = np.any(rel_curr_res > target_relative_tol_currents)

    #             log.append(['The coeffs applied to the current vec = ', self.currents_nk_solver.coeffs])
    #             log.append(['The final residual on the current (relative): max =', np.amax(rel_curr_res), 'mean =', np.mean(rel_curr_res)])

    #             n_it += 1

    #         self.time += self.dt_step

    #         self.step_complete_assign(working_relative_tol_GS)

    # def nlstepper_GS(self, active_voltage_vec,
    #                             target_relative_tol_currents=.1,
    #                             use_extrapolation=False,
    #                             working_relative_tol_GS=.01,
    #                             target_relative_unexplained_residual=.6,
    #                             max_n_directions=4,
    #                             max_Arnoldi_iterations=5,
    #                             max_collinearity=.3,
    #                             step_size_curr=1,
    #                             scaling_with_n=0,
    #                             curr_eps=1e-4,
    #                             clip=3,
    #                             threshold=1.5,
    #                             clip_hard=1.5,
    #                             verbose=False,
    #                             ):
    #     """Alternative solution method for the full nonlinear problem based on solving
    #     the root problem in the currents while remaining on exact GS solutions.
    #     Less performant than method above, suffers from collinearity problems.
    #     To be checked.

    #     Parameters
    #     ----------
    #     active_voltage_vec : _type_
    #         _description_
    #     target_relative_tol_currents : float, optional
    #         _description_, by default .1
    #     use_extrapolation : bool, optional
    #         _description_, by default False
    #     working_relative_tol_GS : float, optional
    #         _description_, by default .01
    #     target_relative_unexplained_residual : float, optional
    #         _description_, by default .6
    #     max_n_directions : int, optional
    #         _description_, by default 4
    #     max_Arnoldi_iterations : int, optional
    #         _description_, by default 5
    #     max_collinearity : float, optional
    #         _description_, by default .3
    #     step_size_curr : int, optional
    #         _description_, by default 1
    #     scaling_with_n : int, optional
    #         _description_, by default 0
    #     curr_eps : _type_, optional
    #         _description_, by default 1e-4
    #     clip : int, optional
    #         _description_, by default 3
    #     threshold : float, optional
    #         _description_, by default 1.5
    #     clip_hard : float, optional
    #         _description_, by default 1.5
    #     verbose : bool, optional
    #         _description_, by default False

    #     Returns
    #     -------
    #     _type_
    #         _description_
    #     """

    #     # self.central_2  = (1 + (self.step_no>0))
    #     if use_extrapolation*(self.step_no > self.extrapolator_input_size):
    #         self.trial_currents = 1.0*self.currents_guess

    #     else:
    #         self.trial_currents = self.hatIy1_iterative_cycle(self.hatIy,
    #                                                           active_voltage_vec,
    #                                                           rtol_NK=self.rtol_NK)

    #     res_curr = self.F_function_curr_GS(self.trial_currents, active_voltage_vec, self.rtol_NK)
    #     rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #     control = np.any(rel_curr_res > target_relative_tol_currents)

    #     args_nk = [active_voltage_vec, self.rtol_NK]

    #     if verbose:
    #         print('starting: curr residual', np.amax(rel_curr_res))
    #     log = []

    #     n_it = 0

    #     while control:

    #         if verbose:
    #             for _ in log:
    #                 print(_)

    #         log = []

    #         self.currents_nk_solver.Arnoldi_iteration(  x0=self.trial_currents, #trial_current expansion point
    #                                                     dx=res_curr, #first vector for current basis
    #                                                     R0=res_curr, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                                     F_function=self.F_function_curr_GS,
    #                                                     args=args_nk,
    #                                                     step_size=step_size_curr,
    #                                                     scaling_with_n=scaling_with_n,
    #                                                     target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector
    #                                                     max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                                     max_Arnoldi_iterations=max_Arnoldi_iterations,
    #                                                     max_collinearity=max_collinearity,
    #                                                     clip=clip,
    #                                                     threshold=threshold,
    #                                                     clip_hard=clip_hard)

    #         self.trial_currents += self.currents_nk_solver.dx#*blend_curr

    #         res_curr = self.F_function_curr_GS(self.trial_currents, active_voltage_vec, self.rtol_NK)
    #         rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #         control = np.any(rel_curr_res > target_relative_tol_currents)

    #         log.append([n_it, 'full cycle curr residual', np.amax(rel_curr_res), np.mean(rel_curr_res)])

    #         n_it += 1
    #         # print('cycle:', np.amax(rel_res0), np.mean(rel_res0))

    #         # r_dpsi = abs(self.eq2.plasma_psi - note_psi)
    #         # r_dpsi /= (np.amax(note_psi) - np.amin(note_psi))
    #         # control += np.any(r_dpsi > rtol_psi)

    #     self.time += self.dt_step

    #     # plt.figure()
    #     # plt.imshow(self.profiles2.jtor - self.jtor_m1)
    #     # plt.colorbar()
    #     # plt.show()

    #     # self.dpsi = self.eq2.plasma_psi - self.eq1.plasma_psi
    #     # plt.figure()
    #     # plt.imshow(self.dpsi)
    #     # plt.colorbar()
    #     # plt.show()

    #     # plt.figure()
    #     # plt.imshow(self.NK.tokamak_psi - note_tokamak_psi)
    #     # plt.colorbar()
    #     # plt.show()

    #     self.step_complete_assign(self.simplified_c, self.eq2.plasma_psi, working_relative_tol_GS)

    #     flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)

    #     return flag

    # def nlstepper1(self, active_voltage_vec,
    #                     profile_parameter=None,
    #                     profile_coefficients=None,
    #                     target_relative_tol_currents=.005,
    #                     target_relative_tol_GS=.002,
    #                     working_relative_tol_GS=.0005,
    #                     target_relative_unexplained_residual=.5,
    #                     max_n_directions=5,
    #                     max_Arnoldi_iterations=6,
    #                     max_collinearity=.3,
    #                     step_size_psi=2.,
    #                     step_size_curr=.8,
    #                     scaling_with_n=0,
    #                     relative_tol_for_nk_psi=.002,
    #                     blend_GS=.5,
    #                     blend_psi=1,
    #                     curr_eps=1e-5,
    #                     max_no_NK_psi=1.,
    #                     clip=5,
    #                     threshold=1.5,
    #                     clip_hard=1.5,
    #                     verbose=0,
    #                     linear_only=False):
    #     """The main stepper function.
    #     If linear_only = True, this advances the linearised problem.
    #     If linear_only = False, a solution of the full non-linear problem is seeked using
    #     a combination of NK methods.
    #     When a solution has been found, time is advanced by self.dt_step,
    #     currents are recorded in self.currents_vec and profile properties
    #     in self.eq1 and self.profiles1.
    #     The solver's algorithm proceeds like below:
    #     1) solve linearised problem for initial guess of the currents and solve associated GS,
    #     assign trial_plasma_psi and trial_currents (and consequent tokamak_psi);
    #     2) if pair [trial_plasma_psi, tokamak_psi] fails static GS check (control_GS),
    #     update trial_plasma_psi using GS solution;
    #     3) at fixed trial_currents (and consequent tokamak_psi) update trial_plasma_psi
    #     using NK solver for the associated root problem;
    #     4) at fixed trial_plasma_psi, update trial_currents (and consequent tokamak_psi)
    #     using NK solver for the associated root problem;
    #     5) if convergence on the current residuals is not achieved or static GS check
    #     fails, restart from point 2;
    #     6) the pair [trial_currents, trial_plasma_psi] solves the nonlinear dynamic problem,
    #     assign values to self.currents_vec, self.eq1 and self.profiles1.

    #     Parameters
    #     ----------
    #     active_voltage_vec : np.array
    #         Vector of active voltages for the active coils, applied between t and t+dt.
    #     profile_parameter : None or float for new paxis or betap
    #         Set to None when the profile parameter (paxis or betap) is left unchanged
    #         with respect to the previous timestep. Set here desired value otherwise.
    #     profile_coefficients : None or tuple (alpha_m, alpha_n)
    #         Set to None when the profile coefficients alpha_m and alpha_n are left unchanged
    #         with respect to the previous timestep. Set here desired values otherwise.
    #     target_relative_tol_currents : float, optional, by default .01
    #         Relative tolerance in the currents required for convergence.
    #     target_relative_tol_GS : float, optional, by default .01
    #         Relative tolerance in the plasma flux required for convergence.
    #     working_relative_tol_GS : float, optional, by default .002
    #         Tolerance used when solving all static GS problems, expressed in
    #         terms of the change in the plasma flux due to 1 timestep of evolution.
    #     target_relative_unexplained_residual : float, optional, by default .5
    #         Used in the NK solvers. Inclusion of additional basis vectors is
    #         stopped if the fraction of unexplained_residual is < target_relative_unexplained_residual.
    #     max_n_directions : int, optional, by default 3
    #         Used in the NK solvers. Inclusion of additional basis vectors is
    #         stopped if max_n_directions have already been included.
    #     max_Arnoldi_iterations : int, optional, by default 4
    #         Used in the NK solvers. Inclusion of additional basis vectors is
    #         stopped if max_n_directions have already been considered for inclusion,
    #         though not necessarily included.
    #     max_collinearity : float, optional, by default .3
    #         Used in the NK solvers. The basis vector being considered is rejected
    #         if scalar product with any of the previously included is larger than max_collinearity.
    #     step_size_psi : float, optional, by default 2.
    #         Used by the NK solver applied to the root problem in the plasma flux.
    #         l2 norm of proposed step.
    #     step_size_curr : float, optional, by default .8
    #         Used by the NK solver applied to the root problem in the currents.
    #         l2 norm of proposed step.
    #     scaling_with_n : int, optional, by default 0
    #         Used in the NK solvers. Allows to further scale dx candidate steps by factor
    #         (1 + self.n_it)**scaling_with_n
    #     relative_tol_for_nk_psi : float, optional, by default .002
    #         NK solver for the root problem in the plasma flux is not used if
    #         the associated residual is < self.rtol_NK/relative_tol_for_nk_psi
    #     max_no_NK_psi : float, optional, by default 1.
    #         Maximum number of consecutive times the NK solver for the root problem in the plasma flux
    #         can be shortcutted.
    #     blend_GS : float, optional, by default .5
    #         Blend coefficient used in trial_plasma_psi updates at step 2 of the algorithm above.
    #         Should be between 0 and 1.
    #     blend_psi : float, optional, by default 1.
    #         Blend coefficient used in trial_plasma_psi updates at step 3 of the algorithm above.
    #         Should be between 0 and 1.
    #     curr_eps : float, optional, by default 1e-5
    #         Used in calculating the relative convergence on the currents. Min value of the current
    #         step. Avoids divergence when dividing by the step in the currents.
    #     clip : float, optional, by default 5
    #         Used in the NK solvers. Maximum step size for each accepted basis vector, in units
    #         of the exploratory step.
    #     threshold : float, optional, by default 1.5
    #         Used in the NK solvers to catch cases of untreated (partial) collinearity.
    #         If relative_unexplained_residual>threshold, clip_hard is applied instead of clip.
    #     clip_hard : float, optional, by default 1.5
    #          Used in the NK solvers. Maximum step size for each accepted basis vector, in units
    #         of the exploratory step, for cases of partial collinearity.
    #     verbose : int, optional, by default T
    #         Printouts of convergence process.
    #         Use 1 for printouts with details on each NK cycle.
    #         Use 2 for printouts with deeper intermediate details.
    #     linear_only : bool, optional, by default False
    #         If linear_only = True the solution of the linearised problem is accepted.
    #         If linear_only = False, the convergence criteria are used and a solution of
    #         the full nonlinear problem is seeked.

    #     Returns
    #     -------
    #     int
    #         Number of grid points NOT in the reduced plasma domain that have some plasma in them (Jtor>0).
    #         Depending on the definition of the reduced plasma domain through plasma_domain_mask,
    #         this may mean the plasma contacted the wall. This will stop the dynamics.
    #     """

    #     # check if profile parameter (betap or paxis) is being altered
    #     # and action the change where necessary
    #     self.check_and_change_profiles(profile_parameter=profile_parameter,
    #                                    profile_coefficients=profile_coefficients)

    #     # solves the linearised problem for the currents.
    #     # needs to use the time derivativive of the profile parameters, if they have been changed
    #     if self.profile_change_flag:
    #         self.d_profile_pars_dt = self.d_profile_pars/self.dt_step
    #     else:
    #         self.d_profile_pars_dt = None
    #     self.set_linear_solution(active_voltage_vec, self.d_profile_pars_dt)
    #     # Solution and GS equilibrium are assigned to self.trial_currents and self.trial_plasma_psi

    #     if linear_only:
    #         # assign currents and plasma flux to self.currents_vec, self.eq1 and self.profile1 and complete step
    #         self.step_complete_assign(working_relative_tol_GS, from_linear=True)

    #     else:
    #         # seek solution of the full nonlinear problem

    #         # this assigns to self.eq2 and self.profiles2
    #         # also records self.tokamak_psi corresponding to self.trial_currents in 2d
    #         res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec).copy()

    #         # uses self.trial_currents and self.currents_vec_m1 to relate res_curr above to step in the currents
    #         rel_curr_res = 1.0*self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #         max_rel_curr_res = np.amax(rel_curr_res)
    #         control = max_rel_curr_res > target_relative_tol_currents

    #         # pair self.trial_currents and self.trial_plasma_psi are a GS solution
    #         r_res_GS = 0
    #         control_GS = 0

    #         max_rel_res = np.array([max_rel_curr_res, r_res_GS])
    #         target_tolerances = np.array([target_relative_tol_currents, target_relative_tol_GS])

    #         args_nk = [active_voltage_vec, self.rtol_NK]

    #         if verbose:
    #             print('starting numerical solve:')
    #             print('max(relative residual on current eqs) =', max_rel_curr_res, 'mean(residual on current eqs) =', np.mean(rel_curr_res))
    #             # print(self.F_function_ceq_GS(self.trial_currents, *args_nk))
    #         log = []

    #         # counter for instances in which the NK solver in psi has been shortcutted
    #         n_no_NK_psi = 0

    #         # counter for number of solution cycles
    #         n_it = 0

    #         while control:
    #             if verbose:
    #                 for _ in log:
    #                     print(_)

    #             log = [self.text_nk_cycle.format(nkcycle = n_it)]

    #             max_rel_res /= target_tolerances
    #             max_rel_res = np.where(max_rel_res<10, max_rel_res, 10)
    #             self.psi_gs_alpha = np.exp(max_rel_res)
    #             self.psi_gs_alpha /= np.sum(self.psi_gs_alpha)
    #             self.psi_gs_alpha = [0,1]
    #             print(max_rel_res/target_tolerances, self.psi_gs_alpha)

    #             # update plasma flux if trial_currents and plasma_flux exceedingly far from GS solution
    #             # if control_GS:
    #             #     self.NK.forward_solve(self.eq2, self.profiles2, self.rtol_NK)
    #             #     self.trial_plasma_psi *= (1 - blend_GS)
    #             #     self.trial_plasma_psi += blend_GS * self.eq2.plasma_psi

    #             # prepare for NK algorithms: 1d vectors needed for independent variable
    #             self.trial_plasma_psi = self.trial_plasma_psi.reshape(-1)
    #             self.tokamak_psi = self.tokamak_psi.reshape(-1)

    #             # calculate initial residual for the root problem in psi
    #             res_psi = self.F_function_psi_GS(trial_plasma_psi=self.trial_plasma_psi,
    #                                             active_voltage_vec=active_voltage_vec,
    #                                             rtol_NK=self.rtol_NK).copy()
    #             del_res_psi = (np.amax(res_psi) - np.amin(res_psi))
    #             # print('del_res_psi', del_res_psi)

    #             if (del_res_psi > self.rtol_NK/relative_tol_for_nk_psi)+(n_no_NK_psi > max_no_NK_psi):
    #                 n_no_NK_psi = 0
    #                 # NK algorithm to solve the root problem in psi
    #                 self.psi_nk_solver.Arnoldi_iteration(x0=self.trial_plasma_psi, #trial_current expansion point
    #                                                     dx=res_psi.copy(), #first vector for current basis
    #                                                     R0=res_psi, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                                     F_function=self.F_function_psi_GS,
    #                                                     args=args_nk,
    #                                                     step_size=step_size_psi,
    #                                                     scaling_with_n=scaling_with_n,
    #                                                     target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector
    #                                                     max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                                     max_Arnoldi_iterations=max_Arnoldi_iterations,
    #                                                     max_collinearity=max_collinearity,
    #                                                     clip=clip,
    #                                                     threshold=threshold,
    #                                                     clip_hard=clip_hard)

    #                 # update trial_plasma_psi according to NK solution
    #                 self.trial_plasma_psi += self.psi_nk_solver.dx*blend_psi
    #                 psi_text = [[self.text_psi_1, self.psi_nk_solver.coeffs]]

    #                 res_psi = self.F_function_psi_GS(trial_plasma_psi=self.trial_plasma_psi,
    #                                             active_voltage_vec=active_voltage_vec,
    #                                             rtol_NK=self.rtol_NK).copy()
    #                 del_res_psi = (np.amax(res_psi) - np.amin(res_psi))
    #                 # print('del_res_psi', del_res_psi)

    #             else:
    #                 # NK algorithm has been shortcutted, keep count
    #                 n_no_NK_psi += 1
    #                 psi_text = [self.text_psi_0.format(skippedno = n_no_NK_psi, psi_res = del_res_psi)]

    #             # prepare for NK solver on the currents, 2d plasma flux needed
    #             self.trial_plasma_psi = self.trial_plasma_psi.reshape(self.nx, self.ny)

    #             # calculates initial residual for the root problem in the currents
    #             # assumes the just updated self.trial_plasma_psi
    #             res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec).copy()
    #             rel_curr_res = abs(res_curr / self.curr_step)
    #             interm_text = ['The intermediate residuals on the current: max =', np.amax(rel_curr_res), 'mean =', np.mean(rel_curr_res)]

    #             if verbose-1:
    #                 log.append(psi_text)
    #                 log.append(interm_text)

    #             # NK algorithm to solve the root problem in the currents
    #             self.currents_nk_solver.Arnoldi_iteration(  x0=self.trial_currents,
    #                                                         dx=res_curr.copy(),
    #                                                         R0=res_curr,
    #                                                         F_function=self.F_function_curr,
    #                                                         args=[active_voltage_vec],
    #                                                         step_size=step_size_curr,
    #                                                         scaling_with_n=scaling_with_n,
    #                                                         target_relative_unexplained_residual=target_relative_unexplained_residual,
    #                                                         max_n_directions=max_n_directions,
    #                                                         max_Arnoldi_iterations=max_Arnoldi_iterations,
    #                                                         max_collinearity=max_collinearity,
    #                                                         clip=clip,
    #                                                         threshold=threshold,
    #                                                         clip_hard=clip_hard)
    #             # update trial_currents according to NK solution
    #             self.trial_currents += self.currents_nk_solver.dx

    #             # check convergence properties of the pair [trial_currents, trial_plasma_psi]:
    #             # relative convergence on the currents:
    #             res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec).copy()
    #             rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #             max_rel_curr_res = np.amax(rel_curr_res)
    #             control = max_rel_curr_res > target_relative_tol_currents
    #             # control = np.any(rel_curr_res > target_relative_tol_currents)
    #             # relative convergence on the GS problem
    #             r_res_GS = 1.0*self.calculate_rel_tolerance_GS(self.trial_plasma_psi)
    #             control_GS = (r_res_GS > target_relative_tol_GS)
    #             control += control_GS
    #             max_rel_res = np.array([max_rel_curr_res, r_res_GS])

    #             log.append(['The coeffs applied to the current vec = ', self.currents_nk_solver.coeffs])
    #             log.append(['The final residual on the current (relative): max =', np.amax(rel_curr_res), 'mean =', np.mean(rel_curr_res)])
    #             self.ceq_res = 1.0*self.F_function_ceq_GS(self.trial_currents, *args_nk)
    #             log.append(['The final residual on the current (relative): max =', np.amax(self.ceq_res), 'mean =', np.mean(self.ceq_res)])
    #             log.append(['Residuals on GS eq (relative): ', r_res_GS])

    #             # one full cycle completed
    #             n_it += 1

    #         # convergence checks succeeded, complete step
    #         self.step_complete_assign(working_relative_tol_GS)

    #     # # check plasma is still fully contained in the plasma reduced domain
    #     # flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)

    #     # return flag

    # def nlstepper3(self, active_voltage_vec,
    #                     profile_parameter=None,
    #                     profile_coefficients=None,
    #                     target_relative_tol_currents=.005,
    #                     target_relative_tol_GS=.002,
    #                     working_relative_tol_GS=.0005,
    #                     target_relative_unexplained_residual=.5,
    #                     max_n_directions=5,
    #                     max_Arnoldi_iterations=6,
    #                     max_collinearity=.3,
    #                     step_size_psi=2.,
    #                     step_size_curr=.8,
    #                     scaling_with_n=0,
    #                     relative_tol_for_nk_psi=.002,
    #                     blend_GS=.5,
    #                     blend_psi=1,
    #                     curr_eps=1e-5,
    #                     max_no_NK_psi=1.,
    #                     clip=5,
    #                     threshold=1.5,
    #                     clip_hard=1.5,
    #                     verbose=0,
    #                     linear_only=False):
    #     """The main stepper function.
    #     If linear_only = True, this advances the linearised problem.
    #     If linear_only = False, a solution of the full non-linear problem is seeked using
    #     a combination of NK methods.
    #     When a solution has been found, time is advanced by self.dt_step,
    #     currents are recorded in self.currents_vec and profile properties
    #     in self.eq1 and self.profiles1.
    #     The solver's algorithm proceeds like below:
    #     1) solve linearised problem for initial guess of the currents and solve associated GS,
    #     assign trial_plasma_psi and trial_currents (and consequent tokamak_psi);
    #     2) if pair [trial_plasma_psi, tokamak_psi] fails static GS check (control_GS),
    #     update trial_plasma_psi using GS solution;
    #     3) at fixed trial_currents (and consequent tokamak_psi) update trial_plasma_psi
    #     using NK solver for the associated root problem;
    #     4) at fixed trial_plasma_psi, update trial_currents (and consequent tokamak_psi)
    #     using NK solver for the associated root problem;
    #     5) if convergence on the current residuals is not achieved or static GS check
    #     fails, restart from point 2;
    #     6) the pair [trial_currents, trial_plasma_psi] solves the nonlinear dynamic problem,
    #     assign values to self.currents_vec, self.eq1 and self.profiles1.

    #     Parameters
    #     ----------
    #     active_voltage_vec : np.array
    #         Vector of active voltages for the active coils, applied between t and t+dt.
    #     profile_parameter : None or float for new paxis or betap
    #         Set to None when the profile parameter (paxis or betap) is left unchanged
    #         with respect to the previous timestep. Set here desired value otherwise.
    #     profile_coefficients : None or tuple (alpha_m, alpha_n)
    #         Set to None when the profile coefficients alpha_m and alpha_n are left unchanged
    #         with respect to the previous timestep. Set here desired values otherwise.
    #     target_relative_tol_currents : float, optional, by default .01
    #         Relative tolerance in the currents required for convergence.
    #     target_relative_tol_GS : float, optional, by default .01
    #         Relative tolerance in the plasma flux required for convergence.
    #     working_relative_tol_GS : float, optional, by default .002
    #         Tolerance used when solving all static GS problems, expressed in
    #         terms of the change in the plasma flux due to 1 timestep of evolution.
    #     target_relative_unexplained_residual : float, optional, by default .5
    #         Used in the NK solvers. Inclusion of additional basis vectors is
    #         stopped if the fraction of unexplained_residual is < target_relative_unexplained_residual.
    #     max_n_directions : int, optional, by default 3
    #         Used in the NK solvers. Inclusion of additional basis vectors is
    #         stopped if max_n_directions have already been included.
    #     max_Arnoldi_iterations : int, optional, by default 4
    #         Used in the NK solvers. Inclusion of additional basis vectors is
    #         stopped if max_n_directions have already been considered for inclusion,
    #         though not necessarily included.
    #     max_collinearity : float, optional, by default .3
    #         Used in the NK solvers. The basis vector being considered is rejected
    #         if scalar product with any of the previously included is larger than max_collinearity.
    #     step_size_psi : float, optional, by default 2.
    #         Used by the NK solver applied to the root problem in the plasma flux.
    #         l2 norm of proposed step.
    #     step_size_curr : float, optional, by default .8
    #         Used by the NK solver applied to the root problem in the currents.
    #         l2 norm of proposed step.
    #     scaling_with_n : int, optional, by default 0
    #         Used in the NK solvers. Allows to further scale dx candidate steps by factor
    #         (1 + self.n_it)**scaling_with_n
    #     relative_tol_for_nk_psi : float, optional, by default .002
    #         NK solver for the root problem in the plasma flux is not used if
    #         the associated residual is < self.rtol_NK/relative_tol_for_nk_psi
    #     max_no_NK_psi : float, optional, by default 1.
    #         Maximum number of consecutive times the NK solver for the root problem in the plasma flux
    #         can be shortcutted.
    #     blend_GS : float, optional, by default .5
    #         Blend coefficient used in trial_plasma_psi updates at step 2 of the algorithm above.
    #         Should be between 0 and 1.
    #     blend_psi : float, optional, by default 1.
    #         Blend coefficient used in trial_plasma_psi updates at step 3 of the algorithm above.
    #         Should be between 0 and 1.
    #     curr_eps : float, optional, by default 1e-5
    #         Used in calculating the relative convergence on the currents. Min value of the current
    #         step. Avoids divergence when dividing by the step in the currents.
    #     clip : float, optional, by default 5
    #         Used in the NK solvers. Maximum step size for each accepted basis vector, in units
    #         of the exploratory step.
    #     threshold : float, optional, by default 1.5
    #         Used in the NK solvers to catch cases of untreated (partial) collinearity.
    #         If relative_unexplained_residual>threshold, clip_hard is applied instead of clip.
    #     clip_hard : float, optional, by default 1.5
    #          Used in the NK solvers. Maximum step size for each accepted basis vector, in units
    #         of the exploratory step, for cases of partial collinearity.
    #     verbose : int, optional, by default T
    #         Printouts of convergence process.
    #         Use 1 for printouts with details on each NK cycle.
    #         Use 2 for printouts with deeper intermediate details.
    #     linear_only : bool, optional, by default False
    #         If linear_only = True the solution of the linearised problem is accepted.
    #         If linear_only = False, the convergence criteria are used and a solution of
    #         the full nonlinear problem is seeked.

    #     Returns
    #     -------
    #     int
    #         Number of grid points NOT in the reduced plasma domain that have some plasma in them (Jtor>0).
    #         Depending on the definition of the reduced plasma domain through plasma_domain_mask,
    #         this may mean the plasma contacted the wall. This will stop the dynamics.
    #     """

    #     self.psi_gs_alpha = [0,1]

    #     # check if profile parameter (betap or paxis) is being altered
    #     # and action the change where necessary
    #     self.check_and_change_profiles(profile_parameter=profile_parameter,
    #                                    profile_coefficients=profile_coefficients)

    #     # solves the linearised problem for the currents.
    #     # needs to use the time derivativive of the profile parameters, if they have been changed
    #     if self.profile_change_flag:
    #         self.d_profile_pars_dt = self.d_profile_pars/self.dt_step
    #     else:
    #         self.d_profile_pars_dt = None
    #     self.set_linear_solution(active_voltage_vec, self.d_profile_pars_dt)
    #     # Solution and GS equilibrium are assigned to self.trial_currents and self.trial_plasma_psi

    #     if linear_only:
    #         # assign currents and plasma flux to self.currents_vec, self.eq1 and self.profile1 and complete step
    #         self.step_complete_assign(working_relative_tol_GS, from_linear=True)

    #     else:
    #         # seek solution of the full nonlinear problem

    #         # this assigns to self.eq2 and self.profiles2
    #         # also records self.tokamak_psi corresponding to self.trial_currents in 2d
    #         res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec).copy()

    #         # uses self.trial_currents and self.currents_vec_m1 to relate res_curr above to step in the currents
    #         rel_curr_res = 1.0*self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #         max_rel_curr_res = np.amax(rel_curr_res)
    #         control = max_rel_curr_res > target_relative_tol_currents

    #         # pair self.trial_currents and self.trial_plasma_psi are a GS solution
    #         r_res_GS = 0
    #         control_GS = 0

    #         target_tolerances = np.array([target_relative_tol_currents, target_relative_tol_GS])

    #         args_nk = [active_voltage_vec, self.rtol_NK]

    #         if verbose:
    #             print('starting numerical solve:')
    #             print('max(relative residual on current eqs) =', max_rel_curr_res, 'mean(residual on current eqs) =', np.mean(rel_curr_res))
    #             # print(self.F_function_ceq_GS(self.trial_currents, *args_nk))
    #         log = []

    #         # counter for instances in which the NK solver in psi has been shortcutted
    #         n_no_NK_psi = 0

    #         # counter for number of solution cycles
    #         n_it = 0

    #         while control:
    #             if verbose:
    #                 for _ in log:
    #                     print(_)

    #             log = [self.text_nk_cycle.format(nkcycle = n_it)]

    #             self.currents_nk_solver.Arnoldi_iteration(  x0=self.trial_currents,
    #                                                         dx=res_curr.copy(),
    #                                                         R0=res_curr,
    #                                                         F_function=self.F_function_curr,
    #                                                         args=[active_voltage_vec],
    #                                                         step_size=step_size_curr,
    #                                                         scaling_with_n=scaling_with_n,
    #                                                         target_relative_unexplained_residual=target_relative_unexplained_residual,
    #                                                         max_n_directions=max_n_directions,
    #                                                         max_Arnoldi_iterations=max_Arnoldi_iterations,
    #                                                         max_collinearity=max_collinearity,
    #                                                         clip=clip,
    #                                                         threshold=threshold,
    #                                                         clip_hard=clip_hard)
    #             # update trial_currents according to NK solution
    #             self.trial_currents += self.currents_nk_solver.dx

    #             res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec).copy()

    #             # uses self.trial_currents and self.currents_vec_m1 to relate res_curr above to step in the currents
    #             rel_curr_res = 1.0*self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #             max_rel_curr_res = np.amax(rel_curr_res)

    #             r_res_GS = 1.0*self.calculate_rel_tolerance_GS(self.trial_plasma_psi)
    #             max_rel_res = np.array([max_rel_curr_res, r_res_GS])

    #             max_rel_res /= target_tolerances
    #             max_rel_res = np.where(max_rel_res<10, max_rel_res, 10)
    #             self.psi_gs_alpha = np.exp(max_rel_res)
    #             self.psi_gs_alpha /= np.sum(self.psi_gs_alpha)

    #             print(max_rel_res/target_tolerances, self.psi_gs_alpha)

    #             # prepare for NK algorithms: 1d vectors needed for independent variable
    #             # self.trial_plasma_psi = self.trial_plasma_psi.reshape(-1)
    #             # self.tokamak_psi = self.tokamak_psi.reshape(-1)

    #             # calculate initial residual for the root problem in psi
    #             res_psi = self.F_function_psi_GS(trial_plasma_psi=self.trial_plasma_psi,
    #                                              active_voltage_vec=active_voltage_vec,
    #                                              rtol_NK=self.rtol_NK).copy()
    #             del_res_psi = (np.amax(res_psi) - np.amin(res_psi))
    #             # print('del_res_psi', del_res_psi)

    #             if (del_res_psi > self.rtol_NK/relative_tol_for_nk_psi)+(n_no_NK_psi > max_no_NK_psi):
    #                 n_no_NK_psi = 0
    #                 # NK algorithm to solve the root problem in psi
    #                 self.psi_nk_solver.Arnoldi_iteration(x0=self.trial_plasma_psi, #trial_current expansion point
    #                                                     dx=res_psi.copy(), #first vector for current basis
    #                                                     R0=res_psi, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                                     F_function=self.F_function_psi_GS,
    #                                                     args=args_nk,
    #                                                     step_size=step_size_psi,
    #                                                     scaling_with_n=scaling_with_n,
    #                                                     target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector
    #                                                     max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                                     max_Arnoldi_iterations=max_Arnoldi_iterations,
    #                                                     max_collinearity=max_collinearity,
    #                                                     clip=clip,
    #                                                     threshold=threshold,
    #                                                     clip_hard=clip_hard)

    #                 # update trial_plasma_psi according to NK solution
    #                 self.trial_plasma_psi += self.psi_nk_solver.dx*blend_psi
    #                 psi_text = [[self.text_psi_1, self.psi_nk_solver.coeffs]]

    #                 res_psi = self.F_function_psi_GS(trial_plasma_psi=self.trial_plasma_psi,
    #                                             active_voltage_vec=active_voltage_vec,
    #                                             rtol_NK=self.rtol_NK).copy()
    #                 del_res_psi = (np.amax(res_psi) - np.amin(res_psi))
    #                 # print('del_res_psi', del_res_psi)

    #             else:
    #                 # NK algorithm has been shortcutted, keep count
    #                 n_no_NK_psi += 1
    #                 psi_text = [self.text_psi_0.format(skippedno = n_no_NK_psi, psi_res = del_res_psi)]

    #             # prepare for NK solver on the currents, 2d plasma flux needed
    #             self.trial_plasma_psi = self.trial_plasma_psi.reshape(self.nx, self.ny)

    #             # # calculates initial residual for the root problem in the currents
    #             # # assumes the just updated self.trial_plasma_psi
    #             # res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec).copy()
    #             # rel_curr_res = abs(res_curr / self.curr_step)
    #             # interm_text = ['The intermediate residuals on the current: max =', np.amax(rel_curr_res), 'mean =', np.mean(rel_curr_res)]

    #             # if verbose-1:
    #             #     log.append(psi_text)
    #             #     log.append(interm_text)

    #             # # NK algorithm to solve the root problem in the currents
    #             # self.currents_nk_solver.Arnoldi_iteration(  x0=self.trial_currents,
    #             #                                             dx=res_curr.copy(),
    #             #                                             R0=res_curr,
    #             #                                             F_function=self.F_function_curr,
    #             #                                             args=[active_voltage_vec],
    #             #                                             step_size=step_size_curr,
    #             #                                             scaling_with_n=scaling_with_n,
    #             #                                             target_relative_unexplained_residual=target_relative_unexplained_residual,
    #             #                                             max_n_directions=max_n_directions,
    #             #                                             max_Arnoldi_iterations=max_Arnoldi_iterations,
    #             #                                             max_collinearity=max_collinearity,
    #             #                                             clip=clip,
    #             #                                             threshold=threshold,
    #             #                                             clip_hard=clip_hard)
    #             # # update trial_currents according to NK solution
    #             # self.trial_currents += self.currents_nk_solver.dx

    #             # check convergence properties of the pair [trial_currents, trial_plasma_psi]:
    #             # relative convergence on the currents:
    #             res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec).copy()
    #             rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
    #             max_rel_curr_res = np.amax(rel_curr_res)
    #             control = max_rel_curr_res > target_relative_tol_currents
    #             # control = np.any(rel_curr_res > target_relative_tol_currents)
    #             # relative convergence on the GS problem
    #             r_res_GS = 1.0*self.calculate_rel_tolerance_GS(self.trial_plasma_psi)
    #             control_GS = (r_res_GS > target_relative_tol_GS)
    #             control += control_GS

    #             log.append(['The coeffs applied to the current vec = ', self.currents_nk_solver.coeffs])
    #             log.append(['The final residual on the current (relative): max =', np.amax(rel_curr_res), 'mean =', np.mean(rel_curr_res)])
    #             self.ceq_res = 1.0*self.F_function_ceq_GS(self.trial_currents, *args_nk)
    #             log.append(['The final residual on the current (relative): max =', np.amax(self.ceq_res), 'mean =', np.mean(self.ceq_res)])
    #             log.append(['Residuals on GS eq (relative): ', r_res_GS])

    #             # one full cycle completed
    #             n_it += 1

    #         # convergence checks succeeded, complete step
    #         self.step_complete_assign(working_relative_tol_GS)

    #     # # check plasma is still fully contained in the plasma reduced domain
    #     # flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)

    #     # return flag
