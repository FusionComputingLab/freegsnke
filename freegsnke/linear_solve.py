"""
Implements the object that advances the linearised dynamics.

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
from scipy.linalg import solve, solve_sylvester

from .implicit_euler import implicit_euler_solver


class linear_solver:
    """
    Linearised plasma–metal circuit solver.

    This class provides the interface between the linearised system of
    coupled plasma and metal circuit equations and the implicit-Euler
    time integrator. It constructs the matrices required for time
    integration, computes linear growth rates, and advances the
    linearised dynamical system in time.

    The metal circuit equations are expressed in a reduced basis defined
    by the mode decomposition matrix ``P``, in which the active coils
    remain in the physical current basis and the passive structures are
    represented by vessel eigenmodes.

    The solver requires the Jacobian of the plasma current distribution
    with respect to the independent current variables, ``dIy/dI``, which
    determines the linear plasma response.
    """

    def __init__(
        self,
        coil_numbers,
        Lambdam1,
        P,
        Pm1,
        Rm1,
        Mey,
        plasma_norm_factor,
        plasma_resistance_1d,
        max_internal_timestep=0.0001,
        full_timestep=0.0001,
    ):
        """
        Initialise the linearised circuit solver.

        Precomputes matrices describing the coupling between plasma currents
        and metal circuits in the reduced modal basis, and creates the
        implicit-Euler integrator used to advance the linearised system.

        Parameters
        ----------
        coil_numbers : tuple[int, int]
            Tuple ``(n_active_coils, n_coils)`` giving the number of active
            coils and the total number of conducting elements.
        Lambdam1 : ndarray
            State matrix of the metal circuit equations in the reduced basis.
            Active coils remain in the physical current basis, while passive
            structures are represented by vessel eigenmodes.
        P : ndarray
            Change-of-basis matrix mapping currents from the reduced modal
            basis to the physical conductor basis.
        Pm1 : ndarray
            Inverse of ``P``, mapping currents from the physical conductor
            basis to the reduced modal basis.
        Rm1 : ndarray
            Inverse resistance matrix of all conducting elements. This is
            typically diagonal.
        Mey : ndarray
            Mutual inductance matrix between plasma current elements and
            metal conductors (active coils and passive structures).
        plasma_norm_factor : float
            Scaling factor applied to the plasma current variable to improve
            conditioning and keep its magnitude comparable to metal currents.
        plasma_resistance_1d : ndarray
            Effective plasma resistance associated with each plasma current
            element in the reduced plasma domain.
        max_internal_timestep : float, optional
            Maximum timestep used internally by the implicit-Euler solver.
            Longer requested timesteps are subdivided into smaller steps.
        full_timestep : float, optional
            External timestep associated with a single solver advance.
        """

        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.plasma_norm_factor = plasma_norm_factor

        self.P = P
        self.Pm1 = Pm1
        self.Rm1 = Rm1
        # self.RP = np.diag(eq.tokamak.coil_resist) @ P
        # self.RP_inv = np.linalg.solve(self.RP.T @ self.RP, self.RP.T)
        # self.RP_inv_Mey = np.matmul(self.RP_inv, Mey)
        self.Pm1Rm1 = Pm1 @ Rm1
        self.Pm1Rm1Mey = np.matmul(self.Pm1Rm1, Mey)
        self.MyeP = np.matmul(Mey.T, P).T

        # if Lambdam1 is None:
        #     self.Lambdam1 = Pm1 @ (Rm1 @ (eq.tokamak.coil_self_ind @ P))
        # else:
        self.Lambdam1 = Lambdam1
        self.n_independent_vars = np.shape(self.Lambdam1)[0]

        self.Mmatrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )
        self.M0matrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )
        self.dMmatrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )

        self.n_active_coils, self.n_coils = coil_numbers

        self.solver = implicit_euler_solver(
            Mmatrix=np.eye(self.n_independent_vars + 1),
            Rmatrix=np.eye(self.n_independent_vars + 1),
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.full_timestep,
        )

        self.plasma_resistance_1d = plasma_resistance_1d

        # dummy vessel voltage vector
        self.empty_U = np.zeros(np.shape(self.Pm1Rm1)[1])
        # dummy voltage vec for eig modes
        self.forcing = np.zeros(self.n_independent_vars + 1)
        self.profiles_forcing = np.zeros(self.n_independent_vars + 1)

    def reset_plasma_resistivity(self, plasma_resistance_1d):
        """
        Update the plasma resistivity profile used by the linearised solver.

        Replaces the vector of effective plasma resistances associated with
        the reduced plasma domain. This automatically invalidates the current
        linearisation point so that system matrices will be rebuilt using the
        updated resistivity.

        Parameters
        ----------
        plasma_resistance_1d : ndarray
            Vector of effective plasma resistance coefficients for all grid
            points in the reduced plasma domain. Each entry represents the
            geometric resistance factor (2π * resistivity divided by cell area).
        """
        self.plasma_resistance_1d = plasma_resistance_1d
        self.set_linearization_point(None, None, None, None)

    def reset_timesteps(self, max_internal_timestep, full_timestep):
        """
        Update the timesteps used by the implicit-Euler integrator.

        Parameters
        ----------
        max_internal_timestep : float
            Maximum internal timestep used by the implicit-Euler solver.
            Larger timesteps are subdivided into steps no larger than this value.
        full_timestep : float
            External timestep associated with a single advance of the
            circuit equations.
        """
        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.solver.set_timesteps(
            full_timestep=full_timestep, max_internal_timestep=max_internal_timestep
        )

    def set_linearization_point(self, dIydI, dIydtheta, hatIy0, Myy_hatIy0):
        """
        Set or update the linearisation point for the coupled plasma–metal system.

        This method defines the equilibrium state around which the system is
        linearised and rebuilds the system matrices used by the implicit-Euler
        solver.

        Parameters
        ----------
        dIydI : ndarray or None
            Jacobian of plasma cell currents with respect to independent metal
            currents (active coils, vessel modes, and normalised total plasma
            current). Typically computed from a Grad–Shafranov solve using finite
            differences.
        dIydtheta : ndarray or None
            Jacobian of plasma cell currents with respect to plasma profile
            parameterisation variables.
        hatIy0 : ndarray or None
            Normalised equilibrium plasma current distribution over the reduced
            plasma domain. This vector is scaled to sum to one.
        Myy_hatIy0 : ndarray or None
            Precomputed product of the plasma–plasma coupling matrix with the
            equilibrium current distribution, provided by the plasma coupling
            handler.

        Notes
        -----
        Any argument set to None is left unchanged from the previous linearisation
        point.

        This call rebuilds the system matrix and reinitialises the implicit-Euler
        solver using the updated linearised dynamics.
        """
        if dIydI is not None:
            self.dIydI = dIydI
        if dIydtheta is not None:
            self.dIydtheta = dIydtheta
        if hatIy0 is not None:
            self.hatIy0 = hatIy0
        if Myy_hatIy0 is not None:
            self.Myy_hatIy0 = Myy_hatIy0

        self.build_Mmatrix()

        self.solver = implicit_euler_solver(
            Mmatrix=self.Mmatrix,
            Rmatrix=np.eye(self.n_independent_vars + 1),
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.full_timestep,
        )

    def build_Mmatrix(
        self,
    ):
        """Initialises the pseudo-inductance matrix of the problem
        M\dot(x) + Rx = forcing
        using the linearisation Jacobian.

                          \Lambda^-1 + P^-1R^-1Mey A        P^-1R^-1Mey B
        M = M0 + dM =  (                                                       )
                           J(Myy A + MyeP)/Rp                J Myy B/Rp

        This also builds the forcing:
                    P^-1R^-1 Voltage         P^-1R^-1Mey
        forcing = (                   ) - (                 ) C \dot{theta}
                            0                  J Myy/Rp

        where A = dIy/dId
              B = dIy/dIp
              C = dIy/plasmapars

        Parameters
        ----------
        None given explicitly, they are all given by the object attributes.

        """

        nRp = (
            np.sum(self.plasma_resistance_1d * self.hatIy0 * self.hatIy0)
            * self.plasma_norm_factor
        )

        # M0 matrix
        self.M0matrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )
        # metal-metal before plasma
        self.M0matrix[: self.n_independent_vars, : self.n_independent_vars] = np.copy(
            self.Lambdam1
        )
        # metal to plasma
        self.M0matrix[-1, :-1] = np.dot(self.MyeP, self.hatIy0)
        self.M0matrix[-1, :] /= nRp

        # dM matrix
        self.dMmatrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )
        # metal-metal plasma-mediated
        self.dMmatrix[: self.n_independent_vars, : self.n_independent_vars] = np.matmul(
            self.Pm1Rm1Mey, self.dIydI[:, :-1]
        )
        # plasma to metal
        self.dMmatrix[:-1, -1] = np.dot(self.Pm1Rm1Mey, self.dIydI[:, -1])
        # metal to plasma plasma-mediated
        self.dMmatrix[-1, :-1] = np.dot(self.dIydI[:, :-1].T, self.Myy_hatIy0)
        self.dMmatrix[-1, -1] = np.dot(self.dIydI[:, -1], self.Myy_hatIy0)
        self.dMmatrix[-1, :] /= nRp

        self.Mmatrix = self.M0matrix + self.dMmatrix

        # build necessary terms to incorporate forcing term from variations of the profile parameters
        # MIdot + RI = V - self.Vm1Rm12Mey_plus@self.dIydpars@d_profiles_pars_dt
        self.forcing_pars_matrix = None
        if self.dIydtheta is not None:
            Pm1Rm1Mey_plus = np.concatenate(
                (self.Pm1Rm1Mey, self.Myy_hatIy0[np.newaxis] / nRp), axis=0
            )
            self.forcing_pars_matrix = np.matmul(Pm1Rm1Mey_plus, self.dIydtheta)

    def stepper(
        self,
        It,
        active_voltage_vec,
        dtheta_dt,
    ):
        """Executes the time advancement. Uses the implicit_euler instance.

        Parameters
        ----------
        It : np.array
            vector of all independent currents that are solved for by the linearides problem, in terms of normal modes:
            (active currents, vessel normal modes, total plasma current divided by normalisation factor)
        active_voltage_vec : np.array
            voltages applied to the active coils
        dtheta_dt : np.array
            Vector of plasma current density profile parameters derivateives with respect to t.

        Returns
        -------
        Itpdt : ndarray
            Updated state vector after one full implicit-Euler timestep.
        """

        # baseline forcing term (from the active coil voltages)
        self.empty_U[: self.n_active_coils] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Pm1Rm1, self.empty_U)
        self.forcing[-1] = 0.0

        # additional forcing due to the time derivative of profile parameters
        if self.forcing_pars_matrix is not None:
            self.forcing -= np.dot(self.forcing_pars_matrix, dtheta_dt)

        Itpdt = self.solver.full_stepper(It, self.forcing)

        return Itpdt

    def calculate_linear_growth_rate(
        self,
    ):
        """Looks into the eigenvecotrs of the "M" matrix to find the negative singular values,
        which correspond to the growth rates of instabilities.

        Parameters
        ----------
        parameters are passed in as object attributes
        """

        # full set of characteristic timescales (circuits + plasma)
        evalues, evectors = np.linalg.eig(self.Mmatrix)
        # ord = np.argsort(evalues)
        self.all_timescales = -evalues  # [ord]
        self.all_modes = evectors  # [:, ord]

        # extract just the positive (i.e. unstable) eigenvalues
        mask = self.all_timescales > 0
        self.instability_timescale = self.all_timescales[mask]
        self.growth_rates = 1 / self.instability_timescale

        # full set of characteristic timescales (circuits only, no plasma)
        evalues, evectors = np.linalg.eig(self.Mmatrix[:-1, :-1])
        # ord = np.argsort(evalues)
        self.all_timescales_const_Ip = -evalues  # [ord]
        self.all_modes_const_Ip = evectors  # [:, ord]

        # extract just the positive (i.e. unstable) eigenvalues
        mask = self.all_timescales_const_Ip > 0
        self.instability_timescale_const_Ip = self.all_timescales_const_Ip[mask]
        self.growth_rates_const_Ip = 1 / self.instability_timescale_const_Ip

        # extract the unstable mode in this case, used in other calculations
        self.unstable_modes = self.all_modes_const_Ip[:, mask]
        self.unstable_modes /= np.linalg.norm(self.unstable_modes, axis=0)

    def calculate_pseudo_rigid_projections(self, dRZdI):
        """Projects the unstable modes on the vectors of currents
        which best isolate an R or a Z movement of the plasma


        Parameters
        ----------
        dRZdI : np.array
            Jacobian of Rcurrent and Zcurrent shifts wrt the modes,
            as calculated in nonlinear_solve

        Returns
        -------
        np.array
            proj[i,0] is the scalar product of the unstable mode i on the vector of modes resulting in an Rcurrent shift
            proj[i,1] is the scalar product of the unstable mode i on the vector of modes resulting in an Zcurrent shift
        """

        # calculate vectors of currents for R and Z movements
        rigid_VC = np.linalg.pinv(dRZdI[:, :-1])
        rigid_VC /= np.linalg.norm(rigid_VC, axis=0)
        # project on unstable mode
        proj = np.sum(
            rigid_VC[:, np.newaxis, :] * self.unstable_modes[:, :, np.newaxis], axis=0
        )
        return proj

    def calculate_stability_margin(
        self,
    ):
        """
        Here we calculate the stability margin parameter from:

        https://iopscience.iop.org/article/10.1088/0029-5515/45/8/021

        Parameters
        ----------
        parameters are passed in as object attributes
        """

        # extract the L and S matrices
        n = self.n_independent_vars
        L = self.M0matrix[0:n, 0:n]
        S = -self.dMmatrix[0:n, 0:n]

        # find e'values
        A = np.linalg.solve(L, S) - np.eye(n)
        self.all_stability_margins = np.sort(np.linalg.eigvals(A))

        # extract stability margin
        mask = self.all_stability_margins > 0
        self.stability_margin = self.all_stability_margins[
            mask
        ]  # the positive (i.e. unstable) eigenvalues
