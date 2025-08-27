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
    """Interface between the linearised system of circuit equations and the implicit-Euler
    time stepper. Calculates the linear growth rate and solves the linearised dynamical problem.
    It needs the Jacobian of the plasma current distribution with respect to all of the
    independent currents, dIy/dI.
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
        """Instantiates the linear_solver object, with inputs computed mostly
        within the circuit_equation_metals object.
        Based on the input plasma properties and coupling matrices, it prepares:
        - an instance of the implicit Euler solver implicit_euler_solver()
        - internal time-stepper for the implicit-Euler

        Parameters
        ----------
        eq : class
            FreeGSNKE equilibrium object.
        Lambdam1: np.array
            State matrix of the circuit equations for the metal in normal mode form:
            P is the identity on the active coils and diagonalises the isolated dynamics
            of the passive coils, R^{-1}L_{passive}
        P: np.array
            change of basis matrix, as defined above, with modes appropriately removed
        Pm1: np.array
            Inverse of the change of basis matrix, as defined above, with modes appropriately removed
        Rm1: np.array
            matrix of all metal resitances to the power of -1. Diagonal.
        Mey: np.array
            matrix of inductance values between grid points in the reduced plasma domain and all metal coils
            (active coils and passive-structure filaments)
            Calculated by the metal_currents object
        plasma_norm_factor: float
            an overall factor to work with a rescaled plasma current, so that
            it's within a comparable range with metal currents
        max_internal_timestep: float
            internal integration timestep of the implicit-Euler solver, to be used
            as substeps over the <<full_timestep>> interval
        full_timestep: float
            full timestep requested to the implicit-Euler solver
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
        """Resets the value of the plasma resistivity,
        throught the vector of 'geometric restistances' in the plasma domain

        Parameters
        ----------
        plasma_resistance_1d : ndarray
            Vector of 2pi resistivity R/dA for all domain grid points in the reduced plasma domain
        """
        self.plasma_resistance_1d = plasma_resistance_1d
        self.set_linearization_point(None, None, None, None)

    def reset_timesteps(self, max_internal_timestep, full_timestep):
        """Resets the integration timesteps, calling self.solver.set_timesteps

        Parameters
        ----------
        max_internal_timestep : float
            integration substep of the ciruit equation, calling an implicit-Euler solver
        full_timestep : float
            integration timestep of the circuit equation
        """
        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.solver.set_timesteps(
            full_timestep=full_timestep, max_internal_timestep=max_internal_timestep
        )

    def set_linearization_point(self, dIydI, dIydtheta, hatIy0, Myy_hatIy0):
        """Initialises an implicit-Euler solver with the appropriate matrices for the linearised dynamic problem.

        Parameters
        ----------
        dIydI : np.array
            partial derivatives of plasma-cell currents on the reduced plasma domain with respect to all intependent metal currents
            (active coil currents, vessel normal modes, total plasma current divided by plasma_norm_factor).
            These would typically come from having solved the forward Grad-Shafranov problem. Finite difference Jacobian.
            Calculated by the nl_solver object
        dIydtheta : np.array
            partial derivatives of plasma-cell currents on the reduced plasma domain with respect to all plasma current density
            profile parameters
        hatIy0 : np.array
            Plasma current distribution on the reduced plasma domain (1d) of the equilibrium around which the dynamics is linearised.
            This is normalised by the total plasma current, so that this vector sums to 1.
        Myy_hatIy0 : np.array
            The matrix product np.dot(Myy, hatIy0) in the same reduced domain as hatIy0
            This is provided by Myy_handler
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
