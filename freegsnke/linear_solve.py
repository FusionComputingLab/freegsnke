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
from scipy.linalg import solve_sylvester

from . import machine_config
from .implicit_euler import implicit_euler_solver


class linear_solver:
    """Interface between the linearised system of circuit equations and the implicit-Euler
    time stepper. Calculates the linear growth rate and solves the linearised dynamical problem.
    It needs the Jacobian of the plasma current distribution with respect to all of the
    independent currents, dIy/dI.
    """

    def __init__(
        self,
        Lambdam1,
        Pm1,
        Rm1,
        Mey,
        Myy,
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
        Lambdam1: np.array
            State matrix of the circuit equations for the metal in normal mode form:
            P is the identity on the active coils and diagonalises the isolated dynamics
            of the passive coils, R^{-1/2}L_{passive}R^{-1/2}
        Pm1: np.array
            change of basis matrix, as defined above, to the power of -1
        Rm1: np.array
            matrix of all metal resitances to the power of -1. Diagonal.
        Mey: np.array
            matrix of inductance values between grid points in the reduced plasma domain and all metal coils
            (active coils and passive-structure filaments)
            Calculated by the metal_currents object
        Myy: np.array
            inductance matrix of grid points in the reduced plasma domain
            Calculated by plasma_current object
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

        if Lambdam1 is None:
            self.Lambdam1 = Pm1 @ (Rm1 @ (machine_config.coil_self_ind @ (Pm1.T)))
        else:
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

        self.Pm1 = Pm1
        self.Rm1 = Rm1
        self.Pm1Rm1 = Pm1 @ Rm1
        self.Pm1Rm1Mey = np.matmul(self.Pm1Rm1, Mey)
        self.MyeP_T = Pm1 @ Mey
        self.Myy = Myy

        self.n_active_coils = machine_config.n_active_coils

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
        self.profile_forcing = np.zeros(self.n_independent_vars + 1)

    def reset_plasma_resistivity(self, plasma_resistance_1d):
        """Resets the value of the plasma resistivity,
        throught the vector of 'geometric restistances' in the plasma domain

        Parameters
        ----------
        plasma_resistance_1d : ndarray
            Vector of 2pi resistivity R/dA for all domain grid points in the reduced plasma domain
        """
        self.plasma_resistance_1d = plasma_resistance_1d
        self.set_linearization_point(None, None)

    def reset_timesteps(self, max_internal_timestep, full_timestep):
        """Resets the integration timesteps, calling self.solver.set_timesteps

        Parameters
        ----------
        max_internal_timestep: float
            integration substep of the ciruit equation, calling an implicit-Euler solver
        full_timestep: float
            integration timestep of the circuit equation
        """
        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.solver.set_timesteps(
            full_timestep=full_timestep, max_internal_timestep=max_internal_timestep
        )

    def set_linearization_point(self, dIydI, hatIy0):
        """Initialises an implicit-Euler solver with the appropriate matrices for the linearised dynamic problem.

        Parameters
        ----------
        dIydI = np.array
            partial derivatives of plasma-cell currents on the reduced plasma domain with respect to all intependent metal currents
            (active coil currents, vessel normal modes, total plasma current divided by plasma_norm_factor).
            These would typically come from having solved the forward Grad-Shafranov problem. Finite difference Jacobian.
            Calculated by the nl_solver object
        hatIy0 = np.array
            Plasma current distribution on the reduced plasma domain (1d) of the equilibrium around which the dynamics is linearised.
            This is normalised by the total plasma current, so that this vector sums to 1.
        """
        if dIydI is not None:
            self.dIydI = dIydI
        if hatIy0 is not None:
            self.hatIy0 = hatIy0

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
        Here we take C=0, that is the linearised dynamics does not account for evolving
        plasma parameters atm.


        Parameters
        ----------
        None given explicitly, they are all given by the object attributes.

        """

        self.M0matrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )
        self.dMmatrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )

        nRp = (
            np.sum(self.plasma_resistance_1d * self.hatIy0 * self.hatIy0)
            * self.plasma_norm_factor
        )

        # metal-metal before plasma
        self.M0matrix[: self.n_independent_vars, : self.n_independent_vars] = np.copy(
            self.Lambdam1
        )
        # metal-metal plasma-mediated
        self.dMmatrix[: self.n_independent_vars, : self.n_independent_vars] = np.matmul(
            self.Pm1Rm1Mey, self.dIydI[:, :-1]
        )

        # plasma to metal
        self.dMmatrix[:-1, -1] = np.dot(self.Pm1Rm1Mey, self.dIydI[:, -1])

        # metal to plasma
        self.M0matrix[-1, :-1] = np.dot(self.MyeP_T, self.hatIy0)
        # metal to plasma plasma-mediated
        self.dMmatrix[-1, :-1] = np.dot(
            np.matmul(self.Myy, self.dIydI[:, :-1]).T, self.hatIy0
        )

        JMyy = np.dot(self.Myy, self.hatIy0)
        self.dMmatrix[-1, -1] = np.dot(self.dIydI[:, -1], JMyy)

        self.dMmatrix[-1, :] /= nRp
        self.M0matrix[-1, :] /= nRp

        self.Mmatrix = self.M0matrix + self.dMmatrix

        # build necessary terms to incorporate forcing term from variations of the profile parameters
        # MIdot + RI = V - self.Vm1Rm12Mey_plus@self.dIydpars@d_profile_pars_dt
        # if self.dIydpars is not None:
        #     Pm1Rm1Mey_plus = np.concatenate(
        #         (self.Pm1Rm1Mey, JMyy[np.newaxis] / nRp), axis=0
        #     )
        #     self.forcing_pars_matrix = np.matmul(Pm1Rm1Mey_plus, self.dIydpars)

    def stepper(self, It, active_voltage_vec, d_profile_pars_dt=None):
        """Executes the time advancement. Uses the implicit_euler instance.

        Parameters
        ----------
        It = np.array
            vector of all independent currents that are solved for by the linearides problem, in terms of normal modes:
            (active currents, vessel normal modes, total plasma current divided by normalisation factor)
        active_voltage_vec = np.array
            voltages applied to the active coils
        d_profile_pars_dt = np.array
            time derivative of the profile parameters, not used atm
        other parameters are passed in as object attributes
        """
        self.empty_U[: self.n_active_coils] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Pm1Rm1, self.empty_U)
        self.forcing[-1] = 0.0

        # add forcing term from time derivative of profile parameters
        if d_profile_pars_dt is not None:
            self.forcing -= np.dot(self.forcing_pars_matrix, d_profile_pars_dt)

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
        # full set of characteristic timescales
        self.all_timescales = np.sort(np.linalg.eigvals(self.Mmatrix))
        # full set of characteristic timescales of the metal circuit equations
        self.all_timescales_const_Ip = np.sort(
            np.linalg.eigvals(self.Mmatrix[:-1, :-1])
        )
        mask = self.all_timescales < 0
        # the negative (i.e. unstable) eigenvalues
        self.instability_timescale = -self.all_timescales[mask]
        self.growth_rates = 1 / self.instability_timescale
        mask = self.all_timescales_const_Ip < 0
        self.instability_timescale_const_Ip = -self.all_timescales_const_Ip[mask]
        self.growth_rates_const_Ip = 1 / self.instability_timescale_const_Ip
