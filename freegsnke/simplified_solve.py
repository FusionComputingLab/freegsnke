"""
Implements a 'simplified' set of discretised current equations (coupling metals and plasma)
in which the normalised distribution of the plasma current is assumed known (but not the magnitude of the plasma current).
This makes the system of current equations linear.

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

from . import machine_config
from .implicit_euler import implicit_euler_solver


class simplified_solver_J1:
    """Takes the full system of circuit equations (discretised in time and over the reduced plasma domain)
    and applies that  $$I_y(t+dt) = \hat{I_y}*I_p(t+dt)$$
    where $\hat{I_y}$ is assigned and such that np.sum(\hat{I_y})=1.
    With this hypothesis, the system can be (linearly) solved to find ALL of the extensive currents at t+dt
    (metal current and total plasma current).
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
        full_timestep=0.0001,
    ):
        """Initialises the solver for the extensive currents.

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
        plasma_resistance_1d: np.array
            plasma reistance in each (reduced domain) plasma cell, R_yy, raveled to be of the same shape as I_y,
            the lumped total plasma resistance is obtained by contracting
            \hat{I_y}R_{yy}\hat{I_{y}} = I_y R_{yy} I_{y} / I_{p}^2
        full_timestep: float
            full timestep requested to the implicit-Euler solver

        """

        self.max_internal_timestep = full_timestep
        self.full_timestep = full_timestep
        self.plasma_norm_factor = plasma_norm_factor

        self.n_independent_vars = np.shape(Lambdam1)[0]
        self.Mmatrix = np.eye(self.n_independent_vars + 1)
        self.Mmatrix[:-1, :-1] = Lambdam1
        self.Lambdam1 = Lambdam1

        self.Lmatrix = np.copy(self.Mmatrix)

        self.Pm1 = Pm1
        self.Rm1 = Rm1
        self.Pm1Rm1 = Pm1 @ Rm1
        self.Pm1Rm1Mey = np.matmul(self.Pm1Rm1, Mey)
        self.MyeP_T = Pm1 @ Mey
        self.Myy = Myy

        self.n_active_coils = machine_config.n_active_coils
        self.n_coils = machine_config.n_coils

        self.plasma_resistance_1d = plasma_resistance_1d

        # sets up implicit euler to solve system of
        # - metal circuit eq
        # - plasma circuit eq
        # NB the solver is initialized here but the matrices are set up
        # at each timestep using the method prepare_solver
        self.solver = implicit_euler_solver(
            Mmatrix=self.Mmatrix,
            Rmatrix=np.eye(self.n_independent_vars + 1),
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.full_timestep,
        )

        # dummy vessel voltage vector
        self.empty_U = np.zeros(self.n_coils)
        # dummy voltage vec for eig modes
        self.forcing = np.zeros(self.n_independent_vars + 1)
        # dummy voltage vec for residuals
        self.residuals = np.zeros(self.n_independent_vars + 1)

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

    def reset_plasma_resistivity(self, plasma_resistance_1d):
        """Resets the value of the plasma resistivity,
        throught the vector of 'geometric resistances' in the plasma domain

        Parameters
        ----------
        plasma_resistance_1d : ndarray
            Vector of (2pi resistivity R/dA) for all domain grid points in the reduced plasma domain
        """
        self.plasma_resistance_1d = plasma_resistance_1d

    def prepare_solver(self, hatIy_left, hatIy_0, hatIy_1, active_voltage_vec):
        """Computes the actual matrices that are needed in the ODE for the extensive currents
         and that must be passed to the implicit-Euler solver.

        Parameters
        ----------
        hatIy_left: np.array
            guess for gridded plasma current to left-contract the plasma evolution equation
            (e.g. at time t, or t+dt, or a combination)
        hatIy_0: np.array
            gridded plasma current to left-contract the plasma evolution equation at time t
        hatIy_1: np.array
            (guessed) gridded plasma current to left-contract the plasma evolution equation at time t+dt
        active_voltage_vec: np.array
            voltages applied to the active coils
        """

        Rp = np.sum(self.plasma_resistance_1d * hatIy_left * hatIy_1)
        self.Rp = Rp

        self.Mmatrix[-1, :-1] = np.dot(self.MyeP_T, hatIy_left) / (
            Rp * self.plasma_norm_factor
        )
        self.Lmatrix[-1, :-1] = np.copy(self.Mmatrix[-1, :-1])

        simplified_mutual = self.Pm1Rm1Mey * self.plasma_norm_factor
        self.Mmatrix[:-1, -1] = np.dot(simplified_mutual, hatIy_1)
        self.Lmatrix[:-1, -1] = np.dot(simplified_mutual, hatIy_0)

        simplified_self_left = np.dot(self.Myy, hatIy_left) / Rp
        simplified_self_1 = np.dot(simplified_self_left, hatIy_1)
        simplified_self_0 = np.dot(simplified_self_left, hatIy_0)
        self.Mmatrix[-1, -1] = simplified_self_1
        self.Lmatrix[-1, -1] = simplified_self_0

        self.solver.set_Lmatrix(self.Lmatrix)
        self.solver.set_Mmatrix(self.Mmatrix)
        # recalculate the inverse operator
        self.solver.calc_inverse_operator()

        self.empty_U[: self.n_active_coils] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Pm1Rm1, self.empty_U)

    def stepper(self, It, hatIy_left, hatIy_0, hatIy_1, active_voltage_vec):
        """Computes and returns the set of extensive currents at time t+dt

        Parameters
        ----------
        It: np.array
            vector of all extensive currents at time t: It = (all metals, plasma)
            with dimension self.n_independent_vars + 1. Metal currents expressed in
            terms of normal modes.
        hatIy_left: np.array
            normalised plasma current distribution on the reduced domain.
            This is used to left-contract the plasma evolution equation
            (e.g. at time t, or t+dt, or a combination)
        hatIy_0: np.array
            normalised plasma current distribution on the reduced domain at time t
        hatIy_1: np.array
            normalised plasma current distribution on the reduced domain at time t+dt
        active_voltage_vec: np.array
            voltages applied to the active coils

        Returns
        -------
        Itpdt: np.array
            currents (active coils, vessel eigenmodes, total plasma current) at time t+dt
        """
        self.prepare_solver(hatIy_left, hatIy_0, hatIy_1, active_voltage_vec)
        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt

    def ceq_residuals(self, I_0, I_1, hatIy_left, hatIy_0, hatIy_1, active_voltage_vec):
        """Computes and returns the set of residual for the full lumped circuit equations
        (all metals in normal modes plus contracted plasma eq.) given extensive currents and
        normalised plasma distributions at both times t and t+dt. Uses

        Parameters
        ----------
        I_0: np.array
            vector of all extensive currents at time t: It = (all metals, plasma)
            with dimension self.n_independent_vars + 1. Metal currents expressed in
            terms of normal modes.
        I_1: np.array
            as above at time t+dt.
        hatIy_left: np.array
            normalised plasma current distribution on the reduced domain.
            This is used to left-contract the plasma evolution equation
            (e.g. at time t, or t+dt, or a combination)
        hatIy_0: np.array
            normalised plasma current distribution on the reduced domain at time t
        hatIy_1: np.array
            normalised plasma current distribution on the reduced domain at time t+dt
        active_voltage_vec: np.array
            voltages applied to the active coils

        Returns
        -------
        np.array
            Residual of the circuit eq, lumped for the plasma: dimensions are self.n_independent_vars + 1.
        """
        residuals = np.zeros_like(I_1)
        empty_U = np.zeros(self.n_coils)
        forcing = np.zeros_like(I_1)

        # prepare time derivatives
        Id_dot = (I_1 - I_0)[:-1]
        Iy_dot = hatIy_1 * I_1[-1] - hatIy_0 * I_0[-1]
        # prepare forcing term
        empty_U[: self.n_active_coils] = active_voltage_vec
        forcing[:-1] = np.dot(self.Pm1Rm1, empty_U)
        # prepare the lumped plasma resistance
        Rp = np.sum(self.plasma_resistance_1d * hatIy_left * hatIy_1)

        # metal dimensions
        res_met = np.dot(self.Lambdam1, Id_dot)
        res_met += np.dot(self.Pm1Rm1Mey, Iy_dot) * self.plasma_norm_factor
        # plasma lump
        res_pl = np.dot(self.Myy, Iy_dot)
        res_pl += np.dot(self.MyeP_T, Id_dot) / self.plasma_norm_factor
        res_pl = np.dot(res_pl, hatIy_left)
        res_pl /= Rp
        # build residual vector
        residuals[:-1] = res_met
        residuals[-1] = res_pl

        # add resistive and forcing terms
        residuals += (I_1 - forcing) * self.full_timestep

        return residuals
