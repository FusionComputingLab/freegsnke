"""
Implements implicit time integration.

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

import math

import numpy as np


class implicit_euler_solver:
    """An implicit Euler time stepper for the linearized circuit equations. Solves an equation of type

    $$M\dot{I} + RI = F$$,

    with generic M, R and F. The internal_stepper and full_stepper solve for I(t+dt) using

    $$I(t+dt) = (M + Rdt)^{-1} (Fdt + MI(t))$$.

    The implementation actually allows for

    $$I(t+dt) = (M + Rdt)^{-1} (Fdt + LI(t))$$

    with M != L
    """

    def __init__(self, Mmatrix, Rmatrix, full_timestep, max_internal_timestep):
        """Sets up the implicit euler solver

        Parameters
        ----------
        Mmatrix : np.ndarray
            (NxN) Mutual inductance matrix
        Rmatrix : np.ndarray
            (NxN) Diagonal resistance matrix
        full_timestep : float
            Full timestep (dt) for the stepper
        max_internal_timestep : float
            Maximum size of the intermediate timesteps taken during the stepper.
            If max_internal_timestep < full_timestep, multiple steps are taken up to dt=full_timestep
        """
        self.Mmatrix = Mmatrix
        self.Lmatrix = Mmatrix
        self.Rmatrix = Rmatrix
        self.dims = np.shape(Mmatrix)[0]
        self.set_timesteps(full_timestep, max_internal_timestep)
        self.empty_U = np.zeros(self.dims)  # dummy voltage vector

    def set_Mmatrix(self, Mmatrix):
        """Updates the mutual inductance matrix.

        Parameters
        ----------
        Mmatrix : np.ndarray
            (NxN) Mutual inductance matrix
        """
        self.Mmatrix = Mmatrix

    def set_Lmatrix(self, Lmatrix):
        """Set a separate mutual inductance matrix L != M.

        Parameters
        ----------
        Lmatrix : np.ndarray
            (NxN) Mutual inductance matrix
        """
        self.Lmatrix = Lmatrix

    def set_Rmatrix(self, Rmatrix):
        """Updates the resistance matrix.

        Parameters
        ----------
        Rmatrix : np.ndarray
            (NxN) Diagonal resistance matrix
        """
        self.Rmatrix = Rmatrix

    def calc_inverse_operator(self):
        """Calculates the inverse operator (M + Rdt)^-1
        Note this needs done when M or R are updated
        """
        self.inverse_operator = np.linalg.inv(
            self.Mmatrix + self.internal_timestep * self.Rmatrix
        )

    def set_timesteps(self, full_timestep, max_internal_timestep):
        """Sets the timesteps for the stepper and (re)calculate the inverse operator

        Parameters
        ----------
        full_timestep : float
            Full timestep (dt) for the stepper
        max_internal_timestep : float
            Maximum size of the intermediate timesteps taken during the stepper.
            If max_internal_timestep < full_timestep, multiple steps are taken up to dt=full_timestep
        """
        self.full_timestep = full_timestep
        self.max_internal_timestep = max_internal_timestep
        self.n_steps = math.ceil(full_timestep / max_internal_timestep)
        self.internal_timestep = self.full_timestep / self.n_steps
        self.calc_inverse_operator()

    def internal_stepper(self, It, dtforcing):
        """Calculates the next internal timestep I(t + internal_timestep)

        Parameters
        ----------
        It : np.ndarray
            Length N vector of the currents, I, at time t
        dtforcing : np.ndarray
            Lenght N vector of the forcing F dt,  at time t
            multiplied by self.internal_timestep
        """
        Itpdt = np.dot(self.inverse_operator, dtforcing + np.dot(self.Lmatrix, It))
        return Itpdt

    def full_stepper(self, It, forcing):
        """Calculates the next full timestep I(t + `self.full_timestep`) by repeatedly
        solving for the internal timestep I(t + `self.internal_timestep`) for `self.n_steps` steps

        Parameters
        ----------
        It : np.ndarray
            Length N vector of the currents, I, at time t
        forcing : np.ndarray
            Lenght N vector of the forcing, F,  at time t
        """
        dtforcing = forcing * self.internal_timestep

        for _ in range(self.n_steps):
            It = self.internal_stepper(It, dtforcing)
            # self.intermediate_results[:, i] = It

        return It
