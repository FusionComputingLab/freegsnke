"""
Implements the core Newton Krylov nonlinear solver used by both static GS solver and evolutive solver. 

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


class nksolver:
    """Implementation of Newton Krylow algorithm for solving
    a generic root problem of the type
    F(x, other args) = 0
    in the variable x -- F(x) should have the same dimensions as x.
    Problem must be formulated so that x is a 1d np.array.

    In practice, given a guess x_0 and F(x_0) = R_0
    it aims to find the best step dx such that
    F(x_0 + dx) is minimum.
    """

    def __init__(self, problem_dimension, verbose=False):
        """Instantiates the class.

        Parameters
        ----------
        problem_dimension : int
            Dimension of independent variable.
            np.shape(x) = problem_dimension
            x is a 1d vector.


        """

        self.problem_dimension = problem_dimension
        self.dummy_hessenberg_residual = np.zeros(problem_dimension)
        self.dummy_hessenberg_residual[0] = 1.0
        self.verbose = verbose

    def Arnoldi_unit(
        self,
        x0,
        dx,
        R0,
        F_function,
        args,
    ):
        """Explores direction dx and proposes new direction for next exploration.

        Parameters
        ----------
        x0 : 1d np.array, np.shape(x0) = self.problem_dimension
            The expansion point x_0
        dx : 1d np.array, np.shape(dx) = self.problem_dimension
            The first direction to be explored. This will be sized appropriately.
        R0 : 1d np.array, np.shape(R0) = self.problem_dimension
            Residual of the root problem F_function at expansion point x_0
        F_function : 1d np.array, np.shape(x0) = self.problem_dimension
            Function representing the root problem at hand
        args : list
            Additional arguments for using function F
            F = F(x, *args)

        Returns
        -------
        new_candidate_step : 1d np.array, with same self.problem_dimension
            The direction to be explored next

        """

        # calculate residual at explored point x0+dx
        res_calculated = False
        while res_calculated is False:
            try:
                candidate_x = x0 + dx
                R_dx = F_function(candidate_x, *args)
                res_calculated = True
            except:
                dx *= 0.75
        useful_residual = R_dx - R0
        self.G[:, self.n_it] = useful_residual

        # append to Hessenberg matrix
        self.Hm[: self.n_it + 1, self.n_it] = np.sum(
            self.Qn[:, : self.n_it + 1] * useful_residual[:, np.newaxis], axis=0
        )

        # ortogonalise wrt previous directions
        next_candidate = useful_residual - np.sum(
            self.Qn[:, : self.n_it + 1]
            * self.Hm[: self.n_it + 1, self.n_it][np.newaxis, :],
            axis=1,
        )

        # append to Hessenberg matrix and normalize
        self.Hm[self.n_it + 1, self.n_it] = np.linalg.norm(next_candidate)
        # normalise the candidate direction for next iteration
        next_candidate /= self.Hm[self.n_it + 1, self.n_it]

        # build the relevant Givens rotation
        givrot = np.eye(self.n_it + 2)
        rho = np.dot(self.Omega[self.n_it], self.Hm[: self.n_it + 1, self.n_it])
        rr = (rho**2 + self.Hm[self.n_it + 1, self.n_it] ** 2) ** 0.5
        givrot[-2, -2] = givrot[-1, -1] = rho / rr
        givrot[-2, -1] = self.Hm[self.n_it + 1, self.n_it] / rr
        givrot[-1, -2] = -1.0 * givrot[-2, -1]
        # update Omega matrix
        Omega = np.eye(self.n_it + 2)
        Omega[:-1, :-1] = 1.0 * self.Omega
        self.Omega = np.matmul(givrot, Omega)
        return next_candidate

    def Arnoldi_iteration(
        self,
        x0,
        dx,
        R0,
        F_function,
        args,
        step_size,
        scaling_with_n,
        target_relative_unexplained_residual,
        max_n_directions,
        clip,
    ):
        """Performs the iteration of the NK solution method:
        1) explores direction dx
        2) checks what fraction of the residual can be (linearly) canceled
        3) restarts if not satisfied
        The best candidate step combining all explored directions is stored at self.dx

        Parameters
        ----------
        x0 : 1d np.array, np.shape(x0) = self.problem_dimension
            The expansion point x_0
        dx : 1d np.array, np.shape(dx) = self.problem_dimension
            The first direction to be explored. This will be sized appropriately.
        R0 : 1d np.array, np.shape(R0) = self.problem_dimension
            Residual of the root problem F_function at expansion point x_0
        F_function : 1d np.array, np.shape(x0) = self.problem_dimension
            Function representing the root problem at hand
        args : list
            Additional arguments for using function F
            F = F(x, *args)
        step_size : float
            l2 norm of proposed step in units of the residual norm
        scaling_with_n : float
            allows to further scale dx candidate steps as a function of the iteration number n_it, by a factor
            (1 + self.n_it)**scaling_with_n
        target_relative_explained_residual : float between 0 and 1
            terminates iteration when such a fraction of the initial residual R0
            can be (linearly) cancelled
        max_n_directions : int
            terminates iteration even though condition on
            explained residual is not met
        clip : float
            maximum step size for each explored direction, in units
            of exploratory step dx_i
        """

        nR0 = np.linalg.norm(R0)
        self.max_dim = max_n_directions + 1

        # orthogonal basis in x space
        self.Q = np.zeros((self.problem_dimension, self.max_dim))
        # orthonormal basis in x space
        self.Qn = np.zeros((self.problem_dimension, self.max_dim))

        # basis in residual space
        self.G = np.zeros((self.problem_dimension, self.max_dim))

        # QR decomposition of Hm: Hm = T@R
        self.Omega = np.array([[1]])

        # Hessenberg matrix
        self.Hm = np.zeros((self.max_dim + 1, self.max_dim))

        # resize step based on residual
        adjusted_step_size = step_size * nR0

        # prepare for first direction exploration
        self.n_it = 0
        self.n_it_tot = 0
        this_step_size = adjusted_step_size * ((1 + self.n_it) ** scaling_with_n)

        dx /= np.linalg.norm(dx)
        self.Qn[:, self.n_it] = np.copy(dx)
        dx *= this_step_size
        self.Q[:, self.n_it] = np.copy(dx)

        explore = 1
        while explore:
            # build Arnoldi update
            dx = self.Arnoldi_unit(x0, dx, R0, F_function, args)

            explore = self.n_it < max_n_directions
            self.explained_residual = np.abs(self.Omega[-1, 0])
            explore *= self.explained_residual > target_relative_unexplained_residual

            # prepare for next step
            if explore:
                self.n_it += 1
                self.Qn[:, self.n_it] = np.copy(dx)
                this_step_size = adjusted_step_size * (
                    (1 + self.n_it) ** scaling_with_n
                )
                dx *= this_step_size
                self.Q[:, self.n_it] = np.copy(dx)

        self.coeffs = -nR0 * np.dot(
            np.linalg.inv(self.Omega[:-1] @ self.Hm[: self.n_it + 2, : self.n_it + 1]),
            self.Omega[:-1, 0],
        )
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        self.dx = np.sum(
            self.Q[:, : self.n_it + 1] * self.coeffs[np.newaxis, :], axis=1
        )
