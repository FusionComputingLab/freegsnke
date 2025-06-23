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

    def __init__(
        self, problem_dimension, l2_reg=1e-6, collinearity_reg=1e-6, verbose=False
    ):
        """Instantiates the class.

        Parameters
        ----------
        problem_dimension : int
            Dimension of independent variable.
            np.shape(x) = problem_dimension
            x is a 1d vector.
        l2_reg : float
            Tychonoff regularization coeff
        collinearity_reg : float
            Tychonoff regularization coeff which further penalizes collinear terms

        """

        self.problem_dimension = problem_dimension
        self.dummy_hessenberg_residual = np.zeros(problem_dimension)
        self.dummy_hessenberg_residual[0] = 1.0
        self.verbose = verbose
        self.set_regularization(l2_reg, collinearity_reg)
        # self.force_sign_alignment = force_sign_alignment

    def Arnoldi_unit(
        self,
        x0,
        dx,
        R0,
        # nR0,
        F_function,
        args,
        build_next=True,
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

        # res_now = np.copy(R0)
        # calculate residual at explored point x0+dx
        res_calculated = False
        dx1 = np.copy(dx)
        while res_calculated is False:
            try:
                candidate_x = x0 + dx1
                R_dx = F_function(candidate_x, *args)
                res_calculated = True
            except:
                dx1 *= 0.75
                self.Q[:, self.n_it] *= 0.75
        useful_residual = R_dx - R0
        # dot_product = np.dot(useful_residual, R0)

        # if self.force_sign_alignment and (dot_product > 0):
        #     # need sign reversal!
        #     print(f"term {self.n_it} being reversed")
        #     res_calculated = False
        #     dx1 = -np.copy(dx)
        #     self.Qn[:, self.n_it] *= -1
        #     self.Q[:, self.n_it] *= -1
        #     while res_calculated is False:
        #         try:
        #             candidate_x = x0 + dx1
        #             R_dx = F_function(candidate_x, *args)
        #             res_calculated = True
        #         except:
        #             dx1 *= 0.75
        #             self.Q[:, self.n_it] *= 0.75
        #     useful_residual = R_dx - R0

        self.n_G[self.n_it] = np.linalg.norm(useful_residual)
        self.G[:, self.n_it] = useful_residual
        self.Gn[:, self.n_it] = useful_residual / self.n_G[self.n_it]
        self.collinearity[: self.n_it, self.n_it] = np.sum(
            self.Gn[:, self.n_it, np.newaxis] * self.Gn[:, : self.n_it], axis=0
        )
        # print('coll', self.n_it, self.collinearity[:self.n_it, self.n_it])

        if build_next:
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

            # # build the relevant Givens rotation
            # givrot = np.eye(self.n_it + 2)
            # rho = np.dot(self.Omega[self.n_it], self.Hm[: self.n_it + 1, self.n_it])
            # rr = (rho**2 + self.Hm[self.n_it + 1, self.n_it] ** 2) ** 0.5
            # givrot[-2, -2] = givrot[-1, -1] = rho / rr
            # givrot[-2, -1] = self.Hm[self.n_it + 1, self.n_it] / rr
            # givrot[-1, -2] = -1.0 * givrot[-2, -1]
            # # update Omega matrix
            # Omega = np.eye(self.n_it + 2)
            # Omega[:-1, :-1] = 1.0 * self.Omega
            # self.Omega = np.matmul(givrot, Omega)
            return next_candidate

    def set_regularization(self, l2_reg, collinearity_reg):
        """Sets the regularization coeffs

        Parameters
        ----------
        l2_reg : float
            Tychonoff regularization coeff
        collinearity_reg : float
            Tychonoff regularization coeff which further penalizes collinear terms
        """
        self.l2_reg = l2_reg
        self.collinearity_reg = collinearity_reg

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
        # l2_reg=1e-5,
        # collinearity_reg=1e-6,
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
        self.x0 = np.copy(x0)
        self.R0 = np.copy(R0)

        self.relative_unexplained_residuals = []
        nR0 = np.linalg.norm(R0)
        self.nR0 = 1.0 * nR0
        self.max_dim = int(max_n_directions + 1)

        # orthogonal basis in x space
        self.Q = np.zeros((self.problem_dimension, self.max_dim))
        # orthonormal basis in x space
        self.Qn = np.zeros((self.problem_dimension, self.max_dim))

        # basis in residual space
        self.G = np.zeros((self.problem_dimension, self.max_dim))
        # orthonormal basis in residual space
        self.Gn = np.zeros((self.problem_dimension, self.max_dim))
        # norms of residual vectors
        self.n_G = np.zeros(self.max_dim)

        self.collinearity = np.zeros((self.max_dim, self.max_dim))

        # QR decomposition of Hm: Hm = T@R
        # self.Omega = np.array([[1]])

        # Hessenberg matrix
        self.Hm = np.zeros((self.max_dim + 1, self.max_dim))

        # resize step based on residual
        adjusted_step_size = step_size * nR0

        # prepare for first direction exploration
        self.n_it = 0
        self.n_it_tot = 0
        this_step_size = adjusted_step_size * ((1 + self.n_it) ** scaling_with_n)

        dx /= np.linalg.norm(dx)
        # # new addition
        # if clip_quantiles is not None:
        #     q1, q2 = np.quantile(dx, clip_quantiles)
        #     dx = np.clip(dx, q1, q2)

        self.Qn[:, self.n_it] = np.copy(dx)
        dx *= this_step_size
        self.Q[:, self.n_it] = np.copy(dx)

        explore = 1
        while explore:
            # build Arnoldi update
            dx = self.Arnoldi_unit(x0, dx, R0, F_function, args)

            # prepare to calculate explained residual
            collinearity_penalty = np.diag(
                np.max(
                    1
                    / (1 - np.abs(self.collinearity[: self.n_it + 1, : self.n_it + 1]))
                    ** 2,
                    axis=0,
                )
                - 1
            )
            collinear_aware_regulariz = (
                np.eye(self.n_it + 1) * self.l2_reg
                + collinearity_penalty * self.collinearity_reg
            )
            self.collinear_aware_regulariz = collinear_aware_regulariz * nR0**2

            # solve the regularised least sq problem
            coeffs = np.dot(
                np.linalg.inv(
                    self.G[:, : self.n_it + 1].T @ self.G[:, : self.n_it + 1]
                    + self.collinear_aware_regulariz
                ),
                np.dot(self.G[:, : self.n_it + 1].T, -R0),
            )
            coeffs = np.clip(coeffs, -clip, clip)
            # calculare the corresponding fraction of residual that is currently explained
            expl_res = np.sum(
                self.G[:, : self.n_it + 1] * coeffs[np.newaxis, :], axis=1
            )
            self.relative_unexplained_residuals.append(
                np.linalg.norm(R0 + expl_res) / nR0
            )

            explore = self.n_it < max_n_directions
            explore *= (
                self.relative_unexplained_residuals[-1]
                > target_relative_unexplained_residual
            )

            # prepare for next step
            if explore:
                self.n_it += 1
                # # new addition
                # if clip_quantiles is not None:
                #     q1, q2 = np.quantile(dx, clip_quantiles)
                #     dx = np.clip(dx, q1, q2)
                self.Qn[:, self.n_it] = np.copy(dx)
                this_step_size = adjusted_step_size * (
                    (1 + self.n_it) ** scaling_with_n
                )
                dx *= this_step_size
                self.Q[:, self.n_it] = np.copy(dx)

        # self.coeffs = -nR0 * np.dot(
        #     np.linalg.inv(self.Omega[:-1] @ self.Hm[: self.n_it + 2, : self.n_it + 1]),
        #     self.Omega[:-1, 0],
        # )

        # collinearity = np.sum(self.G[:,np.newaxis,:self.n_it + 1]*self.G[:,:self.n_it + 1,np.newaxis],axis=0)
        # d_collinearity = np.diag(collinearity)**.5
        # collinearity /= (d_collinearity[:, np.newaxis] * d_collinearity[np.newaxis, :])
        # self.collinearity = np.abs(np.triu(collinearity, 1))
        # d_collinearity = np.diag(np.max(1/(1-np.abs(self.collinearity))**2, axis=0))

        # collinear_aware_regulariz = np.eye(self.n_it + 1)*1e-4
        # collinear_aware_regulariz += d_collinearity*1e-4
        # self.collinear_aware_regulariz = collinear_aware_regulariz * nR0**2

        # Hm_ = np.copy(self.Hm[: self.n_it + 2, : self.n_it + 1])
        # self.coeffs = -self.sign * nR0 * np.dot(np.linalg.inv(Hm_.T@Hm_ + self.collinear_aware_regulariz), Hm_[0])
        # self.vanilla_coeffs = np.dot(np.linalg.inv(self.G[:, : self.n_it + 1].T@self.G[:, : self.n_it + 1] + self.collinear_aware_regulariz), np.dot(self.G[:, : self.n_it + 1].T, -R0))

        self.coeffs = np.copy(coeffs)
        self.dx = np.sum(self.Q[:, : self.n_it + 1] * coeffs[np.newaxis, :], axis=1)

    # def review_Arnoldi_iteration(
    #     self,
    #     F_function,
    #     args,
    #     target_relative_unexplained_residual,
    #     clip,
    #     l2_reg=1e-4,
    #     collinearity_reg=1e-4,
    #     threshold=0.1,
    # ):

    #     # resize the directions in x space
    #     self.Q = self.Q[:, : self.n_it + 1] * self.coeffs[np.newaxis, :]

    #     # # select those that's worth analyzing
    #     # mask = (self.n_G[:self.n_it + 1] * self.coeffs) > threshold*self.nR0
    #     # max_n_directions = np.sum(mask.astype(float))
    #     # # apply the selection
    #     # self.Q = self.Q[:, mask]
    #     max_n_directions = 1.0 * self.n_it

    #     self.relative_unexplained_residuals_review = []
    #     self.n_it = 0
    #     explore = 1

    #     while explore:
    #         self.Arnoldi_unit(
    #             self.x0, self.Q[:, self.n_it], self.R0, self.nR0, F_function, args
    #         )

    #         # prepare to calculate explained residual
    #         collinearity_penalty = np.diag(
    #             np.max(
    #                 1
    #                 / (1 - np.abs(self.collinearity[: self.n_it + 1, : self.n_it + 1]))
    #                 ** 2,
    #                 axis=0,
    #             )
    #             - 1
    #         )
    #         collinear_aware_regulariz = (
    #             np.eye(self.n_it + 1) * l2_reg + collinearity_penalty * collinearity_reg
    #         )
    #         self.collinear_aware_regulariz = collinear_aware_regulariz * self.nR0**2

    #         # solve the regularised least sq problem
    #         coeffs = np.dot(
    #             np.linalg.inv(
    #                 self.G[:, : self.n_it + 1].T @ self.G[:, : self.n_it + 1]
    #                 + self.collinear_aware_regulariz
    #             ),
    #             np.dot(self.G[:, : self.n_it + 1].T, -self.R0),
    #         )
    #         coeffs = np.clip(coeffs, -clip, clip)
    #         # calculare the corresponding fraction of residual that is currently explained
    #         expl_res = np.sum(
    #             self.G[:, : self.n_it + 1] * coeffs[np.newaxis, :], axis=1
    #         )
    #         self.relative_unexplained_residuals_review.append(
    #             np.linalg.norm(self.R0 + expl_res) / self.nR0
    #         )

    #         explore = self.n_it < max_n_directions
    #         explore *= (
    #             self.relative_unexplained_residuals_review[-1]
    #             > target_relative_unexplained_residual
    #         )
    #         self.n_it += 1

    #     self.coeffs_review = np.copy(coeffs)
    #     self.dx_review = np.sum(
    #         self.Q[:, : self.n_it + 1] * coeffs[np.newaxis, :], axis=1
    #     )
