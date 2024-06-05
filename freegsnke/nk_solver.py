import numpy as np

import matplotlib.pyplot as plt


class nksolver:
    """Implementation of Newton Krylow algorithm for solving
    a generic root problem of the type
    F(x, other args) = 0
    in the variable x.
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


        """

        self.problem_dimension = problem_dimension
        # self.verbose=verbose

    def least_square_problem(self, R0, nR0, G, Q, clip, threshold, clip_hard):
        """Solves the following least square problem
        min || G*coeffs + R0 ||^2
        in the coefficients coeffs, and calculates the corresponding best step
        dx = coeffs * Q

        Parameters
        ----------
        R0 : 1d np.array, np.shape(R0) = self.problem_dimension
            Residual at expansion point x_0
        nR0 : float
            l2 norm of R0
        G : 2d np.array, np.shape(G) = [variable, self.problem_dimension]
            Collection of values F(x_0 + dx_i) - R_0
        Q : 2d np.array, np.shape(Q) = [variable, self.problem_dimension]
            Collection of values dx_i
        clip : float
            maximum step size for each explored direction, in units
            of exploratory step dx_i
        threshold : float
            catches cases of untreated (partial) collinearity
        clip_hard : float
            maximum step size for cases of untreated (partial) collinearity
        """
        # print('initial residual', nR0)
        self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)), G.T), -R0)
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        self.explained_residual = np.sum(G * self.coeffs[np.newaxis, :], axis=1)
        self.relative_unexplained_residual = (
            np.linalg.norm(self.explained_residual + R0) / nR0
        )
        if self.relative_unexplained_residual > threshold:
            self.coeffs = np.clip(self.coeffs, -clip_hard, clip_hard)
        self.dx = np.sum(Q * self.coeffs[np.newaxis, :], axis=1)

    def Arnoldi_unit(
        self,
        x0,  # trial expansion point
        dx,  # first vector for current basis
        R0,  # residual at trial_current expansion point: Fresidual(trial_current)
        F_function,
        args,
        step_size,
        max_rel_step_size
    ):
        """Explores direction dx by calculating and storing residual F(x_0 + dx)

        Parameters
        ----------
        x0 : 1d np.array, np.shape(x0) = self.problem_dimension
            The expansion point x_0
        dx : 1d np.array, np.shape(dx) = self.problem_dimension
            The direction to be explored. This will be sized appropriately.
        R0 : 1d np.array, np.shape(R0) = self.problem_dimension
            Residual at expansion point x_0
        F_function : 1d np.array, np.shape(x0) = self.problem_dimension
            Function representing the root problem at hand
        args : list
            Additional arguments for using function F
            F(x_0 + dx, *args)
        step_size : float
            l2 norm of proposed step.


        Returns
        -------
        new_candidate_step : 1d np.array, np.shape(dx_new) = self.problem_dimension
            The direction to be explored next

        """
        # if self.verbose:
        #     print('0 - R0', R0)
        # print('initial residual', R0)
        candidate_step = step_size * dx / np.linalg.norm(dx)
        if max_rel_step_size:
            del_step = np.amax(candidate_step) - np.amin(candidate_step)
            del_x0 = np.amax(x0) - np.amin(x0)
            if del_step/del_x0 > max_rel_step_size:
                print('step resized!')
                candidate_step *= np.abs(max_rel_step_size*del_x0/del_step)
        # plt.imshow(candidate_step.reshape(65,129))
        # plt.colorbar()
        # plt.title('resized step')
        # plt.show()
        res_calculated = False
        while res_calculated is False:
            try:
                candidate_x = x0 + candidate_step
                R_dx = F_function(candidate_x, *args)
                res_calculated = True

            except:
                candidate_step *= .75
                print('candidate_step', np.linalg.norm(candidate_step))
        # plt.imshow(args[1].jtor)
        # plt.colorbar()
        # plt.title('jtor')
        # plt.show()
        # if self.verbose:
        #     print('2 - R0', R0)
        # print('2 - R00', R00)
        useful_residual = R_dx - R0
        # print('useful_residual ', R_dx)
        # plt.imshow(useful_residual.reshape(65,129))
        # plt.colorbar()
        # plt.title('useful_residual')
        # plt.show()
        # self.last_candidate_step = 1.0*candidate_step
        # self.last_useful_residual = 1.0*useful_residual
        # print('candidate_x ', candidate_x)

        self.Q[:, self.n_it] = candidate_step.copy()
        self.Qn[:, self.n_it] = self.Q[:, self.n_it] / np.linalg.norm(
            self.Q[:, self.n_it]
        )

        self.G[:, self.n_it] = useful_residual.copy()
        self.Gn[:, self.n_it] = self.G[:, self.n_it] / np.linalg.norm(
            self.G[:, self.n_it]
        )

        # self.Hm[:self.n_it+1, self.n_it] = np.sum(self.Qn[:,:self.n_it+1] * useful_residual[:,np.newaxis], axis=0)

        # next_candidate = useful_residual - np.sum(self.Qn[:,:self.n_it+1] * self.Hm[:self.n_it+1, self.n_it][:, np.newaxis])

        # if self.verbose:
        #     print(self.n_it)
        #     print('x0', x0)
        #     print('candidate_x', candidate_x)
        #     # print('last_candidate_step', self.last_candidate_step)
        #     print('R_dx', R_dx)
        #     print('R0', R0)
        #     print(self.n_it, self.G[:,:self.n_it+1])

        # orthogonalize with respect to previously attemped directions
        useful_residual -= np.sum(
            np.sum(
                self.Qn[:, : self.n_it + 1] * useful_residual[:, np.newaxis],
                axis=0,
                keepdims=True,
            )
            * self.Qn[:, : self.n_it + 1],
            axis=1,
        )

        return useful_residual

    def Arnoldi_iteration(
        self,
        x0,  # trial_current expansion point
        dx,  # first vector for current basis
        R0,  # circuit eq. residual at trial_current expansion point: F_function(x0)
        F_function,
        args,
        step_size,
        scaling_with_n,
        target_relative_unexplained_residual,
        max_n_directions,  # max number of basis vectors (must be less than number of modes + 1)
        max_Arnoldi_iterations,
        max_collinearity,
        clip,
        threshold,
        clip_hard,
        max_rel_step_size=False
    ):
        """Performs the iteration of the NK solution method:
        1) explores direction dx
        2) computes and stores new residual
        3) builds new candidate direction to continue exploring
        Calculates best candidate step, stored at self.dx

        Parameters
        ----------
        x0 : 1d np.array, np.shape(x0) = self.problem_dimension
            The expansion point x_0
        dx : 1d np.array, np.shape(dx) = self.problem_dimension
            The first direction to be explored.
        R0 : 1d np.array, np.shape(R0) = self.problem_dimension
            Residual at expansion point x_0
        F_function : 1d np.array, np.shape(x0) = self.problem_dimension
            Function representing the root problem at hand
        args : list
            Additional arguments for using function F
            F(x_0 + dx, *args)
        step_size : float
            l2 norm of proposed step
        scaling_with_n : float
            allows to further scale dx candidate steps by factor
            (1 + self.n_it)**scaling_with_n
        target_relative_explained_residual : float between 0 and 1
            terminates iteration when exploration can explain this
            fraction of the initial residual R0
        max_n_directions : int
            terminates iteration even though condition on
            explained residual is not met
        max_Arnoldi_iterations : int
            terminates iteration after attempting to explore
            this number of directions
        max_collinearity : float between 0 and 1
            rejects a candidate direction if resulting residual
            is collinear to any of those stored previously
        clip : float
            maximum step size for each explored direction, in units
            of exploratory step dx_i
        threshold : float
            catches cases of untreated (partial) collinearity
        clip_hard : float
            maximum step size for cases of untreated (partial) collinearity


        """

        self.nR0 = np.linalg.norm(R0)

        # basis in x space
        self.Q = np.zeros((self.problem_dimension, max_n_directions + 1))
        # orthonormal basis in x space
        self.Qn = np.zeros((self.problem_dimension, max_n_directions + 1))
        # basis in residual space
        self.G = np.zeros((self.problem_dimension, max_n_directions + 1))
        # orthonormal basis in residual space
        self.Gn = np.zeros((self.problem_dimension, max_n_directions + 1))

        # self.Hm = np.zeros((self.problem_dimension+1, max_n_directions+1))

        self.n_it = 0
        self.n_it_tot = 0
        adjusted_step_size = step_size * self.nR0

        # print('norm trial_sol', np.linalg.norm(trial_sol))

        explore = 1
        while explore:
            this_step_size = adjusted_step_size * ((1 + self.n_it) ** scaling_with_n)
            dx = self.Arnoldi_unit(x0, dx, R0, F_function, args, this_step_size, max_rel_step_size=max_rel_step_size)

            not_collinear_check = 1 - np.any(
                np.sum(
                    self.Gn[:, : self.n_it] * self.Gn[:, self.n_it : self.n_it + 1],
                    axis=0,
                )
                > max_collinearity
            )
            self.n_it_tot += 1
            if not_collinear_check:
                self.n_it += 1
                # print(self.n_it, self.G[:,:self.n_it])
                self.least_square_problem(
                    R0,
                    self.nR0,
                    G=self.G[:, : self.n_it],
                    Q=self.Q[:, : self.n_it],
                    clip=clip,
                    threshold=threshold,
                    clip_hard=clip_hard,
                )
                # if self.verbose:
                #     print('rel_unexpl_res', self.relative_unexplained_residual)
                explained_residual_check = (
                    self.relative_unexplained_residual
                    > target_relative_unexplained_residual
                )
            # else:
            #     print('collinear!, rejected', self.n_it)

            explore = explained_residual_check * (
                self.n_it_tot < max_Arnoldi_iterations
            )
            explore *= self.n_it < max_n_directions
            print('relative_unexplained_residual', self.relative_unexplained_residual)

            # print('dx, ', np.linalg.norm(self.dx))
