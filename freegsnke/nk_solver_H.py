import numpy as np


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
        self.dummy_hessenberg_residual = np.zeros(problem_dimension)
        self.dummy_hessenberg_residual[0] = 1.0
        self.verbose = verbose

    # def least_square_problem(self, R0, nR0, G, Q, clip, threshold, clip_hard):
    #     """Solves the following least square problem
    #     min || G*coeffs + R0 ||^2
    #     in the coefficients coeffs, and calculates the corresponding best step
    #     dx = coeffs * Q

    #     Parameters
    #     ----------
    #     R0 : 1d np.array, np.shape(R0) = self.problem_dimension
    #         Residual at expansion point x_0
    #     nR0 : float
    #         l2 norm of R0
    #     G : 2d np.array, np.shape(G) = [variable, self.problem_dimension]
    #         Collection of values F(x_0 + dx_i) - R_0
    #     Q : 2d np.array, np.shape(Q) = [variable, self.problem_dimension]
    #         Collection of values dx_i
    #     clip : float
    #         maximum step size for each explored direction, in units
    #         of exploratory step dx_i
    #     threshold : float
    #         catches cases of untreated (partial) collinearity
    #     clip_hard : float
    #         maximum step size for cases of untreated (partial) collinearity
    #     """

    #     self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)), G.T), -R0)
    #     self.coeffs = np.clip(self.coeffs, -clip, clip)
    #     self.explained_residual = np.sum(G * self.coeffs[np.newaxis, :], axis=1)
    #     self.relative_unexplained_residual = (
    #         np.linalg.norm(self.explained_residual + R0) / nR0
    #     )
    #     # if self.relative_unexplained_residual > threshold:
    #     #     self.coeffs = np.clip(self.coeffs, -clip_hard, clip_hard)
    #     self.dx = np.sum(Q * self.coeffs[np.newaxis, :], axis=1)

    def least_square_problem_hessenberg(
        self, R0, nR0, G, Q, Hm, clip, threshold, clip_hard
    ):
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
        self.coeffs = np.matmul(
            np.matmul(np.linalg.inv(np.matmul(Hm.T, Hm)), Hm.T),
            -nR0 * self.dummy_hessenberg_residual[: self.n_it + 1],
        )
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        self.explained_residual = np.linalg.norm(np.sum(G * self.coeffs[np.newaxis, :], axis=1))/nR0
        # self.relative_unexplained_residual = (
            # np.linalg.norm(self.explained_residual + R0) / nR0
        # )
        # print(self.n_it, self.relative_unexplained_residual)
        # if self.relative_unexplained_residual > threshold:
        #     self.coeffs = np.clip(self.coeffs, -clip_hard, clip_hard)
        self.dx = np.sum(Q * self.coeffs[np.newaxis, :], axis=1)

    def Arnoldi_unit(
        self,
        x0,  # trial expansion point
        dx,  # norm 1
        # ndx,
        R0,  # residual at trial_current expansion point: Fresidual(trial_current)
        F_function,
        args,
        # step_size
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
        # build residual
        candidate_x = x0 + dx
        R_dx = np.copy(F_function(candidate_x, *args))
        useful_residual = R_dx - R0
        self.G[:,self.n_it] = useful_residual
        
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
        next_candidate /= self.Hm[self.n_it + 1, self.n_it]

        # build the relevant Givens rotation
        givrot = np.eye(self.n_it+2)
        rho = np.dot(self.Omega[self.n_it], self.Hm[:self.n_it+1,self.n_it])
        rr = (rho**2 + self.Hm[self.n_it+1, self.n_it]**2)**.5
        givrot[-2, -2] = givrot[-1, -1] = rho/rr
        givrot[-2, -1] = self.Hm[self.n_it+1, self.n_it]/rr
        givrot[-1, -2] = -1.0*givrot[-2, -1]
        # update Omega matrix
        Omega = np.eye(self.n_it+2)
        Omega[:-1,:-1] = 1.0*self.Omega
        self.Omega = np.matmul(givrot, Omega)
        return next_candidate  # this is norm 1


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
        clip,
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
        self.Hm = np.zeros((self.max_dim+1, self.max_dim))

        # resize step based on residual
        adjusted_step_size = step_size * nR0
        # print('initial residual 0', R0)

        # prepare for first direction exploration
        self.n_it = 0
        self.n_it_tot = 0
        this_step_size = adjusted_step_size * ((1 + self.n_it) ** scaling_with_n)
        
        dx /= np.linalg.norm(dx)
        self.Qn[:, self.n_it] = np.copy(dx)
        dx *= this_step_size
        self.Q[:, self.n_it] = np.copy(dx)

        # print('initial residual 2', R0)

        explore = 1
        while explore:
            # build Arnoldi update
            dx = self.Arnoldi_unit(x0, dx, R0, F_function, args)

            explore = (self.n_it < max_n_directions)
            self.explained_residual = np.abs(self.Omega[-1,0])
            explore *= (self.explained_residual > target_relative_unexplained_residual)
           
            # prepare for next step
            if explore:
                self.n_it += 1
                self.Qn[:, self.n_it] = np.copy(dx)
                this_step_size = adjusted_step_size * ((1 + self.n_it) ** scaling_with_n)
                dx *= this_step_size
                self.Q[:, self.n_it] = np.copy(dx)

        self.coeffs = -nR0 * np.dot(np.linalg.inv(self.Omega[:-1]@self.Hm[:self.n_it+2, :self.n_it+1]), self.Omega[:-1,0])
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        self.dx = np.sum(self.Q[:, :self.n_it+1] * self.coeffs[np.newaxis, :], axis=1)

