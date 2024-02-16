import freegs
import numpy as np
from freegs.gradshafranov import Greens
from copy import deepcopy

from . import nk_solver as nk_solver


class NKGSsolver:
    """Solver for the non-linear forward Grad Shafranov (GS)
    static problem. Here, the GS problem is written as a root
    problem in the plasma flux psi. This root problem is
    passed to and solved by the NewtonKrylov solver itself,
    class nk_solver.

    The solution domain is set at instantiation time, through the
    input freeGS equilibrium object.

    The non-linear solver itself is called using the 'solve' method.
    """

    def __init__(self, eq):
        """Instantiates the solver object.
        Based on the domain grid of the input equilibrium object, it prepares
        - the linear solver 'self.linear_GS_solver'
        - the response matrix of boundary grid points 'self.greens_boundary'


        Parameters
        ----------
        eq : a freeGS equilibrium object.
             The domain grid defined by (eq.R, eq.Z) is the solution domain
             adopted for the GS problems. Calls to the nonlinear solver will
             use the grid domain set at instantiation time. Re-instantiation
             is necessary in order to change the propertes of either grid or
             domain.

        """

        # eq is an Equilibrium instance, it has to have the same domain and grid as
        # the ones the solver will be called on

        R = eq.R
        Z = eq.Z
        self.R = R
        self.Z = Z
        R_1D = R[:, 0]
        Z_1D = Z[0, :]

        # for reshaping
        nx, ny = np.shape(R)
        self.nx = nx
        self.ny = ny

        # for integration
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]
        self.dRdZ = dR * dZ

        self.nksolver = nk_solver.nksolver(problem_dimension=self.nx * self.ny)

        # linear solver for del*Psi=RHS with fixed RHS
        self.linear_GS_solver = freegs.multigrid.createVcycle(
            nx,
            ny,
            freegs.gradshafranov.GSsparse4thOrder(
                eq.R[0, 0], eq.R[-1, 0], eq.Z[0, 0], eq.Z[0, -1]
            ),
            nlevels=1,
            ncycle=1,
            niter=2,
            direct=True,
        )

        # List of indices on the boundary
        bndry_indices = np.concatenate(
            [
                [(x, 0) for x in range(nx)],
                [(x, ny - 1) for x in range(nx)],
                [(0, y) for y in range(ny)],
                [(nx - 1, y) for y in range(ny)],
            ]
        )
        self.bndry_indices = bndry_indices

        # matrices of responses of boundary locations to each grid positions
        greenfunc = Greens(
            R[np.newaxis, :, :],
            Z[np.newaxis, :, :],
            R_1D[bndry_indices[:, 0]][:, np.newaxis, np.newaxis],
            Z_1D[bndry_indices[:, 1]][:, np.newaxis, np.newaxis],
        )
        # Prevent infinity/nan by removing Greens(x,y;x,y)
        zeros = np.ones_like(greenfunc)
        zeros[
            np.arange(len(bndry_indices)), bndry_indices[:, 0], bndry_indices[:, 1]
        ] = 0
        self.greenfunc = greenfunc * zeros * self.dRdZ

        # RHS/Jtor
        self.rhs_before_jtor = -freegs.gradshafranov.mu0 * eq.R

    def freeboundary(self, plasma_psi, tokamak_psi, profiles, rel_psi_error):
        """Imposes boundary conditions on set of boundary points.

        Parameters
        ----------
        plasma_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the plasma
        tokamak_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the tokamak alone, including all metal currents,
            in both active coils and passive structures
        profiles : freeGS profile object
            profile object describing target plasma properties,
            used to calculate current density jtor
        """

        # tokamak_psi is psi from the currents assigned to the tokamak coils in eq, ie.
        # tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)

        # jtor and RHS given tokamak_psi above and the input plasma_psi
        self.jtor = profiles.Jtor(
            self.R,
            self.Z,
            (tokamak_psi + plasma_psi).reshape(self.nx, self.ny),
            rel_psi_error=rel_psi_error,
        )
        self.rhs = self.rhs_before_jtor * self.jtor

        # calculates and assignes boundary conditions
        self.psi_boundary = np.zeros_like(self.R)
        psi_bnd = np.sum(self.greenfunc * self.jtor[np.newaxis, :, :], axis=(-1, -2))

        self.psi_boundary[:, 0] = psi_bnd[: self.nx]
        self.psi_boundary[:, -1] = psi_bnd[self.nx : 2 * self.nx]
        self.psi_boundary[0, :] = psi_bnd[2 * self.nx : 2 * self.nx + self.ny]
        self.psi_boundary[-1, :] = psi_bnd[2 * self.nx + self.ny :]

        self.rhs[0, :] = self.psi_boundary[0, :]
        self.rhs[:, 0] = self.psi_boundary[:, 0]
        self.rhs[-1, :] = self.psi_boundary[-1, :]
        self.rhs[:, -1] = self.psi_boundary[:, -1]

    def F_function(self, plasma_psi, tokamak_psi, profiles, rel_psi_error=0):
        """Nonlinear Grad Shafranov equation written as a root problem
        F(plasma_psi) \equiv [\delta* - J](plasma_psi)
        The plasma_psi that solves the Grad Shafranov problem satisfies
        F(plasma_psi) = [\delta* - J](plasma_psi) = 0


        Parameters
        ----------
        plasma_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the plasma
        tokamak_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the tokamak alone, including all metal currents,
            in both active coils and passive structures
        profiles : freeGS profile object
            profile object describing target plasma properties,
            used to calculate current density jtor

        Returns
        -------
        residual : np.array of size eq.nx*eq.ny
            residual of the GS equation
        """

        self.freeboundary(plasma_psi, tokamak_psi, profiles, rel_psi_error)
        residual = plasma_psi - (
            self.linear_GS_solver(self.psi_boundary, self.rhs)
        ).reshape(-1)
        return residual

    def port_critical(self, eq, profiles):
        """Transfers critical points info from profile to eq after GS solution or Jtor calculation

        Parameters
        ----------
        eq : freeGS equilibrium object
            Equilibrium on which to record values
        profiles : freeGS profile object
            Profiles object which has been used to calculate Jtor.
        """
        eq.xpt = np.copy(profiles.xpt)
        eq.opt = np.copy(profiles.opt)
        eq.psi_axis = eq.opt[0, 2]

        # eq.psi_bndry = eq.xpt[0,2]
        eq.psi_bndry = profiles.psi_bndry
        eq.flag_limiter = profiles.flag_limiter

    def forward_solve(
        self,
        eq,
        profiles,
        target_relative_tolerance,
        max_solving_iterations=50,
        Picard_handover=0.07,
        step_size=2.5,
        scaling_with_n=-1.2,
        target_relative_unexplained_residual=0.25,
        max_n_directions=8,
        max_Arnoldi_iterations=10,
        max_collinearity=0.99,
        clip=10,
        threshold=3,
        clip_hard=2,
        verbose=False,
    ):
        """The method that actually solves the forward GS problem.

        A forward problem is specified by the 2 freeGS objects eq and profiles.
        The first specifies the metal currents (throught eq.tokamak)
        and the second specifies the desired plasma properties
        (i.e. plasma current and profile functions).

        The plasma_psi which solves the given GS problem is assigned to
        the input eq, and can be found at eq.plasma_psi.

        Parameters
        ----------
        eq : freeGS equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : freeGS profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        target_relative_tolerance : float
            NK iterations are interrupted when this criterion is
            satisfied. Relative convergence
        max_solving_iterations : int
            NK iterations are interrupted when this limit is surpassed
        Picard_handover : float
            Value of relative tolerance above which a Picard iteration
            is performed instead of a full NK call
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
        verbose : bool
            flag to allow warning messages when Picard is used instead of NK

        """

        picard_flag = 0
        self.profiles = profiles
        trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
        self.tokamak_psi = (eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)).reshape(-1)

        res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, self.profiles)
        # print('initial residual', res0)
        rel_change = np.amax(np.abs(res0))
        del_psi = np.amax(trial_plasma_psi) - np.amin(trial_plasma_psi)
        rel_change /= del_psi

        args = [self.tokamak_psi, self.profiles, rel_change]

        iterations = 0
        while (rel_change > target_relative_tolerance) * (
            iterations < max_solving_iterations
        ):

            if rel_change > Picard_handover:
                # using Picard instead of NK
                trial_plasma_psi -= res0
                picard_flag = 1

            else:
                self.nksolver.Arnoldi_iteration(
                    x0=trial_plasma_psi,  # trial_current expansion point
                    dx=res0,  # first vector for current basis
                    R0=res0,  # circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                    F_function=self.F_function,
                    args=args,
                    step_size=step_size,
                    scaling_with_n=scaling_with_n,
                    target_relative_unexplained_residual=target_relative_unexplained_residual,
                    max_n_directions=max_n_directions,  # max number of basis vectors (must be less than number of modes + 1)
                    max_Arnoldi_iterations=max_Arnoldi_iterations,
                    max_collinearity=max_collinearity,
                    clip=clip,
                    threshold=threshold,
                    clip_hard=clip_hard,
                )
                # print(self.nksolver.coeffs)
                trial_plasma_psi += self.nksolver.dx

            res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, self.profiles)
            rel_change = np.amax(np.abs(res0))
            rel_change /= np.amax(trial_plasma_psi) - np.amin(trial_plasma_psi)
            self.relative_change = 1.0 * rel_change
            args[2] = rel_change

            iterations += 1

        # update eq with new solution
        eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()

        # update plasma current
        eq._current = np.sum(profiles.jtor) * self.dRdZ
        eq._profiles = deepcopy(profiles)

        self.port_critical(eq=eq, profiles=profiles)

        if picard_flag and verbose:
            print("Picard was used instead of NK in at least 1 cycle.")

        # if max_iter was hit, then message:
        if iterations >= max_solving_iterations:
            print(
                "failed to converge to the requested relative tolerance of {} with less than {} iterations".format(
                    target_relative_tolerance, max_solving_iterations
                )
            )
            print(f"last relative psi change = {rel_change}")

    # def inverse_solve(
    #     self,
    #     eq,
    #     profiles,
    #     target_relative_tolerance,
    #     constrain,
    #     max_solving_iterations=30,
    #     Picard_handover=0.07,
    #     blend=.5,
    #     step_size=2.5,
    #     scaling_with_n=-1.2,
    #     target_relative_unexplained_residual=0.25,
    #     max_n_directions=8,
    #     max_Arnoldi_iterations=10,
    #     max_collinearity=0.99,
    #     clip=10,
    #     threshold=3,
    #     clip_hard=2,
    #     verbose=False,
    # ):
    #     """The method to solve the inverse GS problems, therefore a constrain object is required.

    #     Parameters
    #     ----------
    #     eq : freeGS equilibrium object
    #         Used to extract the assigned metal currents, which in turn are
    #         used to calculate the according self.tokamak_psi
    #     profiles : freeGS profile object
    #         Specifies the target properties of the plasma.
    #         These are used to calculate Jtor(psi)
    #     target_relative_tolerance : float
    #         NK iterations are interrupted when this criterion is
    #         satisfied. Relative convergence
    #     constrain : freeGS constrain object
    #         specifies the desired constraints on the configuration of magnetic flux (xpoints and isoflux, as in FreeGS)
    #     max_solving_iterations : int
    #         NK iterations are interrupted when this limit is surpassed
    #     Picard_handover : float
    #         Value of relative tolerance above which a Picard iteration
    #         is performed instead of a full NK call
    #     step_size : float
    #         l2 norm of proposed step
    #     scaling_with_n : float
    #         allows to further scale dx candidate steps by factor
    #         (1 + self.n_it)**scaling_with_n
    #     target_relative_explained_residual : float between 0 and 1
    #         terminates iteration when exploration can explain this
    #         fraction of the initial residual R0
    #     max_n_directions : int
    #         terminates iteration even though condition on
    #         explained residual is not met
    #     max_Arnoldi_iterations : int
    #         terminates iteration after attempting to explore
    #         this number of directions
    #     max_collinearity : float between 0 and 1
    #         rejects a candidate direction if resulting residual
    #         is collinear to any of those stored previously
    #     clip : float
    #         maximum step size for each explored direction, in units
    #         of exploratory step dx_i
    #     threshold : float
    #         catches cases of untreated (partial) collinearity
    #     clip_hard : float
    #         maximum step size for cases of untreated (partial) collinearity
    #     verbose : bool
    #         flag to allow warning message in case of failed convergence within requested max_solving_iterations

    #     """

    #     picard_flag = 0
    #     self.profiles = profiles
    #     trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
    #     self.tokamak_psi = (eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)).reshape(-1)

    #     res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, self.profiles)
    #     # print('initial residual', res0)
    #     rel_change = np.amax(np.abs(res0))
    #     del_psi = np.amax(trial_plasma_psi) - np.amin(trial_plasma_psi)
    #     rel_change /= del_psi

    #     args = [self.tokamak_psi, self.profiles, rel_change]

    #     iterations = 0
    #     while (rel_change > target_relative_tolerance) * (
    #         iterations < max_solving_iterations
    #     ):

    #         if rel_change > Picard_handover:
    #             # using Picard instead of NK
    #             trial_plasma_psi -= res0
    #             picard_flag = 1

    #         else:
    #             self.nksolver.Arnoldi_iteration(
    #                 x0=trial_plasma_psi,  # trial_current expansion point
    #                 dx=res0,  # first vector for current basis
    #                 R0=res0,  # circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
    #                 F_function=self.F_function,
    #                 args=args,
    #                 step_size=step_size,
    #                 scaling_with_n=scaling_with_n,
    #                 target_relative_unexplained_residual=target_relative_unexplained_residual,
    #                 max_n_directions=max_n_directions,  # max number of basis vectors (must be less than number of modes + 1)
    #                 max_Arnoldi_iterations=max_Arnoldi_iterations,
    #                 max_collinearity=max_collinearity,
    #                 clip=clip,
    #                 threshold=threshold,
    #                 clip_hard=clip_hard,
    #             )
    #             # print(self.nksolver.coeffs)
    #             trial_plasma_psi += self.nksolver.dx

    #         # update eq with new solution
    #         eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()
    #         # adjust coil currents, using freeGS leastsquares
    #         constrain(eq)
    #         # update tokamak_psi accordingly
    #         self.tokamak_psi = (eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)).reshape(-1)

    #         res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, self.profiles)
    #         rel_change = np.amax(np.abs(res0))
    #         rel_change /= np.amax(trial_plasma_psi) - np.amin(trial_plasma_psi)
    #         self.relative_change = 1.0 * rel_change
    #         args[2] = rel_change

    #         iterations += 1

    #     # # update eq with new solution
    #     # eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()

    #     # update plasma current
    #     eq._current = np.sum(profiles.jtor) * self.dRdZ
    #     eq._profiles = profiles

    #     self.port_critical(eq=eq, profiles=profiles)

    #     if picard_flag and verbose:
    #         print("Picard was used instead of NK in at least 1 cycle.")

    #     # if max_iter was hit, then message:
    #     if iterations >= max_solving_iterations:
    #         print(
    #             "failed to converge to the requested relative tolerance of {} with less than {} iterations".format(
    #                 target_relative_tolerance,
    #                 max_solving_iterations
    #             )
    #         )
    #         print(f"last relative psi change = {rel_change}")

    def solve(
        self,
        eq,
        profiles,
        target_relative_tolerance,
        constrain=None,
        max_solving_iterations=30,
        Picard_handover=0.07,
        step_size=2.5,
        scaling_with_n=-1.2,
        target_relative_unexplained_residual=0.25,
        max_n_directions=8,
        max_Arnoldi_iterations=10,
        max_collinearity=0.99,
        clip=10,
        threshold=3,
        clip_hard=2,
        verbose=False,
    ):
        """The method to solve the GS problems, both forward and inverse.
        Syntax is analogous to freeGS:
            - an inverse solve is specified by the 'constrain' input,
            which includes the desired constraints on the configuration of magnetic flux (xpoints and isoflux, as in FreeGS).
            The optimization over the coil currents uses the freeGS implementation, as a simple regularised least square problem.
            - a forward solve has constrain=None. Please see forward_solve for details.


        Parameters
        ----------
        eq : freeGS equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : freeGS profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        target_relative_tolerance : float
            NK iterations are interrupted when this criterion is
            satisfied. Relative convergence
        constrain : freeGS constrain object
            specifies the desired constraints on the configuration of magnetic flux (xpoints and isoflux, as in FreeGS)
        max_solving_iterations : int
            NK iterations are interrupted when this limit is surpassed
        Picard_handover : float
            Value of relative tolerance above which a Picard iteration
            is performed instead of a full NK call
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
        verbose : bool
            flag to allow warning message in case of failed convergence within requested max_solving_iterations

        """

        if constrain is None:
            self.forward_solve(
                eq,
                profiles,
                target_relative_tolerance,
                max_solving_iterations,
                Picard_handover,
                step_size,
                scaling_with_n,
                target_relative_unexplained_residual,
                max_n_directions,
                max_Arnoldi_iterations,
                max_collinearity,
                clip,
                threshold,
                clip_hard,
                verbose,
            )

        else:
            freegs.solve(
                eq, profiles, constrain, rtol=target_relative_tolerance, show=False
            )
