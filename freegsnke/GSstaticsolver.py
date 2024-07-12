from copy import deepcopy

import freegsfast
import numpy as np
from freegsfast.gradshafranov import Greens

from . import nk_solver_H as nk_solver

import matplotlib.pyplot as plt

class NKGSsolver:
    """Solver for the non-linear forward Grad Shafranov (GS)
    static problem. Here, the GS problem is written as a root
    problem in the plasma flux psi. This root problem is
    passed to and solved by the NewtonKrylov solver itself,
    class nk_solver.

    The solution domain is set at instantiation time, through the
    input FreeGSFast equilibrium object.

    The non-linear solver itself is called using the 'solve' method.
    """

    def __init__(self, eq):
        """Instantiates the solver object.
        Based on the domain grid of the input equilibrium object, it prepares
        - the linear solver 'self.linear_GS_solver'
        - the response matrix of boundary grid points 'self.greens_boundary'


        Parameters
        ----------
        eq : a freegsfast equilibrium object.
             The domain grid defined by (eq.R, eq.Z) is the solution domain
             adopted for the GS problems. Calls to the nonlinear solver will
             use the grid domain set at instantiation time. Re-instantiation
             is necessary in order to change the propertes of either grid or
             domain.

        """

        # eq is an Equilibrium instance, it has to have the same domain and grid as
        # the ones the solver will be called on

        self.eq = eq
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
        self.linear_GS_solver = freegsfast.multigrid.createVcycle(
            nx,
            ny,
            freegsfast.gradshafranov.GSsparse4thOrder(
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
                [(0, y) for y in np.arange(1, ny - 1)],
                [(nx - 1, y) for y in np.arange(1, ny - 1)],
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
        self.rhs_before_jtor = -freegsfast.gradshafranov.mu0 * eq.R

        self.angle_shift = np.linspace(0,1,4)
        self.twopi = np.pi*2
        # self.shifts_uncoupled = np.array([[-1,0],[0,1],[1,0],[0,-1]])

    def freeboundary(self, plasma_psi, tokamak_psi, profiles):  # , rel_psi_error):
        """Imposes boundary conditions on set of boundary points.

        Parameters
        ----------
        plasma_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the plasma
        tokamak_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the tokamak alone, including all metal currents,
            in both active coils and passive structures
        profiles : freegsfast profile object
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
            # rel_psi_error=rel_psi_error,
        )
        self.rhs = self.rhs_before_jtor * self.jtor

        # calculates and assignes boundary conditions
        self.psi_boundary = np.zeros_like(self.R)
        psi_bnd = np.sum(self.greenfunc * self.jtor[np.newaxis, :, :], axis=(-1, -2))

        self.psi_boundary[:, 0] = psi_bnd[: self.nx]
        self.psi_boundary[:, -1] = psi_bnd[self.nx : 2 * self.nx]
        self.psi_boundary[0, 1 : self.ny - 1] = psi_bnd[
            2 * self.nx : 2 * self.nx + self.ny - 2
        ]
        self.psi_boundary[-1, 1 : self.ny - 1] = psi_bnd[2 * self.nx + self.ny - 2 :]

        self.rhs[0, 1 : self.ny - 1] = self.psi_boundary[0, 1 : self.ny - 1]
        self.rhs[:, 0] = self.psi_boundary[:, 0]
        self.rhs[-1, 1 : self.ny - 1] = self.psi_boundary[-1, 1 : self.ny - 1]
        self.rhs[:, -1] = self.psi_boundary[:, -1]

    def F_function(self, plasma_psi, tokamak_psi, profiles):  # , rel_psi_error=0):
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
        profiles : freegsfast profile object
            profile object describing target plasma properties,
            used to calculate current density jtor

        Returns
        -------
        residual : np.array of size eq.nx*eq.ny
            residual of the GS equation
        """

        self.freeboundary(plasma_psi, tokamak_psi, profiles)  # , rel_psi_error)
        residual = plasma_psi - (
            self.linear_GS_solver(self.psi_boundary, self.rhs)
        ).reshape(-1)
        return residual

    def port_critical(self, eq, profiles):
        """Transfers critical points info from profile to eq after GS solution or Jtor calculation

        Parameters
        ----------
        eq : freegsfast equilibrium object
            Equilibrium on which to record values
        profiles : freegsfast profile object
            Profiles object which has been used to calculate Jtor.
        """
        eq.solved = True

        eq.xpt = np.copy(profiles.xpt)
        eq.opt = np.copy(profiles.opt)
        eq.psi_axis = eq.opt[0, 2]

        eq.psi_bndry = profiles.psi_bndry
        eq.flag_limiter = profiles.flag_limiter

        eq._current = np.sum(profiles.jtor) * self.dRdZ
        eq._profiles = deepcopy(profiles)

        eq.tokamak_psi = self.tokamak_psi.reshape(self.nx, self.ny)

    
    def calculate_explore_metric(self, trial_plasma_psi, profiles):
        """Calculates the F_function residual and normalizes
        to a 0d value for initial plasma_psi exploration

        Parameters
        ----------
        plasma_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the plasma
        profiles : freegsfast profile object
            profile object describing target plasma properties,
            used to calculate current density jtor

        """
        
        res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
        metric, res0 = self.explore_metric(res0, trial_plasma_psi)
        plt.imshow(trial_plasma_psi.reshape(self.nx, self.ny))
        plt.colorbar()
        plt.title('metric = '+str(metric))
        plt.show()
        return metric, res0
    
    
    def explore_metric(self, res0, trial_plasma_psi):
        """Calculates the F_function residual and normalizes
        to a 0d value for initial plasma_psi exploration

        Parameters
        ----------
        plasma_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the plasma
        profiles : freegsfast profile object
            profile object describing target plasma properties,
            used to calculate current density jtor

        """

        metric = np.linalg.norm(res0)/np.linalg.norm(trial_plasma_psi)
        # if Jsize_coeff != None:
        #     metric -= Jsize_coeff*np.sum(profiles.jtor>0)/np.sum(profiles.mask_inside_limiter)
        return metric, res0


    def psi_explore(self, reference_trial_psi, reference_trial_pars, profiles, std_shifts, ref_res0=None, successf_shift=None):
        """Can perform an initial exploration of the space of gaussian plasma_psis, to find 
        a better initial guess to start the actual solver on. Exploration 
        includes the plasma_psi 'centre', normalization and exponential degree.

        Parameters
        ----------
        reference_trial_psi : np.array of size 4
            current values for [xc, yc, norm, exp]
        profiles : freegsfast profile object
            profile object describing target plasma properties,
            used to calculate current density jtor
        std_shifts : np.array of size 4
            standard dev on steps for [xc, yc, norm, exp]
        ref_res0 : np.array of size eq.nx*eq.ny
            result of F_function on reference_trial_psi, by default None
        successf_shift : np.array of size 4
            previous succesful step values for [xc, yc, norm, exp]

    
        """
        if ref_res0 is None:
            ref_metric, ref_res0 = self.calculate_explore_metric(reference_trial_psi, profiles)
        else:
            ref_metric, ref_res0 = self.explore_metric(ref_res0, reference_trial_psi)
        
        if successf_shift is not None:
            shift = 1.0*successf_shift
        else:
            shift = np.random.randn(4)*std_shifts
        
        try:
            self.trial_plasma_pars = reference_trial_pars + shift
            if self.trial_plasma_pars[0]>1:
                self.trial_plasma_pars = 1
            self.trial_plasma_psi = self.eq.create_psi_plasma_default(gpars=self.trial_plasma_pars).reshape(-1)

            metric, res0 = self.calculate_explore_metric(self.trial_plasma_psi, profiles)
            print('metrics', metric, ref_metric)
            if metric < ref_metric:
                return self.trial_plasma_psi, self.trial_plasma_pars, metric, res0, shift
        except:
            pass

        return reference_trial_psi, reference_trial_pars, ref_metric, ref_res0, None


    def forward_solve(
        self,
        eq,
        profiles,
        target_relative_tolerance,
        max_solving_iterations=50,
        Picard_handover=0.15,
        step_size=2.5,
        scaling_with_n=-1.2,
        target_relative_unexplained_residual=0.3,
        max_n_directions=12,
        clip=10,
        verbose=False,
        max_rel_step_size=0.25,
        max_rel_update_size=0.2,
    ):
        """The method that actually solves the forward GS problem.

        A forward problem is specified by the 2 freegsfast objects eq and profiles.
        The first specifies the metal currents (throught eq.tokamak)
        and the second specifies the desired plasma properties
        (i.e. plasma current and profile functions).

        The plasma_psi which solves the given GS problem is assigned to
        the input eq, and can be found at eq.plasma_psi.

        Parameters
        ----------
        eq : freegsfast equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : freegsfast profile object
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
        forcing_Picard = False
        trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
        self.tokamak_psi = (eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)).reshape(-1)

        log = []

        control_trial_psi = False
        n_up = 0.0 + 4 * eq.solved
        # this tries to cure cases where plasma_psi is not large enough in modulus
        # causing no core mask to exist
        while (control_trial_psi is False) and (n_up < 5):
            try:
                res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
                control_trial_psi = True
                log.append("first residual found " + str(np.linalg.norm(res0)))
                jmap = 1.0*(profiles.jtor>0)
            except:
                trial_plasma_psi /= 0.8
                n_up += 1
                log.append("residual failed, try /.8")
        # this is in case the above did not work
        # then use standard initialization
        # and grow peak until core mask exists
        if control_trial_psi is False:
            log.append("Default plasma_psi initialization and adjustment invoked")
            eq.plasma_psi = trial_plasma_psi = eq.create_psi_plasma_default(adaptive_centre=True)
            eq.adjust_psi_plasma()
            trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
            res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
            control_trial_psi = True
            # n_up = 0
            # while (control_trial_psi is False) and (n_up < 10):
            #     try:
            #         res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
            #         control_trial_psi = True
            #     except:
            #         trial_plasma_psi /= .6
            #         n_up += 1
            #         print('/.6')

        ares0 = np.linalg.norm(res0)
        del_psi = np.linalg.norm(trial_plasma_psi)
        # a_and_r_res0 = ares0/del_psi # + 0.5*ares0
        # print('a_and_r_res0', a_and_r_res0, ares0)

        # # record for debugging
        # self.first_jtor = profiles.jtor.copy()

        # # if there's been no increase in trial_plasma_psi
        # # check if it would be advantageous to decrease it
        # n_check = (n_up < 1)
        # while n_check:
        #     try:
        #         n_trial_plasma_psi = .8 * trial_plasma_psi
        #         n_del_psi = .8 * del_psi
        #         res_n = self.F_function(n_trial_plasma_psi, self.tokamak_psi, profiles)
        #         ares_n = np.amax(res_n) - np.amin(res_n)
        #         a_and_r_res_n = ares_n/n_del_psi # + .5*ares_n
        #         n_check = (a_and_r_res_n < .9*a_and_r_res0)*(a_and_r_res0 > .3)
        #         if n_check:
        #             print('*.8 -- a_and_r_res_n', a_and_r_res_n, ares_n)
        #             trial_plasma_psi = 1.0 * n_trial_plasma_psi
        #             del_psi = 1.0 * n_del_psi
        #             res0 = 1.0 * res_n
        #             ares0 = 1.0 * ares_n
        #             a_and_r_res0 = 1.0 * a_and_r_res_n
        #             # step_size *= .9
        #     except:
        #         n_check = False

        log.append("del_psi " + str(del_psi))

        self.jtor_at_start = profiles.jtor.copy()

        # res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
        # print('initial residual', res0)
        # rel_change = np.amax(np.abs(res0))
        # del_psi = np.amax(trial_plasma_psi) - np.amin(trial_plasma_psi)
        rel_change = ares0 / del_psi
        self.relative_change = 1.0 * rel_change

        args = [self.tokamak_psi, profiles]  # , rel_change]

        starting_direction = np.copy(res0)

        if verbose:
            for x in log:
                print(x)


        log = []
        iterations = 0
        while (rel_change > target_relative_tolerance) * (
            iterations < max_solving_iterations
        ):
            # plt.imshow(profiles.jtor)
            # plt.colorbar()
            # plt.title('Jtor')
            # plt.show()
            # plt.imshow(res0.reshape(self.nx, self.ny))
            # plt.colorbar()
            # plt.title('residual')
            # plt.show()
            # plt.imshow(trial_plasma_psi.reshape(self.nx, self.ny))
            # plt.colorbar()
            # plt.title('trial_psi')
            # plt.show()

            if rel_change > Picard_handover or forcing_Picard:
                log.append("Picard iteration" + str(iterations))
                # using Picard instead of NK
                res0_2d = res0.reshape(self.nx, self.ny)
                update = -0.5*(res0_2d + res0_2d[:,::-1]).reshape(-1)
                print('Picard update')
                # trial_plasma_psi -= res0
                picard_flag = True
                forcing_Picard = False
                print('done Picard update')

            else:
                print('NK update')
                log.append("NK iteration " + str(iterations))
                picard_flag = False
                self.nksolver.Arnoldi_iteration(
                    x0=1.0*trial_plasma_psi,  # trial_current expansion point
                    dx=1.0*starting_direction,  # first vector for current basis
                    R0=1.0*res0,  # circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                    F_function=self.F_function,
                    args=args,
                    step_size=step_size,
                    scaling_with_n=scaling_with_n,
                    target_relative_unexplained_residual=target_relative_unexplained_residual,
                    max_n_directions=max_n_directions,  # max number of basis vectors (must be less than number of modes + 1)
                    # max_Arnoldi_iterations=max_Arnoldi_iterations,
                    # max_collinearity=max_collinearity,
                    clip=clip,
                    # threshold=threshold,
                    # clip_hard=clip_hard,
                    # max_rel_step_size=max_rel_step_size,
                )
                # print(self.nksolver.coeffs)
                update = 1.0 * self.nksolver.dx
                # limit update size where necessary
                print('done NK update')
                print('self.nksolver.coeffs',self.nksolver.coeffs, self.nksolver.explained_residual)

            del_update = np.linalg.norm(update)
            if del_update / del_psi > max_rel_update_size:
                print("update > max_rel_update_size. Reduced.")
                # log.append("update > max_rel_update_size. Reduced.")
                update *= np.abs(max_rel_update_size * del_psi / del_update)

            # plt.imshow(update.reshape(self.nx, self.ny))
            # plt.colorbar()
            # plt.title('Update')
            # plt.show()

            new_residual_flag = True
            while new_residual_flag:
                print('start new_residual_flag')
                try:
                    n_trial_plasma_psi = trial_plasma_psi + update
                    new_res0 = self.F_function(
                        n_trial_plasma_psi, self.tokamak_psi, profiles
                    )
                    new_jmap = 1.0*(profiles.jtor>0)
                    print('jmap_difference', np.sum((new_jmap-jmap)**2))
                    new_rel_change = np.linalg.norm(new_res0)
                    n_del_psi = np.linalg.norm(n_trial_plasma_psi)
                    new_rel_change = new_rel_change / n_del_psi
                    new_residual_flag = False
                    print('new_rel_change', new_rel_change)
                    # plt.imshow(n_trial_plasma_psi.reshape(self.nx, self.ny))
                    # plt.colorbar()
                    # plt.title('n_trial_plasma_psi')
                    # plt.show()
                    # plt.imshow(new_res0.reshape(self.nx, self.ny))
                    # plt.colorbar()
                    # plt.title('new_residual')
                    # plt.show()
                except:
                    log.append(
                        "Trigger update reduction due to failure to find an Xpoint, try *.75"
                    )
                    update *= 0.75

            if new_rel_change < self.relative_change:
                trial_plasma_psi = n_trial_plasma_psi.copy()
                residual_collinearity = np.sum(res0 * new_res0) / (
                    np.linalg.norm(res0) * np.linalg.norm(new_res0)
                )
                res0 = 1.0 * new_res0
                if (residual_collinearity > 0.9) and (picard_flag is False):
                    log.append("New starting_direction used due to collinear residuals")
                    # print('residual_collinearity', residual_collinearity)
                    # forcing_Picard = True
                    starting_direction = np.sin(
                        np.linspace(0, 2 * np.pi, self.nx) * 1.5 * np.random.random()
                    )[:, np.newaxis]
                    starting_direction = (
                        starting_direction
                        * np.sin(
                            np.linspace(0, 2 * np.pi, self.ny)
                            * 1.5
                            * np.random.random()
                        )[np.newaxis, :]
                    )
                    starting_direction = starting_direction.reshape(-1)
                    starting_direction *= trial_plasma_psi
                    # plt.imshow(starting_direction.reshape(self.nx, self.ny))
                    # plt.colorbar()
                    # plt.title('new starting_direction used!')
                    # plt.show()
                else:
                    starting_direction = np.copy(res0)
                rel_change = 1.0 * new_rel_change
                del_psi = 1.0 * n_del_psi
            else:
                log.append(
                    "Increase in residual, update reduction triggered."
                )
                print("Increase in residual, update reduction triggered. Returning")
                # if 
                # return
                update *= 0.5
                # plt.imshow(update.reshape(self.nx, self.ny))
                # plt.colorbar()
                # plt.title('Update ')
                # plt.show()
                trial_plasma_psi += update
                res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
                starting_direction = np.copy(res0)
                rel_change = np.linalg.norm(res0)
                rel_change /= np.linalg.norm(trial_plasma_psi)

            self.relative_change = 1.0 * rel_change
            # args[2] = 1.0*rel_change
            log.append("rel_change " + str(rel_change))

            if verbose:
                for x in log:
                    print(x)

            log = []

            iterations += 1

        # update eq with new solution
        eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()

        # plt.imshow(res0.reshape(self.nx, self.ny))
        # plt.colorbar()
        # plt.title('residual')
        # plt.show()
        # plt.imshow(trial_plasma_psi.reshape(self.nx, self.ny))
        # plt.colorbar()
        # plt.title('trial_psi')
        # plt.show()

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
    #     eq : freegsfast equilibrium object
    #         Used to extract the assigned metal currents, which in turn are
    #         used to calculate the according self.tokamak_psi
    #     profiles : freegsfast profile object
    #         Specifies the target properties of the plasma.
    #         These are used to calculate Jtor(psi)
    #     target_relative_tolerance : float
    #         NK iterations are interrupted when this criterion is
    #         satisfied. Relative convergence
    #     constrain : freegsfast constrain object
    #         specifies the desired constraints on the configuration of magnetic flux (xpoints and isoflux, as in FreeGSFast)
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
    #         # adjust coil currents, using freegsfast leastsquares
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
        max_solving_iterations=50,
        Picard_handover=0.07,
        blend=0.0,
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
        Syntax is analogous to FreeGSFast:
            - an inverse solve is specified by the 'constrain' input,
            which includes the desired constraints on the configuration of magnetic flux (xpoints and isoflux, as in FreeGSFast).
            The optimization over the coil currents uses the FreeGSFast implementation, as a simple regularised least square problem.
            - a forward solve has constrain=None. Please see forward_solve for details.


        Parameters
        ----------
        eq : freegsfast equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : freegsfast profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        target_relative_tolerance : float
            NK iterations are interrupted when this criterion is
            satisfied. Relative convergence
        constrain : freegsfast constrain object
            specifies the desired constraints on the configuration of magnetic flux (xpoints and isoflux, as in FreeGSFast)
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
            freegsfast.solve(
                eq,
                profiles,
                constrain,
                rtol=target_relative_tolerance,
                show=False,
                blend=blend,
            )
