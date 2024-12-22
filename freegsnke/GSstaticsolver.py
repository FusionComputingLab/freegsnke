import warnings
from copy import deepcopy

import freegs4e
import matplotlib.pyplot as plt
import numpy as np
from freegs4e.gradshafranov import Greens

from . import nk_solver_H as nk_solver


class NKGSsolver:
    """Solver for the non-linear forward Grad Shafranov (GS)
    static problem. Here, the GS problem is written as a root
    problem in the plasma flux psi. This root problem is
    passed to and solved by the NewtonKrylov solver itself,
    class nk_solver.

    The solution domain is set at instantiation time, through the
    input FreeGS4E equilibrium object.

    The non-linear solver itself is called using the 'solve' method.
    """

    def __init__(self, eq):
        """Instantiates the solver object.
        Based on the domain grid of the input equilibrium object, it prepares
        - the linear solver 'self.linear_GS_solver'
        - the response matrix of boundary grid points 'self.greens_boundary'


        Parameters
        ----------
        eq : a freegs4e equilibrium object.
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
        self.linear_GS_solver = freegs4e.multigrid.createVcycle(
            nx,
            ny,
            freegs4e.gradshafranov.GSsparse4thOrder(
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
        self.rhs_before_jtor = -freegs4e.gradshafranov.mu0 * eq.R

        self.angle_shift = np.linspace(0, 1, 4)
        self.twopi = np.pi * 2
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
        profiles : freegs4e profile object
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

        self.rhs[0, :] = self.psi_boundary[0, :]
        self.rhs[:, 0] = self.psi_boundary[:, 0]
        self.rhs[-1, :] = self.psi_boundary[-1, :]
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
        profiles : freegs4e profile object
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
        eq : freegs4e equilibrium object
            Equilibrium on which to record values
        profiles : freegs4e profile object
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
        profiles : freegs4e profile object
            profile object describing target plasma properties,
            used to calculate current density jtor

        """

        res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
        metric, res0 = self.explore_metric(res0, trial_plasma_psi)
        plt.imshow(trial_plasma_psi.reshape(self.nx, self.ny))
        plt.colorbar()
        plt.title("metric = " + str(metric))
        plt.show()
        return metric, res0

    def explore_metric(self, res0, trial_plasma_psi):
        """Calculates the F_function residual and normalizes
        to a 0d value for initial plasma_psi exploration

        Parameters
        ----------
        plasma_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the plasma
        profiles : freegs4e profile object
            profile object describing target plasma properties,
            used to calculate current density jtor

        """

        metric = np.linalg.norm(res0) / np.linalg.norm(trial_plasma_psi)
        # if Jsize_coeff != None:
        #     metric -= Jsize_coeff*np.sum(profiles.jtor>0)/np.sum(profiles.mask_inside_limiter)
        return metric, res0

    def psi_explore(
        self,
        reference_trial_psi,
        reference_trial_pars,
        profiles,
        std_shifts,
        ref_res0=None,
        successf_shift=None,
    ):
        """Can perform an initial exploration of the space of gaussian plasma_psis, to find
        a better initial guess to start the actual solver on. Exploration
        includes the plasma_psi 'centre', normalization and exponential degree.

        Parameters
        ----------
        reference_trial_psi : np.array of size 4
            current values for [xc, yc, norm, exp]
        profiles : freegs4e profile object
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
            ref_metric, ref_res0 = self.calculate_explore_metric(
                reference_trial_psi, profiles
            )
        else:
            ref_metric, ref_res0 = self.explore_metric(ref_res0, reference_trial_psi)

        if successf_shift is not None:
            shift = 1.0 * successf_shift
        else:
            shift = np.random.randn(4) * std_shifts

        try:
            self.trial_plasma_pars = reference_trial_pars + shift
            if self.trial_plasma_pars[0] > 1:
                self.trial_plasma_pars = 1
            self.trial_plasma_psi = self.eq.create_psi_plasma_default(
                gpars=self.trial_plasma_pars
            ).reshape(-1)

            metric, res0 = self.calculate_explore_metric(
                self.trial_plasma_psi, profiles
            )
            print("metrics", metric, ref_metric)
            if metric < ref_metric:
                return (
                    self.trial_plasma_psi,
                    self.trial_plasma_pars,
                    metric,
                    res0,
                    shift,
                )
        except:
            pass

        return reference_trial_psi, reference_trial_pars, ref_metric, ref_res0, None

    def relative_norm_residual(self, res, psi):
        return np.linalg.norm(res) / np.linalg.norm(psi)

    def relative_del_residual(self, res, psi):
        del_psi = np.amax(psi) - np.amin(psi)
        del_res = np.amax(res) - np.amin(res)
        return del_res / del_psi, del_psi

    def forward_solve(
        self,
        eq,
        profiles,
        target_relative_tolerance,
        max_solving_iterations=50,
        Picard_handover=0.15,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.2,
        max_n_directions=16,
        clip=10,
        verbose=False,
        max_rel_update_size=0.2,
    ):
        """The method that actually solves the forward GS problem.

        A forward problem is specified by the 2 freegs4e objects eq and profiles.
        The first specifies the metal currents (throught eq.tokamak)
        and the second specifies the desired plasma properties
        (i.e. plasma current and profile functions).

        The plasma_psi which solves the given GS problem is assigned to
        the input eq, and can be found at eq.plasma_psi.

        Parameters
        ----------
        eq : freegs4e equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : freegs4e profile object
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
        verbose : bool
            flag to allow warning messages when Picard is used instead of NK

        """

        picard_flag = 0
        # forcing_Picard = False
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
                log.append("Initial guess for plasma_psi successful, residual found.")
                # jmap = 1.0 * (profiles.jtor > 0)
            except:
                trial_plasma_psi /= 0.8
                n_up += 1
                log.append("Initial guess for plasma_psi failed, trying to scale...")
        # this is in case the above did not work
        # then use standard initialization
        # and grow peak until core mask exists
        if control_trial_psi is False:
            log.append("Default plasma_psi initialisation and adjustment invoked.")
            eq.plasma_psi = trial_plasma_psi = eq.create_psi_plasma_default(
                adaptive_centre=True
            )
            eq.adjust_psi_plasma()
            trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
            res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
            control_trial_psi = True

        self.jtor_at_start = profiles.jtor.copy()

        norm_rel_change = self.relative_norm_residual(res0, trial_plasma_psi)
        rel_change, del_psi = self.relative_del_residual(res0, trial_plasma_psi)
        self.relative_change = 1.0 * rel_change
        self.norm_rel_change = [norm_rel_change]
        # log.append("del_psi " + str(del_psi))

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

            if rel_change > Picard_handover:  # or forcing_Picard:
                log.append("-----")
                log.append("Picard iteration: " + str(iterations))
                # using Picard instead of NK

                if picard_flag < 3:
                    res0_2d = res0.reshape(self.nx, self.ny)
                    update = -0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                    # print('Picard update')
                    # trial_plasma_psi -= res0
                    picard_flag += 1
                else:
                    update = -1.0 * res0
                    picard_flag = 1
                # forcing_Picard = False
                # print('done Picard update')

            else:
                # print('NK update')
                log.append("-----")
                log.append("Newton-Krylov iteration: " + str(iterations))
                picard_flag = False
                self.nksolver.Arnoldi_iteration(
                    x0=trial_plasma_psi.copy(),  # trial_current expansion point
                    dx=starting_direction.copy(),  # first vector for current basis
                    R0=res0.copy(),  # circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                    F_function=self.F_function,
                    args=args,
                    step_size=step_size,
                    scaling_with_n=scaling_with_n,
                    target_relative_unexplained_residual=target_relative_unexplained_residual,
                    max_n_directions=max_n_directions,  # max number of basis vectors (must be less than number of modes + 1)
                    # max_Arnoldi_iterations=max_Arnoldi_iterations,
                    # max_collinearity=max_collinearity,
                    clip=clip,
                )
                # print(self.nksolver.coeffs)
                update = 1.0 * self.nksolver.dx

            del_update = np.amax(update) - np.amin(update)
            if del_update / del_psi > max_rel_update_size:
                # log.append("update > max_rel_update_size. Reduced.")
                update *= np.abs(max_rel_update_size * del_psi / del_update)

            new_residual_flag = True
            while new_residual_flag:
                # print('start new_residual_flag')
                try:
                    n_trial_plasma_psi = trial_plasma_psi + update
                    new_res0 = self.F_function(
                        n_trial_plasma_psi, self.tokamak_psi, profiles
                    )
                    # new_jmap = 1.0 * (profiles.jtor > 0)
                    # print('jmap_difference', np.sum((new_jmap-jmap)**2))
                    new_norm_rel_change = self.relative_norm_residual(
                        new_res0, n_trial_plasma_psi
                    )
                    new_rel_change, new_del_psi = self.relative_del_residual(
                        new_res0, n_trial_plasma_psi
                    )

                    new_residual_flag = False

                except:
                    log.append(
                        "Trigger update reduction due to failure to find an X-point, trying *0.75."
                    )
                    update *= 0.75

            if new_norm_rel_change < self.norm_rel_change[-1]:
                trial_plasma_psi = n_trial_plasma_psi.copy()
                try:
                    residual_collinearity = np.sum(res0 * new_res0) / (
                        np.linalg.norm(res0) * np.linalg.norm(new_res0)
                    )
                    res0 = 1.0 * new_res0
                    if (residual_collinearity > 0.9) and (picard_flag is False):
                        log.append(
                            "New starting_direction used due to collinear residuals."
                        )
                        # print('residual_collinearity', residual_collinearity)
                        # forcing_Picard = True
                        starting_direction = np.sin(
                            np.linspace(0, 2 * np.pi, self.nx)
                            * 1.5
                            * np.random.random()
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

                    else:
                        starting_direction = np.copy(res0)
                except:
                    starting_direction = np.copy(res0)
                rel_change = 1.0 * new_rel_change
                norm_rel_change = 1.0 * new_norm_rel_change
                del_psi = 1.0 * new_del_psi
            else:
                log.append("Increase in residual, update reduction triggered.")
                reduce_by = self.relative_change / rel_change
                new_residual_flag = True
                while new_residual_flag:
                    try:
                        n_trial_plasma_psi = trial_plasma_psi + update * reduce_by
                        res0 = self.F_function(
                            n_trial_plasma_psi, self.tokamak_psi, profiles
                        )
                        new_residual_flag = False
                    except:
                        reduce_by *= 0.75

                starting_direction = np.copy(res0)
                trial_plasma_psi = n_trial_plasma_psi.copy()
                norm_rel_change = self.relative_norm_residual(res0, trial_plasma_psi)
                rel_change, del_psi = self.relative_del_residual(res0, trial_plasma_psi)

            self.relative_change = 1.0 * rel_change
            self.norm_rel_change.append(norm_rel_change)
            # args[2] = 1.0*rel_change
            log.append("...relative error =  " + str(rel_change))

            if verbose:
                for x in log:
                    print(x)

            log = []

            iterations += 1

        # update eq with new solution
        eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()

        self.port_critical(eq=eq, profiles=profiles)

        # if picard_flag and verbose:
        #     print("Picard was used instead of NK in at least 1 cycle.")

        if iterations >= max_solving_iterations:
            warnings.warn(
                f"Forward solve failed to converge to requested relative tolerance of "
                + f"{target_relative_tolerance} with less than {max_solving_iterations} "
                + f"iterations. Last relative psi change: {rel_change}."
            )

    def get_currents(self, eq):
        # coils = list(eq.tokamak.getCurrents().keys())
        current_vec = np.zeros(self.len_control_coils)
        for i, coil in enumerate(self.control_coils):
            current_vec[i] = eq.tokamak[coil].current
        return current_vec

    def assign_currents(self, eq, current_vec):
        # coils = list(eq.tokamak.getCurrents().keys())
        for i, coil in enumerate(self.control_coils):
            eq.tokamak[coil].current = current_vec[i]

    def update_currents(self, constrain, eq, profiles):
        aux_tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)
        constrain(eq)
        self.tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)

        if hasattr(profiles, "limiter_core_mask"):
            norm_delta = np.linalg.norm(
                (self.tokamak_psi - aux_tokamak_psi)[profiles.limiter_core_mask]
            ) / np.linalg.norm(
                (self.tokamak_psi + aux_tokamak_psi)[profiles.limiter_core_mask]
            )
        else:
            norm_delta = 1
        # print("norm_delta", norm_delta)

        return norm_delta

    def inverse_solve(
        self,
        eq,
        profiles,
        target_relative_tolerance,
        constrain,
        verbose=False,
        max_solving_iterations=20,
        max_iter_per_update=5,
        Picard_handover=0.1,
        step_size=2.5,
        scaling_with_n=-1.0,
        max_n_directions=16,
        clip=10,
        max_rel_update_size=0.2,
        forward_tolerance_increase=5,
    ):
        """Inverse solver using the NK implementation.

        An inverse problem is specified by the 2 freegs4e objects eq and profiles,
        plus a constrain freeGS4e object.
        The first specifies the metal currents (throught eq.tokamak)
        The second specifies the desired plasma properties
        (i.e. plasma current and profile functions).
        The constrain object collects the desired magnetic constraints.

        The plasma_psi which solves the given GS problem is assigned to
        the input eq, and can be found at eq.plasma_psi.
        The coil currents with satisfy the magnetic constraints are
        assigned to eq.tokamak

        Parameters
        ----------
        eq : freegs4e equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : freegs4e profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        target_relative_tolerance : float
            NK iterations are interrupted when this criterion is
            satisfied. Relative convergence
        constrain : freegs4e constrain object
        verbose : bool
            flag to allow warning messages when Picard is used instead of NK
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
        forward_tolerance_increase : float
            after coil currents are updated, the interleaved forward problems
            are requested to converge to a tolerance that is tighter by a factor
            forward_tolerance_increase with respect to the change in flux caused
            by the current updates over the plasma core

        """

        log = []

        self.control_coils = list(eq.tokamak.getCurrents().keys())
        control_mask = np.arange(len(self.control_coils))[
            np.array([eq.tokamak[coil].control for coil in self.control_coils])
        ]
        self.control_coils = [self.control_coils[i] for i in control_mask]
        self.len_control_coils = len(self.control_coils)

        log.append("-----")
        log.append("Picard iteration: " + str(0))

        # use freegs4e Picard solver for initial steps to a shallow tolerance
        freegs4e.solve(
            eq,
            profiles,
            constrain,
            rtol=4e-2,
            show=False,
            blend=0.0,
        )

        iterations = 0
        rel_change_full = 1

        while (rel_change_full > target_relative_tolerance) * (
            iterations < max_solving_iterations
        ):
            
            log.append("-----")
            log.append("Newton-Krylov iteration: " + str(iterations+1))
                
            norm_delta = self.update_currents(constrain, eq, profiles)
            self.forward_solve(
                eq,
                profiles,
                target_relative_tolerance=norm_delta / forward_tolerance_increase,
                max_solving_iterations=max_iter_per_update,
                Picard_handover=Picard_handover,
                step_size=step_size,
                scaling_with_n=-scaling_with_n,
                max_n_directions=max_n_directions,
                clip=clip,
                verbose=False,
                max_rel_update_size=max_rel_update_size,
            )
            rel_change_full = 1.0 * self.relative_change
            iterations += 1
            log.append("...relative error =  " + str(rel_change_full))

            if verbose:
                for x in log:
                    print(x)

            log = []


        if iterations >= max_solving_iterations:
            warnings.warn(
                f"Inverse solve failed to converge to requested relative tolerance of "
                + f"{target_relative_tolerance} with less than {max_solving_iterations} "
                + f"iterations. Last relative psi change: {rel_change_full}. "
                + f"Last current change caused a relative update of tokamak_psi in the core of: {norm_delta}."
            )
            

    def solve(
        self,
        eq,
        profiles,
        target_relative_tolerance,
        constrain=None,
        max_solving_iterations=50,
        Picard_handover=0.1,
        blend=0.0,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.2,
        max_n_directions=16,
        clip=10,
        verbose=False,
        forward_tolerance_increase=5,
        picard=True,
    ):
        """The method to solve the GS problems, both forward and inverse.
            - an inverse solve is specified by the 'constrain' input,
            which includes the desired constraints on the configuration of magnetic flux (xpoints and isoflux, as in FreeGS4E).
            The optimization over the coil currents also uses the FreeGS4E implementation, as a simple regularised least square problem.
            - a forward solve has constrain=None. Please see forward_solve for details.


        Parameters
        ----------
        eq : freegs4e equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : freegs4e profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        target_relative_tolerance : float
            NK iterations are interrupted when this criterion is
            satisfied. Relative convergence
        constrain : freegs4e constrain object
            specifies the desired constraints on the configuration of magnetic flux (xpoints and isoflux, as in FreeGS4E)
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
        forward_tolerance_increase : float
            after coil currents are updated, the interleaved forward problems
            are requested to converge to a tolerance that is tighter by a factor
            forward_tolerance_increase with respect to the change in flux caused
            by the current updates over the plasma core
        verbose : bool
            flag to allow warning message in case of failed convergence within requested max_solving_iterations
        picard : bool
            flag to choose whether inverse solver uses Picard or Newton-Krylov iterations
        """

        # forward solve
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
                # max_Arnoldi_iterations,
                # max_collinearity,
                clip,
                # threshold,
                # clip_hard,
                verbose,
            )

        # inverse solve
        else:
            if picard == True: # uses picard iterations (from freegs4e)
                freegs4e.solve(
                    eq,
                    profiles,
                    constrain,
                    verbose,
                    rtol=target_relative_tolerance,
                    show=False,
                    blend=blend
                )
            else: # uses Newton-Krylov iterations from freegsnke
                self.inverse_solve(
                    eq,
                    profiles,
                    target_relative_tolerance,
                    constrain,
                    verbose
                )