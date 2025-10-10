"""
Applies the Newton Krylov solver Object to the static GS problem.
Implements both forward and inverse GS solvers.

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

from copy import deepcopy

import freegs4e
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
    input FreeGSNKE equilibrium object.

    The non-linear solvers are called using the 'forward_solve', 'inverse_solve' or generic 'solve' methods.
    """

    def __init__(
        self,
        eq,
        l2_reg=1e-6,
        collinearity_reg=1e-6,
    ):
        """Instantiates the solver object.
        Based on the domain grid of the input equilibrium object, it prepares
        - the linear solver 'self.linear_GS_solver'
        - the response matrix of boundary grid points 'self.greens_boundary'


        Parameters
        ----------
        eq : a FreeGSNKE equilibrium object.
             The domain grid defined by (eq.R, eq.Z) is the solution domain
             adopted for the GS problems. Calls to the nonlinear solver will
             use the grid domain set at instantiation time. Re-instantiation
             is necessary in order to change the propertes of either grid or
             domain.
        l2_reg : float
            Tychonoff regularization coeff used by the nonlinear solver
        collinearity_reg : float
            Tychonoff regularization coeff which further penalizes collinear terms used by the nonlinear solver

        """

        # eq is an Equilibrium instance, it has to have the same domain and grid as
        # the ones the solver will be called on

        self.eqR = eq.R
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

        self.nksolver = nk_solver.nksolver(
            problem_dimension=self.nx * self.ny,
            l2_reg=l2_reg,
            collinearity_reg=collinearity_reg,
        )

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

    def freeboundary(self, plasma_psi, tokamak_psi, profiles):
        """Imposes boundary conditions on set of boundary points.

        Parameters
        ----------
        plasma_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the plasma
        tokamak_psi : np.array of size eq.nx*eq.ny
            magnetic flux due to the tokamak alone, including all metal currents,
            in both active coils and passive structures
        profiles : FreeGSNKE profile object
            profile object describing target plasma properties.
            Used to calculate current density jtor
        """

        # tokamak_psi is the psi contribution due to the currents assigned to the tokamak coils in eq, ie.
        # tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)

        # jtor and RHS given tokamak_psi above and the input plasma_psi
        self.jtor = profiles.Jtor(
            self.R,
            self.Z,
            (tokamak_psi + plasma_psi).reshape(self.nx, self.ny),
        )
        self.rhs = self.rhs_before_jtor * self.jtor

        # calculates and imposes the boundary conditions
        self.psi_boundary = np.zeros_like(self.R)
        # weighted sum over the last two axes.
        # "contract" axis 1 of greenfunc with axis 0 of jtor
        # contract axis 2 of greenfunc with axis 1 of jtor
        psi_bnd = np.tensordot(self.greenfunc, self.jtor, axes=([1, 2], [0, 1]))

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

    def F_function(self, plasma_psi, tokamak_psi, profiles):
        """Residual of the nonlinear Grad Shafranov equation written as a root problem
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

        self.freeboundary(plasma_psi, tokamak_psi, profiles)
        residual = plasma_psi - (
            self.linear_GS_solver(self.psi_boundary, self.rhs)
        ).reshape(-1)
        return residual

    def port_critical(self, eq, profiles):
        """Transfers critical points and other useful info from profile to equilibrium object,
        after GS solution.

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Equilibrium on which to record values
        profiles : FreeGSNKE profile object
            Profiles object which has been used to calculate Jtor.
        """
        eq.solved = True

        eq.xpt = np.copy(profiles.xpt)
        eq.opt = np.copy(profiles.opt)
        eq.psi_axis = eq.opt[0, 2]

        eq.psi_bndry = profiles.psi_bndry
        eq.flag_limiter = profiles.flag_limiter

        eq._current = np.sum(profiles.jtor) * self.dRdZ
        eq._profiles = profiles.copy()

        try:
            eq.tokamak_psi = self.tokamak_psi.reshape(self.nx, self.ny)
        except:
            pass

    def relative_norm_residual(self, res, psi):
        """Calculates a normalised relative residual, based on linalg.norm

        Parameters
        ----------
        res : ndarray
            Residual
        psi : ndarray
            plasma_psi

        Returns
        -------
        float
            Relative normalised residual
        """
        return np.linalg.norm(res) / np.linalg.norm(psi)

    def relative_del_residual(self, res, psi):
        """Calculates a normalised relative residual, based on the norm max(.) - min(.)

        Parameters
        ----------
        res : ndarray
            Residual
        psi : ndarray
            plasma_psi

        Returns
        -------
        float, float
            Relative normalised residual, norm(plasma_psi)
        """
        del_psi = np.amax(psi) - np.amin(psi)
        del_res = np.amax(res) - np.amin(res)
        return del_res / del_psi, del_psi

    def forward_solve(
        self,
        eq,
        profiles,
        target_relative_tolerance,
        max_solving_iterations=100,
        Picard_handover=0.11,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.2,
        max_n_directions=16,
        max_rel_update_size=0.2,
        clip=10,
        # clip_quantiles=None,
        force_up_down_symmetric=False,
        verbose=False,
        suppress=False,
    ):
        """The method that actually solves the forward static GS problem.

        A forward problem is specified by the 2 FreeGSNKE objects eq and profiles.
        The first specifies the metal currents (throught eq.tokamak)
        and the second specifies the desired plasma properties
        (i.e. plasma current and profile functions).

        The plasma_psi which solves the given GS problem is assigned to
        the input eq, and can be found at eq.plasma_psi.

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : FreeGSNKE profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        target_relative_tolerance : float
            NK iterations are interrupted when this criterion is
            satisfied. Relative convergence for the residual F(plasma_psi)
        max_solving_iterations : int
            NK iterations are interrupted when this limit is surpassed
        Picard_handover : float
            Value of relative tolerance above which a Picard iteration
            is performed instead of a NK call
        step_size : float
            l2 norm of proposed step, in units of the size of the residual R0
        scaling_with_n : float
            allows to further scale the proposed steps as a function of the
            number of previous steps already attempted
            (1 + self.n_it)**scaling_with_n
        target_relative_unexplained_residual : float between 0 and 1
            terminates internal iterations when the considered directions
            can (linearly) explain such a fraction of the initial residual R0
        max_n_directions : int
            terminates iteration even though condition on
            explained residual is not met
        max_rel_update_size : float
            maximum relative update, in norm, to plasma_psi. If larger than this,
            the norm of the update is reduced
        clip : float
            maximum size of the update due to each explored direction, in units
            of exploratory step used to calculate the finite difference derivative
        force_up_down_symmetric : bool
            If True, the solver is forced on up-down symmetric trial solutions
        verbose : bool
            flag to allow progress printouts
        suppress : bool
            flag to allow suppress all printouts
        """

        if suppress:
            verbose = False

        picard_flag = 0
        if force_up_down_symmetric:
            trial_plasma_psi = 0.5 * (eq.plasma_psi + eq.plasma_psi[:, ::-1]).reshape(
                -1
            )
            self.shape = np.shape(eq.plasma_psi)
        else:
            trial_plasma_psi = np.copy(eq.plasma_psi).reshape(-1)
        # self.tokamak_psi = (eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)).reshape(-1)
        self.tokamak_psi = eq.tokamak.getPsitokamak(vgreen=eq._vgreen).reshape(-1)

        log = []
        log.append("-----")
        log.append("Forward static solve starting...")

        control_trial_psi = False
        n_up = 0.0 + 4 * eq.solved
        # this tries to cure cases where plasma_psi is not large enough in modulus
        # causing no core mask to exist
        while (control_trial_psi is False) and (n_up < 10):
            try:
                res0 = self.F_function(trial_plasma_psi, self.tokamak_psi, profiles)
                control_trial_psi = True
                log.append("Initial guess for plasma_psi successful, residual found.")

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

        self.best_relative_change = 1.0 * rel_change
        self.best_psi = trial_plasma_psi

        args = [self.tokamak_psi, profiles]

        starting_direction = np.copy(res0)

        log.append(f"Initial relative error = {rel_change:.2e}")
        if verbose:
            for x in log:
                print(x)

        self.initial_rel_residual = 1.0 * rel_change

        log = []
        log.append("-----")
        iterations = 0
        while (rel_change > target_relative_tolerance) * (
            iterations < max_solving_iterations
        ):
            if rel_change > Picard_handover:
                log.append("Picard iteration: " + str(iterations))
                # using Picard instead of NK

                if picard_flag < min(max_solving_iterations - 1, 3):
                    # make picard update to the flux up-down symmetric
                    # this combats the instability of picard iterations
                    res0_2d = res0.reshape(self.nx, self.ny)
                    res0 = 0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                    picard_flag += 1
                else:
                    # update = -1.0 * res0
                    picard_flag = 1

                # # test Picard update
                # nres0 = np.linalg.norm(res0)
                # successful = False
                # while successful == False:
                #     try:
                #         res1 = self.F_function(
                #             trial_plasma_psi - res0, self.tokamak_psi, profiles
                #         )
                #         nres1 = np.linalg.norm(res1)
                #         successful = True
                #     except:
                #         res0 *= .8

                # standard Picard update
                update = -1.0 * res0

                # if successful:
                #     if nres1 > 1.5 * nres0:
                #         vals = [-1, 0]
                #         res_vals = [nres1, nres0]
                #         counter_picard = 0
                #         while (
                #             res_vals[-1] < res_vals[-2]
                #             and successful
                #             and counter_picard < 10
                #         ):
                #             try:
                #                 new_res = self.F_function(
                #                     trial_plasma_psi + (vals[-1] + 1) * res0,
                #                     self.tokamak_psi,
                #                     profiles,
                #                 )
                #                 vals.append(vals[-1] + 1)
                #                 res_vals.append(np.linalg.norm(new_res))
                #                 successful = True
                #                 counter_picard += 1
                #             except:
                #                 successful = False
                #         if counter_picard < 10:
                #             # find best quadratic polyfit
                #             poly_coeffs = np.polyfit(vals, res_vals, deg=2)
                #             coeff_picard = (
                #                 0.5
                #                 * max(abs(poly_coeffs[1] / poly_coeffs[0]), 0.3)
                #                 * np.sign(poly_coeffs[1] / poly_coeffs[0])
                #             )
                #             update = -res0 * coeff_picard
                #             if verbose:
                #                 print(
                #                     "custom Picard accepted, with coeff",
                #                     coeff_picard,
                #                 )

            else:
                # using NK
                log.append("-----")
                log.append("Newton-Krylov iteration: " + str(iterations))
                picard_flag = False
                self.nksolver.Arnoldi_iteration(
                    x0=trial_plasma_psi.copy(),
                    dx=starting_direction.copy(),
                    R0=res0.copy(),
                    F_function=self.F_function,
                    args=args,
                    step_size=step_size,
                    scaling_with_n=scaling_with_n,
                    target_relative_unexplained_residual=target_relative_unexplained_residual,
                    max_n_directions=max_n_directions,
                    clip=clip,
                    # clip_quantiles=clip_quantiles,
                )
                update = 1.0 * self.nksolver.dx
                log.append(
                    f"...number of Krylov vectors used =  {len(self.nksolver.coeffs)}"
                )

            if force_up_down_symmetric:
                log.append("Forcing up-dpwn symmetry of the plasma.")
                update = update.reshape(self.shape)
                update = 0.5 * (update + update[:, ::-1]).reshape(-1)

            del_update = np.amax(update) - np.amin(update)
            if del_update / del_psi > max_rel_update_size:
                # Reduce the size of the update as found too large
                update *= np.abs(max_rel_update_size * del_psi / del_update)
                log.append("Update too large, resized.")

            new_residual_flag = True
            while new_residual_flag:
                try:
                    # check update does not cause the disappearance of the Opoint
                    n_trial_plasma_psi = trial_plasma_psi + update
                    new_res0 = self.F_function(
                        n_trial_plasma_psi, self.tokamak_psi, profiles
                    )
                    new_norm_rel_change = self.relative_norm_residual(
                        new_res0, n_trial_plasma_psi
                    )
                    new_rel_change, new_del_psi = self.relative_del_residual(
                        new_res0, n_trial_plasma_psi
                    )

                    new_residual_flag = False

                except:
                    log.append(
                        "Update resizing triggered due to failure to find a critical points."
                    )
                    update *= 0.75

            if new_norm_rel_change < 1.2 * self.norm_rel_change[-1]:
                # accept update
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
                        # Generate a random Krylov vector to continue the exploration
                        # This is arbitrary and can be improved
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
                reduce_by = self.relative_change / new_rel_change
                log.append("Increase in residual, update reduction triggered.")
                # log.append(reduce_by)
                new_residual_flag = True
                while new_residual_flag:
                    try:
                        n_trial_plasma_psi = trial_plasma_psi + update * reduce_by
                        res0 = self.F_function(
                            n_trial_plasma_psi, self.tokamak_psi, profiles
                        )
                        new_residual_flag = False
                    except:
                        log.append("reduction!")
                        reduce_by *= 0.75

                starting_direction = np.copy(res0)
                trial_plasma_psi = n_trial_plasma_psi.copy()
                norm_rel_change = self.relative_norm_residual(res0, trial_plasma_psi)
                rel_change, del_psi = self.relative_del_residual(res0, trial_plasma_psi)

                # compare to best on record
                if rel_change < self.best_relative_change:
                    self.best_relative_change = 1.0 * rel_change
                    self.best_psi = np.copy(trial_plasma_psi)

            self.relative_change = 1.0 * rel_change
            self.norm_rel_change.append(norm_rel_change)
            log.append(f"...relative error =  {rel_change:.2e}")
            log.append("-----")

            if verbose:
                for x in log:
                    print(x)

            log = []

            iterations += 1

        # update eq with new solution
        # compare to best on record
        if self.best_relative_change < rel_change:
            self.relative_change = 1.0 * self.best_relative_change
            trial_plasma_psi = np.copy(self.best_psi)
            profiles.Jtor(
                self.R,
                self.Z,
                (self.tokamak_psi + trial_plasma_psi).reshape(self.nx, self.ny),
            )
        eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()

        self.port_critical(eq=eq, profiles=profiles)

        if not suppress:
            if rel_change > target_relative_tolerance:
                print(
                    f"Forward static solve DID NOT CONVERGE. Tolerance {rel_change:.2e} (vs. requested {target_relative_tolerance:.2e}) reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
                )
            else:
                print(
                    f"Forward static solve SUCCESS. Tolerance {rel_change:.2e} (vs. requested {target_relative_tolerance:.2e}) reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
                )

    def get_rel_delta_psit(self, delta_current, profiles, vgreen):
        """Calculates the relative change to the tokamak_psi in the core region
        caused by the requested current change 'delta_current'.

        Parameters
        ----------
        delta_current : np.array
            Vector of requested current changes.
        profiles : freegsnke profile object
            Used to source the core mask
        vgreen : np.array
            The green functions of the relevant coils.
            For example eq._vgreen

        """
        if hasattr(profiles, "diverted_core_mask"):
            if profiles.diverted_core_mask is not None:
                core_mask = np.copy(profiles.diverted_core_mask)
            else:
                core_mask = np.ones_like(self.eqR)
        else:
            core_mask = np.ones_like(self.eqR)
        rel_delta_psit = np.linalg.norm(
            np.sum(
                delta_current[:, np.newaxis, np.newaxis]
                * vgreen
                * core_mask[np.newaxis],
                axis=0,
            )
        )
        rel_delta_psit /= np.linalg.norm(self.tokamak_psi) + 1e-6
        return rel_delta_psit

    def get_rel_delta_psi(self, new_psi, previous_psi, profiles):
        """Calculates the relative change between new_psi and previous_psi
        in the core region

        Parameters
        ----------
        new_psi : np.array
            Flattened flux function
        previous_psi : np.array
            _descrFlattened flux functioniption_
        profiles : freegsnke profile object
            Used to source the core mask

        """
        if hasattr(profiles, "diverted_core_mask"):
            if profiles.diverted_core_mask is not None:
                core_mask = np.copy(profiles.diverted_core_mask)
            else:
                core_mask = np.ones_like(self.eqR)
        else:
            core_mask = np.ones_like(self.eqR)
        core_mask = core_mask.reshape(-1)
        rel_delta_psit = np.linalg.norm((new_psi - previous_psi) * core_mask)
        rel_delta_psit /= np.linalg.norm((new_psi + previous_psi) * core_mask)
        return rel_delta_psit

    def optimize_currents(
        self,
        eq,
        profiles,
        constrain,
        target_relative_tolerance,
        relative_psit_size=1e-3,
        l2_reg=1e-12,
        verbose=False,
    ):
        """Calculates requested current changes for the coils available to control
        using the actual Jacobian rather than assuming the Jacobian is given by the
        green functions.

        Parameters
        ----------
        eq : freegsnke equilibrium object
            Used to source coil currents and plasma_psi
        profiles : freegsnke equilibrium object
            Provides the plasma properties
        constrain : freegnske inverse_optimizer object
            Provides information on the coils available for control
        target_relative_tolerance : float
            Target tolerance applied to GS when building the Jacobian
        relative_psit_size : float, optional
            Used to size the finite difference steps that define the Jacobian, by default 1e-3
        l2_reg : float, optional
            The Tichonov regularization factor applied to the least sq problem, by default 1e-12
        verbose : bool
            Print output
        """

        self.dbdI = np.zeros((np.size(constrain.b), constrain.n_control_coils))
        self.dummy_current = np.zeros(constrain.n_control_coils)

        full_current_vec = np.copy(eq.tokamak.current_vec)

        self.forward_solve(
            eq=eq,
            profiles=profiles,
            target_relative_tolerance=target_relative_tolerance,
            suppress=True,
        )
        delta_current, loss = constrain.optimize_currents(
            full_currents_vec=full_current_vec,
            trial_plasma_psi=eq.plasma_psi,
            l2_reg=1e-12,
        )
        b0 = np.copy(constrain.b)
        rel_delta_psit = self.get_rel_delta_psit(
            delta_current, profiles, eq._vgreen[constrain.control_mask]
        )
        adj_factor = min(1, relative_psit_size / rel_delta_psit)
        # print(delta_current, rel_delta_psit, adj_factor)
        delta_current *= adj_factor
        # delta_current_ = np.where(delta_current > 0.1, delta_current, 0.1*np.ones_like(delta_current))

        # print(delta_current)

        for i in range(constrain.n_control_coils):
            if verbose:
                print(
                    f" - calculating derivatives for coil {i + 1}/{constrain.n_control_coils}"
                )

            currents = np.copy(self.dummy_current)
            currents[i] = 1.0 * delta_current[i]
            currents = full_current_vec + constrain.rebuild_full_current_vec(currents)
            self.eq2 = eq.create_auxiliary_equilibrium()
            self.eq2.tokamak.set_all_coil_currents(currents)
            self.forward_solve(
                eq=self.eq2,
                profiles=profiles,
                target_relative_tolerance=target_relative_tolerance,
                suppress=True,
            )
            constrain.optimize_currents(
                full_currents_vec=currents,
                trial_plasma_psi=self.eq2.plasma_psi,
                l2_reg=1e-12,
            )
            self.dbdI[:, i] = (constrain.b - b0) / delta_current[i]

        if type(l2_reg) == float:
            reg_matrix = l2_reg * np.eye(constrain.n_control_coils)
        else:
            reg_matrix = np.diag(l2_reg)
        mat = np.linalg.inv(np.matmul(self.dbdI.T, self.dbdI) + reg_matrix)
        Newton_delta_current = np.dot(mat, np.dot(self.dbdI.T, -b0))
        loss = np.linalg.norm(b0 + np.dot(self.dbdI, Newton_delta_current))

        return Newton_delta_current, loss

    def inverse_solve(
        self,
        eq,
        profiles,
        constrain,
        target_relative_tolerance,
        target_relative_psit_update=1e-3,
        max_solving_iterations=100,
        max_iter_per_update=5,
        Picard_handover=0.15,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.3,
        max_n_directions=16,
        clip=10,
        # clip_quantiles=None,
        max_rel_update_size=0.15,
        threshold_val=0.18,
        l2_reg=1e-9,
        forward_tolerance_increase=100,
        # forward_tolerance_increase_factor=1.5,
        max_rel_psit=0.02,
        damping_factor=0.995,
        use_full_Jacobian=True,
        full_jacobian_handover=[1e-5, 7e-3],
        l2_reg_fj=1e-8,
        force_up_down_symmetric=False,
        verbose=False,
        suppress=False,
    ):
        """Inverse solver.

        Both coil currents and plasma flux are sought. Needs a set of desired magnetic constraints.

        An inverse problem is specified by the 2 FreeGSNKE objects, eq and profiles,
        plus a freegsnke "constrain" (or Inverse_optimizer) object.
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
        eq : FreeGSNKE equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : FreeGSNKE profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        constrain : freegsnke inverse_optimizer object
            Specifies the coils available for control and the desired magnetic constraints
        target_relative_tolerance : float
            The desired tolerance for the plasma flux.
            At fixed coil currents, this is the tolerance imposed to the GS problem.
            This has to be satisfied for the inverse problem to be considered solved.
        target_relative_psit_update : float
            The relative update to psi_tokamak caused by the update in the control currents
            has to be lower than this target value for the inverse problem to be considered
            successfully solved.
        max_solving_iterations : int
            Maximum number of solving iterations. The solver is interuupted when this is met.
        max_iter_per_update : int
            Maximum number of interations allowed to the forward solver in each cycle of the inverse solver.
        Picard_handover : float
            Value of relative tolerance above which a Picard iteration
            is performed instead of a NK call in the forward solve
        step_size : float
            l2 norm of proposed step, in units of the size of the residual R0
        scaling_with_n : float
            allows to further scale the proposed steps as a function of the
            number of previous steps already attempted
            (1 + self.n_it)**scaling_with_n
        target_relative_unexplained_residual : float between 0 and 1
            terminates internal iterations when the considered directions
            can (linearly) explain such a fraction of the initial residual R0
        max_n_directions : int
            terminates iteration even though condition on
            explained residual is not met
        max_rel_update_size : float
            maximum relative update, in norm, to plasma_psi. If larger than this,
            the norm of the update is reduced
        clip : float
            maximum size of the update due to each explored direction, in units
            of exploratory step used to calculate the finite difference derivative
        l2_reg : either float or 1d np.array with len=self.n_control_coils
            The regularization factor applied when green functions are used as Jacobians
        forward_tolerance_increase : float
            after coil currents are updated, the interleaved forward problems
            are requested to converge to a tolerance that is tighter by a factor
            forward_tolerance_increase with respect to the change in flux caused
            by the current updates over the plasma core
        forward_tolerance_increase_factor : float
            iterations that do not result in improvement trigger an increase in the tolerance
            applied to the forward problems. The increase corresponds to a factor forward_tolerance_increase_factor
        max_rel_psit : float
            The maximum relative change that the requested updates in the control currents are allowed to cause.
        damping_factor : float
            This applies a damping of damping_factor**number_of_iterations to max_rel_psit,
            to encourage convergence
        full_jacobian_handover : float
            When the forward problems achieve this tolerance level,
            self.optimize_currents is called instead of constrain.optimize_currents.
            This means that the actual Jacobians are used rather than the green functions.
        l2_reg_fj : float
            The regularization factor applied when the full Jacobians are used.
        verbose : bool
            flag to allow progress printouts
        suppress : bool
            flag to allow suppress all printouts
        """

        if suppress:
            verbose = False

        if verbose:
            print("-----")
            print("Inverse static solve starting...")

        iterations = 0
        damping = 1
        self.rel_psit_updates = [max_rel_psit]
        previous_rel_delta_psit = 1
        self.constrain_loss = []
        check_core_mask = False

        # If not calling the solver from self.solve
        # then you need to ensure that the vectorised currents are in place in tokamak object!
        # Make sure any currents that are not controlled are assigned using eq.tokamak.set_coil_current()
        # Otherwise the following tokamak_psi will not be the correct one!
        full_currents_vec = np.copy(eq.tokamak.current_vec)
        self.tokamak_psi = eq.tokamak.getPsitokamak(vgreen=eq._vgreen)

        # prepare all green functions needed by the current optimizations
        constrain.prepare_for_solve(eq)

        check_equilibrium = False
        try:
            GS_residual = self.F_function(
                tokamak_psi=self.tokamak_psi.reshape(-1),
                plasma_psi=eq.plasma_psi.reshape(-1),
                profiles=profiles,
            )
            if verbose:
                print("Initial guess for plasma_psi successful, residual found.")
            rel_change_full, del_psi = self.relative_del_residual(
                GS_residual, eq.plasma_psi
            )
            if rel_change_full < threshold_val:
                check_equilibrium = True
            if profiles.diverted_core_mask is not None:
                check_core_mask = True
        except:
            pass

        if verbose:
            print(f"Initial relative error = {rel_change_full:.2e}")
            print("-----")

        while (
            (rel_change_full > target_relative_tolerance)
            + (previous_rel_delta_psit > target_relative_psit_update)
        ) * (iterations < max_solving_iterations):
            if verbose:
                print("Iteration: " + str(iterations))

            if check_equilibrium:
                # this_max_rel_psit = min(max_rel_psit, np.mean(self.rel_psit_updates[-6:]))
                this_max_rel_psit = np.mean(self.rel_psit_updates[-6:])
                this_max_rel_update_size = 1.0 * max_rel_update_size
                if type(l2_reg) == float:
                    this_l2_reg = 1.0 * l2_reg
                else:
                    this_l2_reg = np.array(l2_reg)
                if (previous_rel_delta_psit < target_relative_psit_update) * (
                    rel_change_full < 50 * target_relative_tolerance
                ):
                    # use more iterations if 'close to solution'
                    this_max_iter_per_update = 50
                else:
                    this_max_iter_per_update = 1.0 * max_iter_per_update
            else:
                this_max_rel_psit = False
                this_max_rel_update_size = max(max_rel_update_size, 0.3)
                this_max_iter_per_update = 1
                if type(l2_reg) == float:
                    this_l2_reg = 1e-4 * l2_reg
                else:
                    this_l2_reg = 1e-4 * np.array(l2_reg)

            if (
                use_full_Jacobian
                * (rel_change_full < full_jacobian_handover[0])
                * (previous_rel_delta_psit < full_jacobian_handover[1])
            ):
                if verbose:
                    print(
                        "Using full Jacobian (of constraints wrt coil currents) to optimsise currents."
                    )

                # use complete Jacobian: psi_plasma changes with the coil currents
                delta_current, loss = self.optimize_currents(
                    eq=eq,
                    profiles=profiles,
                    constrain=constrain,
                    target_relative_tolerance=target_relative_tolerance,
                    relative_psit_size=this_max_rel_psit,
                    l2_reg=l2_reg_fj,
                    verbose=verbose,
                )
            else:
                if verbose:
                    print(
                        "Using simplified Green's Jacobian (of constraints wrt coil currents) to optimise the currents."
                    )
                # use Greens as Jacobian: i.e. psi_plasma is assumed fixed
                delta_current, loss = constrain.optimize_currents(
                    full_currents_vec=full_currents_vec,
                    trial_plasma_psi=eq.plasma_psi,
                    l2_reg=this_l2_reg,
                )
            self.constrain_loss.append(loss)

            rel_delta_psit = self.get_rel_delta_psit(
                delta_current, profiles, eq._vgreen[constrain.control_mask]
            )
            # print("requested rel_delta_psit", rel_delta_psit)
            if this_max_rel_psit:
                # resize update to the control currents so to limit the relative change of the tokamak flux to this_max_rel_psit
                if constrain.curr_loss < 1:
                    damping *= damping_factor
                adj_factor = damping * min(1, this_max_rel_psit / rel_delta_psit)
                # apply the resizing
            else:
                adj_factor = 1.0
            delta_current *= adj_factor
            previous_rel_delta_psit = rel_delta_psit * adj_factor

            if check_core_mask:
                # make sure that the update of the control currents does not cause a loss of the Opoint or of the Xpoints
                delta_tokamak_psi = np.sum(
                    delta_current[:, np.newaxis, np.newaxis]
                    * eq._vgreen[constrain.control_mask],
                    axis=0,
                ).reshape(-1)

                resize = True
                while resize:
                    try:
                        GS_residual = self.F_function(
                            tokamak_psi=self.tokamak_psi.reshape(-1)
                            + delta_tokamak_psi,
                            plasma_psi=eq.plasma_psi.reshape(-1),
                            profiles=profiles,
                        )
                        if len(profiles.xpt):
                            # The update is approved:
                            resize = False
                    except:
                        pass

                    if resize:
                        if verbose:
                            print("Resizing of the control current update triggered!")
                        delta_current *= 0.75
                        delta_tokamak_psi *= 0.75
                        previous_rel_delta_psit *= 0.75

            self.rel_psit_updates.append(previous_rel_delta_psit)

            # apply the update to the control currents
            full_currents_vec += constrain.rebuild_full_current_vec(delta_current)
            eq.tokamak.set_all_coil_currents(full_currents_vec)
            if verbose:
                print(
                    f"Change in coil currents (being controlled): {[f'{val:.2e}' for val in delta_current]}"
                )
                print(f"Constraint losses = {loss:.2e}")
                print(
                    f"Relative update of tokamak psi (in plasma core): {previous_rel_delta_psit:.2e}"
                )

            # if loss > self.constrain_loss[-1]:
            #     forward_tolerance_increase *= forward_tolerance_increase_factor

            # set tolerance for the upcoming forward solve
            if previous_rel_delta_psit < target_relative_psit_update:
                tolerance = 1.0 * target_relative_tolerance
                this_max_rel_update_size = 50
            else:
                tolerance = max(
                    min(previous_rel_delta_psit / forward_tolerance_increase, 1e-3),
                    target_relative_tolerance,
                )

            # forward solve to update the plasma_psi based on the new currents
            if verbose:
                print(f"Handing off to forward solve (with updated currents).")

            self.forward_solve(
                eq,
                profiles,
                target_relative_tolerance=tolerance,
                max_solving_iterations=this_max_iter_per_update,
                Picard_handover=Picard_handover,
                step_size=step_size,
                scaling_with_n=scaling_with_n,
                target_relative_unexplained_residual=target_relative_unexplained_residual,
                max_n_directions=max_n_directions,
                clip=clip,
                verbose=False,
                max_rel_update_size=this_max_rel_update_size,
                force_up_down_symmetric=force_up_down_symmetric,
                suppress=True,
            )
            rel_change_full = 1.0 * self.relative_change

            iterations += 1

            if verbose:
                print(f"Relative error =  {rel_change_full:.2e}")
                print("-----")

            if rel_change_full < threshold_val:
                check_equilibrium = True
            else:
                check_equilibrium = False

        if not suppress:
            if rel_change_full > target_relative_tolerance:
                print(
                    f"Inverse static solve DID NOT CONVERGE. Tolerance {rel_change_full:.2e} (vs. requested {target_relative_tolerance}) reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
                )
            else:
                print(
                    f"Inverse static solve SUCCESS. Tolerance {rel_change_full:.2e} (vs. requested {target_relative_tolerance}) reached in {int(iterations)}/{int(max_solving_iterations)} iterations."
                )

    def solve(
        self,
        eq,
        profiles,
        constrain=None,
        target_relative_tolerance=1e-5,
        target_relative_psit_update=1e-3,
        max_solving_iterations=100,
        max_iter_per_update=5,
        Picard_handover=0.1,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.3,
        max_n_directions=16,
        clip=10,
        # clip_quantiles=None,
        max_rel_update_size=0.15,
        l2_reg=1e-9,
        forward_tolerance_increase=100,
        # forward_tolerance_increase_factor=1.5,
        max_rel_psit=0.01,
        damping_factor=0.98,
        use_full_Jacobian=True,
        full_jacobian_handover=[1e-5, 7e-3],
        l2_reg_fj=1e-8,
        force_up_down_symmetric=False,
        verbose=False,
        suppress=False,
    ):
        """The method to solve the GS problems, both forward and inverse.
            - an inverse solve is specified by the 'constrain' input,
            which includes the desired constraints on the configuration of magnetic flux.
            Please see inverse_solve for details.
            - a forward solve has constrain=None. Please see forward_solve for details.


        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : FreeGSNKE profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        constrain : freegsnke inverse_optimizer object
            Specifies the coils available for control and the desired magnetic constraints
        target_relative_tolerance : float
            The desired tolerance for the plasma flux.
            At fixed coil currents, this is the tolerance imposed to the GS problem.
            This has to be satisfied for the inverse problem to be considered solved.
        target_relative_psit_update : float
            The relative update to psi_tokamak caused by the update in the control currents
            has to be lower than this target value for the inverse problem to be considered
            successfully solved.
        max_solving_iterations : int
            Maximum number of solving iterations. The solver is interuupted when this is met.
        max_iter_per_update : int
            Maximum number of interations allowed to the forward solver in each cycle of the inverse solver.
        Picard_handover : float
            Value of relative tolerance above which a Picard iteration
            is performed instead of a NK call in the forward solve
        step_size : float
            l2 norm of proposed step, in units of the size of the residual R0
        scaling_with_n : float
            allows to further scale the proposed steps as a function of the
            number of previous steps already attempted
            (1 + self.n_it)**scaling_with_n
        target_relative_unexplained_residual : float between 0 and 1
            terminates internal iterations when the considered directions
            can (linearly) explain such a fraction of the initial residual R0
        max_n_directions : int
            terminates iteration even though condition on
            explained residual is not met
        max_rel_update_size : float
            maximum relative update, in norm, to plasma_psi. If larger than this,
            the norm of the update is reduced
        clip : float
            maximum size of the update due to each explored direction, in units
            of exploratory step used to calculate the finite difference derivative
        l2_reg : either float or 1d np.array with len=self.n_control_coils
            The regularization factor applied when green functions are used as Jacobians
        forward_tolerance_increase : float
            after coil currents are updated, the interleaved forward problems
            are requested to converge to a tolerance that is tighter by a factor
            forward_tolerance_increase with respect to the change in flux caused
            by the current updates over the plasma core
        forward_tolerance_increase_factor : float
            iterations that do not result in improvement trigger an increase in the tolerance
            applied to the forward problems. The increase corresponds to a factor forward_tolerance_increase_factor
        max_rel_psit : float
            The maximum relative change that the requested updates in the control currents are allowed to cause.
        damping_factor : float
            This applies a damping of damping_factor**number_of_iterations to max_rel_psit,
            to encourage convergence
        full_jacobian_handover : float
            When the forward problems achieve this tolerance level,
            self.optimize_currents is called instead of constrain.optimize_currents.
            This means that the actual Jacobians are used rather than the green functions.
        l2_reg_fj : float
            The regularization factor applied when the full Jacobians are used.
        verbose : bool
            flag to allow progress printouts
        suppress : bool
            flag to allow suppress all printouts
        """

        # ensure vectorised currents are in place in tokamak object
        eq.tokamak.getCurrentsVec()

        # forward solve
        eq._separatrix_data_flag = False
        if constrain is None:
            self.forward_solve(
                eq=eq,
                profiles=profiles,
                target_relative_tolerance=target_relative_tolerance,
                max_solving_iterations=max_solving_iterations,
                Picard_handover=Picard_handover,
                step_size=step_size,
                scaling_with_n=scaling_with_n,
                target_relative_unexplained_residual=target_relative_unexplained_residual,
                max_n_directions=max_n_directions,
                clip=clip,
                verbose=verbose,
                max_rel_update_size=max_rel_update_size,
                force_up_down_symmetric=force_up_down_symmetric,
                suppress=suppress,
            )

        else:
            self.inverse_solve(
                eq=eq,
                profiles=profiles,
                constrain=constrain,
                target_relative_tolerance=target_relative_tolerance,
                target_relative_psit_update=target_relative_psit_update,
                max_solving_iterations=max_solving_iterations,
                max_iter_per_update=max_iter_per_update,
                Picard_handover=Picard_handover,
                step_size=step_size,
                scaling_with_n=scaling_with_n,
                target_relative_unexplained_residual=target_relative_unexplained_residual,
                max_n_directions=max_n_directions,
                clip=clip,
                max_rel_update_size=max_rel_update_size,
                l2_reg=l2_reg,
                forward_tolerance_increase=forward_tolerance_increase,
                # forward_tolerance_increase_factor=forward_tolerance_increase_factor,
                max_rel_psit=max_rel_psit,
                damping_factor=damping_factor,
                full_jacobian_handover=full_jacobian_handover,
                use_full_Jacobian=use_full_Jacobian,
                l2_reg_fj=l2_reg_fj,
                force_up_down_symmetric=force_up_down_symmetric,
                verbose=verbose,
                suppress=suppress,
            )


# def old_inverse_solve(
#         self,
#         eq,
#         profiles,
#         target_relative_tolerance,
#         constrain,
#         verbose=False,
#         max_solving_iterations=20,
#         max_iter_per_update=5,
#         Picard_handover=0.1,
#         initial_Picard=True,
#         step_size=2.5,
#         scaling_with_n=-1.0,
#         target_relative_unexplained_residual=0.3,
#         max_n_directions=16,
#         clip=10,
#         clip_quantiles=None,
#         max_rel_update_size=0.2,
#         forward_tolerance_increase=5,
#     ):
#         """Inverse solver using the NK implementation.

#         An inverse problem is specified by the 2 FreeGSNKE objects, eq and profiles,
#         plus a constrain freeGS4e object.
#         The first specifies the metal currents (throught eq.tokamak)
#         The second specifies the desired plasma properties
#         (i.e. plasma current and profile functions).
#         The constrain object collects the desired magnetic constraints.

#         The plasma_psi which solves the given GS problem is assigned to
#         the input eq, and can be found at eq.plasma_psi.
#         The coil currents with satisfy the magnetic constraints are
#         assigned to eq.tokamak

#         Parameters
#         ----------
#         eq : FreeGSNKE equilibrium object
#             Used to extract the assigned metal currents, which in turn are
#             used to calculate the according self.tokamak_psi
#         profiles : FreeGSNKE profile object
#             Specifies the target properties of the plasma.
#             These are used to calculate Jtor(psi)
#         target_relative_tolerance : float
#             NK iterations are interrupted when this criterion is
#             satisfied. Relative convergence for the residual F(plasma_psi)
#         constrain : freegs4e constrain object
#         verbose : bool
#             flag to allow progress printouts
#         max_solving_iterations : int
#             NK iterations are interrupted when this limit is surpassed
#         Picard_handover : float
#             Value of relative tolerance above which a Picard iteration
#             is performed instead of a NK call
#         step_size : float
#             l2 norm of proposed step, in units of the size of the residual R0
#         scaling_with_n : float
#             allows to further scale the proposed steps as a function of the
#             number of previous steps already attempted
#             (1 + self.n_it)**scaling_with_n
#         target_relative_explained_residual : float between 0 and 1
#             terminates internal iterations when the considered directions
#             can (linearly) explain such a fraction of the initial residual R0
#         max_n_directions : int
#             terminates iteration even though condition on
#             explained residual is not met
#         max_rel_update_size : float
#             maximum relative update, in norm, to plasma_psi. If larger than this,
#             the norm of the update is reduced
#         clip : float
#             maximum size of the update due to each explored direction, in units
#             of exploratory step used to calculate the finite difference derivative
#         forward_tolerance_increase : float
#             after coil currents are updated, the interleaved forward problems
#             are requested to converge to a tolerance that is tighter by a factor
#             forward_tolerance_increase with respect to the change in flux caused
#             by the current updates over the plasma core

#         """

#         log = []

#         # self.control_coils = list(eq.tokamak.getCurrents().keys())
#         # control_mask = np.arange(len(self.control_coils))[
#         #     np.array([eq.tokamak[coil].control for coil in self.control_coils])
#         # ]
#         # self.control_coils = [self.control_coils[i] for i in control_mask]
#         # self.len_control_coils = len(self.control_coils)

#         if initial_Picard:
#             # use freegs4e Picard solver for initial steps to a shallow tolerance
#             freegs4e.solve(
#                 eq,
#                 profiles,
#                 constrain,
#                 rtol=4e-2,
#                 show=False,
#                 blend=0.0,
#             )

#         iterations = 0
#         rel_change_full = 1

#         while (rel_change_full > target_relative_tolerance) * (
#             iterations < max_solving_iterations
#         ):

#             log.append("-----")
#             log.append("Newton-Krylov iteration: " + str(iterations))

#             norm_delta = self.update_currents(constrain, eq, profiles)

#             self.forward_solve(
#                 eq,
#                 profiles,
#                 target_relative_tolerance=norm_delta / forward_tolerance_increase,
#                 max_solving_iterations=max_iter_per_update,
#                 Picard_handover=Picard_handover,
#                 step_size=step_size,
#                 scaling_with_n=-scaling_with_n,
#                 target_relative_unexplained_residual=target_relative_unexplained_residual,
#                 max_n_directions=max_n_directions,
#                 clip=clip,
#                 clip_quantiles=clip_quantiles,
#                 verbose=False,
#                 max_rel_update_size=max_rel_update_size,
#             )
#             rel_change_full = 1.0 * self.relative_change
#             iterations += 1
#             log.append("...relative error =  " + str(rel_change_full))

#             if verbose:
#                 for x in log:
#                     print(x)

#             log = []

#         if iterations >= max_solving_iterations:
#             warnings.warn(
#                 f"Inverse solve failed to converge to requested relative tolerance of "
#                 + f"{target_relative_tolerance} with less than {max_solving_iterations} "
#                 + f"iterations. Last relative psi change: {rel_change_full}. "
#                 + f"Last current change caused a relative update of tokamak_psi in the core of: {norm_delta}."
#             )
