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
    input FreeGSNKE equilibrium object.

    The non-linear solvers are called using the 'forward_solve', 'inverse_solve' or generic 'solve' methods.
    """

    def __init__(self, eq):
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
        eq._profiles = deepcopy(profiles)

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
        target_relative_explained_residual : float between 0 and 1
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
        verbose : bool
            flag to allow progress printouts
        """

        picard_flag = 0
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

        args = [self.tokamak_psi, profiles]

        starting_direction = np.copy(res0)

        if verbose:
            for x in log:
                print(x)

        log = []
        iterations = 0
        while (rel_change > target_relative_tolerance) * (
            iterations < max_solving_iterations
        ):

            if rel_change > Picard_handover:
                log.append("-----")
                log.append("Picard iteration: " + str(iterations))
                # using Picard instead of NK

                if picard_flag < 3:
                    # make picard update to the flux up-down symmetric
                    # this combats the instability of picard iterations
                    res0_2d = res0.reshape(self.nx, self.ny)
                    update = -0.5 * (res0_2d + res0_2d[:, ::-1]).reshape(-1)
                    picard_flag += 1
                else:
                    update = -1.0 * res0
                    picard_flag = 1

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
                )
                update = 1.0 * self.nksolver.dx

            del_update = np.amax(update) - np.amin(update)
            if del_update / del_psi > max_rel_update_size:
                # Reduce the size of the update as found too large
                update *= np.abs(max_rel_update_size * del_psi / del_update)

            new_residual_flag = True
            while new_residual_flag:
                try:
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
            log.append("...relative error =  " + str(rel_change))

            if verbose:
                for x in log:
                    print(x)

            log = []

            iterations += 1

        # update eq with new solution
        eq.plasma_psi = trial_plasma_psi.reshape(self.nx, self.ny).copy()

        self.port_critical(eq=eq, profiles=profiles)

        if iterations >= max_solving_iterations:
            warnings.warn(
                f"Forward solve failed to converge to requested relative tolerance of "
                + f"{target_relative_tolerance} with less than {max_solving_iterations} "
                + f"iterations. Last relative psi change: {rel_change}."
            )

    def get_currents(self, eq):
        current_vec = np.zeros(self.len_control_coils)
        for i, coil in enumerate(self.control_coils):
            current_vec[i] = eq.tokamak[coil].current
        return current_vec

    def assign_currents(self, eq, current_vec):
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
        initial_Picard=True,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.3,
        max_n_directions=16,
        clip=10,
        max_rel_update_size=0.2,
        forward_tolerance_increase=5,
    ):
        """Inverse solver using the NK implementation.

        An inverse problem is specified by the 2 FreeGSNKE objects, eq and profiles,
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
        eq : FreeGSNKE equilibrium object
            Used to extract the assigned metal currents, which in turn are
            used to calculate the according self.tokamak_psi
        profiles : FreeGSNKE profile object
            Specifies the target properties of the plasma.
            These are used to calculate Jtor(psi)
        target_relative_tolerance : float
            NK iterations are interrupted when this criterion is
            satisfied. Relative convergence for the residual F(plasma_psi)
        constrain : freegs4e constrain object
        verbose : bool
            flag to allow progress printouts
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
        target_relative_explained_residual : float between 0 and 1
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

        if initial_Picard:
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
            log.append("Newton-Krylov iteration: " + str(iterations))

            norm_delta = self.update_currents(constrain, eq, profiles)
            self.forward_solve(
                eq,
                profiles,
                target_relative_tolerance=norm_delta / forward_tolerance_increase,
                max_solving_iterations=max_iter_per_update,
                Picard_handover=Picard_handover,
                step_size=step_size,
                scaling_with_n=-scaling_with_n,
                target_relative_unexplained_residual=target_relative_unexplained_residual,
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
        max_iter_per_update=5,
        Picard_handover=0.1,
        step_size=2.5,
        scaling_with_n=-1.0,
        target_relative_unexplained_residual=0.3,
        max_n_directions=16,
        clip=10,
        verbose=False,
        max_rel_update_size=0.2,
        forward_tolerance_increase=5,
        blend=0,
        picard=True,
    ):
        """The method to solve the GS problems, both forward and inverse.
            - an inverse solve is specified by the 'constrain' input,
            which includes the desired constraints on the configuration of magnetic flux (xpoints and isoflux, as in FreeGS4E).
            The optimization over the coil currents also uses the FreeGS4E implementation, as a simple regularised least square problem.
            - a forward solve has constrain=None. Please see forward_solve for details.


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
        constrain : freegs4e constrain object
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
        target_relative_explained_residual : float between 0 and 1
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
        forward_tolerance_increase : float
            after coil currents are updated, the interleaved forward problems
            are requested to converge to a tolerance that is tighter by a factor
            forward_tolerance_increase with respect to the change in flux caused
            by the current updates over the plasma core
        verbose : bool
            flag to allow progress printouts
        picard : bool
            flag to choose whether inverse solver uses Picard or Newton-Krylov iterations
        """

        # forward solve
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
            )

        else:
            if picard == True:  # uses picard iterations (from freegs4e)
                freegs4e.solve(
                    eq=eq,
                    profiles=profiles,
                    constrain=constrain,
                    rtol=target_relative_tolerance,
                    show=False,
                    blend=blend,
                )
                self.port_critical(eq=eq, profiles=profiles)

            else:  # uses Newton-Krylov iterations from freegsnke
                self.inverse_solve(
                    eq=eq,
                    profiles=profiles,
                    target_relative_tolerance=target_relative_tolerance,
                    constrain=constrain,
                    max_solving_iterations=max_solving_iterations,
                    max_iter_per_update=max_iter_per_update,
                    Picard_handover=Picard_handover,
                    step_size=step_size,
                    scaling_with_n=scaling_with_n,
                    target_relative_unexplained_residual=target_relative_unexplained_residual,
                    max_n_directions=max_n_directions,
                    clip=clip,
                    verbose=verbose,
                    max_rel_update_size=max_rel_update_size,
                    forward_tolerance_increase=forward_tolerance_increase,
                )
