"""
Defines class that represents the virtual circuits. 

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

from __future__ import annotations

from datetime import datetime
from typing import Callable

import numpy as np

from .jtor_update import fit_lao85_betap_li_ip


class VirtualCircuit:
    """
    The class for storing/recording virtual circuits that have been built
    using the VirtualCircuitHandling class.
    """

    def __init__(
        self,
        name: str,
        eq: object,
        profiles: object,
        shape_matrix: np.ndarray,
        VCs_matrix: np.ndarray,
        target_names: list[str],
        coils: list[str],
        target_calculator: Callable[[object], np.ndarray],
        profile_adjuster: Callable | None = None,
    ) -> None:
        """
        Store the key quantities from the VirtualCircuitHandling calculations.

        Parameters
        ----------
        name : str
            Name to call the VCs (e.g. super-X VCs).
        eq : object
            The equilibrium object used to build the VCs.
        profiles : object
            The profiles object used to build the VCs.
        shape_matrix : np.array
            The array storing the Jacobians between the targets and coils given in 'targets'
            and 'coils'.
        VCs_matrix : np.array
            The array storing the VCs between the targets and coils given in 'targets'
            and 'coils'.
        target_names : list
            A list of the target names, e.g [Rin, Rout, Rx, Zx, ...] (must be same length as array from target_calculator).
        coils : list
            The list of coils used to calculate the shape_matrix and VCs_matrix.
        target_calculator : function
            Function returning an array of the shape targets (VC will be calculated for ALL of these targets).
        profile_adjuster : callable, optional
            Callable used while building or applying the VC to update the profile
            object after coil currents have been changed. This is useful for VCs
            at fixed profile-derived quantities, such as fixed beta_p and li.
        """

        self.name = name
        self.eq = eq
        self.profiles = profiles
        self.shape_matrix = shape_matrix
        self.VCs_matrix = VCs_matrix
        self.target_names = target_names
        self.coils = coils
        self.target_calculator = target_calculator
        self.profile_adjuster = profile_adjuster


def make_lao85_betap_li_profile_adjuster(
    reference_eq,
    betap=None,
    li=None,
    li_method="internalInductance2",
    alpha=None,
    beta=None,
    target_relative_tolerance=None,
    max_solving_iterations=100,
    optimizer_kwargs=None,
    use_metric_jacobian=True,
    reuse_metric_jacobian=True,
    jacobian_rel_step=1e-4,
    regularization_weight=1e-6,
):
    """
    Build a profile adjuster for virtual circuits at fixed beta_p and li.

    The returned callable is suitable for the ``profile_adjuster`` argument of
    :meth:`VirtualCircuitHandling.calculate_VC` and
    :meth:`VirtualCircuitHandling.apply_VC`. After each coil-current
    perturbation, it refits the supplied Lao85 profile coefficients so that the
    new static equilibrium preserves:

    - ``poloidalBeta1()``, matching the ``ConstrainBetapIp`` beta_p definition;
    - ``getattr(eq, li_method)()``, defaulting to ``internalInductance2``.

    Parameters
    ----------
    reference_eq : object
        Solved equilibrium defining the fixed targets. If ``betap`` or ``li``
        are not supplied, they are read from this equilibrium.
    betap, li : float, optional
        Explicit target values. Defaults to ``reference_eq.poloidalBeta1()`` and
        ``getattr(reference_eq, li_method)()``.
    li_method : str, optional
        Equilibrium method used to evaluate internal inductance.
    alpha, beta : array-like of length 2, optional
        Initial Lao coefficients used by the fitter. If omitted, the first two
        coefficients from the current profile object are used at each call.
    target_relative_tolerance : float, optional
        Static solve tolerance used inside the fitter. If omitted, the VC
        builder's tolerance is used.
    max_solving_iterations : int, optional
        Maximum static solve iterations for each fitted trial profile.
    optimizer_kwargs : dict, optional
        Additional keyword arguments passed to ``scipy.optimize.least_squares``.
    use_metric_jacobian : bool, optional
        Whether to build an explicit finite-difference Jacobian of the solved
        beta_p and li metrics with respect to the Lao coefficients.
    reuse_metric_jacobian : bool, optional
        Whether to reuse the first calculated metric Jacobian for subsequent VC
        perturbation solves. This is usually appropriate for local VC
        construction, where all perturbations stay close to the reference
        equilibrium. Set false to rebuild the Jacobian for each profile fit.
    jacobian_rel_step : float, optional
        Relative perturbation used by the explicit metric Jacobian.
    regularization_weight : float, optional
        Coefficient regularisation weight passed to ``fit_lao85_betap_li_ip``.

    Returns
    -------
    callable
        Callback returning ``(profiles, True)`` because it solves the static GS
        problem as part of the fit.
    """

    target_betap = reference_eq.poloidalBeta1() if betap is None else betap
    target_li = getattr(reference_eq, li_method)() if li is None else li
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    fit_results = []
    reusable_metric_jacobian = [None]

    def adjuster(eq, profiles, solver, vc_target_relative_tolerance):
        fit_alpha = _first_two_lao_coefficients(profiles.alpha, alpha, "alpha")
        fit_beta = _first_two_lao_coefficients(profiles.beta, beta, "beta")
        fit_tolerance = (
            vc_target_relative_tolerance
            if target_relative_tolerance is None
            else target_relative_tolerance
        )

        fitted_profiles, fit_result = fit_lao85_betap_li_ip(
            eq,
            solver,
            Ip=profiles.Ip,
            fvac=profiles.fvac(),
            betap=target_betap,
            li=target_li,
            alpha=fit_alpha,
            beta=fit_beta,
            alpha_logic=profiles.alpha_logic,
            beta_logic=profiles.beta_logic,
            li_method=li_method,
            Raxis=profiles.Raxis,
            target_relative_tolerance=fit_tolerance,
            max_solving_iterations=max_solving_iterations,
            optimizer_kwargs=optimizer_kwargs,
            regularization_weight=regularization_weight,
            use_metric_jacobian=use_metric_jacobian,
            metric_jacobian=reusable_metric_jacobian[0],
            jacobian_rel_step=jacobian_rel_step,
        )
        if (
            reuse_metric_jacobian
            and reusable_metric_jacobian[0] is None
            and fit_result.metric_jacobian is not None
        ):
            reusable_metric_jacobian[0] = fit_result.metric_jacobian
        fit_results.append(fit_result)
        return fitted_profiles, True

    adjuster.betap = target_betap
    adjuster.li = target_li
    adjuster.li_method = li_method
    adjuster.fit_results = fit_results
    adjuster.reusable_metric_jacobian = reusable_metric_jacobian

    return adjuster


def _first_two_lao_coefficients(profile_coefficients, override, name):
    """Return the two Lao coefficients supplied before logic terms are appended."""
    if override is not None:
        coefficients = np.asarray(override, dtype=float)
    else:
        coefficients = np.asarray(profile_coefficients[:2], dtype=float)

    if coefficients.shape != (2,):
        raise ValueError(f"{name} must contain exactly two Lao coefficients.")

    return coefficients


class VirtualCircuitHandling:
    """
    The virtual circuits handling class.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialises the virtual circuits.

        Parameters
        ----------

        """

        # name to store the VC under
        self.default_VC_name = f"VC_{datetime.today().strftime('%Y%m%d')}"

    def define_solver(
        self, solver: object, target_relative_tolerance: float = 1e-7
    ) -> None:
        """
        Sets the solver in the VC class.

        Parameters
        ----------
        solver : object
            The static Grad-Shafranov solver object.
        target_relative_tolerance : float
            Target relative tolerance to be met by the solver.

        Returns
        -------
        None
            Modifies the class object in place.
        """

        self.solver = solver
        self.target_relative_tolerance = target_relative_tolerance

    def build_current_vec(self, eq: object, coils: list[str]) -> None:
        """
        For the given equilibrium, this function stores the coil currents
        (for those listed in 'coils') in the class object.

        Parameters
        ----------
        eq : object
            The equilibrium object.
        coils : list
            List of strings containing the names of the coil currents to be stored.

        Returns
        -------
        None
            Modifies the class object in place.
        """

        # empty array for the currents
        self.currents_vec = np.zeros(len(coils))

        # set the currents
        for i, coil in enumerate(coils):
            self.currents_vec[i] = eq.tokamak[coil].current

    def assign_currents(
        self, currents_vec: np.ndarray, coils: list[str], eq: object
    ) -> None:
        """
        For the given equilibrium, this function assigns the coil currents
        (for those listed in 'coils') in the class object.

        Parameters
        ----------
        currents_vec : np.array
            Vector of coil currents to be assigned to the eq object using the coil
            names in 'coils.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        eq : object
            The equilibrium object.

        Returns
        -------
        None
            Modifies the class object in place.
        """

        # directly assign the currents
        for i, coil in enumerate(coils):
            eq.tokamak.set_coil_current(coil, currents_vec[i])

    def assign_currents_solve_GS(
        self,
        currents_vec: np.ndarray,
        coils: list[str],
        target_relative_tolerance: float,
    ) -> None:
        """
        Assigns the coil currents in 'currents_vec' to a private equilibrium object and
        then solve using the static GS solver.

        Parameters
        ----------
        currents_vec : np.array
            Input current values to be assigned. Format as in self.assign_currents.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        target_relative_tolerance : float
            Target relative tolerance to be met by the solver.

        Returns
        -------
        None
            Modifies the class (and other private) object(s) in place.
        """

        # Profile-adjusted finite differences must share one fitted baseline.
        if self.profile_adjuster is not None:
            self._profiles2 = self._baseline_profiles.copy()

        # assign currents
        self.assign_currents(currents_vec, coils, eq=self._eq2)

        # solve for equilibrium, optionally first adjusting profile parameters
        self._profiles2 = self.solve_GS_with_profile_adjuster(
            self._eq2,
            self._profiles2,
            target_relative_tolerance,
            profile_adjuster=self.profile_adjuster,
            suppress=False,
        )

    def solve_GS_with_profile_adjuster(
        self,
        eq,
        profiles,
        target_relative_tolerance,
        profile_adjuster=None,
        suppress=True,
    ):
        """
        Solve a static GS problem, optionally updating the profile first.

        ``profile_adjuster`` is called after any caller-side current changes and
        before the final target evaluation. It may either return a profile object
        that still needs solving, or ``(profile, solved)`` where ``solved=True``
        indicates that the adjuster has already run a static solve.

        Parameters
        ----------
        eq : object
            Equilibrium object to solve.
        profiles : object
            Profile object used by the static solve.
        target_relative_tolerance : float
            Target relative tolerance to be met by the solver.
        profile_adjuster : callable, optional
            Optional callback with signature ``(eq, profiles, solver,
            target_relative_tolerance)``.
        suppress : bool, optional
            Whether to suppress static solver output when this method performs
            the solve itself.

        Returns
        -------
        object
            The profile object associated with the solved equilibrium.
        """

        solved_by_adjuster = False
        if profile_adjuster is not None:
            adjusted = profile_adjuster(
                eq,
                profiles,
                self.solver,
                target_relative_tolerance,
            )
            if isinstance(adjusted, tuple):
                profiles, solved_by_adjuster = adjusted
            elif adjusted is not None:
                profiles = adjusted

        if not solved_by_adjuster:
            try:
                self.solver.forward_solve(
                    eq,
                    profiles,
                    target_relative_tolerance=target_relative_tolerance,
                    suppress=suppress,
                )
            except AttributeError:
                raise AttributeError("Solver not defined. Call define_solver() first.")

        return profiles

    def prepare_build_dIydI_j(
        self,
        j: int,
        coils: list[str],
        target_dIy: float,
        starting_dI: float,
        min_curr: float = 1e-4,
        max_curr: float = 300,
    ) -> None:
        """
        Prepares to compute the term d(Iy)/dI_j of the Jacobian by
        inferring the value of delta(I_j) corresponding to a change delta(I_y)
        with norm(delta(I_y)) = target_dIy.

        Here:
            - Iy is the flattened vector of plasma currents (on the computational grid).
            - I_j is the current in the jth coil.

        Parameters
        ----------
        j : int
            Index identifying the current to be varied. Indexes as in self.currents_vec.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        target_dIy : float
            Target value for the norm of delta(I_y), from which the finite difference derivative is calculated.
        starting_dI : float
            Initial value to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        min_curr : float, optional, by default 1e-4
            If inferred current value is below min_curr, clip to min_curr.
        max_curr : float, optional, by default 300
            If inferred current value is above max_curr, clip to max_curr.

        Returns
        -------
        None
            Modifies the class (and other private) object(s) in place.
        """

        # copy of currents
        currents = np.copy(self.currents_vec)

        # perturb current j
        currents[j] += starting_dI

        # assign current to the coil and solve static GS problem
        self.assign_currents_solve_GS(currents, coils, self.target_relative_tolerance)

        # difference between plasma current vectors (before and after the solve)
        dIy_0 = self._eq2.limiter_handler.Iy_from_jtor(self._profiles2.jtor) - self.Iy

        # relative norm of plasma current change
        norm_dIy_0 = np.linalg.norm(dIy_0)
        if norm_dIy_0 < 1e-10:
            raise ZeroDivisionError(
                "Norm of change in jtor is near-zero for this Jacobian, please increase 'starting_dI' parameter."
            )
        else:
            rel_ndIy_0 = norm_dIy_0 / self._nIy

        # scale the starting_dI to match the target
        final_dI = starting_dI * target_dIy / rel_ndIy_0

        # clip small/large currents
        final_dI = np.clip(final_dI, min_curr, max_curr)

        # store
        self.final_dI_record[j] = final_dI

    def build_dIydI_j(
        self,
        j: int,
        coils: list[str],
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Computes the term d(Iy)/dI_j of the Jacobian as a finite difference derivative,
        using the value of delta(I_j) inferred earlier by self.prepare_build_dIydI_j.

        Here:
            - Iy is the flattened vector of plasma currents (on the computational grid).
            - I_j is the current in the jth coil.

        Parameters
        ----------
        j : int
            Index identifying the current to be varied. Indexes as in self.currents_vec.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        verbose: bool
            Display output (or not).

        Returns
        -------
        np.array
            The column of the shape (Jacobian) matrix corresponding to coil j,
            i.e. d(targets)/dI_j.
        """

        # print some output
        if verbose:
            print(f"Coil {coils[j]}")

        # store dI
        final_dI = 1.0 * self.final_dI_record[j]

        # copy of currents
        currents = np.copy(self.currents_vec)

        # perturb current
        currents[j] += final_dI

        # assign current to the coil and solve static GS problem
        self.assign_currents_solve_GS(currents, coils, self.target_relative_tolerance)

        # calculate finite difference of targets wrt to the coil current
        self._target_vec_1 = self.target_calculator(self._eq2)

        dtargets = self._target_vec_1 - self._targets_vec

        return dtargets / final_dI

    @staticmethod
    def calculate_matrix_inverse(
        matrix: np.ndarray,
        tikhonov_lambda: np.ndarray | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Compute inverse of a generically non-square matrix
        By default Moore Penrose inverse is used (np.pinv).
        If Tikhonov_lambda is provided then Tikhonov regularisation is applied
        and inv = [M^T M + diag(lambda)]^-1 M^T

        Parameters
        ----------
        matrix : np.ndarray
            matrix to be inverted
        tikhonov_lambda : np.ndarray, optional
            1d array of tikhonov coefficients, or 2d diagonal matrix of coefficients.
            Must have size/shape consistent with matrix.shape[1].
        verbose : bool, optional
            Display output (or not).

        Returns
        -------
        inverse : np.ndarray
            inverse of matrix
        """

        # convert tensorflow to numpy
        matrix = np.asarray(matrix)

        # use regular moore-penrose pseudo inverse
        if tikhonov_lambda is None:
            if verbose:
                print("VC computing using Moore-Penrose pseudoinverse.")
            inverse = np.linalg.pinv(matrix)

        # use tikhonov regularisation in the inverse calculation
        else:
            if verbose:
                print("VC computed using Tikhonov regularised inverse. ")
            tikhonov_lambda = np.asarray(
                tikhonov_lambda
            )  # convert tensorflow to numpy.
            n_cols = matrix.shape[1]

            if tikhonov_lambda.ndim == 1:
                if tikhonov_lambda.shape[0] != n_cols:
                    raise ValueError(
                        f"tikhonov_lambda length {tikhonov_lambda.shape[0]} "
                        f"must match matrix column count {n_cols}."
                    )
                tikhonov_matrix = np.diag(tikhonov_lambda)

            elif tikhonov_lambda.ndim == 2:
                if tikhonov_lambda.shape != (n_cols, n_cols):
                    raise ValueError(
                        f"tikhonov_lambda shape {tikhonov_lambda.shape} "
                        f"must be ({n_cols}, {n_cols}) to match matrix column count."
                    )
                if not np.allclose(tikhonov_lambda, np.diag(np.diag(tikhonov_lambda))):
                    raise ValueError(
                        "tikhonov_lambda 2d array must be a diagonal matrix."
                    )
                tikhonov_matrix = tikhonov_lambda

            else:
                raise ValueError(
                    f"tikhonov_lambda must be 1d or 2d, got {tikhonov_lambda.ndim}d."
                )

            inverse = np.linalg.solve(matrix.T @ matrix + tikhonov_matrix, matrix.T)

        return inverse

    def calculate_VC(
        self,
        eq: object,
        profiles: object,
        coils: list[str],
        target_names: list[str],
        target_calculator: Callable[[object], np.ndarray],
        target_dIy: float = 1e-3,
        starting_dI: np.ndarray | None = None,
        min_starting_dI: float = 50,
        verbose: bool = False,
        tikhonov_lambda: np.ndarray | None = None,
        name: str | None = None,
        profile_adjuster: Callable | None = None,
    ) -> None:
        """
        Calculate the "virtual circuits" matrix:

            V = (S^T S)^(-1) S^T,

        which is the Moore-Penrose pseudo-inverse of the shape (Jacobian) matrix S:

            S_ij = dT_i / dI_j.

        This represents the sensitivity of target parameters T_i to changes in coil
        currents I_j.

        Parameters
        ----------
        eq : object
            The equilibrium object.
        profiles : object
            The profiles object.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        target_names : list
            A list of the target names, e.g [Rin, Rout, Rx, Zx, ...] (must be same length as array from target_calculator).
        target_calculator : function
            Function returning an array of the shape targets (VC will be calculated for ALL of these targets).
        target_dIy : float
            Target value for the norm of delta(I_y), from which the finite difference derivative is calculated.
        starting_dI : array
            Initial current perturbations [Amps] to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        min_starting_dI : float
            Minimum starting_dI value to be used as delta(I_j): to infer the slope of norm(delta(I_y))/delta(I_j).
        verbose: bool
            Display output (or not).
        tikhonov_lambda : np.ndarray, optional
            Tikhonov regularisation coefficients to use when inverting the shape
            matrix. See calculate_matrix_inverse for details. If None (default),
            the Moore-Penrose pseudo-inverse is used instead.
        name: str
            Name to store the VC under (in the 'VirtualCircuit' class).
        profile_adjuster : callable, optional
            Optional callback used to update profile parameters after each coil
            current perturbation and before target derivatives are evaluated.
            The callback receives ``(eq, profiles, solver,
            target_relative_tolerance)`` and returns either an updated profile
            object or ``(updated_profile, solved)``. If ``solved`` is true, the
            callback is assumed to have already solved the static GS problem.
            Baseline targets are evaluated after this adjustment, and every
            finite-difference perturbation starts from the same adjusted profile.

        Returns
        -------
        None
            Modifies the class (and other private) object(s) in place.

        """

        # store original currents
        self.build_current_vec(eq, coils)

        # store function to calculate targets from equilibrium
        if target_calculator is None:
            raise ValueError("You need to input a 'target_calculator' function!")
        self.target_calculator = target_calculator

        self.profile_adjuster = profile_adjuster

        # Establish the adjusted baseline before evaluating finite differences.
        profiles = self.solve_GS_with_profile_adjuster(
            eq=eq,
            profiles=profiles,
            target_relative_tolerance=self.target_relative_tolerance,
            profile_adjuster=profile_adjuster,
        )

        self._targets_vec = self.target_calculator(eq)
        if target_names is None:
            raise ValueError("You need to input a list of 'target_names'!")
        elif len(target_names) != len(self._targets_vec):
            raise ValueError(
                "Number of 'target_names' does not match length of array from 'target_calculator' function!"
            )
        self.target_names = target_names

        # store the flattened plasma current vector (and its norm)
        self.Iy = eq.limiter_handler.Iy_from_jtor(profiles.jtor).copy()
        self._nIy = np.linalg.norm(self.Iy)

        # define starting_dI using currents if not given
        if starting_dI is None:
            starting_dI = np.abs(self.currents_vec.copy()) * target_dIy
            starting_dI = np.where(
                starting_dI > min_starting_dI, starting_dI, min_starting_dI
            )

        if verbose:
            print("--- Stage one ---")
            print(
                f"Re-sizing each initial coil current shift so that it produces a {np.round(target_dIy*100,2)}% change in plasma current density from the input equilibrium."
            )

        # storage matrices
        shape_matrix = np.zeros((len(self._targets_vec), len(coils)))
        self.final_dI_record = np.zeros(len(coils))

        # make copies of the newly solved equilibrium and profile objects
        # these are used for all GS solves below
        self._eq2 = eq.create_auxiliary_equilibrium()
        self._baseline_profiles = profiles.copy()
        self._profiles2 = self._baseline_profiles.copy()

        # for each coil, prepare by inferring delta(I_j) corresponding to a change delta(I_y)
        # with norm(delta(I_y)) = target_dIy
        for j in np.arange(len(coils)):
            self.prepare_build_dIydI_j(j, coils, target_dIy, starting_dI[j])
            if verbose:
                print(
                    f"Coil {coils[j]} (original current shift = {np.round(starting_dI[j],2)} [A] --> scaled current shift {np.round(self.final_dI_record[j],2)} [A])."
                )

        if verbose:
            print("--- Stage two ---")
            print(
                "Building the shape matrix (Jacobian) of the shape parameter changes wrt scaled current shifts for each coil:"
            )

        # for each coil, build the Jacobian using the value of delta(I_j) inferred earlier
        # by self.prepare_build_dIydI_j.
        for j in np.arange(len(coils)):
            # each shape matrix row is derivative of targets wrt the final coil current change
            shape_matrix[:, j] = self.build_dIydI_j(j, coils, verbose)

        # store the data in its own (new) class
        if name is None:
            name = self.default_VC_name

        if verbose:
            print("--- Stage three ---")
            print("Inverting the shape matrix to get the virtual circuit matrix.")
            print(f"VC object stored under name: '{name}'.")

        # vc_matrix is the pseudo inverse of shape_matrix
        vc_matrix = self.calculate_matrix_inverse(
            shape_matrix, tikhonov_lambda=tikhonov_lambda, verbose=verbose
        )

        # store the VC object dynamically
        store_VC = VirtualCircuit(
            name=name,
            eq=eq,
            profiles=profiles,
            shape_matrix=shape_matrix,
            VCs_matrix=vc_matrix,
            target_names=target_names,
            coils=coils,
            target_calculator=target_calculator,
            profile_adjuster=profile_adjuster,
        )
        setattr(self, name, store_VC)

    def apply_VC(
        self,
        eq: object,
        profiles: object,
        VC_object: VirtualCircuit,
        requested_target_shifts: list[float],
        verbose: bool = False,
        profile_adjuster: Callable | None = None,
    ) -> tuple[object, object, np.ndarray, np.ndarray]:
        """
        Here we apply the VC matrix V to requested shifts in the target quantities (dT),
        obtaining the shift in the currents (in coils, dI) required to achieve this:

            dI = V * dT.

        Applying the current shifts to the existing currents, we
        re-solve the equilibrium and return to user.

        Parameters
        ----------
        eq : object
            The equilibrium object upon which to apply the VCs.
        profiles : object
            The profiles object upon which to apply the VCs.
        VC_object : an instance of the VirtualCircuit class
            Specifies the virtual circuit matrix and properties.
        requested_target_shifts : list
            List of floats containing the shifts in all of the relevant targets.
            Same order as VC_object.target_names.
        verbose: bool
            Display output (or not).
        profile_adjuster : callable, optional
            Optional callback used to update profile parameters after applying
            the VC coil-current shifts. If omitted, the adjuster stored on the
            ``VC_object`` is used.

        Returns
        -------
        object
            Returns the equilibrium object after applying the shifted currents.
        object
            Returns the profiles object after applying the shifted currents.
        np.array
            Array of new target values.
        np.array
            Array of old target values.
        """

        # verify targets, coils, and shifts all match those used to generate VCs
        if len(requested_target_shifts) != VC_object.VCs_matrix.shape[1]:
            raise ValueError(
                "The length of 'requested_target_shifts' does not match the list of targets "
                "associated with the supplied VC object!"
            )

        # calculate current shifts required using shape matrix (for stability)
        # uses least squares solver to solve S*dI = dT
        # where dT are the target shifts and dI the current shifts
        current_shifts = np.linalg.lstsq(
            VC_object.shape_matrix, np.array(requested_target_shifts), rcond=None
        )[0]

        if verbose:
            print(f"Currents shifts from VCs:")
            print(f"{VC_object.coils} = {current_shifts}.")

        if profile_adjuster is None:
            profile_adjuster = getattr(VC_object, "profile_adjuster", None)

        # re-solve static GS problem (to make sure it's solved already)
        profiles = self.solve_GS_with_profile_adjuster(
            eq=eq,
            profiles=profiles,
            target_relative_tolerance=self.target_relative_tolerance,
            profile_adjuster=profile_adjuster,
        )

        # calculate the targets
        # if not hasattr(self, "target_calculator"):
        #     self.target_calculator = VC_object.target_calculator
        old_target_values = VC_object.target_calculator(eq)

        # store copies of the eq and profile objects
        eq_new = eq.create_auxiliary_equilibrium()
        profiles_new = profiles.copy()

        # assign currents to the required coils in the eq object
        new_currents = [
            eq_new.tokamak.getCurrents()[name] + current_shifts[i]
            for i, name in enumerate(VC_object.coils)
        ]
        self.assign_currents(new_currents, VC_object.coils, eq=eq_new)

        # solve for the new equilibrium
        profiles_new = self.solve_GS_with_profile_adjuster(
            eq=eq_new,
            profiles=profiles_new,
            target_relative_tolerance=self.target_relative_tolerance,
            profile_adjuster=profile_adjuster,
        )

        # calculate new target values and the difference vs. the old
        new_target_values = VC_object.target_calculator(eq_new)

        if verbose:
            print(f"Targets shifts from VCs:")
            print(
                f"{VC_object.target_names} = {new_target_values - old_target_values}."
            )

        return eq_new, profiles_new, new_target_values, old_target_values
