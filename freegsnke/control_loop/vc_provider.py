"""
Module to compute virtual circuits (VCs) and interface with the PCS class in a simulation.

Provides two ways to obtain VCs:
- Compute a VC on demand at a given simulation time, using FreeGSNKE's built-in
  ``VirtualCircuitHandling`` functionality, for "real time updates".
- Generate a fixed schedule of VCs up front, to pass directly to the PCS class.

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

# defers evaluation of "X | None"-style annotations to strings, so this
# module still imports cleanly on Python 3.9 (requires-python >=3.9 in
# pyproject.toml; bare `|` unions otherwise need Python >=3.10)
from __future__ import annotations

import abc
from typing import Callable, Optional

import numpy as np

from freegsnke.virtual_circuits import VirtualCircuitHandling


class VirtualCircuitProvider(abc.ABC):
    """
    Defines the interface for a Virtual Circuit provider.
    """

    def __init__(
        self,
        vcg_targets_ctrl: list[str],
        vcg_targets_calc: list[str],
        vcg_coils_calc: list[str],
    ) -> None:

        # Confguration for VC computations
        self.vcg_targets_ctrl = vcg_targets_ctrl
        self.vcg_targets_calc = vcg_targets_calc
        self.vcg_coils_calc = vcg_coils_calc

        print(f"New VCs will be computed for {self.vcg_targets_ctrl}")
        print(
            f"The Jacobian matrix computation and inversion is performed with :\n{self.vcg_targets_calc} \n{self.vcg_coils_calc}"
        )

    @abc.abstractmethod
    def get_vc(
        self,
        targets: list[str],
        targets_calc: list[str],
        coils: list[str],
        coils_calc: list[str],
        input_data: tuple | np.ndarray,
        tikhonov_lambda: np.ndarray | None = None,
        verbose: bool = False,
    ) -> np.ndarray | None:
        """
        Gets a Virtual Circuit for the given timestamp and observables requested from
        the registry.

        Parameters
        ----------
        targets : list[str]
            User-facing names of targets to include in the returned matrix
            (order is preserved). Must be a subset of ``targets_calc``.
        targets_calc : list[str]
            Targets actually used in the VC calculation (sensitivity
            calculation and inversion). May be a superset of ``targets``,
            e.g. if extra targets are needed to condition the inversion.
        coils : list[str]
            Full list of coils defining the output matrix row ordering.
        coils_calc : list[str]
            Subset of coils actually used in the VC calculation.
        input_data : tuple
            Tuple of inputs required for VC computation, obtained from .get_inputs() method.
        tikhonov_lambda : np.ndarray, optional
            Regularisation parameter(s) passed through to
            ``VirtualCircuitHandling.calculate_VC``.
        verbose : bool, optional
            If True, print progress messages while computing the VC.
            Default is False.

        Returns
        -------
        vc : np.ndarray | None
            virtual circuit matrix to be used by the control voltages class or None if
            no virtual circuit could be obtained or constructed.
        """
        pass

    @abc.abstractmethod
    def get_inputs_from_eq(
        self, eq: object, profiles: object
    ) -> tuple | np.ndarray | None:
        """
        Method to obtain input_data from equilibrium and profiles

        Parameters
        ----------
        eq : object
            Equilibrium object.
        profiles : object
            Plasma profile data.

        Returns
        -------
        input_data : tuple or np.ndarray
            data formatted to pass to input_data argument of get_vc()
        """
        pass


class VCGenerator(VirtualCircuitProvider):
    """
    Virtual Circuit (VC) generator based on FreeGSNKE's
    ``VirtualCircuitHandling`` infrastructure to interface with PCS class
    in the vc_generator argument.

    See each method's own docstring for details: ``get_targets``, ``get_vc``,
    ``get_inputs_from_eq``, and ``generate_fixed_schedule``.
    """

    def __init__(
        self,
        solver: object,
        target_calculator: Callable[[object], np.ndarray],
        target_names: list[str],
        vcg_targets_ctrl: list[str],
        vcg_targets_calc: list[str],
        vcg_coils_calc: list[str],
    ) -> None:
        """
        Initialise the VC generator and bind it to a solver.

        This sets up a ``VirtualCircuitHandling`` instance and registers the solver object required for VC computations.

        Parameters
        ----------
        solver : object
            A FreeGSNKE solver instance used internally by
            ``VirtualCircuitHandling`` to compute virtual circuits.
        target_calculator : Callable[[object], np.ndarray]
            Function to compute array of shape targets from a given equilibrium.
            Same as the target_calculator used by ``VirtualCircuitHandling.calculate_VC``.
        target_names : list[str]
            list of target names associated with the outputs of target_calculator.
        vcg_targets_ctrl : list of str , optional
            List of targets to be controlled using the emulated VC's. Must be subset of
            ctrl_targets, and subset/equal to emulated_VC_targets_calc. Those not defined in this list will be taken from waveform-defined
            VCs.
        vcg_targets_calc : list of str , optional
            List of targets to be used when performing pseudoinverse of jacobian when calculating the emulated VC.
        vcg_coils_calc : list of str, optional
            List of coils to use in emulated VC compuation. These are coils to use in computing shape sensitivity matrix.

        """
        # Confguration for VC computations
        super().__init__(
            vcg_targets_ctrl=vcg_targets_ctrl,
            vcg_targets_calc=vcg_targets_calc,
            vcg_coils_calc=vcg_coils_calc,
        )

        self.VCH = VirtualCircuitHandling()
        self.VCH.define_solver(solver)
        self.target_calculator = target_calculator
        self.target_names = target_names

        # construct a dictionary to allow for different ordering or a subset of targets to be used in computation.
        self.target_calculator_dict = {
            name: (lambda eq, i=i: self.target_calculator(eq)[i])
            for i, name in enumerate(self.target_names)
        }

    def _create_target_calculator(
        self, targets: list[str]
    ) -> Callable[[object], np.ndarray]:
        """
        Assemble array function for chosen targets out of dictionary of
        target functions.

        Builds a single callable of function expected by
        ``VirtualCircuitHandling.calculate_VC`` (as its ``target_calculator``
        argument).

        Parameters
        ----------
        targets : list[str]
            Names of targets to evaluate, in the desired output order.
            Each name must be a key in ``self.target_calculator_dict``.

        Returns
        -------
        Callable[[object], np.ndarray]
            Function that takes an equilibrium ``eq`` and returns a 1D
            ``np.ndarray`` of shape ``(len(targets),)`` with the evaluated
            target values, in the same order as ``targets``.
        """

        if targets == self.target_names:
            # return target calculator if target ordering doesn't change
            return self.target_calculator

        else:
            # reorder the target calculator outputs if targets are different order or a subset
            def array_func(eq):
                return np.array(
                    [self.target_calculator_dict[targ](eq) for targ in targets]
                )

            return array_func

    def get_targets(self, outputs: list[str], input_data: tuple) -> np.ndarray:
        """
        Evaluate the current values of a set of targets for a given
        equilibrium.

        Parameters
        ----------
        outputs : list[str]
            Names of the targets to evaluate. Each name must be a key in
            ``self.target_calculator_dict``.
        input_data : tuple
            Tuple of inputs as returned by ``get_inputs_from_eq``, i.e.
            ``(equilibrium, profiles)``. Only the equilibrium (first
            element) is used here.

        Returns
        -------
        np.ndarray
            1D array of shape ``(len(outputs),)`` containing the evaluated
            target values, in the same order as ``outputs``.
        """
        eq = input_data[0]

        # construct target calculator
        target_calculator = self._create_target_calculator(outputs)
        return target_calculator(eq)

    def get_vc(
        self,
        targets: list[str],
        targets_calc: list[str],
        coils: list[str],
        coils_calc: list[str],
        input_data: tuple,
        tikhonov_lambda: np.ndarray | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Compute the virtual circuit (VC) matrix for a given set of targets and coils.
        Only a subset of coils may be used for the VC computation, but the returned
        matrix is expanded to include all coils provided in ``coils``.

        Parameters
        ----------
        targets : list[str]
            User-facing names of targets to include in the returned matrix
            (order is preserved). Must be a subset of ``targets_calc``.
        targets_calc : list[str]
            Targets actually used in the VC calculation (sensitivity
            calculation and inversion). May be a superset of ``targets``,
            e.g. if extra targets are needed to condition the inversion.
        coils : list[str]
            Full list of coils defining the output matrix row ordering.
        coils_calc : list[str]
            Subset of coils actually used in the VC calculation.
        input_data : tuple
            Tuple of inputs required for VC computation. Obtained from .get_inputs() method.
            Expected to be ``(equilibrium, profiles)``.
        tikhonov_lambda : np.ndarray, optional
            Regularisation parameter(s) passed through to
            ``VirtualCircuitHandling.calculate_VC``.
        verbose : bool, optional
            If True, print progress messages while computing the VC.
            Default is False.

        Returns
        -------
        vc_matrix : np.ndarray
            Expanded virtual circuit matrix of shape
            (len(coils), len(targets))

        Raises
        ------
        ValueError
            If ``targets_calc`` contains a target with no corresponding
            entry in ``self.target_calculator_dict``.
        """

        if not set(targets_calc).issubset(self.target_calculator_dict.keys()):
            raise ValueError(
                "All chosen control targets in `targets_calc` must have a corresponding function in the target_calculator_dict"
            )

        # get inputs
        eq = input_data[0]
        profiles = input_data[1]

        # construct target calculator
        target_calculator = self._create_target_calculator(targets_calc)

        # compute VC
        self.VCH.calculate_VC(
            eq=eq,
            profiles=profiles,
            coils=coils_calc,
            target_names=targets_calc,
            target_calculator=target_calculator,
            tikhonov_lambda=tikhonov_lambda,
            name="latest_VC",
            verbose=verbose,
        )
        vc_matrix = self.VCH.latest_VC.VCs_matrix

        # fill out full vc matrix
        vc_matrix_big_temp = np.zeros((len(coils), len(targets_calc)))

        # fill out rows, keeping target order
        index_coils = {coil: i for i, coil in enumerate(coils)}
        coil_indices = [index_coils[coil] for coil in coils_calc]
        vc_matrix_big_temp[coil_indices, :] = vc_matrix

        # select columns: targets is a subset of targets_calc
        index_targets = {target: i for i, target in enumerate(targets_calc)}
        target_indices = [index_targets[targ] for targ in targets]
        vc_matrix_big = vc_matrix_big_temp[:, target_indices]

        return vc_matrix_big

    def get_inputs_from_eq(self, eq: object, profiles: object) -> tuple[object, object]:
        """
        Package equilibrium and profile data into the input format expected
        by ``get_vc``.

        Parameters
        ----------
        eq : object
            Equilibrium object.
        profiles : object
            Plasma profile data.

        Returns
        -------
        tuple
            ``(eq, profiles)``
        """
        return eq, profiles

    def generate_fixed_schedule(
        self,
        times: list[float],
        eq_list: list[object],
        profile_list: list[object],
        coils: list[str],
        targets_calc: Optional[list[str]] = None,
        targets_ctrl: Optional[list[str]] = None,
        coils_calc: Optional[list[str]] = None,
        tikhonov_lambda: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Generate the shape-target virtual circuit (VC) entries for a fixed
        schedule, in the format expected by the ``circuits_data`` argument of
        ``VirtualCircuitsController``/``PlasmaControlSystem``.

        For each timestamp in ``times``, a VC matrix is computed from the
        corresponding equilibrium/profile pair in ``eq_list``/``profile_list``,
        using ``targets_calc`` and ``coils_calc`` for the underlying
        sensitivity calculation and inversion. The resulting per-coil
        coefficients for each target in ``self.target_names`` (the full set
        of targets this generator was initialised with) are stored over
        time, with targets not in ``targets_ctrl`` left as all-zero arrays
        (i.e. reported to PCS but not actively controlled).

        Note that this method only builds the shape-target entries. The
        plasma-current VC and any feedforward coil drives (``"<coil>_ref"``)
        are not shape targets computed here and must be added separately
        before the result is used as ``circuits_data``.

        Parameters
        ----------
        times : list[float]
            Timestamps for the start of each VC phase.
        eq_list : list[object]
            Equilibria used to compute the VC for each phase (one entry per
            timestamp in ``times``).
        profile_list : list[object]
            Equilibrium profiles used to compute the VC for each phase (one
            entry per timestamp in ``times``).
        coils : list[str]
            Full list of coils defining the output matrix column ordering.
        targets_calc : list[str], optional
            Targets actually used in the VC calculation (sensitivity
            calculation and inversion). Must be a subset of
            ``self.target_names``.
            Defaults to list set when initialising class
        targets_ctrl : list[str], optional
            List of targets that are going to be controlled, with non-zero
            VC arrays. Must be a subset of ``self.target_names`` and of
            ``targets_calc``.
            Defaults to list set when initialising class
        coils_calc : list[str], optional
            Subset of coils actually used in the VC calculation.
            Defaults to list set when initialising class
        tikhonov_lambda : np.ndarray, optional
            Regularisation parameter(s) passed through to ``get_vc`` (and in
            turn to ``VirtualCircuitHandling.calculate_VC``) for every phase
            in the schedule.
        verbose : bool, optional
            If True, print progress messages as each phase is computed.
            Default is False.

        Returns
        -------
        schedule : dict
            One entry per target in ``self.target_names``, each a dict with:
                "times" : np.ndarray, shape (len(times),)
                    the schedule timestamps for this target
                "vals" : np.ndarray, shape (len(times), len(coils))
                    that target's coil coefficients at each scheduled time
            Targets not in ``targets_ctrl`` are left with all-zero "vals".
            Plus a ``"coil_order"`` entry giving ``coils``.

        Raises
        ------
        ValueError
            If ``targets_ctrl`` is not a subset of ``self.target_names`` or
            of ``targets_calc``;
            if ``targets_calc`` is not a subset of ``self.target_names``;
            if ``coils_calc`` is not a subset of ``coils``;
            if ``eq_list``/``profile_list`` do not match ``times`` in length.
        """
        # assign defaults for coils/targets from VCG class if not provided
        if coils_calc is None:
            coils_calc = self.vcg_coils_calc
        if targets_ctrl is None:
            targets_ctrl = self.vcg_targets_ctrl
        if targets_calc is None:
            targets_calc = self.vcg_targets_calc

        target_names_set = set(self.target_names)
        targets_ctrl_set = set(targets_ctrl)
        targets_calc_set = set(targets_calc)
        coils_set = set(coils)
        coils_calc_set = set(coils_calc)
        n_times = len(times)
        n_coils = len(coils)

        if not targets_ctrl_set.issubset(target_names_set):
            raise ValueError(
                "`targets_ctrl` must be a subset of `self.target_names`; "
                f"found targets not in target_names: {sorted(targets_ctrl_set - target_names_set)}"
            )

        if not targets_ctrl_set.issubset(targets_calc_set):
            raise ValueError(
                "`targets_ctrl` must be a subset of `targets_calc`; "
                f"found targets not in targets_calc: {sorted(targets_ctrl_set - targets_calc_set)}"
            )

        if not targets_calc_set.issubset(target_names_set):
            raise ValueError(
                "`targets_calc` must be a subset of `self.target_names`; "
                f"found targets not in target_names: {sorted(targets_calc_set - target_names_set)}"
            )

        if not coils_calc_set.issubset(coils_set):
            raise ValueError(
                "`coils_calc` must be a subset of `coils`; "
                f"found coils not in coils: {sorted(coils_calc_set - coils_set)}"
            )

        if len(eq_list) != n_times:
            raise ValueError(
                f"`eq_list` must have the same length as `times` ({n_times}), got {len(eq_list)}"
            )
        if len(profile_list) != n_times:
            raise ValueError(
                f"`profile_list` must have the same length as `times` ({n_times}), got {len(profile_list)}"
            )

        # initialise: all-zero coil-coefficient arrays for every target this
        # generator supports; targets not in targets_ctrl are left at zero
        # (uncontrolled)
        schedule = {
            targ: {
                "times": np.asarray(times, dtype=float).copy(),
                "vals": np.zeros((n_times, n_coils)),
            }
            for targ in self.target_names
        }
        schedule["coil_order"] = coils

        if verbose:
            print("Calculating VC schedule...")

        for idx, t in enumerate(times):

            if verbose:
                print(f"---> time {t}s")

            input_data = self.get_inputs_from_eq(eq_list[idx], profile_list[idx])

            # calculate VC matrix for this phase, shape (n_coils, len(targets_ctrl))
            vc_matrix_big = self.get_vc(
                targets=targets_ctrl,
                targets_calc=targets_calc,
                coils=coils,
                coils_calc=coils_calc,
                input_data=input_data,
                tikhonov_lambda=tikhonov_lambda,
                verbose=verbose,
            )

            # populate schedule, keeping non-controlled targets at zero
            for j, targ in enumerate(targets_ctrl):
                schedule[targ]["vals"][idx, :] = vc_matrix_big[:, j]

        if verbose:
            print("--- done! ---")

        return schedule
