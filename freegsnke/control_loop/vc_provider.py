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
        targets_ctrl: list[str],
        targets_calc: list[str],
        coils_calc: list[str],
        vc_update_rate: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """
        Store the coil/target configuration used for every VC computation
        made by this provider.

        Parameters
        ----------
        targets_ctrl : list[str]
            List of targets to be controlled using the emulated VCs. Must be a
            subset of ``ctrl_targets``, and subset/equal to ``targets_calc``.
            Those not defined in this list will be taken from waveform-defined
            VCs.
        targets_calc : list[str]
            List of targets to be used when performing the pseudoinverse of the
            Jacobian when calculating the emulated VC.
        coils_calc : list[str]
            List of coils to use in the emulated VC computation. These are the
            coils used in computing the shape sensitivity matrix.
        vc_update_rate : float, optional
            How often, in seconds, ``VirtualCircuitsController.run_control`` should
            recompute the emulated VC via ``get_vc``. Default is 0.0, i.e. a new VC
            is computed at every control step.
        verbose : bool, optional
            If True, print the configuration used for VC computations.
            Default is False.
        """

        # Configuration for VC computations
        self.targets_ctrl = targets_ctrl
        self.targets_calc = targets_calc
        self.coils_calc = coils_calc
        self.vc_update_rate = vc_update_rate

        if verbose:
            print(f"New VCs will be computed for {self.targets_ctrl}")
            print(
                f"The Jacobian matrix computation and inversion is performed with :\n{self.targets_calc} \n{self.coils_calc}"
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
        targets_ctrl: list[str],
        targets_calc: list[str],
        coils_calc: list[str],
        coils: list[str] | None = None,
        vc_update_rate: float = 0.0,
        n_vc_workers: int = 1,
        verbose: bool = False,
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
        targets_ctrl : list[str]
            List of targets to be controlled using the emulated VC's. Must be subset of
            ctrl_targets, and subset/equal to targets_calc. Those not defined in this list will be taken from waveform-defined
            VCs.
        targets_calc : list[str]
            List of targets to be used when performing pseudoinverse of jacobian when calculating the emulated VC.
        coils_calc : list[str]
            List of coils to use in emulated VC compuation. These are coils to use in computing shape sensitivity matrix.
        coils : list[str], optional
            Full list of coils defining the output matrix column ordering, used only by
            ``generate_fixed_schedule``. Not needed if this generator is only used via
            ``get_vc``/``get_inputs_from_eq`` (e.g. through ``VirtualCircuitsController``),
            where the full coil list is instead supplied per-call by the controller.
        vc_update_rate : float, optional
            How often, in seconds, ``VirtualCircuitsController.run_control`` should
            recompute the emulated VC via ``get_vc``. Default is 0.0, i.e. a new VC
            is computed at every control step.
        n_vc_workers : int, optional
            Number of worker processes used to build independent virtual-circuit
            shape-matrix columns. Default is 1, retaining serial calculation.
        verbose : bool, optional
            If True, print the configuration used for VC computations.
            Default is False.

        """
        # Configuration for VC computations
        super().__init__(
            targets_ctrl=targets_ctrl,
            targets_calc=targets_calc,
            coils_calc=coils_calc,
            vc_update_rate=vc_update_rate,
            verbose=verbose,
        )
        self.coils = coils
        self.n_vc_workers = n_vc_workers

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
                """
                Evaluate all requested target quantities for a given equilibrium.

                Parameters
                ----------
                eq : object
                    Equilibrium (or similar state) object passed to each target
                    calculator function.

                Returns
                -------
                np.ndarray
                    1D array of computed target values, in the same order as
                    `targets`, obtained by calling ``self.target_calculator_dict[targ](eq)``
                    for each ``targ`` in `targets`.
                """
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

        Notes
        -----
        As a side effect, stores the shape (Jacobian) matrix used to compute this
        VC as ``self.latest_shape_matrix``, of shape
        (len(targets_calc), len(coils_calc)) -- i.e. *not* expanded/reordered to
        match ``coils``/``targets`` like the returned ``vc_matrix`` is.

        Raises
        ------
        ValueError
            If ``targets_calc`` contains a target with no corresponding
            entry in ``self.target_calculator_dict``; if ``input_data`` is
            None.
        """

        if not set(targets_calc).issubset(self.target_calculator_dict.keys()):
            raise ValueError(
                "All chosen control targets in `targets_calc` must have a corresponding function in the target_calculator_dict"
            )

        if input_data is None:
            raise ValueError(
                "`input_data` must not be None; obtain it via "
                "`get_inputs_from_eq(eq, profiles)`."
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
            n_vc_workers=self.n_vc_workers,
        )
        vc_matrix = self.VCH.latest_VC.VCs_matrix

        # store the Jacobian (shape matrix) used to compute this VC, for
        # external bookkeeping (e.g. VirtualCircuitsController.jacobian_list)
        self.latest_shape_matrix = self.VCH.latest_VC.shape_matrix

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
        tikhonov_lambda: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Generate the shape-target virtual circuit (VC) entries for a fixed
        schedule, in the format expected by the ``circuits_data`` argument of
        ``VirtualCircuitsController``/``PlasmaControlSystem``.

        For each timestamp in ``times``, a VC matrix is computed from the
        corresponding equilibrium/profile pair in ``eq_list``/``profile_list``,
        using ``self.targets_calc`` and ``self.coils_calc`` (set when
        this ``VCGenerator`` was initialised) for the underlying sensitivity
        calculation and inversion. The resulting per-coil coefficients for
        each target in ``self.target_names`` (the full set of targets this
        generator was initialised with) are stored over time, with targets
        not in ``self.targets_ctrl`` left as all-zero arrays (i.e.
        reported to PCS but not actively controlled).

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
                "vals" : np.ndarray, shape (len(times), len(self.coils))
                    that target's coil coefficients at each scheduled time
            Targets not in ``self.targets_ctrl`` are left with all-zero
            "vals". Plus a ``"coil_order"`` entry giving ``self.coils``.

        Raises
        ------
        ValueError
            If ``self.coils`` was not provided when this ``VCGenerator`` was
            initialised;
            if ``self.targets_ctrl`` is not a subset of
            ``self.target_names`` or of ``self.targets_calc``;
            if ``self.targets_calc`` is not a subset of
            ``self.target_names``;
            if ``self.coils_calc`` is not a subset of ``self.coils``;
            if ``eq_list``/``profile_list`` do not match ``times`` in length.
        """
        if self.coils is None:
            raise ValueError(
                "`self.coils` was not provided; pass `coils` when initialising "
                "this `VCGenerator` in order to use `generate_fixed_schedule`."
            )

        target_names_set = set(self.target_names)
        targets_ctrl_set = set(self.targets_ctrl)
        targets_calc_set = set(self.targets_calc)
        coils_set = set(self.coils)
        coils_calc_set = set(self.coils_calc)
        n_times = len(times)
        n_coils = len(self.coils)

        if not targets_ctrl_set.issubset(target_names_set):
            raise ValueError(
                "`self.targets_ctrl` must be a subset of `self.target_names`; "
                f"found targets not in target_names: {sorted(targets_ctrl_set - target_names_set)}"
            )

        if not targets_ctrl_set.issubset(targets_calc_set):
            raise ValueError(
                "`self.targets_ctrl` must be a subset of `self.targets_calc`; "
                f"found targets not in targets_calc: {sorted(targets_ctrl_set - targets_calc_set)}"
            )

        if not targets_calc_set.issubset(target_names_set):
            raise ValueError(
                "`self.targets_calc` must be a subset of `self.target_names`; "
                f"found targets not in target_names: {sorted(targets_calc_set - target_names_set)}"
            )

        if not coils_calc_set.issubset(coils_set):
            raise ValueError(
                "`self.coils_calc` must be a subset of `self.coils`; "
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
        # generator supports; targets not in self.targets_ctrl are left at
        # zero (uncontrolled)
        schedule = {
            targ: {
                "times": np.asarray(times, dtype=float).copy(),
                "vals": np.zeros((n_times, n_coils)),
            }
            for targ in self.target_names
        }
        schedule["coil_order"] = self.coils

        if verbose:
            print("Calculating VC schedule...")

        for idx, t in enumerate(times):

            if verbose:
                print(f"---> time {t}s")

            input_data = self.get_inputs_from_eq(eq_list[idx], profile_list[idx])

            # calculate VC matrix for this phase, shape (n_coils, len(targets_ctrl))
            vc_matrix_big = self.get_vc(
                targets=self.targets_ctrl,
                targets_calc=self.targets_calc,
                coils=self.coils,
                coils_calc=self.coils_calc,
                input_data=input_data,
                tikhonov_lambda=tikhonov_lambda,
                verbose=verbose,
            )

            # populate schedule, keeping non-controlled targets at zero
            for j, targ in enumerate(self.targets_ctrl):
                schedule[targ]["vals"][idx, :] = vc_matrix_big[:, j]

        if verbose:
            print("--- done! ---")

        return schedule
