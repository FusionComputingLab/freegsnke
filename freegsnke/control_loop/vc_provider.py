"""
Module to compute VCs and interface with PCS class in a simulation
Compute a VC at a given time in simulation, computed with built in FreeGSNKE VirtualCircuit functionality, for 'real time updates'.
Generate a fixed schedule of VC's to pass to PCS class, using a custom but fixed scheduled approach.

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

import abc

import numpy as np

from freegsnke.virtual_circuits import VirtualCircuitHandling


class VirtualCircuitProvider(abc.ABC):
    """
    Defines the interface for a Virtual Circuit provider.
    """

    def __init__(
        self,
    ):
        """
        Initialise the virtual circuit provider.
        """

    @abc.abstractmethod
    def get_vc(
        self,
        targets: list[str],
        targets_calc: list[str],
        coils: list[str],
        coils_calc: list[str],
        input_data: tuple | np.ndarray,
        tikhonov_lambda: np.ndarray = None,
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
        Returns
        -------
        vc : VirtualCircuit | None
            virtual circuit matrix to be used by the control voltages class or None if
            no virtual circuit could be obtained or constructed.
        """
        pass

    def get_inputs_from_eq(self, eq: object, profiles: object):
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
        input_data : tuple or array
            data formatted to pass to input_data argument of get_vc()
        """
        pass


class VCGenerator(VirtualCircuitProvider):
    """
    Virtual Circuit (VC) generator based on FreeGSNKE's
    ``VirtualCircuitHandling`` infrastructure to interface with PCS class
    in the vc_generator argument.


    Methods
    -------
    get_targets(outputs, input_data)
        Evaluate the current values of a set of targets directly from an
        equilibrium, without performing any VC/inversion calculation.
    get_vc(targets, targets_calc, coils, coils_calc, input_data, tikhonov_lambda=None)
        Compute the virtual circuit matrix for a chosen subset of targets
        and coils, optionally using a larger set of targets/coils
        internally to condition the inversion.
    get_inputs_from_eq(eq, profiles)
        Package an equilibrium and its profiles into the ``input_data``
        tuple format expected by ``get_vc`` and ``get_targets``.
    generate_fixed_schedule(times, eqi_list, profile_list, targets_all, targets_ctrl, targets_calc, coils_all, coils_calc)
        Build a full time-dependent VC schedule across multiple
        equilibria, suitable for use as ``circuits_data`` in a
        ``VirtualCircuitsController`` or plasma control system (PCS).
    """

    def __init__(self, solver, target_calculator, target_names):
        """
        Initialise the VC generator and bind it to a solver.

        This sets up a ``VirtualCircuitHandling`` instance and registers the solver object required for VC computations.

        Parameters
        ----------
        solver : object
            A FreeGSNKE solver instance used internally by
            ``VirtualCircuitHandling`` to compute virtual circuits.
        target_calculator : array function
            Function to compute array of shape targets from a given equilibrium.
            Same as the target_calculator used by ``VirtualCircuitHandling.calculate_VC``.
        target_names = list[str]
            list of target names associated with the outputs of target_calculator.
        """

        self.VCH = VirtualCircuitHandling()
        self.VCH.define_solver(solver)
        self.target_calculator = target_calculator
        self.target_names = target_names

        # construct a dictionary to allow for different ordering or a subset of targets to be used in computation.
        self.target_calculator_dict = {
            name: (lambda eq, i=i: self.target_calculator(eq)[i])
            for i, name in enumerate(self.target_names)
        }

    def _create_target_calculator(self, targets):
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

    def get_targets(self, outputs: list[str], input_data):
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
        tikhonov_lambda: np.ndarray = None,
    ):
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

        Returns
        -------
        vc_matrix : np.ndarray
            Expanded virtual circuit matrix of shape
            (len(coils), len(targets))
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

        print("calc targets")
        print(target_calculator(eq))

        # compute VC
        self.VCH.calculate_VC(
            eq=eq,
            profiles=profiles,
            coils=coils_calc,
            target_names=targets_calc,
            target_calculator=target_calculator,
            tikhonov_lambda=tikhonov_lambda,
            name="latest_VC",
        )
        vc_matrix = self.VCH.latest_VC.VCs_matrix

        ## fill out full vc matrix
        vc_matrix_big_temp = np.zeros((len(coils), len(targets_calc)))

        # fill out rows, keeping target order
        index_coils = {coil: i for i, coil in enumerate(coils)}
        for i, coil in enumerate(coils_calc):
            ind = index_coils[coil]
            vc_matrix_big_temp[ind, :] = 1.0 * vc_matrix[i, :]

        # select columns :  targets is subset of targets_calc
        index_targets_temp = {target: i for i, target in enumerate(targets_calc)}
        target_indices = [index_targets_temp[targ] for targ in targets]
        vc_matrix_big = np.take(vc_matrix_big_temp, target_indices, axis=1)

        return vc_matrix_big

    def get_inputs_from_eq(self, eq, profiles):
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
        times: np.ndarray,
        eqi_list: list[object],
        profile_list: list[object],
        targets_all: list[str],
        targets_ctrl: list[str],
        targets_calc: list[str],
        coils_all: list[str],
        coils_calc: list[str],
        plasma_schedule: dict,
        ff_drives: dict | None = None,
    ) -> dict:
        """
        Generate a circuits schedule that can be passed into the circuits_data
        argument of the VirtualCircuitsController and PCS classes.

        For each timestamp in ``times``, a VC matrix is computed from the
        corresponding equilibrium/profile pair in ``eqi_list``/``profile_list``,
        using ``targets_calc`` and ``coils_calc`` for the underlying
        sensitivity calculation and inversion. The resulting per-coil
        coefficients for each target in ``targets_all`` are stored over time,
        with targets not in ``targets_ctrl`` left as all-zero arrays (i.e.
        reported to PCS but not actively controlled). The resulting schedule is
        a dictionary with control targets as keys, and each value being a dictionary
        key/items specified by ``"times"`` (np.ndarray) and ``"vals"`` (np.ndarray) entries,


        Inputs:
        -------
        times : np.ndarray
            array of timestamps for the start of the next VC phase
        eqi_list : list[object]
            list of equilibria used to compute the VC for that phase
            (one entry per timestamp in ``times``)
        profile_list : list[object]
            list of equilibrium profiles used to compute the VC for that
            phase (one entry per timestamp in ``times``)
        targets_all : list[str]
            list of targets to provide to PCS
        targets_ctrl : list[str]
            list of targets that are going to be controlled, with non-zero
            VC arrays. Must be a subset of ``targets_all`` and ``targets_calc``.
        targets_calc : list[str]
            list of targets used in VC computation (sensitivity calculation
            and inversion). Must be a subset of ``targets_all``.
        coils_all : list[str]
            list of all coils passed to systems category
        coils_calc : list[str]
            subset of coils actually used in the VC calculation
        plasma_schedule : dict
            dictionary for plasma vc schedule of the form {"plasma" : }
        ff_drives : dict, optional
            Feed-forward coil drive schedules, keyed by ``"{coilname}_ref"``
            for every coil in ``coils_all``. If ``None`` (default), all feed-forward drives
            are set to zero for the full duration of ``times`` (plus a
            10-unit buffer past the final timestamp).

        Returns:
        --------
        schedule : dict
            dictionary of VC schedule to be passed as circuits_data to PCS
            class. Contains one entry per target in ``targets_all``, each a
            dict with:
                "times" : np.ndarray, shape (len(times),)
                    the schedule timestamps for this target
                "vals" : np.ndarray, shape (len(times), len(coils_all))
                    that target's coil coefficients at each scheduled time
            Targets not in ``targets_ctrl`` are left with all-zero "vals".

        Raises:
        -------
        ValueError
            If ``targets_ctrl`` is not a subset of ``targets_all`` or of
            ``targets_calc``;
            if ``targets_all`` is not a subset of ``targets_calc``;
            if ``coils_calc`` is not a subset of ``coils_all``;
            if ``eqi_list``/``profile_list`` do not match
            ``times`` in length.
        """
        targets_all_set = set(targets_all)
        targets_ctrl_set = set(targets_ctrl)
        targets_calc_set = set(targets_calc)
        coils_all_set = set(coils_all)
        coils_calc_set = set(coils_calc)
        n_times = len(times)
        n_coils = len(coils_all)

        if not targets_ctrl_set.issubset(targets_all_set):
            raise ValueError(
                "`targets_ctrl` must be a subset of `targets_all`; "
                f"found targets not in targets_all: {sorted(targets_ctrl_set - targets_all_set)}"
            )

        if not targets_ctrl_set.issubset(targets_calc_set):
            raise ValueError(
                "`targets_ctrl` must be a subset of `targets_calc`; "
                f"found targets not in targets_calc: {sorted(targets_ctrl_set - targets_calc_set)}"
            )

        if not targets_calc_set.issubset(targets_all):
            raise ValueError(
                "`targets_calc` must be a subset of `targets_all`; "
                f"found targets not in targets_all: {sorted(targets_ctrl_set - targets_calc_set)}"
            )

        if not coils_calc_set.issubset(coils_all_set):
            raise ValueError(
                "`coils_calc` must be a subset of `coils_all`; "
                f"found coils not in coils_all: {sorted(coils_calc_set - coils_all_set)}"
            )

        if len(eqi_list) != n_times:
            raise ValueError(
                f"`eqi_list` must have the same length as `times` ({n_times}), got {len(eqi_list)}"
            )
        if len(profile_list) != n_times:
            raise ValueError(
                f"`profile_list` must have the same length as `times` ({n_times}), got {len(profile_list)}"
            )

        # initialise: all-zero coil-coefficient arrays for every target;
        # targets not in targets_ctrl are left at zero (uncontrolled)
        schedule = {
            targ: {
                "times": np.asarray(times, dtype=float).copy(),
                "vals": np.zeros((n_times, n_coils)),
            }
            for targ in targets_all
        }

        # Add plasma schedule
        if plasma_schedule is None:
            raise ValueError(f"Please provide a plasma vc schedule")
        else:
            # check dictoinary of correct form
            for key, entry in plasma_schedule.items():
                if (
                    not isinstance(entry, dict)
                    or "times" not in entry
                    or "vals" not in entry
                ):
                    raise ValueError(
                        f"`plasma_schedule['{key}']` must be a dict with 'times' and 'vals' keys"
                    )
                for arr in entry["vals"]:
                    assert (
                        len(arr) == n_coils
                    ), f"plasma virtual circuit array must have {n_coils} entries"

            # add plasma schedule to schedule
            schedule.update(plasma_schedule)

        # Build shaping VC schedule
        targets_ctrl_set = set(targets_ctrl)

        for t_idx in range(n_times):
            input_data = self.get_inputs_from_eq(eqi_list[t_idx], profile_list[t_idx])

            # calculate VC matrix for this phase, shape (n_coils, len(targets_all))
            vc_matrix_big = self.get_vc(
                targets=targets_ctrl,
                targets_calc=targets_calc,
                coils=coils_all,
                coils_calc=coils_calc,
                input_data=input_data,
            )

            # populate schedule, keeping non-controlled targets at zero
            for j, targ in enumerate(targets_ctrl):
                schedule[targ]["vals"][t_idx, :] = vc_matrix_big[:, j]

            schedule["coil_order"] = coils_all

        if ff_drives is None:
            print(f"default ff coil drives set to zero")
            tmin = times[0]
            tmax = times[-1]
            zeros_dict = {
                "times": np.array([tmin, tmax + 10]),
                "vals": np.array([0.0, 0.0]),
            }
            for coil in coils_all:  # linearly interpolated
                schedule[coil + "_ref"] = zeros_dict
        else:
            # Check input is a dictionary, and that it has the correct keys "{coilname}_ref"
            if not isinstance(ff_drives, dict):
                raise TypeError(
                    f"`ff_drives` must be a dict mapping coil names to schedules, got {type(ff_drives)}"
                )

            expected_keys = {coil + "_ref" for coil in coils_all}
            provided_keys = set(ff_drives.keys())

            missing_keys = expected_keys - provided_keys
            if missing_keys:
                raise ValueError(
                    f"`ff_drives` is missing entries for coils: {sorted(missing_keys)}"
                )

            # Check each entry has the required "times"/"vals" structure
            for key, entry in ff_drives.items():
                if (
                    not isinstance(entry, dict)
                    or "times" not in entry
                    or "vals" not in entry
                ):
                    raise ValueError(
                        f"`ff_drives['{key}']` must be a dict with 'times' and 'vals' keys"
                    )

            # populate schedule directly from the user-provided feed-forward drives
            for coil in coils_all:
                schedule[coil + "_ref"] = ff_drives[coil + "_ref"]

        print(f"Circuits data generated!")

        return schedule
