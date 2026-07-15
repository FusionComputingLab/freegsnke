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

import numpy as np

from freegsnke.virtual_circuits import VirtualCircuitHandling


class VCGenerator:
    """
    Virtual Circuit (VC) generator based on FreeGSNKE's
    ``VirtualCircuitHandling`` infrastructure.

    """

    def __init__(self, solver, target_calculator_dict):
        """
        Initialise the VC generator and bind it to a  solver.

        This sets up a ``VirtualCircuitHandling`` instance and registers the solver object required for VC computations.

        Parameters
        ----------
        solver : object
            A FreeGSNKE solver instance used internally by
            ``VirtualCircuitHandling`` to compute virtual circuits.
        target_calculator_dict : dict[str, Callable]
            Dictionary mapping user-facing target names to functions that,
            given an equilibrium, return the corresponding scalar target
            value. Used to assemble the array function passed into
            ``VirtualCircuitHandling.calculate_VC``.
        """

        self.VCH = VirtualCircuitHandling()
        self.VCH.define_solver(solver)

        self.target_calculator_dict = target_calculator_dict

    def _create_target_calculator(self, targets):
        """Assemble array function for chosen targets out of dictionary of target functions"""

        def array_func(eq):
            return np.array(
                [self.target_calculator_dict[targ](eq)[0] for targ in targets]
            )

        return array_func

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

        The VC matrix maps coil current perturbations to changes in the selected
        plasma shape or position targets. Only a subset of coils may be use
        for the VC computation, but the returned matrix is expanded to include
        all coils provided in ``coils``.

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
            Tuple of inputs required for VC computation.
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
        vc_matrix_big = np.zeros((len(coils), len(targets)))

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

    def get_inputs_from_equi(self, eq, profiles):
        """
        Package equilibrium and profile data into the input format expected
        by ``get_vc``.

        This method exists for compatibility with higher-level infrastructure
        (e.g. observable registries).

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
        reported to PCS but not actively controlled).

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
            and inversion). Must be a superset of ``targets_all``.
        coils_all : list[str]
            list of all coils passed to systems category
        coils_calc : list[str]
            subset of coils actually used in the VC calculation

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
            ``targets_calc``; if ``targets_all`` is not a subset of
            ``targets_calc``; if ``coils_calc`` is not a subset of
            ``coils_all``; or if ``eqi_list``/``profile_list`` do not match
            ``times`` in length.
        """
        targets_all_set = set(targets_all)
        targets_ctrl_set = set(targets_ctrl)
        targets_calc_set = set(targets_calc)
        coils_all_set = set(coils_all)
        coils_calc_set = set(coils_calc)

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

        if not coils_calc_set.issubset(coils_all_set):
            raise ValueError(
                "`coils_calc` must be a subset of `coils_all`; "
                f"found coils not in coils_all: {sorted(coils_calc_set - coils_all_set)}"
            )

        n_times = len(times)
        if len(eqi_list) != n_times:
            raise ValueError(
                f"`eqi_list` must have the same length as `times` ({n_times}), got {len(eqi_list)}"
            )
        if len(profile_list) != n_times:
            raise ValueError(
                f"`profile_list` must have the same length as `times` ({n_times}), got {len(profile_list)}"
            )

        n_coils = len(coils_all)

        # initialise: all-zero coil-coefficient arrays for every target;
        # targets not in targets_ctrl are left at zero (uncontrolled)
        schedule = {
            targ: {
                "times": np.asarray(times, dtype=float).copy(),
                "vals": np.zeros((n_times, n_coils)),
            }
            for targ in targets_all
        }

        targets_ctrl_set = set(targets_ctrl)

        for t_idx in range(n_times):
            input_data = self.get_inputs_from_equi(eqi_list[t_idx], profile_list[t_idx])

            # calculate VC matrix for this phase, shape (n_coils, len(targets_all))
            vc_matrix_big = self.get_vc(
                targets=targets_ctrl,
                targets_calc=targets_calc,
                coils=coils_all,
                coils_calc=coils_calc,
                input_data=input_data,
            )

            # populate schedule, keeping non-controlled targets at zero
            for j, targ in enumerate(targets_all):
                if targ in targets_ctrl_set:
                    schedule[targ]["vals"][t_idx, :] = vc_matrix_big[:, j]

        return schedule
