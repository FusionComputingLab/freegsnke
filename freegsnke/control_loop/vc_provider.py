"""
Defines the base class, `VirtualCircuitProvider`, for a Virtual Circuit provider. Such
a provider promises to provide a Virtual Circuit given a timestamp and a means to
extract observables regarding the equilibrium. The mechanism by which the Virtual
Circuit is produced, and the observables that are or are not requested for the purpose
of Virtual Circuit construction is not constrained.

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
import time
from copy import deepcopy

import numpy as np

from freegsnke.virtual_circuits import VirtualCircuit, VirtualCircuitHandling


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
            return np.array([self.target_calculator_dict[targ](eq) for targ in targets])

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

        # compute VC
        self.VCH.calculate_VC(
            eq=eq,
            profiles=profiles,
            coils=coils_calc,
            targets=targets_calc,
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
