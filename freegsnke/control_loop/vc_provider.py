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

    This class acts as an adapter between a control or optimisation framework
    and FreeGSNKE's internal VC computation routines. It allows:

    - Mapping between user-facing and internal target names
    - Computation of virtual circuit matrices for a selected subset of coils
    - Optional access to sensitivity (shape/derivative) matrices

    The class assumes that equilibrium and profile objects are provided
    externally (e.g. via an observable registry).
    """

    def __init__(self, solver):
        """
        Initialise the VC generator and bind it to a FreeGSNKE solver.

        This sets up a ``VirtualCircuitHandling`` instance and registers the
        solver object required for VC computations. It also defines the default
        set of supported targets and their internal naming conventions.

        Default available targets are:

        - ``"R_in"``           : Inner midplane radius
        - ``"R_out"``          : Outer midplane radius
        - ``"Rx_lower"``       : Lower X-point radial position
        - ``"Zx_lower"``       : Lower X-point vertical position
        - ``"Rx_upper"``       : Upper X-point radial position
        - ``"Zx_upper"``       : Upper X-point vertical position
        - ``"Rs_lower_outer"`` : Lower outer strike-point radius
        - ``"Rs_upper_outer"`` : Upper outer strike-point radius

        Parameters
        ----------
        solver : object
            A FreeGSNKE solver instance used internally by
            ``VirtualCircuitHandling`` to compute virtual circuits.
        """

        self.VCH = VirtualCircuitHandling()
        self.VCH.define_solver(solver)

        # intenrnal names of targets, as prescribed in freegsnke.
        self.target_names_internal = [
            "R_in",
            "R_out",
            "Rx_lower",
            "Zx_lower",
            "Rx_upper",
            "Zx_upper",
            "Rs_lower_outer",
            "Rs_upper_outer",
        ]
        self.target_names_user = deepcopy(self.target_names_internal)
        self.targets_user_to_internal = dict(
            zip(self.target_names_user, self.target_names_internal)
        )

    def rename_targets(self, names_user: list[str], names_internal: list[str]):
        """
        Rename target labels exposed to the user or control code.

        This method allows user-facing target names to differ from the
        internal FreeGSNKE naming scheme. The mapping is order-dependent:
        each entry in ``names_user`` corresponds to the entry at the same
        index in ``names_internal``.

        Parameters
        ----------
        names_user : list[str]
            New target names to be exposed to the user.
        names_internal : list[str]
            Existing internal target names to be replaced.

        Returns
        -------
        None
            Updates ``target_names_user`` and ``targets_user_to_internal`` in-place.
        """

        internal_to_user = dict(zip(names_internal, names_user))
        user_to_internal = dict(zip(names_user, names_internal))

        new_labels = []
        for label in self.target_names_internal:
            if label in internal_to_user.keys():
                new_labels.append(internal_to_user[label])
            else:
                new_labels.append(label)
        self.target_names_user = new_labels

        for key, item in user_to_internal.items():
            self.targets_user_to_internal[key] = item

        print("Targets renamed")
        print(self.target_names_user)
        print("user to internal", self.targets_user_to_internal)

    def get_vc(
        self,
        targets: list[str],
        coils: list[str],
        coils_calc: list[str],
        input_data: tuple,
        sensitivity=False,
    ):
        """
        Compute the virtual circuit (VC) matrix for a given set of targets and coils.

        The VC matrix maps coil current perturbations to changes in the selected
        plasma shape or position targets. Only a subset of coils may be used
        for the VC computation, but the returned matrix is expanded to include
        all coils provided in ``coils``.

        Parameters
        ----------
        targets : list[str]
            User-facing names of targets to include (order is preserved).
        coils : list[str]
            Full list of coils defining the output matrix row ordering.
        coils_calc : list[str]
            Subset of coils actually used in the VC calculation.
        input_data : tuple
            Tuple of inputs required for VC computation.
            Expected to be ``(equilibrium, profiles)``.
        sensitivity : bool, optional
            If ``True``, return the sensitivity (shape/derivative) matrix instead
            of the VC matrix. Default is ``False``.

        Returns
        -------
        vc_matrix : np.ndarray
            Expanded virtual circuit matrix of shape
            ``(len(coils), len(targets))`` if ``sensitivity=False``.
        derivative_matrix : np.ndarray
            Sensitivity (shape) matrix if ``sensitivity=True``.
        """

        # get inputs
        # print(input_data)
        eq = input_data[0]
        profiles = input_data[1]
        # print(eq)
        # print(profiles)

        t1 = time.time()
        # print(f"Computing VCs for {targets}")
        # print("targets user", targets)
        # convert back to internal names
        targets_internal = [self.targets_user_to_internal[t] for t in targets]
        # print("targets internal ", targets_internal)

        # print("coils for calc", coils_calc)

        # compute vc's
        self.VCH.calculate_VC(
            eq=eq,
            profiles=profiles,
            coils=coils_calc,
            targets=targets_internal,
            targets_options=None,
        )
        vc_matrix = self.VCH.latest_VC.VCs_matrix
        derivative_matrix = self.VCH.latest_VC.shape_matrix
        # print("small matrix shape")
        # print(np.shape(vc_matrix))

        # larger full matrix, including zeros
        vc_matrix_big = np.zeros((len(coils), len(targets)))
        # print("big matrix shape")
        # print(np.shape(vc_matrix_big))

        # index dict
        index_coils = {coil: i for i, coil in enumerate(coils)}
        # fill out rows, keeping target order
        for i, coil in enumerate(coils_calc):
            ind = index_coils[coil]
            vc_matrix_big[ind, :] = 1.0 * vc_matrix[i, :]

        t2 = time.time()
        # print("VC compute time", t2 - t1)
        if sensitivity == False:
            return vc_matrix_big
        elif sensitivity == True:
            return derivative_matrix

    def get_inputs(self, eq, profiles):
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
