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

from freegsnke.observable_registry import ObservableRegistry
from freegsnke.virtual_circuits import VirtualCircuit, VirtualCircuitHandling


class VirtualCircuitProvider(abc.ABC):
    """
    Defines the interface for a Virtual Circuit provider.

    TODO(Matthew): We here have get_vc require the ObservableRegistry, but as this can
                    be in-principle stateless (and if stateful it can be keyed on
                    timestamp).
    """

    def __init__(self, observable_registry: ObservableRegistry | None = None):
        """
        Initialise the virtual circuit provider.

        Parameters
        ----------
        observable_registry : ObservableRegistry | None (default: None)
            The observable registry to set the provider to use.
        """
        self._observable_registry = None
        if observable_registry is None or self._validate_observable_registry(
            observable_registry
        ):
            self._observable_registry = observable_registry

    @abc.abstractmethod
    def get_vc(
        self,
        targets: list[str],
        coils: list[str],
        coils_calc: list[str],
        input_data,
    ) -> np.ndarray | None:
        """
        Gets a Virtual Circuit for the given timestamp and observables requested from
        the registry.

        Parameters
        ----------
        targets : list[str]
            list of targets to get a virtual circuit for
        coils : list[str]
            list of coils to return VC matrix for.
        coils_calc : list[str]
            list of coils to use for computing VC/jacobians.
            Must be subset or equal to coils above
        input_data :
            data needed to compute/retrieve VC.
            For example array of inputs for NN emulators, eq/profile for Freegsnke.

        Returns
        -------
        vc : VirtualCircuit | None
            virtual circuit object to be used by the control voltages class or None if
            no virtual circuit could be obtained or constructed.
        """
        pass

    def set_observable_registry(self, observable_registry: ObservableRegistry) -> bool:
        """
        Sets observable registry to provided registry if it provides the necessary
        observables for the provider to execute get_vc correctly.

        Parameters
        ----------
        observable_registry : ObservableRegistry | None (default: None)
            The observable registry to set the provider to use.
        """
        if not self._validate_observable_registry(observable_registry):
            return False

        self._observable_registry = observable_registry
        return True

    @abc.abstractmethod
    def _validate_observable_registry(
        self, observable_registry: ObservableRegistry
    ) -> bool:
        """
        Determine if the provided observable registry satisfies the necessary
        requirements for get_vc to be executed correctly. E.g. does it provide access to
        all the physical parameters of an equilibrium needed by a model.

        Parameters
        ----------
        observable_registry : ObservableRegistry
            The observable registry to validate.
        """
        pass


class VCGenerator(VirtualCircuitProvider):
    """
    Class to generate VC's using the built in FreeGSNKE VirtualCircuitHandling class.
    """

    def __init__(self, solver):
        """
        Create instance of VC handling class, and assign solver

        Default available targets for computing are
            - "R_in": inner midplane radius.
            - "R_out": outer midplane radius.
            - "Rx_lower": lower X-point (radial) position.
            - "Zx_lower": lower X-point (vertical) position.
            - "Rx_upper": upper X-point (radial) position.
            - "Zx_upper": upper X-point (vertical) position.
            - "Rs_lower_outer": lower strikepoint (radial) position.
            - "Rs_upper_outer": upper strikepoint (radial) position.

        Parameters
        ----------
        solver : obj
            freegsnke solver object
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
        Rename targets for ease of the user/interfacing with control code.
        Provide two ordered lists of new and old variable names, and list of variable names modified.

        Parameters
        ----------
        names_user :  list[str]
            list of new names for use by user
        names_internal : list[str]
            ordered list of old/internal names corresponding to new names in names_user

        Returns
        -------
        None
            modifies class attributes target_names_user and output_user_to_internal dictionary
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
        Compute and return VC matrix

        Parameters
        ----------
        targets : list[str]
            list of targets to use in VC matrix
        coils : list[str]
            list of coils to use in VC matrix
        inputs : list
            list of inputs. Here it is equilibrium and profiles
        sensitivity : bool
            Optional fag to return sensitivity matrix instead. Defaults to False and will return vc matrix

        Returns
        -------
        vc_matrix : np.ndarray
            virtual circuit matrix
        """
        # get inputs
        print(input_data)
        eq = input_data[0]
        profiles = input_data[1]
        print(eq)
        print(profiles)

        t1 = time.time()
        print(f"Computing VCs for {targets}")
        print("targets user", targets)
        # convert back to internal names
        targets_internal = [self.targets_user_to_internal[t] for t in targets]
        print("targets internal ", targets_internal)

        print("coils for calc", coils_calc)

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
        print("small matrix shape")
        print(np.shape(vc_matrix))

        # larger full matrix, including zeros
        vc_matrix_big = np.zeros((len(coils), len(targets)))
        print("big matrix shape")
        print(np.shape(vc_matrix_big))

        # index dict
        index_coils = {coil: i for i, coil in enumerate(coils)}
        # fill out rows, keeping target order
        for i, coil in enumerate(coils_calc):
            ind = index_coils[coil]
            vc_matrix_big[ind, :] = 1.0 * vc_matrix[i, :]

        t2 = time.time()
        print("VC compute time", t2 - t1)
        if sensitivity == False:
            return vc_matrix_big
        elif sensitivity == True:
            return derivative_matrix

    def get_inputs(self, eq, profiles):
        """method to get inputs

        For now just returns eq, profiles. This will be tweaked into the obs registry somehow
        """
        return eq, profiles

    def set_observable_registry(self, observable_registry: ObservableRegistry) -> bool:
        """
        Sets observable registry to provided registry if it provides the necessary
        observables for the provider to execute get_vc correctly.

        Parameters
        ----------
        observable_registry : ObservableRegistry | None (default: None)
            The observable registry to set the provider to use.
        """
        if not self._validate_observable_registry(observable_registry):
            return False

        self._observable_registry = observable_registry
        return True

    def _validate_observable_registry(
        self, observable_registry: ObservableRegistry
    ) -> bool:
        """
        Determine if the provided observable registry satisfies the necessary
        requirements for get_vc to be executed correctly. E.g. does it provide access to
        all the physical parameters of an equilibrium needed by a model.

        Parameters
        ----------
        observable_registry : ObservableRegistry
            The observable registry to validate.
        """
        pass
