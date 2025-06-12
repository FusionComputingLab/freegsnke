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

from freegsnke.observable_registry import ObservableRegistry
from freegsnke.virtual_circuits import VirtualCircuit


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
        """
        self._observable_registry = None
        if observable_registry is None or self._validate_observable_registry(
            observable_registry
        ):
            self._observable_registry = observable_registry

    @abc.abstractmethod
    def get_vc(
        self,
        timestamp: float,
        targets: list[str],
    ) -> VirtualCircuit | None:
        """
        Gets a Virtual Circuit for the given timestamp and observables requested from
        the registry.

        Parameters
        ----------
        timestamp : float (4 decimal places)
            timestamp at which the virtual circuit should be retrieved
        targets : list[str]
            list of targets to get a virtual circuit for

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
        observable_registry : ObservableRegistry
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
        """
        pass
