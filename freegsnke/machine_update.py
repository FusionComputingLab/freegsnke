"""
Defines the FreeGSNKE machine object, which inherits from the FreeGS4E machine object. 

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

import freegs4e


class Machine(freegs4e.machine.Machine):
    """Same as freegs4e.machine.Machine.
    It can have an additional freegs4e.machine.Wall object which specifies the limiter's properties.
    """

    def __init__(self, coils, wall=None, limiter=None):
        """Instantiates the Machine, same as freegs4e.machine.Machine.

        Parameters
        ----------
        coils : FreeGS4E coils[(label, Coil|Circuit|Solenoid]
            List of coils
        wall : FreeGS4E machine.Wall object
            It is only used to display the wall in plots.
        limiter : FreeGS4E machine.Wall object
            This is the limiter. Used to define limiter plasma configurations.
        """
        super().__init__(coils, wall)
        self.limiter = limiter

    def set_machine_description(
        self,
        active_coils_data=None,
        passive_coils_data=None,
        limiter_data=None,
        wall_data=None,
        magnetic_probe_data=None,
        active_coils_path=None,
        passive_coils_path=None,
        limiter_path=None,
        wall_path=None,
        magnetic_probe_path=None,
        refine_mode="G",
        preserve_currents=True,
    ):
        """
        Update this machine in place from direct data dictionaries or pickle paths.

        This accepts the same machine description inputs as
        :func:`freegsnke.build_machine.tokamak`, but applies the new geometry to
        the existing object and refreshes derived FreeGSNKE machine attributes.

        Parameters
        ----------
        active_coils_data : dict, optional
            Dictionary containing the active coil description.
        passive_coils_data : list, optional
            List containing passive structure descriptions. If omitted, no
            passive structures are added.
        limiter_data : list, optional
            List of limiter boundary points.
        wall_data : list, optional
            List of wall boundary points. If omitted, the limiter is used as the
            wall.
        magnetic_probe_data : dict, optional
            Dictionary containing magnetic probe descriptions.
        active_coils_path : str, optional
            Path to a pickle file containing the active coil description.
        passive_coils_path : str, optional
            Path to a pickle file containing passive structure descriptions.
        limiter_path : str, optional
            Path to a pickle file containing limiter boundary points.
        wall_path : str, optional
            Path to a pickle file containing wall boundary points.
        magnetic_probe_path : str, optional
            Path to a pickle file containing magnetic probe descriptions.
        refine_mode : str, optional
            Refinement mode for extended passive structures. Defaults to
            ``"G"``; ``"LH"`` and ``"GQ"`` are also supported. ``"GQ"``
            requires every polygon passive to have four vertices.
        preserve_currents : bool, optional
            If True, currents for labels that are present in both the old and new
            machine descriptions are copied onto the updated coil objects.

        Returns
        -------
        Machine
            This machine object, updated in place.

        Notes
        -----
        This method updates only machine-level state. If this machine is already
        attached to an equilibrium, use
        :meth:`freegsnke.equilibrium_update.Equilibrium.update_machine_description`
        instead so equilibrium Greens functions and limiter masks are refreshed.
        Existing nonlinear solver objects should be reinstantiated after a
        geometry change because they cache machine-dependent matrices and mode
        decompositions.
        """

        from .build_machine import update_tokamak

        return update_tokamak(
            self,
            active_coils_data=active_coils_data,
            passive_coils_data=passive_coils_data,
            limiter_data=limiter_data,
            wall_data=wall_data,
            magnetic_probe_data=magnetic_probe_data,
            active_coils_path=active_coils_path,
            passive_coils_path=passive_coils_path,
            limiter_path=limiter_path,
            wall_path=wall_path,
            magnetic_probe_path=magnetic_probe_path,
            refine_mode=refine_mode,
            preserve_currents=preserve_currents,
        )

    def update_active_coil(self, coil_name, active_coil_data, preserve_current=True):
        """
        Update one active coil/circuit in place from direct machine-description data.

        Parameters
        ----------
        coil_name : str
            Existing active coil/circuit label to replace.
        active_coil_data : dict
            Machine-description entry for ``coil_name``. This is the same data
            structure stored under the coil label in the active-coils pickle or
            direct active-coils dictionary.
        preserve_current : bool, optional
            If True, the old current on ``coil_name`` is copied onto the
            replacement coil/circuit. Defaults to True.

        Returns
        -------
        Machine
            This machine object, updated in place.

        Notes
        -----
        Only the named active coil/circuit and the R/M entries that depend on it
        are recalculated. If this machine is already attached to an equilibrium,
        use :meth:`freegsnke.equilibrium_update.Equilibrium.update_active_coil`
        instead so equilibrium Greens functions are refreshed. Existing
        nonlinear solver objects should be reinstantiated after a geometry
        change because they cache machine-dependent matrices and mode
        decompositions.
        """

        from .build_machine import update_active_coil

        return update_active_coil(
            self,
            coil_name=coil_name,
            active_coil_data=active_coil_data,
            preserve_current=preserve_current,
        )
