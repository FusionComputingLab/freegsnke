"""
Functions that build tokamak objects in FreeGSNKE (from file or otherwise). 

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

import os
import pickle
from copy import deepcopy

import numpy as np
from freegs4e.coil import Coil
from freegs4e.machine import Circuit, Wall
from freegs4e.multi_coil import MultiCoil

from .copying import copy_into
from .machine_config import (
    append_tokamak_R_and_M_entries,
    build_tokamak_R_and_M,
    insert_tokamak_R_and_M_entries,
    remove_tokamak_R_and_M_entry,
)
from .machine_update import Machine
from .magnetic_probes import Probes
from .passive_structure import PassiveStructure
from .refine_passive import generate_refinement


def tokamak(
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
):
    """
    Load the standarised input data required to build the tokamak machine.

    These dictionaries/lists/arrays can either be provided directly or loaded from pickle files.

    At minimum, the tokamak requires active coil data and a limiter (to contain the plasma). The passives,
    the wall, and the magnetic probes are optional.

    Parameters
    ----------
    active_coils_data : dict, optional
        Dictionary containing the active coil data.
    passive_coils_data : dict, optional
        Dictionary containing the passive structure data.
    limiter_data : dict, optional
        Dictionary containing the limiter data.
    wall_data : dict, optional
        Dictionary containing the wall data.
    magnetic_probe_data : dict, optional
        Dictionary containing the magnetic probes data.
    active_coils_path : str, optional
        Path to the pickle file containing the active coil data.
    passive_coils_path : str, optional
        Path to the pickle file containing the passive structure data.
    limiter_path : str, optional
        Path to the pickle file containing the limiter data.
    wall_path : str, optional
        Path to the pickle file containing the wall data.
    magnetic_probe_path : str, optional
        Path to the pickle file containing the magnetic probe data.
    refine_mode : str, optional
        Choose the refinement mode for extended passive structures (input as polygons), by default
        'G' for 'grid' (use 'LH' for alternative mode using a Latin Hypercube implementation).

    Returns
    -------
    tokamak : class
        Returns an object containing the tokamak machine decsription.
    """

    components = build_tokamak_components(
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
    )

    # build the tokamak
    tokamak = Machine(
        components["coil_circuits"],
        wall=components["wall"],
        limiter=components["limiter"],
    )
    apply_tokamak_components(tokamak, components, rebuild_R_and_M=False)

    print("Tokamak built.")

    return tokamak


def build_tokamak_components(
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
):
    """
    Build the reusable pieces of a FreeGSNKE machine description.

    This is shared by initial tokamak construction and in-place machine updates,
    so direct dictionaries and pickle-backed descriptions are handled identically.

    At minimum, active coil data and limiter data must be provided, either as
    dictionaries/lists or as pickle paths. Passive structures, wall, and magnetic
    probes are optional and follow the same defaults as :func:`tokamak`.

    Parameters
    ----------
    active_coils_data : dict, optional
        Dictionary containing the active coil description.
    passive_coils_data : list, optional
        List containing passive structure descriptions. If omitted, no passive
        structures are added.
    limiter_data : list, optional
        List of limiter boundary points.
    wall_data : list, optional
        List of wall boundary points. If omitted, the limiter is used as the wall.
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
        Refinement mode for extended passive structures. Defaults to ``"G"``.

    Returns
    -------
    dict
        Dictionary containing FreeGS4E coil circuits, wall and limiter objects,
        FreeGSNKE coil metadata, coil counts, and the probe object needed to
        initialise or update a :class:`freegsnke.machine_update.Machine`.
    """

    # check data can be loaded correctly
    active_coils, passive_coils, limiter, wall = load_data_dicts(
        active_coils_data=active_coils_data,
        passive_coils_data=passive_coils_data,
        limiter_data=limiter_data,
        wall_data=wall_data,
        active_coils_path=active_coils_path,
        passive_coils_path=passive_coils_path,
        limiter_path=limiter_path,
        wall_path=wall_path,
    )

    # build the actives into their circuits
    coil_circuits = build_actives(active_coils=active_coils)
    n_active_coils = len(coil_circuits)

    # build a vectorised coil dictionary for use throughout freegsnke
    coils_dict = build_active_coil_dict(active_coils=active_coils)

    # coil circuit names
    coil_names = list(coils_dict.keys())

    # add the passive structures to the coil_circuits list
    coil_circuits, coils_dict, coil_names = build_passives(
        passive_coils=passive_coils,
        coil_circuits=coil_circuits,
        coils_dict=coils_dict,
        coil_names=coil_names,
        refine_mode=refine_mode,
    )
    n_passive_coils = len(coil_circuits) - n_active_coils

    # add the limiter
    r_limiter = [entry["R"] for entry in limiter]
    z_limiter = [entry["Z"] for entry in limiter]

    # add the wall
    r_wall = [entry["R"] for entry in wall]
    z_wall = [entry["Z"] for entry in wall]

    probes = Probes(
        coils_dict=coils_dict,
        magnetic_probe_data=magnetic_probe_data,
        magnetic_probe_path=magnetic_probe_path,
    )

    return {
        "coil_circuits": coil_circuits,
        "wall": Wall(r_wall, z_wall),
        "limiter": Wall(r_limiter, z_limiter),
        "coils_dict": coils_dict,
        "coils_list": coil_names,
        "n_active_coils": n_active_coils,
        "n_passive_coils": n_passive_coils,
        "n_coils": n_active_coils + n_passive_coils,
        "probes": probes,
        "machine_description_data": {
            "active_coils": deepcopy(active_coils),
            "passive_coils": deepcopy(passive_coils),
            "limiter": deepcopy(limiter),
            "wall": deepcopy(wall),
            "magnetic_probes": deepcopy(magnetic_probe_data),
        },
    }


def apply_tokamak_components(
    tokamak, components, preserve_currents=True, rebuild_R_and_M=True
):
    """
    Apply a built machine description to an existing tokamak object.

    Parameters
    ----------
    tokamak : Machine
        Tokamak object to update in place.
    components : dict
        Output from :func:`build_tokamak_components`.
    preserve_currents : bool, optional
        If True, currents for coils with matching labels are copied across to the
        updated machine description.
    rebuild_R_and_M : bool, optional
        Recalculate resistance and inductance matrices after updating geometry.

    Returns
    -------
    tokamak : Machine
        The same object, updated in place.

    Notes
    -----
    This updates only the machine object. If the machine is already attached to
    an equilibrium, use
    :meth:`freegsnke.equilibrium_update.Equilibrium.update_machine_description`
    so that equilibrium-level Greens functions and limiter masks are refreshed
    as well. Existing nonlinear solver objects should be reinstantiated after a
    geometry change because they cache machine-dependent matrices and mode data.
    """

    old_currents = tokamak.getCurrents() if preserve_currents else {}
    old_coils_dict = getattr(tokamak, "coils_dict", None)
    old_coils_list = getattr(tokamak, "coils_list", None)
    old_machine_description_data = getattr(tokamak, "_machine_description_data", None)

    changed_coils, topology_changed = _changed_coil_labels(
        old_coils_dict=old_coils_dict,
        old_coils_list=old_coils_list,
        new_coils_dict=components["coils_dict"],
        new_coils_list=components["coils_list"],
        old_machine_description_data=old_machine_description_data,
        new_machine_description_data=components["machine_description_data"],
    )
    if not topology_changed and old_coils_dict is not None:
        _reuse_unchanged_coil_components(tokamak, components, changed_coils)

    tokamak.coils = components["coil_circuits"]
    tokamak.wall = components["wall"]
    tokamak.limiter = components["limiter"]

    tokamak.coils_dict = components["coils_dict"]
    tokamak.coils_list = components["coils_list"]
    tokamak.n_active_coils = components["n_active_coils"]
    tokamak.n_passive_coils = components["n_passive_coils"]
    tokamak.n_coils = components["n_coils"]
    tokamak.probes = components["probes"]
    tokamak._machine_description_data = components["machine_description_data"]
    tokamak._last_machine_update_changed_coils = changed_coils
    tokamak._last_machine_update_topology_changed = topology_changed

    tokamak.current_vec = np.zeros(tokamak.n_coils)
    tokamak.current_dummy_vec = np.zeros(tokamak.n_coils)
    tokamak.coil_names = list(tokamak.getCurrents().keys())
    tokamak.coil_order = {}
    for i, coil in enumerate(tokamak.coil_names):
        tokamak.coil_order[coil] = i

    if preserve_currents:
        for label, current in old_currents.items():
            if label in tokamak.coil_order:
                tokamak.set_coil_current(label, current)

    if topology_changed:
        build_tokamak_R_and_M(tokamak, rebuild=rebuild_R_and_M)
    else:
        build_tokamak_R_and_M(tokamak, changed_coils=changed_coils)

    return tokamak


def build_active_coil_component(coil_name, active_coil_data):
    """
    Build one active coil/circuit and its FreeGSNKE metadata.

    This helper accepts the value normally stored under ``coil_name`` in the
    active-coil machine-description dictionary. It reuses the same builders as
    full tokamak construction, so single coils and compound active circuits are
    interpreted identically in whole-machine and one-coil update paths.

    Parameters
    ----------
    coil_name : str
        Label of the active coil/circuit to build.
    active_coil_data : dict
        Machine-description entry for this active coil/circuit.

    Returns
    -------
    tuple
        ``((coil_name, coil_or_circuit), coil_metadata)`` where the first item
        is suitable for insertion into ``tokamak.coils`` and the second is the
        corresponding ``tokamak.coils_dict[coil_name]`` entry.

    Raises
    ------
    ValueError
        If the supplied data cannot be converted into exactly one active
        coil/circuit with the requested label.
    """

    active_coils = {coil_name: deepcopy(active_coil_data)}
    coil_circuits = build_actives(active_coils=active_coils)
    coils_dict = build_active_coil_dict(active_coils=active_coils)

    if len(coil_circuits) != 1 or coil_name not in coils_dict:
        raise ValueError(
            f"Could not build active coil '{coil_name}'. Check the supplied data format."
        )

    built_label, built_coil = coil_circuits[0]
    if built_label != coil_name:
        raise ValueError(
            f"Active coil data for '{coil_name}' produced label '{built_label}'."
        )

    return (built_label, built_coil), coils_dict[coil_name]


def update_active_coil(tokamak, coil_name, active_coil_data, preserve_current=True):
    """
    Update one active coil/circuit on an existing tokamak object.

    Only the named active coil object, its ``coils_dict`` metadata, cached raw
    machine-description entry, and the affected resistance/inductance matrix
    entries are updated. Passive structures, limiter/wall objects, magnetic
    probe descriptions, and unchanged coil objects are left in place.

    Parameters
    ----------
    tokamak : Machine
        Existing tokamak object to update in place.
    coil_name : str
        Existing active coil/circuit label to replace.
    active_coil_data : dict
        Machine-description entry for ``coil_name``.
    preserve_current : bool, optional
        If True, the old current on ``coil_name`` is copied onto the replacement
        coil/circuit. Defaults to True.

    Returns
    -------
    Machine
        The same tokamak object, updated in place.

    Raises
    ------
    ValueError
        If ``coil_name`` is not an existing active coil/circuit label.

    Notes
    -----
    This updates only machine-level state. If the machine is attached to an
    equilibrium, use
    :meth:`freegsnke.equilibrium_update.Equilibrium.update_active_coil` so that
    equilibrium-level Greens functions are refreshed as well. Existing nonlinear
    solver objects should be reinstantiated after a geometry change because
    they cache machine-dependent matrices and mode decompositions.
    """

    if not hasattr(tokamak, "coil_order") or coil_name not in tokamak.coil_order:
        raise ValueError(f"Tokamak does not contain coil label '{coil_name}'.")
    if not tokamak.coils_dict.get(coil_name, {}).get("active", False):
        raise ValueError(f"Coil label '{coil_name}' is not an active coil/circuit.")

    machine_description_data = getattr(tokamak, "_machine_description_data", None)
    old_active_data = None
    if machine_description_data is not None:
        old_active_data = machine_description_data.get("active_coils", {}).get(
            coil_name
        )

    if old_active_data is not None and _machine_description_values_equal(
        old_active_data, active_coil_data
    ):
        tokamak._last_machine_update_changed_coils = []
        tokamak._last_machine_update_topology_changed = False
        build_tokamak_R_and_M(tokamak, changed_coils=[])
        return tokamak

    old_current = tokamak[coil_name].current
    coil_index = tokamak.coil_order[coil_name]
    built_component, coil_metadata = build_active_coil_component(
        coil_name, active_coil_data
    )

    tokamak.coils[coil_index] = built_component
    tokamak.coils_dict[coil_name] = coil_metadata

    if machine_description_data is None:
        machine_description_data = {"active_coils": {}}
    elif "active_coils" not in machine_description_data:
        machine_description_data["active_coils"] = {}
    machine_description_data["active_coils"][coil_name] = deepcopy(active_coil_data)
    tokamak._machine_description_data = machine_description_data

    if hasattr(tokamak, "probes"):
        tokamak.probes.coils_dict = tokamak.coils_dict
        if hasattr(tokamak.probes, "coil_names"):
            tokamak.probes.coil_names = list(tokamak.coils_dict.keys())

    if preserve_current:
        tokamak.set_coil_current(coil_name, old_current)
    elif hasattr(tokamak, "current_vec"):
        tokamak.current_vec[coil_index] = tokamak[coil_name].current

    tokamak._last_machine_update_changed_coils = [coil_name]
    tokamak._last_machine_update_topology_changed = False
    build_tokamak_R_and_M(tokamak, changed_coils=[coil_name])
    return tokamak


def build_passive_structure_component(name, passive_data, refine_mode="G"):
    """
    Build one passive structure and its FreeGSNKE metadata.

    This helper accepts one item of the passive-coils machine-description
    list (a ``"name"`` key is not required; ``name`` always takes precedence).
    It reuses :func:`build_passives` so a single passive structure is built
    exactly as it would be within a full machine build.

    Parameters
    ----------
    name : str
        Label for the passive structure.
    passive_data : dict
        Machine-description entry for this passive structure.
    refine_mode : str, optional
        Refinement mode for extended (polygonal) passive structures. Defaults
        to ``"G"``.

    Returns
    -------
    tuple
        ``((name, coil_or_passive_structure), coil_metadata)`` where the first
        item is suitable for insertion into ``tokamak.coils`` and the second is
        the corresponding ``tokamak.coils_dict[name]`` entry.

    Raises
    ------
    ValueError
        If the supplied data cannot be converted into exactly one passive
        structure with the requested label.
    """

    passive_entry = deepcopy(passive_data)
    passive_entry["name"] = name

    coil_circuits, coils_dict, _ = build_passives(
        passive_coils=[passive_entry],
        coil_circuits=[],
        coils_dict={},
        coil_names=[],
        refine_mode=refine_mode,
    )

    if len(coil_circuits) != 1 or name not in coils_dict:
        raise ValueError(
            f"Could not build passive structure '{name}'. Check the supplied data format."
        )

    return coil_circuits[0], coils_dict[name]


def _index_passive_coils(passive_coils):
    """
    Map passive-structure label to its ``(index, raw entry)`` within a
    passive-coils list, using the same default naming as
    :func:`build_passives` (``coil.get("name", f"passive_{i}")``).

    Parameters
    ----------
    passive_coils : list of dict
        Raw passive-coils machine-description list.

    Returns
    -------
    dict
        Mapping of resolved label to ``(index, entry)`` in ``passive_coils``.
    """

    by_name = {}
    for i, entry in enumerate(passive_coils):
        name = entry.get("name", f"passive_{i}")
        by_name[name] = (i, entry)
    return by_name


def update_passive_structure(
    tokamak, name, passive_data, preserve_current=True, refine_mode="G"
):
    """
    Update one passive structure on an existing tokamak object, in place.

    Only the named passive-structure object, its ``coils_dict`` metadata, and
    the resistance/inductance matrix entries it affects are recalculated.
    Active coils, other passive structures, limiter/wall objects, and magnetic
    probe descriptions are left in place. This does not change the set or
    ordering of passive structures; use :func:`add_passive_structure` or
    :func:`remove_passive_structure` for that.

    Parameters
    ----------
    tokamak : Machine
        Existing tokamak object to update in place.
    name : str
        Existing passive-structure label to replace.
    passive_data : dict
        Machine-description entry for ``name`` (same format as one item of the
        passive-coils list).
    preserve_current : bool, optional
        If True, the old current on ``name`` is copied onto the replacement
        passive structure. Defaults to True.
    refine_mode : str, optional
        Refinement mode for extended (polygonal) passive structures. Defaults
        to ``"G"``.

    Returns
    -------
    Machine
        The same tokamak object, updated in place.

    Raises
    ------
    ValueError
        If ``name`` is not an existing passive-structure label.

    Notes
    -----
    This updates only machine-level state. If the machine is attached to an
    equilibrium, use
    :meth:`freegsnke.equilibrium_update.Equilibrium.update_passive_structure`
    so that equilibrium-level Greens functions are refreshed as well. Existing
    nonlinear solver objects should be reinstantiated after a geometry change
    because they cache machine-dependent matrices and mode decompositions.
    """

    if not hasattr(tokamak, "coil_order") or name not in tokamak.coil_order:
        raise ValueError(
            f"Tokamak does not contain passive structure label '{name}'."
        )
    if tokamak.coils_dict.get(name, {}).get("active", False):
        raise ValueError(f"Coil label '{name}' is not a passive structure.")

    machine_description_data = getattr(tokamak, "_machine_description_data", None)
    old_passive_coils = list(
        (machine_description_data or {}).get("passive_coils", []) or []
    )
    old_by_name = _index_passive_coils(old_passive_coils)

    new_entry = deepcopy(passive_data)
    new_entry["name"] = name

    if name in old_by_name and _machine_description_values_equal(
        old_by_name[name][1], new_entry
    ):
        tokamak._last_machine_update_changed_coils = []
        tokamak._last_machine_update_topology_changed = False
        build_tokamak_R_and_M(tokamak, changed_coils=[])
        return tokamak

    old_current = tokamak[name].current
    coil_index = tokamak.coil_order[name]
    built_component, coil_metadata = build_passive_structure_component(
        name, passive_data, refine_mode=refine_mode
    )

    tokamak.coils[coil_index] = built_component
    tokamak.coils_dict[name] = coil_metadata

    if name in old_by_name:
        old_passive_coils[old_by_name[name][0]] = new_entry
    else:
        old_passive_coils.append(new_entry)

    if machine_description_data is None:
        machine_description_data = {}
    machine_description_data["passive_coils"] = old_passive_coils
    tokamak._machine_description_data = machine_description_data

    if hasattr(tokamak, "probes"):
        tokamak.probes.coils_dict = tokamak.coils_dict
        if hasattr(tokamak.probes, "coil_names"):
            tokamak.probes.coil_names = list(tokamak.coils_dict.keys())

    if preserve_current:
        tokamak.set_coil_current(name, old_current)
    elif hasattr(tokamak, "current_vec"):
        tokamak.current_vec[coil_index] = tokamak[name].current

    tokamak._last_machine_update_changed_coils = [name]
    tokamak._last_machine_update_topology_changed = False
    build_tokamak_R_and_M(tokamak, changed_coils=[name])
    return tokamak


def add_passive_structure(tokamak, passive_data, name=None, refine_mode="G"):
    """
    Add one new passive structure to an existing tokamak object, in place.

    Only the resistance/inductance matrix row/column for the new passive
    structure are calculated; entries for existing coils are reused.

    Parameters
    ----------
    tokamak : Machine
        Existing tokamak object to update in place.
    passive_data : dict
        Machine-description entry for the new passive structure (same format
        as one item of the passive-coils list).
    name : str, optional
        Label for the new passive structure. If omitted, ``passive_data["name"]``
        is used, and if that is also absent a default of the form
        ``f"passive_{tokamak.n_passive_coils}"`` is used.
    refine_mode : str, optional
        Refinement mode for extended (polygonal) passive structures. Defaults
        to ``"G"``.

    Returns
    -------
    Machine
        The same tokamak object, updated in place.

    Raises
    ------
    ValueError
        If a coil/passive-structure with the resolved label already exists.

    Notes
    -----
    This updates only machine-level state. If the machine is attached to an
    equilibrium, use
    :meth:`freegsnke.equilibrium_update.Equilibrium.add_passive_structure` so
    that equilibrium-level Greens functions are extended as well. Existing
    nonlinear solver objects should be reinstantiated after a geometry change
    because they cache machine-dependent matrices and mode decompositions,
    whose dimensionality will itself have changed.
    """

    if name is None:
        name = passive_data.get("name", f"passive_{tokamak.n_passive_coils}")

    if hasattr(tokamak, "coil_order") and name in tokamak.coil_order:
        raise ValueError(f"Tokamak already contains a coil label '{name}'.")

    built_component, coil_metadata = build_passive_structure_component(
        name, passive_data, refine_mode=refine_mode
    )

    tokamak.coils.append(built_component)
    tokamak.coils_dict[name] = coil_metadata
    tokamak.coils_list.append(name)
    tokamak.n_passive_coils += 1
    tokamak.n_coils += 1

    tokamak.coil_names = list(tokamak.getCurrents().keys())
    tokamak.coil_order = {}
    for i, coil in enumerate(tokamak.coil_names):
        tokamak.coil_order[coil] = i

    tokamak.getCurrentsVec()
    tokamak.current_dummy_vec = np.zeros(tokamak.n_coils)

    machine_description_data = getattr(tokamak, "_machine_description_data", None)
    if machine_description_data is None:
        machine_description_data = {}
    old_passive_coils = list(machine_description_data.get("passive_coils", []) or [])
    new_entry = deepcopy(passive_data)
    new_entry["name"] = name
    old_passive_coils.append(new_entry)
    machine_description_data["passive_coils"] = old_passive_coils
    tokamak._machine_description_data = machine_description_data

    if hasattr(tokamak, "probes"):
        tokamak.probes.coils_dict = tokamak.coils_dict
        if hasattr(tokamak.probes, "coil_names"):
            tokamak.probes.coil_names = list(tokamak.coils_dict.keys())

    tokamak._last_machine_update_changed_coils = [name]
    tokamak._last_machine_update_topology_changed = True
    append_tokamak_R_and_M_entries(tokamak, [name])
    return tokamak


def remove_passive_structure(tokamak, name):
    """
    Remove one passive structure from an existing tokamak object, in place.

    Only the resistance/inductance matrix row/column for the removed passive
    structure are dropped; entries for the remaining coils are reused.

    Parameters
    ----------
    tokamak : Machine
        Existing tokamak object to update in place.
    name : str
        Existing passive-structure label to remove.

    Returns
    -------
    Machine
        The same tokamak object, updated in place.

    Raises
    ------
    ValueError
        If ``name`` is not an existing passive-structure label.

    Notes
    -----
    This updates only machine-level state. If the machine is attached to an
    equilibrium, use
    :meth:`freegsnke.equilibrium_update.Equilibrium.remove_passive_structure`
    so that equilibrium-level Greens functions are shrunk as well. Existing
    nonlinear solver objects should be reinstantiated after a geometry change
    because they cache machine-dependent matrices and mode decompositions,
    whose dimensionality will itself have changed.
    """

    if not hasattr(tokamak, "coil_order") or name not in tokamak.coil_order:
        raise ValueError(
            f"Tokamak does not contain passive structure label '{name}'."
        )
    if tokamak.coils_dict.get(name, {}).get("active", False):
        raise ValueError(f"Coil label '{name}' is not a passive structure.")

    index = tokamak.coil_order[name]

    del tokamak.coils[index]
    del tokamak.coils_dict[name]
    del tokamak.coils_list[index]
    tokamak.n_passive_coils -= 1
    tokamak.n_coils -= 1

    tokamak.coil_names = list(tokamak.getCurrents().keys())
    tokamak.coil_order = {}
    for i, coil in enumerate(tokamak.coil_names):
        tokamak.coil_order[coil] = i

    tokamak.getCurrentsVec()
    tokamak.current_dummy_vec = np.zeros(tokamak.n_coils)

    machine_description_data = getattr(tokamak, "_machine_description_data", None)
    if machine_description_data is not None:
        old_passive_coils = list(
            machine_description_data.get("passive_coils", []) or []
        )
        by_name = _index_passive_coils(old_passive_coils)
        if name in by_name:
            del old_passive_coils[by_name[name][0]]
        machine_description_data["passive_coils"] = old_passive_coils
        tokamak._machine_description_data = machine_description_data

    if hasattr(tokamak, "probes"):
        tokamak.probes.coils_dict = tokamak.coils_dict
        if hasattr(tokamak.probes, "coil_names"):
            tokamak.probes.coil_names = list(tokamak.coils_dict.keys())

    tokamak._last_machine_update_changed_coils = [name]
    tokamak._last_machine_update_topology_changed = True
    remove_tokamak_R_and_M_entry(tokamak, index)
    return tokamak


def add_active_coil(tokamak, coil_name, active_coil_data):
    """
    Add one new active coil/circuit to an existing tokamak object, in place.

    The new coil is inserted immediately before any passive structures, so
    that the active/passive partition of ``tokamak.coils_list`` (everything
    before ``tokamak.n_active_coils`` is active, everything after is passive)
    stays contiguous. Only the new coil's resistance/inductance matrix
    row/column are calculated; entries for existing coils are reused.

    Parameters
    ----------
    tokamak : Machine
        Existing tokamak object to update in place.
    coil_name : str
        Label for the new active coil/circuit.
    active_coil_data : dict
        Machine-description entry for the new active coil/circuit (same
        format as one entry of the active-coils dictionary).

    Returns
    -------
    Machine
        The same tokamak object, updated in place.

    Raises
    ------
    ValueError
        If a coil/passive-structure with the given label already exists.

    Notes
    -----
    This updates only machine-level state. If the machine is attached to an
    equilibrium, use
    :meth:`freegsnke.equilibrium_update.Equilibrium.add_active_coil` so that
    equilibrium-level Greens functions are extended as well. Existing
    nonlinear solver objects should be reinstantiated after a geometry change
    because they cache machine-dependent matrices and mode decompositions,
    whose dimensionality will itself have changed.
    """

    if hasattr(tokamak, "coil_order") and coil_name in tokamak.coil_order:
        raise ValueError(f"Tokamak already contains a coil label '{coil_name}'.")

    built_component, coil_metadata = build_active_coil_component(
        coil_name, active_coil_data
    )

    insert_index = tokamak.n_active_coils
    tokamak.coils.insert(insert_index, built_component)
    tokamak.coils_dict[coil_name] = coil_metadata
    tokamak.coils_list.insert(insert_index, coil_name)
    tokamak.n_active_coils += 1
    tokamak.n_coils += 1

    tokamak.coil_names = list(tokamak.getCurrents().keys())
    tokamak.coil_order = {}
    for i, coil in enumerate(tokamak.coil_names):
        tokamak.coil_order[coil] = i

    tokamak.getCurrentsVec()
    tokamak.current_dummy_vec = np.zeros(tokamak.n_coils)

    machine_description_data = getattr(tokamak, "_machine_description_data", None)
    if machine_description_data is None:
        machine_description_data = {}
    active_coils_data_dict = dict(
        machine_description_data.get("active_coils", {}) or {}
    )
    active_coils_data_dict[coil_name] = deepcopy(active_coil_data)
    machine_description_data["active_coils"] = active_coils_data_dict
    tokamak._machine_description_data = machine_description_data

    if hasattr(tokamak, "probes"):
        tokamak.probes.coils_dict = tokamak.coils_dict
        if hasattr(tokamak.probes, "coil_names"):
            tokamak.probes.coil_names = list(tokamak.coils_dict.keys())

    tokamak._last_machine_update_changed_coils = [coil_name]
    tokamak._last_machine_update_topology_changed = True
    insert_tokamak_R_and_M_entries(tokamak, insert_index, [coil_name])
    return tokamak


def remove_active_coil(tokamak, coil_name):
    """
    Remove one active coil/circuit from an existing tokamak object, in place.

    Only the removed coil's resistance/inductance matrix row/column are
    dropped; entries for the remaining coils are reused.

    Parameters
    ----------
    tokamak : Machine
        Existing tokamak object to update in place.
    coil_name : str
        Existing active coil/circuit label to remove.

    Returns
    -------
    Machine
        The same tokamak object, updated in place.

    Raises
    ------
    ValueError
        If ``coil_name`` is not an existing active coil/circuit label.

    Notes
    -----
    This updates only machine-level state. If the machine is attached to an
    equilibrium, use
    :meth:`freegsnke.equilibrium_update.Equilibrium.remove_active_coil` so
    that equilibrium-level Greens functions are shrunk as well. Existing
    nonlinear solver objects should be reinstantiated after a geometry change
    because they cache machine-dependent matrices and mode decompositions,
    whose dimensionality will itself have changed.
    """

    if not hasattr(tokamak, "coil_order") or coil_name not in tokamak.coil_order:
        raise ValueError(f"Tokamak does not contain active coil label '{coil_name}'.")
    if not tokamak.coils_dict.get(coil_name, {}).get("active", False):
        raise ValueError(f"Coil label '{coil_name}' is not an active coil/circuit.")

    index = tokamak.coil_order[coil_name]

    del tokamak.coils[index]
    del tokamak.coils_dict[coil_name]
    del tokamak.coils_list[index]
    tokamak.n_active_coils -= 1
    tokamak.n_coils -= 1

    tokamak.coil_names = list(tokamak.getCurrents().keys())
    tokamak.coil_order = {}
    for i, coil in enumerate(tokamak.coil_names):
        tokamak.coil_order[coil] = i

    tokamak.getCurrentsVec()
    tokamak.current_dummy_vec = np.zeros(tokamak.n_coils)

    machine_description_data = getattr(tokamak, "_machine_description_data", None)
    if machine_description_data is not None:
        active_coils_data_dict = dict(
            machine_description_data.get("active_coils", {}) or {}
        )
        active_coils_data_dict.pop(coil_name, None)
        machine_description_data["active_coils"] = active_coils_data_dict
        tokamak._machine_description_data = machine_description_data

    if hasattr(tokamak, "probes"):
        tokamak.probes.coils_dict = tokamak.coils_dict
        if hasattr(tokamak.probes, "coil_names"):
            tokamak.probes.coil_names = list(tokamak.coils_dict.keys())

    tokamak._last_machine_update_changed_coils = [coil_name]
    tokamak._last_machine_update_topology_changed = True
    remove_tokamak_R_and_M_entry(tokamak, index)
    return tokamak


def _reuse_unchanged_coil_components(tokamak, components, changed_coils):
    """
    Reuse coil objects and metadata for labels whose input description is unchanged.

    Reusing unchanged entries keeps cached matrices and Greens functions
    consistent with the actual coil objects, and avoids treating passive
    refinement noise as a real geometry change.

    Parameters
    ----------
    tokamak : Machine
        Existing tokamak object whose current coil objects and metadata are
        the candidates for reuse.
    components : dict
        Newly built components, as returned by :func:`build_tokamak_components`.
        Updated in place: entries for unchanged labels have their
        ``coil_circuits`` and ``coils_dict`` entries replaced with the
        corresponding objects from ``tokamak``.
    changed_coils : list
        Labels that should be treated as changed and therefore left untouched
        (i.e. not reused from ``tokamak``).

    Returns
    -------
    None
        ``components`` is modified in place.
    """

    changed = set(changed_coils)
    old_coils = dict(tokamak.coils)
    for i, label in enumerate(components["coils_list"]):
        if label in changed:
            continue
        if label in old_coils and label in tokamak.coils_dict:
            components["coil_circuits"][i] = (label, old_coils[label])
            components["coils_dict"][label] = tokamak.coils_dict[label]


def _changed_coil_labels(
    old_coils_dict,
    old_coils_list,
    new_coils_dict,
    new_coils_list,
    old_machine_description_data,
    new_machine_description_data,
):
    """
    Determine which coil labels changed between two machine descriptions.

    Raw machine-description data is preferred over rebuilt coil metadata so that
    unchanged passive structures are not marked as changed solely because their
    refinement was regenerated.

    Parameters
    ----------
    old_coils_dict : dict or None
        Existing ``tokamak.coils_dict``, or None if the tokamak has not been
        built with coil metadata before.
    old_coils_list : list or None
        Existing ``tokamak.coils_list``, or None if unavailable.
    new_coils_dict : dict
        Newly built ``coils_dict`` (from :func:`build_tokamak_components`).
    new_coils_list : list
        Newly built ``coils_list`` (from :func:`build_tokamak_components`).
    old_machine_description_data : dict or None
        Cached raw machine-description data (``active_coils``, ``passive_coils``,
        etc.) from the existing tokamak, or None if unavailable.
    new_machine_description_data : dict
        Raw machine-description data for the newly built components.

    Returns
    -------
    changed_coils : list
        Labels whose underlying description changed (or all labels, if the
        coil topology changed).
    topology_changed : bool
        True if the set/order of coil labels differs between the old and new
        descriptions (or no prior description exists), in which case
        ``changed_coils`` is simply ``new_coils_list``.
    """

    if old_coils_dict is None or old_coils_list is None:
        return list(new_coils_list), True
    if old_coils_list != new_coils_list:
        return list(new_coils_list), True

    if old_machine_description_data is None:
        return (
            _changed_coil_labels_from_metadata(
                old_coils_dict, old_coils_list, new_coils_dict
            ),
            False,
        )

    changed = []
    old_actives = old_machine_description_data.get("active_coils", {})
    new_actives = new_machine_description_data.get("active_coils", {})
    passive_data_changed = not _machine_description_values_equal(
        old_machine_description_data.get("passive_coils", []),
        new_machine_description_data.get("passive_coils", []),
    )

    for label in new_coils_list:
        if new_coils_dict[label].get("active", False):
            if not _machine_description_values_equal(
                old_actives.get(label), new_actives.get(label)
            ):
                changed.append(label)
        elif passive_data_changed:
            changed.append(label)

    return changed, False


def _changed_coil_labels_from_metadata(old_coils_dict, old_coils_list, new_coils_dict):
    """
    Fallback changed-label detection based on built coil metadata.

    Used when no cached raw machine-description data is available to compare
    against, so coil metadata dictionaries are compared directly instead.

    Parameters
    ----------
    old_coils_dict : dict
        Existing ``tokamak.coils_dict``.
    old_coils_list : list
        Existing ``tokamak.coils_list``, defining the labels to check.
    new_coils_dict : dict
        Newly built ``coils_dict`` to compare against.

    Returns
    -------
    list
        Labels whose metadata entry differs between ``old_coils_dict`` and
        ``new_coils_dict``.
    """

    changed = []
    for label in old_coils_list:
        if not _machine_description_values_equal(
            old_coils_dict.get(label), new_coils_dict.get(label)
        ):
            changed.append(label)
    return changed


def _machine_description_values_equal(old_value, new_value):
    """
    Recursively compare machine-description values, including numpy arrays.

    Handles the value types found in machine-description dictionaries (nested
    dicts, lists/tuples, numpy arrays, and plain scalars/strings), since a
    plain ``==`` comparison raises or gives the wrong answer for arrays and
    nested containers.

    Parameters
    ----------
    old_value : any
        Value from the existing/old machine description.
    new_value : any
        Value from the newly built machine description.

    Returns
    -------
    bool
        True if the two values are recursively equal, False otherwise.
    """

    if isinstance(old_value, dict) and isinstance(new_value, dict):
        if set(old_value.keys()) != set(new_value.keys()):
            return False
        return all(
            _machine_description_values_equal(old_value[key], new_value[key])
            for key in old_value
        )

    if isinstance(old_value, (list, tuple)) and isinstance(new_value, (list, tuple)):
        if len(old_value) != len(new_value):
            return False
        return all(
            _machine_description_values_equal(old_item, new_item)
            for old_item, new_item in zip(old_value, new_value)
        )

    if isinstance(old_value, np.ndarray) or isinstance(new_value, np.ndarray):
        try:
            return np.array_equal(np.asarray(old_value), np.asarray(new_value))
        except ValueError:
            return False

    try:
        return old_value == new_value
    except ValueError:
        return False


def update_tokamak(
    tokamak,
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
    Update an existing tokamak from direct data dictionaries or pickle paths.

    This is the functional equivalent of
    :meth:`freegsnke.machine_update.Machine.set_machine_description`. It accepts
    the same machine-description inputs as :func:`tokamak`, rebuilds the
    FreeGS4E coil/circuit objects and FreeGSNKE coil metadata, and recalculates
    the resistance and inductance matrices on the existing machine object.

    Parameters
    ----------
    tokamak : Machine
        Existing tokamak object to update in place.
    active_coils_data : dict, optional
        Dictionary containing the active coil description.
    passive_coils_data : list, optional
        List containing passive structure descriptions.
    limiter_data : list, optional
        List of limiter boundary points.
    wall_data : list, optional
        List of wall boundary points.
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
        Refinement mode for extended passive structures. Defaults to ``"G"``.
    preserve_currents : bool, optional
        If True, currents for labels that are present in both the old and new
        machine descriptions are copied onto the updated coil objects.

    Returns
    -------
    Machine
        The same tokamak object, updated in place.
    """

    components = build_tokamak_components(
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
    )
    apply_tokamak_components(
        tokamak,
        components,
        preserve_currents=preserve_currents,
        rebuild_R_and_M=True,
    )
    print("Tokamak updated.")
    return tokamak


def load_data_dicts(
    active_coils_data=None,
    passive_coils_data=None,
    limiter_data=None,
    wall_data=None,
    active_coils_path=None,
    passive_coils_path=None,
    limiter_path=None,
    wall_path=None,
):
    """
    Load the standarised input data required to build the tokamak machine.

    These dictionaries/lists/arrays can either be provided directly or loaded from pickle files.

    Parameters
    ----------
    active_coils_data : dict, optional
        Dictionary containing the active coil data.
    passive_coils_data : dict, optional
        Dictionary containing the passive structure data.
    limiter_data : dict, optional
        Dictionary containing the limiter data.
    wall_data : dict, optional
        Dictionary containing the wall data.
    active_coils_path : str, optional
        Path to the pickle file containing the active coil data.
    passive_coils_path : str, optional
        Path to the pickle file containing the passive structure data.
    limiter_path : str, optional
        Path to the pickle file containing the limiter data.
    wall_path : str, optional
        Path to the pickle file containing the wall data.

    Returns
    -------
    active_coils_data : dict
        Dictionary containing active coil data.
    passive_coils_data : dict
        Dictionary containing passive structure data.
    limiter_data : dict
        Dictionary containing the limiter data.
    wall_data : dict
        Dictionary containing the wall data.
    """

    # actives required
    if active_coils_data is not None and active_coils_path is not None:
        raise ValueError(
            "The user needs to provide only one of 'active_coils_data' or 'active_coils_path', not both."
        )
    elif active_coils_data is None and active_coils_path is None:
        raise ValueError(
            "The user needs to provide either 'active_coils_data' or 'active_coils_path'."
        )
    elif active_coils_path is not None:
        with open(active_coils_path, "rb") as f:
            active_coils_data = pickle.load(f)
            print("Active coils --> built from pickle file.")
    else:
        print("Active coils --> built from user-provided data.")

    # passives not strictly required
    if passive_coils_data is not None and passive_coils_path is not None:
        raise ValueError(
            "The user needs to provide only one of 'passive_coils_data' or 'passive_coils_path', not both."
        )
    elif passive_coils_data is None and passive_coils_path is None:
        passive_coils_data = []  # default to empty list
        print("Passive structures --> none provided.")
    elif passive_coils_path is not None:
        with open(passive_coils_path, "rb") as f:
            passive_coils_data = pickle.load(f)
            print("Passive structures --> built from pickle file.")
    else:
        print("Passive structures --> built from user-provided data.")

    # limiter required
    if limiter_data is not None and limiter_path is not None:
        raise ValueError(
            "The user needs to provide only one of 'limiter_data' or 'limiter_path', not both."
        )
    elif limiter_data is None and limiter_path is None:
        raise ValueError(
            "The user needs to provide either 'limiter_data' or 'limiter_path'."
        )
    elif limiter_path is not None:
        with open(limiter_path, "rb") as f:
            limiter_data = pickle.load(f)
            print("Limiter --> built from pickle file.")
    else:
        print("Limiter --> built from user-provided data.")

    # wall not strictly required
    if wall_data is not None and wall_path is not None:
        raise ValueError(
            "The user needs to provide only one of 'wall_data' or 'wall_path', not both."
        )
    elif wall_data is None and wall_path is None:
        wall_data = limiter_data  # default to the limiter
        print("Wall --> none provided, setting equal to limiter.")
    elif wall_path is not None:
        with open(wall_path, "rb") as f:
            wall_data = pickle.load(f)
            print("Wall --> built from pickle file.")
    else:
        print("Wall --> built from user-provided data.")

    return active_coils_data, passive_coils_data, limiter_data, wall_data


def build_actives(
    active_coils,
):
    """
    Build the coils (and any circuits) in FreeGSNKE using the MultiCoil and Circuit
    functionality from FreeGS4E.

    Parameters
    ----------
    active_coils : dict, optional
        Dictionary containing the active coil data.

    Returns
    -------
    coils : list
        List of coils and circuits to be ingested by FreeGSNKE/FreeGS4E.
    """

    # store list of all coils built
    coils = []

    # loop over all coils in dictionary
    for name in active_coils:

        # single coil (e.g a solenoid)
        if "R" in active_coils[name] or "Z" in active_coils[name]:
            try:
                # initialise Multicoil and set attributes
                multicoil = MultiCoil(active_coils[name]["R"], active_coils[name]["Z"])
                multicoil.dR = active_coils[name]["dR"]
                multicoil.dZ = active_coils[name]["dZ"]
                multicoil.resistivity = active_coils[name]["resistivity"]

                # add to list in its own Circuit
                coils.append(
                    (
                        name,
                        Circuit(
                            [
                                (
                                    name,
                                    multicoil,
                                    float(active_coils[name]["polarity"])
                                    * float(active_coils[name]["multiplier"]),
                                ),
                            ]
                        ),
                    ),
                )
            except:
                print(
                    f"Could not build the coil {active_coils[name]}, check its format."
                )

        # multiple coils linked in a circuit (e.g. an up-down pair of shaping coils)
        else:
            try:

                # create a circuit of coils
                circuit_list = []

                # loop over each coil in circuit
                for ind in active_coils[name]:

                    # initialise Multicoil and set attributes
                    multicoil = MultiCoil(
                        active_coils[name][ind]["R"], active_coils[name][ind]["Z"]
                    )
                    multicoil.dR = active_coils[name][ind]["dR"]
                    multicoil.dZ = active_coils[name][ind]["dZ"]
                    multicoil.resistivity = active_coils[name][ind]["resistivity"]

                    # add to coils in circuit
                    circuit_list.append(
                        (
                            name + ind,
                            multicoil,
                            float(active_coils[name][ind]["polarity"])
                            * float(active_coils[name][ind]["multiplier"]),
                        )
                    )

                # add circuit to list
                coils.append(
                    (
                        name,
                        Circuit(circuit_list),
                    )
                )

            except:
                print(
                    f"Could not build the coil {active_coils[name]}, check its format."
                )

    return coils


def build_passives(
    passive_coils,
    coil_circuits,
    coils_dict,
    coil_names,
    refine_mode,
):
    """
    Build the passive structures in FreeGSNKE using the PassiveStructure function.

    Parameters
    ----------
    passive_coils : dict
        Dictionary containing data for passive coils.
    coil_circuits : list
        List of coil circuit objects.
    coils_dict : dict
        Dictionary of coil data.
    coil_names : list
        List of circuit/coil names and passive structures.
    refine_mode : str, optional
        Choose the refinement mode for extended passive structures (input as polygons), by default
        'G' for 'grid' (use 'LH' for alternative mode using a Latin Hypercube implementation).

    Returns
    -------
    coil_circuits : list
        List of coil circuit objects.
    coils_dict : dict
        Dictionary of coil data.
    coil_names : list
        List of circuit/coil names and passive structures.
    """

    # parameters to set the refinement of extended passive structures
    # values are in number of individual filaments per m^2 (per area) and per m (per length)
    default_min_refine_per_area = 3e3
    default_min_refine_per_length = 200

    # loop over passive coils
    for i, coil in enumerate(passive_coils):

        # include name if provided, else use default
        try:
            name = coil["name"]
        except:
            name = f"passive_{i}"

        # add entry to list
        coil_names.append(name)

        # if vertices provided, build them as polygons
        if np.size(coil["R"]) > 1:

            # how much do we refine the polygons?
            try:
                min_refine_per_area = 1.0 * coil["min_refine_per_area"]
            except:
                min_refine_per_area = 1.0 * default_min_refine_per_area
            try:
                min_refine_per_length = 1.0 * coil["min_refine_per_length"]
            except:
                min_refine_per_length = 1.0 * default_min_refine_per_length

            # build the passive structure Polygon
            ps = PassiveStructure(
                R=coil["R"],
                Z=coil["Z"],
                min_refine_per_area=min_refine_per_area,
                min_refine_per_length=min_refine_per_length,
                refine_mode=refine_mode,
            )

            # add to circuits list
            coil_circuits.append(((name, ps)))

            # add coils_dict entry
            coils_dict[name] = {}
            coils_dict[name]["active"] = False
            coils_dict[name]["vertices"] = np.array((coil["R"], coil["Z"]))
            coils_dict[name]["coords"] = np.array(
                [ps.filaments[:, 0], ps.filaments[:, 1]]
            )
            coils_dict[name]["area"] = ps.area

            filament_size = (ps.area / len(ps.filaments)) ** 0.5
            coils_dict[name]["dR"] = filament_size
            coils_dict[name]["dZ"] = filament_size

            coils_dict[name]["polarity"] = np.array([1])
            coils_dict[name]["resistivity_over_area"] = (
                coil["resistivity"] / coils_dict[name]["area"]
            )
            # multiplier is used to distribute current over the passive structure
            coils_dict[name]["multiplier"] = np.array([1 / len(ps.filaments)])

        # if vertices not provided, build passive structure as individual filament
        else:
            coil_circuits.append(
                (
                    (
                        name,
                        Coil(
                            R=coil["R"],
                            Z=coil["Z"],
                            area=coil["dR"] * coil["dZ"],
                            control=False,
                        ),
                    )
                )
            )

            # add coils_dict entry
            coils_dict[name] = {}
            coils_dict[name]["active"] = False
            coils_dict[name]["coords"] = np.array((coil["R"], coil["Z"]))[:, np.newaxis]
            coils_dict[name]["dR"] = coil["dR"]
            coils_dict[name]["dZ"] = coil["dZ"]
            coils_dict[name]["polarity"] = np.array([1])
            coils_dict[name]["multiplier"] = np.array([1])
            coils_dict[name]["resistivity_over_area"] = coil["resistivity"] / (
                coil["dR"] * coil["dZ"]
            )

    return coil_circuits, coils_dict, coil_names


def build_active_coil_dict(active_coils):
    """
    Create vectorised version of the active coil properties in a dictionary for use
    throughout FreeGSNKE.

    Parameters
    ----------
    active_coils : dict, optional
        Dictionary containing the active coil data.

    Returns
    -------
    coils_dict : dict
        Dictionary with vectorised properties of all active coils.
    """

    # initialise
    coils_dict = {}

    # loop over each entry
    for i, name in enumerate(active_coils):

        # single coil (e.g a solenoid)
        if "R" in active_coils[name] or "Z" in active_coils[name]:
            try:
                coils_dict[name] = {}
                coils_dict[name]["active"] = True
                coils_dict[name]["coords"] = np.array(
                    [active_coils[name]["R"], active_coils[name]["Z"]]
                )
                coils_dict[name]["polarity"] = np.array(
                    [active_coils[name]["polarity"]] * len(active_coils[name]["R"])
                )
                coils_dict[name]["dR"] = active_coils[name]["dR"]
                coils_dict[name]["dZ"] = active_coils[name]["dZ"]
                coils_dict[name]["resistivity_over_area"] = active_coils[name][
                    "resistivity"
                ] / (active_coils[name]["dR"] * active_coils[name]["dZ"])
                coils_dict[name]["multiplier"] = np.array(
                    [active_coils[name]["multiplier"]] * len(active_coils[name]["R"])
                )

            except:
                print(
                    f"Could not build the coil {active_coils[name]}, check its format."
                )

        # multiple coils linked in a circuit (e.g. an up-down pair of shaping coils)
        else:
            try:
                coils_dict[name] = {}
                coils_dict[name]["active"] = True

                coords_R = []
                for ind in active_coils[name].keys():
                    coords_R.extend(active_coils[name][ind]["R"])

                coords_Z = []
                for ind in active_coils[name].keys():
                    coords_Z.extend(active_coils[name][ind]["Z"])
                coils_dict[name]["coords"] = np.array([coords_R, coords_Z])

                polarity = []
                for ind in active_coils[name].keys():
                    polarity.extend(
                        [active_coils[name][ind]["polarity"]]
                        * len(active_coils[name][ind]["R"])
                    )
                coils_dict[name]["polarity"] = np.array(polarity)

                multiplier = []
                for ind in active_coils[name].keys():
                    multiplier.extend(
                        [active_coils[name][ind]["multiplier"]]
                        * len(active_coils[name][ind]["R"])
                    )
                coils_dict[name]["multiplier"] = np.array(multiplier)

                coils_dict[name]["dR"] = active_coils[name][
                    list(active_coils[name].keys())[0]
                ]["dR"]
                coils_dict[name]["dZ"] = active_coils[name][
                    list(active_coils[name].keys())[0]
                ]["dZ"]

                coils_dict[name]["resistivity_over_area"] = active_coils[name][
                    list(active_coils[name].keys())[0]
                ]["resistivity"] / (coils_dict[name]["dR"] * coils_dict[name]["dZ"])

            except:
                print(
                    f"Could not build the coil {active_coils[name]}, check its format."
                )

    return coils_dict


def copy_tokamak(tokamak: Machine):
    """
    Create a copy of a tokamak Machine object with controlled deep/shallow copying.

    This function duplicates the core Machine object while carefully controlling
    which internal structures are shared and which are copied. It is intended to
    produce a new independent tokamak instance suitable for simulation branching
    or parameter variation.

    Copy behaviour:
        - Uses `tokamak.copy()` as the base object copy.
        - Performs shallow copies of coil container structures.
        - Copies selected numerical arrays using `copy_into`.
        - Shares some reference-type attributes (e.g. probes).

    Parameters
    ----------
    tokamak : Machine
        Original tokamak machine object to be copied.

    Returns
    -------
    Machine
        New tokamak instance with copied geometry and selected attributes.

    Notes
    -----
    - `coils_dict` is shallow-copied (dictionary structure copied, values shared).
    - `coils_list` is shallow-copied (list container copied, objects shared).
    - `coil_resist`, `coil_self_ind`, and `current_vec` are copied using
      `copy_into` (typically deep or array-level copy depending on implementation).
    - Cached machine-description data and update metadata are copied so direct
      machine-description updates remain available on the new object.
    - `probes` is shared by reference (not copied).
    - This function does not guarantee full independence of all mutable
      substructures unless `copy_into` enforces deep copying.
    """
    new_tokamak = tokamak.copy()

    new_tokamak.coils_dict = tokamak.coils_dict.copy()
    new_tokamak.coils_list = tokamak.coils_list[::]
    new_tokamak.n_active_coils = tokamak.n_active_coils
    new_tokamak.n_passive_coils = tokamak.n_passive_coils
    new_tokamak.n_coils = tokamak.n_coils
    new_tokamak.limiter = tokamak.limiter

    copy_into(tokamak, new_tokamak, "coil_resist", mutable=True, strict=False)
    copy_into(tokamak, new_tokamak, "coil_self_ind", mutable=True, strict=False)
    copy_into(tokamak, new_tokamak, "current_vec", mutable=True, strict=False)
    copy_into(
        tokamak,
        new_tokamak,
        "_machine_description_data",
        mutable=True,
        strict=False,
        allow_deepcopy=True,
    )
    copy_into(
        tokamak,
        new_tokamak,
        "_last_machine_update_changed_coils",
        mutable=True,
        strict=False,
        allow_deepcopy=True,
    )
    copy_into(
        tokamak,
        new_tokamak,
        "_last_machine_update_topology_changed",
        mutable=True,
        strict=False,
    )

    # add probe object attribute to tokamak (not strictly required)
    new_tokamak.probes = tokamak.probes

    return new_tokamak


if __name__ == "__main__":
    for coil_name in active_coils:
        print([pol for pol in active_coils[coil_name]])
