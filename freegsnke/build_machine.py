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

import numpy as np
from freegs4e.coil import Coil
from freegs4e.machine import Circuit, Wall
from freegs4e.multi_coil import MultiCoil

from .machine_config import build_tokamak_R_and_M
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

    # build the tokamak
    tokamak = Machine(
        coil_circuits, wall=Wall(r_wall, z_wall), limiter=Wall(r_limiter, z_limiter)
    )

    # store some additional data
    tokamak.coils_dict = coils_dict
    tokamak.coils_list = coil_names
    tokamak.n_active_coils = n_active_coils
    tokamak.n_passive_coils = n_passive_coils
    tokamak.n_coils = n_active_coils + n_passive_coils

    # add probe object attribute to tokamak (not strictly required)
    tokamak.probes = Probes(
        coils_dict=coils_dict,
        magnetic_probe_data=magnetic_probe_data,
        magnetic_probe_path=magnetic_probe_path,
    )

    # build the R and M matrices in place (in tokamak)
    build_tokamak_R_and_M(tokamak)

    print("Tokamak built.")

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
        List of circuit\coil names and passive structures.
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
        List of circuit\coil names and passive structures.
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
    new_tokamak = tokamak.copy()

    new_tokamak.coils_dict = tokamak.coils_dict.copy()
    new_tokamak.coils_list = tokamak.coils_list[::]
    new_tokamak.n_active_coils = tokamak.n_active_coils
    new_tokamak.n_passive_coils = tokamak.n_passive_coils
    new_tokamak.n_coils = tokamak.n_coils

    # add probe object attribute to tokamak (not strictly required)
    new_tokamak.probes = tokamak.probes

    return new_tokamak


if __name__ == "__main__":
    for coil_name in active_coils:
        print([pol for pol in active_coils[coil_name]])
