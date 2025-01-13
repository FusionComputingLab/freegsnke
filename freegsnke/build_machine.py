"""
Routines to build the FreeGSNKE Machine Object from the provided machine description.

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
from freegs4e.machine import Circuit, Solenoid, Wall
from freegs4e.multi_coil import MultiCoil

from .machine_update import Machine
from .magnetic_probes import Probes
from .passive_structure import PassiveStructure
from .refine_passive import generate_refinement

# these parameters set the refinement of extended passive structures
# values are in number of individual filaments per m^2 (per area) and per m (per length)
# please amend accordingly to the needs of your use case
default_min_refine_per_area = 3e3
default_min_refine_per_length = 200

passive_coils_path = os.environ.get("PASSIVE_COILS_PATH", None)
if passive_coils_path is None:
    raise ValueError("PASSIVE_COILS_PATH environment variable not set.")

active_coils_path = os.environ.get("ACTIVE_COILS_PATH", None)
if active_coils_path is None:
    raise ValueError("ACTIVE_COILS_PATH environment variable not set.")

wall_path = os.environ.get("WALL_PATH", None)
if wall_path is None:
    raise ValueError("WALL_PATH environment variable not set.")

limiter_path = os.environ.get("LIMITER_PATH", None)
if limiter_path is None:
    raise ValueError("LIMITER_PATH environment variable not set.")

with open(passive_coils_path, "rb") as f:
    passive_coils = pickle.load(f)

with open(active_coils_path, "rb") as f:
    active_coils = pickle.load(f)

with open(wall_path, "rb") as f:
    wall = pickle.load(f)

with open(limiter_path, "rb") as f:
    limiter = pickle.load(f)

if "Solenoid" not in active_coils:
    print("No coil named Solenoid among the active coils.")


def tokamak(refine_mode="G"):
    """Builds the Machine object using the provided geometry info.
    Using MultiCoil to represent coils with different locations for each strand.
    Passive structures are handled by the dedicated object.

    Parameters
    ----------
    refine_mode : str, optional
        refinement mode for extended passive structures (inputted as polygons), by default 'G' for 'grid'
        Use 'LH' for alternative mode using a Latin Hypercube implementation.

    Returns
    -------
    FreeGSNKE tokamak machine object.
    """
    coils = []
    for coil_name in active_coils:
        if coil_name == "Solenoid":
            # Add the solenoid if any
            multicoil = MultiCoil(
                active_coils["Solenoid"]["R"], active_coils["Solenoid"]["Z"]
            )
            multicoil.dR = active_coils["Solenoid"]["dR"]
            multicoil.dZ = active_coils["Solenoid"]["dZ"]
            coils.append(
                (
                    "Solenoid",
                    Circuit(
                        [
                            (
                                "Solenoid",
                                multicoil,
                                float(active_coils["Solenoid"]["polarity"])
                                * float(active_coils["Solenoid"]["multiplier"]),
                            ),
                        ]
                    ),
                ),
            )

        else:
            # Add active coils
            circuit_list = []
            for ind in active_coils[coil_name]:
                multicoil = MultiCoil(
                    active_coils[coil_name][ind]["R"],
                    active_coils[coil_name][ind]["Z"],
                )
                multicoil.dR = active_coils[coil_name][ind]["dR"]
                multicoil.dZ = active_coils[coil_name][ind]["dZ"]
                circuit_list.append(
                    (
                        coil_name + ind,
                        multicoil,
                        float(active_coils[coil_name][ind]["polarity"])
                        * float(active_coils[coil_name][ind]["multiplier"]),
                    )
                )
            coils.append(
                (
                    coil_name,
                    Circuit(circuit_list),
                )
            )

    coils_dict = build_active_coil_dict(active_coils=active_coils)
    coils_list = list(coils_dict.keys())

    # Add passive coils
    for i, coil in enumerate(passive_coils):
        # include name if provided
        try:
            coil_name = coil["name"]
        except:
            coil_name = f"passive_{i}"
        coils_list.append(coil_name)

        if np.size(coil["R"]) > 1:
            # refine if vertices provided

            # keep refinement filaments grouped
            # i.e. use passive structure object
            try:
                min_refine_per_area = 1.0 * coil["min_refine_per_area"]
            except:
                min_refine_per_area = 1.0 * default_min_refine_per_area
            try:
                min_refine_per_length = 1.0 * coil["min_refine_per_length"]
            except:
                min_refine_per_length = 1.0 * default_min_refine_per_length

            ps = PassiveStructure(
                R=coil["R"],
                Z=coil["Z"],
                min_refine_per_area=min_refine_per_area,
                min_refine_per_length=min_refine_per_length,
            )
            coils.append(((coil_name, ps)))

            # add coil_dict entry
            coils_dict[coil_name] = {}
            coils_dict[coil_name]["active"] = False
            coils_dict[coil_name]["vertices"] = np.array((coil["R"], coil["Z"]))
            coils_dict[coil_name]["coords"] = np.array(
                [ps.filaments[:, 0], ps.filaments[:, 1]]
            )
            coils_dict[coil_name]["area"] = ps.area
            filament_size = (ps.area / len(ps.filaments)) ** 0.5
            coils_dict[coil_name]["dR"] = filament_size
            coils_dict[coil_name]["dZ"] = filament_size
            coils_dict[coil_name]["polarity"] = np.array([1])
            # here 'multiplier' is used to normalise the green functions,
            # this is needed because currents are distributed over the passive structure
            coils_dict[coil_name]["multiplier"] = np.array([1 / len(ps.filaments)])
            # this is resistivity divided by area
            coils_dict[coil_name]["resistivity_over_area"] = (
                coil["resistivity"] / coils_dict[coil_name]["area"]
            )

        else:
            # passive structure is not refined, just an individual filament
            coils.append(
                (
                    (
                        coil_name,
                        Coil(
                            R=coil["R"],
                            Z=coil["Z"],
                            area=coil["dR"] * coil["dZ"],
                            control=False,
                        ),
                    )
                )
            )
            # add coil_dict entry
            coils_dict[coil_name] = {}
            coils_dict[coil_name]["active"] = False
            coils_dict[coil_name]["coords"] = np.array((coil["R"], coil["Z"]))[
                :, np.newaxis
            ]
            coils_dict[coil_name]["dR"] = coil["dR"]
            coils_dict[coil_name]["dZ"] = coil["dZ"]
            coils_dict[coil_name]["polarity"] = np.array([1])
            coils_dict[coil_name]["multiplier"] = np.array([1])
            # this is resistivity divided by area
            coils_dict[coil_name]["resistivity_over_area"] = coil["resistivity"] / (
                coil["dR"] * coil["dZ"]
            )

    # Add walls
    r_wall = [entry["R"] for entry in wall]
    z_wall = [entry["Z"] for entry in wall]

    # Add limiter
    r_limiter = [entry["R"] for entry in limiter]
    z_limiter = [entry["Z"] for entry in limiter]

    tokamak_machine = Machine(
        coils, wall=Wall(r_wall, z_wall), limiter=Wall(r_limiter, z_limiter)
    )

    tokamak_machine.coils_dict = coils_dict
    tokamak_machine.coils_list = coils_list

    # Number of active coils
    tokamak_machine.n_active_coils = len(active_coils)
    # Total number of coils
    tokamak_machine.n_coils = len(list(coils_dict.keys()))

    # Save coils_dict
    machine_path = os.path.join(
        os.path.split(active_coils_path)[0], "machine_data.pickle"
    )
    with open(machine_path, "wb") as f:
        pickle.dump(coils_dict, f)

    # add probe object attribute to tokamak
    tokamak_machine.probes = Probes(coils_dict)

    return tokamak_machine


def build_active_coil_dict(active_coils):
    """Adds vectorised properties of all active coils to a dictionary for use throughout FreeGSNKE.

    Parameters
    ----------
    active_coils : dictionary
        input dictionary, user defined with properties of the active coils

    Returns
    -------
    dictionary
        includes vectorised properties of all active coils
    """

    coils_dict = {}
    for i, coil_name in enumerate(active_coils):
        if coil_name == "Solenoid":
            coils_dict[coil_name] = {}
            coils_dict[coil_name]["active"] = True
            coils_dict[coil_name]["coords"] = np.array(
                [active_coils[coil_name]["R"], active_coils[coil_name]["Z"]]
            )
            coils_dict[coil_name]["polarity"] = np.array(
                [active_coils[coil_name]["polarity"]]
                * len(active_coils[coil_name]["R"])
            )
            coils_dict[coil_name]["dR"] = active_coils[coil_name]["dR"]
            coils_dict[coil_name]["dZ"] = active_coils[coil_name]["dZ"]
            # this is resistivity divided by area
            coils_dict[coil_name]["resistivity_over_area"] = active_coils[coil_name][
                "resistivity"
            ] / (active_coils[coil_name]["dR"] * active_coils[coil_name]["dZ"])
            coils_dict[coil_name]["multiplier"] = np.array(
                [active_coils[coil_name]["multiplier"]]
                * len(active_coils[coil_name]["R"])
            )
            continue

        coils_dict[coil_name] = {}
        coils_dict[coil_name]["active"] = True

        coords_R = []
        for ind in active_coils[coil_name].keys():
            coords_R.extend(active_coils[coil_name][ind]["R"])

        coords_Z = []
        for ind in active_coils[coil_name].keys():
            coords_Z.extend(active_coils[coil_name][ind]["Z"])
        coils_dict[coil_name]["coords"] = np.array([coords_R, coords_Z])

        polarity = []
        for ind in active_coils[coil_name].keys():
            polarity.extend(
                [active_coils[coil_name][ind]["polarity"]]
                * len(active_coils[coil_name][ind]["R"])
            )
        coils_dict[coil_name]["polarity"] = np.array(polarity)

        multiplier = []
        for ind in active_coils[coil_name].keys():
            multiplier.extend(
                [active_coils[coil_name][ind]["multiplier"]]
                * len(active_coils[coil_name][ind]["R"])
            )
        coils_dict[coil_name]["multiplier"] = np.array(multiplier)

        coils_dict[coil_name]["dR"] = active_coils[coil_name][
            list(active_coils[coil_name].keys())[0]
        ]["dR"]
        coils_dict[coil_name]["dZ"] = active_coils[coil_name][
            list(active_coils[coil_name].keys())[0]
        ]["dZ"]

        # this is resistivity divided by area
        coils_dict[coil_name]["resistivity_over_area"] = active_coils[coil_name][
            list(active_coils[coil_name].keys())[0]
        ]["resistivity"] / (coils_dict[coil_name]["dR"] * coils_dict[coil_name]["dZ"])

    return coils_dict


if __name__ == "__main__":
    for coil_name in active_coils:
        print([pol for pol in active_coils[coil_name]])
