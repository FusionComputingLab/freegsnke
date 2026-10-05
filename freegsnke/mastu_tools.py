"""
Contains various functions required to load MAST-U experimental shot data, ready for 
simulation in FreeGSNKE. Also contains additional functions that may be of use in the
simulations themselves. 

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

from __future__ import annotations

import math
import pickle
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pyuda
import scipy as sp
import shapely as sh
from freegs4e import critical
from numpy import abs, argmax, clip, linspace, pi

# --------------------------------
# EXTRACTING EFIT++ DATA


def get_machine_data(
    save_path: str | None = None,
    shot: int = 45425,
    split_passives: bool = True,
) -> None:
    """
    This functions builds the active coil, passive structure, wall, and limiter machine description pickle
    files for MAST-U (for a given shot number) required by FreeGSNKE.

    Parameters
    ----------
    save_path : str, optional
        Path in which to save the machine pickle files.
    shot : int, optional
        MAST-U shot number.
    split_passives : bool, optional
        If True, we model the passive structures as parallelograms (recommended), if False, we
        model as point current sources.

    Returns
    -------
    None
        Builds pickle files for the machine description in the `save_path` directory.
    """

    # path required
    if save_path is None:
        raise ValueError(
            "'save_path' cannot be None. Please provide a valid path to save the machine data."
        )

    # set up pyUDA client
    client = pyuda.Client()

    # store data in dictionary form
    data = {}

    # limiter structure
    limiter = client.geometry("/limiter/efit", shot)
    data["geometry_limiter"] = dict(r=limiter.data.R, z=limiter.data.Z)

    # active poloidal field coil geometry data
    pfcoil = client.geometry("/magnetics/pfcoil", shot)
    dict2 = {}
    for child in pfcoil.data.children:
        dict1 = {}
        for grandchild in child.children:
            dict0 = None
            try:
                material = grandchild.material
                coordinates = grandchild.children[1]
                r = coordinates.centreR
                z = coordinates.centreZ
                dr = coordinates.dR
                dz = coordinates.dZ
                turns = coordinates.effectiveTurnCount
                dict0 = dict(r=r, z=z, dr=dr, dz=dz, turns=turns)
            except AttributeError as err:
                print(err)
            dict2[child.name] = dict0
    data["geometry_pfcoil"] = dict2

    # passive structure geometry data
    passive = client.geometry("/passive/efit", shot)
    dict2 = {}
    for child in passive.data.children:
        dict1 = None
        try:
            coordinates = child.children[0]
            r = coordinates.centreR
            z = coordinates.centreZ
            dr = coordinates.dR
            dz = coordinates.dZ
            ang1 = coordinates.shapeAngle1
            ang2 = coordinates.shapeAngle2
            rho = coordinates.resistivity
            dict1 = dict(r=r, z=z, dr=dr, dz=dz, ang1=ang1, ang2=ang2, rho=rho)
            try:
                efitGroup = coordinates.efitGroup
                elementLabels = coordinates.elementLabels
                dict1["efitGroup"] = efitGroup
                dict1["elementLabels"] = elementLabels
            except AttributeError as err:
                pass
        # not everything is in a group, for example the coil cases (not grouped) and tiles (not used)
        except AttributeError as err:
            print(err)
        if dict1 is not None:
            dict2[child.name] = dict1
    data["geometry_passive"] = dict2

    # magnetic probe geometry data (fluxloop and pickups)

    # efit fluxloop data
    flux_names = client.get("/epm/input/constraints/fluxloops/shortname", shot).data
    flux_r = client.get("/epm/input/constraints/fluxloops/rvalues", shot).data
    flux_z = client.get("/epm/input/constraints/fluxloops/zvalues", shot).data
    data["fluxloops"] = dict(names=flux_names, r=flux_r, z=flux_z)

    # efit pickup coil data
    pickup_names = client.get(
        "/epm/input/constraints/magneticprobes/shortname", shot
    ).data
    pickup_r = client.get("/epm/input/constraints/magneticprobes/rcentre", shot).data
    pickup_z = client.get("/epm/input/constraints/magneticprobes/zcentre", shot).data
    pickup_pol_angle = client.get(
        "/epm/input/constraints/magneticprobes/poloidalOrientation", shot
    ).data
    pickup_tor_angle = client.get(
        "/epm/input/constraints/magneticprobes/toroidalangle", shot
    ).data
    data["pickups"] = dict(
        names=pickup_names,
        r=pickup_r,
        z=pickup_z,
        pol_ang=pickup_pol_angle,
        tor_ang=pickup_tor_angle,
    )

    # NOW BUILD THE PICKLE FILES FROM THE DATA IN THE ABOVE DICTIONARY

    # default global machine quantities that may not be present in UDA data
    eta_copper = 1.55e-8  # resistivity in Ohm*m, for active coils

    # ------------
    # ACTIVE COILS
    active_coils_uda = data["geometry_pfcoil"]

    # extract data into required form
    active_coils = {}

    # coil definitions (do not modify)
    p1 = {
        "R": np.hstack(
            (active_coils_uda["p1_inner"]["r"], active_coils_uda["p1_outer"]["r"])
        ),
        "Z": np.hstack(
            (active_coils_uda["p1_inner"]["z"], active_coils_uda["p1_outer"]["z"])
        ),
        "dR": np.mean(
            np.hstack(
                (active_coils_uda["p1_inner"]["dr"], active_coils_uda["p1_outer"]["dr"])
            )
        ),
        "dZ": np.mean(
            np.hstack(
                (active_coils_uda["p1_inner"]["dz"], active_coils_uda["p1_outer"]["dz"])
            )
        ),
        "polarity": 1,
        "resistivity": eta_copper,
        "multiplier": 0.5,
    }

    px_upper = {
        "R": active_coils_uda["px_upper"]["r"],
        "Z": active_coils_uda["px_upper"]["z"],
        "dR": np.mean(active_coils_uda["px_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["px_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    px_lower = {
        "R": active_coils_uda["px_lower"]["r"],
        "Z": active_coils_uda["px_lower"]["z"],
        "dR": np.mean(active_coils_uda["px_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["px_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d1_upper = {
        "R": active_coils_uda["d1_upper"]["r"],
        "Z": active_coils_uda["d1_upper"]["z"],
        "dR": np.mean(active_coils_uda["d1_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["d1_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d1_lower = {
        "R": active_coils_uda["d1_lower"]["r"],
        "Z": active_coils_uda["d1_lower"]["z"],
        "dR": np.mean(active_coils_uda["d1_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["d1_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d2_upper = {
        "R": active_coils_uda["d2_upper"]["r"],
        "Z": active_coils_uda["d2_upper"]["z"],
        "dR": np.mean(active_coils_uda["d2_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["d2_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d2_lower = {
        "R": active_coils_uda["d2_lower"]["r"],
        "Z": active_coils_uda["d2_lower"]["z"],
        "dR": np.mean(active_coils_uda["d2_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["d2_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d3_upper = {
        "R": active_coils_uda["d3_upper"]["r"],
        "Z": active_coils_uda["d3_upper"]["z"],
        "dR": np.mean(active_coils_uda["d3_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["d3_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d3_lower = {
        "R": active_coils_uda["d3_lower"]["r"],
        "Z": active_coils_uda["d3_lower"]["z"],
        "dR": np.mean(active_coils_uda["d3_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["d3_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    dp_upper = {
        "R": active_coils_uda["dp_upper"]["r"],
        "Z": active_coils_uda["dp_upper"]["z"],
        "dR": np.mean(active_coils_uda["dp_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["dp_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    dp_lower = {
        "R": active_coils_uda["dp_lower"]["r"],
        "Z": active_coils_uda["dp_lower"]["z"],
        "dR": np.mean(active_coils_uda["dp_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["dp_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d5_upper = {
        "R": active_coils_uda["d5_upper"]["r"],
        "Z": active_coils_uda["d5_upper"]["z"],
        "dR": np.mean(active_coils_uda["d5_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["d5_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d5_lower = {
        "R": active_coils_uda["d5_lower"]["r"],
        "Z": active_coils_uda["d5_lower"]["z"],
        "dR": np.mean(active_coils_uda["d5_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["d5_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d6_upper = {
        "R": active_coils_uda["d6_upper"]["r"],
        "Z": active_coils_uda["d6_upper"]["z"],
        "dR": np.mean(active_coils_uda["d6_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["d6_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d6_lower = {
        "R": active_coils_uda["d6_lower"]["r"],
        "Z": active_coils_uda["d6_lower"]["z"],
        "dR": np.mean(active_coils_uda["d6_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["d6_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d7_upper = {
        "R": active_coils_uda["d7_upper"]["r"],
        "Z": active_coils_uda["d7_upper"]["z"],
        "dR": np.mean(active_coils_uda["d7_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["d7_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    d7_lower = {
        "R": active_coils_uda["d7_lower"]["r"],
        "Z": active_coils_uda["d7_lower"]["z"],
        "dR": np.mean(active_coils_uda["d7_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["d7_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    p4_upper = {
        "R": active_coils_uda["p4_upper"]["r"],
        "Z": active_coils_uda["p4_upper"]["z"],
        "dR": np.mean(active_coils_uda["p4_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["p4_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    p4_lower = {
        "R": active_coils_uda["p4_lower"]["r"],
        "Z": active_coils_uda["p4_lower"]["z"],
        "dR": np.mean(active_coils_uda["p4_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["p4_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    p5_upper = {
        "R": active_coils_uda["p5_upper"]["r"],
        "Z": active_coils_uda["p5_upper"]["z"],
        "dR": np.mean(active_coils_uda["p5_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["p5_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    p5_lower = {
        "R": active_coils_uda["p5_lower"]["r"],
        "Z": active_coils_uda["p5_lower"]["z"],
        "dR": np.mean(active_coils_uda["p5_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["p5_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    pc = {
        "R": active_coils_uda["pc"]["r"],
        "Z": active_coils_uda["pc"]["z"],
        "dR": np.mean(active_coils_uda["pc"]["dr"]),
        "dZ": np.mean(active_coils_uda["pc"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 0.5,
    }

    p6_upper = {
        "R": active_coils_uda["p6_upper"]["r"],
        "Z": active_coils_uda["p6_upper"]["z"],
        "dR": np.mean(active_coils_uda["p6_upper"]["dr"]),
        "dZ": np.mean(active_coils_uda["p6_upper"]["dz"]),
        "resistivity": eta_copper,
        "polarity": 1,
        "multiplier": 1,
    }

    # note the reversed polarity here
    p6_lower = {
        "R": active_coils_uda["p6_lower"]["r"],
        "Z": active_coils_uda["p6_lower"]["z"],
        "dR": np.mean(active_coils_uda["p6_lower"]["dr"]),
        "dZ": np.mean(active_coils_uda["p6_lower"]["dz"]),
        "resistivity": eta_copper,
        "polarity": -1,
        "multiplier": 1,
    }

    # define symmetric (up/down coils linked in circuits) active coils dictionary
    active_coils = {}

    active_coils["p1"] = p1

    active_coils["p4"] = {}
    active_coils["p4"]["1"] = p4_upper
    active_coils["p4"]["2"] = p4_lower

    active_coils["p5"] = {}
    active_coils["p5"]["1"] = p5_upper
    active_coils["p5"]["2"] = p5_lower

    active_coils["px"] = {}
    active_coils["px"]["1"] = px_upper
    active_coils["px"]["2"] = px_lower

    active_coils["d1"] = {}
    active_coils["d1"]["1"] = d1_upper
    active_coils["d1"]["2"] = d1_lower

    active_coils["d2"] = {}
    active_coils["d2"]["1"] = d2_upper
    active_coils["d2"]["2"] = d2_lower

    active_coils["d3"] = {}
    active_coils["d3"]["1"] = d3_upper
    active_coils["d3"]["2"] = d3_lower

    active_coils["d5"] = {}
    active_coils["d5"]["1"] = d5_upper
    active_coils["d5"]["2"] = d5_lower

    active_coils["d6"] = {}
    active_coils["d6"]["1"] = d6_upper
    active_coils["d6"]["2"] = d6_lower

    active_coils["d7"] = {}
    active_coils["d7"]["1"] = d7_upper
    active_coils["d7"]["2"] = d7_lower

    active_coils["dp"] = {}
    active_coils["dp"]["1"] = dp_upper
    active_coils["dp"]["2"] = dp_lower

    active_coils["pc"] = pc

    active_coils["p6"] = {}
    active_coils["p6"]["1"] = p6_upper
    active_coils["p6"]["2"] = p6_lower

    # save data: this pickle file can be used when a symmetric MAST-U machine
    # description is required.
    pickle.dump(active_coils, open(f"{save_path}/MAST-U_active_coils.pickle", "wb"))

    # define non-symmetric (up/down coils not linked in circuits) active coils dictionary
    active_coils_nonsym = {}

    active_coils_nonsym["p1"] = p1

    active_coils_nonsym["p4_upper"] = {}
    active_coils_nonsym["p4_upper"]["1"] = p4_upper
    active_coils_nonsym["p4_lower"] = {}
    active_coils_nonsym["p4_lower"]["1"] = p4_lower

    active_coils_nonsym["p5_upper"] = {}
    active_coils_nonsym["p5_upper"]["1"] = p5_upper
    active_coils_nonsym["p5_lower"] = {}
    active_coils_nonsym["p5_lower"]["1"] = p5_lower

    active_coils_nonsym["px_upper"] = {}
    active_coils_nonsym["px_upper"]["1"] = px_upper
    active_coils_nonsym["px_lower"] = {}
    active_coils_nonsym["px_lower"]["1"] = px_lower

    active_coils_nonsym["d1_upper"] = {}
    active_coils_nonsym["d1_upper"]["1"] = d1_upper
    active_coils_nonsym["d1_lower"] = {}
    active_coils_nonsym["d1_lower"]["1"] = d1_lower

    active_coils_nonsym["d2_upper"] = {}
    active_coils_nonsym["d2_upper"]["1"] = d2_upper
    active_coils_nonsym["d2_lower"] = {}
    active_coils_nonsym["d2_lower"]["1"] = d2_lower

    active_coils_nonsym["d3_upper"] = {}
    active_coils_nonsym["d3_upper"]["1"] = d3_upper
    active_coils_nonsym["d3_lower"] = {}
    active_coils_nonsym["d3_lower"]["1"] = d3_lower

    active_coils_nonsym["d5_upper"] = {}
    active_coils_nonsym["d5_upper"]["1"] = d5_upper
    active_coils_nonsym["d5_lower"] = {}
    active_coils_nonsym["d5_lower"]["1"] = d5_lower

    active_coils_nonsym["d6_upper"] = {}
    active_coils_nonsym["d6_upper"]["1"] = d6_upper
    active_coils_nonsym["d6_lower"] = {}
    active_coils_nonsym["d6_lower"]["1"] = d6_lower

    active_coils_nonsym["d7_upper"] = {}
    active_coils_nonsym["d7_upper"]["1"] = d7_upper
    active_coils_nonsym["d7_lower"] = {}
    active_coils_nonsym["d7_lower"]["1"] = d7_lower

    active_coils_nonsym["dp_upper"] = {}
    active_coils_nonsym["dp_upper"]["1"] = dp_upper
    active_coils_nonsym["dp_lower"] = {}
    active_coils_nonsym["dp_lower"]["1"] = dp_lower

    active_coils_nonsym["pc"] = pc

    active_coils_nonsym["p6_upper"] = {}
    active_coils_nonsym["p6_upper"]["1"] = p6_upper
    active_coils_nonsym["p6_lower"] = {}
    active_coils_nonsym["p6_lower"]["1"] = p6_lower
    active_coils_nonsym["p6_lower"]["1"]["polarity"] = 1

    # save data: this pickle file can be used when a non-symmetric MAST-U machine
    # description is required.
    pickle.dump(
        active_coils_nonsym,
        open(f"{save_path}/MAST-U_active_coils_nonsym.pickle", "wb"),
    )

    # ------------
    # LIMITER/WALL
    limiter_uda = data["geometry_limiter"]

    # extract data into required form
    limiter = []
    for i in range(len(limiter_uda["r"])):
        limiter.append({"R": limiter_uda["r"][i], "Z": limiter_uda["z"][i]})

    # save
    pickle.dump(limiter, open(f"{save_path}/MAST-U_limiter.pickle", "wb"))

    # save: here we set the wall to be the same as the MAST-U limiter.
    pickle.dump(limiter, open(f"{save_path}/MAST-U_wall.pickle", "wb"))

    # ------------
    # PASSIVE STRUCTURES
    passive_coils_uda = data["geometry_passive"]

    # strucutres to be excluded from simulations (as they're not in EFIT++)
    excluded_structures = [
        "centrecolumn_tiles",
        "div_tiles_lower",
        "div_tiles_upper",
        "nose_baffle_tiles_upper",
        "nose_baffle_tiles_lower",
        "cryopump_upper",
        "cryopump_lower",
    ]

    # calculate the total area for each non-excluded EFIT group
    # --> this is for assigning the passive currents  later on(see further below)
    group_total_area = {}
    for name in passive_coils_uda.keys():
        if name not in excluded_structures:
            coil_data = passive_coils_uda[name]

            if "elementLabels" in coil_data:  # do this for the EFIT group passives only
                for i in range(0, len(coil_data["r"])):

                    group = coil_data["efitGroup"][i]
                    area = coil_data["dr"][i] * coil_data["dz"][i]

                    if group in group_total_area:
                        group_total_area[group] += area
                    else:
                        group_total_area[group] = area

    # extract data into required dictionary form
    passive_coils = []

    # if 'True', we pass the parallelogram  vertices to FreeGSNKE so they can be
    # optionally sub-divided further for better modelling
    if split_passives:
        for name in passive_coils_uda.keys():
            if name not in excluded_structures:
                coil_data = passive_coils_uda[name]

                if "elementLabels" in coil_data:
                    for i in range(0, len(coil_data["r"])):

                        temp = get_element_vertices(
                            coil_data["r"][i],
                            coil_data["z"][i],
                            coil_data["dr"][i],
                            coil_data["dz"][i],
                            coil_data["ang1"][i],
                            coil_data["ang2"][i],
                            version=0.0,
                            close_shape=False,
                        )

                        passive_coils.append(
                            {
                                "R": temp[0],
                                "Z": temp[1],
                                "resistivity": coil_data["rho"],
                                "efitGroup": coil_data["efitGroup"][i],
                                "element": name,
                                "name": coil_data["elementLabels"][i],
                                "current_multiplier": coil_data["dr"][i]
                                * coil_data["dz"][i]
                                / group_total_area[coil_data["efitGroup"][i]],
                            }
                        )
                else:
                    group_area = np.sum(coil_data["dr"] * coil_data["dz"])
                    for i in range(0, len(coil_data["r"])):

                        temp = get_element_vertices(
                            coil_data["r"][i],
                            coil_data["z"][i],
                            coil_data["dr"][i],
                            coil_data["dz"][i],
                            coil_data["ang1"][i],
                            coil_data["ang2"][i],
                            version=0.0,
                            close_shape=False,
                        )

                        passive_coils.append(
                            {
                                "R": temp[0],
                                "Z": temp[1],
                                "resistivity": coil_data["rho"],
                                "element": name,
                                "name": name + f"_{i}",
                                "current_multiplier": coil_data["dr"][i]
                                * coil_data["dz"][i]
                                / group_area,
                            }
                        )

    # if 'False', we pass each parallelogram centre coords and lengths to
    # FreeGSNKE
    else:
        for name in passive_coils_uda.keys():
            # these passive structures are not used in EFIT
            if name not in excluded_structures:

                coil_data = passive_coils_uda[name]
                if "elementLabels" in coil_data:
                    for i in range(0, len(coil_data["r"])):
                        passive_coils.append(
                            {
                                "R": coil_data["r"][i],
                                "Z": coil_data["z"][i],
                                "dR": coil_data["dr"][i],
                                "dZ": coil_data["dz"][i],
                                "resistivity": coil_data["rho"],
                                "efitGroup": coil_data["efitGroup"][i],
                                "element": name,
                                "name": coil_data["elementLabels"][i],
                                "current_multiplier": coil_data["dr"][i]
                                * coil_data["dz"][i]
                                / group_total_area[coil_data["efitGroup"][i]],
                            }
                        )
                else:
                    group_area = np.sum(coil_data["dr"] * coil_data["dz"])
                    for i in range(0, len(coil_data["r"])):
                        passive_coils.append(
                            {
                                "R": coil_data["r"][i],
                                "Z": coil_data["z"][i],
                                "dR": coil_data["dr"][i],
                                "dZ": coil_data["dz"][i],
                                "resistivity": coil_data["rho"],
                                "element": name,
                                "name": name + f"_{i}",
                                "current_multiplier": coil_data["dr"][i]
                                * coil_data["dz"][i]
                                / group_area,
                            }
                        )

    # save data
    pickle.dump(
        passive_coils,
        open(f"{save_path}/MAST-U_passive_coils.pickle", "wb"),
    )

    # ------------
    # MAGNETIC PROBES

    # data
    fluxloops_uda = data["fluxloops"]

    # extract data into required form
    flux_loops = []
    for i in range(len(fluxloops_uda["r"])):
        flux_loops.append(
            {
                "name": fluxloops_uda["names"][i],
                "position": np.array(
                    [fluxloops_uda["r"][i][0], fluxloops_uda["z"][i][0]]
                ),
            }
        )

    # data
    pickups_uda = data["pickups"]

    # extract data into required form
    pickups = []
    for i in range(len(pickups_uda["names"])):

        # calculate normalised orientation directions based on poloidal angles
        r_pol_hat = np.cos(pickups_uda["pol_ang"][i])
        z_pol_hat = np.sin(pickups_uda["pol_ang"][i])
        pickups.append(
            {
                "name": pickups_uda["names"][i],
                "position": np.array([pickups_uda["r"][i], 0, pickups_uda["z"][i]]),
                "orientation_vector": np.array(
                    [r_pol_hat, pickups_uda["tor_ang"][i], z_pol_hat]
                ),
            }
        )

    # save
    pickle.dump(
        {"flux_loops": flux_loops, "pickups": pickups},
        open(f"{save_path}/MAST-U_magnetic_probes.pickle", "wb"),
    )

    print("MAST-U geometry data successfully extracted and pickle files built.")


# ------------
# ------------
def get_efit_data(
    client: pyuda.Client,
    active_coils_path: str,
    passive_coils_path: str,
    shot: int = 45425,
    zero_passives: bool = False,
    data_type: str = "magnetics",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict, dict]:
    """
    Extract the key EFIT++ reconstruction data at each time slice so that we can use it in
    FreeGSNKE to carry out static forward GS solves. Only time slices which EFIT++ managed to
    converge on are returned.

    Parameters
    ----------
    client : pyuda.Client
        The pyUDA client.
    active_coils_path : str
        Path to active coils pickle.
    passive_coils_path : str
        Path to passive coils pickle.
    shot : int, optional
        MAST-U shot number.
    zero_passives : bool, optional
        If True, we set the currents in the MAST-U passive structures to zero, if False, we use
        the currents found by EFIT++.
    data_type : str, optional
        Which EFIT++ reconstruction to use: 'magnetics' for the magnetics-only reconstruction
        (profile parameters are the Lao85 p'/FF' polynomial coefficients), or
        'magnetics_and_mse' for the magnetics + motional Stark effect reconstruction (profile
        parameters are the tension-spline p'/FF' knot locations/values/tensions).

    Returns
    -------
    np.array
        EFIT++ reconstruction times [s] (converged time slices only).
    np.array
        Index of each converged time slice within the full (unfiltered) EFIT++ reconstruction.
        Useful for aligning with data fetched separately over the full, unfiltered time base
        (e.g. via extract_EFIT_outputs / extract_EFIT_outputs_splines).
    np.array
        Total plasma current [Amps] at each (converged) EFIT++ reconstruction time.
    np.array
        Vaccum toroidal field parameter at each (converged) EFIT++ reconstruction time.
    dict
        Profile parameters at each (converged) EFIT++ reconstruction time.
        For data_type='magnetics' this contains:
            'alpha', 'beta' : p'/FF' Lao85 polynomial coefficients.
            'alpha_logic', 'beta_logic' : p'/FF' edge boundary condition logicals.
        For data_type='magnetics_and_mse' this contains:
            'pp_knots', 'ffp_knots' : p'/FF' tension-spline knot locations.
            'pp_values', 'ffp_values' : p'/FF' tension-spline values at the knot points.
            'pp_values_2nd', 'ffp_values_2nd' : p'/FF' tension-spline second deriv. values at
                the knot points.
            'pp_tension', 'ffp_tension' : p'/FF' tension-spline tension values.
    dict
        Dictionary of active coil and passive structure currents: within each key is an
        array of currents (one at each converged EFIT++ reconstruction time). These are
        currents for the symmetric active coil set up.
    dict
        Dictionary of active coil and passive structure currents: within each key is an
        array of currents (one at each converged EFIT++ reconstruction time). These are
        currents for the non-symmetric active coil set up.
    """

    if data_type == "magnetics":
        prefix = "epm"
    elif data_type == "magnetics_and_mse":
        prefix = "epq"
    else:
        raise ValueError(
            "'data_type' must be either 'magnetics' or 'magnetics_and_mse'."
        )

    # load data
    Ip = client.get(
        f"/{prefix}/input/constraints/plasmacurrent/computed", shot
    ).data  # plasma current
    fvac = client.get(f"/{prefix}/input/bvacradiusproduct", shot).data  # fvac

    if data_type == "magnetics":
        alpha = client.get(
            f"/{prefix}/output/numericaldetails/degreesoffreedom/pprimecoeffs", shot
        ).data  # pprime coefficients
        beta = client.get(
            f"/{prefix}/output/numericaldetails/degreesoffreedom/ffprimecoeffs", shot
        ).data  # ffprime coefficients
        alpha_logic = client.get(
            f"/{prefix}/input/numericalcontrols/pp/edge", shot
        ).data  # pprime logical
        beta_logic = client.get(
            f"/{prefix}/input/numericalcontrols/ffp/edge", shot
        ).data  # ffprime logical

        profile_params = dict(
            alpha=alpha,
            beta=beta,
            alpha_logic=alpha_logic,
            beta_logic=beta_logic,
        )
    else:
        pp_coeffs = client.get(
            f"/{prefix}/output/numericaldetails/degreesoffreedom/pprimecoeffs", shot
        ).data  # pprime coefficients (contains values at knots, second deriv. values at knots)
        ffp_coeffs = client.get(
            f"/{prefix}/output/numericaldetails/degreesoffreedom/ffprimecoeffs", shot
        ).data  # ffprime coefficients (contains values at knots, second deriv. values at knots)
        pp_values = pp_coeffs[
            :, 0::2
        ]  # every second element (starting from zero) is the value at a knot
        pp_values_2nd = pp_coeffs[
            :, 1::2
        ]  # every second element (starting from one) is the value of the second deriv. at a knot
        ffp_values = ffp_coeffs[
            :, 0::2
        ]  # every second element (starting from zero) is the value at a knot
        ffp_values_2nd = ffp_coeffs[
            :, 1::2
        ]  # every second element (starting from one) is the value of the second deriv. at a knot

        pp_tension = client.get(
            f"/{prefix}/input/numericalcontrols/pp/tens", shot
        ).data  # pprime tension value
        ffp_tension = client.get(
            f"/{prefix}/input/numericalcontrols/ffp/tens", shot
        ).data  # ffprime tension value
        pp_knots_raw = client.get(
            f"/{prefix}/input/numericalcontrols/pp/knt", shot
        ).data  # pprime knot locations
        ffp_knots_raw = client.get(
            f"/{prefix}/input/numericalcontrols/ffp/knt", shot
        ).data  # ffprime knot locations
        pp_knots = pp_knots_raw[:, pp_knots_raw[0, :] > -1]
        ffp_knots = ffp_knots_raw[:, ffp_knots_raw[0, :] > -1]

        profile_params = dict(
            pp_knots=pp_knots,
            ffp_knots=ffp_knots,
            pp_values=pp_values,
            ffp_values=ffp_values,
            pp_values_2nd=pp_values_2nd,
            ffp_values_2nd=ffp_values_2nd,
            pp_tension=pp_tension,
            ffp_tension=ffp_tension,
        )

    # active coil\passive structure currents need to be done carefully
    current_labels = client.get(
        f"/{prefix}/input/constraints/pfcircuits/shortname", shot
    ).data  # active/passive coil current names
    currents_values = client.get(
        f"/{prefix}/input/constraints/pfcircuits/computed", shot
    ).data  # active/passive coil current values

    # Active coils
    currents = {}
    currents_nonsym = {}

    with open(active_coils_path, "rb") as file:
        active_coils = pickle.load(file)

    efit_names = current_labels[0:24]  # active coil names in efit

    # loop through active coil names in freegsnke to set them with UDA data
    for active_coil_name in active_coils.keys():

        # special cases for specific coil names
        if active_coil_name == "Solenoid":
            # find indices for Solenoid coils in efit_names
            indices = [i for i, efit_name in enumerate(efit_names) if "p1" in efit_name]
            currents["Solenoid"] = currents_values[:, indices[0]] if indices else None
            currents_nonsym["Solenoid"] = (
                currents_values[:, indices[0]] if indices else None
            )
        else:
            # find indices for other coils in efit_names
            indices = [
                i
                for i, efit_name in enumerate(efit_names)
                if active_coil_name in efit_name
            ]
            if indices:
                polarity = np.sign(currents_values[:, indices[0]])
                average_current = np.sum(
                    np.abs(currents_values[:, indices]), axis=1
                ) / len(indices)
                currents[active_coil_name] = polarity * average_current
                for j in indices:
                    currents_nonsym[efit_names[j]] = currents_values[:, j]

            else:
                currents[active_coil_name] = None

    # passive structures
    with open(passive_coils_path, "rb") as file:
        passive_coils = pickle.load(file)

    for i in range(0, len(passive_coils)):
        coil = passive_coils[i]

        if "efitGroup" in coil:
            group_name = coil["efitGroup"]
        else:
            group_name = coil["element"]
        ind = current_labels.tolist().index(group_name)

        if zero_passives:
            currents[coil["name"]] = 0.0
            currents_nonsym[coil["name"]] = 0.0
        else:
            currents[coil["name"]] = (
                currents_values[:, ind] * coil["current_multiplier"]
            )
            currents_nonsym[coil["name"]] = (
                currents_values[:, ind] * coil["current_multiplier"]
            )

    # keep only the time slices that EFIT++ successfully converged on
    status = client.get(f"/{prefix}/equilibriumstatusinteger", shot)
    time_indices = np.where(status.data == 1)[0]
    times = status.time.data[time_indices]

    Ip = Ip[time_indices]
    fvac = fvac[time_indices]
    for key in profile_params:
        profile_params[key] = profile_params[key][time_indices]
    for key in currents:
        currents[key] = currents[key][time_indices]
    for key in currents_nonsym:
        currents_nonsym[key] = currents_nonsym[key][time_indices]

    print(
        f"{len(status.data) - len(time_indices)} of {len(status.data)} EFIT++ time slices "
        "excluded (did not converge)."
    )

    return (
        times,
        time_indices,
        Ip,
        fvac,
        profile_params,
        currents,
        currents_nonsym,
    )


def extract_EFIT_outputs(
    client: pyuda.Client, shot: int, time_indices: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
    dict,
]:
    """
    Extract the key (magnetics-only) EFIT++ reconstruction output data at each
    time slice so that we can compare to FreeGSNKE.

    Parameters
    ----------
    client : pyuda.Client
        The pyUDA client.
    shot : int
        MAST-U shot number.
    time_indices : np.ndarray of int
        Indices (into the full, unfiltered EFIT++ reconstruction time base) of the time slices
        at which to extract the EFIT++ output data (e.g. as returned by `get_efit_data`).

    Returns
    -------
    np.array
        Total plasma flux map [Webers/(2pi)] at each EFIT++ reconstruction time.
    np.array
        Total plasma flux on the magnetic axis [Webers/(2pi)] at each EFIT++ reconstruction time.
    np.array
        Total plasma flux on the plasma boundary [Webers/(2pi)] at each EFIT++ reconstruction time.
    np.array
        Toroidal plasma current density map [Amps/m^2] at each EFIT++ reconstruction time.
    np.array
        The (R,Z) location of the magnetic axis at each EFIT++ reconstruction time.
    np.array
        The inner and outer R location of the plasma boundary at the midplane at each EFIT++ reconstruction time.
    np.array
        The (R,Z) location of the X-points at each EFIT++ reconstruction time.
    np.array
        The p' profile (vs. normalised psi) at each EFIT++ reconstruction time.
    np.array
        The FF' profile (vs. normalised psi) at each EFIT++ reconstruction time.
    np.array
        The strikepoints at each EFIT++ reconstruction time (not always very accurate).
    dict
        Dictionary of (target and computed) fluxloop readings used by EFIT++ at each reconstruction time.
    dict
        Dictionary of (target and computed) pickup coil readings used by EFIT++ at each reconstruction time.
    """

    # equilibrium data
    psi_total = client.get("/epm/output/profiles2d/poloidalflux", shot).data[
        time_indices, :, :
    ]  # total poloidal flux (units= Webers/2*pi)
    psi_axis = client.get("/epm/output/globalparameters/psiaxis", shot).data[
        time_indices
    ]  # flux on magnetic axis
    psi_boundary = client.get("/epm/output/globalparameters/psiboundary", shot).data[
        time_indices
    ]  # flux on plasma boundary
    jtor = client.get("/epm/output/profiles2d/jphi", shot).data[
        time_indices, :, :
    ]  # plasma current density
    magnetic_axis = np.array(
        [
            client.get("/epm/output/globalparameters/magneticaxis/r", shot).data[
                time_indices
            ],
            client.get("/epm/output/globalparameters/magneticaxis/z", shot).data[
                time_indices
            ],
        ]
    ).T  # magnetic axis coords
    midplane_inner_outer_radii = np.array(
        [
            client.get("/epm/output/separatrixgeometry/rmidplanein", shot).data[
                time_indices
            ],
            client.get("/epm/output/separatrixgeometry/rmidplaneout", shot).data[
                time_indices
            ],
        ]
    ).T  # midplane inner/outer radii coords
    x_points = np.array(
        [
            client.get("/epm/output/separatrixgeometry/xpointr", shot).data[
                time_indices
            ],
            client.get("/epm/output/separatrixgeometry/xpointz", shot).data[
                time_indices
            ],
        ]
    ).T  # x-points in flux field
    pprime = client.get("/epm/output/fluxfunctionprofiles/staticpprime", shot).data[
        time_indices
    ]  # pressure profile function
    ffprime = client.get("/epm/output/fluxfunctionprofiles/ffprime", shot).data[
        time_indices
    ]  # toroidal current density profile
    strike_points = np.array(
        [
            client.get("/epm/output/separatrixgeometry/strikepointr", shot).data[
                time_indices
            ],
            client.get("/epm/output/separatrixgeometry/strikepointz", shot).data[
                time_indices
            ],
        ]
    ).T  # strikepoint coords

    # fluxloop data
    flux_names = client.get("/epm/input/constraints/fluxloops/shortname", shot).data
    flux_target = client.get(
        "/epm/input/constraints/fluxloops/target", shot
    ).data  # the data (not needed)
    flux_computed = client.get(
        "/epm/input/constraints/fluxloops/computed", shot
    ).data  # the data
    flux_sigmas = client.get(
        "/epm/input/constraints/fluxloops/sigmas", shot
    ).data  # the "errors"
    flux_weights = client.get("/epm/input/constraints/fluxloops/weights", shot).data
    indices = flux_weights[0, :]  # just selects ones that are used in EFIT
    fluxloop_data = dict(
        names=flux_names[(indices == 1)],
        target=flux_target[:, (indices == 1)],
        computed=flux_computed[:, (indices == 1)],
        sigmas=flux_sigmas[:, (indices == 1)],
    )

    # pickup coil data
    pickup_names = client.get(
        "/epm/input/constraints/magneticprobes/shortname", shot
    ).data
    pickup_target = client.get(
        "/epm/input/constraints/magneticprobes/target", shot
    ).data  # the data
    pickup_computed = client.get(
        "/epm/input/constraints/magneticprobes/computed", shot
    ).data  # the data
    pickup_sigmas = client.get(
        "/epm/input/constraints/magneticprobes/sigmas", shot
    ).data  # the "errors"
    pickup_weights = client.get(
        "/epm/input/constraints/magneticprobes/weights", shot
    ).data
    indices = pickup_weights[0, :]  # just selects ones that are used in EFIT

    pickup_data = dict(
        names=pickup_names[(indices == 1)],
        target=pickup_target[:, (indices == 1)],
        computed=pickup_computed[:, (indices == 1)],
        sigmas=pickup_sigmas[:, (indices == 1)],
    )

    return (
        psi_total,
        psi_axis,
        psi_boundary,
        jtor,
        magnetic_axis,
        midplane_inner_outer_radii,
        x_points,
        pprime,
        ffprime,
        strike_points,
        fluxloop_data,
        pickup_data,
    )


def extract_EFIT_outputs_splines(
    client: pyuda.Client, shot: int, time_indices: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
    dict,
]:
    """
    Extract the key (magnetics + motional stark effect) EFIT++ reconstruction output data at each
    time slice so that we can compare to FreeGSNKE.

    Parameters
    ----------
    client : pyuda.Client
        The pyUDA client.
    shot : int
        MAST-U shot number.
    time_indices : np.ndarray of int
        Indices (into the full, unfiltered EFIT++ reconstruction time base) of the time slices
        at which to extract the EFIT++ output data (e.g. as returned by `get_efit_data`).

    Returns
    -------
    np.array
        Total plasma flux map [Webers/(2pi)] at each EFIT++ reconstruction time.
    np.array
        Total plasma flux on the magnetic axis [Webers/(2pi)] at each EFIT++ reconstruction time.
    np.array
        Total plasma flux on the plasma boundary [Webers/(2pi)] at each EFIT++ reconstruction time.
    np.array
        Toroidal plasma current density map [Amps/m^2] at each EFIT++ reconstruction time.
    np.array
        The (R,Z) location of the magnetic axis at each EFIT++ reconstruction time.
    np.array
        The inner and outer R location of the plasma boundary at the midplane at each EFIT++ reconstruction time.
    np.array
        The (R,Z) location of the X-points at each EFIT++ reconstruction time.
    np.array
        The p' profile (vs. normalised psi) at each EFIT++ reconstruction time.
    np.array
        The FF' profile (vs. normalised psi) at each EFIT++ reconstruction time.
    np.array
        The strikepoints at each EFIT++ reconstruction time (not always very accurate).
    dict
        Dictionary of (target and computed) fluxloop readings used by EFIT++ at each reconstruction time.
    dict
        Dictionary of (target and computed) pickup coil readings used by EFIT++ at each reconstruction time.
    """

    # equilibrium data
    psi_total = client.get("/epq/output/profiles2d/poloidalflux", shot).data[
        time_indices, :, :
    ]  # total poloidal flux (units= Webers/2*pi)
    psi_axis = client.get("/epq/output/globalparameters/psiaxis", shot).data[
        time_indices
    ]  # flux on magnetic axis
    psi_boundary = client.get("/epq/output/globalparameters/psiboundary", shot).data[
        time_indices
    ]  # flux on plasma boundary
    jtor = client.get("/epq/output/profiles2d/jphi", shot).data[
        time_indices, :, :
    ]  # plasma current density
    magnetic_axis = np.array(
        [
            client.get("/epq/output/globalparameters/magneticaxis/r", shot).data[
                time_indices
            ],
            client.get("/epq/output/globalparameters/magneticaxis/z", shot).data[
                time_indices
            ],
        ]
    ).T  # magnetic axis coords
    midplane_inner_outer_radii = np.array(
        [
            client.get("/epq/output/separatrixgeometry/rmidplanein", shot).data[
                time_indices
            ],
            client.get("/epq/output/separatrixgeometry/rmidplaneout", shot).data[
                time_indices
            ],
        ]
    ).T  # midplane inner/outer radii coords
    x_points = np.array(
        [
            client.get("/epq/output/separatrixgeometry/xpointr", shot).data[
                time_indices
            ],
            client.get("/epq/output/separatrixgeometry/xpointz", shot).data[
                time_indices
            ],
        ]
    ).T  # x-points in flux field
    pprime = client.get("/epq/output/fluxfunctionprofiles/staticpprime", shot).data[
        time_indices
    ]  # pressure profile function
    ffprime = client.get("/epq/output/fluxfunctionprofiles/ffprime", shot).data[
        time_indices
    ]  # toroidal current density profile
    strike_points = np.array(
        [
            client.get("/epq/output/separatrixgeometry/strikepointr", shot).data[
                time_indices
            ],
            client.get("/epq/output/separatrixgeometry/strikepointz", shot).data[
                time_indices
            ],
        ]
    ).T  # strikepoint coords

    # fluxloop data
    flux_names = client.get("/epq/input/constraints/fluxloops/shortname", shot).data
    flux_target = client.get(
        "/epq/input/constraints/fluxloops/target", shot
    ).data  # the data (not needed)
    flux_computed = client.get(
        "/epq/input/constraints/fluxloops/computed", shot
    ).data  # the data
    flux_sigmas = client.get(
        "/epq/input/constraints/fluxloops/sigmas", shot
    ).data  # the "errors"
    flux_weights = client.get("/epq/input/constraints/fluxloops/weights", shot).data
    indices = flux_weights[0, :]  # just selects ones that are used in EFIT
    fluxloop_data = dict(
        names=flux_names[(indices == 1)],
        target=flux_target[:, (indices == 1)],
        computed=flux_computed[:, (indices == 1)],
        sigmas=flux_sigmas[:, (indices == 1)],
    )

    # pickup coil data
    pickup_names = client.get(
        "/epq/input/constraints/magneticprobes/shortname", shot
    ).data
    pickup_target = client.get(
        "/epq/input/constraints/magneticprobes/target", shot
    ).data  # the data
    pickup_computed = client.get(
        "/epq/input/constraints/magneticprobes/computed", shot
    ).data  # the data
    pickup_sigmas = client.get(
        "/epq/input/constraints/magneticprobes/sigmas", shot
    ).data  # the "errors"
    pickup_weights = client.get(
        "/epq/input/constraints/magneticprobes/weights", shot
    ).data
    indices = pickup_weights[0, :]  # just selects ones that are used in EFIT

    pickup_data = dict(
        names=pickup_names[(indices == 1)],
        target=pickup_target[:, (indices == 1)],
        computed=pickup_computed[:, (indices == 1)],
        sigmas=pickup_sigmas[:, (indices == 1)],
    )

    return (
        psi_total,
        psi_axis,
        psi_boundary,
        jtor,
        magnetic_axis,
        midplane_inner_outer_radii,
        x_points,
        pprime,
        ffprime,
        strike_points,
        fluxloop_data,
        pickup_data,
    )


# --------------------------------
# ADDITIONAL FUNCTIONS


def get_element_vertices(
    centreR: float,
    centreZ: float,
    dR: float,
    dZ: float,
    a1: float,
    a2: float,
    version: float = 0.1,
    close_shape: bool = False,
) -> list:
    """
    Convert EFIT++ description of parallelograms to four vertices (used in FreeGSNKE
    passive structures).

    Code courtesy of Lucy Kogan (UKAEA).

    Parameters
    ----------
    centreR : float
        Centre (R) of the parallelogram.
    centreZ : float
        Centre (Z) of the parallelogram.
    dR : float
        Width of the the parallelogram.
    dZ : float
        Height of the the parallelogram.
    a1 : float
        Angle between the horizontal and the base of the parallelogram (zero for rectangles).
    a2 : float
        Angle between the horizontal and the right side of the parallelogram (zero for rectangles).
    version : float, optional
        Geometry version (backwards compatibilty for bug in < V0.1). Use default.
    close_shape : bool, optional
        Repeat first vertex to close the shape if set to True.

    Returns
    -------
    list
        Returns list of vertics with radial 'rr' and vertical 'zz' positions and
        the original width 'dR' and height 'dZ' of the shape.
    """

    if a1 == 0.0 and a2 == 0.0:
        # Rectangle
        rr = [
            centreR - dR / 2.0,
            centreR - dR / 2.0,
            centreR + dR / 2.0,
            centreR + dR / 2.0,
        ]
        zz = [
            centreZ - dZ / 2.0,
            centreZ + dZ / 2.0,
            centreZ + dZ / 2.0,
            centreZ - dZ / 2.0,
        ]
    elif version == 0.1:
        # Parallelogram
        Lx1 = math.cos(math.radians(a1)) * dR
        Lx2 = math.sin(math.radians(a2)) * dZ
        Lx = Lx1 + Lx2

        Lz1 = math.sin(math.radians(a1)) * dR
        Lz2 = math.cos(math.radians(a2)) * dZ
        Lz = Lz1 + Lz2

        rr = [
            centreR - Lx / 2,  # A
            centreR - Lx / 2 + Lx2,  # B
            centreR + Lx / 2,  # C
            centreR - Lx / 2 + Lx1,
        ]  # D

        zz = [
            centreZ - Lz / 2,
            centreZ - Lz / 2 + Lz2,
            centreZ + Lz / 2,
            centreZ - Lz / 2 + Lz1,
        ]
    else:
        # Parallelogram (different definitions of dR, dZ, angle1 and angle2)
        a1_tan = 0.0
        a2_tan = 0.0
        if a1 > 0.0:
            a1_tan = np.tan(a1 * np.pi / 180.0)

        if a2 > 0.0:
            a2_tan = 1.0 / np.tan(a2 * np.pi / 180.0)

        rr = [
            centreR - dR / 2.0 - dZ / 2.0 * a2_tan,
            centreR + dR / 2.0 - dZ / 2.0 * a2_tan,
            centreR + dR / 2.0 + dZ / 2.0 * a2_tan,
            centreR - dR / 2.0 + dZ / 2.0 * a2_tan,
        ]

        zz = [
            centreZ - dZ / 2.0 - dR / 2.0 * a1_tan,
            centreZ - dZ / 2.0 + dR / 2.0 * a1_tan,
            centreZ + dZ / 2.0 + dR / 2.0 * a1_tan,
            centreZ + dZ / 2.0 - dR / 2.0 * a1_tan,
        ]

    if close_shape:
        rr.append(rr[0])
        zz.append(zz[0])

    return [rr, zz, dR, dZ]


def find_strikepoints(
    R: np.ndarray, Z: np.ndarray, psi: np.ndarray, psi_boundary: float, wall: np.ndarray
) -> np.ndarray | None:
    """
    Find the strikepoints of an equilibrium with the wall.

    Parameters
    ----------
    R : np.ndarray (nx, nz)
        2D array of major radius coordinates [m]
    Z : np.ndarray (nx, nz)
        2D array of vertical coordinates [m]
    psi : np.ndarray (nx, nz)
        2D poloidal flux map [Wb]
    psi_boundary : float
        Value of psi at the plasma boundary [Wb]
    wall : np.ndarray (N, 2)
        Wall/limiter coordinates as (R, Z) pairs [m]

    Returns
    -------
    np.ndarray or None
        Array of strikepoint coordinates, shape (M, 2), or None if no
        intersections are found.
    """

    # find contour object for psi_boundary
    cs = plt.contour(R, Z, psi, levels=[psi_boundary])
    plt.close()  # this isn't the most elegant but we don't need the plot itself

    # for each item in the contour object there's a list of points in (r,z) (i.e. a line)
    psi_boundary_lines = []
    for i, item in enumerate(cs.allsegs[0]):
        psi_boundary_lines.append(item)

    # use the shapely package to find where each psi_boundary_line intersects the limiter (or not)
    strikes = []
    curve1 = sh.LineString(wall)
    for j, line in enumerate(psi_boundary_lines):
        curve2 = sh.LineString(line)

        # find the intersection points
        intersection = curve2.intersection(curve1)

        # extract intersection points
        if intersection.geom_type == "Point":
            strikes.append(np.squeeze(np.array(intersection.xy).T))
        elif intersection.geom_type == "MultiPoint":
            strikes.append(
                np.squeeze(np.array([geom.xy for geom in intersection.geoms]))
            )

    # check how many strikepoints
    if len(strikes) == 0:
        out = None
    else:
        out = np.concatenate(strikes, axis=0)

    return out


def Separatrix(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    ntheta: int,
    psival: float = 1.0,
    theta_grid: np.ndarray | None = None,
    input_opoint: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute separatrix coordinates for a given equilibrium flux map.

    This function traces the ψ = const surface corresponding to the separatrix
    (default psival = 1.0 in normalized coordinates) starting from the magnetic
    axis and marching outwards in poloidal angle.

    The separatrix is sampled at equally spaced geometric poloidal angles,
    with care taken to avoid sampling exactly at the X-point locations.

    Parameters
    ----------
    R : ndarray
        Radial grid coordinates (2D array).
    Z : ndarray
        Vertical grid coordinates (2D array).
    psi : ndarray
        Poloidal flux on the (R, Z) grid.
    ntheta : int
        Number of poloidal angle samples along the separatrix.
    psival : float, optional
        Target normalized flux value defining the separatrix.
    theta_grid : ndarray, optional
        User-specified poloidal angle grid. If None, a uniform grid is used.
    input_opoint : tuple, optional
        User-specified magnetic axis (R0, Z0). If None, it is inferred.

    Returns
    -------
    points : ndarray
        Array of separatrix points (R, Z) sampled along θ.
    theta_grid : ndarray
        Poloidal angle grid used for the construction.
    """

    opoint, xpoint = critical.find_critical(R, Z, psi)

    psinorm = (psi - opoint[0][2]) / (xpoint[0][2] - opoint[0][2])

    psifunc = sp.interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psinorm)

    if input_opoint is None:
        r0, z0 = opoint[0][0:2]
    else:
        r0, z0 = input_opoint

    if theta_grid is None:
        theta_grid = np.linspace(0, 2 * pi, ntheta, endpoint=False)
    dtheta = theta_grid[1] - theta_grid[0]

    # Avoid putting theta grid points exactly on the X-points
    xpoint_theta = np.arctan2(xpoint[0][0] - r0, xpoint[0][1] - z0)
    xpoint_theta = xpoint_theta * (xpoint_theta >= 0) + (xpoint_theta + 2 * pi) * (
        xpoint_theta < 0
    )  # let's make it between 0 and 2*pi
    # How close in theta to allow theta grid points to the X-point
    TOLERANCE = 2.0e-4
    if any(abs(theta_grid - xpoint_theta) < TOLERANCE):
        # warn("Theta grid too close to X-point, shifting by half-step")
        # print('Im shifting the grid!')
        theta_grid += (
            dtheta / 2 * np.ones(ntheta) * (abs(theta_grid - xpoint_theta) < TOLERANCE)
        )

    isoflux = []
    for theta in theta_grid:
        r, z = find_psisurface(
            psifunc,
            R,
            Z,
            r0,
            z0,
            r0 + 10.0 * np.sin(theta),
            z0 + 10.0 * np.cos(theta),
            psival=psival,
            n=1000,
        )
        isoflux.append((r, z))

    threshold = 0.1  # exlude points this far away from other nearest point
    points = np.array(isoflux)
    distances = sp.spatial.distance.cdist(points, points)
    min_distances = np.min(
        np.where(distances == 0, np.inf, distances), axis=1
    )  # Exclude distances to itself
    far_points = np.where(min_distances > threshold)[0]
    points[far_points] = None

    return points, theta_grid


def find_psisurface(
    psifunc: Callable,
    R: np.ndarray,
    Z: np.ndarray,
    r0: float,
    z0: float,
    r1: float,
    z1: float,
    psival: float = 1.0,
    n: int = 100,
) -> tuple[float, float]:
    """
    Find an intersection point of a ψ = const surface along a straight line.

    This routine samples a straight line from an initial point inside the
    separatrix to a point outside it, evaluates the flux along the line, and
    interpolates to locate the position where ψ equals the target value.

    Parameters
    ----------
    psifunc : callable
        Interpolated flux function ψ(R, Z).
    R : ndarray
        Radial grid (used for domain clipping).
    Z : ndarray
        Vertical grid (used for domain clipping).
    r0 : float
        Starting radial coordinate (assumed inside target surface).
    z0 : float
        Starting vertical coordinate (assumed inside target surface).
    r1 : float
        Ending radial coordinate (outside target surface).
    z1 : float
        Ending vertical coordinate (outside target surface).
    psival : float, optional
        Target flux value defining the surface.
    n : int, optional
        Number of points sampled along the line.

    Returns
    -------
    r : float
        Radial coordinate of ψ = psival intersection.
    z : float
        Vertical coordinate of ψ = psival intersection.
    """
    # Clip (r1,z1) to be inside domain
    # Shorten the line so that the direction is unchanged
    if abs(r1 - r0) > 1e-6:
        rclip = clip(r1, np.min(R), np.max(R))
        z1 = z0 + (z1 - z0) * abs((rclip - r0) / (r1 - r0))
        r1 = rclip

    if abs(z1 - z0) > 1e-6:
        zclip = clip(z1, np.min(Z), np.max(Z))
        r1 = r0 + (r1 - r0) * abs((zclip - z0) / (z1 - z0))
        z1 = zclip

    r = linspace(r0, r1, n)
    z = linspace(z0, z1, n)

    pnorm = psifunc(r, z, grid=False)

    if hasattr(psival, "__len__"):
        pass

    else:
        # Only one value
        ind = argmax(pnorm > psival)

        # Edited by Bhavin 31/07/18
        # Changed 1.0 to psival in f
        # make f gradient to psival surface
        f = (pnorm[ind] - psival) / (pnorm[ind] - pnorm[ind - 1])

        r = (1.0 - f) * r[ind] + f * r[ind - 1]
        z = (1.0 - f) * z[ind] + f * z[ind - 1]

    return r, z


def max_euclidean_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    Compute the maximum Euclidean distance between corresponding points in two sets.

    Points containing NaN values are excluded before the distance calculation.

    Parameters
    ----------
    points1 : ndarray
        First set of (x, y) points.
    points2 : ndarray
        Second set of (x, y) points with the same shape as points1.

    Returns
    -------
    float
        Maximum Euclidean distance between valid corresponding points.
        Returns NaN if no valid points remain.
    """
    valid_indices = np.logical_not(
        np.any(np.isnan(points1), axis=1) | np.any(np.isnan(points2), axis=1)
    )
    points1_valid = points1[valid_indices]
    points2_valid = points2[valid_indices]
    if len(points1_valid) == 0 or len(points2_valid) == 0:
        return np.nan
    return np.max(np.sqrt(np.sum((points1_valid - points2_valid) ** 2, axis=1)))


def separatrix_areas(
    separatrix_1: np.ndarray, separatrix_2: np.ndarray
) -> tuple[float, sh.Polygon, sh.Polygon]:
    """
    Compute a geometric similarity metric between two separatrix shapes.

    This function constructs convex hull polygons from two sets of (R, Z)
    points, then compares their overlap using union and intersection areas.
    The resulting metric η is based on the non-overlapping area relative
    to the total area, as defined in Bardsley et al. (2024, Nuclear Fusion).

    Parameters
    ----------
    separatrix_1 : ndarray
        Array of shape (N, 2) containing (R, Z) points of the first separatrix.
    separatrix_2 : ndarray
        Array of shape (M, 2) containing (R, Z) points of the second separatrix.

    Returns
    -------
    eta : float
        Normalised non-overlap metric between the two separatrices.
    polygon1 : shapely.geometry.Polygon
        Convex hull polygon of separatrix_1.
    polygon2 : shapely.geometry.Polygon
        Convex hull polygon of separatrix_2.
    """

    # create Polygon objects from the points using Shapely package
    polygon1 = sh.Polygon(separatrix_1).convex_hull
    polygon2 = sh.Polygon(separatrix_2).convex_hull

    # calculate union and intersection of the two polygons
    union_polygon = polygon1.union(polygon2)
    intersection_polygon = polygon1.intersection(polygon2)

    # Calculate the area of the non-overlapping regions
    non_overlapping_area = union_polygon.area - intersection_polygon.area

    # metric from paper
    eta = non_overlapping_area / (polygon1.area + polygon2.area)

    return eta, polygon1, polygon2
