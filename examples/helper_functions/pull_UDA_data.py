"""

This file contains a number of useful functions required to run the "example3" notebook. 
  
Machine description code courtesy of Geof Cunningham, modified by Kamran Pentland (UKAEA). 

"""

import math
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pyuda
import scipy as sp
import shapely as sh
from freegs4e import critical
from numpy import (
    abs,
    amax,
    arctan2,
    argmax,
    argmin,
    clip,
    cos,
    dot,
    linspace,
    pi,
    sin,
    sqrt,
    sum,
    zeros,
)
from scipy.interpolate import interp1d

# --------------------------------
# MACHINE DESCRIPTION DATA
# recovers the machine description data and builds the required pickle files for FreeGSNKE


def get_machine_data(
    shot=45425,  # choose a shot
    split_passives=True,  # True = model passive structures as parallelograms (recommended), False = model as point current sources
):

    # pull data from pyUDA client
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
                # print(vars(grandchild))
                material = grandchild.material
                # Is there a better method for this? - maybe do like Lucy did with efitGroups
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
                # print(efitGroup)
                # print(elementLabels)
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

    # BUILD THE PICKLE FILES FROM THE DATA IN THE ABOVE DICTIONARY

    # default global machine quantities that may not be present in UDA data
    eta_copper = 1.55e-8  # resistivity in Ohm*m, for active coils

    # ------------
    # ACTIVE COILS
    active_coils_uda = data["geometry_pfcoil"]

    # extract data into required form
    active_coils = {}

    # coil definitions (do not modify)
    Solenoid = {
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

    # define symmetric active coils dictionary
    active_coils = {}

    active_coils["Solenoid"] = Solenoid

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

    active_coils["dp"] = {}
    active_coils["dp"]["1"] = dp_upper
    active_coils["dp"]["2"] = dp_lower

    active_coils["d5"] = {}
    active_coils["d5"]["1"] = d5_upper
    active_coils["d5"]["2"] = d5_lower

    active_coils["d6"] = {}
    active_coils["d6"]["1"] = d6_upper
    active_coils["d6"]["2"] = d6_lower

    active_coils["d7"] = {}
    active_coils["d7"]["1"] = d7_upper
    active_coils["d7"]["2"] = d7_lower

    active_coils["p4"] = {}
    active_coils["p4"]["1"] = p4_upper
    active_coils["p4"]["2"] = p4_lower

    active_coils["p5"] = {}
    active_coils["p5"]["1"] = p5_upper
    active_coils["p5"]["2"] = p5_lower

    active_coils["p6"] = {}
    active_coils["p6"]["1"] = p6_upper
    active_coils["p6"]["2"] = p6_lower

    # save data: this pickle file can be used when a symmetric MAST-U machine
    # description is required.
    pickle.dump(
        active_coils, open("../machine_configs/MAST-U/MAST-U_active_coils.pickle", "wb")
    )

    # define non-symmetric active coils dictionary
    active_coils_nonsym = {}

    active_coils_nonsym["Solenoid"] = Solenoid

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

    active_coils_nonsym["dp_upper"] = {}
    active_coils_nonsym["dp_upper"]["1"] = dp_upper
    active_coils_nonsym["dp_lower"] = {}
    active_coils_nonsym["dp_lower"]["1"] = dp_lower

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

    active_coils_nonsym["p4_upper"] = {}
    active_coils_nonsym["p4_upper"]["1"] = p4_upper
    active_coils_nonsym["p4_lower"] = {}
    active_coils_nonsym["p4_lower"]["1"] = p4_lower

    active_coils_nonsym["p5_upper"] = {}
    active_coils_nonsym["p5_upper"]["1"] = p5_upper
    active_coils_nonsym["p5_lower"] = {}
    active_coils_nonsym["p5_lower"]["1"] = p5_lower

    active_coils_nonsym["p6_upper"] = {}
    active_coils_nonsym["p6_upper"]["1"] = p6_upper
    active_coils_nonsym["p6_lower"] = {}
    active_coils_nonsym["p6_lower"]["1"] = p6_lower
    active_coils_nonsym["p6_lower"]["1"]["polarity"] = 1

    # save data: this pickle file can be used when a non-symmetric MAST-U machine
    # description is required.
    pickle.dump(
        active_coils_nonsym,
        open("../machine_configs/MAST-U/MAST-U_active_coils_nonsym.pickle", "wb"),
    )

    # ------------
    # LIMITER/WALL
    limiter_uda = data["geometry_limiter"]

    # extract data into required form
    limiter = []
    for i in range(len(limiter_uda["r"])):
        limiter.append({"R": limiter_uda["r"][i], "Z": limiter_uda["z"][i]})

    # save
    pickle.dump(limiter, open("../machine_configs/MAST-U/MAST-U_limiter.pickle", "wb"))

    # save: here we set the wall to be the same as the MAST-U limiter.
    pickle.dump(limiter, open("../machine_configs/MAST-U/MAST-U_wall.pickle", "wb"))

    # ------------
    # PASSIVE STRUCTURES
    passive_coils_uda = data["geometry_passive"]

    # strucutres to be excluded from simulations (as they're not in EFIT)
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
        open("../machine_configs/MAST-U/MAST-U_passive_coils.pickle", "wb"),
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
        open("../machine_configs/MAST-U/MAST-U_magnetic_probes.pickle", "wb"),
    )

    # DONE
    print("MAST-U geometry data successfully extracted and pickle files built.")


# ------------
# ------------
def load_efit_times_and_status(client, shot=45425):
    """
    Extract the shot status (converged or not) and times from the EFIT data.

    """

    # load data
    status = client.get("/epm/equilibriumstatusinteger", shot)

    return status.time.data, status.data


# ------------
# ------------
def load_efit_times_and_status_splines(client, shot=45425):
    """
    Extract the shot status (converged or not) and times from the EFIT data.

    """

    # load data
    status = client.get("/epq/equilibriumstatusinteger", shot)

    return status.time.data, status.data


# ------------
# ------------
def load_static_solver_inputs(client, shot=45425, zero_passives=False):
    """
    Extract the EFIT++ data required for the static solve.

    """

    # load data
    Ip = client.get(
        "/epm/input/constraints/plasmacurrent/computed", shot
    ).data  # plasma current
    fvac = client.get("/epm/input/bvacradiusproduct", shot).data  # fvac
    alpha = client.get(
        "/epm/output/numericaldetails/degreesoffreedom/pprimecoeffs", shot
    ).data  # pprime coefficients
    beta = client.get(
        "/epm/output/numericaldetails/degreesoffreedom/ffprimecoeffs", shot
    ).data  # ffprime coefficients
    alpha_logic = client.get(
        "/epm/input/numericalcontrols/pp/edge", shot
    ).data  # pprime logical
    beta_logic = client.get(
        "/epm/input/numericalcontrols/ffp/edge", shot
    ).data  # ffprime logical

    # active coil\passive structure currents need to be done carefully
    current_labels = client.get(
        "/epm/input/constraints/pfcircuits/shortname", shot
    ).data  # active/passive coil current names
    currents_values = client.get(
        "/epm/input/constraints/pfcircuits/computed", shot
    ).data  # active/passive coil current values

    # Active coils
    currents = {}
    currents_nonsym = {}
    currents_discrepancy = {}

    with open("../machine_configs/MAST-U//MAST-U_active_coils.pickle", "rb") as file:
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
            currents_discrepancy["Solenoid"] = 0.0
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
                currents_discrepancy[active_coil_name] = polarity * np.abs(
                    np.abs(currents_values[:, indices[0]]) - average_current
                )
                for j in indices:
                    currents_nonsym[efit_names[j]] = currents_values[:, j]

            else:
                currents[active_coil_name] = None

    # passive structures
    with open("../machine_configs/MAST-U//MAST-U_passive_coils.pickle", "rb") as file:
        passive_coils = pickle.load(file)

    # Passive structures
    for i in range(0, len(passive_coils)):
        coil = passive_coils[i]

        if "efitGroup" in coil:
            group_name = coil["efitGroup"]
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
                # currents[coil["name"]] = efit_currents['currents_input'][t,ind]
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
                # currents[coil["name"]] = efit_currents['currents_input'][t,ind]

    return (
        Ip,
        fvac,
        alpha,
        beta,
        alpha_logic,
        beta_logic,
        currents,
        currents_nonsym,
        currents_discrepancy,
    )


# ------------
# ------------
def load_static_solver_inputs_splines(client, shot=45425, zero_passives=False):
    """
    Extract the EFIT++ data required for the static solve.

    """

    # load data
    Ip = client.get(
        "/epq/input/constraints/plasmacurrent/computed", shot
    ).data  # plasma current
    fvac = client.get("/epq/input/bvacradiusproduct", shot).data  # fvac
    pp_coeffs = client.get(
        "/epq/output/numericaldetails/degreesoffreedom/pprimecoeffs", shot
    ).data  # pprime coefficients (contains values at knots, second deriv. values at knots)
    ffp_coeffs = client.get(
        "/epq/output/numericaldetails/degreesoffreedom/ffprimecoeffs", shot
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
        "/epq/input/numericalcontrols/pp/tens", shot
    ).data  # pprime tension value
    ffp_tension = client.get(
        "/epq/input/numericalcontrols/ffp/tens", shot
    ).data  # ffprime tension value
    pp_knots_raw = client.get(
        "/epq/input/numericalcontrols/pp/knt", shot
    ).data  # pprime knot locations
    ffp_knots_raw = client.get(
        "/epq/input/numericalcontrols/ffp/knt", shot
    ).data  # ffprime knot locations
    pp_knots = pp_knots_raw[:, pp_knots_raw[0, :] > -1]
    ffp_knots = ffp_knots_raw[:, ffp_knots_raw[0, :] > -1]

    # active coil\passive structure currents need to be done carefully
    current_labels = client.get(
        "/epq/input/constraints/pfcircuits/shortname", shot
    ).data  # active/passive coil current names
    currents_values = client.get(
        "/epq/input/constraints/pfcircuits/computed", shot
    ).data  # active/passive coil current values

    # Active coils
    currents = {}
    currents_nonsym = {}
    currents_discrepancy = {}

    with open("../machine_configs/MAST-U//MAST-U_active_coils.pickle", "rb") as file:
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
            currents_discrepancy["Solenoid"] = 0.0
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
                currents_discrepancy[active_coil_name] = polarity * np.abs(
                    np.abs(currents_values[:, indices[0]]) - average_current
                )
                for j in indices:
                    currents_nonsym[efit_names[j]] = currents_values[:, j]

            else:
                currents[active_coil_name] = None

    # passive structures
    with open("../machine_configs/MAST-U//MAST-U_passive_coils.pickle", "rb") as file:
        passive_coils = pickle.load(file)

    # Passive structures
    for i in range(0, len(passive_coils)):
        coil = passive_coils[i]

        if "efitGroup" in coil:
            group_name = coil["efitGroup"]
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
                # currents[coil["name"]] = efit_currents['currents_input'][t,ind]
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
                # currents[coil["name"]] = efit_currents['currents_input'][t,ind]

    return (
        Ip,
        fvac,
        pp_knots,
        ffp_knots,
        pp_values,
        ffp_values,
        pp_values_2nd,
        ffp_values_2nd,
        pp_tension,
        ffp_tension,
        currents,
        currents_nonsym,
        currents_discrepancy,
    )


# ------------
def extract_EFIT_outputs(client, shot, time_indices):
    """
    Load EFIT++ output data for chosen targets below at time indices required.

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


# ------------
def extract_EFIT_outputs_splines(client, shot, time_indices):
    """
    Load EFIT++ output data for chosen targets below at time indices required.

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


# ------------
def get_coil_info(coil_list, coil):
    # Given a coil recovers the coil position and orientation data into
    # a dictionary and appends it to a list of coils.

    # Get coil location
    coordinates = coil["data"]["coordinate"]
    position = np.array([coordinates.r, coordinates.phi, coordinates.z])

    # Get coil orientation
    vector = coil["data"]["orientation"]["unit_vector"]
    orientation = np.array([vector.r, vector.phi, vector.z])

    # Create coil dictionary
    coil_dict = {
        "name": coil.name,
        "position": position,
        "orientation": coil["data"]["orientation"].measurement_direction,
        "orientation_vector": orientation,
    }

    coil_list.append(coil_dict)


# ------------
def get_coils(data, coil_list):
    # Recovers the data for all the pickup coils.

    child_names = [child.name for child in data.children]

    if "data" in child_names:
        get_coil_info(coil_list, data)
    else:
        for child in data.children:
            get_coils(child, coil_list)


# ------------
def get_element_vertices(
    centreR, centreZ, dR, dZ, a1, a2, version=0.1, close_shape=False
):
    """
    Convert EFIT description of rectangles / parallelograms to vertices (used
    passive structures).

                xxxx     ---             xxxxxxxxxxx
            xxxx   x      |            xx        xx
    --- xxxx       x      |          xx        xx
     |  x          x      dZ       xx        xx
     dZ x       xxxx      |      xx        xx
     |  x   xxxx   ^      |    xx        xx   ^
    --- xxxx    A1 )     --- xxxxxxxxxxxx A2 )
        |----dR----|         |-----dR---|

    :param centreR: R-position of centre of shape
    :param centreZ: Z-position of centre of shape
    :param dR: Width
    :param dZ: Height
    :param a1: angle1 as defined above. zero for rectangles
    :param a2: angle2 as defined above. zero for rectangles.
    :param version: geometry version (backwards compatibilty for bug in < V0.1
    :param close_shape: Repeat first vertex to close the shape if set to True
    :return:

    Code courtesy of Lucy Kogan (UKAEA)

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


# ------------
def find_strikepoints(R, Z, psi_total, psi_boundary, limiter):
    """
    This function can be used to find the strikepoints of an equilibrium using
    the:
        - R and Z grids (2D)
        - psi_total (2D) (i.e. the poloidal flux map)
        - psi_boundary (single value)
        - limiter/wall coordinates (N x 2)

    """

    # find contour object for psi_boundary
    cs = plt.contour(R, Z, psi_total, levels=[psi_boundary])
    plt.close()  # this isn't the most elegant but we don't need the plot itself

    # for each item in the contour object there's a list of points in (r,z) (i.e. a line)
    psi_boundary_lines = []
    for i, item in enumerate(cs.allsegs[0]):
        psi_boundary_lines.append(item)

    # use the shapely package to find where each psi_boundary_line intersects the limiter (or not)
    strikes = []
    curve1 = sh.LineString(limiter)
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


# ------------
def Separatrix(R, Z, psi, ntheta, psival=1.0, theta_grid=None, input_opoint=None):
    """Find the R, Z coordinates of the separatrix for equilbrium
    eq. Returns a tuple of (R, Z, R_X, Z_X), where R_X, Z_X are the
    coordinates of the X-point on the separatrix. Points are equally
    spaced in geometric poloidal angle.

    If opoint, xpoint or psi are not given, they are calculated from eq

    eq - Equilibrium object
    opoint - List of O-point tuples of (R, Z, psi)
    xpoint - List of X-point tuples of (R, Z, psi)
    ntheta - Number of points to find
    psi - Grid of psi on (R, Z)
    axis - A matplotlib axis object to plot points on
    input_opoint - a user-chosen magnetic axis from which to find separatrix
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


# ------------
def find_psisurface(psifunc, R, Z, r0, z0, r1, z1, psival=1.0, n=100):
    """
    eq      - Equilibrium object
    (r0,z0) - Start location inside separatrix
    (r1,z1) - Location outside separatrix

    n - Number of starting points to use
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


# ------------
def max_euclidean_distance(points1, points2):
    """
    Calculate the maximum Euclidean distance between corresponding points in two sets.
    Exclude points with 'None' values.
    """
    valid_indices = np.logical_not(
        np.any(np.isnan(points1), axis=1) | np.any(np.isnan(points2), axis=1)
    )
    points1_valid = points1[valid_indices]
    points2_valid = points2[valid_indices]
    if len(points1_valid) == 0 or len(points2_valid) == 0:
        return np.nan
    return np.max(np.sqrt(np.sum((points1_valid - points2_valid) ** 2, axis=1)))


# ------------
def median_euclidean_distance(points1, points2):
    """
    Calculate the maximum Euclidean distance between corresponding points in two sets.
    Exclude points with 'None' values.
    """
    valid_indices = np.logical_not(
        np.any(np.isnan(points1), axis=1) | np.any(np.isnan(points2), axis=1)
    )
    points1_valid = points1[valid_indices]
    points2_valid = points2[valid_indices]
    if len(points1_valid) == 0 or len(points2_valid) == 0:
        return np.nan
    return np.median(np.sqrt(np.sum((points1_valid - points2_valid) ** 2, axis=1)))


# ------------
def separatrix_areas(separatrix_1, separatrix_2):
    """
    This function can be used to find the poloidal area of each separatrix given
    and the similiarity of the two as calculated using equation 8 in Bardsely et al
    2024 (Nuclear Fusion) ("Decoupled magnetic control of spherical tokamak
    divertors via vacuum harmonic constraints"). Inputs:
        - separatrix_1 (and 2): np.array of (n x 2) (r,z) points
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
