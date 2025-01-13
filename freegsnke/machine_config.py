"""
Checks and/or calculates resistance and inductance machine model based on the provided machine geometry.

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
from deepdiff import DeepDiff
from freegs4e.gradshafranov import Greens, mu0

from .refine_passive import generate_refinement

active_coils_path = os.environ.get("ACTIVE_COILS_PATH", None)
if active_coils_path is None:
    raise ValueError("ACTIVE_COILS_PATH environment variable not set.")


def self_ind_circular_loop(R, r):
    """Self inductance of a circular loop with radius R and width r.

    Parameters
    ----------
    R : float or ndarray
        Radius of the loop
    r : float or ndarray
        Width of the filament

    Returns
    -------
    float or ndarray
        Self inductance values
    """
    return mu0 * R * (np.log(8 * R / r) - 0.5)


def check_self_inductance_and_resistance(coils_dict):
    """Checks if file with pre-calculated resistance and inductance value is in place,
    and if so if the machine model corresponds.

    Parameters
    ----------
    coils_dict : dictionary
        Dictionary containing the FreeGSNKE machine description, i.e. containing vectorised coil info.
        Built by build_machine.py and stored by the machine object at machine.coil_dict

    Returns
    -------
    ndarrays
        resistance and inductance model
    """

    needs_calculating = False

    # Check for existence of resistance matrix, self inductance matrix
    # and coil order data files. Calculates and saves them if they don't exist.
    self_inductance_path = os.environ.get("RESISTANCE_INDUCTANCE_PATH", None)
    if self_inductance_path is None:
        self_inductance_path = os.path.join(
            os.path.split(active_coils_path)[0], "resistance_inductance_data.pickle"
        )
        if not os.path.exists(self_inductance_path):
            needs_calculating = True
        else:
            with open(self_inductance_path, "rb") as f:
                data = pickle.load(f)
    else:
        with open(self_inductance_path, "rb") as f:
            data = pickle.load(f)

    # check input tokamak and retrieved files refer to the same machine,
    if needs_calculating is False:
        check = DeepDiff(data["coils_dict"], coils_dict) == {}
        if check is False:
            needs_calculating = True

    # calculate where necessary
    if needs_calculating:
        print(
            "At least one of the self inductance and resistance data files does"
            " not exist. Calculating them now."
        )

        coils_order, coil_resist, coil_self_ind = calculate_all(coils_dict)
        data_to_save = {
            "coils_dict": coils_dict,
            "coils_order": coils_order,
            "coil_resist": coil_resist,
            "coil_self_ind": coil_self_ind,
        }

        # Save self inductance and resistance matrices, plus list of ordered coils
        with open(self_inductance_path, "wb") as f:
            pickle.dump(data_to_save, f)

    else:
        coils_order = data["coils_order"]
        coil_resist = data["coil_resist"]
        coil_self_ind = data["coil_self_ind"]

    return coils_order, coil_resist, coil_self_ind


def calculate_all(coils_dict):
    """Calculates resistance and inductance model from the provided machine geometry.

    Parameters
    ----------
    coils_dict : dictionary
        Dictionary containing the FreeGSNKE machine description, i.e. containing vectorised coil info.
        Built by build_machine.py and stored by the machine object at machine.coil_dict

    """
    n_coils = len(list(coils_dict.keys()))
    coil_resist = np.zeros(n_coils)
    coil_self_ind = np.zeros((n_coils, n_coils))

    coils_order = []
    for i, labeli in enumerate(coils_dict.keys()):
        coils_order.append(labeli)

    for i, labeli in enumerate(coils_order):
        # for coil-coil flux
        # mutual inductance = 2pi * (sum of all Greens(R_i,Z_i, R_j,Z_j) on n_i*n_j terms, where n is the number of windings)

        # note that while the equation above is valid for active coils, where each filament carries the nominal current,
        # this is not valid for refined passive structures, where each filament carries a factor 1/n_filaments of the total current
        # and for which a mean of the greens (rather than the sum) should be used instead, which is accounted through the 'multiplier'

        coords_i = coils_dict[labeli]["coords"]

        for j, labelj in enumerate(coils_order):
            if j >= i:
                coords_j = coils_dict[labelj]["coords"]

                greenm = Greens(
                    coords_i[0][np.newaxis, :],
                    coords_i[1][np.newaxis, :],
                    coords_j[0][:, np.newaxis],
                    coords_j[1][:, np.newaxis],
                )

                # Recalculate the diagonal terms of greenm using self_ind_circular_loop
                if j == i:
                    # The linear sum dr = dR + dZ (rather than (dR**2+dZ**2/pi)**.5 is mutuated from Fiesta)
                    rr = np.array([coils_dict[labeli]["dR"]]) + np.array(
                        [coils_dict[labeli]["dZ"]]
                    )
                    greenm[np.arange(len(coords_i[0])), np.arange(len(coords_i[0]))] = (
                        self_ind_circular_loop(R=coords_i[0], r=rr) / (2 * np.pi)
                    )

                greenm *= coils_dict[labelj]["polarity"][:, np.newaxis]
                greenm *= coils_dict[labelj]["multiplier"][:, np.newaxis]
                greenm *= coils_dict[labeli]["polarity"][np.newaxis, :]
                greenm *= coils_dict[labeli]["multiplier"][np.newaxis, :]
                coil_self_ind[i, j] = np.sum(greenm)
                coil_self_ind[j, i] = coil_self_ind[i, j]

        # resistance = 2pi * (resistivity/area) * (number of loops * mean_radius)
        # note the multiplier is used as refined passives have number of loops = 1
        coil_resist[i] = (
            coils_dict[labeli]["resistivity_over_area"]
            * coils_dict[labeli]["multiplier"][0]
            * np.sum(coords_i[0])
        )
    coil_self_ind *= 2 * np.pi
    coil_resist *= 2 * np.pi

    return coils_order, coil_resist, coil_self_ind


# Load machine using coils_dict
machine_path = os.path.join(os.path.split(active_coils_path)[0], "machine_data.pickle")
with open(machine_path, "rb") as f:
    coils_dict = pickle.load(f)

# Number of active coils
n_active_coils = np.sum([coils_dict[coil]["active"] for coil in coils_dict])
# Total number of coils
n_coils = len(list(coils_dict.keys()))

# Executes checks and calculations where needed:
coils_order, coil_resist, coil_self_ind = check_self_inductance_and_resistance(
    coils_dict
)
