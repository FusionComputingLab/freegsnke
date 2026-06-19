"""
Checks and/or calculates resistance and inductances matrices
based on a provided machine description.

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


def build_tokamak_R_and_M(tokamak, rebuild=False, changed_coils=None):
    """
    Build the resistance (R) and inductance (M) matrices for the machine within the tokamak
    object. This will construct R and M for the active coils (and any passive structures).

    If already present, they will not be re-calculated.



            # for coil-coil flux
            # mutual inductance = 2pi * (sum of all Greens(R_i,Z_i, R_j,Z_j) on n_i*n_j terms, where n is the number of windings)

            # note that while the equation above is valid for active coils, where each filament carries the nominal current,
            # this is not valid for refined passive structures. Their filament currents are normalized quadrature weights
            # of the total structure current, which is accounted through the 'multiplier'


            # resistance = 2pi * (resistivity/area) * (number of loops * mean_radius)
            # note the multiplier is used as refined passives have number of loops = 1

    Parameters
    ----------
    tokamak : class
        The tokamak object.
    rebuild : bool, optional
        Recalculate and replace existing resistance and inductance matrices
        even when ``tokamak.coil_resist`` or ``tokamak.coil_self_ind`` already
        exist. This should be set when the machine geometry has been updated in
        place.
    changed_coils : list of str, optional
        Coil/circuit labels whose geometry or metadata changed. When existing
        matrices are available and the coil ordering has not changed, only the
        resistance entries and mutual-inductance rows/columns involving these
        labels are recalculated.

    Returns
    -------
    tokamak : class
        Returns an object containing the (new) R and M data.
    """

    has_matrices = hasattr(tokamak, "coil_resist") and hasattr(tokamak, "coil_self_ind")

    if has_matrices and changed_coils is not None and not rebuild:
        changed_coils = [name for name in changed_coils if name in tokamak.coils_list]
        if len(changed_coils) == 0:
            print(
                "Resistance (R) and inductance (M) matrices --> reused; no coil geometry changes detected."
            )
            return

        if len(tokamak.coil_resist) == tokamak.n_coils and np.shape(
            tokamak.coil_self_ind
        ) == (tokamak.n_coils, tokamak.n_coils):
            _update_tokamak_R_and_M_entries(tokamak, changed_coils)
            print(
                "Resistance (R) and inductance (M) matrices --> updated for changed coils only."
            )
            return

    # calculate R and M if they don't exist
    if has_matrices and not rebuild:
        print(
            "Resistance (R) and inductance (M) matrices already exist for these actives (and passives, if present). Check the tokamak object."
        )
    else:

        # storage matrices
        R = np.zeros(tokamak.n_coils)
        M = np.zeros((tokamak.n_coils, tokamak.n_coils))

        # loop over each coil
        for i, name_i in enumerate(tokamak.coils_list):

            # coords of coil i
            coords_i = tokamak.coils_dict[name_i]["coords"]

            # loop over each coil (again) to calcualte inductances
            for j, name_j in enumerate(tokamak.coils_list):
                if j >= i:
                    # coords of coil j
                    coords_j = tokamak.coils_dict[name_j]["coords"]

                    M[i, j] = _calc_mutual_inductance_entry(tokamak, name_i, name_j)
                    M[j, i] = M[i, j]

            # calculate resistance of coil
            R[i] = _calc_resistance_entry(tokamak, name_i)

        # store in tokamak object
        tokamak.coil_resist = R * 2 * np.pi
        tokamak.coil_self_ind = M * 2 * np.pi

        print(
            "Resistance (R) and inductance (M) matrices --> built using actives (and passives if present)."
        )


def _update_tokamak_R_and_M_entries(tokamak, changed_coils):
    """Update only R/M entries affected by changed coil labels."""

    changed_indices = [tokamak.coils_list.index(name) for name in changed_coils]
    for i in changed_indices:
        name_i = tokamak.coils_list[i]
        tokamak.coil_resist[i] = _calc_resistance_entry(tokamak, name_i) * 2 * np.pi

        for j, name_j in enumerate(tokamak.coils_list):
            if j >= i:
                val = _calc_mutual_inductance_entry(tokamak, name_i, name_j)
            else:
                val = _calc_mutual_inductance_entry(tokamak, name_j, name_i)
            tokamak.coil_self_ind[i, j] = val * 2 * np.pi
            tokamak.coil_self_ind[j, i] = val * 2 * np.pi


def _calc_resistance_entry(tokamak, coil_name):
    """Calculate the unscaled resistance entry for one coil label."""

    coords = tokamak.coils_dict[coil_name]["coords"]
    return tokamak.coils_dict[coil_name]["resistivity_over_area"] * np.sum(
        tokamak.coils_dict[coil_name]["multiplier"] * coords[0]
    )


def _calc_mutual_inductance_entry(tokamak, name_i, name_j):
    """Calculate the unscaled mutual-inductance entry for two coil labels."""

    coords_i = tokamak.coils_dict[name_i]["coords"]
    coords_j = tokamak.coils_dict[name_j]["coords"]

    green_m = Greens(
        coords_i[0][np.newaxis, :],
        coords_i[1][np.newaxis, :],
        coords_j[0][:, np.newaxis],
        coords_j[1][:, np.newaxis],
    )

    if name_j == name_i:
        rr = np.array([tokamak.coils_dict[name_i]["dR"]]) + np.array(
            [tokamak.coils_dict[name_i]["dZ"]]
        )
        green_m[np.arange(len(coords_i[0])), np.arange(len(coords_i[0]))] = (
            self_ind_circular_loop(R=coords_i[0], dR=rr) / (2 * np.pi)
        )

    green_m *= tokamak.coils_dict[name_i]["polarity"][np.newaxis, :]
    green_m *= tokamak.coils_dict[name_i]["multiplier"][np.newaxis, :]

    green_m *= tokamak.coils_dict[name_j]["polarity"][:, np.newaxis]
    green_m *= tokamak.coils_dict[name_j]["multiplier"][:, np.newaxis]

    return np.sum(green_m)


def self_ind_circular_loop(R, dR):
    """
    Calculate the self inductance of a circular loop with radius
    R and width dR.

    Parameters
    ----------
    R : float or ndarray
        Radial position of the loop [m].
    dR : float or ndarray
        Radial width of the filament [m].

    Returns
    -------
    float or ndarray
        Self inductance..
    """
    return mu0 * R * (np.log(8 * R / dR) - 0.5)
