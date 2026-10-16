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

from __future__ import annotations

import os
import pickle
from copy import deepcopy

import numpy as np
from deepdiff import DeepDiff

from .gradshafranov import Greens, mu0
from .refine_passive import generate_refinement


def build_tokamak_R_and_M(
    tokamak: object, rebuild: bool = False, changed_coils: list[str] | None = None
) -> None:
    """
    Build the resistance (R) and inductance (M) matrices for the machine within the tokamak
    object. This will construct R and M for the active coils (and any passive structures).

    If already present, they will not be re-calculated.

    Notes
    -----
    - The mutual inductance between coils is 2pi * (sum of all
      Greens(R_i, Z_i, R_j, Z_j) terms weighted by the polarities and
      multipliers of the filaments).
    - This sum is valid for active coils, where each filament carries the
      nominal current. It is not valid as such for refined passive structures,
      where each filament carries a factor 1/n_filaments of the total current
      and a mean of the Greens functions (rather than the sum) is needed; this
      is accounted for through the ``multiplier``.
    - The resistance is 2pi * (resistivity/area) * (number of loops * mean
      radius). The multiplier is used because refined passives have a number
      of loops equal to 1.

    Parameters
    ----------
    tokamak : Machine
        The tokamak object; ``coil_resist`` and ``coil_self_ind`` are set (or
        updated) in place.
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


def append_tokamak_R_and_M_entries(tokamak: object, new_labels: list[str]) -> None:
    """
    Extend the existing resistance (R) and inductance (M) matrices with
    entries for coil labels that have just been appended to the machine.

    Parameters
    ----------
    tokamak : Machine
        The tokamak object. ``tokamak.coils_list`` and ``tokamak.coils_dict``
        must already include ``new_labels``, appended in order at the end of
        ``coils_list``, and ``tokamak.coil_resist``/``tokamak.coil_self_ind``
        must already exist for every other (pre-existing) label.
    new_labels : list of str
        Labels of the newly appended coils, in the order they were appended.

    Notes
    -----
    ``tokamak.coil_resist`` and ``tokamak.coil_self_ind`` are replaced in
    place with the extended matrices.
    """

    old_n = len(tokamak.coil_resist)
    new_n = old_n + len(new_labels)

    R = np.zeros(new_n)
    R[:old_n] = tokamak.coil_resist
    M = np.zeros((new_n, new_n))
    M[:old_n, :old_n] = tokamak.coil_self_ind

    for offset, name_i in enumerate(new_labels):
        i = old_n + offset
        R[i] = _calc_resistance_entry(tokamak, name_i) * 2 * np.pi

        for j, name_j in enumerate(tokamak.coils_list):
            if j > i:
                continue
            val = _calc_mutual_inductance_entry(tokamak, name_j, name_i) * 2 * np.pi
            M[i, j] = val
            M[j, i] = val

    tokamak.coil_resist = R
    tokamak.coil_self_ind = M


def insert_tokamak_R_and_M_entries(
    tokamak: object, index: int, new_labels: list[str]
) -> None:
    """
    Insert resistance (R) and inductance (M) matrix entries for coil labels
    that have just been inserted into the machine at a given position, shifting
    any existing entries at or after that position along.

    Unlike :func:`append_tokamak_R_and_M_entries`, this supports inserting new
    coils before existing ones (e.g. a new active coil, which must be inserted
    immediately before the first passive structure to keep the active/passive
    partition of ``tokamak.coils_list`` contiguous).

    Parameters
    ----------
    tokamak : Machine
        The tokamak object. ``tokamak.coils_list`` and ``tokamak.coils_dict``
        must already include ``new_labels``, inserted in order starting at
        ``index``, and ``tokamak.coil_resist``/``tokamak.coil_self_ind`` must
        already exist for every other (pre-existing) label.
    index : int
        Position at which ``new_labels`` were inserted into ``tokamak.coils_list``.
    new_labels : list of str
        Labels of the newly inserted coils, in the order they were inserted.

    Notes
    -----
    ``tokamak.coil_resist`` and ``tokamak.coil_self_ind`` are replaced in
    place with the extended matrices.
    """

    old_n = len(tokamak.coil_resist)
    n_new = len(new_labels)
    new_n = old_n + n_new

    old_R = tokamak.coil_resist
    old_M = tokamak.coil_self_ind

    R = np.zeros(new_n)
    R[:index] = old_R[:index]
    R[index + n_new :] = old_R[index:]

    M = np.zeros((new_n, new_n))
    M[:index, :index] = old_M[:index, :index]
    M[:index, index + n_new :] = old_M[:index, index:]
    M[index + n_new :, :index] = old_M[index:, :index]
    M[index + n_new :, index + n_new :] = old_M[index:, index:]

    for offset, name_i in enumerate(new_labels):
        i = index + offset
        R[i] = _calc_resistance_entry(tokamak, name_i) * 2 * np.pi

        for j, name_j in enumerate(tokamak.coils_list):
            # Skip only a *later* new coil in this same batch (index <= j < index
            # + n_new): its coupling with name_i will be computed when we reach
            # it as name_i in a later iteration of this loop. Every other coil -
            # including pre-existing ones that now sit at j > i because name_i
            # was inserted before them - must still be computed here.
            if index <= j < index + n_new and j > i:
                continue
            val = _calc_mutual_inductance_entry(tokamak, name_j, name_i) * 2 * np.pi
            M[i, j] = val
            M[j, i] = val

    tokamak.coil_resist = R
    tokamak.coil_self_ind = M


def remove_tokamak_R_and_M_entry(tokamak: object, index: int) -> None:
    """
    Remove the resistance (R) and inductance (M) matrix entries at a given
    coil position.

    Parameters
    ----------
    tokamak : Machine
        The tokamak object, with ``coil_resist``/``coil_self_ind`` already
        built for the machine prior to removal.
    index : int
        Position (row/column) to remove from the existing matrices. This
        should be the coil's index before it is removed from
        ``tokamak.coils_list``.

    Notes
    -----
    ``tokamak.coil_resist`` and ``tokamak.coil_self_ind`` are replaced in
    place with the shrunk matrices.
    """

    tokamak.coil_resist = np.delete(tokamak.coil_resist, index)
    tokamak.coil_self_ind = np.delete(
        np.delete(tokamak.coil_self_ind, index, axis=0), index, axis=1
    )


def _update_tokamak_R_and_M_entries(tokamak: object, changed_coils: list[str]) -> None:
    """
    Update only R/M entries affected by changed coil labels.

    Parameters
    ----------
    tokamak : Machine
        The tokamak object with existing ``coil_resist`` and ``coil_self_ind``
        matrices, which are modified in place.
    changed_coils : list of str
        Labels of the coils whose resistance entries and inductance
        rows/columns are recalculated.
    """

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


def _calc_resistance_entry(tokamak: object, coil_name: str) -> float:
    """
    Calculate the unscaled resistance entry for one coil label.

    Parameters
    ----------
    tokamak : Machine
        The tokamak object providing ``coils_dict``.
    coil_name : str
        Label of the coil.

    Returns
    -------
    float
        Resistance entry, not yet multiplied by the factor 2*pi.
    """

    coords = tokamak.coils_dict[coil_name]["coords"]
    val = (
        tokamak.coils_dict[coil_name]["resistivity_over_area"]
        * tokamak.coils_dict[coil_name]["multiplier"][0]
        * np.sum(coords[0])
    )
    return float(np.asarray(val).squeeze())


def _calc_mutual_inductance_entry(tokamak: object, name_i: str, name_j: str) -> float:
    """
    Calculate the unscaled mutual-inductance entry for two coil labels.

    For ``name_i == name_j`` the diagonal (self-inductance) terms use
    `self_ind_circular_loop`.

    Parameters
    ----------
    tokamak : Machine
        The tokamak object providing ``coils_dict``.
    name_i : str
        Label of the first coil.
    name_j : str
        Label of the second coil.

    Returns
    -------
    float
        Inductance entry, not yet multiplied by the factor 2*pi.
    """

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


def self_ind_circular_loop(
    R: float | np.ndarray, dR: float | np.ndarray
) -> float | np.ndarray:
    """
    Calculate the self inductance of a circular loop with radius
    R and width dR.

    Parameters
    ----------
    R : float or np.ndarray
        Radial position of the loop [m].
    dR : float or np.ndarray
        Radial width of the filament [m].

    Returns
    -------
    float or np.ndarray
        Self inductance in H.
    """
    return mu0 * R * (np.log(8 * R / dR) - 0.5)
