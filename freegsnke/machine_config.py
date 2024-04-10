import os
import pickle
from copy import deepcopy

import numpy as np
from freegs.gradshafranov import Greens
from deepdiff import DeepDiff


active_coils_path = os.environ.get("ACTIVE_COILS_PATH", None)
if active_coils_path is None:
    raise ValueError("ACTIVE_COILS_PATH environment variable not set.")

def Greens_with_depth(Rc, Zc, R, Z, dR, dZ, tol=1e-6):
    mask = np.abs(R - Rc) < tol
    mask *= np.abs(Z - Zc) < tol
    small_r = np.sqrt(dR * dZ / np.pi)
    Rc = np.where(mask, Rc - small_r / 2, Rc)
    greens = Greens(Rc, Zc, R, Z)
    return greens

def check_self_inductance_and_resistance(coils_dict):

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
        check = (DeepDiff(data['coils_dict'], coils_dict)=={})
        if check is False:
            needs_calculating = True

    # calculate where necessary
    if needs_calculating:
        print(
        "At least one of the self inductance and resistance data files does"
        " not exist. Calculating them now."
        )

        coils_order, coil_resist, coil_self_ind = calculate_all(coils_dict)
        data_to_save = {'coils_dict':coils_dict,
                        'coils_order':coils_order, 
                        'coil_resist':coil_resist, 
                        'coil_self_ind':coil_self_ind}
        
        # Save self inductance and resistance matrices, plus list of ordered coils
        with open(self_inductance_path, "wb") as f:
            pickle.dump(data_to_save, f)
    
    else:
        coils_order = data['coils_order']
        coil_resist = data['coil_resist']
        coil_self_ind = data['coil_self_ind']

    return coils_order, coil_resist, coil_self_ind

def calculate_all(coils_dict):
    """_summary_

    Parameters
    ----------
    coils_dict : dictionary
        dictionary containing vectorised coil info. 
        Created in build_machine.py and stored at tokamak.coil_dict

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
        for j, labelj in enumerate(coils_order):
            greenm = Greens_with_depth(
                coils_dict[labeli]["coords"][0][np.newaxis, :],
                coils_dict[labeli]["coords"][1][np.newaxis, :],
                coils_dict[labelj]["coords"][0][:, np.newaxis],
                coils_dict[labelj]["coords"][1][:, np.newaxis],
                np.array([coils_dict[labeli]["dR"]])[:, np.newaxis],
                np.array([coils_dict[labeli]["dZ"]])[:, np.newaxis],
            )

            greenm *= coils_dict[labelj]["polarity"][:, np.newaxis]
            greenm *= coils_dict[labelj]["multiplier"][:, np.newaxis]
            greenm *= coils_dict[labeli]["polarity"][np.newaxis, :]
            greenm *= coils_dict[labeli]["multiplier"][np.newaxis, :]
            coil_self_ind[i, j] = np.sum(greenm)
        # resistance = 2pi * (resistivity/area) * (number of loops * mean_radius)
        # voltages in terms of total applied voltage
        coil_resist[i] = coils_dict[labeli]["resistivity"] * np.sum(
            coils_dict[labeli]["coords"][0]
        )
    coil_self_ind *= 2 * np.pi
    coil_resist *= 2 * np.pi

    return coils_order, coil_resist, coil_self_ind

# Load machine using coils_dict
machine_path = os.path.join(
            os.path.split(active_coils_path)[0], "machine_data.pickle"
        )
with open(machine_path, "rb") as f:
    coils_dict = pickle.load(f)

# Number of active coils
n_active_coils = np.sum([coils_dict[coil]["active"] for coil in coils_dict])
# Total number of coils
n_coils = len(list(coils_dict.keys()))

# Executes checks and calculations where needed:
coils_order, coil_resist, coil_self_ind = check_self_inductance_and_resistance(coils_dict)


# # not actually used in code, user provides relevant values
# # eta_copper = 1.55e-8  # Resistivity in Ohm*m, for active coils
# # eta_steel = 5.5e-7  # In Ohm*m, for passive structures




# # Create dictionary of coils
# coils_dict = {}
# for i, coil_name in enumerate(active_coils):
#     if coil_name == "Solenoid":
#         coils_dict[coil_name] = {}
#         coils_dict[coil_name]["coords"] = np.array(
#             [active_coils[coil_name]["R"], active_coils[coil_name]["Z"]]
#         )
#         coils_dict[coil_name]["polarity"] = np.array(
#             [active_coils[coil_name]["polarity"]] * len(active_coils[coil_name]["R"])
#         )
#         coils_dict[coil_name]["dR"] = active_coils[coil_name]["dR"]
#         coils_dict[coil_name]["dZ"] = active_coils[coil_name]["dZ"]
#         # this is resistivity divided by area
#         coils_dict[coil_name]["resistivity"] = active_coils[coil_name][
#             "resistivity"
#         ] / (active_coils[coil_name]["dR"] * active_coils[coil_name]["dZ"])
#         coils_dict[coil_name]["multiplier"] = np.array(
#             [active_coils[coil_name]["multiplier"]] * len(active_coils[coil_name]["R"])
#         )
#         continue
#     coils_dict[coil_name] = {}

#     coords_R = []
#     for ind in active_coils[coil_name].keys():
#         coords_R.extend(active_coils[coil_name][ind]["R"])

#     coords_Z = []
#     for ind in active_coils[coil_name].keys():
#         coords_Z.extend(active_coils[coil_name][ind]["Z"])
#     coils_dict[coil_name]["coords"] = np.array([coords_R, coords_Z])

#     polarity = []
#     for ind in active_coils[coil_name].keys():
#         polarity.extend(
#             [active_coils[coil_name][ind]["polarity"]]
#             * len(active_coils[coil_name][ind]["R"])
#         )
#     coils_dict[coil_name]["polarity"] = np.array(polarity)

#     multiplier = []
#     for ind in active_coils[coil_name].keys():
#         multiplier.extend(
#             [active_coils[coil_name][ind]["multiplier"]]
#             * len(active_coils[coil_name][ind]["R"])
#         )
#     coils_dict[coil_name]["multiplier"] = np.array(multiplier)

#     coils_dict[coil_name]["dR"] = active_coils[coil_name][
#         list(active_coils[coil_name].keys())[0]
#     ]["dR"]
#     coils_dict[coil_name]["dZ"] = active_coils[coil_name][
#         list(active_coils[coil_name].keys())[0]
#     ]["dZ"]

#     # this is resistivity divided by area
#     coils_dict[coil_name]["resistivity"] = active_coils[coil_name][
#         list(active_coils[coil_name].keys())[0]
#     ]["resistivity"] / (coils_dict[coil_name]["dR"] * coils_dict[coil_name]["dZ"])

# for i, coil in enumerate(passive_coils):
#     tkey = "passive_" + str(i)
#     coils_dict[tkey] = {}
#     coils_dict[tkey]["coords"] = np.array((coil["R"], coil["Z"]))[:, np.newaxis]
#     coils_dict[tkey]["dR"] = coil["dR"]
#     coils_dict[tkey]["dZ"] = coil["dZ"]
#     coils_dict[tkey]["polarity"] = np.array([1])
#     coils_dict[tkey]["multiplier"] = np.array([1])
#     # this is resistivity divided by area
#     coils_dict[tkey]["resistivity"] = coil["resistivity"] / (coil["dR"] * coil["dZ"])

# needs_calculating = check_self_inductance_and_resistance()

#     # Save self inductance and resistance matrices, plus list of ordered coils
#     with open(self_inductance_path, "wb") as f:
#         pickle.dump(coil_self_ind, f)
#     with open(resistance_path, "wb") as f:
#         pickle.dump(coil_resist, f)
#     with open(coils_order_path, "wb") as f:
#         pickle.dump(coils_order, f)


# # # Extract normal modes
# # # 0. active + passive
# # R12 = np.diag(coil_resist**.5)
# # Rm12 = np.diag(coil_resist**-.5)
# # Mm1 = np.linalg.inv(coil_self_ind)
# # lm1r = R12@Mm1@R12
# # rm1l = Rm12@coil_self_ind@Rm12
# # # w,v = np.linalg.eig(R12@(Mm1@R12))
# # # ordw = np.argsort(w)
# # # w_active = w[ordw]
# # # Vmatrix_full = ((v.T)[ordw]).T

# # # 1. active coils
# # w,v = np.linalg.eig(lm1r[:n_active_coils, :n_active_coils])
# # ordw = np.argsort(w)
# # w_active = w[ordw]
# # Vmatrix_active = ((v.T)[ordw]).T

# # # 2. passive structures
# # w,v = np.linalg.eig(lm1r[n_active_coils:, n_active_coils:])
# # ordw = np.argsort(w)
# # w_passive = w[ordw]
# # Vmatrix_passive = ((v.T)[ordw]).T

# # # compose full
# # Vmatrix = np.zeros((n_coils, n_coils))
# # # Vmatrix[:n_active_coils, :n_active_coils] = 1.0*Vmatrix_active
# # Vmatrix[:n_active_coils, :n_active_coils] = np.eye(n_active_coils)
# # Vmatrix[n_active_coils:, n_active_coils:] = 1.0*Vmatrix_passive


# # TODO: Unit tests
# # if __name__ == "__main__":
