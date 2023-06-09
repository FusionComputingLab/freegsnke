import os
from copy import deepcopy
import numpy as np
import pickle
from freegs.gradshafranov import Greens


eta_copper = 1.55e-8 # Resistivity in Ohm*m, for active coils 
eta_steel = 5.5e-7 # In Ohm*m, for passive structures


# Load active and passive structures and wall location
passive_coils_path = os.environ.get('PASSIVE_COILS_PATH', None)
if passive_coils_path is None:
    raise ValueError('PASSIVE_COILS_PATH environment variable not set.')

active_coils_path = os.environ.get('ACTIVE_COILS_PATH', None)
if active_coils_path is None:
    raise ValueError('ACTIVE_COILS_PATH environment variable not set.')

wall_path = os.environ.get('WALL_PATH', None)
if wall_path is None:
    raise ValueError('WALL_PATH environment variable not set.')

with open(passive_coils_path, 'rb') as f:
    passive_coils = pickle.load(f)

with open(active_coils_path, 'rb') as f:
    active_coils = pickle.load(f)

with open(wall_path, 'rb') as f:
    wall = pickle.load(f)


# Number of active coils
n_active_coils = len(active_coils)

# Total number of coils
n_coils = n_active_coils + len(passive_coils)


# Check for existence of resistance matrix and self inductance matrix data
# files. Calculates and saves them if they don't exist.
calculate_self_inductance_and_resistance = False

self_inductance_path = os.environ.get('SELF_INDUCTANCE_PATH', None)
if self_inductance_path is None:
    self_inductance_path = os.path.join(os.path.split(active_coils_path)[0],
                                        'self_inductance.pk')
    if not os.path.exists(self_inductance_path):
        calculate_self_inductance_and_resistance = True
    else:
        with open(self_inductance_path, 'rb') as f:
            coil_self_ind = pickle.load(f)
else:
    with open(self_inductance_path, 'rb') as f:
        coil_self_ind = pickle.load(f)

resistance_path = os.environ.get('RESISTANCE_PATH', None)
if resistance_path is None:
    resistance_path = os.path.join(os.path.split(active_coils_path)[0],
                                   'resistance.pk')
    if not os.path.exists(resistance_path):
        calculate_self_inductance_and_resistance = True
    else:
        with open(resistance_path, 'rb') as f:
            coil_resist = pickle.load(f)
else:
    with open(resistance_path, 'rb') as f:
        coil_resist = pickle.load(f)


if calculate_self_inductance_and_resistance:
    # Create dictionary of coils
    coils_dict = {}
    for i, coil_name in enumerate(active_coils):
        if coil_name == 'solenoid':
            coils_dict[coil_name] = {}
            coils_dict[coil_name]['coords'] = np.array([active_coils[coil_name]["R"], active_coils[coil_name]["Z"]])
            coils_dict[coil_name]['polarity'] = np.array([active_coils[coil_name]["polarity"]] * len(active_coils[coil_name]["R"]))
            coils_dict[coil_name]['dR'] = active_coils[coil_name]["dR"]
            coils_dict[coil_name]['dZ'] = active_coils[coil_name]["dZ"]
            coils_dict[coil_name]['resistivity'] = active_coils[coil_name]["resistivity"]
            continue
        coils_dict[coil_name] = {}

        coords_R = []
        for ind in active_coils[coil_name].keys():
            coords_R.extend(active_coils[coil_name][ind]["R"])

        coords_Z = []
        for ind in active_coils[coil_name].keys():
            coords_Z.extend(active_coils[coil_name][ind]["Z"])
        coils_dict[coil_name]['coords'] = np.array([coords_R, coords_Z])

        polarity = []
        for ind in active_coils[coil_name].keys():
            polarity.extend([active_coils[coil_name][ind]["polarity"]] * len(active_coils[coil_name][ind]["R"]))
        coils_dict[coil_name]['polarity'] = np.array(polarity)

        # coils_dict[coil_name]['polarity'] = np.array([active_coils[coil_name][ind]["polarity"] * len(active_coils[coil_name][ind]["R"]) for ind in active_coils[coil_name].keys()])
        coils_dict[coil_name]['dR'] = active_coils[coil_name][list(active_coils[coil_name].keys())[0]]["dR"]
        coils_dict[coil_name]['dZ'] = active_coils[coil_name][list(active_coils[coil_name].keys())[0]]["dZ"]
        coils_dict[coil_name]['resistivity'] = active_coils[coil_name][list(active_coils[coil_name].keys())[0]]["resistivity"] / (coils_dict[coil_name]['dR'] * coils_dict[coil_name]['dZ'])
    
    for i, coil in enumerate(passive_coils):
        tkey = 'passive_' + str(i)
        coils_dict[tkey] = {}
        coils_dict[tkey]['coords'] = np.array((coil["R"], coil["Z"]))[:, np.newaxis]
        coils_dict[tkey]['polarity'] = np.array([1])
        coils_dict[tkey]['resistivity'] = coil["resistivity"] / (coil["dR"] * coil["dZ"])

    nloops_per_coil = np.zeros(n_coils, dtype=int)
    coil_resist = np.zeros(n_coils)
    coil_self_ind = np.zeros((n_coils, n_coils))

    for i,labeli in enumerate(coils_dict.keys()):
        nloops_per_coil[i] = len(coils_dict[labeli]['coords'][0])    
        #for coil-coil flux
        for j,labelj in enumerate(coils_dict.keys()):
            greenm = Greens(coils_dict[labeli]['coords'][0][np.newaxis,:],
                            coils_dict[labeli]['coords'][1][np.newaxis,:],
                            coils_dict[labelj]['coords'][0][:,np.newaxis],
                            coils_dict[labelj]['coords'][1][:,np.newaxis])
            
            greenm *= coils_dict[labelj]['polarity'][:,np.newaxis]  # TODO: multiplier
            greenm *= coils_dict[labelj]['multiplier'][:,np.newaxis]
            greenm *= coils_dict[labeli]['polarity'][np.newaxis,:]
            greenm *= coils_dict[labeli]['multiplier'][np.newaxis,:]
            coil_self_ind[i,j] = np.sum(greenm)
        # resistance = resistivity/area * number of loops * mean_radius * 2pi
        # voltages in terms of total applied voltage
        coil_resist[i] = coils_dict[labeli]['resistivity']*np.sum(coils_dict[labeli]['coords'][0])

    # Save self inductance and resistance matrices
    with open(self_inductance_path, 'wb') as f:
        pickle.dump(coil_self_ind, f)
    with open(resistance_path, 'wb') as f:
        pickle.dump(coil_resist, f)


# Extract normal modes
# 0. active + passive
R12 = np.diag(coil_resist**.5)
Rm12 = np.diag(coil_resist**-.5)
Mm1 = np.linalg.inv(coil_self_ind)
lm1r = R12@Mm1@R12
rm1l = Rm12@coil_self_ind@Rm12
# w,v = np.linalg.eig(R12@(Mm1@R12))
# ordw = np.argsort(w)
# w_active = w[ordw]
# Vmatrix_full = ((v.T)[ordw]).T

# 1. active coils
w,v = np.linalg.eig(lm1r[:n_active_coils, :n_active_coils])
ordw = np.argsort(w)
w_active = w[ordw]
Vmatrix_active = ((v.T)[ordw]).T

# 2. passive structures
w,v = np.linalg.eig(lm1r[n_active_coils:, n_active_coils:])
ordw = np.argsort(w)
w_passive = w[ordw]
Vmatrix_passive = ((v.T)[ordw]).T

# compose full 
Vmatrix = np.zeros((n_coils, n_coils))
# Vmatrix[:n_active_coils, :n_active_coils] = 1.0*Vmatrix_active
Vmatrix[:n_active_coils, :n_active_coils] = np.eye(n_active_coils)
Vmatrix[n_active_coils:, n_active_coils:] = 1.0*Vmatrix_passive


# TODO: Unit tests
# if __name__ == "__main__":