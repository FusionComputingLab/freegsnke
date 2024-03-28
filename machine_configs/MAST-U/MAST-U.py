import pickle

import numpy as np
from freegs.gradshafranov import Greens

eta_copper = 1.55e-8  # resistivity in Ohm*m, for active coils
eta_steel = 5.5e-7  # in Ohm*m, for passive structures

d1_upper_r = [
    0.35275,
    0.36745015,
    0.38215014,
    0.39685005,
    0.35275,
    0.35275,
    0.35275,
    0.35275,
    0.35275039,
    0.36745039,
    0.36745039,
    0.36745039,
    0.36745039,
    0.36745,
    0.38215002,
    0.38215002,
    0.38215002,
    0.38215002,
    0.39685014,
    0.39685014,
    0.39685014,
    0.39685014,
    0.39685008,
    0.41155013,
    0.41155013,
    0.41155013,
    0.41155013,
    0.4115501,
    0.42625037,
    0.42625007,
    0.42625007,
    0.42625007,
    0.42625007,
    0.41155002,
    0.4262501,
]

d1_upper_z = [
    1.60924995,
    1.60924995,
    1.60924995,
    1.60924995,
    1.59455001,
    1.57984996,
    1.5651499,
    1.55044997,
    1.53574991,
    1.53574991,
    1.55044997,
    1.5651499,
    1.57984996,
    1.59455001,
    1.57984996,
    1.5651499,
    1.55044997,
    1.53574991,
    1.53574991,
    1.55044997,
    1.5651499,
    1.57984996,
    1.59455001,
    1.59455001,
    1.57984996,
    1.5651499,
    1.55044997,
    1.53574991,
    1.53574991,
    1.55044997,
    1.5651499,
    1.57984996,
    1.59455001,
    1.60924995,
    1.60924995,
]

d1_lower_z = [x * -1 for x in d1_upper_z]

d2_upper_r = [
    0.60125011,
    0.58655024,
    0.60125017,
    0.60125017,
    0.60125023,
    0.58655,
    0.58655,
    0.57185012,
    0.57185036,
    0.57185042,
    0.55715007,
    0.55715007,
    0.55715001,
    0.54245019,
    0.54245019,
    0.54245001,
    0.52775019,
    0.52775025,
    0.52775025,
    0.57185012,
    0.55715013,
    0.54245007,
    0.52774996,
]

d2_upper_z = [
    1.75705004,
    1.75705004,
    1.74234998,
    1.72765005,
    1.71294999,
    1.71294999,
    1.72765005,
    1.74234998,
    1.72765005,
    1.71294999,
    1.71294999,
    1.72765005,
    1.74234998,
    1.74234998,
    1.72765005,
    1.71294999,
    1.71294999,
    1.72765005,
    1.74234998,
    1.75705004,
    1.75705004,
    1.75705004,
    1.75705004,
]

d2_lower_z = [x * -1 for x in d2_upper_z]

d3_upper_r = [
    0.82854998,
    0.8432501,
    0.84325004,
    0.84325004,
    0.82855022,
    0.82855004,
    0.8285504,
    0.81384999,
    0.81385022,
    0.81385005,
    0.79915011,
    0.79915005,
    0.79915005,
    0.78445005,
    0.78444934,
    0.78445005,
    0.76975012,
    0.76975018,
    0.76975018,
    0.76975006,
    0.78445035,
    0.79915041,
    0.81384987,
]

d3_upper_z = [
    2.00405002,
    2.00405002,
    1.98935008,
    1.97465003,
    1.95995009,
    1.97465003,
    1.98935008,
    1.98935008,
    1.97465003,
    1.95995009,
    1.95995009,
    1.97465003,
    1.98935008,
    1.98935008,
    1.97465003,
    1.95995009,
    1.95995009,
    1.97465003,
    1.98935008,
    2.00405002,
    2.00405002,
    2.00405002,
    2.00405002,
]

d3_lower_z = [x * -1 for x in d3_upper_z]

d5_upper_r = [
    1.90735006,
    1.92205048,
    1.92205,
    1.92205,
    1.92205,
    1.92205,
    1.92205,
    1.90735018,
    1.9073503,
    1.9073503,
    1.9073503,
    1.9073503,
    1.90735006,
    1.89265001,
    1.89265013,
    1.89265013,
    1.89265013,
    1.89265013,
    1.89265001,
    1.87794995,
    1.87795019,
    1.87795019,
    1.87795019,
    1.87795019,
    1.87795019,
    1.87795019,
    1.89265037,
]

d5_upper_z = [
    1.99409997,
    1.99409997,
    1.97940004,
    1.96469998,
    1.95000005,
    1.93529999,
    1.92060006,
    1.9059,
    1.92060006,
    1.93529999,
    1.95000005,
    1.96469998,
    1.97940004,
    1.97940004,
    1.96469998,
    1.95000005,
    1.93529999,
    1.92060006,
    1.9059,
    1.9059,
    1.92060006,
    1.93529999,
    1.95000005,
    1.96469998,
    1.97940004,
    1.99409997,
    1.99409997,
]

d5_lower_z = [x * -1 for x in d5_upper_z]

d6_upper_r = [
    1.30704987,
    1.32175004,
    1.32175004,
    1.32174993,
    1.30705011,
    1.32174993,
    1.30704999,
    1.29235005,
    1.30704999,
    1.29235005,
    1.27765,
    1.27765,
    1.26295006,
    1.27765,
    1.26294994,
    1.24825013,
    1.26294994,
    1.24825001,
    1.24825001,
    1.24825013,
    1.26294994,
    1.27765,
    1.29234993,
]

d6_upper_z = [
    1.44564998,
    1.44564998,
    1.46034992,
    1.47504997,
    1.48974991,
    1.48975003,
    1.47504997,
    1.46034992,
    1.46035004,
    1.47504997,
    1.48975003,
    1.47504997,
    1.46034992,
    1.46034992,
    1.47504997,
    1.48974991,
    1.48975003,
    1.47504997,
    1.46034992,
    1.44564998,
    1.44564998,
    1.44564998,
    1.44564998,
]

d6_lower_z = [x * -1 for x in d6_upper_z]

d7_upper_r = [
    1.54205,
    1.55675006,
    1.55675006,
    1.55675006,
    1.54205012,
    1.55674994,
    1.54205,
    1.52735007,
    1.54204988,
    1.52735007,
    1.51265013,
    1.52734995,
    1.51265001,
    1.49794996,
    1.51265001,
    1.49794996,
    1.48325002,
    1.48325002,
    1.48325002,
    1.48325002,
    1.49795008,
    1.51265001,
    1.52734995,
]

d7_upper_z = [
    1.44564998,
    1.44564998,
    1.46034992,
    1.47504997,
    1.48974991,
    1.48974991,
    1.47504997,
    1.46034992,
    1.46034992,
    1.47504997,
    1.48974991,
    1.48974991,
    1.47504997,
    1.46035004,
    1.46034992,
    1.47504997,
    1.48974991,
    1.47504997,
    1.46035004,
    1.44564998,
    1.44564998,
    1.44564998,
    1.44564998,
]

d7_lower_z = [x * -1 for x in d7_upper_z]

dp_upper_r = [
    0.93285,
    0.94755,
    0.93285,
    0.94755,
    0.96224999,
    0.96224999,
    0.88875002,
    0.90345001,
    0.91815001,
    0.91815001,
    0.90345001,
    0.88875002,
    0.96224999,
    0.94755,
    0.93285,
    0.96224999,
    0.94755,
    0.93285,
    0.91815001,
    0.90345001,
    0.88875002,
    0.88874996,
    0.90345001,
    0.91815001,
]

dp_upper_z = [
    1.48634994,
    1.48634994,
    1.47165,
    1.47165,
    1.48634994,
    1.47165,
    1.47165,
    1.47165,
    1.47165,
    1.48634994,
    1.48634994,
    1.48634994,
    1.51574993,
    1.51574993,
    1.51574993,
    1.50105,
    1.50105,
    1.50105,
    1.51574993,
    1.51574993,
    1.51574993,
    1.50105,
    1.50105,
    1.50105,
]

dp_lower_z = [x * -1 for x in dp_upper_z]

p4_upper_r = [
    1.43500018,
    1.53500021,
    1.51000023,
    1.48500025,
    1.46000016,
    1.43500006,
    1.43500006,
    1.46100008,
    1.43500018,
    1.46100008,
    1.48700011,
    1.4610002,
    1.48700011,
    1.51300013,
    1.48700011,
    1.51300013,
    1.53900015,
    1.51300013,
    1.53900003,
    1.56500018,
    1.53900015,
    1.56500006,
    1.56500006,
]

p4_upper_z = [
    1.04014993,
    1.03714991,
    1.03714991,
    1.03714991,
    1.03714991,
    1.07814991,
    1.1161499,
    1.15414989,
    1.15414989,
    1.1161499,
    1.07814991,
    1.07814991,
    1.1161499,
    1.15414989,
    1.15414989,
    1.1161499,
    1.07814991,
    1.07814991,
    1.1161499,
    1.15414989,
    1.15414989,
    1.1161499,
    1.07814991,
]

p4_lower_z = [x * -1 for x in p4_upper_z]

p5_upper_r = [
    1.58500004,
    1.61000001,
    1.63499999,
    1.65999997,
    1.68499994,
    1.58500004,
    1.58500004,
    1.58500004,
    1.63499999,
    1.63499999,
    1.63499999,
    1.65999997,
    1.65999997,
    1.65999997,
    1.68499994,
    1.68500006,
    1.68500006,
    1.71500003,
    1.71500003,
    1.71500003,
    1.71500003,
    1.6099776,
    1.60997999,
]

p5_upper_z = [
    0.41065004,
    0.41065004,
    0.41065004,
    0.41065004,
    0.41065004,
    0.37165004,
    0.33265004,
    0.29365003,
    0.37165004,
    0.33265004,
    0.29365003,
    0.29365003,
    0.33262005,
    0.37165004,
    0.37165004,
    0.33265004,
    0.29365003,
    0.29365006,
    0.33265004,
    0.37165004,
    0.41065004,
    0.31147972,
    0.35528255,
]

p5_lower_z = [x * -1 for x in p5_upper_z]

p6_upper_r = [
    1.2887001,
    1.2887001,
    1.30900013,
    1.2887001,
    1.30900013,
    1.33414996,
    1.33414996,
    1.35444999,
    1.33414996,
    1.35444999,
]

p6_upper_z = [
    0.99616498,
    0.97586501,
    0.95556498,
    0.95556498,
    0.97586501,
    0.931265,
    0.91096503,
    0.89066499,
    0.89066499,
    0.91096503,
]

p6_lower_z = [x * -1 for x in p6_upper_z]

px_upper_r = [
    0.24849965,
    0.24849975,
    0.24849974,
    0.2344998,
    0.24849974,
    0.24849974,
    0.24849972,
    0.24849972,
    0.24849972,
    0.24849971,
    0.24849971,
    0.24849971,
    0.24849969,
    0.24849969,
    0.24849969,
    0.24849968,
    0.24849968,
    0.24849968,
    0.24849966,
    0.24849966,
    0.24849966,
    0.24849965,
    0.23449969,
    0.23449969,
    0.23449971,
    0.23449971,
    0.23449971,
    0.23449971,
    0.23449972,
    0.23449972,
    0.23449974,
    0.23449974,
    0.23449974,
    0.23449975,
    0.23449975,
    0.23449977,
    0.23449977,
    0.23449977,
    0.23449978,
    0.23449978,
    0.2344998,
    0.2344998,
]

px_upper_z = [
    1.41640627,
    1.03640544,
    1.0554055,
    1.03164983,
    1.07440555,
    1.0934056,
    1.11240554,
    1.13140559,
    1.15040565,
    1.1694057,
    1.18840575,
    1.20740581,
    1.22640586,
    1.24540591,
    1.26440585,
    1.2834059,
    1.30240595,
    1.32140601,
    1.34040606,
    1.35940611,
    1.37840617,
    1.39740622,
    1.41164911,
    1.39264905,
    1.37364912,
    1.35464919,
    1.33564925,
    1.31664932,
    1.29764926,
    1.27864933,
    1.2596494,
    1.24064946,
    1.22164953,
    1.20264947,
    1.18364954,
    1.16464961,
    1.14564967,
    1.12664974,
    1.10764968,
    1.08864975,
    1.06964982,
    1.05064988,
]

px_lower_z = [x * -1 for x in px_upper_z]

# import os
# this_dir, this_filename = os.path.split(__file__)
# passive_path = os.path.join(this_dir,'pass_coils_n.pk')
# with open(passive_path,'rb') as handle:
#     pass_coil_dict = pickle.load(handle)

# from . import populate_cancoils
# coilcans_dict = populate_cancoils.pop_coilcans()
# # dict with key:value entries like  'can_P5lower_7': {'R': 1.7435, 'Z': -0.31025, 'dR': 0.003, 'dZ': 0.0935}
# # is using coilcans, these are all filaments, so use eta_material*2*pi*tc['R']/(tc['dR']*tc['dZ']) for the filament resistance
# multicoilcans_dict = populate_cancoils.pop_multicoilcans()
# # dict with key:values like 'can_P5lower': {'R':[list_of_Rs],'Z':[list_of_Zs],'series':sum(2*pi*R/(dR*dZ))}
# # if using multicoilcans, multiply 'series' by eta_material to get the resistance

# multican=True

# #section of active coil loops
# dRc = 0.0127
# dZc = 0.0127
# these dRc and dZc are not really used below, each piece of metal has its own geometry data, but still

coils_dict = {}

coils_dict["Solenoid"] = {}
coils_dict["Solenoid"]["coords"] = np.array(
    [[0.19475] * 324, np.linspace(-1.581, 1.581, 324)]
)
coils_dict["Solenoid"]["polarity"] = np.array(
    [1] * len(coils_dict["Solenoid"]["coords"][0])
)
coils_dict["Solenoid"]["dR"] = 0.012
coils_dict["Solenoid"]["dZ"] = 0.018

coils_dict["Px"] = {}
coils_dict["Px"]["coords"] = np.array(
    [px_upper_r + px_upper_r, px_upper_z + px_lower_z]
)
coils_dict["Px"]["polarity"] = np.array([1] * len(coils_dict["Px"]["coords"][0]))
coils_dict["Px"]["dR"] = 0.011
coils_dict["Px"]["dZ"] = 0.018

coils_dict["D1"] = {}
coils_dict["D1"]["coords"] = np.array(
    [d1_upper_r + d1_upper_r, d1_upper_z + d1_lower_z]
)
coils_dict["D1"]["polarity"] = np.array([1] * len(coils_dict["D1"]["coords"][0]))
coils_dict["D1"]["dR"] = 0.0127
coils_dict["D1"]["dZ"] = 0.0127

coils_dict["D2"] = {}
coils_dict["D2"]["coords"] = np.array(
    [d2_upper_r + d2_upper_r, d2_upper_z + d2_lower_z]
)
coils_dict["D2"]["polarity"] = np.array([1] * len(coils_dict["D2"]["coords"][0]))
coils_dict["D2"]["dR"] = 0.0127
coils_dict["D2"]["dZ"] = 0.0127

coils_dict["D3"] = {}
coils_dict["D3"]["coords"] = np.array(
    [d3_upper_r + d3_upper_r, d3_upper_z + d3_lower_z]
)
coils_dict["D3"]["polarity"] = np.array([1] * len(coils_dict["D3"]["coords"][0]))
coils_dict["D3"]["dR"] = 0.0127
coils_dict["D3"]["dZ"] = 0.0127

coils_dict["Dp"] = {}
coils_dict["Dp"]["coords"] = np.array(
    [dp_upper_r + dp_upper_r, dp_upper_z + dp_lower_z]
)
coils_dict["Dp"]["polarity"] = np.array([1] * len(coils_dict["Dp"]["coords"][0]))
coils_dict["Dp"]["dR"] = 0.0127
coils_dict["Dp"]["dZ"] = 0.0127

coils_dict["D5"] = {}
coils_dict["D5"]["coords"] = np.array(
    [d5_upper_r + d5_upper_r, d5_upper_z + d5_lower_z]
)
coils_dict["D5"]["polarity"] = np.array([1] * len(coils_dict["D5"]["coords"][0]))
coils_dict["D5"]["dR"] = 0.0127
coils_dict["D5"]["dZ"] = 0.0127

coils_dict["D6"] = {}
coils_dict["D6"]["coords"] = np.array(
    [d6_upper_r + d6_upper_r, d6_upper_z + d6_lower_z]
)
coils_dict["D6"]["polarity"] = np.array([1] * len(coils_dict["D6"]["coords"][0]))
coils_dict["D6"]["dR"] = 0.0127
coils_dict["D6"]["dZ"] = 0.0127

coils_dict["D7"] = {}
coils_dict["D7"]["coords"] = np.array(
    [d7_upper_r + d7_upper_r, d7_upper_z + d7_lower_z]
)
coils_dict["D7"]["polarity"] = np.array([1] * len(coils_dict["D7"]["coords"][0]))
coils_dict["D7"]["dR"] = 0.0127
coils_dict["D7"]["dZ"] = 0.0127

coils_dict["P4"] = {}
coils_dict["P4"]["coords"] = np.array(
    [p4_upper_r + p4_upper_r, p4_upper_z + p4_lower_z]
)
coils_dict["P4"]["polarity"] = np.array([1] * len(coils_dict["P4"]["coords"][0]))
coils_dict["P4"]["dR"] = 0.024
coils_dict["P4"]["dZ"] = 0.037

coils_dict["P5"] = {}
coils_dict["P5"]["coords"] = np.array(
    [p5_upper_r + p5_upper_r, p5_upper_z + p5_lower_z]
)
coils_dict["P5"]["polarity"] = np.array([1] * len(coils_dict["P5"]["coords"][0]))
coils_dict["P5"]["dR"] = 0.024
coils_dict["P5"]["dZ"] = 0.037

coils_dict["P6"] = {}
coils_dict["P6"]["coords"] = np.array(
    [p6_upper_r + p6_upper_r, p6_upper_z + p6_lower_z]
)
coils_dict["P6"]["polarity"] = np.array([1] * len(p6_upper_r) + [-1] * len(p6_upper_r))
coils_dict["P6"]["dR"] = 0.02836
coils_dict["P6"]["dZ"] = 0.02836

# get number of active coils
N_active = len(coils_dict.keys())
# insert resistance-related info:
for key in coils_dict.keys():
    coils_dict[key]["resistivity"] = eta_copper / (
        coils_dict[key]["dR"] * coils_dict[key]["dZ"]
    )  # (dRc*dZc)


import os

# import passive structure details:
this_dir, this_filename = os.path.split(__file__)
passive_path = os.path.join(this_dir, "Fiesta_full_passive.pk")
with open(passive_path, "rb") as handle:
    pass_coils_dict = pickle.load(handle)
pass_coils = pass_coils_dict[0]

for i, coil in enumerate(pass_coils):
    tkey = "pass_" + str(i)
    coils_dict[tkey] = {}
    coils_dict[tkey]["coords"] = coil[:2][:, np.newaxis]
    coils_dict[tkey]["polarity"] = np.array([1])
    coils_dict[tkey]["resistivity"] = coil[-1] / (coil[2] * coil[3])


# calculate coil-coil inductances and coil resistances
nloops_per_coil = np.zeros(len(coils_dict.keys()))
coil_resist = np.zeros(len(coils_dict.keys()))
coil_self_ind = np.zeros((len(coils_dict.keys()), len(coils_dict.keys())))
# for i,labeli in enumerate(coils_dict.keys()):
#     nloops_per_coil[i] = len(coils_dict[labeli]['coords'][0])
#     #for coil-coil flux
#     for j,labelj in enumerate(coils_dict.keys()):
#         greenm = Greens(coils_dict[labeli]['coords'][0][np.newaxis,:],
#                         coils_dict[labeli]['coords'][1][np.newaxis,:],
#                         coils_dict[labelj]['coords'][0][:,np.newaxis],
#                         coils_dict[labelj]['coords'][1][:,np.newaxis])

#         greenm *= coils_dict[labelj]['polarity'][:,np.newaxis]
#         greenm *= coils_dict[labeli]['polarity'][np.newaxis,:]
#         coil_self_ind[i,j] = np.sum(greenm)
# resistance = resistivity/area * number of loops * mean_radius * 2pi
# voltages in terms of total applied voltage
# coil_resist[i] = coils_dict[labeli]['resistivity']*np.sum(coils_dict[labeli]['coords'][0])
# freeGS greens = Mij/2pi
# coil_self_ind *= 2*np.pi
# coil_resist *= 2*np.pi

# active coils are up-down symmetric: no inductance on p6
# coil_self_ind[:11,11] = 0
# coil_self_ind[11,:11] = 0


# # calculations above replaced with final result
coil_self_ind = pass_coils_dict[1]
coil_resist = pass_coils_dict[2]

n_coils = len(coil_resist)


# extract normal modes

# 0. active + passive
R12 = np.diag(coil_resist**0.5)
Rm12 = np.diag(coil_resist**-0.5)
Mm1 = np.linalg.inv(coil_self_ind)
lm1r = R12 @ Mm1 @ R12
rm1l = Rm12 @ coil_self_ind @ Rm12
# w,v = np.linalg.eig(R12@(Mm1@R12))
# ordw = np.argsort(w)
# w_active = w[ordw]
# Vmatrix_full = ((v.T)[ordw]).T


# 1. active coils
w, v = np.linalg.eig(lm1r[:N_active, :N_active])
ordw = np.argsort(w)
w_active = w[ordw]
Vmatrix_active = ((v.T)[ordw]).T


# 2. passive structures
w, v = np.linalg.eig(lm1r[N_active:, N_active:])
ordw = np.argsort(w)
w_passive = w[ordw]
Vmatrix_passive = ((v.T)[ordw]).T


# compose full
Vmatrix = np.zeros((n_coils, n_coils))
# Vmatrix[:N_active, :N_active] = 1.0*Vmatrix_active
Vmatrix[:N_active, :N_active] = np.eye(N_active)
Vmatrix[N_active:, N_active:] = 1.0 * Vmatrix_passive


from freegs.coil import Coil
from freegs.machine import Circuit, Machine, Solenoid, Wall
from freegs.multi_coil import MultiCoil


# define MASTU machine including passive structures
# note that PC has been eliminated entirely (vanilla FreeGS includes it)
def MASTU_wpass():
    """MAST-Upgrade, using MultiCoil to represent coils with different locations
    for each strand.
    """
    coils = [
        ("Solenoid", Solenoid(0.19475, -1.581, 1.581, 324, control=False)),
        # ("Pc", MultiCoil(pc_r, pc_z)),
        (
            "Px",
            Circuit(
                [
                    ("PxU", MultiCoil(px_upper_r, px_upper_z), 1.0),
                    ("PxL", MultiCoil(px_upper_r, px_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D1",
            Circuit(
                [
                    ("D1U", MultiCoil(d1_upper_r, d1_upper_z), 1.0),
                    ("D1L", MultiCoil(d1_upper_r, d1_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D2",
            Circuit(
                [
                    ("D2U", MultiCoil(d2_upper_r, d2_upper_z), 1.0),
                    ("D2L", MultiCoil(d2_upper_r, d2_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D3",
            Circuit(
                [
                    ("D3U", MultiCoil(d3_upper_r, d3_upper_z), 1.0),
                    ("D3L", MultiCoil(d3_upper_r, d3_lower_z), 1.0),
                ]
            ),
        ),
        (
            "Dp",
            Circuit(
                [
                    ("DPU", MultiCoil(dp_upper_r, dp_upper_z), 1.0),
                    ("DPL", MultiCoil(dp_upper_r, dp_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D5",
            Circuit(
                [
                    ("D5U", MultiCoil(d5_upper_r, d5_upper_z), 1.0),
                    ("D5L", MultiCoil(d5_upper_r, d5_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D6",
            Circuit(
                [
                    ("D6U", MultiCoil(d6_upper_r, d6_upper_z), 1.0),
                    ("D6L", MultiCoil(d6_upper_r, d6_lower_z), 1.0),
                ]
            ),
        ),
        (
            "D7",
            Circuit(
                [
                    ("D7U", MultiCoil(d7_upper_r, d7_upper_z), 1.0),
                    ("D7L", MultiCoil(d7_upper_r, d7_lower_z), 1.0),
                ]
            ),
        ),
        (
            "P4",
            Circuit(
                [
                    ("P4U", MultiCoil(p4_upper_r, p4_upper_z), 1.0),
                    ("P4L", MultiCoil(p4_upper_r, p4_lower_z), 1.0),
                ]
            ),
        ),
        (
            "P5",
            Circuit(
                [
                    ("P5U", MultiCoil(p5_upper_r, p5_upper_z), 1.0),
                    ("P5L", MultiCoil(p5_upper_r, p5_lower_z), 1.0),
                ]
            ),
        ),
        (
            "P6",
            Circuit(
                [
                    ("P6U", MultiCoil(p6_upper_r, p6_upper_z), 1.0),
                    ("P6L", MultiCoil(p6_upper_r, p6_lower_z), -1.0),
                ]
            ),
        ),
    ]

    # here we must add the passive-structure coils
    # e.g. ( "pas_1", Coil(R, Z) )
    for i, coil in enumerate(pass_coils):
        tkey = "pass_" + str(i)
        coils.append(
            (tkey, Coil(R=coil[0], Z=coil[1], area=coil[2] * coil[3], control=False))
        )
    #
    #
    rwall = [
        1.56442,
        1.73298,
        1.34848,
        1.0882,
        0.902253,
        0.903669,
        0.533866,
        0.538011,
        0.332797,
        0.332797,
        0.334796,
        0.303115,
        0.305114,
        0.269136,
        0.271135,
        0.260841,
        0.260841,
        0.271135,
        0.269136,
        0.305114,
        0.303115,
        0.334796,
        0.332797,
        0.332797,
        0.538598,
        0.534469,
        0.90563,
        0.904219,
        1.0882,
        1.34848,
        1.73018,
        1.56442,
        1.37999,
        1.37989,
        1.19622,
        1.19632,
        1.05537,
        1.05528,
        0.947502,
        0.905686,
        0.899143,
        0.883388,
        0.867681,
        0.851322,
        0.833482,
        0.826063,
        0.822678,
        0.821023,
        0.820691,
        0.822887,
        0.827573,
        0.839195,
        0.855244,
        0.877567,
        0.899473,
        1.18568,
        1.279,
        1.296,
        1.521,
        1.521,
        1.8,
        1.8,
        1.521,
        1.521,
        1.296,
        1.279,
        1.18568,
        0.899473,
        0.877567,
        0.855244,
        0.839195,
        0.827573,
        0.822887,
        0.820691,
        0.821023,
        0.822678,
        0.826063,
        0.833482,
        0.851322,
        0.867681,
        0.883388,
        0.899143,
        0.905686,
        0.947502,
        1.05528,
        1.05537,
        1.19632,
        1.19622,
        1.37989,
        1.37999,
        1.56442,
    ]

    zwall = [
        1.56424,
        1.67902,
        2.06041,
        2.05946,
        1.87565,
        1.87424,
        1.50286,
        1.49874,
        1.29709,
        1.094,
        1.094,
        0.8475,
        0.8475,
        0.565,
        0.565,
        0.495258,
        -0.507258,
        -0.577,
        -0.577,
        -0.8595,
        -0.8595,
        -1.106,
        -1.106,
        -1.30909,
        -1.5099,
        -1.51403,
        -1.88406,
        -1.88547,
        -2.06614,
        -2.06519,
        -1.68099,
        -1.56884,
        -1.57688,
        -1.57673,
        -1.58475,
        -1.5849,
        -1.59105,
        -1.59091,
        -1.59561,
        -1.59556,
        -1.59478,
        -1.59026,
        -1.58087,
        -1.56767,
        -1.54624,
        -1.52875,
        -1.51517,
        -1.49624,
        -1.47724,
        -1.44582,
        -1.41923,
        -1.38728,
        -1.35284,
        -1.3221,
        -1.30018,
        -1.0138,
        -0.8423,
        -0.8202,
        -0.8202,
        -0.25,
        -0.25,
        0.25,
        0.25,
        0.8156,
        0.8156,
        0.8377,
        1.0092,
        1.29558,
        1.3175,
        1.34824,
        1.38268,
        1.41463,
        1.44122,
        1.47264,
        1.49164,
        1.51057,
        1.52415,
        1.54164,
        1.56307,
        1.57627,
        1.58566,
        1.59018,
        1.59096,
        1.59101,
        1.58631,
        1.58645,
        1.5803,
        1.58015,
        1.57213,
        1.57228,
        1.56424,
    ]

    return Machine(coils, Wall(rwall, zwall))
