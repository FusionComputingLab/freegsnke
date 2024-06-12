import json
import pickle
import numpy as np
import itertools 

resistivity = 1.55e-8

f = open('coilset.json')
d = json.load(f)

coil_names = list(d.keys())

active_coils_dict = {}
for name in coil_names:

    if name[-1] == 'r':
        circ_name = name[0:-2]
        position = 'lower'
    else:
        circ_name = name
        active_coils_dict[name] = {}
        position = 'upper'


    nr = round(float(d[name]['dx'])/0.1)
    nz = round(float(d[name]['dz'])/0.1)

    if nr == 0:
        nr = nr+1
    if nz == 0:
        nz = nz+1
    
    nfil = nr*nz

    r = np.linspace(d[name]['x'] - float(d[name]['dx'])/2,
                    d[name]['x'] + float(d[name]['dx'])/2,
                    nr)
    
    z = np.linspace(d[name]['z'] - float(d[name]['dz'])/2,
                    d[name]['z'] + float(d[name]['dz'])/2,
                    nz)

    print(z)
    c = np.array(list(itertools.product(r,z)))

    active_coils_dict[circ_name][position] = {
        "R" : list(c[:,0]),
        "Z" : list(c[:,1]),
        "dR": float(d[name]['dx']),
        "dZ": float(d[name]['dz']),
        "resistivity": resistivity,
        "polarity": d[name]['polarity'],
        "multiplier": 1/nfil
    }
f.close()

# add a solenoid to keep the code happy 
active_coils_dict["Solenoid"] = {
    "R": [0.9375]*150,
    "Z": list(np.linspace(-3, 3, 150)),
    "dR": 0.0321,
    "dZ": 0.04,
    "polarity": 1,
    "resistivity": resistivity,
    "multiplier": 1
}


with open("active_coils.pickle",'wb') as f:
    pickle.dump(active_coils_dict, f)

f.close()