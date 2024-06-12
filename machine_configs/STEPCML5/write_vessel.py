import json
import pickle
import numpy as np
import itertools 


f = open('vessel.json')
d = json.load(f)

passive_coils = []

for R,Z,dR,dZ,resistivity in zip(d['R'],d['Z'],d['dR'],d['dZ'],d['resistivity']):

    nr = round(dR/0.1)
    nz = round(dZ/0.1)

    if nr == 0:
        nr = nr+1
    if nz == 0:
        nz = nz+1

    nfil = nr*nz

    rfils = np.linspace(R - dR/2,
                    R + dR/2,
                    nr)
    zfils = np.linspace(Z - dZ/2,
                    Z + dZ/2,
                    nz)

    c = np.array(list(itertools.product(rfils,zfils)))
    print(nr,nz)

    for fil in c:
        passive_coils.append({
            "R": fil[0],
            "Z": fil[1],
            "dR": dR/nr,
            "dZ": dZ/nz,
            "resistivity": resistivity
        })

print(passive_coils)
with open("passive_coils.pickle", "wb") as f:
    pickle.dump(passive_coils, f)