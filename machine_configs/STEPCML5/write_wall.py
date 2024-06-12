import csv
import numpy as np
import pickle

R_limiter = []
Z_limiter = []

with open('SPR045_with_vertical_tile.csv') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        vals = row
        R_limiter.append(float(vals[2]))
        Z_limiter.append(float(vals[3]))


limiter = []
for r,z in zip(R_limiter,Z_limiter):
    limiter.append({"R":r,"Z":z})

print(limiter)

with open ("limiter.pickle", "wb") as f:
    pickle.dump(limiter, f)

wall = limiter

with open("wall.pickle", "wb") as f:
    pickle.dump(wall, f)