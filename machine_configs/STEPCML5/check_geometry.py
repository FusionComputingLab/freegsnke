import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 8))
ax.set_xlabel('r')
ax.set_ylabel('z')

with open("active_coils.pickle","rb") as f:
    active_coils = pickle.load(f)

with open("limiter.pickle","rb") as f:
    limiter = pickle.load(f)

# Limiter
ax.plot(
    [l["R"] for l in limiter] + [limiter[0]["R"]],
    [l["Z"] for l in limiter] + [limiter[0]["Z"]],
    c='k',
    ls='--',
    label='Wall and limiter'
)

plt.savefig('geometry.png')