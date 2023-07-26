import numpy as np
from . import machine_config

"""This calculates matrix data needed for normal mode decomposition of the vessel.
Resistance data (coil_resist) and metal mutual inductance matrix (coil_self_ind)
are as calculated in machine_config.py 
Matrix data calculated here is used to reformulate the system of circuit eqs,
primarily in circuit_eq_metal.py
"""

# active + passive
R12 = np.diag(machine_config.coil_resist**.5)
Rm12 = np.diag(machine_config.coil_resist**-.5)
Mm1 = np.linalg.inv(machine_config.coil_self_ind)
lm1r = R12@Mm1@R12
rm1l = Rm12@machine_config.coil_self_ind@Rm12

# 1. active coils
# normal modes are not used for the active coils, 
# but they're calculated here in case needed
mm1 = np.linalg.inv(machine_config.coil_self_ind[:machine_config.n_active_coils, :machine_config.n_active_coils])
r12 = np.diag(machine_config.coil_resist[:machine_config.n_active_coils]**.5)
w,v = np.linalg.eig(r12 @ mm1 @ r12)
ordw = np.argsort(w)
w_active = w[ordw]
Vmatrix_active = ((v.T)[ordw]).T

# 2. passive structures
r12 = np.diag(machine_config.coil_resist[machine_config.n_active_coils:]**.5)
mm1 = np.linalg.inv(machine_config.coil_self_ind[machine_config.n_active_coils:, machine_config.n_active_coils:])
w,v = np.linalg.eig(r12 @ mm1 @ r12)
ordw = np.argsort(w)
w_passive = w[ordw]
Vmatrix_passive = ((v.T)[ordw]).T

# a sign convention for the normal modes is set, otherwise same mode could have opposite signs 
# in repeat calculcations and across machines, which may hinder reproducibility
# The way this is achieved is somewhat arbitrary: 
Vmatrix_passive /= np.sign(np.sum(Vmatrix_passive, axis=0, keepdims=True))

if np.any(w_active < 0):
    print('Negative eigenvalues in active coils! Please check coil sizes and coordinates.')
if np.any(w_passive < 0):
    print('Negative eigenvalues in passive vessel! Please check coil sizes and coordinates.')


# compose full 
Vmatrix = np.zeros((machine_config.n_coils, machine_config.n_coils))
# Vmatrix[:n_active_coils, :n_active_coils] = 1.0*Vmatrix_active
Vmatrix[:machine_config.n_active_coils, :machine_config.n_active_coils] = np.eye(machine_config.n_active_coils)
Vmatrix[machine_config.n_active_coils:, machine_config.n_active_coils:] = 1.0*Vmatrix_passive


# TODO: Unit tests
# if __name__ == "__main__":