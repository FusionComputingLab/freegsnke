"""
Calculates matrix data needed for normal mode decomposition of the vessel.

Copyright 2024 Nicola C. Amorisco, George K. Holt, Kamran Pentland, Adriano Agnello, Alasdair Ross, Matthijs Mars.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
"""

import numpy as np


class mode_decomposition:
    """Sets up the vessel mode decomposition to be used by the dynamic solver(s)"""

    def __init__(self, coil_resist, coil_self_ind, n_coils, n_active_coils):
        """Instantiates the class.
        Matrix data calculated here is used to reformulate the system of circuit eqs,
        primarily in circuit_eq_metal.py

        Parameters
        ----------
        coil_resist : np.array
            1d array of resistance values for all machine conducting elements,
            including both active coils and passive structures.
            Normally calculated (or fetched) in machine_config.py
        coil_self_ind : np.array
            2d matrix of mutual inductances between all pairs of machine conducting elements,
            including both active coils and passive structures
            Normally calculated (or fetched) in machine_config.py
        """

        # check number of coils is compatible with data provided
        check = len(coil_resist) == n_coils
        check *= np.size(coil_self_ind) == n_coils**2
        if check == False:
            raise ValueError(
                "Resistance vector or self inductance matrix are not compatible with number of coils"
            )

        self.n_active_coils = n_active_coils
        self.n_coils = n_coils
        self.coil_resist = coil_resist
        self.coil_self_ind = coil_self_ind

        # active + passive
        self.rm1l_non_symm = np.diag(self.coil_resist**-1.0) @ self.coil_self_ind

        # 1. active coils
        # normal modes are not used for the active coils,
        # but they're calculated here for the check on negative eigenvalues below
        mm1 = np.linalg.inv(
            self.coil_self_ind[: self.n_active_coils, : self.n_active_coils]
        )
        r12 = np.diag(self.coil_resist[: self.n_active_coils] ** 0.5)
        w, v = np.linalg.eig(r12 @ mm1 @ r12)
        ordw = np.argsort(w)
        w_active = w[ordw]

        # 2. passive structures
        r12 = np.diag(self.coil_resist[self.n_active_coils :] ** 0.5)
        mm1 = np.linalg.inv(
            self.coil_self_ind[self.n_active_coils :, self.n_active_coils :]
        )
        w, v = np.linalg.eig(r12 @ mm1 @ r12)
        ordw = np.argsort(w)
        self.w_passive = w[ordw]
        Pmatrix_passive = ((v.T)[ordw]).T

        # A sign convention for the sign of the normal modes is set
        # The way this is achieved is just a choice:
        Pmatrix_passive /= np.sign(np.sum(Pmatrix_passive, axis=0, keepdims=True))

        if np.any(w_active < 0):
            print(
                "Negative eigenvalues in active coils! Please check coil sizes and coordinates."
            )
        if np.any(self.w_passive < 0):
            print(
                "Negative eigenvalues in passive vessel! Please check coil sizes and coordinates."
            )

        # compose full
        self.Pmatrix = np.zeros((self.n_coils, self.n_coils))
        # Pmatrix[:n_active_coils, :n_active_coils] = 1.0*Pmatrix_active
        self.Pmatrix[: self.n_active_coils, : self.n_active_coils] = np.eye(
            self.n_active_coils
        )
        self.Pmatrix[self.n_active_coils :, self.n_active_coils :] = (
            1.0 * Pmatrix_passive
        )
