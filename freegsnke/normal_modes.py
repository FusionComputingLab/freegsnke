"""
Calculates matrix data needed for normal mode decomposition of the vessel.

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

import numpy as np


class mode_decomposition:
    """Sets up the vessel mode decomposition to be used by the dynamic solver(s)"""

    def __init__(
        self,
        coil_resist,
        coil_self_ind,
        n_coils,
        n_active_coils,
        passive_reflection_operator=None,
        symmetry_tolerance=1e-8,
    ):
        """Instantiates the class.
        Matrix data calculated here is used to reformulate the system of circuit eqs,
        primarily in circuit_eq_metal.py

        Parameters
        ----------
        coil_resist : np.array
            1d array of resistance values for all machine conducting elements,
            including both active coils and passive structures.
        coil_self_ind : np.array
            2d matrix of mutual inductances between all pairs of machine conducting elements,
            including both active coils and passive structures
        passive_reflection_operator : np.array, optional
            Matrix mapping passive currents to their up-down reflected counterparts.
            When supplied, passive modes are classified as even or odd in Z.
        symmetry_tolerance : float, optional
            Maximum departure of a mode's parity from exactly +1 or -1.
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

        # 1. active coils
        # normal modes are not used for the active coils,
        # but they're calculated here for the check on negative eigenvalues below
        r12 = np.diag(self.coil_resist[: self.n_active_coils] ** 0.5)
        mm = self.coil_self_ind[: self.n_active_coils, : self.n_active_coils]
        w, v = np.linalg.eig(r12 @ np.linalg.solve(mm, r12))
        ordw = np.argsort(w)
        w_active = w[ordw]

        # 2. passive structures
        rm1 = np.diag(self.coil_resist[self.n_active_coils :] ** -1)
        mm = self.coil_self_ind[self.n_active_coils :, self.n_active_coils :]
        w, v = np.linalg.eig(rm1 @ mm)
        # w as calculated here are timescales
        # here we switch to frequencies
        w = 1.0 / w
        ordw = np.argsort(w)
        self.w_passive = w[ordw]
        Pmatrix_passive = v[:, ordw]

        self.passive_mode_parity = None
        if passive_reflection_operator is not None:
            reflection = np.asarray(passive_reflection_operator)
            n_passive = self.n_coils - self.n_active_coils
            if reflection.shape != (n_passive, n_passive):
                raise ValueError(
                    "'passive_reflection_operator' must have shape "
                    f"({n_passive}, {n_passive})."
                )
            parity = np.einsum(
                "im,ij,jm->m", Pmatrix_passive, reflection, Pmatrix_passive
            ) / np.einsum("im,im->m", Pmatrix_passive, Pmatrix_passive)
            if np.any(np.abs(np.abs(parity) - 1) > symmetry_tolerance):
                raise ValueError(
                    "Passive normal modes are not purely even or odd in Z. "
                    "Check that the machine resistance and inductance data are "
                    "up-down symmetric."
                )
            self.passive_mode_parity = np.where(parity >= 0, 1, -1)
            Pmatrix_passive = 0.5 * (
                Pmatrix_passive
                + reflection
                @ (Pmatrix_passive * self.passive_mode_parity[np.newaxis, :])
            )
            Pmatrix_passive /= np.linalg.norm(Pmatrix_passive, axis=0)

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
        # self.Pmatrixm1 = np.zeros((self.n_coils, self.n_coils))
        # set active
        self.Pmatrix[: self.n_active_coils, : self.n_active_coils] = np.eye(
            self.n_active_coils
        )
        # self.Pmatrixm1[: self.n_active_coils, : self.n_active_coils] = np.eye(
        #     self.n_active_coils
        # )
        # set passive
        self.Pmatrix[self.n_active_coils :, self.n_active_coils :] = (
            1.0 * Pmatrix_passive
        )
        # self.Pmatrixm1[self.n_active_coils :, self.n_active_coils :] = (
        #     1.0 * Pmatrix_passive_m1
        # )

        # calculate the inverse
        self.Pmatrix_inverse = np.linalg.solve(
            self.Pmatrix.T @ self.Pmatrix, self.Pmatrix.T
        )

    def normal_modes_greens(self, eq_vgreen, mode_matrix=None):
        """
        Calculates the green functions of the vessel normal modes,
        i.e. the psi flux per unit current for each mode.

        Parameters
        ----------
        eq_vgreen : np.array
            the vectorised green functions of each coil.
            Can be found at eq._vgreen. np.shape(eq_vgreen)=(n_coils, nx, ny)
        mode_matrix : np.array, optional
            Transformation from retained mode currents to physical currents.
            Required when calculating Green functions for a reduced basis. If
            omitted, the existing full-basis transformation is used.
        """

        if mode_matrix is not None:
            return np.einsum("im,irz->mrz", mode_matrix, eq_vgreen)

        return np.einsum("mi,irz->mrz", self.Pmatrix_inverse, eq_vgreen)
