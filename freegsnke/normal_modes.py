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
            Maximum relative commutator error permitted between passive
            dynamics and the supplied reflection operator.
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
        passive_dynamics = rm1 @ mm
        self.passive_mode_parity = None
        if passive_reflection_operator is None:
            timescales, modes = np.linalg.eig(passive_dynamics)
            frequencies = 1.0 / timescales
            order = np.argsort(frequencies)
            self.w_passive = frequencies[order]
            Pmatrix_passive = modes[:, order]
        else:
            reflection = np.asarray(passive_reflection_operator)
            n_passive = self.n_coils - self.n_active_coils
            if reflection.shape != (n_passive, n_passive):
                raise ValueError(
                    "'passive_reflection_operator' must have shape "
                    f"({n_passive}, {n_passive})."
                )
            identity = np.eye(n_passive)
            if not np.allclose(reflection, reflection.T) or not np.allclose(
                reflection @ reflection, identity
            ):
                raise ValueError(
                    "'passive_reflection_operator' must be a symmetric involution."
                )
            relative_commutator = np.linalg.norm(
                reflection @ passive_dynamics - passive_dynamics @ reflection
            ) / np.linalg.norm(passive_dynamics)
            if relative_commutator > symmetry_tolerance:
                raise ValueError(
                    "Passive dynamics do not commute with up-down reflection. "
                    "Check the machine resistance and inductance data."
                )

            reflection_values, reflection_vectors = np.linalg.eigh(reflection)
            frequencies = []
            modes = []
            parities = []
            # Solve each parity block separately so degenerate even and odd
            # eigenvalues cannot be returned as arbitrary mixed modes.
            for parity in (-1, 1):
                basis = reflection_vectors[:, np.isclose(reflection_values, parity)]
                if not basis.shape[1]:
                    continue
                timescales, reduced_modes = np.linalg.eig(
                    basis.T @ passive_dynamics @ basis
                )
                frequencies.extend(1.0 / timescales)
                modes.extend((basis @ reduced_modes).T)
                parities.extend([parity] * len(timescales))

            frequencies = np.real_if_close(np.asarray(frequencies))
            modes = np.real_if_close(np.asarray(modes).T)
            parities = np.asarray(parities)
            order = np.argsort(frequencies)
            self.w_passive = frequencies[order]
            Pmatrix_passive = modes[:, order]
            self.passive_mode_parity = parities[order]

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
