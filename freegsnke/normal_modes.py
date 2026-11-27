"""
Calculates matrix data needed for normal mode decomposition of the vessel.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import numpy as np


def _validate_passive_reflection(reflection: np.ndarray, n_passive: int) -> np.ndarray:
    """Validate a finite symmetric involution before it changes machine state.

    Parameters
    ----------
    reflection : np.ndarray
        Candidate passive reflection permutation matrix.
    n_passive : int
        Number of passive conducting structures.

    Returns
    -------
    np.ndarray
        Validated floating-point reflection matrix of shape
        ``(n_passive, n_passive)``.
    """
    reflection = np.asarray(reflection, dtype=float)
    if reflection.shape != (n_passive, n_passive):
        raise ValueError(
            f"'passive_reflection_operator' must have shape ({n_passive}, {n_passive})."
        )
    if (
        not np.isfinite(reflection).all()
        or not np.allclose(reflection, reflection.T)
        or not np.allclose(reflection @ reflection, np.eye(n_passive))
    ):
        raise ValueError(
            "'passive_reflection_operator' must be a finite symmetric involution."
        )
    return reflection


class mode_decomposition:
    """Set up the vessel mode decomposition used by the dynamic solvers.

    When a symmetric reflection operator is supplied, the passive dynamics are
    decomposed into orthogonal even and odd subspaces before their normal modes
    are calculated. This prevents degenerate eigenvalues from producing modes
    with arbitrary mixed parity.
    """

    def __init__(
        self,
        coil_resist: np.ndarray,
        coil_self_ind: np.ndarray,
        n_coils: int,
        n_active_coils: int,
        passive_reflection_operator: np.ndarray | None = None,
        symmetry_tolerance: float = 1e-8,
    ) -> None:
        """Instantiates the class.
        Matrix data calculated here is used to reformulate the system of circuit eqs,
        primarily in circuit_eq_metal.py

        Parameters
        ----------
        coil_resist : np.ndarray
            1d array of resistance values for all machine conducting elements,
            including both active coils and passive structures.
        coil_self_ind : np.array
            2d matrix of mutual inductances between all pairs of machine conducting elements,
            including both active coils and passive structures
        n_coils : int
            Total number of active and passive conducting elements.
        n_active_coils : int
            Number of active circuits at the start of the current vector.
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
            n_passive = self.n_coils - self.n_active_coils
            reflection = _validate_passive_reflection(
                passive_reflection_operator, n_passive
            )
            relative_commutator = np.linalg.norm(
                reflection @ passive_dynamics - passive_dynamics @ reflection
            ) / np.linalg.norm(passive_dynamics)
            if relative_commutator > symmetry_tolerance:
                raise ValueError(
                    "Passive dynamics do not commute with up-down reflection. "
                    "Check the machine resistance and inductance data."
                )

            # Let A = R_p^-1 L_p and S be passive reflection. For an exactly
            # symmetric machine, SA = AS. Since S is a symmetric involution,
            # its +1 and -1 eigenspaces are orthogonal and invariant under A.
            # Solving Q_+^T A Q_+ and Q_-^T A Q_- separately therefore gives
            # dynamically decoupled modes with exact parity, even when their
            # eigenvalues are degenerate.
            reflection_values, reflection_vectors = np.linalg.eigh(reflection)
            frequencies = []
            modes = []
            parities = []
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

    def normal_modes_greens(
        self, eq_vgreen: np.ndarray, mode_matrix: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Calculate the Green functions of the vessel normal modes.

        If ``I_m`` contains the physical metal currents and ``I_d`` contains
        the normal-mode currents, the current transformation is

        ``I_m = Pmatrix @ I_d``.

        Since the physical-coil Green functions satisfy
        ``psi = I_m.T @ G_m``, substitution gives

        ``psi = I_d.T @ Pmatrix.T @ G_m``.

        The modal Green functions are therefore ``Pmatrix.T @ G_m``.  The
        inverse transformation is instead used to recover modal currents from
        physical currents and must not be applied to the Green functions.

        Parameters
        ----------
        eq_vgreen : np.ndarray
            The vectorised Green functions of each coil, found at
            ``eq._vgreen``, with shape (n_coils, nx, ny).

        mode_matrix : np.ndarray or None, optional
            Transformation from retained mode currents to physical currents.
            Defaults to the full mode matrix.

        Returns
        -------
        np.ndarray
            Green functions of the retained modes, shape (n_modes, nx, ny).
        """

        if mode_matrix is None:
            mode_matrix = self.Pmatrix
        return np.tensordot(mode_matrix, eq_vgreen, axes=([0], [0]))
