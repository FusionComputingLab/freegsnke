"""
Defines the plasma_current Object, which handles the lumped parameter model
used as an effective circuit equation for the plasma.

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

from .gradshafranov import Greens

# class plasma_current:
#     """Implements the plasma circuit equation in projection on $I_{y}^T$:

#     $$I_{y}^T/I_p (M_{yy} \dot{I_y} + M_{ye} \dot{I_e} + R_p I_y) = 0$$
#     """

#     def __init__(self, plasma_pts, Rm1, P, plasma_resistance_1d, Mye):
#         """Implements the object dealing with the plasma circuit equation in projection on $I_y$,
#         I_y being the plasma toroidal current density distribution:

#         $$I_{y}^T/I_p (M_{yy} \dot{I_y} + M_{ye} \dot{I_e} + R_p I_y) = 0$$

#         Parameters
#         ----------
#         plasma_pts : freegsnke.limiter_handler.plasma_pts
#             Domain points in the domain that are included in the evolutive calculations.
#             A typical choice would be all domain points inside the limiter. Defaults to None.
#         Rm1 : np.ndarray
#             The diagonal matrix of all metal vessel resistances to the power of -1 ($R^{-1}$).
#         P : np.ndarray
#             Matrix used to change basis from normal mode currents to vessel metal currents.
#         plasma_resistance_1d : np.ndarray
#             Vector of plasma resistance values for all grid points in the reduced plasma domain.
#             plasma_resistance_1d = 2pi resistivity R/dA for all plasma_pts
#         Mye : np.ndarray
#             Matrix of mutual inductances between plasma grid points and all vessel coils.

#         """

#         self.plasma_pts = plasma_pts
#         self.Rm1 = Rm1
#         self.P = P
#         self.Mye = Mye
#         self.Ryy = plasma_resistance_1d
#         self.Myy_matrix = self.Myy()

#     def reset_modes(self, P):
#         """Allows a reset of the attributes set up at initialization time following a change
#         in the properties of the selected normal modes for the passive structures.

#         Parameters
#         ----------
#         P : np.ndarray
#             New change of basis matrix.
#         """
#         self.P = P


#     def Myy(
#         plasma_pts,
#     ):
#         """Calculates the matrix of mutual inductances between all plasma grid points

#         Parameters
#         ----------
#         plasma_pts : np.ndarray
#             Array with R and Z coordinates of all the points inside the limiter

#         Returns
#         -------
#         Myy : np.ndarray
#             Array of mutual inductances between plasma grid points
#         """
#         greenm = Greens(
#             plasma_pts[:, np.newaxis, 0],
#             plasma_pts[:, np.newaxis, 1],
#             plasma_pts[np.newaxis, :, 0],
#             plasma_pts[np.newaxis, :, 1],
#         )
#         return 2 * np.pi * greenm


class Myy_handler:
    """
    Handles all operations involving the Myy matrix.

    Myy is the mutual inductance matrix of the domain grid points. To reduce
    memory usage, the domain on which Myy is built and stored is set
    adaptively, so as to cover the plasma. This object handles that adaptivity.
    """

    def __init__(
        self, limiter_handler: object, layer_size: int = 5, tolerance: int = 3
    ) -> None:
        """
        Instantiate the object.

        Parameters
        ----------
        limiter_handler : object
            FreeGSNKE limiter handler (i.e. ``eq.limiter_handler``). Sets the
            properties of the domain grid and those of the limiter.
        layer_size : int, optional
            Used when recalculating Myy. A layer of `layer_size` pixels is added
            to envelop the mask defined by the plasma. This 'broadened' mask
            defines the pixels included in the Myy matrix.
        tolerance : int, optional
            Used to check if Myy needs recalculating. Myy is not recalculated if
            the mask defined by the plasma region, broadened by `tolerance`
            pixels, is fully contained in the domain of the current Myy matrix.
        """

        limiter_handler.build_reduced_rect_domain()

        self.reduce_rect_domain = limiter_handler.reduce_rect_domain
        self.extract_index_mask = limiter_handler.extract_index_mask
        self.rebuild_map2d = limiter_handler.rebuild_map2d
        self.broaden_mask = limiter_handler.broaden_mask

        self.mask_inside_limiter = limiter_handler.mask_inside_limiter
        self.mask_inside_limiter_red = self.reduce_rect_domain(self.mask_inside_limiter)

        self.idxs_mask_red = self.extract_index_mask(self.mask_inside_limiter_red)

        self.gg = self.grid_greens(
            self.reduce_rect_domain(limiter_handler.eqR),
            self.reduce_rect_domain(limiter_handler.eqZ),
        )

        self.layer_size = layer_size
        self.tolerance = tolerance

    def grid_greens(self, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Calculate the Green's function values on the reduced rectangular domain.

        The values are computed on the minimal rectangular region that fully
        encompasses the limiter, using that the Green's functions are invariant
        under vertical translations.

        Parameters
        ----------
        R : np.ndarray
            Like eq.R, but on the rectangular reduced domain, i.e.
            ``self.reduce_rect_domain(limiter_handler.eqR)``. Only ``R[:, 0]``
            is used.
        Z : np.ndarray
            Like eq.Z, but on the rectangular reduced domain. Uniform spacing
            along the second axis is assumed.

        Returns
        -------
        np.ndarray
            Array of shape (nR, nR, nZ) with entry ``[i, j, k]`` equal to
            2 pi times the Green's function between ``(R[i, 0], k dz)`` and
            ``(R[j, 0], 0)``.
        """

        dz = Z[0, 1] - Z[0, 0]
        nZ = np.shape(Z)[1]

        ggreens = Greens(
            R[:, 0][:, np.newaxis, np.newaxis],
            dz * np.arange(nZ)[np.newaxis, np.newaxis, :],
            R[:, 0][np.newaxis, :, np.newaxis],
            0,
        )

        return 2 * np.pi * ggreens

    def build_mask_from_hatIy(self, hatIy: np.ndarray, layer_size: int) -> np.ndarray:
        """
        Build the mask used by `build_Myy_from_mask` from a hatIy map.

        The mask is broadened by `layer_size` pixels, and the limiter mask is
        taken into account.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on the reduced plasma domain, e.g. inside the limiter.
        layer_size : int
            Number of pixels by which the mask is broadened.

        Returns
        -------
        np.ndarray
            Boolean mask on the reduced rectangular domain, of the same shape
            as ``self.mask_inside_limiter_red``.
        """
        hatIy_mask = hatIy > 0
        hatIy_rect_red = self.rebuild_map2d(
            hatIy_mask, self.mask_inside_limiter_red, self.idxs_mask_red
        )
        hatIy_broad_rect_red = self.broaden_mask(hatIy_rect_red, layer_size=layer_size)
        hatIy_broad_rect_red *= self.mask_inside_limiter_red
        return hatIy_broad_rect_red

    def build_Myy_from_mask(self, mask: np.ndarray) -> None:
        """
        Build the Myy matrix only including domain points in the input mask.

        Sets ``myy_mask_red``, ``outside_myy_mask``, ``idxs_myy_mask_red`` and
        ``myy``.

        Parameters
        ----------
        mask : np.ndarray
            Mask of the domain points to include. Defined on the reduced
            rectangular domain grid, i.e. the smallest rectangular domain around
            the limiter mask (same size as ``self.mask_inside_limiter_red``).
        """
        self.myy_mask_red = mask
        self.outside_myy_mask = np.logical_not(mask)

        nmask = np.sum(mask)

        self.idxs_myy_mask_red = self.extract_index_mask(mask)

        r_idxs = np.tile(
            self.idxs_myy_mask_red[0][:, np.newaxis],
            (1, nmask),
        )
        dz_idxs = np.abs(
            self.idxs_myy_mask_red[1][np.newaxis, :]
            - self.idxs_myy_mask_red[1][:, np.newaxis]
        )

        self.myy = self.gg[r_idxs, r_idxs.T, dz_idxs]

    def force_build_Myy(self, hatIy: np.ndarray) -> None:
        """
        Build the Myy matrix only including domain points in the input vector.

        The vector is broadened by ``layer_size`` pixels and need not be a mask.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on the reduced plasma domain, e.g. inside the limiter.
        """

        hatIy_broad_rect_red = self.build_mask_from_hatIy(
            hatIy, layer_size=self.layer_size
        )
        self.build_Myy_from_mask(hatIy_broad_rect_red)

    def check_Myy(self, hatIy: np.ndarray) -> int:
        """
        Check whether Myy needs rebuilding for the input hatIy.

        hatIy, broadened by ``tolerance`` pixels, is compared with the current
        Myy mask. Note 1: tolerance should be smaller than ``layer_size`` in
        `build_mask_from_hatIy`. Note 2: tolerance should be larger than the
        number of pixels by which the plasma is expected to move every
        timestep of the evolution.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on the reduced plasma domain, e.g. inside the limiter.

        Returns
        -------
        int
            Number of broadened-hatIy pixels lying outside the current Myy
            mask. Nonzero means Myy should be rebuilt.
        """
        hatIy_broad_rect_red = self.build_mask_from_hatIy(
            hatIy, layer_size=self.tolerance
        )
        flag = np.sum(hatIy_broad_rect_red[self.outside_myy_mask])
        return flag

    def dot(self, hatIy: np.ndarray) -> np.ndarray:
        """
        Multiply Myy by a vector defined on the reduced plasma domain.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on the reduced plasma domain, e.g. inside the limiter.

        Returns
        -------
        np.ndarray
            Resulting 1d vector on the same domain.
        """
        # first bring hatIy from the reduced domain to the current myy domain
        hatIy_rect_red = self.rebuild_map2d(
            hatIy, self.mask_inside_limiter_red, self.idxs_mask_red
        )
        hatIy_myy_red = hatIy_rect_red[self.myy_mask_red]

        # perform the dot product
        result = np.dot(self.myy, hatIy_myy_red)

        # bring result back to the reduced plasma domain
        result_rect_red = self.rebuild_map2d(
            result, self.mask_inside_limiter_red, self.idxs_myy_mask_red
        )
        result_red = result_rect_red[self.mask_inside_limiter_red]

        return result_red
