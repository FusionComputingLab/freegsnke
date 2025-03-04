"""
Defines the plasma_current Object, which handles the lumped parameter model 
used as an effective circuit equation for the plasma.

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
from freegs4e.gradshafranov import Greens

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

    def __init__(self, limiter_handler):

        self.mask_inside_limiter = limiter_handler.mask_inside_limiter
        limiter_handler.build_reduced_rect_domain()

        self.reduce_rect_domain = limiter_handler.reduce_rect_domain
        self.extract_index_mask = limiter_handler.extract_index_mask
        self.rebuild_map2d = limiter_handler.rebuild_map2d
        self.broaden_mask = limiter_handler.broaden_mask

        self.gg = self.grid_greens(limiter_handler.eqR_red, limiter_handler.eqZ_red)

        # self.r_idxs = np.tile(
        #     limiter_handler.idxs_mask[0][:, np.newaxis],
        #     (1, len(limiter_handler.plasma_pts)),
        # )
        # self.dz_idxs = np.abs(
        #     limiter_handler.idxs_mask[1][np.newaxis, :]
        #     - limiter_handler.idxs_mask[1][:, np.newaxis]
        # )

    def grid_greens(self, R, Z):

        dz = Z[0, 1] - Z[0, 0]
        nZ = np.shape(Z)[1]

        ggreens = Greens(
            R[:, 0][:, np.newaxis, np.newaxis],
            dz * np.arange(nZ)[np.newaxis, np.newaxis, :],
            R[:, 0][np.newaxis, :, np.newaxis],
            0,
        )

        return 2 * np.pi * ggreens

    def build_mask_from_hatIy(self, hatIy, layer_size=5):
        """Builds the mask that will be used by build_myy_from_mask
        based on the hatIy map. The mask is broadened by a number of pixels
        equal to layer mask. The limiter mask is taken into account.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on reduced plasma domain, e.g. inside the limiter
        layer_size : int, optional
            _description_, by default 3
        """
        hatIy_mask = hatIy > 0
        hatIy_rect_full = self.rebuild_map2d(hatIy_mask)
        hatIy_broad = self.broaden_mask(hatIy_rect_full, layer_size=layer_size)
        hatIy_broad *= self.mask_inside_limiter
        return hatIy_broad

    def build_myy_from_mask(self, mask):
        """Build the Myy matrix only including domain points in the input mask

        Parameters
        ----------
        mask : np.ndarray
            mask of the domain points to include on the full domain grid, e.g. eq.R
        """
        self.myy_mask = mask
        self.outside_myy_mask = np.logical_not(mask)[self.mask_inside_limiter]

        mask_red = self.reduce_rect_domain(mask)
        nmask = np.sum(mask_red)

        idxs_mask_red = self.extract_index_mask(mask_red)

        r_idxs = np.tile(
            idxs_mask_red[0][:, np.newaxis],
            (1, nmask),
        )
        dz_idxs = np.abs(
            idxs_mask_red[1][np.newaxis, :] - idxs_mask_red[1][:, np.newaxis]
        )

        self.myy = self.gg[r_idxs, r_idxs.T, dz_idxs]

    def source_Myy(self, hatIy):
        """Returns the Myy matrix. Resets the domain when the input hatIy
        is not fully inside the current myy_mask

        Parameters
        ----------
        hatIy : hatIy : np.ndarray
            1d vector on reduced plasma domain, e.g. inside the limiter
        """

        if np.sum(hatIy[self.outside_myy_mask]):
            hatIy_mask = self.build_mask_from_hatIy(hatIy)
            self.build_myy_from_mask(hatIy_mask)

        return self.myy

    def compose_Myy(
        self,
    ):

        return self.gg[self.r_idxs, self.r_idxs.T, self.dz_idxs]

    def dot(self, vec):

        return np.dot(self.compose_Myy(), vec)

    def matmul(self, mat):

        return np.matmul(self.compose_Myy(), mat)
