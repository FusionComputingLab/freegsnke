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


class plasma_current:
    """Implements the plasma circuit equation in projection on $I_{y}^T$:

    $$I_{y}^T/I_p (M_{yy} \dot{I_y} + M_{ye} \dot{I_e} + R_p I_y) = 0$$
    """

    def __init__(self, plasma_pts, Rm1, P, plasma_resistance_1d, Mye):
        """Implements the object dealing with the plasma circuit equation in projection on $I_y$,
        I_y being the plasma toroidal current density distribution:

        $$I_{y}^T/I_p (M_{yy} \dot{I_y} + M_{ye} \dot{I_e} + R_p I_y) = 0$$

        Parameters
        ----------
        plasma_pts : freegsnke.limiter_handler.plasma_pts
            Domain points in the domain that are included in the evolutive calculations.
            A typical choice would be all domain points inside the limiter. Defaults to None.
        Rm1 : np.ndarray
            The diagonal matrix of all metal vessel resistances to the power of -1 ($R^{-1}$).
        P : np.ndarray
            Matrix used to change basis from normal mode currents to vessel metal currents.
        plasma_resistance_1d : np.ndarray
            Vector of plasma resistance values for all grid points in the reduced plasma domain.
            plasma_resistance_1d = 2pi resistivity R/dA for all plasma_pts
        Mye : np.ndarray
            Matrix of mutual inductances between plasma grid points and all vessel coils.

        """

        self.plasma_pts = plasma_pts
        self.Rm1 = Rm1
        self.P = P
        self.Mye = Mye
        self.Ryy = plasma_resistance_1d
        self.Myy_matrix = self.Myy()

    def reset_modes(self, P):
        """Allows a reset of the attributes set up at initialization time following a change
        in the properties of the selected normal modes for the passive structures.

        Parameters
        ----------
        P : np.ndarray
            New change of basis matrix.
        """
        self.P = P

    def Myy(
        self,
    ):
        """Calculates the matrix of mutual inductances between all plasma grid points

        Parameters
        ----------
        plasma_pts : np.ndarray
            Array with R and Z coordinates of all the points inside the limiter

        Returns
        -------
        Myy : np.ndarray
            Array of mutual inductances between plasma grid points
        """
        greenm = Greens(
            self.plasma_pts[:, np.newaxis, 0],
            self.plasma_pts[:, np.newaxis, 1],
            self.plasma_pts[np.newaxis, :, 0],
            self.plasma_pts[np.newaxis, :, 1],
        )
        return 2 * np.pi * greenm
