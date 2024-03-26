import numpy as np
from freegs.gradshafranov import Greens


class plasma_current:
    """Implements the plasma circuit equation in projection on $I_{y}^T$:

    $$I_{y}^T/I_p (M_{yy} \dot{I_y} + M_{ye} \dot{I_e} + R_p I_y) = 0$$
    """

    def __init__(self, plasma_grids, Rm1, P, plasma_resistance_1d, Mye):
        """Implements a class for evaluating the plasma circuit equation in projection on $I_{y}^T$:

        $$I_{y}^T/I_p (M_{yy} \dot{I_y} + M_{ye} \dot{I_e} + R_p I_y) = 0$$

        Parameters
        ----------
        plasma_grids : freegsnke.plasma_grids
            Grids object defining the reduced plasma domain.
        Rm12 : np.ndarray
            The diagonal matrix of all metal vessel resistances to the power of -1 ($R^{-1}$).
        P : np.ndarray
            Matrix used to change basis from vessel metal to normal modes. See normal_modes.py
        plasma_resistance_1d : np.ndarray
            Vector on plasma resistance values for all grid points in the reduced plasma domain.
        Mye : np.ndarray
            Matrix of mutual inductances between plasma grid points and all vessel coils.

        """

        self.plasma_grids = plasma_grids
        self.Rm1 = Rm1
        self.P = P
        self.Mye = Mye
        # self.Pm1Rm1 = (P.T) @ Rm1
        # self.MyeRm12V = np.matmul(Mye, self.Rm12V)
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
        # self.Rm12V = self.Rm12 @ V
        # self.MyeRm12V = np.matmul(self.Mye, self.Rm12V)

    def Myy(
        self,
    ):
        """Calculates the matrix of mutual inductances between plasma grid points

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
            self.plasma_grids.plasma_pts[:, np.newaxis, 0],
            self.plasma_grids.plasma_pts[:, np.newaxis, 1],
            self.plasma_grids.plasma_pts[np.newaxis, :, 0],
            self.plasma_grids.plasma_pts[np.newaxis, :, 1],
        )
        return 2 * np.pi * greenm

    # def current_residual(self, red_Iy0, red_Iy1, red_Iydot, Iddot):
    #     """Calculates the residual of the circuit equation, written in terms of the normal modes.

    #     $$I_{\text{residual}} = \frac{I_{y,0}^T}{I_{p,0}} (M_{yy} \dot{I_y} + M_{ye} R^{-1/2} V \dot{I_e} + R_{yy} I_y)/R_{p,0},$$

    #     where the plasma resistivity is found as:

    #     $$R_{p,0} =  I_{y,0}^T/I{p,0} R_{yy} I_{y,0}/I_{p,0}$$

    #     .

    #     Parameters
    #     ----------
    #     red_Iy0 : np.ndarray
    #         The plasma current distribution at timestep $t$ on the reduced plasma grid, 1d.
    #     red_Iy1 : np.ndarray
    #         The plasma current distribution at timestep $t + \delta t$ on the reduced plasma grid, 1d.
    #     red_Iydot : np.ndarray
    #         The time derivative of the current in the plasma on the reduced plasma grid
    #     Iddot : np.ndarray
    #         The time derivative of the current, in terms of the normal modes

    #     Returns
    #     -------
    #     float
    #         $I_{\text{residual}}$, the residual of lumped circuit equation of the plasma, see above.
    #     """
    #     Ip0 = np.sum(red_Iy0)
    #     norm_red_Iy0 = red_Iy0 / Ip0

    #     Fy = np.dot(self.Myy, red_Iydot)
    #     Fe = np.dot(self.MyeRm12V, Iddot)
    #     Fr = self.Ryy * red_Iy1
    #     Ftot = Fy + Fe + Fr

    #     residual = np.dot(norm_red_Iy0, Ftot)
    #     residual *= 1 / np.sum(norm_red_Iy0 * Fr)

    #     return residual
