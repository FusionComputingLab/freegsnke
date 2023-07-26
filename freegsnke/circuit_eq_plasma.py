import numpy as np


class plasma_current:
    """Implements the plasma circuit equation in projection on $I_{y}^T$:
    
    $$I_{y}^T/I_p (M_{yy} \dot{I_y} + M_{ye} \dot{I_e} + R_p I_y) = 0$$
    """

    def __init__(self, plasma_grids, Rm12, V, plasma_resistance_1d, Mye):
        """Implements a class for evaluating the plasma circuit equation in projection on $I_{y}^T$:
    
        $$I_{y}^T/I_p (M_{yy} \dot{I_y} + M_{ye} \dot{I_e} + R_p I_y) = 0$$

        Parameters
        ----------
        plasma_grids : Grids
            Grids object defining the reduced plasma domain.
        Rm12 : np.ndarray
            The diagonal matrix of vessel resistances to the power of -1/2 ($R^{-1/2}$).
        V : np.ndarray
            Vessel structure voltages.
        plasma_resistance_1d : np.ndarray
            Plasma resistivity ($R_p$) in the reduced plasma domain.
        Mye : np.ndarray
            Natrix of mutual inductances between plasma grid points and all vessel coils

        """

        self.Myy = plasma_grids.Myy()
        self.Rm12 = Rm12
        self.V = V
        self.Rm12V = self.Rm12@V
        self.Mye = Mye
        self.MyeRm12V = np.matmul(Mye, self.Rm12V)
        self.Ryy = plasma_resistance_1d
    
    # def reduced_Iy(self, full_Iy):
    #     Iy = full_Iy[self.mask_inside_limiter]
    #     return Iy


    # def Iydot(self, Iy1, Iy0, dt):
    #     full_Iydot = (Iy1-Iy0)/dt
    #     Iydot = self.reduced_Iy(full_Iydot)
    #     return Iydot

    def reset_modes(self, V):
        """Recalculates the factors given a new coil voltage $V$

        Parameters
        ----------
        V : np.ndarray
            Vessel structure voltages.
    
        """
        self.V = V
        self.Rm12V = self.Rm12@V
        self.MyeRm12V = np.matmul(self.Mye, self.Rm12V)


    def current_residual(
            self,  
            red_Iy0, 
            red_Iy1,
            red_Iydot,
            Iddot
            ):
        """
        Solves the circuit equation to get the residual current using: 

        $$I_{\text{residual}} = \frac{I_{y,0}^T}{I_{p,0}} (M_{yy} \dot{I_y} + M_{ye} R^{-1/2} V \dot{I_e} + R_{yy} I_y)/R_{p,0},$$
        
        where the plasma resistivity is found as: 
        
        $$R_{p,0} =  I_{y,0}^T/I{p,0} R_{yy} I_{y,0}/I_{p,0}$$

        Parameters
        ----------
        red_Iy0 : np.ndarray
            The current in the plasma at timestep $t$ on the reduced plasma grid 
        red_Iy1 : np.ndarray
            The current in the plasma at timestep $t + \Delta t$ on the reduced plasma grid 
        red_Iydot : _type_dI
            The time derivative of the current in the plasma on the reduced plasma grid
        Iddot : _type_ 
            The time derivative of the current in the vessel modes

        Returns
        -------
        np.ndarray
            Residual current for all vessel structure. 
        """
        Ip0 = np.sum(red_Iy0)
        norm_red_Iy0 = red_Iy0/Ip0
        
        Fy = np.dot(self.Myy, red_Iydot)
        Fe = np.dot(self.MyeRm12V, Iddot)
        Fr = self.Ryy*red_Iy1
        Ftot = Fy+Fe+Fr

        residual = np.dot(norm_red_Iy0, Ftot)
        residual *= 1/np.sum(norm_red_Iy0*Fr)

        return residual

