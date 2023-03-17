import numpy as np

from . import plasma_grids

class plasma_current:
    # implements the plasma circuit equation
    # in projection on Iy.T:
    # Iy.T/Ip (Myy Iydot + Mye Iedot + Rp Iy) = 0

    def __init__(self, reference_eq, Rm12V, plasma_resistance_1d):

        plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(reference_eq.R, reference_eq.Z)
        self.Mye = (plasma_grids.Mey(plasma_pts)).T
        self.Myy = plasma_grids.Myy(plasma_pts)
        self.MyeRm12V = np.matmul(self.Mye, Rm12V)
        self.Ryy = plasma_resistance_1d
    
    # def reduced_Iy(self, full_Iy):
    #     Iy = full_Iy[self.mask_inside_limiter]
    #     return Iy


    # def Iydot(self, Iy1, Iy0, dt):
    #     full_Iydot = (Iy1-Iy0)/dt
    #     Iydot = self.reduced_Iy(full_Iydot)
    #     return Iydot


    def current_residual(self,  red_Iy0, 
                                red_Iy1,
                                red_Iydot,
                                Iddot):

        # residual = Iy0.T/Ip0 (Myy Iydot + Mey Iedot + Ryy Iy)
        # residual here = Iy0.T/Ip0 (Myy Iydot + Mey Rm12 V Iddot + Ryy Iy)/Rp0
        # where Rp0 =  Iy0.T/Ip0 Ryy Iy0/Ip0

        Ip0 = np.sum(red_Iy0)
        
        Fy = np.dot(self.Myy, red_Iydot)
        Fe = np.dot(self.MyeRm12V, Iddot)
        Fr = self.Ryy*red_Iy1
        Ftot = Fy+Fe+Fr

        residual = np.dot(red_Iy0, Ftot)

        # residual *= 1/Ip0 * Ip0**2/np.dot(red_Iy0,red_Iy0*self.Ryy)
        residual *= Ip0/np.sum(red_Iy0*red_Iy0*self.Ryy)

        return residual

