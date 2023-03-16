import numpy as np

from . import plasma_grids

class plasma_current:
    # implements the plasma circuit equation
    # in projection on Iy.T:
    # Iy.T/Ip (Myy Iydot + Mye Iedot + Rp Iy) = 0

    def __init__(self, reference_eq, Rm12V):

        plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(reference_eq.R, reference_eq.Z)
        self.Mye = (plasma_grids.Mey(plasma_pts)).T
        self.Myy = plasma_grids.Myy(plasma_pts)
        self.MyeRm12V = np.matmul(self.Mye, Rm12V)

    
    # def reduced_Iy(self, full_Iy):
    #     Iy = full_Iy[self.mask_inside_limiter]
    #     return Iy


    # def Iydot(self, Iy1, Iy0, dt):
    #     full_Iydot = (Iy1-Iy0)/dt
    #     Iydot = self.reduced_Iy(full_Iydot)
    #     return Iydot


    def current_residual(self,  norm_red_Iy1, Ip,
                                red_Iydot,
                                Iddot,
                                Rp):

        # residual = Iy.T/Ip (Myy Iydot + Mey Iedot + Ryy Iy)
        # residual here = Iy.T/Ip (Myy Iydot + Mey Rm12 V Iddot + Ryy Iy)/Rp
        # with Rp = Iy.T/Ip Ryy Iy/Ip

        # norm_red_Iy1 = red_Iy1/Ip
        # Rp = np.dot(norm_red_Iy1, Ryy*norm_red_Iy1)
        Fy = np.dot(self.Myy, red_Iydot)
        Fe = np.dot(self.MyeRm12V, Iddot)
        Ftot = Fy+Fe
        residual = np.dot(norm_red_Iy1, Ftot)/Rp
        residual += Ip
        return residual

