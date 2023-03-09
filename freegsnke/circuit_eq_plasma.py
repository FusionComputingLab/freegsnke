import numpy as np

from . import plasma_grids

class plasma_current:
    # implements the plasma circuit equation
    # in projection on Iy.T:
    # Iy.T/Ip (Myy Iydot + Mye Iedot + Rp Iy) = 0

    def __init__(self, reference_eq):

        plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(reference_eq.R, reference_eq.Z)
        self.Mye = (plasma_grids.Mey(plasma_pts)).T
        self.Myy = plasma_grids.Myy(plasma_pts)

    
    def reduced_Iy(self, full_Iy):
        Iy = full_Iy[self.mask_inside_limiter]
        return Iy


    def Iydot(self, Iy1, Iy0, dt):
        full_Iydot = (Iy1-Iy0)/dt
        Iydot = self.reduced_Iy(full_Iydot)
        return Iydot


    def current_residual(self,  red_Iy1, Ip,
                                red_Iydot,
                                Iedot,
                                Rp):

        # residual = Iy.T/Ip (Myy/Rp Iydot + Mey/Rp Iedot + Iy)

        # Iydot = self.Iydot(Iy1, Iy0, dt)
        # Iy1 = self.reduced_Iy(Iy1)
        # Ip = np.sum(red_Iy1)

        Fy = self.Myy@red_Iydot
        Fe = self.Mye@Iedot
        Ftot = (Fy+Fe)/Rp
        Ftot += red_Iy1
        residual = np.dot(red_Iy1, Ftot)/Ip
        return residual

