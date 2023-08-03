import freegs
import numpy as np
from freegs import critical
from . import limiter
from . import plasma_grids

from matplotlib.path import Path



def build_mask_inside_limiter(eq, limiter):
    """Uses the coordinates of points along the edge of the limiter region
    to generate the mask of contained domain points.

    Parameters
    ----------
    eq : freeGS Equilibrium object
        Specifies the domain properties
    limiter : freeGS.machine.Wall object
        Specifies the limiter contour points
    Returns
    -------
    mask_inside_limiter : np.array
        Mask over the full domain of grid points inside the limiter region.
    """
    verts = np.concatenate((np.array(limiter.R)[:,np.newaxis],
                            np.array(limiter.Z)[:,np.newaxis]), axis=-1)
    path = Path(verts)

    points = np.concatenate((eq.R[:,:,np.newaxis], eq.Z[:,:,np.newaxis]), axis=-1)
    
    mask_inside_limiter = path.contains_points(points.reshape(-1,2))
    mask_inside_limiter = mask_inside_limiter.reshape(np.shape(eq.R))
    return mask_inside_limiter


class ConstrainBetapIp(freegs.jtor.ConstrainBetapIp):
    """FreeGS profile class with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, limiter, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        mask_inside_limiter : np.array
            Boole mask, it is True inside the limiter. Same size as full domain grid: (eq.nx, eq.ny)
        """
        super().__init__(*args, **kwargs)
        self.mask_inside_limiter = build_mask_inside_limiter(eq, limiter)
        self.limiter_mask_out = plasma_grids.make_layer_mask(self.mask_inside_limiter, layer_size=1)
        self.limiter_mask_in = plasma_grids.make_layer_mask(np.invert(self.mask_inside_limiter), layer_size=1)
        if not hasattr(self, 'fast'):
            self.Jtor = self._Jtor
        else:
            self.Jtor = self.Jtor_fast

    def _Jtor(self, R, Z, psi, psi_bndry=None):
        """Replaces the original FreeGS Jtor method if FreeGSfast is not available."""
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)

        self.diverted_core_mask = self.jtor>0
        self.psi_bndry, mask, self.limiter_flag = limiter.core_mask_limiter(psi,
                                                                        self.xpt[0][2],
                                                                        self.diverted_core_mask,
                                                                        self.limiter_mask_out,
                                                                        self.limiter_mask_in)
        self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
        return self.jtor

    def Jtor_fast(self, R, Z, psi, psi_bndry=None):
        """Used when FreeGSfast is available."""
        self.diverted_core_mask = super().Jtor_part1(R, Z, psi, psi_bndry)
        if self.diverted_core_mask is None:
            print('no xpt')
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = psi_bndry, None, False
        else: 
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = limiter.core_mask_limiter(psi,
                                                                                                self.psi_bndry,
                                                                                                self.diverted_core_mask,
                                                                                                self.limiter_mask_out,
                                                                                                self.limiter_mask_in)
        self.jtor = super().Jtor_part2(R, Z, psi, self.psi_bndry, self.limiter_core_mask)
        return self.jtor
    
    
    

class ConstrainPaxisIp(freegs.jtor.ConstrainPaxisIp):
    """FreeGS profile class with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, limiter, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : freeGS Equilibrium object
            Specifies the domain properties
        limiter : freeGS.machine.Wall object
            Specifies the limiter contour points
        """
        super().__init__(*args, **kwargs) 
        self.mask_inside_limiter = build_mask_inside_limiter(eq, limiter)
        self.limiter_mask_out = plasma_grids.make_layer_mask(self.mask_inside_limiter, layer_size=1)
        self.limiter_mask_in = plasma_grids.make_layer_mask(np.invert(self.mask_inside_limiter), layer_size=1)
        if not hasattr(self, 'fast'):
            self.Jtor = self._Jtor
        else:
            self.Jtor = self.Jtor_fast

    def _Jtor(self, R, Z, psi, psi_bndry=None):
        """Replaces the original FreeGS Jtor method if FreeGSfast is not available."""
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)

        self.diverted_core_mask = self.jtor>0
        self.psi_bndry, mask, self.limiter_flag = limiter.core_mask_limiter(psi,
                                                                        self.xpt[0][2],
                                                                        self.diverted_core_mask,
                                                                        self.limiter_mask_out,
                                                                        self.limiter_mask_in)
        self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
        return self.jtor

    def Jtor_fast(self, R, Z, psi, psi_bndry=None):
        """Used when FreeGSfast is available."""
        self.diverted_core_mask = super().Jtor_part1(R, Z, psi, psi_bndry)
        if self.diverted_core_mask is None:
            print('no xpt')
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = psi_bndry, None, False
        else: 
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = limiter.core_mask_limiter(psi,
                                                                                                self.psi_bndry,
                                                                                                self.diverted_core_mask,
                                                                                                self.limiter_mask_out,
                                                                                                self.limiter_mask_in)
        self.jtor = super().Jtor_part2(R, Z, psi, self.psi_bndry, self.limiter_core_mask)
        return self.jtor