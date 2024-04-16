from freegs.gradshafranov import Greens, GreensBr, GreensBz, mu0
import numpy as np

from .refine_passive import generate_refinement, find_area

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import freegs

class PassiveStructure(freegs.coil.Coil):
    """Inherits from freegs.coil.Coil.
    Object to implement passive structures.
    Rather than listing large number of filaments it builds the 
    relevant green function matrix to distribute currents uniformly.
    """

    def __init__(self, R, Z, refine_mode='G', min_refine_per_area=3e3, min_refine_per_lenght=100):
        """Instantiates the Machine, same as freegs.machine.Machine.

        Parameters
        ----------
        R : array
            List of vertex coordinates, defining passive structure polygon.
        Z : array
            List of vertex coordinates, defining passive structure polygon.
        refine_mode : str, optional
            refinement mode for passive structures inputted as polygons, by default 'G' for 'grid'
            Use 'LH' for alternative mode using a Latin Hypercube implementation.
        """
        
        res = find_area(R, Z, 1e3)
        self.area = res[0]
        self.R = res[-2]
        self.Z = res[-1]
        self.Len = np.linalg.norm(res[-3]) 

        self.turns = 1
        self.control = False
        self.current = 0

        self.Rpolygon = np.array(R)
        self.Zpolygon = np.array(Z)
        self.vertices = np.concatenate((self.Rpolygon[:,np.newaxis], self.Zpolygon[:,np.newaxis]), axis=-1)
        self.polygon = Polygon(self.vertices, facecolor = 'k', alpha=.75)

        self.refine_mode = refine_mode
        self.n_refine = int(max(
                                1, self.area * min_refine_per_area, 
                                self.Len * min_refine_per_lenght)
                            )
        self.filaments = self.build_refining_filaments()
        
       

        self.greens = {}

        

    def create_RZ_key(self, R, Z):
        """
        Produces tuple (Rmin,Rmax,Zmin,Zmax,nx,ny) to access correct greens function.
        """
        RZ_key = (np.min(R), np.max(R), np.min(Z), np.max(Z), np.size(R))
        return RZ_key

    def build_refining_filaments(self,):
        """Builds the grid used for the refinement"""

        filaments, area = generate_refinement(self.Rpolygon, self.Zpolygon, self.n_refine, self.refine_mode)
        return filaments

    def build_control_psi(self, R, Z):
        """Builds controlPsi, controlBr, controlBz for a new set of R, Z grids.

        Parameters
        ----------
        R : array
            Grid on which to calculate the greens
        Z : array
            Grid on which to calculate the greens
        """

        greens_psi = Greens(self.filaments[:,0].reshape([-1]+[1]*R.ndim), 
                            self.filaments[:,1].reshape([-1]+[1]*R.ndim), 
                            R[np.newaxis], Z[np.newaxis])
        greens_psi = np.mean(greens_psi, axis=0)

        RZ_key = self.create_RZ_key(R,Z)
        try :
            self.greens[RZ_key]['psi'] = greens_psi
        except :
            self.greens[RZ_key] = {'psi': greens_psi}     
                              
        
    def build_control_br(self, R, Z):
        """Builds controlPsi, controlBr, controlBz for a new set of R, Z grids."""

        greens_br = GreensBr(self.filaments[:,0].reshape([-1]+[1]*R.ndim), 
                            self.filaments[:,1].reshape([-1]+[1]*R.ndim), 
                            R[np.newaxis], Z[np.newaxis])
        greens_br = np.mean(greens_br, axis=0)

        RZ_key = self.create_RZ_key(R,Z)
        try :
            self.greens[RZ_key]['Br'] = greens_br
        except :
            self.greens[RZ_key] = {'Br': greens_br}

    def build_control_bz(self, R, Z):
        """Builds controlPsi, controlBr, controlBz for a new set of R, Z grids."""

        greens_bz = GreensBz(self.filaments[:,0].reshape([-1]+[1]*R.ndim), 
                            self.filaments[:,1].reshape([-1]+[1]*R.ndim), 
                            R[np.newaxis], Z[np.newaxis])
        greens_bz = np.mean(greens_bz, axis=0)

        RZ_key = self.create_RZ_key(R,Z)
        try :
            self.greens[RZ_key]['Bz'] = greens_bz
        except :
            self.greens[RZ_key] = {'Bz': greens_bz}
            

    def controlPsi(self, R, Z):
        """
        Retrieve poloidal flux at (R,Z) due to a unit current
        or calculate where necessary.
        """

        RZ_key = self.create_RZ_key(R,Z)
        try: 
            greens_ = self.greens[RZ_key]['psi']
        except:
            self.build_control_psi(R, Z)
            greens_ = self.greens[RZ_key]['psi']
        return greens_

    def controlBr(self, R, Z):
        """
        Retrieve Br at (R,Z) due to a unit current
        or calculate where necessary.
        """

        RZ_key = self.create_RZ_key(R,Z)
        try: 
            greens_ = self.greens[RZ_key]['Br']
        except:
            self.build_control_br(R, Z)
            greens_ = self.greens[RZ_key]['Br']
        return greens_   

    def controlBz(self, R, Z):
        """
        Retrieve Bz at (R,Z) due to a unit current
        or calculate where necessary.
        """

        RZ_key = self.create_RZ_key(R,Z)
        try: 
            greens_ = self.greens[RZ_key]['Bz']
        except:
            self.build_control_bz(R, Z)
            greens_ = self.greens[RZ_key]['Bz']
        return greens_    
    
    def plot(self, axis=None, show=False):
        """Plot the passive structure polygon"""

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        self.polygon = Polygon(self.vertices, 
                               facecolor = 'grey', 
                               edgecolor='k',
                               linewidth=2)

        axis.add_patch(self.polygon)
        return axis