"""  Class to implement magnetic probes computations
- sets up probe object, containing the types and locations of the probes
- methods to extract the 'measurements' by each probe from an equilibrium.
"""

import os
import numpy as np
import pickle

from freegs.gradshafranov import Greens, GreensBr, GreensBz
from . import machine_config 



class probe():
    """ 
    Class to implement magnetic probes
    - flux loops: compute psi(R,Z) 
    - pickup coils: compute B(R,phi,Z).nhat (nhat is unit vector orientation of the probe)

    Inputs
    - equilibrium object - contains grid, plasma profile, plasma and coil currents, coil positions.

    Methods 
    - get_coil_currents(eq) : returns current values in all the coils. NB different equilibrium to that used in init
    - greens_all_coils() : returns array with greens function for each coil and probe combination.
    - psi_all_coils(eq): returns list of psi values at each flux loop position. contributions from all coils.
    - psi_from_plasma(eq): returns list of psi values at each flux loop position. contributions from plasma itself.
    - calc_flux_value(eq): returns total flux at each probe position (sum of previous two outputs)

    - 
        
    """
    def __init__(self, eq):
        """ 
        Initialise the following
        - read the probe dictionaries from file
        - set coil positions from equilibrium object
        - set grid size/spacing from equilibrium object
        - create array of greens functions with positions of coils and probes
        """
        # extract probe dictionaries, and set variables for each probe type
        probe_path = os.environ.get("PROBE_PATH",None)
        if probe_path is None:
            raise ValueError("PROBE_PATH environment variable not set.")
    
        with open(probe_path, 'rb') as file:
            probe_dict = pickle.load(file) 

        # set coil lists and probe lists
        self.floops = probe_dict['flux_loops']
        self.pickups = probe_dict['pickups']

        #tokamak is a list of all the coils (active pasive etc.)
        # take coil info from machine_config.py where coil_dict is defined
        self.coil_names = [name  for name in eq.tokamak.getCurrents()]
        self.coil_dict = machine_config.coils_dict

        # self.coils_order = [labeli for i, labeli in enumerate(self.coil_dict.keys())]

        #FLUX LOOPS
        self.floop_pos = np.array([probe['position'] for probe in self.floops])
        self.floop_pos_R = np.array([probe['position'][0] for  probe in self.floops])
        self.floop_pos_Z  = np.array([probe['position'][1] for probe in self.floops])
        self.floop_order = [probe['name']for probe in self.floops]
  

        # create greens function array for all coils, eval at position of flux loop probes
        

        # PICKUP COILS
        # greens functions for pickups - runs over grid G[i][j][k] = greens for coil i, probe pos j, grid



    """
    Things for flux loops
    """
    def greens_psi_single_coil(self,coil_key):
        """
        Create array of greens functions for given coil evaluate at all flux loop positions
        - defines array of greens for each filament at each probe.
        - multplies by polarity and multiplier
        - then sums over filaments to return greens function for probes from a given coil
        """
        #### need to include multiplicities and polarities. here?
        pol = self.coil_dict[coil_key]['polarity'][np.newaxis,:]
        mul = self.coil_dict[coil_key]['multiplier'][np.newaxis,:]
        greens_filaments = Greens(self.coil_dict[coil_key]['coords'][0],
                            self.coil_dict[coil_key]['coords'][1],
                            self.floop_pos_R[:,np.newaxis],
                            self.floop_pos_Z[:,np.newaxis])
        greens_filaments *= pol 
        greens_filaments *= mul 
        greens_psi_coil = np.sum(greens_filaments,axis=1)

        return greens_psi_coil

    def greens_psi_all_coils(self):
        """
        Create 2d array of greens functions for all coils and at all probe positions
        - array[i][j] is greens function for coil i evaluated at probe position j
        """
        array = []
        for key in self.coil_dict.keys():
            array.append(self.greens_psi_single_coil(key))
        return array 
    
    def get_coil_currents(self,eq):
        """ 
        create list of coil currents from the equilibrium
        """
        array_of_coil_currents = np.zeros(len(self.coil_names))
        for i, label in enumerate(self.coil_names): 
            array_of_coil_currents[i] = eq.tokamak[label].current
    
        return array_of_coil_currents 


    def psi_all_coils(self,eq):
        """
        compute flux function summed over all coils. 
        returns array of flux values at the positions of the floop probes
        """
        array_of_coil_currents = self.get_coil_currents(eq)
        green_psi_coils = self.greens_psi_all_coils()

        psi_from_all_coils = np.sum(green_psi_coils * array_of_coil_currents[:,np.newaxis], axis=0) 
        return psi_from_all_coils
        

    def green_psi_plasma(self,eq):
        """ 
        Compute greens function at probes from the plasma currents .
        - plasma current source in grid from solve. grid points contained in eq object 
        """
        rgrid = eq.R
        zgrid = eq.Z 

        greens = Greens(rgrid[:,:,np.newaxis],
                        zgrid[:,:,np.newaxis],
                        self.floop_pos[0][np.newaxis,np.newaxis,:],
                        self.floop_pos[1][np.newaxis, np.newaxis,:])
        
        return greens


    def psi_from_plasma(self,eq):
        """
        Calculate flux function contribution from the plasma
        returns array of flux values from plasma at position of floop probes
        """
        plasma_greens = self.green_psi_plasma(eq)

        plasma_current_distribution = eq._profiles.jtor #toroidal current distribution from plasma equilibrium
    
        Psi_from_plasma = np.sum(plasma_greens*plasma_current_distribution[:,:,np.newaxis], axis=(0,1)) 
        return Psi_from_plasma


    def calculate_flux_value(self,eq):
        """ 
        total flux for all floop probes
        """
        return self.psi_all_coils(eq) + self.psi_from_plasma(eq)


    """
    Things for pickup coils
    """


    def pickup_position(self):
        """
        Return array of positions of pickup coils    
        coordinates are [r,theta,z]
        """
        return[el['position'] for el in self.pickups]

    def pickup_orientation(self):
        """ 
        Return array of orientation vectors for the pickup coil probes
        """
        return [el['orientation_vector'] for el in self.pickups]
    

    # Version1 - plasma compuation uses scipy.interpolate which is slow.
    # def pickup_value(self,eq):
    #     """ 
    #     Computes value of B.n at each pickup probe postition.
    #     Uses Br,Btor,Bz methods defined already for equilibrium object. 
    #     # i'm asumming that these already do the appropriate sums over the coils/grid/plasma etc.
    #     """
    #     BdotN = []
    #     for i, el in enumerate(self.pickups):
    #         # only need r, z positions as B = B(r,z)
    #         R,Z = self.pickup_position[i][0], self.pickup_position[i][2]
    #         orientation_vec = self.pickup_orientation[i]
    #         Bvec = [eq.Br(R,Z),eq.Btor(R,Z),eq.Bz(R,Z)]

    #         BdotN.append(np.dot(Bvec,orientation_vec))
        
    #     return BdotN
    
#  BUILDING FROM SCRATCH
    def Greens_B_Plasma(self,eq):
        """
        3d array of greens functions evaluated for all coils at each point in g
        """
        pass 

    def plasma_current_dist(self,eq):
        """ 
        Compute the current values in the plasma over the grid
        """
        return eq.tokamak._profiles.jtor 
    def calculate_pickup_value(self,eq):
        """ 
        Method to compute and return B.n, for a given pickup coil
        """
        pass