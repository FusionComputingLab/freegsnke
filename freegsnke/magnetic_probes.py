"""  Class to implement magnetic probes computations
- sets up probe object, containing the types and locations of the probes
- methods to extract the 'measurements' by each probe from an equilibrium.
"""

import os
import pickle
import numpy as np


from freegs.gradshafranov import Greens, GreensBr, GreensBz
from . import machine_config



class Probe():
    """ 
    Class to implement magnetic probes:
    - flux loops: compute psi(R,Z) 
    - pickup coils: compute B(R,phi,Z).nhat (nhat is unit vector orientation of the probe)

    Inputs:
    - equilibrium object - contains grid, plasma profile, plasma and coil currents, coil positions.
    N.B:- in init the eq object provides machine /domain/position information
        - in methods the equilibrium provides currents and other aspects that evolve in solve().

    Attributes:
    - floops,pickups = dictionaries with name, position, orientation of the probes
    - floops_positions etc.  = extract individual arrays of positions, orientations etc.
    - greens_psi = greens functions for psi, evaluated at flux loop positions
    - greens_br/bz = greens functions for Br and Bz, evaluated at pickup coil positions
    - greens_psi_plasma = greens functions for psi from plasma current, evaluated at flux loop positions
    - greens_brbz_plasma = greens functions for Br and Bz from plasma, evaluated at pickup coil positions

    Methods:
    - get_coil_currents(eq): returns current values in all the coils from equilibrium object.
    - get_plasma_current(eq): returns toroidal current values at each plasma grid point, taken from equilibrium input.
    - greens_all_coils(): returns array with greens function for each coil and probe combination.
    - psi_all_coils(eq): returns list of psi values at each flux loop position, summed over all coils.
    - psi_from_plasma(eq): returns list of psi values at each flux loop position from plasma itself.
    - calc_flux_value(eq): returns total flux at each probe position (sum of previous two outputs).

    - Br(eq)/ Bz(eq) : computes radial/z component of magnetic field, sum of coil and plasma contributions
    - Btor(eq) : extracts toroidal magnetic field (outside of plasma), evaluated at 
        
    """
    def __init__(self, eq):
        """ 
        Initialise the following
        - read the probe dictionaries from file
        - set probe positions and orientations
        - set coil positions from equilibrium object
        - create arrays of greens functions with positions of coils/plasma currents and probes
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
        # positions, number of probes, ordering 
        self.floop_pos = np.array([probe['position'] for probe in self.floops])
        self.number_floops = np.shape(self.floop_pos)[0]        #number of probes
        self.floop_order = [probe['name']for probe in self.floops]

        # # Initilaise Greens functions Gpsi
        self.greens_psi_coils_floops = self.greens_psi_all_coils(eq)
        self.greens_psi_plasma_floops = self.green_psi_plasma(eq)

        # # PICKUP COILS
        # # Positions and orientations - 3d vectors of [R, theta, Z]
        self.pickup_pos = np.array([el['position'] for el in self.pickups])
        self.pickup_or = np.array([el['orientation_vector'] for el in self.pickups])
        self.number_pickups = np.shape(self.pickup_pos)[0]
        self.pickup_order = [probe['name']for probe in self.pickups]

        # # Initialise greens functions for pickups
        self.greens_br_coils_pickup, self.greens_bz_coils_pickup = self.greens_BrBz_all_coils('pickups')
        self.greens_br_plasma_pickup, self.greens_bz_plasma_pickup = self.greens_BrBz_plasma(eq,'pickups')

        # Other probes - to add in future...


    """
    Things for all probes
    - coil current array
    - plasma current array
    """

    def get_coil_currents(self,eq):
        """ 
        create list of coil currents from the equilibrium
        """
        array_of_coil_currents = np.zeros(len(self.coil_names))
        for i, label in enumerate(self.coil_names):
            array_of_coil_currents[i] = eq.tokamak[label].current

        # could use eq.tokamak.getcurrents() instead
        return array_of_coil_currents

    def get_plasma_current(self,eq):
        """
        equilibirium object contains toroidal current distribution, over the grid positions.
        """
        plasma_current_distribution = eq._profiles.jtor

        return plasma_current_distribution


    """
    Things for flux loops
    """
    def greens_psi_single_coil(self,coil_key, probe = 'floops'):
        """
        Create array of greens functions for given coil evaluate at all probe positions
        - pos_R and pos_Z are arrays of R,Z coordinates of the probes 
        - defines array of greens for each filament at each probe.
        - multiplies by polarity and multiplier
        - then sums over filaments to return greens function for probes from a given coil
        """
        pos_R = self.floop_pos[:,0]
        pos_Z = self.floop_pos[:,1]

        pol = self.coil_dict[coil_key]['polarity'][np.newaxis,:]
        mul = self.coil_dict[coil_key]['multiplier'][np.newaxis,:]
        greens_filaments = 2* np.pi* Greens(self.coil_dict[coil_key]['coords'][0][np.newaxis,:],
                            self.coil_dict[coil_key]['coords'][1][np.newaxis,:],
                            pos_R[:,np.newaxis],
                            pos_Z[:,np.newaxis])
        greens_filaments *= pol
        greens_filaments *= mul
        greens_psi_coil = np.sum(greens_filaments,axis=1)

        return greens_psi_coil

    def greens_psi_all_coils(self, probe = 'floops'):
        """
        Create 2d array of greens functions for all coils and at all probe positions
        - array[i][j] is greens function for coil i evaluated at probe position j
        """
            
        array = np.array([]).reshape(0,self.number_floops)
        for key in self.coil_dict.keys():
            array = np.vstack((array, self.greens_psi_single_coil(key,probe)))
        return array

    def psi_floop_all_coils(self,eq, probe = 'floops'):
        """
        compute flux function summed over all coils. 
        returns array of flux values at the positions of the floop probes
        """
        array_of_coil_currents = self.get_coil_currents(eq)
        if probe == 'floops':
            greens = self.greens_psi_coils_floops

        psi_from_all_coils = np.sum(greens * array_of_coil_currents[:,np.newaxis], axis=0)
        self.floop_psi = psi_from_all_coils
        return psi_from_all_coils

    def green_psi_plasma(self,eq,probe = 'floops'):
        """ 
        Compute greens function at probes from the plasma currents .
        - plasma current source in grid from solve. grid points contained in eq object 
        """
        if probe == 'floops':
            pos_R = self.floop_pos[:,0]
            pos_Z = self.floop_pos[:,0]

        rgrid = eq.R # FOR LATER: can speed up only calcualte in plasma, J = 0 outiside
        zgrid = eq.Z
        greens = 2 * np.pi *Greens(rgrid[:,:,np.newaxis],
                        zgrid[:,:,np.newaxis],
                        pos_R[np.newaxis,np.newaxis,:],
                        pos_Z[np.newaxis, np.newaxis,:])

        return greens

    def psi_from_plasma(self,eq,probe = 'floops'):
        """
        Calculate flux function contribution from the plasma
        returns array of flux values from plasma at position of floop probes
        """
        plasma_current_distribution = eq._profiles.jtor #toroidal current distribution from plasma equilibrium

        if probe == 'floops':
            plasma_greens = self.greens_psi_plasma_floops

        psi_from_plasma = np.sum(plasma_greens*plasma_current_distribution[:,:,np.newaxis], axis=(0,1))
        return psi_from_plasma


    def calculate_fluxloop_value(self,eq):
        """ 
        total flux for all floop probes
        """

        return self.psi_floop_all_coils(eq) + self.psi_from_plasma(eq)

    """
    Things for pickup coils
    """

    def greens_BrBz_single_coil(self,coil_key,probe = 'pickups'):
        """
        Create array of greens functions for given coil evaluate at all pickup positions
        - defines array of greens for each filament at each probe.
        - multiplies by polarity and multiplier
        - then sums over filaments to return greens function for probes from a given coil
        """
        if probe == 'pickups':
            pos_R = self.pickup_pos[:,0]
            pos_Z = self.pickup_pos[:,1]

        pol = self.coil_dict[coil_key]['polarity'][np.newaxis,:]
        mul = self.coil_dict[coil_key]['multiplier'][np.newaxis,:]
        greens_br_filaments = GreensBr(self.coil_dict[coil_key]['coords'][0][np.newaxis,:],
                            self.coil_dict[coil_key]['coords'][1][np.newaxis,:],
                            pos_R[:,np.newaxis],
                            pos_Z[:,np.newaxis])
        greens_br_filaments *= pol
        greens_br_filaments *= mul
        greens_br_coil = np.sum(greens_br_filaments,axis=1)

        greens_bz_filaments = GreensBz(self.coil_dict[coil_key]['coords'][0][np.newaxis,:],
                            self.coil_dict[coil_key]['coords'][1][np.newaxis,:],
                            pos_R[:,np.newaxis],
                            pos_Z[:,np.newaxis])

        greens_bz_filaments *= pol
        greens_bz_filaments *= mul
        greens_bz_coil = np.sum(greens_bz_filaments,axis=1)

        return greens_br_coil, greens_bz_coil

    def greens_BrBz_all_coils(self, probe = 'pickups'):
        """
        Create 2d array of greens functions for all coils and at all probe positions
        - array[i][j] is greens function for coil i evaluated at probe position j
        """
        array_r = np.array([]).reshape(0,self.number_pickups)
        array_z = np.array([]).reshape(0,self.number_pickups)
        for key in self.coil_dict.keys():
            array_r = np.vstack((array_r,self.greens_BrBz_single_coil(key,probe)[0]))
            array_z = np.vstack((array_z,self.greens_BrBz_single_coil(key,probe)[1]))

        return array_r, array_z

    def greens_BrBz_plasma(self,eq,probe = 'pickups'):
        """ 
        Compute greens function at probes from the plasma currents .
        - plasma current source in grid from solve. grid points contained in eq object 
        """
        if probe == 'pickups':
            pos_R = self.pickup_pos[:,0]
            pos_Z = self.pickup_pos[:,2]

        rgrid = eq.R
        zgrid = eq.Z

        greens_br = GreensBr(rgrid[:,:,np.newaxis],
                        zgrid[:,:,np.newaxis],
                        pos_R[np.newaxis,np.newaxis,:],
                        pos_Z[np.newaxis, np.newaxis,:])

        greens_bz = GreensBz(rgrid[:,:,np.newaxis],
                        zgrid[:,:,np.newaxis],
                        pos_R[np.newaxis,np.newaxis,:],
                        pos_Z[np.newaxis, np.newaxis,:])

        return greens_br, greens_bz

    def Br(self,eq, probe = 'pickups'):
        """
        Method to compute total radial magnetic field from coil and plasma
        returns array with Br at each pickup coil probe
        """
        coil_currents = self.get_coil_currents(eq)[:,np.newaxis]
        plasma_current = self.get_plasma_current(eq)[:,:,np.newaxis]
        if probe =='pickups':
            br_coil = np.sum(self.greens_br_coils_pickup*coil_currents,axis = 0)
            br_plasma = np.sum(self.greens_br_plasma_pickup * plasma_current, axis = (0,1))
        return br_coil + br_plasma

    def Bz(self,eq,probe = 'pickups'):
        """
        Method to compute total z component of magnetic field from coil and plasma
        returns array with Bz at each pickup coil probe
        """
        coil_currents = self.get_coil_currents(eq)[:,np.newaxis]
        plasma_current = self.get_plasma_current(eq)[:,:,np.newaxis]
        if probe =='pickups':
            bz_coil = np.sum(self.greens_bz_coils_pickup * coil_currents,axis = 0)
            bz_plasma = np.sum(self.greens_bz_plasma_pickup * plasma_current, axis = (0,1))
        return bz_coil + bz_plasma

    def Btor(self,eq,probe = "pickups"):
        """
        Probes outside of plasma therfore Btor = fvac/R
        returns array of btor for each probe position
        """
        if probe == 'pickups':
            pos_R = self.pickup_pos[:,0]

        btor  = eq._profiles.fvac()/pos_R
        return btor

    def calculate_pickup_value(self,eq,probe = 'pickups'):
        """ 
        Method to compute and return B.n, for a given pickup coil
        """
        orientation = self.pickup_or
        Btotal = np.vstack((self.Br(eq),self.Btor(eq,probe),self.Bz(eq)))

        BdotN = np.sum(orientation*Btotal[np.newaxis,:], axis = 1)

        return BdotN
