import numpy as np
from numpy.linalg import inv

import freegs

from . import MASTU_coils
from .MASTU_coils import coils_dict
from .MASTU_coils import coil_self_ind

import copy

from . import quants_for_emu
from . import evolve_currents

#from picardfast import fast_solve
from .newtonkrylov import NewtonKrylov

class evolve_plasma_NK:
    #interfaces the circuit equation with freeGS and executes dt evolution
    #uses NK solver rather than picard
    
    def __init__(self, profiles, eq):
        
        #instantiate solver on eq's domain
        self.NK = NewtonKrylov(eq)

        
        #eq is the equilibrium object from which to start 
        #its plasma_psi is used to instantiate the eq on which to work on
        #eq1 is the equilibrium object that will be modified at each time step
        self.eq1 = freegs.Equilibrium(tokamak=eq.getMachine(),
                            Rmin=eq.R[0,0], Rmax=eq.R[-1,-1],    # Radial domain
                            Zmin=eq.Z[0,0], Zmax=eq.Z[-1,-1],    # Height range
                            nx=np.shape(eq.R)[0], ny=np.shape(eq.R)[1], # Number of grid points
                            psi = eq.plasma_psi)  
        for i,labeli in enumerate(coils_dict):
            self.eq1.tokamak[labeli].control = False
        
#         #self.tokamak is the one on which to switch control off and assign currents
#         self.tokamak = eq.getMachine()
#         for i,labeli in enumerate(coils_dict):
#             self.tokamak[labeli].control = False
        
        #profiles are the properties of the eq at handoff, 
        #paxis, fvac and alpha values are taken from there and kept fixed thereafter
        
        self.paxis = profiles.paxis
        self.fvac = profiles.fvac
        self.alpha_m = profiles.alpha_m
        self.alpha_n = profiles.alpha_n
        
        
        
        #qfe is an instance of the quants_for_emulation class
        #calculates fluxes of plasma on coils and on itself
        #as well as all other time-dependent quantities needed for time evolution
        self.qfe = quants_for_emu.quants_for_emulation(eq)
        
        #evol_currents is an instance of the evolve_currents class
        #contains Euler stepper
        self.evol_currents = evolve_currents.evolve_currents()
        
        #quantities for position of plasma
        #used to look if hitting the wall
        mask_inside_reactor = np.ones_like(eq.R)
        mask_inside_reactor *= (eq.R>eq.R[11,0])*(eq.R<eq.R[100,0])
        mask_inside_reactor *= (eq.Z<.96+1*eq.R)*(eq.Z>-.96-1*eq.R)
        mask_inside_reactor *= (eq.Z<-2+9.*eq.R)*(eq.Z>2-9.*eq.R)
        mask_inside_reactor *= (eq.Z<2.28-1.1*eq.R)*(eq.Z>-2.28+1.1*eq.R)
        mask_outside_reactor = 1-mask_inside_reactor
        self.mask_outside_reactor = mask_outside_reactor
        self.plasma_against_wall = 0
        
        # this just creates a vector with all currents
        # including coils and plasma
        n_coils = len(coils_dict.keys())
        currents_vec = np.zeros(n_coils+1)
        eq_currents = eq.tokamak.getCurrents()
        for i,labeli in enumerate(coils_dict.keys()):
            currents_vec[i] = eq_currents[labeli]
        currents_vec[-1] = eq.plasmaCurrent()
        #currents_vec[0] = 1000
        self.currents_vec = currents_vec
        self.new_currents = currents_vec
        
        self.npshape = np.shape(eq.plasma_psi)
        self.dt_step = 0
        
        
    def assign_currents(self, currents_vec=None):
        
        if currents_vec is not None:
            self.currents_vec = currents_vec
        
        self.profiles1 = freegs.jtor.ConstrainPaxisIp(self.paxis, # Plasma pressure on axis [Pascals]
                                            self.currents_vec[-1], # Plasma current [Amps]
                                            self.fvac, # vacuum f = R*Bt
                                            alpha_m = self.alpha_m,
                                            alpha_n = self.alpha_n)
        for i,labeli in enumerate(coils_dict):
            self.eq1.tokamak[labeli].current = self.currents_vec[i]
        
        
    def find_dt_evolve(self, U_active, max_rel_change):
        
        self.results = self.qfe.quants_out(self.eq1, self.profiles1)
        self.plasma_against_wall = np.sum(self.mask_outside_reactor*
                                          self.results['separatrix'][1])
        
        rel_change_curr = np.ones(5)
        dt_step = .001
        while np.sum(abs(rel_change_curr[1:])>max_rel_change)>0:
            dt_step /= 1.5
            new_currents = self.evol_currents.new_currents_out(self.eq1, 
                                                               self.results, 
                                                               U_active, 
                                                               dt_step)
            rel_change_curr = (new_currents-self.currents_vec)/(new_currents+self.currents_vec)
            
        self.new_currents = new_currents
        self.dt_step = dt_step
       
    
    def call_NK(self, rtol=1e-7, verbose=False):
        self.NK.solve(self.eq1, self.profiles1, rel_convergence=rtol, verbose=verbose)
    
    
    def do_step(self, U_active, 
                      currents_vec=None, 
                      max_rel_change=.005, 
                      rtol=1e-7,
                      verbose=False):
        """Evolve the equilibrium one time step.
        
        Returns
        -------
        flag : int
            1 if the plasma is impinging on the wall, 0 otherwise.
        """
        #flag to alert when hitting wall
        flag = 0
        
        if self.plasma_against_wall>0:
            print('plasma against the wall!')
            flag = 1
        else:
            self.assign_currents(currents_vec)
            self.find_dt_evolve(U_active, max_rel_change)
       
            self.call_NK(rtol, verbose)
            self.currents_vec = self.new_currents.copy()
        return(flag)

            
        
        
        
        
       
        
