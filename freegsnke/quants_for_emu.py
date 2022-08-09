from freegs.gradshafranov import Greens
import numpy as np

from . import MASTU_coils
from .MASTU_coils import coils_dict

class quants_for_emulation:
    #needs coil_dict from MASTU_coils.py
    
    def __init__(self, eq):
        #pre-builds matrices for inductance calculations on new equilibria
        #eq need not be the actual one, only has to have same grid properties 
        #as those on which calculations will be made by calling other methods
        
        dR_dZ = (eq.R[0,0]-eq.R[1,0])*(eq.Z[0,0]-eq.Z[0,1])
        two_pi_dR_dZ = 2*np.pi*dR_dZ
        
        #reduced grid for plasma-related inductances
        red_idx = np.round(np.array(np.shape(eq.R))*
                           np.array([[12,100],[30,99]])/129).astype(int)
        red_R = eq.R[red_idx[0,0]:red_idx[0,1],
                     red_idx[1,0]:red_idx[1,1]]
        red_Z = eq.Z[red_idx[0,0]:red_idx[0,1],
                     red_idx[1,0]:red_idx[1,1]]
        self.red_idx = red_idx
        
        #for plasma-plasma flux
        #greenmatrix = Greens(red_R[:,:,np.newaxis,np.newaxis], red_Z[:,:,np.newaxis,np.newaxis],
        #                     red_R[np.newaxis,np.newaxis,:,:], red_Z[np.newaxis,np.newaxis,:,:])
        #greenmatrix *= two_pi_dR_dZ*dR_dZ
        #self.greenmatrix = greenmatrix
                
        #for coil-plasma and plasma-coil fluxes 
        resgrid = np.shape(red_Z)
        coil_plasma_greens = np.zeros((len(coils_dict.keys()),resgrid[0],resgrid[1]))
        for i,labeli in enumerate(coils_dict.keys()):
            greenm = Greens(red_R[np.newaxis,:,:], red_Z[np.newaxis,:,:],
                            coils_dict[labeli]['coords'][0][:,np.newaxis,np.newaxis],
                            coils_dict[labeli]['coords'][1][:,np.newaxis,np.newaxis])
            greenm *= coils_dict[labeli]['polarity'][:,np.newaxis,np.newaxis]
            coil_plasma_greens[i] = two_pi_dR_dZ*np.sum(greenm, axis=0)
        self.coil_plasma_greens = coil_plasma_greens
        
        #for separatrix characterization
        self.Zvec = np.arange(len(red_Z[0]))[np.newaxis,:].astype('float64')
        self.Zvec -= np.mean(self.Zvec)
       
        self.dR_dZ = dR_dZ
        self.red_R = red_R

        self.void_matrix = np.zeros((len(coils_dict.keys())+1, len(coils_dict.keys())+1))


        
    def jtor_and_mask(self, eq, profiles):
        # eq is the actual one on which to calculate quantities
        # profiles is the associated ConstrainPaxisIp or ConstrainBetapIp obj
        # call this BEFORE calling method fluxes
        self.psi = eq.psi()

        jtor = profiles.Jtor(eq.R, eq.Z, self.psi)
        self.plasma_mask = jtor>0
        self.red_jtor = jtor[self.red_idx[0,0]:self.red_idx[0,1],
                             self.red_idx[1,0]:self.red_idx[1,1]]
        self.red_plasma_mask = self.red_jtor>0
        
        #separatrix characterization
        self.plasma_zloc = np.stack((np.mean(self.Zvec*self.red_plasma_mask, axis=1),
                                     np.sum(self.red_plasma_mask, axis=1)))
        
        self.tot_current = self.dR_dZ*np.sum(jtor)
        
        #to be multiplied by the resistivity eta_plasma
        plasma_resistance = np.sum(self.red_R*self.red_jtor)
        plasma_resistance *= 2*np.pi*self.dR_dZ/self.tot_current
        self.plasma_resistance = plasma_resistance
        
        
                
    def fluxes(self, eq):
        #needs jtor and plasma masks
        
        #plasma-plasma flux:
        #plasma_self_flux = self.red_jtor[np.newaxis,np.newaxis,:,:]*self.greenmatrix
        #plasma_self_flux *= self.red_plasma_mask[:,:,np.newaxis,np.newaxis]
        #self.plasma_self_flux = np.sum(plasma_self_flux)

        #greenf-free simple plasma-plasma self-flux
        self.simple_plasma_self_ind = np.sum(eq.plasma_psi*self.plasma_mask)
        self.simple_plasma_self_ind *= 2*np.pi*self.dR_dZ/self.tot_current

        #greenf-free simple tot flux on plasma
        #tot_flux_on_plasma = np.sum(self.psi*self.plasma_mask)
        #tot_flux_on_plasma *= 2*np.pi*self.dR_dZ

        #inductance of plasma current on coils, if using Total voltages
        coil_plasma_ind = self.red_jtor[np.newaxis,:,:]*self.coil_plasma_greens
        coil_plasma_ind = np.sum(coil_plasma_ind, axis=(1,2))
        self.coil_plasma_ind = coil_plasma_ind/self.tot_current
        #if using voltage per loop use below:
        #self.coil_plasma_ind = self.coil_plasma_ind/MASTU_coils.nloops_per_coil
        
        #inductance of coils on plasma                                               
        plasma_coil_ind = self.red_plasma_mask[np.newaxis,:,:]*self.coil_plasma_greens
        plasma_coil_ind = np.sum(plasma_coil_ind, axis=(1,2))
        self.plasma_coil_ind = plasma_coil_ind
        

        
    def quants_out(self, eq, profiles):
        #calls all that's needed on eq, in order, and returns results

        self.jtor_and_mask(eq, profiles)
        self.fluxes(eq)
        
        results = {}
        
        #total plasma current, plasma resistance term (to devide by conductivity)
        results['tot_Ip_Rp'] = [self.tot_current, self.plasma_resistance]
        
        #plasma self inductance:
        #results['plasma_self_ind'] = self.simple_plasma_self_ind
        
        #inductance of plasma current on coils
        #includes plasma self inductance!
        results['plasma_ind_on_coils'] = np.concatenate((self.coil_plasma_ind, [self.simple_plasma_self_ind]))
        
        #inductance of coils on plasma
        results['plasma_coil_ind'] = self.plasma_coil_ind

        #for non-linear evolution step
        #results['non_linear_fluxes'] = np.concatenate((self.coil_plasma_flux, [self.tot_flux_on_plasma]), axis=0)
        
        #plasma centroid and width in Z of section inside separatrix
        #in grid units
        results['separatrix'] = [self.plasma_zloc, self.plasma_mask]

        # Lplasma = self.void_matrix.copy()
        # Lplasma[:,-1] += results['plasma_ind_on_coils']
        # Lplasma[-1,:-1] += self.plasma_coil_ind
        # results['Lplasma'] = Lplasma
        
        return results

    



   

        
 

        
        
        
    