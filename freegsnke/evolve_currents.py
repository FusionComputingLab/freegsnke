import numpy as np
from numpy.linalg import inv

from . import MASTU_coils
from .MASTU_coils import coils_dict
from .MASTU_coils import coil_self_ind

class evolve_currents:
    #simple Euler implicit time stepper for the linearized circuit equation
    #would need time-derivatives from emulators to better approximate non linear evolution
    #these are ignored or simplified in current setup
    
    def __init__(self):
        
        self.n_coils = len(MASTU_coils.coil_self_ind)
        #self.dt_step = dt_step
        
        #RESISTENCE
        #use actual values of coil resistances!!!!!
        #value right now is GUESSED so that time evolution is approximately such that
        #1% change in the currents is obtained in 1e-4s
        coils_resistence = (1e2)*np.ones(self.n_coils)
        R_matrix = np.zeros((self.n_coils+1, self.n_coils+1))
        R_matrix[:-1,:-1] = np.diag(coils_resistence)
        self.R_matrix = R_matrix
        
        #INDUCTANCE
        L0_matrix = np.zeros((self.n_coils+1, self.n_coils+1))
        L0_matrix[:-1,:-1] = coil_self_ind
        self.L0_matrix = L0_matrix
        
        
    def initialize_time_t(self, eq, results):
        #adjust quantities for time t
        #results = qfe.quants_out(eq, profiles)
        
        self.Ip_tot = results['tot_Ip_Rp'][0]

        #adjust R matrix for use, with quantities relevant to eq at time t:
        #add entry for plasma resistance, i.e. divide by plasma conductivity 
        #use actual values of resistivity!!!!!
        #value right now is GUESSED so that time evolution is approximately such that
        #1% change in the currents is obtained in 1e-4s
        self.R_matrix[-1,-1] = results['tot_Ip_Rp'][1]*(1e-5)
        #calculate dpci_dw from emulator, for now at random
        dpci_dw = 0#0.2*np.random.randn(self.n_coils)
        self.R_matrix[-1,:-1] = dpci_dw

        #adjust L matrix for use, with quantities relevant to eq at time t:
        L_matrix = np.zeros((self.n_coils+1,self.n_coils+1))
        #add entry for plasma self inductance from emulator, 
        #for now is flux/current
        L_matrix[-1,-1] = results['plasma_self_flux']/self.Ip_tot
        #add inductances of plasma on the coils from emulator,
        #for now is flux/current
        L_matrix[:-1,-1] = results['plasma_flux_on_coils']/self.Ip_tot
        #add inductance of coils on plasma
        L_matrix[-1,:-1] = results['plasma_coil_ind']
        #add ncoil x ncoil term to coil-self-inductances from emulator
        #for now at 0
        L_matrix[:-1,:-1] += 0#(10**-2)*np.random.randn(self.n_coils,self.n_coils)*coil_self_ind
        #add coil self inductances
        L_matrix += self.L0_matrix
        self.L_matrix = L_matrix
        
        #prepare currents
        eq_currents = eq.tokamak.getCurrents()
        currents_vec = np.zeros(self.n_coils+1)
        for i,labeli in enumerate(coils_dict.keys()):
            currents_vec[i] = eq_currents[labeli]
        currents_vec[-1] = self.Ip_tot
        currents_vec[0] = 1000
        self.currents_vec = currents_vec
        
        #adjust input voltages by adding dw term DW from emulators,
        #for now put at 0
        self.Udw_term = 0
        
        
    def stepper(self, U_active, dt_step):
        #U_active only has n_coils, plasma not included
        #at the moment no walls/conductive material other than active coils
        all_Us = np.zeros(self.n_coils+1)
        all_Us[:self.n_coils] = U_active
        all_Us += self.Udw_term
        
        #I_t+1 = (R+L/dt)^-1(U+L/dt I)
        Ldt = self.L_matrix/dt_step
        self.invM = inv(self.R_matrix+Ldt)
        self.LdtI = np.matmul(Ldt,self.currents_vec)
        all_Us += self.LdtI
        
        self.new_currents = np.matmul(self.invM, all_Us)
        
        
    def new_currents_out(self, eq, results, U_active, dt_step):
        
        self.initialize_time_t(eq, results)
        self.stepper(U_active, dt_step)
        
        return(self.new_currents)
        
        
        
        
        