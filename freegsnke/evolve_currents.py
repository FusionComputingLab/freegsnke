import numpy as np
from numpy.linalg import inv

from . import MASTU_coils
# from .MASTU_coils import coils_dict
# from .MASTU_coils import coil_self_ind
# from .MASTU_coils import coil_resist


class evolve_currents:
    #simple Euler implicit time stepper for the linearized circuit equation
    #would need time-derivatives from emulators to better approximate non linear evolution
    #these are ignored or simplified in current setup
    
    def __init__(self):
        
        self.n_coils = len(MASTU_coils.coil_self_ind)
        
        #RESISTENCE
        #use actual values of coil resistances!!!!!
        #value right now is approximated using metal resistivity and section area of coils
        R_matrix = np.zeros((self.n_coils+1, self.n_coils+1))
        R_matrix[:-1,:-1] = np.diag(MASTU_coils.coil_resist)
        self.R_matrix = R_matrix
        
        #INDUCTANCE
        L0_matrix = np.zeros((self.n_coils+1, self.n_coils+1))
        L0_matrix[:-1,:-1] = MASTU_coils.coil_self_ind
        self.L0_matrix = L0_matrix
        self.empty_L = np.zeros((self.n_coils+1,self.n_coils+1))
        
        #dummy voltage vector
        self.empty_U = np.zeros(self.n_coils+1)

        #adaptive steps
        self.n_step = 1
        self.dt_step = .0002
        
        
    def initialize_time_t(self, results):
        #adjust quantities in R and L matrix based on
        #results = qfe.quants_out(eq, profiles) for time t
        #gets currents from eq.tokamak and plasma current from qfe
        
        #self.Ip_tot = results['tot_Ip_Rp'][0]

        #adjust R matrix for use, with quantities relevant to eq at time t:
        #add entry for plasma resistance, i.e. divide by plasma conductivity 
        #use actual values of resistivity!!!!!
        #value right now is GUESSED so that time evolution is approximately such that
        #1% change in the currents is obtained in 1e-4s
        self.R_matrix[-1,-1] = results['tot_Ip_Rp'][1]*MASTU_coils.eta_plasma
        

        #adjust L matrix for use, with quantities relevant to eq at time t:
        L_matrix = self.empty_L.copy()
         #add inductance of coils on plasma
        L_matrix[-1,:-1] = results['plasma_coil_ind']
        #inductances of plasma on the coils 
        #includes plasma self inductance
        L_matrix[:,-1] = results['plasma_ind_on_coils']
        #add coil self inductances
        L_matrix += self.L0_matrix
        self.L_matrix = L_matrix
        
        # #prepare currents
        # eq_currents = eq.tokamak.getCurrents()
        # currents_vec = np.zeros(self.n_coils+1)
        # for i,labeli in enumerate(MASTU_coils.coils_dict.keys()):
        #     currents_vec[i] = eq_currents[labeli]
        # currents_vec[-1] = self.Ip_tot
        # self.currents_vec = currents_vec

        
        
    def stepper(self, U_active, dt_step, dR=0):
        #U_active only provides active voltages,
        #can be any length as long as len(U_active)<=self.n_coils
        #self.n_coils is the number of all active and passive coils now
        #even though at the moment no walls/conductive material other than active coils
        all_Us = self.empty_U.copy()
        all_Us[:MASTU_coils.N_active] = U_active
        #self.all_Us = all_Us.copy()
        
        #implicit Euler
        #I_t+1 = (R+L/dt)^-1(U+L/dt I)
        #dL is used to add dL/dt terms
        Ldt = (self.L_matrix)/dt_step
        self.invM = inv(self.R_matrix + dR + Ldt)
        self.LdtI = np.matmul(Ldt, self.currents_vec)
        all_Us += self.LdtI
        
        new_currents = np.matmul(self.invM, all_Us)
        return new_currents

    

    def determine_stepsize(self, tot_dt, max_dt_step=.0001):
        self.n_step = max(1,round(tot_dt/max_dt_step))
        self.dt_step = tot_dt/self.n_step
    
    def stepper_adapt_first(self, currents_now, U_active, dR):
        #divides dt_step in smaller steps based on relative change to the currents
        #U_active only provides active voltages,
        #can be any length as long as len(U_active)<=self.n_coils
        #self.n_coils is the number of all active and passive coils now
        #even though at the moment no walls/conductive material other than active coils
        self.all_Us = self.empty_U.copy()
        self.all_Us[:MASTU_coils.N_active] = U_active
        #self.all_Us = all_Us.copy()

        #implicit Euler
        #I_t+1 = (R+L/dt)^-1(U+L/dt I)
        #dL is used to add dL/dt terms
        Ldt = (self.L_matrix)/self.dt_step
        invM = inv(self.R_matrix + dR + Ldt)
        for i in range(self.n_step):
            LdtI = np.matmul(Ldt, currents_now)
            now_Us = self.all_Us + LdtI
            currents_now = np.matmul(invM, now_Us)
        
        new_currents = currents_now.copy()
        return new_currents


    def stepper_adapt_repeat(self, currents_now, dR):
        """same as above, but active voltage already in place"""
        
        Ldt = (self.L_matrix)/self.dt_step
        invM = inv(self.R_matrix + dR + Ldt)
        for i in range(self.n_step):
            LdtI = np.matmul(Ldt, currents_now)
            now_Us = self.all_Us + LdtI
            currents_now = np.matmul(invM, now_Us)
        
        new_currents = currents_now.copy()
        return new_currents
        
        
    def new_currents_out(self, eq, results, U_active, dt_step, dR=0):
        #sets R, L and currents
        self.initialize_time_t(eq, results)
        #does the linear solve
        new_currents = self.stepper(U_active, dt_step, dR)
        return new_currents
