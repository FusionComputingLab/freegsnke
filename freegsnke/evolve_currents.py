import numpy as np
from numpy.linalg import inv

from . import MASTU_coils
from . import plasma_grids

class evolve_metal_currents:
    # implicit Euler time stepper for the linearized circuit equation
    # solves an equation of the type
    # MIdot + RI = F
    # with generic M, R and F, whose form depend on the flags
    # plasma inductance is included in the forcing term
    
    def __init__(self, flag_vessel_eig,
                       flag_plasma,
                       reference_eq=0,
                       max_mode_frequency=1,
                       max_internal_timestep=.0001,
                       output_timestep=.0001):
                    
        self.n_coils = len(MASTU_coils.coil_self_ind)
        self.N_active = MASTU_coils.N_active
        
        # allow for 10 timesteps in fastest mode
        self.max_mode_frequency = max_mode_frequency
        self.max_internal_timestep = min(1/(10*max_mode_frequency), max_internal_timestep)
        self.output_timestep = output_timestep
        
        # change solve mode by using set_eig
        self.flag_plasma = flag_plasma
        self.flag_vessel_eig = flag_vessel_eig
        if flag_vessel_eig:
            self.initialize_for_eig(max_mode_frequency=max_mode_frequency)
        else:
            self.initialize_for_no_eig()

        # change if plasma by using set_plasma
        if flag_plasma:
            plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(reference_eq.R, reference_eq.Z)
            self.Mey = plasma_grids.Mey(plasma_pts)            

        #dummy voltage vector
        self.empty_U = np.zeros(self.independent_var)


    def initialize_for_eig(self, max_mode_frequency):
        self.independent_var = np.sum(MASTU_coils.w<max_mode_frequency)
        # Id = Vm1 R**(1/2) I 
        # to change base to truncated modes
        # I = R**(-1/2) V Id 
        self.Vm1 = ((MASTU_coils.Vmatrix).T)[:self.independent_var]
        self.V = (MASTU_coils.Vmatrix)[:, :self.independent_var]

        # equation is Lambda**(-1)Iddot + I = F
        # where Lambda is such that R12@M-1@R12 = V Lambda V-1
        # w are frequences, eigenvalues of Lambda, 
        # so M are timescales
        self.Lambda = MASTU_coils.w[:self.independent_var]
        self.R = np.ones(self.independent_var)

        self.R12 = MASTU_coils.coil_resist**.5
        self.Rm12 = MASTU_coils.coil_resist**(-.5)
        # note M,R, R12, Rm12 are vectors rather than matrices!



    def initialize_for_no_eig(self, ):
        self.independent_var = len(MASTU_coils.coil_self_ind)
        self.M = MASTU_coils.coil_self_ind
        self.Mm1 = MASTU_coils.Mm1
        self.R = np.diag(MASTU_coils.coil_resist)
        self.Mm1R = self.Mm1@self.R

        # set internal timestep and number of steps, if change needed, use set_timesteps
        self.n_steps = int(self.output_timestep/self.max_internal_timestep + .999)
        self.internal_timestep = self.output_timestep/self.n_steps     

        # dictionary to collect inv(Mm1R*dt+1) for various dt
        self.inverse_operator = {}
        self.inverse_operator[self.internal_timestep] = np.linalg.inv(np.eye(self.independent_var)+self.internal_timestep*self.Mm1R)
        self.set_inverse_operator = 1.0*self.inverse_operator[self.internal_timestep]



    def set_plasma(self, flag_plasma):
        if flag_plasma != self.flag_plasma:
            self.flag_plasma = flag_plasma
            if flag_plasma:
                plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(reference_eq.R, reference_eq.Z)
                self.Mey = plasma_grids.Mey(plasma_pts)            


    def set_eig(self, flag_vessel_eig, max_mode_frequency=1):
        if (flag_vessel_eig != self.flag_vessel_eig) or (max_mode_frequency != self.max_mode_frequency):
            self.flag_vessel_eig = flag_vessel_eig
            self.max_mode_frequency = max_mode_frequency
            if flag_vessel_eig:
                self.initialize_for_eig(max_mode_frequency=max_mode_frequency)
            else:
                self.initialize_for_no_eig()


    def set_timesteps(self, output_timestep):
        self.output_timestep = output_timestep
        self.n_steps = int(self.output_timestep/self.max_internal_timestep + .999)
        self.internal_timestep = self.output_timestep/self.n_steps 
        if self.internal_timestep not in self.inverse_operator.keys():
            self.inverse_operator[self.internal_timestep] = np.linalg.inv(np.eye(self.independent_var)+self.internal_timestep*self.Mm1R)
        self.set_inverse_operator = 1.0*self.inverse_operator[self.internal_timestep]



    def forcing_term(self, active_voltage_vec,
                           flag_vessel_eig, 
                           flag_plasma, Iydot=0):
        
        all_Us = self.empty_U.copy()
        all_Us[:MASTU_coils.N_active] = active_voltage_vec

        if flag_plasma:
            all_Us -= self.Mey@Iydot
        
        if flag_vessel_eig:
            all_Us = self.Vm1@(self.Rm12*all_Us)
        else:
            all_Us = self.Mm1@all_Us
        
        return all_Us
    
  
    def analytic_internal_step(self, It, forcing):
        # advances current for equation
        # Lambda^-1 Idot + I = forcing
        # where Lambda is R12@M^-1@R12 = V@Lambda@V^-1
        # I are normal modes: Inormal = V^-1@R12@I
        # forcing here is V^-1@Rm12@(voltage-plasma_inductance), like calculated in forcing_term
        elambdadt = np.exp(-self.internal_timestep*self.Lambda)
        Itpdt = It*elambdadt + (1-elambdadt)*forcing
        return Itpdt


    def numeric_internal_step(self, It, forcing):
        # advances current using equation
        # Itpdt = (Mm1@R*dt+1)^-1(forcing+It)
        # forcing is Mm1@(voltage-plasma_inductance),like calculated in forcing_term
        Itpdt = self.set_inverse_operator@(forcing*self.internal_timestep + It)
        return Itpdt


    def output_stepper(self, It, active_voltage_vec, Iydot=0):
        # input It is full vessel current, not reduced normal mode vector

        forcing = self.forcing_term(active_voltage_vec,
                                    self.flag_vessel_eig, 
                                    self.flag_plasma, Iydot)
        
        if self.flag_vessel_eig:
            advance_f = self.analytic_internal_step
            It = self.Vm1@(self.R12*It)
        else:
            advance_f = self.numeric_internal_step

        for _ in range(self.n_steps):
            It = advance_f(It, forcing)
        
        if self.flag_vessel_eig:
            It = self.Rm12*(self.V@It)
        
        return It

    



        