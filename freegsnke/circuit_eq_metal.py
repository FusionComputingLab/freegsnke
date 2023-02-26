import numpy as np

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
                       full_timestep=.0001):
                    
        self.n_coils = len(MASTU_coils.coil_self_ind)
        self.N_active = MASTU_coils.N_active

        self.flag_vessel_eig = flag_vessel_eig
        self.flag_plasma = flag_plasma

        # dictionary to collect inv(Mm1R*dt+1) for various dt
        self.inverse_operator = {}

        if flag_vessel_eig:
            self.max_mode_frequency = max_mode_frequency
            self.initialize_for_eig()
        else:
            self.max_mode_frequency = 0
            self.initialize_for_no_eig()

        self.set_timesteps(full_timestep=full_timestep, max_internal_timestep=max_internal_timestep)

        # change if plasma by using set_plasma
        if flag_plasma:
            plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(reference_eq.R, reference_eq.Z)
            self.Mey = plasma_grids.Mey(plasma_pts)            

        #dummy voltage vector
        self.empty_U = np.zeros(self.independent_var)




    def initialize_for_eig(self, ):
        self.independent_var = np.sum(MASTU_coils.w < self.max_mode_frequency)
        
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
        # note Lambda, R, R12, Rm12 are vectors rather than matrices!

        self.internal_stepper = self.internal_stepper_eig
        if self.flag_plasma:
            self.forcing_term = self.forcing_term_eig_plasma
        else:
            self.forcing_term = self.forcing_term_eig_no_plasma




    def initialize_for_no_eig(self, ):
        self.independent_var = len(MASTU_coils.coil_self_ind)
        self.M = MASTU_coils.coil_self_ind
        self.Mm1 = MASTU_coils.Mm1
        self.R = np.diag(MASTU_coils.coil_resist)
        self.Rm1 = 1/MASTU_coils.coil_resist #it's a vector!
        self.Mm1R = self.Mm1@self.R
        self.Rm1M = np.diag(1/MASTU_coils.coil_resist)@self.M

        self.internal_stepper = self.internal_stepper_no_eig
        if self.flag_plasma:
            self.forcing_term = self.forcing_term_no_eig_plasma
        else:
            self.forcing_term = self.forcing_term_no_eig_no_plasma




    def set_mode(self, flag_plasma, flag_vessel_eig, reference_eq=0, max_mode_frequency=1):
        
        control = (flag_plasma != self.flag_plasma)
        self.flag_plasma = flag_plasma
        if control*flag_plasma: 
            plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(reference_eq.R, reference_eq.Z)
            self.Mey = plasma_grids.Mey(plasma_pts)        
        
        control += (flag_vessel_eig != self.flag_vessel_eig)
        self.flag_vessel_eig = flag_vessel_eig
        if flag_vessel_eig:
            control += (max_mode_frequency != self.max_mode_frequency)
            self.max_mode_frequency = max_mode_frequency
        if control*flag_vessel_eig:
            self.initialize_for_eig()
        else:
            self.initialize_for_no_eig()



    def calc_inverse_operator(self, dt):
        if dt not in self.inverse_operator.keys():
            self.inverse_operator[dt] = np.linalg.inv(np.eye(self.independent_var) + dt*self.Mm1R)
        self.set_inverse_operator = 1.0*self.inverse_operator[dt]


    def set_timesteps(self, full_timestep, max_internal_timestep):
        self.full_timestep = full_timestep
        self.max_internal_timestep = max_internal_timestep
        self.n_steps = int(self.full_timestep/self.max_internal_timestep + .999)
        self.internal_timestep = self.full_timestep/self.n_steps 
        if self.flag_vessel_eig<1:
            self.calc_inverse_operator(self.internal_timestep)


    def forcing_term_eig_plasma(self, active_voltage_vec, Iydot):
        all_Us = self.empty_U.copy()
        all_Us[:MASTU_coils.N_active] = active_voltage_vec
        all_Us -= self.Mey@Iydot
        all_Us = self.Vm1@(self.Rm12*all_Us)
        return all_Us
    def forcing_term_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        all_Us = self.empty_U.copy()
        all_Us[:MASTU_coils.N_active] = active_voltage_vec
        all_Us = self.Vm1@(self.Rm12*all_Us)
        return all_Us
    def forcing_term_no_eig_plasma(self, active_voltage_vec, Iydot):
        all_Us = self.empty_U.copy()
        all_Us[:MASTU_coils.N_active] = active_voltage_vec
        all_Us -= self.Mey@Iydot
        all_Us = self.Mm1@all_Us
        return all_Us
    def forcing_term_no_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        all_Us = self.empty_U.copy()
        all_Us[:MASTU_coils.N_active] = active_voltage_vec
        all_Us = self.Mm1@all_Us
        return all_Us


    def IvesseltoId(self, Ivessel):
        Id = self.Vm1@(self.R12*Ivessel)
        return Id
    def IdtoIvessel(self, Id):
        Ivessel = self.Rm12*(self.V@Id)
        return Ivessel

    
  
    def internal_stepper_eig(self, It, forcing):
        # advances current for equation
        # Lambda^-1 Idot + I = forcing
        # where Lambda is R12@M^-1@R12 = V@Lambda@V^-1
        # I are normal modes: Inormal = V^-1@R12@I
        # forcing here is V^-1@Rm12@(voltage-plasma_inductance), like calculated in forcing_term
        elambdadt = np.exp(-self.internal_timestep*self.Lambda)
        Itpdt = It*elambdadt + (1-elambdadt)*forcing
        return Itpdt
    def internal_stepper_no_eig(self, It, forcing):
        # advances current using equation
        # Itpdt = (Mm1@R*dt+1)^-1(forcing+It)
        # forcing is Mm1@(voltage-plasma_inductance),like calculated in forcing_term
        Itpdt = self.set_inverse_operator@(forcing*self.internal_timestep + It)
        return Itpdt


    def full_stepper(self, It, active_voltage_vec, Iydot=0):
        # input It and output I(t+dt) are vessel currents if no_eig or normal modes if eig

        forcing = self.forcing_term(active_voltage_vec, Iydot)
        
        for _ in range(self.n_steps):
            It = self.internal_stepper(It, forcing)
        
        return It


    def current_residual(self, It, Itpdt, active_voltage_vec, Iydot=0):
        # returns residual of circuit equations in normal modes:
        # residual = Lambda^-1 Idot + I - forcing
        forcing = self.forcing_term(active_voltage_vec, Iydot=0)
        residual = (Itpdt - It)/(self.Lambda*self.full_timestep)
        residual += Itpdt - forcing
        return residual





        