import numpy as np

from . import machine_config
from . import plasma_grids
from .implicit_euler import implicit_euler_solver



class metal_currents:
    # sets up framework for all metal currents
    # calculates residuals of full non linear circuit eq 
    # sets up to solve metal circuit eq for vacuum shots
    
    def __init__(self, flag_vessel_eig,
                       flag_plasma,
                       reference_eq=0,
                       max_mode_frequency=1,
                       max_internal_timestep=.0001,
                       full_timestep=.0001):
                    
        self.n_coils = len(machine_config.coil_self_ind)
        self.n_active_coils = machine_config.N_active

        self.flag_vessel_eig = flag_vessel_eig
        self.flag_plasma = flag_plasma

        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        
        if flag_vessel_eig:
            self.max_mode_frequency = max_mode_frequency
            self.initialize_for_eig()
        else:
            self.max_mode_frequency = 0
            self.initialize_for_no_eig()


        # change if plasma by using set_plasma
        if flag_plasma:
            plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(reference_eq.R, reference_eq.Z)
            self.Mey = plasma_grids.Mey(plasma_pts)            

        #dummy voltage vector
        self.empty_U = np.zeros(self.n_coils)
        



    def initialize_for_eig(self, ):
        # from passive alone
        self.n_independent_vars = np.sum(machine_config.w_passive < self.max_mode_frequency)
        # include active
        self.n_independent_vars += machine_config.N_active

        # Id = Vm1 R**(1/2) I 
        # to change base to truncated modes
        # I = R**(-1/2) V Id 
        self.Vm1 = ((machine_config.Vmatrix).T)[:self.n_independent_vars]
        self.V = (machine_config.Vmatrix)[:, :self.n_independent_vars]

        # equation is Lambda**(-1)Iddot + I = F
        # where Lambda is such that R12@M-1@R12 = V Lambda V-1
        # w are frequences, eigenvalues of Lambda, 
        self.Lambda = self.Vm1@machine_config.lm1r@self.V
        self.Lambdam1 = self.Vm1@machine_config.rm1l@self.V

        self.R = machine_config.coil_resist
        self.R12 = machine_config.coil_resist**.5
        self.Rm12 = machine_config.coil_resist**-.5
        # R, R12, Rm12 are vectors rather than matrices!

        self.solver = implicit_euler_solver(Mmatrix=self.Lambdam1, 
                                            Rmatrix=np.eye(self.n_independent_vars), 
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.full_timestep)

        if self.flag_plasma:
            self.forcing_term = self.forcing_term_eig_plasma
        else:
            self.forcing_term = self.forcing_term_eig_no_plasma




    def initialize_for_no_eig(self, ):
        self.n_independent_vars = self.n_coils
        self.M = machine_config.coil_self_ind
        self.Mm1 = machine_config.Mm1
        self.R = np.diag(machine_config.coil_resist)
        self.Rm1 = 1/machine_config.coil_resist #it's a vector!
        self.Mm1R = self.Mm1@self.R
        self.Rm1M = np.diag(1/machine_config.coil_resist)@self.M

        # equation is MIdot + RI = F
        self.solver = implicit_euler_solver(Mmatrix=self.M, 
                                            Rmatrix=self.R, 
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.full_timestep)

        if self.flag_plasma:
            self.forcing_term = self.forcing_term_no_eig_plasma
        else:
            self.forcing_term = self.forcing_term_no_eig_no_plasma




    def reset_mode(self, flag_vessel_eig,
                        flag_plasma,
                        reference_eq=0,
                        max_mode_frequency=1,
                        max_internal_timestep=.0001,
                        full_timestep=.0001):
        # allows reset of init inputs
        
        control = (self.max_internal_timestep != max_internal_timestep)
        self.max_internal_timestep = max_internal_timestep

        control += (self.full_timestep != full_timestep)
        self.full_timestep = full_timestep

        control += (flag_plasma != self.flag_plasma)
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





    # def calc_inverse_operator(self, dt):
    #     if dt not in self.inverse_operator.keys():
    #         self.inverse_operator[dt] = np.linalg.inv(np.eye(self.n_independent_vars) + dt*self.Mm1R)
    #     self.set_inverse_operator = 1.0*self.inverse_operator[dt]


    # def set_timesteps(self, full_timestep, max_internal_timestep):
    #     self.full_timestep = full_timestep
    #     self.max_internal_timestep = max_internal_timestep
    #     self.n_steps = int(self.full_timestep/self.max_internal_timestep + .999)
    #     self.internal_timestep = self.full_timestep/self.n_steps 
    #     if self.flag_vessel_eig<1:
    #         self.calc_inverse_operator(self.internal_timestep)


    def forcing_term_eig_plasma(self, active_voltage_vec, Iydot):
        all_Us = np.zeros_like(self.empty_U)
        all_Us[:self.n_active_coils] = active_voltage_vec
        all_Us -= self.Mey@Iydot
        all_Us = np.dot(self.Vm1, self.Rm12*all_Us)
        return all_Us
    def forcing_term_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        all_Us = self.empty_U.copy()
        all_Us[:self.n_active_coils] = active_voltage_vec
        all_Us = np.dot(self.Vm1, self.Rm12*all_Us)
        return all_Us
    def forcing_term_no_eig_plasma(self, active_voltage_vec, Iydot):
        all_Us = self.empty_U.copy()
        all_Us[:self.n_active_coils] = active_voltage_vec
        all_Us -= np.dot(self.Mey, Iydot)
        return all_Us
    def forcing_term_no_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        all_Us = self.empty_U.copy()
        all_Us[:self.n_active_coils] = active_voltage_vec
        return all_Us



    def IvesseltoId(self, Ivessel):
        Id = np.dot(self.Vm1, self.R12*Ivessel)
        return Id
    def IdtoIvessel(self, Id):
        Ivessel = self.Rm12*np.dot(self.V, Id)
        return Ivessel

    
  
    # def internal_stepper_eig(self, It, forcing):
    #     # advances current for equation
    #     # Lambda^-1 Idot + I = forcing
    #     # where Lambda is R12@M^-1@R12 = V@Lambda@V^-1
    #     # I are normal modes: Inormal = V^-1@R12@I
    #     # forcing here is V^-1@Rm12@(voltage-plasma_inductance), like calculated in forcing_term
    #     elambdadt = np.exp(-self.internal_timestep*self.Lambda)
    #     Itpdt = It*elambdadt + (1-elambdadt)*forcing
    #     return Itpdt
    # def internal_stepper_no_eig(self, It, forcing):
    #     # advances current using equation
    #     # Itpdt = (Mm1@R*dt+1)^-1(forcing+It)
    #     # forcing is Mm1@(voltage-plasma_inductance),like calculated in forcing_term
    #     Itpdt = self.set_inverse_operator@(forcing*self.internal_timestep + It)
    #     return Itpdt


    def stepper(self, It, active_voltage_vec, Iydot=0):
        # input It and output I(t+dt) are vessel currents if no_eig or normal modes if eig

        forcing = self.forcing_term(active_voltage_vec, Iydot)
        It = self.solver(It, forcing)
        return It


    # def Iedot(self, It, Itpdt):
    #     iedot = (Itpdt - It)/self.full_timestep
    #     return iedot


    def current_residual(self, Itpdt, Iddot, forcing_term):
        # returns residual of circuit equations in normal modes:
        # residual = Lambda^-1 Idot + I - forcing
        residual = np.dot(self.Lambdam1, Iddot)
        residual += Itpdt 
        residual -= forcing_term
        return residual





        