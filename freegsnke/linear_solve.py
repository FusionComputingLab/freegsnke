import numpy as np

from .implicit_euler import implicit_euler_solver_d
from .implicit_euler import implicit_euler_solver
from . import machine_config



class simplified_solver_J1:
    """
    Takes full system of (discretised) circuit equations and applies that 
    $$I_y(t+dt) = \hat{I_y}*I_p(t+dt)$$
    where $\hat{I_y}$ is assigned (and such that np.sum(\hat{I_y})=1).
    With this hypothesis, the system can be solved to find all of the 
    extensive currents at t+dt.
    """
   
    
    def __init__(self, Lambdam1, Vm1Rm12, Mey, Myy,
                       plasma_norm_factor,
                       plasma_resistance_1d,
                       max_internal_timestep=.0001,
                       full_timestep=.0001):

        
        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.plasma_norm_factor = plasma_norm_factor

        self.n_independent_vars = np.shape(Lambdam1)[0]
        self.Mmatrix = np.eye(self.n_independent_vars + 1)
        self.Mmatrix[:-1,:-1] = Lambdam1

        self.Lmatrix = np.copy(self.Mmatrix)

        self.Vm1Rm12 = Vm1Rm12
        self.Vm1Rm12Mey = np.matmul(Vm1Rm12, Mey)
        self.Myy = Myy

        self.n_active_coils = machine_config.n_active_coils

        self.plasma_resistance_1d = plasma_resistance_1d


        # sets up implicit euler to solve system of 
        # - metal circuit eq
        # - plasma circuit eq
        # solver is initialized here but matrices are set up 
        # at each timestep using prepare_solver
        self.solver = implicit_euler_solver(Mmatrix=self.Mmatrix, 
                                            Rmatrix=np.eye(self.n_independent_vars + 1), 
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.full_timestep)

        # dummy vessel voltage vector
        self.empty_U = np.zeros(np.shape(Vm1Rm12)[1])
        # dummy voltage vec for eig modes
        self.forcing = np.zeros(self.n_independent_vars + 1)
        



    def prepare_solver(self, hatIy_left, hatIy_0, hatIy_1, active_voltage_vec):

        Rp = np.sum(self.plasma_resistance_1d*hatIy_left*hatIy_1)

        simplified_mutual_left = np.dot(self.Vm1Rm12Mey, hatIy_left)
        simplified_mutual_1 = np.dot(self.Vm1Rm12Mey, hatIy_1)
        simplified_mutual_0 = np.dot(self.Vm1Rm12Mey, hatIy_0)

        simplified_self_left = np.dot(self.Myy, hatIy_left)
        simplified_self_1 = np.dot(simplified_self_left, hatIy_1)
        simplified_self_0 = np.dot(simplified_self_left, hatIy_0)

        self.Mmatrix[-1, :-1] = simplified_mutual_left/(Rp*self.plasma_norm_factor)
        self.Lmatrix[-1, :-1] = np.copy(self.Mmatrix[-1, :-1])

        self.Mmatrix[:-1, -1] = simplified_mutual_1*self.plasma_norm_factor
        self.Lmatrix[:-1, -1] = simplified_mutual_0*self.plasma_norm_factor

        self.Mmatrix[-1, -1] = simplified_self_1/Rp
        self.Lmatrix[-1, -1] = simplified_self_0/Rp

        self.solver.set_Lmatrix(self.Lmatrix)
        self.solver.set_Mmatrix(self.Mmatrix)
        self.solver.calc_inverse_operator() # recalculate the inverse operator1

        self.empty_U[:self.n_active_coils] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Vm1Rm12, self.empty_U)

    

    def stepper(self, It, hatIy_left, hatIy_0, hatIy_1, active_voltage_vec):
        self.prepare_solver(hatIy_left, hatIy_0, hatIy_1, active_voltage_vec)
        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt




class simplified_solver_dJ:
    # implements solver of circuit eq + plasma system 
    # in which the direction dJ has been fixed
    # dJ is the direction of the vector dIy, 
    # the plasma current density change over the timestep
    # direction means that sum(dJ) = 1
    
    
    def __init__(self, Lambdam1, Vm1Rm12, Mey, Myy,
                       plasma_norm_factor,
                       plasma_resistance_1d,
                       max_internal_timestep=.0001,
                       full_timestep=.0001):

        
        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        # self.plasma_resistivity = plasma_resistivity
        self.plasma_norm_factor = plasma_norm_factor

        self.n_independent_vars = len(Lambdam1)
        self.Mmatrix = np.eye(self.n_independent_vars+1)
        self.Mmatrix[:-1,:-1] = Lambdam1

        self.Vm1Rm12 = Vm1Rm12
        self.Vm1Rm12Mey = np.matmul(Vm1Rm12, Mey)
        self.Myy = Myy

        self.n_active_coils = machine_config.n_active_coils

        self.plasma_resistance_1d = plasma_resistance_1d

        # sets up implicit euler to solve system of 
        # - metal circuit eq
        # - plasma circuit eq
        # it uses that \deltaIy = dJ*deltaIp
        # where deltaJ is a sum 1 vector and deltaIp is the increment in the total plasma current
        # the simplification consists in using a specified dJ vector rather than the self-consistent one
        # solver is initialized here but matrices are set up 
        # at each timestep using prepare_solver
        self.solver = implicit_euler_solver_d(Mmatrix=self.Mmatrix, 
                                              Rmatrix=np.eye(self.n_independent_vars+1), 
                                              max_internal_timestep=self.max_internal_timestep,
                                              full_timestep=self.full_timestep)

        # dummy vessel voltage vector
        self.empty_U = np.zeros(np.shape(Vm1Rm12)[1])
        # dummy voltage vec for eig modes
        self.forcing = np.zeros(self.n_independent_vars+1)
        
        # dummy Sdiag for the ueler solver
        self.Sdiag = np.ones(self.n_independent_vars+1)



    def prepare_solver(self, norm_red_Iy0,
                            #  norm_red_Iy_m1, 
                             norm_red_Iy_dot, 
                             active_voltage_vec, 
                             Rp,
                             central_2):

        Sp = np.sum(self.plasma_resistance_1d*norm_red_Iy0*norm_red_Iy_dot)/Rp

        simplified_mutual_v = np.dot(self.Vm1Rm12Mey, norm_red_Iy_dot)
        self.Mmatrix[:-1, -1] = simplified_mutual_v*self.plasma_norm_factor

        simplified_mutual_h = np.dot(self.Vm1Rm12Mey, norm_red_Iy0)
        self.Mmatrix[-1, :-1] = simplified_mutual_h/(Rp*self.plasma_norm_factor)

        simplified_plasma_self = np.sum(norm_red_Iy0[:,np.newaxis]*norm_red_Iy_dot[np.newaxis,:]*self.Myy)
        self.Mmatrix[-1, -1] = simplified_plasma_self/Rp

        self.solver.set_Mmatrix(self.Mmatrix)

        self.Sdiag[-1] = Sp
        self.solver.set_Smatrix(central_2*self.Sdiag)

        self.solver.calc_inverse_operator()

        self.empty_U[:self.n_active_coils] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Vm1Rm12, self.empty_U)


    def stepper(self, It, norm_red_Iy0, norm_red_Iy_dot, active_voltage_vec, Rp, central_2):
        self.prepare_solver(norm_red_Iy0, norm_red_Iy_dot, active_voltage_vec, Rp, central_2)
        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt
