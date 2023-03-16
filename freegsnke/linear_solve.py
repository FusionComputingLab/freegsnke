import numpy as np

from .implicit_euler import implicit_euler_solver
from . import MASTU_coils

class simplified_solver:
    # implements simplified solvers to be used as starting point of non-linear solver
    
    
    def __init__(self, Lambdam1, Vm1Rm12, Mey, Myy,
                       plasma_norm_factor,
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

        self.n_active_coils = MASTU_coils.N_active

        # sets up implicit euler to solve system of 
        # - metal circuit eq
        # - plasma circuit eq
        # with the assumption that Iy(t+dt) = Ip(t+dt)*norm_red_Iy 
        # where norm_red_Iy is Iy(t)/Ip(t) and kept fixed
        # solver is initialized here but matrices are set up 
        # at each timestep using prepare_solver
        self.solver = implicit_euler_solver(Mmatrix=self.Mmatrix, 
                                            Rmatrix=np.eye(self.n_independent_vars+1), 
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.full_timestep)

        # dummy vessel voltage vector
        self.empty_U = np.zeros(np.shape(Vm1Rm12)[1])
        # dummy voltage vec for eig modes
        self.forcing = np.zeros(self.n_independent_vars+1)



    def prepare_solver(self, norm_red_Iy, norm_red_Iy_dot, active_voltage_vec, Rp):

        simplified_mutual_v = np.dot(self.Vm1Rm12Mey, norm_red_Iy_dot)
        simplified_mutual_h = np.dot(self.Vm1Rm12Mey, norm_red_Iy)
        self.Mmatrix[:-1, -1] = simplified_mutual_v*self.plasma_norm_factor
        self.Mmatrix[-1, :-1] = (simplified_mutual_h/Rp)/self.plasma_norm_factor

        simplified_plasma_self = np.sum(norm_red_Iy[:,np.newaxis]*norm_red_Iy_dot[np.newaxis,:]*self.Myy)
        self.Mmatrix[-1, -1] = simplified_plasma_self/Rp

        self.solver.set_Mmatrix(self.Mmatrix)

        self.empty_U[:self.n_active_coils] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Vm1Rm12, self.empty_U)

    

    def stepper(self, It, norm_red_Iy, norm_red_Iy_dot, active_voltage_vec, Rp):
        self.prepare_solver(norm_red_Iy, norm_red_Iy_dot, active_voltage_vec, Rp)
        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt
