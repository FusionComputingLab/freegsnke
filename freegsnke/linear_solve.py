import numpy as np

from .implicit_euler import implicit_euler_solver

class simplified_solver:
    # implements simplified solvers to be used as starting point of non-linear solver
    
    
    def __init__(self, Lambdam1, Vm1Rm12, Mey, Myy,
                       plasma_norm_factor,
                       max_internal_timestep=.0001,
                       full_timestep=.0001,
                       plasma_resistance=1e-6):

        
        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.plasma_resistance = plasma_resistance
        self.plasma_norm_factor = plasma_norm_factor

        self.n_independent_vars = len(Lambdam1)
        self.Mmatrix = np.eye(self.n_independent_vars+1)
        self.Mmatrix[:-1,:-1] = Lambdam1

        self.Vm1Rm12 = Vm1Rm12
        self.Vm1Rm12Mey = Vm1Rm12@Mey
        self.Myy = Myy

        self.solver = implicit_euler_solver(Mmatrix=self.Mmatrix, 
                                            Rmatrix=np.eye(self.n_independent_vars+1), 
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.full_timestep)

        # dummy vessel voltage vector
        self.empty_U = np.zeros(np.shape(Vm1Rm12)[1])
        # dummy voltage vec
        self.forcing = np.zeros(self.n_independent_vars+1)



    def prepare_solver(self, J, active_voltage_vec):

        simplified_mutual = np.dot(self.Vm1Rm12Mey, J)
        self.Mmatrix[:-1, -1] = simplified_mutual*self.plasma_norm_factor
        self.Mmatrix[-1, :-1] = (simplified_mutual/self.plasma_resistance)/self.plasma_norm_factor

        simplified_plasma_self = np.sum(J[:,np.newaxis]*J[np.newaxis,:]*self.Myy)/self.plasma_resistance
        self.Mmatrix[-1, -1] = simplified_plasma_self

        self.solver.set_Mmatrix(self.Mmatrix)

        self.empty_U[:len(active_voltage_vec)] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Vm1Rm12, self.empty_U)

    

    def stepper(self, It, J, active_voltage_vec):
        self.prepare_solver(J, active_voltage_vec)
        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt
