import numpy as np

# from .implicit_euler import implicit_euler_solver_d
from .implicit_euler import implicit_euler_solver
from . import machine_config


class linear_solver:

    def __init__(self, Lambdam1, Vm1Rm12, Mey, Myy,
                    #    dIydI, hatIy0,
                       plasma_norm_factor,
                       plasma_resistance_1d,
                       max_internal_timestep=.0001,
                       full_timestep=.0001,
                       ):
        
        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.plasma_norm_factor = plasma_norm_factor

        self.n_independent_vars = np.shape(Lambdam1)[0]
        self.Mmatrix = np.eye(self.n_independent_vars + 1)
        
        self.Lambdam1 = Lambdam1
        self.Vm1Rm12 = Vm1Rm12
        self.Vm1Rm12Mey = np.matmul(Vm1Rm12, Mey)
        self.Myy = Myy
        

        self.n_active_coils = machine_config.n_active_coils

        self.plasma_resistance_1d = plasma_resistance_1d

        # dummy vessel voltage vector
        self.empty_U = np.zeros(np.shape(Vm1Rm12)[1])
        # dummy voltage vec for eig modes
        self.forcing = np.zeros(self.n_independent_vars + 1)

    
    
    def set_linearization_point(self, dIydI, hatIy0):

        self.dIydI = dIydI
        self.hatIy0 = hatIy0

        self.build_Mmatrix()

        self.solver = implicit_euler_solver(Mmatrix=self.Mmatrix, 
                                            Rmatrix=np.eye(self.n_independent_vars + 1), 
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.full_timestep)

       


    def build_Mmatrix(self, ):
        nRp = np.sum(self.plasma_resistance_1d * self.hatIy0 * self.hatIy0)*self.plasma_norm_factor

        self.Mmatrix[:self.n_independent_vars, :self.n_independent_vars] = np.copy(self.Lambdam1)
        self.Mmatrix[:self.n_independent_vars, :self.n_independent_vars] += np.matmul(self.Vm1Rm12Mey, self.dIydI[:,:-1])

        self.Mmatrix[:-1, -1] = np.dot(self.Vm1Rm12Mey, self.dIydI[:,-1])

        mat = np.matmul(self.Myy, self.dIydI[:,:-1]).T
        mat += self.Vm1Rm12Mey
        self.Mmatrix[-1, :-1] = np.dot(mat, self.hatIy0)

        self.Mmatrix[-1,-1] = np.dot(self.hatIy0, np.dot(self.Myy, self.dIydI[:,-1]))

        self.Mmatrix[-1, :] /= nRp



    def stepper(self, It, active_voltage_vec):
        self.empty_U[:self.n_active_coils] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Vm1Rm12, self.empty_U)
        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt
    

    def update_linearization(self, current_record, Iy_record, threshold_svd=.1):
        current_dv = ((current_record - current_record[-1:])[:-1])

        Iy_dv = ((Iy_record - Iy_record[-1:])[:-1]).T
        self.predicted_Iy = np.matmul(self.dIydI, current_dv.T)
        Iy_dv = Iy_dv - self.predicted_Iy

        svd = np.linalg.svd(current_dv.T, full_matrices=False)

        mask = svd[1] > threshold_svd
        
        self.Iy_dv = (Iy_dv@(svd[-1].T)@np.diag(1/svd[1]))

        self.current_dv = (svd[0].T)[np.newaxis]
        delta = self.Iy_dv[:,:, np.newaxis]*self.current_dv/np.sum(self.current_dv**2, axis=-1, keepdims=True)
        delta = delta[:,mask,:]
        delta = np.sum(delta, axis=1)
        
        return delta










