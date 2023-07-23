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

        self.solver = implicit_euler_solver(Mmatrix=np.eye(self.n_independent_vars + 1), 
                                            Rmatrix=np.eye(self.n_independent_vars + 1), 
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.full_timestep)

        self.plasma_resistance_1d = plasma_resistance_1d

        # dummy vessel voltage vector
        self.empty_U = np.zeros(np.shape(Vm1Rm12)[1])
        # dummy voltage vec for eig modes
        self.forcing = np.zeros(self.n_independent_vars + 1)




    def reset_timesteps(self, max_internal_timestep,
                              full_timestep):
        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.solver.set_timesteps(full_timestep=full_timestep,
                                  max_internal_timestep=max_internal_timestep)
    

    
    def set_linearization_point(self, dIydI, hatIy0):

        self.dIydI = dIydI
        self.hatIy0 = hatIy0

        self.build_Mmatrix()

        self.solver = implicit_euler_solver(Mmatrix=self.Mmatrix, 
                                            Rmatrix=np.eye(self.n_independent_vars + 1), 
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.full_timestep)
        
        # self.solver.set_Mmatrix(self.Mmatrix)
        # self.solver.set_timesteps(full_timestep=self.full_timestep,
        #                           max_internal_timestep=self.max_internal_timestep)

        # self.growth_rates = np.sort(np.linalg.eig(self.Mmatrix)[0])

       


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
    

    def prepare_min_update_linearization(self, current_record, Iy_record, threshold_svd):
        self.Iy_dv = ((Iy_record - Iy_record[-1:])[:-1]).T

        self.current_dv = ((current_record - current_record[-1:])[:-1])
        self.abs_current_dv = np.mean(abs(self.current_dv), axis=0)

        U,S,B = np.linalg.svd(self.current_dv.T, full_matrices=False)
        
        mask = (S > threshold_svd)
        S = S[mask]
        U = U[:, mask]
        B = B[mask, :]

        # delta = Iy_dv@(B.T)@np.diag(1/S)@(U.T)
        self.pseudo_inverse = (B.T)@np.diag(1/S)@(U.T)


    def min_update_linearization(self, ):
        self.predicted_Iy = np.matmul(self.dIydI, self.current_dv.T)
        Iy_dv_d = self.Iy_dv - self.predicted_Iy

        delta = Iy_dv_d@self.pseudo_inverse
        return delta


    def calculate_linear_growth_rate(self, ):
        self.all_timescales = np.sort(np.linalg.eig(self.Mmatrix)[0])
        mask = (self.all_timescales < 0)
        self.growth_rates = self.all_timescales[mask]
        

