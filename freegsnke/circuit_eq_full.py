import numpy as np
import copy

from . import implicit_euler
from . import MASTU_coils


class full_circuit_eqs:
    # builds the full system of circuit eq
    # combining metal currents (in normal modes)
    # and plasma - through grid of individual grid points

    def __init__(self, evol_metal_curr, evol_plasma_curr):

        self.N_active = MASTU_coils.N_active

        len_plasma = len(evol_plasma_curr.Ryy)
        len_metal = evol_metal_curr.n_independent_vars
        len_full = len_plasma + len_metal
        self.len_full = len_full

        self.max_internal_timestep = evol_metal_curr.max_internal_timestep
        self.full_timestep = evol_metal_curr.full_timestep

        self.metal_true = np.ones(len_metal)>0

        self.Vm1Rm12 = np.matmul(evol_metal_curr.Vm1, np.diag(evol_metal_curr.Rm12))

        self.full_M_matrix = np.zeros((len_full, len_full))
        self.full_M_matrix[:len_metal, :len_metal] = evol_metal_curr.Lambdam1
        self.full_M_matrix[:len_metal, len_metal:] = np.matmul(self.Vm1Rm12, evol_metal_curr.Mey)
        self.full_M_matrix[len_metal:, :len_metal] = self.full_M_matrix[:len_metal, len_metal:].T
        self.full_M_matrix[len_metal:, len_metal:] = evol_plasma_curr.Myy
        self.full_M_matrix_diag = np.diag(self.full_M_matrix)
        self.masked_M_matrix = 1.0*self.full_M_matrix

        self.full_R_matrix = np.eye(len_full)
        self.full_R_matrix[len_metal:, len_metal:] = np.diag(evol_plasma_curr.Ryy)

        self.solver_full = implicit_euler.implicit_euler_solver(Mmatrix=self.full_M_matrix, 
                                                                Rmatrix=self.full_R_matrix,  
                                                                max_internal_timestep=evol_metal_curr.max_internal_timestep,
                                                                full_timestep=evol_metal_curr.full_timestep)
        self.solver_mask = copy.deepcopy(self.solver_full)

        self.empty_U = np.zeros(len_full)

        self.Vm1Rm12 = self.Vm1Rm12[:evol_metal_curr.n_active_coils, :evol_metal_curr.n_active_coils]
        
    
    def set_solver_on_mask(self, plasma_mask_1d):
        mask_to_zero = np.concatenate((self.metal_true, plasma_mask_1d))
        self.masked_M_matrix = 1.0*self.full_M_matrix
        self.masked_M_matrix[mask_to_zero, :] = 0
        self.masked_M_matrix[:, mask_to_zero] = 0
        np.fill_diagonal(self.masked_M_matrix, self.full_M_matrix_diag)
        self.solver_mask.Lmatrix = 1.0*self.masked_M_matrix
        self.solver_mask.Mmatrix = 1.0*self.masked_M_matrix
        self.solver_mask.calc_inverse_operator()
        

    def stepper(self, It_metal, Iy, active_voltage_vec):
        It = np.concatenate((It_metal, Iy))
        self.empty_U[:self.N_active] = np.dot(self.Vm1Rm12, active_voltage_vec)
        Itpdt = self.solver_full.full_stepper(It=It, forcing=self.empty_U[:len(It)])
        return Itpdt


    def stepper_on_mask(self, It_metal, Iy, active_voltage_vec):
        It = np.concatenate((It_metal, Iy))
        self.empty_U[:self.N_active] = np.dot(self.Vm1Rm12, active_voltage_vec)
        Itpdt = self.solver_mask.full_stepper(It=It, forcing=self.empty_U[:len(It)])
        return Itpdt

    
    def set_and_step(self, It_metal, Iy, active_voltage_vec):
        plasma_mask_1d = (Iy < 1e-6)
        self.set_solver_on_mask(plasma_mask_1d)
        Itpdt = self.stepper_on_mask(It_metal, Iy, active_voltage_vec)
        return Itpdt





    

