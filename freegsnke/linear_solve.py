import numpy as np

# from .implicit_euler import implicit_euler_solver_d
from .implicit_euler import implicit_euler_solver
from . import machine_config


class linear_solver:
    """Interface between the linearised system and an ODE solver, calling a general implicit-Euler for a first-order ODE.
    It needs some matrix combinations as inputs, and a number to quantities related to the plasma current to avoid round-off issues.
    """
    def __init__(self, Lambdam1, Vm1Rm12, Mey, Myy,
                    #    dIydI, hatIy0,
                       plasma_norm_factor,
                       plasma_resistance_1d,
                       max_internal_timestep=.0001,
                       full_timestep=.0001,
                       ):

        """Instantiates the linear_solver object, with inputs computed mostly from circuit_equation_metals.py .
        Based on the input plasma properties and comupling matrices, it prepares:
            - an instance of the implicit Euler solver implicit_euler_solver()
            - internal time-stepper for the implicit-Euler
            - dummy vessel voltages (zeros) in terms of filaments and eigenmodes

        Parameters
        ----------
        Lambdam1: np.array 
            diagonal matrix, inverse of diagonal form of
            Lambdam1 = self.Vm1@normal_modes.rm1l@self.V, where rm1l= Rm12@machine_config.coil_self_ind@Rm12
            V is the identity on the active coils and diagonalises only the passive coils (R^{-1/2}L_{passive}R^{-1/2})
        Vm1Rm12: np.array
            matrix combination V^{-1}R^{-1/2}, where V is defined above
        Mey: np.array 
            matrix of inductances between the reduced plasma domain cells and all metal coils
            (active coils and passive-structure filaments, self.Vm1Rm12Mey below is the one between plasma and modes)
            calculated by plasma_grids.py
        Myy: np.array 
            inductance matrix of reduced plasma domain cells
            calculated by plasma_grids.py
        plasma_norm_factor: float
            an overall number to work with rescaled currents that are within a comparable range
        max_internal_timestep: float
            internal integration timestep of the implicit-Euler solver, to be used as substeps over the <<full_timestep>> interval
        full_timestep: float
            full timestep requested to the implicit-Euler solver

        """
        
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
        """Resets the integration timesteps, calling self.solver.set_timesteps

        Parameters
        ----------
        max_internal_timestep: float
            integration substep of the ciruit equation, calling an implicit-Euler solver
        full_timestep: float
            integration timestep of the circuit equation
        """
        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.solver.set_timesteps(full_timestep=full_timestep,
                                  max_internal_timestep=max_internal_timestep)
    

    
    def set_linearization_point(self, dIydI, hatIy0):
        """Initialises an implicit-Euler solver with the appropriate matrices for the linearised problem.

        Parameters
        ----------
        dIydI = np.array
            partial derivatives of plasma-cell currents on the reduced plasma domain with respect to all <<current>> parameters
            (active coil currents, vessel normal modes, total plasma current divided by plasma_norm_factor).
            These would typically come from having solved the forward Grad-Shafranov problem for different combinations of current parameters.
        hatIy0 = np.array
            plasma-cell currents normalised by the total current at the linearlisation-point (it sums to 1)
        """

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
        """Initialises the pseudo-inductance matrix of the problem M\dot(x)+ Rx=forcing from the linearisation Jacobian.

        Parameters
        ----------
        None given explicitly, they are all given by the object attributes.
        
        """

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
        """Returns a stepper object instance

        Parameters
        ----------
        It = np.array
            vector of all independent currents that are solved for by the linearides problem
            (active currents, vessel normal modes, total plasma current divided by normalisation factor)
        active_voltage_vec = np.array 
            voltages applied to the active coils
        other parameters are passed in as object attributes
        """
        self.empty_U[:self.n_active_coils] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Vm1Rm12, self.empty_U)
        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt
    

    def prepare_min_update_linearization(self, current_record, Iy_record, threshold_svd):
        """Computes quantities to update the linearisation matrices, using a record of recently computed Grad-Shafranov solutions.
        Not used at present.

        Parameters
        ----------
        current_record : np.array
            <<current>> parameter values over a time-horizon
        Iy_record : np.array 
            plasma cell currents (over the reduced domain) over a time-horizon
        threshold_svd : float
            discards singular values that are too small, to obtain a smoother pseudo-inverse
        other parameters are passed in as object attributes
        """
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
        """Returns an updated linearisation of the problem.

        Parameters
        ----------
        parameters are passed in as object attributes
        """
        self.predicted_Iy = np.matmul(self.dIydI, self.current_dv.T)
        Iy_dv_d = self.Iy_dv - self.predicted_Iy

        delta = Iy_dv_d@self.pseudo_inverse
        return delta


    def calculate_linear_growth_rate(self, ):
        """Looks into the eigenvecotrs of the "M" matrix to find the negative singular values, which correspond to the growth rates

        Parameters
        ----------
        parameters are passed in as object attributes
        """
        self.all_timescales = np.sort(np.linalg.eig(self.Mmatrix)[0])
        mask = (self.all_timescales < 0)
        self.growth_rates = self.all_timescales[mask]
        

