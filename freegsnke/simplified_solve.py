import numpy as np

from .implicit_euler import implicit_euler_solver_d
from .implicit_euler import implicit_euler_solver
from . import machine_config



class simplified_solver_J1:
    """Takes the full system of circuit equations (discretised in time and over the reduced plasma domain)
    and applies that  $$I_y(t+dt) = \hat{I_y}*I_p(t+dt)$$
    where $\hat{I_y}$ is assigned and such that np.sum(\hat{I_y})=1.
    With this hypothesis, the system can be solved to find all of the extensive currents at t+dt
    (metals and total plasma current).
    """
   
    
    def __init__(self, Lambdam1, Vm1Rm12, Mey, Myy,
                       plasma_norm_factor,
                       plasma_resistance_1d,
                       max_internal_timestep=.0001,
                       full_timestep=.0001):
        """Initialises the solver for the extensive currents that works with I_y(t+dt) = \hat{I_y}*I_p(t+dt) with given \hat{I_y}.
        \hat{I_y} is such that is must sum to 1 over the plasma integration domain.

        Based on the input plasma properties and coupling matrices, it prepares:
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
        plasma_resistance_1d: np.array 
            plasma reistance in each (reduced domain) plasma cell, R_yy, raveled to be of the same shape as I_y,
            the one-dimensional plasma resistance is obtained by integrating \hat{I_y}R_{yy}\hat{I_{y}}/I_{p}^2
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
        # NB the solver is initialized here but the matrices are set up 
        # at each timestep using prepare_solver
        self.solver = implicit_euler_solver(Mmatrix=self.Mmatrix, 
                                            Rmatrix=np.eye(self.n_independent_vars + 1), 
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.full_timestep)

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


    def prepare_solver(self, hatIy_left, hatIy_0, hatIy_1, active_voltage_vec):
        """Computes the actual matrices that are needed in the ODE for the extensive currents
         and that must be passed to the implicit-Euler solver.
         Due to the time-discretisation for the Euler solver,
         three versions of the gridded plasma current distribution are needed as input.

        Parameters
        ----------
        hatIy_left: np.array
            guess for gridded plasma current to left-contract the plasma evolution equation
            (e.g. at time t, or t+dt, or a combination)
        hatIy_0: np.array
            gridded plasma current to left-contract the plasma evolution equation at time t
        hatIy_1: np.array
            (guessed) gridded plasma current to left-contract the plasma evolution equation at time t+dt
        active_voltage_vec: np.array
            voltages applied to the active coils
        """

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
        """Computes and returns the set of extensive currents at time t+dt

        Parameters
        ----------
        hatIy_left: np.array
            guess for gridded plasma current to left-contract the plasma evolution equation
            (e.g. at time t, or t+dt, or a combination)
        hatIy_0: np.array
            gridded plasma current to left-contract the plasma evolution equation at time t
        hatIy_1: np.array
            (guessed) gridded plasma current to left-contract the plasma evolution equation at time t+dt
        active_voltage_vec: np.array
            voltages applied to the active coils
        
        Returns
        -------
        Itpdt: np.array
            currents (active coils, vessel eigenmodes, total plasma current) at time t+dt
        """
        self.prepare_solver(hatIy_left, hatIy_0, hatIy_1, active_voltage_vec)
        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt




class simplified_solver_dJ:
    """Implements a solver of the circuit equations, including the plasma,
    in which the change of gridded plasma currents is given.
    dJ is the direction of the vector dIy, <direction> means that sum(dJ) = 1
    """
    
    def __init__(self, Lambdam1, Vm1Rm12, Mey, Myy,
                       plasma_norm_factor,
                       plasma_resistance_1d,
                       max_internal_timestep=.0001,
                       full_timestep=.0001):
        """Initialises the solver for the extensive currents that works with 
        I_y(t+dt)-I_y(t) = dJ*(I_p(t+dt)-I_p(t)) = dJ*dIp with given dJ (such that sum(dJ)=1).

        Based on the input plasma properties and coupling matrices, it prepares:
            - an instance of the implicit Euler solver implicit_euler_solver_d() to solve in dIp and dI(metals)
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
        plasma_resistance_1d: np.array 
            plasma reistance in each (reduced domain) plasma cell, R_yy, raveled to be of the same shape as I_y,
            the one-dimensional plasma resistance is obtained by integrating \hat{I_y}R_{yy}\hat{I_{y}}/I_{p}^2
        max_internal_timestep: float
            internal integration timestep of the implicit-Euler solver, to be used as substeps over the <<full_timestep>> interval
        full_timestep: float
            full timestep requested to the implicit-Euler solver

        """

        
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
        # NB the solver is initialized here but the matrices are set up 
        # at each timestep using prepare_solver
        self.solver = implicit_euler_solver_d(Mmatrix=self.Mmatrix, 
                                              Rmatrix=np.eye(self.n_independent_vars+1), 
                                              max_internal_timestep=self.max_internal_timestep,
                                              full_timestep=self.full_timestep)

        # dummy vessel voltage vector
        self.empty_U = np.zeros(np.shape(Vm1Rm12)[1])
        # dummy voltage vec for eig modes
        self.forcing = np.zeros(self.n_independent_vars+1)
        
        # dummy Sdiag for the Euler solver
        self.Sdiag = np.ones(self.n_independent_vars+1)


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
                                  max_internal_timestep=full_timestep)


    def prepare_solver(self, norm_red_Iy0,
                            #  norm_red_Iy_m1, 
                             norm_red_Iy_dot, 
                             active_voltage_vec, 
                             Rp,
                             central_2):
        """Computes the actual matrices that are needed in the ODE for the extensive currents
         and that must be passed to the implicit-Euler solver.
         Due to the time-discretisation for the Euler solver,
         three versions of the gridded plasma current distribution are needed as input.

        Parameters
        ----------
        norm_red_Iy0: np.array
            normalised gridded plasma current, which sums to 1 over the plasma integration domain,
            used to left-contract the plasma evolution equation
        norm_red_Iy_dot: np.array
            (guessed) normalised time-change of the gridded plasma current dJ,
            which sums to 1 over the plasma integration domain
        active_voltage_vec: np.array
            voltages applied to the active coils
        Rp: float
            one-dimensional plasma resistance computed by contracting I_y^2 with plasma_resistance_1d
        central_2: 
            central time-derivative of plasma current
        """

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
        """Computes and returns the set of extensive currents at time t+dt

        Parameters
        ----------
        norm_red_Iy0 : np.array
            gridded plasma current distribution over the (reduced) plasma integration domain, sums to 1
        norm_red_Iy_dot : np.array
            (guessed) gridded change dJ of plasma current distribution over the (reduced) plasma integration domain, sums to 1
        
        active_voltage_vec: np.array
            voltages applied to the active coils
        
        Rp: float
            one-dimensional plasma resistance computed by contracting I_y^2 with plasma_resistance_1d
        central_2: 
            central time-derivative of plasma current


        Returns
        -------
        Itpdt: np.array
            extensive currents (active coils, vessel eigenmodes, total plasma current) at time t+dt
        """
        self.prepare_solver(norm_red_Iy0, norm_red_Iy_dot, active_voltage_vec, Rp, central_2)
        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt
