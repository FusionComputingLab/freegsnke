import numpy as np
import math

class implicit_euler_solver():
    """An implicit Euler time stepper for the linearized circuit equation. Solves an equation of type

    $$M\dot{I} + RI = F$$,

    with generic M, R and F. The internal_stepper and full_stepper solve for I(t+dt) using

    $$I(t+dt) = (M + Rdt)^{-1} (Fdt + MI(t))$$.

    It allows for a different M != L, where

    $$I(t+dt) = (M + Rdt)^{-1} (Fdt + LI(t))$$.
    """
    def __init__(self, Mmatrix, Rmatrix, full_timestep, max_internal_timestep):
        """Sets up the implicit euler solver

        Parameters
        ----------
        Mmatrix : np.ndarray
            (NxN) Mutual inductance matrix
        Rmatrix : np.ndarray
            (NxN) Diagonal resistance matrix
        full_timestep : float
            Full timestep (dt) for the stepper
        max_internal_timestep : float
            Maximum size of the intermediate timesteps taken during the stepper. If max_internal_timestep >= full_timestep, then the stepper only takes one step of size full_timestep
        """
        self.Mmatrix = Mmatrix
        self.Lmatrix = Mmatrix
        self.Rmatrix = Rmatrix
        self.dims = np.shape(Mmatrix)[0]
        self.set_timesteps(full_timestep, max_internal_timestep)
        self.empty_U = np.zeros(self.dims) # dummy voltage vector

    def set_Mmatrix(self, Mmatrix):
        """Updates the mutual inductance matrix. If the mutual inductance matrix and/or the resistance matrix is updated, the inverse operator needs to be recalculated using `self.calc_inverse_operator()`.

        Parameters
        ----------
        Mmatrix : np.ndarray
            (NxN) Mutual inductance matrix
        """
        self.Mmatrix = Mmatrix
    
    def set_Lmatrix(self, Lmatrix):
        """Set a separate mutual inductance matrix L != M.

        Parameters
        ----------
        Lmatrix : np.ndarray
            (NxN) Mutual inductance matrix
        """
        self.Lmatrix = Lmatrix

    def set_Rmatrix(self, Rmatrix):
        """Updates the resistance matrix. If the mutual inductance matrix and/or the resistance matrix is updated, the inverse operator needs to be recalculated using `self.calc_inverse_operator()`.
        
        Parameters
        ----------
        Rmatrix : np.ndarray
            (NxN) Diagonal resistance matrix
        """
        self.Rmatrix = Rmatrix
    
    def calc_inverse_operator(self):
        """Calculates the inverse operator (M + Rdt)^-1
        """
        self.inverse_operator = np.linalg.inv(self.Mmatrix + self.internal_timestep*self.Rmatrix)

    def set_timesteps(self, full_timestep, max_internal_timestep):
        """Sets the timesteps for the stepper and (re)calculate the inverse operator

        Parameters
        ----------
        full_timestep : float
            Full timestep (dt) for the stepper
        max_internal_timestep : float
            Maximum size of the intermediate timesteps taken during the stepper. If max_internal_timestep >= full_timestep, then the stepper only takes one step of size full_timestep
        """
        self.full_timestep = full_timestep
        self.max_internal_timestep = max_internal_timestep
        self.n_steps = math.ceil(full_timestep / max_internal_timestep)
        # self.intermediate_results = np.zeros((self.dims, self.n_steps))
        self.internal_timestep = self.full_timestep/self.n_steps 
        self.calc_inverse_operator()

    def internal_stepper(self, It, dtforcing):
        """Calculates the next internal timestep I(t + internal_timestep)
        
        Parameters
        ----------
        It : np.ndarray
            Length N vector of the currents, I, at time t
        dtforcing : np.ndarray
            Lenght N vector of the forcing, F,  at time t
            multiplied by self.internal_timestep
        """       
        Itpdt = np.dot(self.inverse_operator, dtforcing + np.dot(self.Lmatrix, It))
        return Itpdt

    def full_stepper(self, It, forcing):
        """Calculates the next full timestep I(t + `self.full_timestep`) by repeatedly solving for the internal timestep I(t + `self.internal_timestep`) for `self.n_steps` steps

        Parameters
        ----------
        It : np.ndarray
            Length N vector of the currents, I, at time t
        forcing : np.ndarray
            Lenght N vector of the forcing, F,  at time t
        """
        dtforcing = forcing*self.internal_timestep

        for _ in range(self.n_steps):
            It = self.internal_stepper(It, dtforcing)
            # self.intermediate_results[:, i] = It
        
        return It


class implicit_euler_solver_d():
    """An implicit Euler time stepper for the linearized circuit equation. Solves an equation of type

    $$M\dot{I} + RI = F$$,

    with generic M, R and F. The internal_stepper and full_stepper solve for $dI(t+dt) = I(t+dt) - I(t)$ using

    $$\delta I(t+dt) = dt (M + Rdt)^{-1} (F - RI(t))$$.

    It allows for the possibility of setting a different resistance matrix S in the inverse operator such that the inverse operator is (M + Sdt)^-1 instead of (M + Rdt)^-1.
    
    By default S = R. 
    """
    def __init__(self, Mmatrix, Rmatrix, full_timestep, max_internal_timestep):
        """Sets up the implicit euler solver

        Parameters
        ----------
        Mmatrix : np.ndarray
            (NxN) Mutual inductance matrix
        Rmatrix : np.ndarray
            (NxN) Diagonal resistance matrix
        full_timestep : float
            Full timestep (dt) for the stepper
        max_internal_timestep : float
            Maximum size of the intermediate timesteps taken during the stepper. If max_internal_timestep >= full_timestep, then the stepper only takes one step of size full_timestep
        """
        self.Mmatrix = Mmatrix
        # self.Mmatrixm1 = np.linalg.inv(Mmatrix)
        self.Rmatrix = Rmatrix
        self.Smatrix = Rmatrix
        self.dims = np.shape(Mmatrix)[0]
        self.set_timesteps(full_timestep, max_internal_timestep)
        self.empty_U = np.zeros(self.dims) # dummy voltage vector

    def set_Mmatrix(self, Mmatrix):
        """Updates the mutual inductance matrix M. If the mutual inductance matrix and/or the different resistance matrix, S, is updated, the inverse operator needs to be recalculated using `self.calc_inverse_operator()`.
        Parameters
        ----------
        Mmatrix : np.ndarray
            (NxN) Mutual inductance matrix
        """
        self.Mmatrix = Mmatrix

    def set_Rmatrix(self, Rmatrix):
        """Updates the resistance matrix R.

        Parameters
        ----------
        Rmatrix : np.ndarray
            (NxN) Diagonal resistance matrix
        """
        self.Rmatrix = Rmatrix
        
    def set_Smatrix(self, Sdiag):
        """Sets the different resistance matrix, S, to be used for the stepper. If the mutual inductance matrix and/or this resistance matrix is updated, the inverse operator needs to be recalculated using `self.calc_inverse_operator()`.

        Parameters
        ----------
        Sdiag : np.ndarray
            Lenght N vector of the resistances for each of the components.
        """
        self.Smatrix = np.diag(Sdiag)
    
    def calc_inverse_operator(self):
        """Calculates the inverse operator (M + Sdt)^{-1}.
        """
        self.inverse_operator = np.linalg.inv(self.Mmatrix + self.internal_timestep*self.Smatrix)

    def set_timesteps(self, full_timestep, max_internal_timestep):
        """sets the timesteps for the stepper and (re)calculate the inverse operator

        Parameters
        ----------
        full_timestep : float
            Full timestep (dt) for the stepper
        max_internal_timestep : float
            Maximum size of the intermediate timesteps taken during the stepper. If max_internal_timestep >= full_timestep, then the stepper only takes one step of size full_timestep
        """
        self.full_timestep = full_timestep
        self.max_internal_timestep = max_internal_timestep
        self.n_steps = math.ceil(full_timestep / max_internal_timestep)
        self.intermediate_results = np.zeros((self.dims, self.n_steps))
        self.internal_timestep = self.full_timestep/self.n_steps 
        self.calc_inverse_operator()

    def internal_stepper(self, It, forcing):
        """Calculates the next internal timestep $\delta I(t + `self.internal_timestep`) = dt  (M + Sdt)^{-1} . (F - RI(t))$
        
        Parameters
        ----------
        It : np.ndarray
            Length N vector of the currents, I, at time t
        forcing : np.ndarray
            Lenght N vector of the forcing, F,  at time t
        """
        dI = self.internal_timestep*np.dot(self.inverse_operator, forcing - np.dot(self.Rmatrix, It))
        return dI
    
    def full_stepper(self, It, forcing):
        """Calculates the next full timestep $\delta I(t + dt) = dt  (M + Sdt)^{-1} . (F - RI(t))$, where dt = `self.full_timestep`, by repeatedly solving for the internal timestep I(t + `self.internal_timestep`) for `self.n_steps` steps
        
        Parameters
        ----------
        It : np.ndarray
            Length N vector of the currents, I, at time t
        forcing : np.ndarray
            Lenght N vector of the forcing, F,  at time t
        """
        for i in range(self.n_steps):
            dI = 1.0*self.internal_stepper(It, forcing)
            self.intermediate_results[:, i] = 1.0*dI
            It = It + dI
        
        return np.sum(self.intermediate_results, axis=-1) # only return dI
