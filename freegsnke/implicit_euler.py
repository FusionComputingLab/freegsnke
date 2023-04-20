import numpy as np

class implicit_euler_solver:
    # implicit Euler time stepper for the linearized circuit equation
    # solves an equation of the type
    # MIdot + RI = F
    # with generic M, R and F

    # internal_stepper and full_stepper solve for I(t+dt) using
    # I(t+dt) = (M + Rdt)^-1 . (Fdt + MI(t))

    # it allows for a different M != L , where
    # I(t+dt) = (M + Rdt)^-1 . (Fdt + LI(t))
    
    def __init__(self, Mmatrix, Rmatrix, full_timestep, max_internal_timestep):

        self.Mmatrix = Mmatrix
        self.Lmatrix = Mmatrix

        self.Rmatrix = Rmatrix

        self.dims = np.shape(Mmatrix)[0]

        self.set_timesteps(full_timestep, max_internal_timestep)
        
        #dummy voltage vector
        self.empty_U = np.zeros(self.dims)

    
    def set_Mmatrix(self, Mmatrix):
        self.Mmatrix = Mmatrix
        self.calc_inverse_operator()
    
    def set_Lmatrix(self, Lmatrix):
        self.Lmatrix = Lmatrix

    def set_Rmatrix(self, Rmatrix):
        self.Rmatrix = Rmatrix
        self.calc_inverse_operator()
    
    def calc_inverse_operator(self, ):
        self.inverse_operator = np.linalg.inv(self.Mmatrix + self.internal_timestep*self.Rmatrix)

    def set_timesteps(self, full_timestep, max_internal_timestep):
        self.full_timestep = full_timestep
        self.max_internal_timestep = max_internal_timestep
        self.n_steps = int(full_timestep/max_internal_timestep + .999)
        self.intermediate_results = np.zeros((self.dims, self.n_steps))
        self.internal_timestep = self.full_timestep/self.n_steps 
        self.calc_inverse_operator()



    def internal_stepper(self, It, forcing):
        # executes on self.internal_timestep
        # I(t+dt) = (M + Rdt)^-1 . (Fdt + LI(t))        
        Itpdt = np.dot(self.inverse_operator, forcing*self.internal_timestep + np.dot(self.Lmatrix, It))
        return Itpdt

    def full_stepper(self, It, forcing):
        # executes on self.full_timestep
        # by repeating over self.n_steps
        # I(t+dt) = (M + Rdt)^-1 . (Fdt + LI(t))     
        for i in range(self.n_steps):
            It = self.internal_stepper(It, forcing)
            self.intermediate_results[:, i] = It
        
        return It



class implicit_euler_solver_d:
    # implicit Euler time stepper for the linearized circuit equation
    # solves an equation of the type
    # MIdot + RI = F
    # with generic M, R and F

    # internal_stepper and full_stepper solve for deltaI = I(t+dt)-I(t) using
    # deltaI = dt . (M + Rdt)^-1 . (F - RI(t))

    # allows for the possibility of setting a different resistance matrix S in the inverse operator
    # inverse operator = (M + Sdt)^-1 instead of (M + Rdt)^-1
    # defauls is S=R

    
    def __init__(self, Mmatrix, Rmatrix, full_timestep, max_internal_timestep):

        self.Mmatrix = Mmatrix
        # self.Mmatrixm1 = np.linalg.inv(Mmatrix)

        self.Rmatrix = Rmatrix
        self.Smatrix = Rmatrix

        self.dims = np.shape(Mmatrix)[0]

        self.set_timesteps(full_timestep, max_internal_timestep)
        
        
            

        #dummy voltage vector
        self.empty_U = np.zeros(self.dims)


    def set_Mmatrix(self, Mmatrix):
        self.Mmatrix = Mmatrix

    def set_Rmatrix(self, Rmatrix):
        self.Rmatrix = Rmatrix
        
    def set_Smatrix(self, Sdiag):
        self.Smatrix = np.diag(Sdiag)
    
    def calc_inverse_operator(self, ):
        self.inverse_operator = np.linalg.inv(self.Mmatrix + self.internal_timestep*self.Smatrix)



    def set_timesteps(self, full_timestep, max_internal_timestep):
        self.full_timestep = full_timestep
        self.max_internal_timestep = max_internal_timestep
        self.n_steps = int(full_timestep/max_internal_timestep + .999)
        self.intermediate_results = np.zeros((self.dims, self.n_steps))
        self.internal_timestep = self.full_timestep/self.n_steps 
        self.calc_inverse_operator()


    def internal_stepper(self, It, forcing):
        # executes on self.internal_timestep
        # deltaI = dt . (M + Sdt)^-1 . (F - RI(t))
        dI = self.internal_timestep*np.dot(self.inverse_operator, forcing - np.dot(self.Rmatrix, It))
        return dI


    def full_stepper(self, It, forcing):
        # executes on self.full_timestep
        # by repeating over self.n_steps

        for i in range(self.n_steps):
            dI = 1.0*self.internal_stepper(It, forcing)
            self.intermediate_results[:, i] = 1.0*dI
            It = It + dI
        
        return It