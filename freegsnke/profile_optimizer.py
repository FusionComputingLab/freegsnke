import numpy as np
from . import newtonkrylov
from . import plasma_grids



class optimize_profile:
   
    def __init__(self, 
                 eq,
                 profile,
                 ):
        
        self.plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(eq.R, eq.Z)

        self.profile = profile
        self.NK = newtonkrylov.NewtonKrylov(eq)

        dR = eq.R[1, 0] - eq.R[0, 0]
        dZ = eq.Z[0, 1] - eq.Z[0, 0]
        self.dRdZ = dR*dZ

        #basis in dJ space
        self.red_len = np.len(self.plasma_pts)
        self.G = np.zeros((self.red_len, 3))
        self.Q = np.zeros(3)


    def Iyplasmafromjtor(self, jtor):
        red_Iy = jtor[self.mask_inside_limiter]*self.dRdZ
        return red_Iy


    def solve_and_control(self, eq, profile, targetJ, dresidual,
                             rtol_NK, counter):
        self.NK.solve(eq=eq, profiles=profile, rel_convergence=rtol_NK)
        candidatedJ = self.Iyplasmafromjtor(profile.jtor) - targetJ
        relncandidatedJ = np.linalg.norm(candidatedJ)/dresidual
        counter += 1 
        control =  ((relncandidatedJ<.1) + (relncandidatedJ>10))*(control<2)
        return candidatedJ, control, relncandidatedJ


    def build_jacobian(self, eq, profile, targetJ, dresidual,
                             rtol_NK, 
                             dIp, dalpha_m, dalpha_n):

        alpha_m = 1.0*profile.alpha_m
        alpha_n = 1.0*profile.alpha_n
        Ip = 1.0*profile.Ip

        i=0
        control = 1
        counter = 0
        while control:
            profile.Ip = Ip + dIp
            candidatedJ, control, relncandidatedJ = self.solve_and_control(eq, profile, targetJ, dresidual,
                                                                           rtol_NK, counter)
            dIp = dIp*min(10, relncandidatedJ)
        self.G[:, i] = candidatedJ
        self.Q[i] = dIp

        i=1
        control = 1
        counter = 0
        while control:
            profile.alpha_m = alpha_m + dalpha_m
            candidatedJ, control, relncandidatedJ = self.solve_and_control(eq, profile, targetJ, dresidual,
                                                                           rtol_NK, counter)
            dalpha_m = dalpha_m*min(10, relncandidatedJ)
        self.G[:, i] = candidatedJ
        self.Q[i] = dalpha_m

        i=1
        control = 1
        counter = 0
        while control:
            profile.alpha_n = alpha_n + dalpha_n
            candidatedJ, control, relncandidatedJ = self.solve_and_control(eq, profile, targetJ, dresidual,
                                                                           rtol_NK, counter)
            dalpha_n = dalpha_n*min(10, relncandidatedJ)
        self.G[:, i] = candidatedJ
        self.Q[i] = dalpha_n


    def LSQP(self, Fresidual, G, Q, clip=1):
        #solve the least sq problem in coeffs: min||G*coeffs+Fresidual||^2
        self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)),
                                     G.T), -Fresidual)                            
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        self.explained_res = np.sum(G*self.coeffs[np.newaxis,:], axis=1) 
        #get the associated step in candidate_d_sol space
        self.d_sol_step = np.sum(Q*self.coeffs[np.newaxis,:], axis=1)











        