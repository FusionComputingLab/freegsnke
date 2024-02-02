import numpy as np
from . import newtonkrylov
from . import plasma_grids

import matplotlib.pyplot as plt


class optimize_profile:

    def __init__(
        self,
        eq,
        profile,
    ):

        self.plasma_pts, self.mask_inside_limiter = (
            plasma_grids.define_reduced_plasma_grid(eq.R, eq.Z)
        )

        self.profile = profile
        self.NK = newtonkrylov.NewtonKrylov(eq)

        self.nx = np.shape(eq.R)[0]
        self.ny = np.shape(eq.R)[1]
        dR = eq.R[1, 0] - eq.R[0, 0]
        dZ = eq.Z[0, 1] - eq.Z[0, 0]
        self.dRdZ = dR * dZ
        self.idxs_mask = np.mgrid[0 : self.nx, 0 : self.ny][
            np.tile(self.mask_inside_limiter, (2, 1, 1))
        ].reshape(2, -1)
        self.mapd = np.zeros_like(eq.R)

        # basis in dJ space
        self.red_len = len(self.plasma_pts)
        self.G = np.zeros((self.red_len, 2))
        self.Q = np.zeros((1, 2))

    def rebuild_grid_map(self, red_vec):
        self.mapd[self.idxs_mask[0], self.idxs_mask[1]] = red_vec
        return self.mapd

    def Iyplasmafromjtor(self, jtor):
        red_Iy = jtor[self.mask_inside_limiter] * self.dRdZ
        return red_Iy

    def restart_profile_vals(self, profile, Ip, alpha_m, alpha_n):
        profile.Ip = Ip
        profile.alpha_m = alpha_m
        profile.alpha_n = alpha_n

    def solve_and_control(
        self, eq, profile, red_Iy0, dresidual, rtol_NK, counter, max_iter
    ):
        self.NK.solve(eq=eq, profiles=profile, rel_convergence=rtol_NK)
        candidatedJ = self.Iyplasmafromjtor(profile.jtor) - red_Iy0
        relncandidatedJ = dresidual / np.linalg.norm(candidatedJ)
        counter += 1
        control = ((relncandidatedJ < 0.1) + (relncandidatedJ > 10)) * (
            counter < max_iter
        )
        plt.figure()
        plt.imshow(self.rebuild_grid_map(candidatedJ))
        plt.colorbar()
        plt.title(
            str(profile.Ip)
            + "   "
            + str(profile.alpha_m)
            + "   "
            + str(profile.alpha_n)
        )
        return candidatedJ, control, relncandidatedJ

    def build_jacobian(
        self,
        eq,
        profile,
        red_Iy0,
        dresidual,
        rtol_NK,
        dIp,
        dalpha_m,
        dalpha_n,
        max_iter=3,
    ):

        alpha_m = 1.0 * profile.alpha_m
        alpha_n = 1.0 * profile.alpha_n
        Ip = 1.0 * profile.Ip

        # i=0
        # control = 1
        # counter = 0
        # relncandidatedJ = 1.0
        # while control:
        #     dIp = dIp*min(10, relncandidatedJ)
        #     profile.Ip = Ip + dIp
        #     candidatedJ, control, relncandidatedJ = self.solve_and_control(eq, profile, red_Iy0, dresidual,
        #                                                        rtol_NK, counter, max_iter)

        # print(relncandidatedJ)
        # profile.Ip = 1.0*Ip
        # self.G[:, i] = candidatedJ
        # self.Q[:, i] = dIp

        i = 0
        control = 1
        counter = 0
        relncandidatedJ = 1.0
        while control:
            dalpha_m = dalpha_m * min(10, relncandidatedJ)
            profile.alpha_m = alpha_m + dalpha_m
            candidatedJ, control, relncandidatedJ = self.solve_and_control(
                eq, profile, red_Iy0, dresidual, rtol_NK, counter, max_iter
            )
        print(relncandidatedJ)
        profile.alpha_m = 1.0 * alpha_m
        self.G[:, i] = candidatedJ
        self.Q[:, i] = dalpha_m

        i = 1
        control = 1
        counter = 0
        relncandidatedJ = 1.0
        while control:
            dalpha_n = dalpha_n * min(10, relncandidatedJ)
            profile.alpha_n = alpha_n + dalpha_n
            candidatedJ, control, relncandidatedJ = self.solve_and_control(
                eq, profile, red_Iy0, dresidual, rtol_NK, counter, max_iter
            )
        print(relncandidatedJ)
        profile.alpha_n = 1.0 * alpha_n
        self.G[:, i] = candidatedJ
        self.Q[:, i] = dalpha_n

    def LSQP(self, Fresidual, clip=10):
        # solve the least sq problem in coeffs: min||G*coeffs+Fresidual||^2
        self.coeffs = np.matmul(
            np.matmul(np.linalg.inv(np.matmul(self.G.T, self.G)), self.G.T), -Fresidual
        )
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        self.explained_res = np.sum(self.G * self.coeffs[np.newaxis, :], axis=1)
        # get the associated step in candidate_d_sol space
        self.d_sol_step = np.sum(self.Q * self.coeffs[np.newaxis, :], axis=1)

    def update_profile(self, profile):
        # profile.Ip += self.coeffs[0]*self.Q[0,0]
        profile.alpha_m += self.coeffs[0] * self.Q[0, 0]
        profile.alpha_n += self.coeffs[1] * self.Q[0, 1]
        return profile
