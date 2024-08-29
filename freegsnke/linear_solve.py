import numpy as np
from scipy.linalg import solve_sylvester

from . import machine_config
from .implicit_euler import implicit_euler_solver


class linear_solver:
    """Interface between the linearised system of circuit equations and an ODE solver, calling a general implicit-Euler for a first-order ODE.
    Solves the linearised problem. It needs the Jacobian of the plasma current distribution with respect to the independent currents, dIy/dI.
    """

    def __init__(
        self,
        Lambdam1,
        Pm1,
        Rm1,
        Mey,
        Myy,
        plasma_norm_factor,
        plasma_resistance_1d,
        max_internal_timestep=0.0001,
        full_timestep=0.0001,
    ):
        """Instantiates the linear_solver object, with inputs computed mostly from circuit_equation_metals.py .
        Based on the input plasma properties and coupling matrices, it prepares:
        - an instance of the implicit Euler solver implicit_euler_solver()
        - internal time-stepper for the implicit-Euler

        Parameters
        ----------
        Lambdam1: np.array
            State matrix of the circuit equations for the metal in normal mode form:
            Pm1Rm1MP = Lambdam1
            P is the identity on the active coils and diagonalises the isolated dynamics of the passive coils, R^{-1/2}L_{passive}R^{-1/2}
        Pm1: np.array
            change of basis matrix, as defined above
        Rm1: np.array
            matrix of all metal resitances to the power of -1. Diagonal.
        Mey: np.array
            matrix of inductances between grid points in the reduced plasma domain and all metal coils
            (active coils and passive-structure filaments, self.Vm1Rm12Mey below is the one between plasma and the normal modes)
            calculated using plasma_grids.py
        Myy: np.array
            inductance matrix of grid points in the reduced plasma domain
            calculated using plasma_grids.py
        plasma_norm_factor: float
            an overall number to work with a rescaled plasma current, so that it's within a comparable range
        max_internal_timestep: float
            internal integration timestep of the implicit-Euler solver, to be used as substeps over the <<full_timestep>> interval
        full_timestep: float
            full timestep requested to the implicit-Euler solver
        """

        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep
        self.plasma_norm_factor = plasma_norm_factor

        if Lambdam1 is None:
            self.Lambdam1 = Pm1 @ (Rm1 @ (machine_config.coil_self_ind @ (Pm1.T)))
        else:
            self.Lambdam1 = Lambdam1
        self.n_independent_vars = np.shape(self.Lambdam1)[0]

        self.Mmatrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )
        self.M0matrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )
        self.dMmatrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )

        self.Pm1 = Pm1
        self.Rm1 = Rm1
        self.Pm1Rm1 = Pm1 @ Rm1
        self.Pm1Rm1Mey = np.matmul(self.Pm1Rm1, Mey)
        self.MyeP_T = Pm1 @ Mey
        self.Myy = Myy

        self.n_active_coils = machine_config.n_active_coils

        self.solver = implicit_euler_solver(
            Mmatrix=np.eye(self.n_independent_vars + 1),
            Rmatrix=np.eye(self.n_independent_vars + 1),
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.full_timestep,
        )

        self.plasma_resistance_1d = plasma_resistance_1d

        # dummy vessel voltage vector
        self.empty_U = np.zeros(np.shape(self.Pm1Rm1)[1])
        # dummy voltage vec for eig modes
        self.forcing = np.zeros(self.n_independent_vars + 1)
        self.profile_forcing = np.zeros(self.n_independent_vars + 1)

        self.dIydpars = None

    def reset_timesteps(self, max_internal_timestep, full_timestep):
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
        self.solver.set_timesteps(
            full_timestep=full_timestep, max_internal_timestep=max_internal_timestep
        )

    def set_linearization_point(self, dIydI, dIydpars, hatIy0):
        """Initialises an implicit-Euler solver with the appropriate matrices for the linearised problem.

        Parameters
        ----------
        dIydI = np.array
            partial derivatives of plasma-cell currents on the reduced plasma domain with respect to all intependent <<current>> parameters
            (active coil currents, vessel normal modes, total plasma current divided by plasma_norm_factor).
            These would typically come from having solved the forward Grad-Shafranov problem. Finite difference Jacobian.
        dIydI = np.array
            partial derivatives of plasma-cell currents on the reduced plasma domain with respect to all intependent profile parameters,
            i.e. (alpha_m, alpha_n, paxis OR betap)
        hatIy0 = np.array
            Normalised plasma current distribution on the reduced plasma domain (1d) at the equilibrium of the linearization.
            This vector sums to 1.
        """
        if dIydI is not None:
            self.dIydI = dIydI
        if dIydpars is not None:
            self.dIydpars = dIydpars
        if hatIy0 is not None:
            self.hatIy0 = hatIy0

        self.build_Mmatrix()

        self.solver = implicit_euler_solver(
            Mmatrix=self.Mmatrix,
            Rmatrix=np.eye(self.n_independent_vars + 1),
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.full_timestep,
        )

    def build_Mmatrix(
        self,
    ):
        """Initialises the pseudo-inductance matrix of the problem
        M\dot(x) + Rx = forcing from the linearisation Jacobian.

                          \Lambda^-1 + P^-1R^-1Mey A        P^-1R^-1Mey B
        M = M0 + dM =  (                                                       )
                           J(Myy A + MyeP)/Rp                J Myy B/Rp

        This also builds the forcing:
                    P^-1R^-1 Voltage         P^-1R^-1Mey
        forcing = (                   ) - (                 ) C \dot{theta}
                            0                  J Myy/Rp

        where A = dIy/dId
              B = dIy/dIp
              C = dIy/plasmapars


        Parameters
        ----------
        None given explicitly, they are all given by the object attributes.

        """

        self.M0matrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )
        self.dMmatrix = np.zeros(
            (self.n_independent_vars + 1, self.n_independent_vars + 1)
        )

        nRp = (
            np.sum(self.plasma_resistance_1d * self.hatIy0 * self.hatIy0)
            * self.plasma_norm_factor
        )

        # metal-metal before plasma
        self.M0matrix[: self.n_independent_vars, : self.n_independent_vars] = np.copy(
            self.Lambdam1
        )
        # metal-metal plasma-mediated
        self.dMmatrix[: self.n_independent_vars, : self.n_independent_vars] = np.matmul(
            self.Pm1Rm1Mey, self.dIydI[:, :-1]
        )

        # plasma to metal
        self.dMmatrix[:-1, -1] = np.dot(self.Pm1Rm1Mey, self.dIydI[:, -1])

        # metal to plasma
        self.M0matrix[-1, :-1] = np.dot(self.MyeP_T, self.hatIy0)
        # metal to plasma plasma-mediated
        self.dMmatrix[-1, :-1] = np.dot(
            np.matmul(self.Myy, self.dIydI[:, :-1]).T, self.hatIy0
        )

        JMyy = np.dot(self.Myy, self.hatIy0)
        self.dMmatrix[-1, -1] = np.dot(self.dIydI[:, -1], JMyy)

        self.dMmatrix[-1, :] /= nRp
        self.M0matrix[-1, :] /= nRp

        self.Mmatrix = self.M0matrix + self.dMmatrix

        # build necessary terms to incorporate forcing term from variations of the profile parameters
        # MIdot + RI = V - self.Vm1Rm12Mey_plus@self.dIydpars@d_profile_pars_dt
        if self.dIydpars is not None:
            Pm1Rm1Mey_plus = np.concatenate(
                (self.Pm1Rm1Mey, JMyy[np.newaxis] / nRp), axis=0
            )
            self.forcing_pars_matrix = np.matmul(Pm1Rm1Mey_plus, self.dIydpars)

    def stepper(self, It, active_voltage_vec, d_profile_pars_dt=None):
        """Executes the time advancement. Uses the implicit_euler instance.

        Parameters
        ----------
        It = np.array
            vector of all independent currents that are solved for by the linearides problem, in terms of normal modes:
            (active currents, vessel normal modes, total plasma current divided by normalisation factor)
        active_voltage_vec = np.array
            voltages applied to the active coils
        d_profile_pars_dt = np.array
            time derivative of the profile parameters, in the order (alpha_m, alpha_n, paxis OR betap)
        other parameters are passed in as object attributes
        """
        self.empty_U[: self.n_active_coils] = active_voltage_vec
        self.forcing[:-1] = np.dot(self.Pm1Rm1, self.empty_U)
        self.forcing[-1] = 0.0

        # add forcing term from time derivative of profile parameters
        if d_profile_pars_dt is not None:
            self.forcing -= np.dot(self.forcing_pars_matrix, d_profile_pars_dt)

        Itpdt = self.solver.full_stepper(It, self.forcing)
        return Itpdt

    def calculate_linear_growth_rate(
        self,
    ):
        """Looks into the eigenvecotrs of the "M" matrix to find the negative singular values, which correspond to the growth rates of instabilities.

        Parameters
        ----------
        parameters are passed in as object attributes
        """
        self.all_timescales = np.sort(np.linalg.eigvals(self.Mmatrix))
        self.all_timescales_const_Ip = np.sort(
            np.linalg.eigvals(self.Mmatrix[:-1, :-1])
        )
        mask = self.all_timescales < 0
        self.instability_timescale = self.all_timescales[mask]
        self.growth_rates = 1 / self.instability_timescale
        mask = self.all_timescales_const_Ip < 0
        self.instability_timescale_const_Ip = self.all_timescales_const_Ip[mask]
        self.growth_rates_const_Ip = 1 / self.instability_timescale_const_Ip

    def build_dIydall(self, mask=None):
        """Builds full Jacobian including both extensive currents and profile pars"""
        self.dIydall_full = np.concatenate((self.dIydI, self.dIydpars), axis=-1)
        if mask is not None:
            self.dIydall = self.dIydall_full[mask, :]
        else:
            self.dIydall = self.dIydall_full.copy()

    def assign_from_dIydall(self, mask=None):
        """Uses full Jacobian to assign current and profile components"""
        self.dIydI = self.dIydall_full[:, : self.n_independent_vars + 1]
        self.dIydpars = self.dIydall_full[:, self.n_independent_vars + 1 :]

    def build_n2_diffs(self, vectors):
        """Builds non trivial pairwise differences.

        Parameters
        ----------
        vectors : list of arrays
            These should be the recorded currents and Iys

        Returns
        -------
        list of arrays
            Each array has non-trivial pairwise differences of entries in the input arrays
        """
        diff_1d = []
        size = np.shape(vectors[0])[0]
        idxs = np.tril_indices(size, k=-1)
        idxs = idxs[0] * size + idxs[1]
        for vector in vectors:
            diff_vec_2d = vector[np.newaxis, :, :] - vector[:, np.newaxis, :]
            diff_1d.append((diff_vec_2d.reshape(size * size, -1)[idxs]).T)
        return diff_1d

    def prepare_linearization_update(self, current_record, Iy_record, threshold):
        """Computes quantities to update the linearisation matrices,
        using a record of recently computed Grad-Shafranov solutions.

        Parameters
        ----------
        current_record : np.array
            <<current>> and profile parameter values over a time-horizon
        Iy_record : np.array
            plasma cell currents (over the reduced domain) over a time-horizon
        """

        # self.mask_Iy = (np.sum(Iy_record>0, axis=0)>0)
        self.mask_Iy = (
            np.std(Iy_record, axis=0) / (np.mean(Iy_record, axis=0) + 0.1)
        ) > threshold
        Iy_record = Iy_record[:, self.mask_Iy]
        self.dv, self.dIy = self.build_n2_diffs([current_record, Iy_record])
        self.build_dIydall(mask=self.mask_Iy)
        self.dd = self.dIy - np.matmul(self.dIydall, self.dv)

        # Composing the Sylverster equation
        # where D is the sought Jacobian update
        # dd@dv.T + D@(dv@dv.T) + \lambda R@D == 0
        # standard form is
        # AX + XB = Q

        self.B = np.matmul(self.dv, self.dv.T)
        self.Q = np.matmul(self.dd, self.dv.T)

    def find_linearization_update(self, current_record, Iy_record, R, threshold):
        """Computes the regularised update to the full jacobian.

        Parameters
        ----------
        current_record : np.array
            <<current>> and profile parameter values over a time-horizon
        Iy_record : np.array
            plasma cell currents (over the reduced domain) over a time-horizon
        R : np.array
            the regularization to be applied.
        """
        self.prepare_linearization_update(current_record, Iy_record, threshold)
        reg_matrix = R[self.mask_Iy, :][:, self.mask_Iy]
        self.jacobian_update = solve_sylvester(a=reg_matrix, b=self.B, q=self.Q)

        self.jacobian_update_full = np.zeros_like(self.dIydall_full)
        self.jacobian_update_full[self.mask_Iy, :] = self.jacobian_update

    def apply_linearization_update(
        self,
    ):
        """Uses the precalculated self.jacobian_update to update both
        self.dIydI and self.dIydpars
        """
        self.dIydall_full += self.jacobian_update_full
        self.assign_from_dIydall()

    # def prepare_min_update_linearization(
    #     self, current_record, Iy_record, threshold_svd
    # ):
    #     """Computes quantities to update the linearisation matrices, using a record of recently computed Grad-Shafranov solutions.
    #     To be updated.

    #     Parameters
    #     ----------
    #     current_record : np.array
    #         <<current>> parameter values over a time-horizon
    #     Iy_record : np.array
    #         plasma cell currents (over the reduced domain) over a time-horizon
    #     threshold_svd : float
    #         discards singular values that are too small, to obtain a smoother pseudo-inverse
    #     other parameters are passed in as object attributes
    #     """
    #     self.Iy_dv = ((Iy_record - Iy_record[-1:])[:-1]).T

    #     self.current_dv = (current_record - current_record[-1:])[:-1]
    #     self.abs_current_dv = np.mean(abs(self.current_dv), axis=0)

    #     U, S, B = np.linalg.svd(self.current_dv.T, full_matrices=False)

    #     mask = S > threshold_svd
    #     S = S[mask]
    #     U = U[:, mask]
    #     B = B[mask, :]

    #     # delta = Iy_dv@(B.T)@np.diag(1/S)@(U.T)
    #     self.pseudo_inverse = (B.T) @ np.diag(1 / S) @ (U.T)

    # def min_update_linearization(
    #     self,
    # ):
    #     """Returns a minimum update to the Jacobian dIydI based on a set of recently computed Grad-Shafranov solutions.

    #     Parameters
    #     ----------
    #     parameters are passed in as object attributes
    #     """
    #     self.predicted_Iy = np.matmul(self.dIydI, self.current_dv.T)
    #     Iy_dv_d = self.Iy_dv - self.predicted_Iy

    #     delta = Iy_dv_d @ self.pseudo_inverse
    #     return delta
