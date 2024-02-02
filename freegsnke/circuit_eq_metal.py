import numpy as np

from . import machine_config
from . import normal_modes
from .implicit_euler import implicit_euler_solver


class metal_currents:
    """Sets up framework for all metal currents. Calculates residuals of full
    nonlinear circuit equation. Sets up to solve metal circuit equation for
    vacuum shots.

    Parameters
    ----------

    flag_vessel_eig : bool
        Flag to use vessel eigenmodes.
    flag_plasma : bool
        Whether to include plasma in circuit equation. If True, plasma_grids
        must be provided.
    plasma_grids : freegsnke.plasma_grids
        Plasma grids object. Defaults to None.
    max_mode_frequency : float
        Maximum frequency of vessel eigenmodes to include in circuit equation.
        Defaults to 1.
    max_internal_timestep : float
        Maximum internal timestep for implicit euler solver. Defaults to .0001.
    full_timestep : float
        Full timestep for implicit euler solver. Defaults to .0001.
    """

    def __init__(
        self,
        flag_vessel_eig,
        flag_plasma,
        plasma_grids=None,
        max_mode_frequency=1,
        max_internal_timestep=0.0001,
        full_timestep=0.0001,
    ):

        self.n_coils = len(machine_config.coil_self_ind)
        self.n_active_coils = machine_config.n_active_coils

        self.flag_vessel_eig = flag_vessel_eig
        self.flag_plasma = flag_plasma

        self.max_internal_timestep = max_internal_timestep
        self.full_timestep = full_timestep

        if flag_vessel_eig:
            self.max_mode_frequency = max_mode_frequency
            self.make_selected_mode_mask_from_max_freq()
            self.initialize_for_eig(self.selected_modes_mask)

        else:
            self.max_mode_frequency = 0
            self.initialize_for_no_eig()

        if flag_plasma:
            self.plasma_grids = plasma_grids
            self.Mey = plasma_grids.Mey()

        # Dummy voltage vector
        self.empty_U = np.zeros(self.n_coils)

    def make_selected_mode_mask_from_max_freq(self):
        """Creates a mask for the vessel normal modes to include in the circuit
        equation based on the maximum frequency of the modes.
        """
        selected_modes_mask = normal_modes.w_passive < self.max_mode_frequency
        self.selected_modes_mask = np.concatenate(
            (np.ones(self.n_active_coils).astype(bool), selected_modes_mask)
        )
        self.n_independent_vars = np.sum(self.selected_modes_mask)
        print(
            "Input max_mode_frequency corresponds to ",
            self.n_independent_vars - self.n_active_coils,
            " independent vessel normal modes in addition to the ",
            self.n_active_coils,
            " active coils.",
        )

    def initialize_for_eig(self, selected_modes_mask):
        """Initializes the metal currents object for the case where vessel
        eigenmodes are used.

        Parameters
        ----------
        selected_modes_mask : np.ndarray
            Mask for the vessel normal modes to include in the circuit equation.
        """

        self.selected_modes_mask = selected_modes_mask
        self.n_independent_vars = np.sum(self.selected_modes_mask)

        # Id = Vm1 R**(1/2) I
        # to change base to truncated modes
        # I = R**(-1/2) V Id
        self.Vm1 = ((normal_modes.Vmatrix).T)[selected_modes_mask, :]
        self.V = (normal_modes.Vmatrix)[:, selected_modes_mask]

        # Equation is Lambda**(-1)Iddot + I = F
        # where Lambda is such that R12@M-1@R12 = V Lambda V-1
        # w are frequences, eigenvalues of Lambda
        # Note Lambda is not diagonal because of active/passive mode separation
        self.Lambda = self.Vm1 @ normal_modes.lm1r @ self.V
        self.Lambdam1 = self.Vm1 @ normal_modes.rm1l @ self.V

        self.R = machine_config.coil_resist
        self.R12 = machine_config.coil_resist**0.5
        self.Rm12 = machine_config.coil_resist**-0.5
        # R, R12, Rm12 are vectors rather than matrices!

        self.solver = implicit_euler_solver(
            Mmatrix=self.Lambdam1,
            Rmatrix=np.eye(self.n_independent_vars),
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.full_timestep,
        )

        if self.flag_plasma:
            self.forcing_term = self.forcing_term_eig_plasma
        else:
            self.forcing_term = self.forcing_term_eig_no_plasma

    def initialize_for_no_eig(self):
        """Initializes the metal currents object for the case where vessel
        eigenmodes are not used."""
        self.n_independent_vars = self.n_coils
        self.M = machine_config.coil_self_ind
        self.Mm1 = normal_modes.Mm1
        self.R = np.diag(machine_config.coil_resist)
        self.Rm1 = 1 / machine_config.coil_resist  # it's a vector!
        self.Mm1R = self.Mm1 @ self.R
        self.Rm1M = np.diag(1 / machine_config.coil_resist) @ self.M

        # Equation is MIdot + RI = F
        self.solver = implicit_euler_solver(
            Mmatrix=self.M,
            Rmatrix=self.R,
            max_internal_timestep=self.max_internal_timestep,
            full_timestep=self.full_timestep,
        )

        if self.flag_plasma:
            self.forcing_term = self.forcing_term_no_eig_plasma
        else:
            self.forcing_term = self.forcing_term_no_eig_no_plasma

    def reset_mode(
        self,
        flag_vessel_eig,
        flag_plasma,
        plasma_grids=None,
        max_mode_frequency=1,
        max_internal_timestep=0.0001,
        full_timestep=0.0001,
    ):
        """Resets init inputs.

        Parameters
        ----------
        flag_vessel_eig : bool
            Flag to use vessel eigenmodes.
        flag_plasma : bool
            Whether to include plasma in circuit equation. If True, plasma_grids
            must be provided.
        plasma_grids : freegsnke.plasma_grids
            Plasma grids object. Defaults to None.
        max_mode_frequency : float
            Maximum frequency of vessel eigenmodes to include in circuit
            equation.
        max_internal_timestep : float
            Maximum internal timestep for implicit euler solver.
        full_timestep : float
            Full timestep for implicit euler solver.
        """
        control = self.max_internal_timestep != max_internal_timestep
        self.max_internal_timestep = max_internal_timestep

        control += self.full_timestep != full_timestep
        self.full_timestep = full_timestep

        control += flag_plasma != self.flag_plasma
        self.flag_plasma = flag_plasma

        if control * flag_plasma:
            self.plasma_grids = plasma_grids
            self.Mey = plasma_grids.Mey()

        control += flag_vessel_eig != self.flag_vessel_eig
        self.flag_vessel_eig = flag_vessel_eig

        if flag_vessel_eig:
            control += max_mode_frequency != self.max_mode_frequency
            self.max_mode_frequency = max_mode_frequency
        if control * flag_vessel_eig:
            self.initialize_for_eig(self.selected_modes_mask)
        else:
            self.initialize_for_no_eig()

    def forcing_term_eig_plasma(self, active_voltage_vec, Iydot):
        """Right-hand-side of circuit equation in eigenmode basis with plasma.

        Parameters
        ----------
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray
            Vector of rate of change of plasma currents.

        Returns
        -------
        all_Us : np.ndarray
            Voltages.
        """
        all_Us = np.zeros_like(self.empty_U)
        all_Us[: self.n_active_coils] = active_voltage_vec
        all_Us -= self.Mey @ Iydot
        all_Us = np.dot(self.Vm1, self.Rm12 * all_Us)
        return all_Us

    def forcing_term_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        """Right-hand-side of circuit equation in eigenmode basis without
        plasma.

        Parameters
        ----------
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray, optional
            This is not used.

        Returns
        -------
        all_Us : np.ndarray
            Voltages."""
        all_Us = self.empty_U.copy()
        all_Us[: self.n_active_coils] = active_voltage_vec
        all_Us = np.dot(self.Vm1, self.Rm12 * all_Us)
        return all_Us

    def forcing_term_no_eig_plasma(self, active_voltage_vec, Iydot):
        """Right-hand-side of circuit equation in normal mode basis with plasma.

        Parameters
        ----------
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray
            Vector of rate of change of plasma currents.

        Returns
        -------
        all_Us : np.ndarray
            Voltages.
        """
        all_Us = self.empty_U.copy()
        all_Us[: self.n_active_coils] = active_voltage_vec
        all_Us -= np.dot(self.Mey, Iydot)
        return all_Us

    def forcing_term_no_eig_no_plasma(self, active_voltage_vec, Iydot=0):
        """Right-hand-side of circuit equation in normal mode basis without
        plasma.

        Parameters
        ----------
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray, optional
            This is not used.

        Returns
        -------
        all_Us : np.ndarray
            Voltages.
        """
        all_Us = self.empty_U.copy()
        all_Us[: self.n_active_coils] = active_voltage_vec
        return all_Us

    def IvesseltoId(self, Ivessel):
        """Given Ivessel, returns Id.

        Parameters
        ----------
        Ivessel : np.ndarray
            Vessel currents.

        Returns
        -------
        Id : np.ndarray

        """
        Id = np.dot(self.Vm1, self.R12 * Ivessel)
        return Id

    def IdtoIvessel(self, Id):
        """Given Id, returns Ivessel.

        Parameters
        ----------"""
        Ivessel = self.Rm12 * np.dot(self.V, Id)
        return Ivessel

    def stepper(self, It, active_voltage_vec, Iydot=0):
        """Steps the circuit equation forward in time.

        Parameters
        ----------
        It : np.ndarray
            Currents at time t.
        active_voltage_vec : np.ndarray
            Vector of active coil voltages.
        Iydot : np.ndarray or float, optional
            Vector of rate of change of plasma currents. Defaults to 0.

        Returns
        -------
        It : np.ndarray
            Currents at time t+dt.
        """
        forcing = self.forcing_term(active_voltage_vec, Iydot)
        It = self.solver(It, forcing)
        return It

    def current_residual(self, Itpdt, Iddot, forcing_term):
        """Calculates the residual of the circuit equation in normal modes.

        $$\Lambda^{-1} \dot{I} + I - F = \text{residual}$$.

        Parameters
        ----------
        Itpdt : np.ndarray
            Currents at time t+dt.
        Iddot : np.ndarray
            Rate of change of currents at time t.
        forcing_term : np.ndarray
            Forcing term of circuit equation.

        Returns
        -------
        residual : np.ndarray
            Residual of circuit equation.
        """
        residual = np.dot(self.Lambdam1, Iddot)
        residual += Itpdt
        residual -= forcing_term
        return residual
