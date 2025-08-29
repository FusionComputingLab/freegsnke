"""
Module to implement a Plasma Control System (PCS) in FreeGSNKE.

"""

# imports
import numpy as np

from .coil_activation_category import CoilActivationController
from .pf_category import PFController
from .plasma_category import PlasmaController
from .shape_category import ShapeController
from .systems_category import SystemsController
from .vertical_category import VerticalController
from .virtual_circuits_category import VirtualCircuitsController


class PlasmaControlSystem:
    """
    A high-level class for managing multiple controllers in a plasma
    control system.

    This class integrates several subsystem controllers responsible for different aspects
    of plasma control, including shape control, vertical stabilization, coil current
    regulation, and virtual circuit modelling. It provides a unified interface for
    coordinating these controllers based on time-dependent input waveforms.

    Attributes
    ----------
    active_coils : list of str
        List of all active coils used.

    ctrl_coils : list of str
        List of all active coils being used for shape control.

    solenoid_coils : list of str
        List of all active coils being used for plasma current control.

    vertical_coils : list of str
        List of all active coils being used for vertical control.

    ctrl_targets : list of str
        List of all shape control targets (e.g., X-points, strike points).

    plasma_target : list of str
        List of all plasma control targets.

    PlasmaController : PlasmaController
        Handles plasma current control.

    ShapeController : ShapeController
        Handles shape target control.

    VirtualCircuitsController : VirtualCircuitsController
        Computes unapproved coil currents in ctrl coils using virtual circuits.

    SystemsController : SystemsController
        Applies perturbations and enforces coil current and ramp rate limits to
        find approved coil currents.

    PFController : PFController
        Transforms approved coil currents into ctrl coil voltages.

    VerticalControl : VerticalController
        Controls vertical plasma position via vertical control coil.

    CoilActivationControl : CoilActivationController
        Controls coil resistances, depending on if they're switched on or off.

    emulated_VCs : object, optional
        An optional class object for applying emulated virtual circuits. If not
        provided, deafult waveform-defined VCs will be used.
    """

    def __init__(
        self,
        plasma_data,
        shape_data,
        circuits_data,
        systems_data,
        pf_data,
        vertical_data,
        coil_activation_data,
        active_coils,
        ctrl_coils,
        solenoid_coils,
        vertical_coils,
        ctrl_targets,
        plasma_target,
        emulated_VCs=None,
    ):

        # coil ordering
        self.active_coils = active_coils
        self.ctrl_coils = ctrl_coils
        self.solenoid_coils = solenoid_coils
        self.vertical_coils = vertical_coils

        # shape targets
        self.ctrl_targets = ctrl_targets
        self.plasma_target = plasma_target

        # initialise controllers and assign data to each
        self.PlasmaController = PlasmaController(
            data=plasma_data,
        )

        self.ShapeController = ShapeController(
            data=shape_data,
            ctrl_targets=self.ctrl_targets,
        )

        self.VirtualCircuitsController = VirtualCircuitsController(
            data=circuits_data,
            ctrl_coils=self.ctrl_coils,
            ctrl_targets=self.ctrl_targets,
            plasma_target=self.plasma_target,
            emulated_VCs=emulated_VCs,
        )

        self.SystemsController = SystemsController(
            data=systems_data,
            ctrl_coils=self.ctrl_coils,
        )

        self.PFController = PFController(
            data=pf_data,
        )

        self.VerticalController = VerticalController(
            data=vertical_data,
        )

        self.CoilActivationController = CoilActivationController(
            data=coil_activation_data,
            active_coils=self.active_coils,
        )

    def calculate_ctrl_voltages(
        self,
        t,
        dt,
        ip_meas,
        ip_hist_prev,
        T_meas,
        T_err_prev,
        T_hist_prev,
        I_approved_prev,
        I_meas,
        V_approved_prev,
        zip_meas,
        zipv_meas,
        active_coil_resists,
        emulated_VC_targets=None,
        emulator_coils=None,
        emu_inputs=None,
        verbose=False,
    ):
        """
        Run the full control pipeline to compute approved coil voltage commands.

        This method coordinates all subsystem controllers (plasma current, shape control,
        virtual circuits, systems constraints, PF, and vertical) to compute the final
        voltage commands for the coils. It also returns updated histories and error signals
        for use in the next control cycle.

        Parameters
        ----------
        t : float
            Current time [s].

        dt : float
            Time step [s].

        ip_meas : float
            Measured plasma current [A].

        ip_hist_prev : float
            Previous value of the integrated plasma current error [A.s].

        T_meas : np.ndarray
            Measured values of the shape targets at the current time [m].

        T_err_prev : np.ndarray
            Previously shape target filtered error signal (used for damping) [m].

        T_hist_prev : np.ndarray
            Previous shape target integral term (used for PI control) [m.s].

        I_approved_prev : numpy.ndarray
            Previously approved coil currents [A].

        I_meas : numpy.ndarray
            Measured coil currents at the current time step [A].

        V_approved_prev : numpy.ndarray
            Previously approved control coil voltages from the last control step [V].

        zip_meas : float
            Measured vertical position of the plasma multiplied by measured Ip [A.m].

        zipv_meas : float
            Measured vertical velocity of the plasma multiplied by measured Ip [A.m/s].

        active_coil_resists : numpy.ndarray
            Array of active coil resistances when coils are switched on [Ohms].

        emulated_VC_targets : list of str , optional
            List of targets to be controlled using the emulated VC's. Must be subset of
            ctrl_targets. Those not defined in this list will be taken from waveform-defined
            VCs.

        emulator_coils : list of str, optional
            List of coils to use in emulated VC compuation. These are coils to use in computing shape sensitivity matrix.

        verbose : bool, optional
            If True, prints diagnostic information from subsystem controllers.

        Returns
        -------
        V_active : numpy.ndarray
            Final (all active) coil voltage demands after applying all constraints [V].

        ip_hist : list of float
            Updated integrated plasma current error [A.s].

        T_err : numpy.ndarray
            Updated shape target filtered error signal (used for damping) [m].

        T_hist : list of numpy.ndarray
            Updated shape target integral term (used for PI control) [m.s].

        I_approved : numpy.ndarray
            Approved coil currents after applying perturbations and clipping [A].

        coil_resists : numpy.ndarray
            Active coil resistances to be used (some coils may be on or off at time t) [Ohms].
        """

        # plasma category
        dip_dt, ip_hist = self.PlasmaController.run_control(
            t=t,
            dt=dt,
            ip_meas=ip_meas,
            ip_hist_prev=ip_hist_prev,
        )

        # shape category
        dT_dt, T_err, T_hist = self.ShapeController.run_control(
            t=t,
            dt=dt,
            T_meas=T_meas,
            T_err_prev=T_err_prev,
            T_hist_prev=T_hist_prev,
        )

        # virtual circuits category
        I_unapproved, dI_dt_unapproved = self.VirtualCircuitsController.run_control(
            t=t,
            dt=dt,
            dip_dt=dip_dt,
            dT_dt=dT_dt,
            I_approved_prev=I_approved_prev,
            emulated_VC_targets=emulated_VC_targets,
            emulator_coils=emulator_coils,
            emu_inputs=emu_inputs,
        )

        # systems category
        I_approved, dI_dt_approved = self.SystemsController.run_control(
            t=t,
            dt=dt,
            I_unapproved=I_unapproved,
            dI_dt_unapproved=dI_dt_unapproved,
            verbose=verbose,
        )

        # PF category
        V_ctrl = self.PFController.run_control(
            t=t,
            dt=dt,
            I_meas=I_meas,
            I_approved=I_approved,
            dI_dt_approved=dI_dt_approved,
            V_approved_prev=V_approved_prev,
            verbose=verbose,
        )

        # vertical category
        V_vertical = self.VerticalController.run_control(
            t=t,
            dt=dt,
            ip_meas=ip_meas,
            zip_meas=zip_meas,
            zipv_meas=zipv_meas,
        )

        # coil activations category
        coil_resists = self.CoilActivationController.run_control(
            t=t,
            dt=dt,
            active_coil_resists=active_coil_resists,
        )

        # lookup dictionaries
        ctrl_dict = dict(zip(self.ctrl_coils, V_ctrl))
        vert_dict = dict(zip(self.vertical_coils, np.array([V_vertical])))

        # build active coil voltages vector
        V_active = np.array(
            [ctrl_dict.get(c, vert_dict.get(c, 0.0)) for c in self.active_coils]
        )

        return V_active, ip_hist, T_err, T_hist, I_approved, coil_resists


# class Simulator:
#     """Simulator class to interface PCS control with dynamic solving in freegsnke"""

#     def __init__(
#         self,
#         pcs_controller: PlasmaControlSystem,
#         stepping,
#     ):
#         """initialse simulator

#         Parameters:
#         pcs  : PlasmaControlSystem
#             instance of PCS class to compute voltages
#         stepping : FreeGSNKE NL solver
#             dynamic forward solver in freegsnke

#         """
#         self.stepping = stepping
#         self.controller = pcs_controller

#     def initialise_VCH(
#         self,
#         stepping,
#         target_relative_tolerance: float = 1e-7,
#     ):
#         """initialise the VCH object as class attribute.
#         This must be done after the class is initialised and before first call to calculate_blended_target_deltas


#         Inputs
#         ------
#         stepping : object
#             stepping object, to provide solver information
#         target_relative_tolerance : float
#             target relative tolerance

#         Returns
#         -------
#         None
#             Modifies the class attribute self.VCH
#         """
#         self.VCH = vc.VirtualCircuitHandling()
#         self.VCH.define_solver(
#             stepping.NK, target_relative_tolerance=target_relative_tolerance
#         )
#         print("Initialised VCH in shape controller")

#     def get_observed_from_eqi(
#         self,
#     ):
#         """
#         Compute or retrieve all appropriate observed quantites from equilibrium
#         Shape targets, plasma current, etc.

#         Returns
#         -------

#         """

#         pass

#     def update_eqi(self, voltages, linear=True):
#         """
#         update and solve equilibrium with new voltages as inputs
#         """
#         if linear == True:
#             print("updating equilibrium  : LINEAR")
#             self.stepping.nlstepper(
#                 active_voltage_vec=voltages,
#                 linear_only=True,  # linearise solve only
#                 verbose=False,
#                 # custom_coil_resist=coil_resist,   #options for restistances/inductances used in solve
#                 # custom_self_ind=coil_ind)
#             )
#         elif linear == False:
#             print("updating equilibrium : NON-LINEAR")
#             self.stepping.nlstepper(
#                 active_voltage_vec=voltages,
#                 linear_only=False,  # Non linear solve
#                 verbose=False,
#                 # custom_coil_resist=coil_resist,   #options for restistances/inductances used in solve
#                 # custom_self_ind=coil_ind)
#             )
#         print("equi updated")

#     def simulate(self, control_times):
#         """run simulation ..."""

#         # for time in times
#         # compute voltage request
#         # solve for new equilibrium
#         # compute new plasma shape targets/currents
#         pass

#     def get_fgs_inductance_resistance(self, stepping):
#         """
#         get inductance matrix and coil resistances from stepping object (Freegsnke)

#         Inputs :
#         --------
#         stepping : object
#             stepping object

#         Returns
#         -------
#         inductance_full : np.array
#             full inductance matrix in freegsnke for all active coils

#         coil_resist : np.array
#             coil resistances in freegsnke for all active coils
#         """

#         # assign equi and profiles objects
#         # self.stepping = stepping
#         n_active_coils = (
#             stepping.n_active_coils
#         )  # could also be eq.tokamak.n_active_coils
#         print("number active coils", n_active_coils)
#         tok = stepping.eq1.tokamak
#         active_coils = tok.coils_list[:n_active_coils]

#         inductance_full = tok.coil_self_ind[: len(active_coils), : len(active_coils)]
#         coil_resist = tok.coil_resist[: len(active_coils)]
#         print(
#             "Inductances and resistances retrieved for all active coils :", active_coils
#         )
#         coil_order_dictionary = {coil: i for i, coil in enumerate(active_coils)}

#         return {
#             "inductance_full": inductance_full,
#             "coil_resist": coil_resist,
#             "coils": active_coils,
#             "coil_order_dictionary": coil_order_dictionary,
#         }
