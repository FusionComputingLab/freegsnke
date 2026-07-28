"""
Module to implement a Plasma Control System (PCS) in FreeGSNKE.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
  
You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
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

    vertical_coils : list of str
        List of all active coils being used for vertical control.

    ctrl_targets : list of str
        List of all shape control targets (e.g., X-points, strike points).

    plasma_target : list of str
        List of all plasma control targets.

    shape_control_mode : str
        Select which shape control algorithm to use (see shape_category.py).

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

    vc_generator : object, optional
        An optional class object for applying emulated virtual circuits. If not
        provided, default waveform-defined VCs will be used.

    vc_update_rate : float, optional
        Optional argument to specify how often, in seconds, new VCs are computed with vc_generator.
        If None provided, defaults to zero and new VC computed at every iteration.
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
        vertical_coils,
        ctrl_targets,
        plasma_target,
        shape_control_mode=None,
        vc_generator=None,
        vc_update_rate=None,
    ):
        """
        Initialise the top-level control system, composing all sub-controllers.

        Builds and wires together the individual controllers (plasma, shape,
        virtual circuits, systems, PF, vertical, and coil activation), each
        given its own slice of input data plus whatever coil groupings and
        targets it needs.

        Parameters
        ----------
        plasma_data : dict
            Time-series data passed to `PlasmaController`.
        shape_data : dict
            Time-series data passed to `ShapeController`.
        circuits_data : dict
            Time-series data passed to `VirtualCircuitsController`.
        systems_data : dict
            Time-series data passed to `SystemsController`.
        pf_data : dict
            Time-series data passed to `PFController`.
        vertical_data : dict
            Time-series data passed to `VerticalController`.
        coil_activation_data : dict
            Time-series data passed to `CoilActivationController`.
        active_coils : list of str
            Names of all coils that are active in this configuration. Passed
            to `CoilActivationController`.
        ctrl_coils : list of str
            Names of the coils used for shape/virtual-circuit control. Passed
            to `VirtualCircuitsController` and `SystemsController`.
        vertical_coils : list of str
            Names of the coils used for vertical position control.
        ctrl_targets : list or dict
            Shape control targets, passed to `ShapeController` and
            `VirtualCircuitsController`.
        plasma_target : float or array_like or dict
            Target value(s) for the plasma, passed to
            `VirtualCircuitsController`.
        shape_control_mode : str, optional
            Mode used by `ShapeController` to determine how shape control is
            performed. If None, `ShapeController`'s default mode is used.
        vc_generator : callable, optional
            Generator function used by `VirtualCircuitsController` to produce
            virtual circuit outputs. If None, `VirtualCircuitsController`'s
            default generator is used.
        vc_update_rate : float, optional
            Rate at which `VirtualCircuitsController` updates its virtual
            circuits. If None, `VirtualCircuitsController`'s default update
            rate is used.

        Attributes
        ----------
        active_coils : list of str
            Stored copy of `active_coils`.
        ctrl_coils : list of str
            Stored copy of `ctrl_coils`.
        vertical_coils : list of str
            Stored copy of `vertical_coils`.
        ctrl_targets : list or dict
            Stored copy of `ctrl_targets`.
        plasma_target : float or array_like or dict
            Stored copy of `plasma_target`.
        PlasmaController : PlasmaController
            Sub-controller handling plasma-related quantities.
        ShapeController : ShapeController
            Sub-controller handling shape control.
        VirtualCircuitsController : VirtualCircuitsController
            Sub-controller handling virtual circuit generation and control.
        SystemsController : SystemsController
            Sub-controller handling systems-level control.
        PFController : PFController
            Sub-controller handling PF coil control.
        VerticalController : VerticalController
            Sub-controller handling vertical position control.
        CoilActivationController : CoilActivationController
            Sub-controller handling per-coil activation state.

        """

        # coil ordering
        self.active_coils = active_coils
        self.ctrl_coils = ctrl_coils
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
            mode=shape_control_mode,
        )

        self.VirtualCircuitsController = VirtualCircuitsController(
            data=circuits_data,
            ctrl_coils=self.ctrl_coils,
            ctrl_targets=self.ctrl_targets,
            plasma_target=self.plasma_target,
            vc_generator=vc_generator,
            vc_update_rate=vc_update_rate,
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
        ip_err_prev,
        T_meas,
        T_err_prev,
        T_hist_prev,
        I_approved_prev,
        I_meas,
        V_approved_prev,
        zip_meas,
        zipv_meas,
        active_coil_resists,
        dt_simulator=None,
        emulated_VC_targets=None,
        emulated_VC_targets_calc=None,
        emulator_coils_calc=None,
        emu_inputs=None,
        vc_update_rate=None,
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
            Time step to run the controllers at (must have 'dt = dt_simulator/n' where n is a natural number) [s].

        ip_meas : float
            Measured plasma current [A].

        ip_hist_prev : float
            Previous value of the integrated plasma current error [A.s].

        ip_err_prev : float
            Previous value of the plasma current error [A].

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

        dt_simulator : float
            Time step of the simulator (must have 'dt_simulator = dt*n' where n is a natural number) [s].

        emulated_VC_targets : list of str , optional
            List of targets to be controlled using the emulated VC's. Must be subset of
            ctrl_targets, and subset/equal to emulated_VC_targets_calc. Those not defined in this list will be taken from waveform-defined
            VCs.

        emulated_VC_targets_calc : list of str , optional
            List of targets to be used when performing pseudoinverse of jacobian when calculating the emulated VC.

        emulator_coils_calc : list of str, optional
            List of coils to use in emulated VC compuation. These are coils to use in computing shape sensitivity matrix.

        verbose : bool, optional
            If True, prints diagnostic information from subsystem controllers.

        Returns
        -------
        V_active : numpy.ndarray
            Final (all active) coil voltage demands after applying all constraints [V].

        ip_hist : list of float
            Updated integrated plasma current error [A.s].

        ip_err : list of float
            Updated plasma current error [A].

        T_err : numpy.ndarray
            Updated shape target filtered error signal (used for damping) [m].

        T_hist : list of numpy.ndarray
            Updated shape target integral term (used for PI control) [m.s].

        I_approved : numpy.ndarray
            Approved coil currents after applying perturbations and clipping [A].

        coil_resists : numpy.ndarray
            Active coil resistances to be used (some coils may be on or off at time t) [Ohms].
        """

        # check timesteps align correctly (default: dt_simulator = dt)
        if dt_simulator is None:
            dt_simulator = dt
            n = 1
        else:
            n = round(dt_simulator / dt)  # nearest integer
            if n < 1:
                n = 1  # enforce natural number >= 1

        # call the PCS class (n times per simulator step if requested)
        V_actives = []
        for i in range(0, n):

            # plasma category
            self.dip_dt, ip_hist, ip_err = self.PlasmaController.run_control(
                t=t + (i * dt),
                dt=dt,
                ip_meas=ip_meas,
                ip_hist_prev=ip_hist_prev,
                ip_err_prev=ip_err_prev,
            )

            # update "history" terms
            ip_hist_prev = ip_hist.copy()
            ip_err_prev = ip_err.copy()

            # shape category
            self.dT_dt, T_err, T_hist = self.ShapeController.run_control(
                t=t + (i * dt),
                dt=dt,
                T_meas=T_meas,
                T_err_prev=T_err_prev,
                T_hist_prev=T_hist_prev,
            )

            # update "history" terms
            T_err_prev = T_err.copy()
            T_hist_prev = T_hist.copy()

            # virtual circuits category
            self.I_unapproved, self.dI_dt_unapproved = (
                self.VirtualCircuitsController.run_control(
                    t=t + (i * dt),
                    dt=dt,
                    dip_dt=self.dip_dt,
                    dT_dt=self.dT_dt,
                    I_approved_prev=I_approved_prev,
                    emulated_VC_targets=emulated_VC_targets,
                    emulated_VC_targets_calc=emulated_VC_targets_calc,
                    emulator_coils_calc=emulator_coils_calc,
                    emu_inputs=emu_inputs,
                )
            )

            # systems category
            self.I_approved, self.dI_dt_approved = self.SystemsController.run_control(
                t=t + (i * dt),
                dt=dt,
                I_unapproved=self.I_unapproved,
                dI_dt_unapproved=self.dI_dt_unapproved,
                verbose=verbose,
            )

            # update "history" terms
            I_approved_prev = self.I_approved.copy()

            # PF category
            self.V_ctrl = self.PFController.run_control(
                t=t + (i * dt),
                dt=dt,
                I_meas=I_meas,
                I_approved=self.I_approved,
                dI_dt_approved=self.dI_dt_approved,
                V_approved_prev=V_approved_prev,
                verbose=verbose,
            )

            # update "history" terms
            V_approved_prev = self.V_ctrl.copy()

            # vertical category
            self.V_vertical = self.VerticalController.run_control(
                t=t + (i * dt),
                dt=dt,
                ip_meas=ip_meas,
                zip_meas=zip_meas,
                zipv_meas=zipv_meas,
            )

            # coil activations category
            coil_resists = self.CoilActivationController.run_control(
                t=t + (i * dt),
                dt=dt,
                active_coil_resists=active_coil_resists,
            )

            # lookup dictionaries
            ctrl_dict = dict(zip(self.ctrl_coils, self.V_ctrl))
            vert_dict = dict(zip(self.vertical_coils, np.array([self.V_vertical])))

            # build active coil voltages vector
            V_actives.append(
                np.array(
                    [ctrl_dict.get(c, vert_dict.get(c, 0.0)) for c in self.active_coils]
                )
            )

        # average the requested voltages for use in simulator
        V_active = np.mean(V_actives, axis=0)

        return V_active, ip_hist, ip_err, T_err, T_hist, self.I_approved, coil_resists
