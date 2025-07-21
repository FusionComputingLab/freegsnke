"""
Module for computing feedback control voltages from virtual circuits.

"""

from copy import deepcopy

import numpy as np

from .. import machine_config
from .. import virtual_circuits as vc  # import the virtual circuit class
from ..equilibrium_update import Equilibrium
from ..nonlinear_solve import nl_solver
from ..virtual_circuits import VirtualCircuit
from .shape_scheduling import ShapeTargetScheduler


def get_inductance_resistance(stepping):
    """get inductance matrix and coil resistances from stepping object (Freegsnke)

    Inputs :
    --------
    stepping : object
        stepping object

    Returns
    -------
    inductance_full : np.array
        full inductance matrix in freegsnke for all active coils

    coil_resist : np.array
        coil resistances in freegsnke for all active coils
    """

    # assign equi and profiles objects
    # self.stepping = stepping
    n_active_coils = stepping.n_active_coils  # could also be eq.tokamak.n_active_coils
    print("number active coils", n_active_coils)
    tok = stepping.eq1.tokamak
    active_coils = tok.coils_list[:n_active_coils]

    inductance_full = tok.coil_self_ind[: len(active_coils), : len(active_coils)]
    coil_resist = tok.coil_resist[: len(active_coils)]
    print("Inductances and resistances retrieved for all active coils :", active_coils)
    coil_order_dictionary = {coil: i for i, coil in enumerate(active_coils)}

    return {
        "inductance_full": inductance_full,
        "coil_resist": coil_resist,
        "coils": active_coils,
        "coil_order_dictionary": coil_order_dictionary,
    }


class ShapeController:
    """
    Class to implement control voltages from virtual circuit, and given a set of observed target values, and a set of requested target values.

    Attributes :
    ------------
    eq : eq object
    profiles : profiles object
    active_coils : list of all active coils
    coils : list of coils to be used in controller
    active_coil_order_dictionary : dictionary mapping coil names to their order in the list of active coils
    inductance_full : full inductance matrix for all active coils
    VCH (virtual circuit handling class)
    feedback_target_scheduler (target scheduler class)


    Methods :
    reshape_inductance : retrieve inductance matrix from machine config, and select rows/columns
    calc_vc_from_eq : retrieve from file or compute a virtual circuit object from freegsnke or NN emulator.
    calculate_blended_target_deltas : compute feedback voltages from a virtual circuit object and a set of target shifts.
    control_shape_rates : compute feedback voltages from a time provided by retrieving targets at given time from the target waveformr and computing with calculate_blended_target_deltas.
    """

    def __init__(
        self,
        feedback_target_scheduler: ShapeTargetScheduler,
        active_coils: list[str],
        control_coils: list[str],
        machine_parameters,
        prev_output: np.array = None,
        pi_state=None,
        integral_state=None,
    ):
        """
        Initialize the control voltages class

        Parameters
        ----------
        eq : equilibrium object
            equilibrium object
        profiles : list of profiles
            list of profiles
        feedback_target_scheduler : TargetScheduler object
            TargetScheduler object - contains targets and vc schedule for simulation.
        active_coils : list[str]
            list of all coil names to be used by simulation. Includes shaping and vertical control coils
        control_coils : list[str]
            list of coils used for shape control. These are used by emulators.
        machine_parameters : dict
            dictionary containing full inductance matrix and coil resistances
        prev_output : np.array
            array of target rates (output of shapes category) at previous timestep.
            If not provided it defaults to array of zeros, of size len(all_targets)
        pi_state  : np.array
            array of PI state for all controllable targets. Only used if Integral control active
        integral_state : np.array
            array of integral state for all controllable targets. only used in integral computation.
        """
        self.active_coils = active_coils

        # create a dictionary to map coil names to their order in the list
        self.active_coil_order_dictionary = {
            coil: i for i, coil in enumerate(self.active_coils)
        }

        # assign coils to default or
        if control_coils is None:
            print("initialising with default all active coils")
            self.control_coils = deepcopy(self.active_coils)
        else:
            print("initialising with custom coils")
            self.control_coils = control_coils

        # initialise a target scheduler object
        self.feedback_target_scheduler = feedback_target_scheduler
        print("target scheduler flag :", self.feedback_target_scheduler.vc_flag)

        if prev_output is None:
            self.prev_output = np.zeros(
                len(feedback_target_scheduler.get_all_targets())
            )
        else:
            self.prev_output = prev_output

        # set machine parameters (inductance and resistances for coils)
        self.inductance_full = machine_parameters["inductance_full"]
        self.coil_resist = machine_parameters["coil_resist"]
        self.machine_coils = machine_parameters["coils"]
        self.machine_param_coil_order = machine_parameters["coil_order_dictionary"]

        # # reorder inductance matrix and coil resistances to match coil order
        # # ### ??? inducnace for active coils or control coils ???
        # # self.inductance_full = self.reshape_inductance(coils=self.active_coils)
        # # self.coil_resist = self.reorder_resistance(coils=self.active_coils)
        # self.inductance_full = self.reshape_inductance(coils=self.control_coils)
        # self.coil_resist = self.reorder_resistance(coils=self.control_coils)
        # # reduced inductance matrix for control coils
        # self.inductance_reduced = self.reshape_inductance(coils=self.control_coils)
        # # initialise the VCH object

        ## ?? add in pi state / integral term ??##
        # self.pi_state = pi_state
        # self.x_int = deepcopy(pi_state)
        print("Shape controller initialised")
        print("all active", self.active_coils)
        print("control coils", self.control_coils)
        print("Now please initialise the VCH object with .initialise_VCH(stepping)")

    def initialise_VCH(self, stepping, target_relative_tolerance=1e-7):
        """initialise the VCH object as class attribute.
        This must be done after the class is initialised and before first call to calculate_blended_target_deltas


        Inputs
        ------
        stepping : object
            stepping object, to provide solver information
        target_relative_tolerance : float
            target relative tolerance

        Returns
        -------
        None
            Modifies the class attribute self.VCH
        """
        self.VCH = vc.VirtualCircuitHandling()
        self.VCH.define_solver(
            stepping.NK, target_relative_tolerance=target_relative_tolerance
        )
        print("Initialised VCH in shape controller")

    def reshape_inductance(self, coils=None):
        """
        Select appropriate inductance rows and columns from inductance matrix, given set of coils in the VC.

        parameters
        ----------
        coils : list[str] (optional)
            list of coil names. If None provided, defaults to control_coils

        Returns
        -------
        inductance_reduced : np.array
            inductance matrix of reduced set of coils. Also updates inductance matrix attribute


        """
        if coils is None:  # use default of all active coils from tokamak
            print(
                "Inductance matrix for default of default reduced set of active coils"
            )
            coils = self.control_coils
        else:  # use coils provided and select apropriate part of inductance matrix
            print(f"Inductance matrix for coils provided {coils}")
            pass

        # create mask for selecting part of inductance matrix
        mask = [self.machine_param_coil_order[coil] for coil in coils]
        print("coil ordering mask ", mask)
        inductance_reduced = self.inductance_full[np.ix_(mask, mask)]

        return inductance_reduced

    def reorder_resistance(self, coils):
        """
        Reorder coil resistances to match coil order

        Parameters
        ----------
        coils : list[str]
            ordering of coils to reorder restitance

        Returns
        -------
        coil_resist
            reorders in place the coil resistance array

        """
        mask = [self.machine_param_coil_order[coil] for coil in coils]

        return self.coil_resist[np.ix_(mask)]

    ## this function will be replaced by instance of build virtual circuit class.
    def calc_vc_from_eq(self, targets, eq, profiles, coils=None):
        """
        Compute a VC using freegsnke VirtualCircuitHandling.

        Parameters
        ----------
        eq : object
            equilibrium object

        profiles : object
            profiles object

        targets : list[str]
            list of targets

        coils : list[str]
            list of coils (optional)


        Returns
        -------
        virtual_circuit : object
            virtual circuit object
        """

        # if targets and coils are provided, update targets/coils attributes
        if coils is None:
            coils = self.control_coils

        print("building virtual circuit from freegsnke")
        self.VCH.calculate_VC(
            eq,
            profiles,
            coils=coils,
            targets=targets,
            targets_options=None,
        )

        # get the virtual circuit object
        virtual_circuit = self.VCH.latest_VC

        return virtual_circuit

    def calculate_target_deltas(
        self,
        targets_req,
        targets_obs,
    ):
        """compute the the raw error term targ_required - targ_observed

        Parameters
        ----------
        targets_req : np.array()
            array of required target values
        targets_obs : np.array()
            array of observed/measured target values

        Returns
        -------
        target_deltas : np.array()
            array of deltas
        """

        # check dimensions of target values
        assert len(targets_req) == len(
            targets_obs
        ), "The target required and observed vectors are not the same length"

        # shifts required
        target_deltas = targets_req - targets_obs
        return target_deltas

    def calculate_damped_target_deltas(
        self,
        target_deltas,
        prev_err,
        damp_factor=1,
    ):
        """Calculate the target damped shape deltas.

        output = (1-1/damping)*prev_err + 1/damping *

        Parameters
        ----------
        target_deltas : np.array
            array of the raw difference between observed and requried target
        prev_err : np.array
            array of damped target deltas at previous time step
        damp_factor : float
            damping factor used to compute damped deltas. defaults to 1 (no damping)


        Returns
        -------
        targ_err_t : np.array
            error term (damped) at current time. Has units of m
        target_deltas : np.array
            array of raw error term (T_required - T_observed). Has units of m.
        """

        # total damped target error at current time
        targ_err_t = (1 - 1 / damp_factor) * prev_err + target_deltas / damp_factor

        # update prev_output
        self.prev_output = 1.0 * targ_err_t

        return targ_err_t

    def calculate_integral_deltas(self, deltas, pi_state, dt, K_int_arr, blends):
        """compute updated integral error term and PI state

        Parameters
        ----------
        deltas  : np.array
            array of damped deltas (targ_err from calc proportioal targets method above)
        pi_state : np.array
            PI state at previous time step
        dt : float
            time interval
        K_int_array : np.array
            array of integral gains for all controlled targets
        blends : np.array
            blend array for all the controlled targets

        Returns
        -------
        pi_state : np.array
            updated PI state
        delta_int : np.array
            integral term

        """
        # ??? add in a pi state attribute to class ??
        x_int = pi_state + 0.5 * K_int_arr @ deltas * dt
        pi_state_new = pi_state + blends * K_int_arr @ deltas * dt
        return x_int, pi_state_new

    @staticmethod
    def recompute_vc_from_sensitivity(virtual_circuit, targets):
        """
        Recompute a virtual circuit from the sensitivity matrix, using the targets provided.

        Parameters
        ----------
        virtual_circuit : object
            virtual circuit object

        targets : list[str]
            list of target names

        Returns
        -------
        virtual_circuit : object
            virtual circuit object
        """
        # get the sensitivity matrix, and select columns corresponding to the targets
        sensitivity_matrix = virtual_circuit.shape_matrix
        vc_targ_order_dict = dict(
            zip(virtual_circuit.targets, np.arange(len(virtual_circuit.targets)))
        )
        vc_coil_order_dict = dict(
            zip(virtual_circuit.coils, np.arange(len(virtual_circuit.coils)))
        )

        # if virtual_circuit.coils != self.control_coils:
        #     print(
        #         "Warning : the virtual circuit provided does not match the control coils"
        #     )
        #     # raise error or shrink the vc to the control coils?????
        #     control_coils_indices = np.array(
        #         [vc_coil_order_dict[coil] for coil in self.control_coils]
        #     )

        sens_reduced = sensitivity_matrix[
            np.array([vc_targ_order_dict[targ] for targ in targets]),
        ]
        vc_mat_reduced = np.linalg.pinv(sens_reduced)

        targs_reduced = [
            virtual_circuit.targets[i]
            for i in np.array([vc_targ_order_dict[targ] for targ in targets])
        ]
        # targ_values = np.array(
        #     [
        #         # virtual_circuit.targets_val[i]
        #         targets_obs[i]
        #         for i in np.array([vc_targ_order_dict[targ] for targ in targets])
        #     ]
        # )

        virtualcircuit = vc.VirtualCircuit(
            name="recomputed_vc",
            eq=virtual_circuit.eq,
            profiles=virtual_circuit.profiles,
            shape_matrix=sens_reduced,
            VCs_matrix=vc_mat_reduced,
            targets=targs_reduced,
            targets_val=None,
            non_standard_targets=None,
            targets_options=None,
            coils=virtual_circuit.coils,
        )

        return virtualcircuit

    def calculate_blended_target_deltas(
        self,
        proportional_deltas,
        targets_blends,
        prop_shape_gains,
        ff_deltas,
        x_int=None,
        integral_gains=None,
    ):
        """
        Compute the blended combination of Feedforward and feedback shape rates.

        blended_rate = (1-blend)*FF_rate + blend * FB_rate_proportional + Integral rate

        Parameters
        ----------
        proportional_deltas : np.array
            array of damped feedback deltas for all targets. Has units of m.
        targets_blends : np.array
            array of blends for all targets. must be same order as targets above.
        prop_shape_gains : array()
            array of proportional shape gains
        ff_deltas : np.array
            array of gradients of the feedfoward waveforms.
        x_int : np.array()
            integral term for targets
        integral_gains : np.array
            integral gains array for all controlled targets
        Returns
        -------
        blended_target_deltas : np.array
            array of blended ff and fb shape target rates. Has units of m/s.
        """

        blended_target_deltas = (
            targets_blends * prop_shape_gains * proportional_deltas
            + (1 - targets_blends) * ff_deltas
        )

        return blended_target_deltas

    def apply_shape_vc(
        self,
        targets: list[str],
        target_deltas: np.array,
        virtual_circuit: VirtualCircuit,
        reshape=False,
    ):
        """
        Apply the virtual circuit to the target deltas.

        Parameters
        ----------
        targets : list[str]
            The targets to apply the virtual circuit to.
        target_deltas : np.array
            The target deltas (rates) from shape/div category. Units m/s
        virtual_circuit : VirtualCircuit
            The virtual circuit object to be applied to the target deltas. Units A/m
        reshape : bool
            flag as to whether to reshape the currents to the order provided active_coils

        Returns
        -------
        np.array
            The coil current rates after being applied to the virtual circuit.
        """

        if targets == virtual_circuit.targets:
            # targets match - do nothing and use VC provided
            pass

        elif set(targets).issubset(set(virtual_circuit.targets)):
            # targets are a subset of the VC targets - recompute VC from sensitivity
            print(
                "targets are a subset of the VC targets - recomputing VC from sensitivity"
            )
            virtual_circuit = self.recompute_vc_from_sensitivity(
                virtual_circuit, targets
            )

        else:
            # targets are not a subset of the VC targets - raise error
            print("targets are not a subset of the VC targets - raising error")
            print("targets", targets)
            print("VC targets", virtual_circuit.targets)
            raise ValueError(
                "The virtual circuit targets do not match the targets requested. Check the VC and Target sequence"
            )

        delta_currents = virtual_circuit.VCs_matrix @ target_deltas

        if reshape == False:
            # leave currents with the coil order associated with the VC scheduler
            return delta_currents

        elif reshape == True:
            print("reshaping current to match active coils order")
            reshaped_currents = np.zeros(len(self.active_coils))
            for i, coil in enumerate(virtual_circuit.coils):
                # PCO patch until we sort this out
                if coil == "pc":
                    continue
                reshaped_currents[self.active_coil_order_dictionary[coil]] = (
                    1.0
                    * delta_currents[
                        i
                    ]  # multiply by 1.0 to get around the pointer/reference feature of python.
                )

            return reshaped_currents

    def apply_vc_2(
        self,
        time_stamp,
        targets: list[str],
        target_deltas: np.array,
    ):
        """Apply VC using the 'list of columns' format instead - for now just for testing

        Parameters
        ----------
        time_stamp : float
            time stamp to retrieve VC
        targets : list[str]
            list of targets to apply VC's to
        target_deltas : np.array
            array of the gained/blended target deltas (units m/s) to apply vc to
            length of delta array must match length of targets list.



        Returns
        -------
        currents_rates : np.array
            current rates (dI/dt)
        """
        # currents_rates = np.zeros(len(self.control_coils))
        currents_rates = np.zeros(
            len(self.feedback_target_scheduler.vc_scheduler.vc_coil_order)
        )
        for i, target in enumerate(targets):
            vc_col = self.feedback_target_scheduler.vc_scheduler.get_vc_2(
                time_stamp, target
            )
            currents_rates += target_deltas[i] * vc_col
        return currents_rates

    def control_shape_rates(
        self,
        time_stamp,
        target_obs,
        eq=None,
        profiles=None,
    ):
        """
        Compute shape rates given a set of target at a time provided.

        Parameters
        ----------
        time_stamp : float
            time stamp of the target to be retrieved
        target_obs : np.array
            array of measured/observed target values.
            Can come from file or from equilibrium in a simulation
        eq : object
            equilibrium object. optional - used to provide inputs to Emulators if vcs not from file
        profiles : object
            profiles object. optional - used to provide inputs to Emulators if vcs not from file


        Returns
        -------
        voltage_array : array
            feedback voltages
        """
        controlled_targets_all = self.feedback_target_scheduler.get_all_targets()
        controlled_targets_fb = (
            self.feedback_target_scheduler.get_fb_controlled_targets(
                time_stamp=time_stamp
            )
        )

        # get proportional gains
        prop_gains_arr = self.feedback_target_scheduler.get_gains(
            targets=controlled_targets_fb, time_stamp=time_stamp, K_type="Kprop"
        )

        # get integral gains
        int_gains_arr = self.feedback_target_scheduler.get_gains(
            targets=controlled_targets_fb, time_stamp=time_stamp, K_type="Kint"
        )
        # get reference desired target values for feedback control
        desired_target_values = self.feedback_target_scheduler.desired_target_values_fb(
            time_stamp
        )

        # get blends array
        blends_arr = self.feedback_target_scheduler.get_blends(
            time_stamp=time_stamp,
        )

        # get ff gradients
        ff_deltas = self.feedback_target_scheduler.feed_forward_gradient(
            time_stamp, targets=controlled_targets_all
        )

        # compute the proportional terms
        damp_factor = self.feedback_target_scheduler.get_damping(time_stamp=time_stamp)

        # raw deltas
        targ_deltas = self.calculate_target_deltas(
            targets_req=desired_target_values,
            targets_obs=target_obs,
        )

        # damped deltas (same as targ_deltas if no damping)
        damped_deltas = self.calculate_damped_target_deltas(
            target_deltas=targ_deltas,
            prev_err=self.prev_output,
            damp_factor=damp_factor,
        )

        # compute the shape rates - apply blends and gains
        shape_rate = self.calculate_blended_target_deltas(
            proportional_deltas=damped_deltas,
            targets_blends=blends_arr,
            prop_shape_gains=prop_gains_arr,
            ff_deltas=ff_deltas,
        )

        return shape_rate

    def control_current_rates(
        self, time_stamp, shape_rate, eq=None, profiles=None, reshape=False
    ):
        """
        Compute current rate (dI/dt) by applying VC to shape rate

        Parameters
        ----------
        time_stamp : float
            time to compute current rates at
        shape_rate : np.array
            shape rates (output of control_shape_rates) function above.
        eq : object
            equilibrium object. optional - used to provide inputs to Emulators if vcs not from file
        profiles : object
            profiles object. optional - used to provide inputs to Emulators if vcs not from file
        reshape : bool (optoinal)
            flag associaed with apply_vc. Reorder the output currents to active coils or not. Defaults to no reshaping.


        Returns
        -------
        current_rates : np.array
            current rate array (dI/dt), with coil order
        """

        controlled_targets_all = self.feedback_target_scheduler.get_all_targets()
        controlled_targets_fb = (
            self.feedback_target_scheduler.get_fb_controlled_targets(
                time_stamp=time_stamp
            )
        )

        # get the virtual circuit object
        virtual_circuit = self.feedback_target_scheduler.get_vc(
            eq=eq,
            profiles=profiles,
            time_stamp=time_stamp,
            coils=self.control_coils,
            targets=controlled_targets_all,
        )

        current_rate = self.apply_shape_vc(
            targets=controlled_targets_all,
            target_deltas=shape_rate,
            virtual_circuit=virtual_circuit,
            reshape=reshape,
        )

        return shape_rate, current_rate

    def feedback_voltage_from_currents(self, current_rate, inductance_matrix=None):
        """
        compute feedback voltage from current rate of change vector
        """
        if inductance_matrix is None:
            inductance_matrix = self.inductance_full

        return np.dot(inductance_matrix, current_rate)
