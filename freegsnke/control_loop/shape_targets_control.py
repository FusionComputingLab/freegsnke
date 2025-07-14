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
    feedback_current_rate_timefunc : compute feedback voltages from a time provided by retrieving targets at given time from the target waveformr and computing with calculate_blended_target_deltas.
    """

    def __init__(
        self,
        feedback_target_scheduler: ShapeTargetScheduler,
        active_coils: list[str],
        control_coils: list[str],
        machine_parameters,
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
            coils : list[str]
                list of coil names to be used for shape control, defaults to all active coils defined in get_active_coils.
            machine_parameters : dict
                dictionary containing full inductance matrix and coil resistances
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

    ### OLD blends function
    # def get_shape_blends(self, targets, time_stamp):
    #     """
    #     Retrieves the blends for the target at time_stamp

    #     Parameters
    #     ----------
    #     time_stamp : float
    #         time stamp of the target to be retrieved

    #     Returns
    #     -------
    #     blends : dict
    #         dictionary of blends for the target at time_stamp
    #     """
    #     blend_arr = []
    #     # replace this...
    #     for target in targets:
    #         interpolation = np.interp(
    #             time_stamp,
    #             self.feedback_target_scheduler.shape_blends[target]["times"],
    #             self.feedback_target_scheduler.shape_blends[target]["vals"],
    #         )
    #         blend_arr.append(interpolation)
    #     print(f"blends for {targets} at time {time_stamp}: {blend_arr}")
    #     return np.array(blend_arr)

    def reshape_inductance(self, coils=None):
        """
        Select appropriate inductance rows and columns from inductance matrix, given set of coils in the VC.

        parameters
        ----------
        coils : list[str]
            list of coil names

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
        """reorder coil resistances to match coil order"""
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

    ### OLD REDUNDANT (i think) - doesn't include damping
    def calculate_gained_target_deltas(
        self,
        targets,
        targets_req,
        targets_obs,
        shape_prop_gain_matrix,
    ):
        """Calculate the target deltas between the required and observed targets

        Parameters
        ----------
        targets : list
            list of target names
        shape_prop_gain_matrix : np.array() (2d array))
            diagonal square matrix of target gains.
        targets_req : array
            array of required/desired target values for each shape target
        targets_obs : array, Optional
            array of target values for each shape target. If not provided, the observed targets are calculated from the equilibrium.


        Returns
        -------
        target_deltas : array
            array of target deltas
        """

        # check dimensions of target values
        assert len(targets_req) == len(
            targets_obs
        ), "The target required and observed vectors are not the same length"

        assert (
            shape_prop_gain_matrix.shape[0]
            == shape_prop_gain_matrix.shape[1]
            == len(targets_req)
        ), "The gain matrix is not the square or same size as the target vector"
        # print("shape gain matrix", shape_prop_gain_matrix)

        # shifts required
        target_deltas = targets_req - targets_obs
        gained_target_deltas = shape_prop_gain_matrix @ target_deltas
        # print("targets names", targets)
        # print("required target deltas", target_deltas)
        # print("gained target deltas", gained_target_deltas)
        return gained_target_deltas, target_deltas

    def calculate_proportional_target_deltas(
        self,
        targets,
        targets_req,
        targets_obs,
        prev_err,
        shape_prop_gain_matrix,
        damp_factor=1,
    ):
        """Calculate the target deltas between the required and observed targets

        Parameters
        ----------
        targets : list
            list of target names
        shape_prop_gain_matrix : np.array() (2d array))
            diagonal square matrix of target gains.
        targets_req : array
            array of required/desired target values for each shape target
        targets_obs : array, Optional
            array of target values for each shape target. If not provided, the observed targets are calculated from the equilibrium.


        Returns
        -------
        target_deltas : array
            array of target deltas
        """

        # check dimensions of target values
        assert len(targets_req) == len(
            targets_obs
        ), "The target required and observed vectors are not the same length"

        assert (
            shape_prop_gain_matrix.shape[0]
            == shape_prop_gain_matrix.shape[1]
            == len(targets_req)
        ), "The gain matrix is not the square or same size as the target vector"
        # print("shape gain matrix", shape_prop_gain_matrix)

        # shifts required
        target_deltas = targets_req - targets_obs
        damp_deltas = target_deltas / damp_factor

        # total damped target error
        targ_err = (1 - 1 / damp_factor) * prev_err + damp_deltas

        gained_target_deltas = shape_prop_gain_matrix @ targ_err
        # print("targets names", targets)
        # print("required target deltas", target_deltas)
        # print("gained target deltas", gained_target_deltas)
        return (
            gained_target_deltas,
            targ_err,
            target_deltas,
        )

    def calculate_integral_deltas(self, deltas, pi_state, dt, K_int_matrix, blends):
        """compute updated integral error term and PI state

        Parameters
        ----------
        deltas  : np.array
            array of damped deltas (targ_err from calc proportioal targets method above)
        pi_state : np.array
            PI state at previous time step
        dt : float
            time interval

        Returns
        -------
        pi_state : np.array
            updated PI state
        delta_int : np.array
            integral term

        """
        # ??? add in a pi state attribute to class ??
        x_int = pi_state + 0.5 * K_int_matrix @ deltas * dt
        pi_state_new = pi_state + blends * K_int_matrix @ deltas * dt
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
        targets,
        proportional_deltas,
        targets_blends,
        ff_deltas,
        shape_prop_gain_matrix,
        x_int=None,
        coils=None,
    ):
        """
        Compute current given a set of target value shifts and vc matrix, at a given time.

        Assigns attributes in place for feedback voltages, and returns them.

        Parameters
        ----------
        targets_req : array
            array function of target values for each control voltage at a given time

        targets_obs : array
            array function of target values for each control voltage at given  time
            Defaults to None, in which case the targets are computed from the equilibrium.

        targets : list
            list of target names. Defaults to None, in which case the targets taken to be all active.

        coils : list
            list of coil names. Defaults to None, in which case the coils taken to be all active.

        virtual_circuit : object
            virtual circuit object. Defaults to None, in which case the virtual circuit is computed from the equilibrium.
            with default currents of the Tokamak minus p6, and solenoid (these are determined differently)

        shape_prop_gain_matrix : array
            diagonal square matrix of target gains. Defaults to identity matrix

        Returns
        -------
        feedback_voltages : array
            feedback voltages
        """
        if coils is None:
            print("coils being set to default active coils reduced")
            coils = self.control_coils

        # compute the shape target deltas
        # gained_target_deltas, _ = self.calculate_gained_target_deltas(
        #     targets=targets,
        #     targets_req=targets_req,
        #     targets_obs=targets_obs,
        #     shape_prop_gain_matrix=shape_prop_gain_matrix,
        # )
        gained_target_deltas = shape_prop_gain_matrix @ proportional_deltas

        blended_target_deltas = (
            targets_blends * gained_target_deltas + (1 - targets_blends) * ff_deltas
        )

        return blended_target_deltas

    def apply_shape_vc(
        self,
        targets: list[str],
        target_deltas: np.array,
        virtual_circuit: VirtualCircuit,
    ):
        """
        Apply the virtual circuit to the target deltas.

        Parameters
        ----------
        targets : list[str]
            The targets to apply the virtual circuit to.
        target_deltas : np.array
            The target deltas (rates) from shape/div category
        virtual_circuit : VirtualCircuit
            The virtual circuit object to be applied to the target deltas.

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
            # print("VC targets now ", virtual_circuit.targets)
            # print("coils now ", virtual_circuit.coils)

        else:
            # targets are not a subset of the VC targets - raise error
            print("targets are not a subset of the VC targets - raising error")
            print("targets", targets)
            print("VC targets", virtual_circuit.targets)
            raise ValueError(
                "The virtual circuit targets do not match the targets requested. Check the VC and Target sequence"
            )
        # do matrix multiplication VC @ G @ delta
        # delta_currents = virtual_circuit.VCs_matrix @ gained_target_deltas
        delta_currents = virtual_circuit.VCs_matrix @ target_deltas
        print("shape current deltas", delta_currents)

        # option 1 reorder currents, fill in zeros and multiply by inductance matrix
        reshaped_currents = np.zeros(len(self.active_coils))
        for i, coil in enumerate(virtual_circuit.coils):
            # voltages_v1[i] = np.dot(inductance_matrix[self.coil_order_dictionary[coil],:], delta_currents[:])
            # PCO patch until we sort this out
            if coil == "pc":
                continue
            reshaped_currents[self.active_coil_order_dictionary[coil]] = delta_currents[
                i
            ]
        print("reshaped currents")
        print(reshaped_currents)

        return reshaped_currents
        # voltages_v1 = np.dot(self.inductance_full, reshaped_currents)

        # print("------------- \n computing voltages \n -------------")
        # print(
        #     "voltages v1 : reorder currents, fill in zeros and multiply by full active coil inductance matrix"
        # )
        # print("voltages v1 : shape", voltages_v1.shape)
        # print(voltages_v1)

        # # option 2 reshape inductance matrix, multiply by currents and then fill in zeros
        # print("doing option 2")
        # print("delta currents", delta_currents)
        # inductance_matrix_controlled = self.reshape_inductance(
        #     coils=virtual_circuit.coils
        # )
        # voltages_v2_controlled = np.dot(inductance_matrix_controlled, delta_currents)
        # # fill in zeros
        # voltages_v2 = np.zeros(len(self.active_coils))
        # for i, coil in enumerate(virtual_circuit.coils):
        #     voltages_v2[self.coil_order_dictionary[coil]] = voltages_v2_controlled[i]

        # print(
        #     "voltages v2 : reshaped inductance matrix, then fill in zeros in voltage vector"
        # )
        # print("voltages v2 : shape", voltages_v2.shape)
        # print(voltages_v2)

        # if we want to keep the latest voltages
        # self.feedback_voltages_v1 = voltages_v1
        # self.feedback_voltages_v2 = voltages_v2

        # return voltages_v1, voltages_v2

    def apply_vc_2(
        self,
        time_stamp,
        targets: list[str],
        target_deltas: np.array,
    ):
        """Apply VC using the 'list of columns' format instead - for now just for testing"""
        # currents_rates = np.zeros(len(self.control_coils))
        currents_rates = np.zeros(
            len(self.feedback_target_scheduler.vc_scheduler.vc_coil_order)
        )
        for i, target in enumerate(targets):
            vc_col = self.feedback_target_scheduler.vc_scheduler.get_vc_2(
                time_stamp, target
            )
            print(np.shape(vc_col))
            print(np.shape(currents_rates))
            currents_rates += target_deltas[i] * vc_col
        return currents_rates

    def feedback_current_rate_timefunc(
        self,
        time_stamp,
        target_obs,
        eq=None,
        profiles=None,
    ):
        """
        Compute current given a set of target value shifts and vc matrix, at a time provided.

        Parameters
        ----------
        time_stamp : float
            time stamp of the target to be retrieved
        eq : object
            equilibrium object

        profiles : object
            profiles object

        # shape_prop_gain_matrix : array
        #     diagonal square matrix of target gains. Defaults to identity matrix

        Returns
        -------
        voltage_array : array
            feedback voltages
        """
        controlled_targets = self.feedback_target_scheduler.get_fb_controlled_targets(
            time_stamp
        )
        print("controlled targets are ", controlled_targets)
        if controlled_targets == []:
            print("no controlled targets at time ", time_stamp)
            return np.zeros(len(self.active_coils))

        gains_arr, shape_prop_gain_matrix = self.feedback_target_scheduler.get_gains(
            targets=controlled_targets, time_stamp=time_stamp, K_type="Kprop"
        )
        print("shape target gains", shape_prop_gain_matrix)
        # get the virtual circuit object
        virtual_circuit = self.feedback_target_scheduler.get_vc(
            eq=eq,
            profiles=profiles,
            time_stamp=time_stamp,
            coils=self.control_coils,
            targets=controlled_targets,
        )

        desired_target_values = self.feedback_target_scheduler.desired_target_values_fb(
            time_stamp
        )

        fb_blends_arr = self.feedback_target_scheduler.get_blends(
            controlled_targets, time_stamp
        )
        print("fb blends", fb_blends_arr)
        ff_deltas = self.feedback_target_scheduler.feed_forward_gradient(
            time_stamp, targets=controlled_targets
        )
        print("ff deltas", ff_deltas)

        # compute the proportional voltages

        gained, prop_deltas, deltas = self.calculate_proportional_target_deltas(
            targets=controlled_targets,
            targets_req=desired_target_values,
            targets_obs=target_obs,
            prev_err=0,
            shape_prop_gain_matrix=shape_prop_gain_matrix,
            damp_factor=1,
        )

        shape_rate = self.calculate_blended_target_deltas(
            targets=controlled_targets,
            proportional_deltas=prop_deltas,
            targets_blends=fb_blends_arr,
            ff_deltas=ff_deltas,
            shape_prop_gain_matrix=shape_prop_gain_matrix,
        )

        current_rate = self.apply_shape_vc(
            targets=controlled_targets,
            target_deltas=shape_rate,
            virtual_circuit=virtual_circuit,
        )

        return current_rate

    def feedback_voltage_from_currents(self, current_rate, inductance_matrix=None):
        """
        compute feedback voltage from current rate of change vector
        """
        if inductance_matrix is None:
            inductance_matrix = self.inductance_full

        return np.dot(inductance_matrix, current_rate)


### TESTING ###
# if __name__ == "__main__":

#     pass
