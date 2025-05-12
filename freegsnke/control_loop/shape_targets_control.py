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
from .target_scheduler import TargetScheduler


class ShapeController:
    """
    Class to implement control voltages from virtual circuit, and given a set of observed target values, and a set of requested target values.

    Attributes :
    ------------
    eq : eq object
    profiles : profiles object
    stepping :  stepping object (nl solver)
    active_coils : list of all active coils
    coils : list of coils to be used in controller
    coil_order_dictionary : dictionary mapping coil names to their order in the list
    inductance_full : full inductance matrix for all active coils
    VCH (virtual circuit handling class)
    feedback_target_scheduler (target scheduler class)


    Methods :
    get_inductance_reduced : retrieve inductance matrix from machine config, and select rows/columns
    calc_vc_from_eq : retrieve from file or compute a virtual circuit object from freegsnke or NN emulator.
    calculate_blended_feedback_current_rate_vc_proportional : compute feedback voltages from a virtual circuit object and a set of target shifts.
    feedback_current_rate_timefunc : compute feedback voltages from a time provided by retrieving targets at given time from the target waveformr and computing with calculate_blended_feedback_current_rate_vc_proportional.
    """

    def __init__(
        self,
        eq: Equilibrium,
        profiles,
        stepping: nl_solver,
        feedback_target_scheduler: ShapeTargetScheduler,
        feedforward_target_scheduler: TargetScheduler = None,
        coils=None,
        inductance_matrix=None,
        coil_resist=None,
    ):
        """
        Initialize the control voltages class

        Parameters
        ----------
            eq : equilibrium object
                equilibrium object
            profiles : list of profiles
                list of profiles
            stepping : Non Linear Solver object
                Non Linear Solver object
            feedback_target_scheduler : TargetScheduler object
                TargetScheduler object - contains targets and vc schedule for simulation.
            coils : list[str]   (optional)
                list of coil names, defaults to all active coils defined in get_active_coils.
            inductance_matrix : np.array (optional)
                inductance matrix, defaults to machine inductance matrix.
        """
        # assign equi and profiles objects
        self.eq = eq
        self.profiles = profiles

        # self.stepping = stepping
        self.n_active_coils = (
            stepping.n_active_coils
        )  # could also be eq.tokamak.n_active_coils
        print("number active coils", self.n_active_coils)
        # initialise targets with defaults or lists given
        # if targets is None:
        #     targets = ["R_in", "R_out", "Rx_lower", "Rs_lower_outer"]
        #     self.targets = targets
        # else:
        #     self.targets = targets

        # set coil lists and dictionary for all active coils
        self.active_coils = self.eq.tokamak.coils_list[: self.n_active_coils]

        self.active_coils_reduced = deepcopy(self.active_coils)

        # .remove("px").remove("p6")

        # create a dictionary to map coil names to their order in the list
        coil_order_dictionary = {coil: i for i, coil in enumerate(self.active_coils)}
        self.coil_order_dictionary = coil_order_dictionary

        # assign coils to default or
        if coils is None:
            print("initialising with default all active coils")
            self.control_coils = deepcopy(self.active_coils_reduced)
        else:
            print("initialising with custom coils")
            self.control_coils = coils

        print("Default targets and current's initialised")
        print("all active", self.active_coils)
        print("control coils", self.control_coils)

        # get inductance matrix (full with all active coils)
        # ??Machine config and inductance matrix will come from stepping function later??
        if inductance_matrix is None:
            self.inductance_full = machine_config.coil_self_ind[
                : len(self.active_coils), : len(self.active_coils)
            ]
            print("Using default inductance matrix from machine config")
            # print("inductance matrix", self.inductance_full)
        else:
            self.inductance_full = inductance_matrix

        if coil_resist is None:
            self.coil_resist = machine_config.coil_resist[: len(self.active_coils)]
            print(
                "No coil resistances provided, using default coil resistances from machine config"
            )
            # print("coil resistances", self.coil_resist)
        else:
            self.coil_resist = coil_resist
        # initialise a VC handling object
        self.VCH = vc.VirtualCircuitHandling()
        self.VCH.define_solver(stepping.NK, target_relative_tolerance=1e-7)

        # initialise a target scheduler object
        self.feedback_target_scheduler = feedback_target_scheduler
        print("target scheduler flag :", self.feedback_target_scheduler.vc_flag)
        if self.feedback_target_scheduler.vc_flag == (
            "Emulator" or "emu" or "emulator"
        ):

            # pre run the emulators so future calls to calculate_blended_feedback_current_rate_vc_proportional are quicker
            start_targs = self.feedback_target_scheduler.target_schedule_dict[
                list(self.feedback_target_scheduler.target_schedule_dict.keys())[0]
            ]
            self.feedback_target_scheduler.vc_scheduler.build_vc(
                eq=self.eq,
                profiles=self.profiles,
                targets=start_targs,
                coils=self.control_coils,
            )

        if feedforward_target_scheduler is None:
            print("No feedforward scheduler provided. will copy the feedback scheduler")
            self.feedforward_target_scheduler = deepcopy(self.feedback_target_scheduler)
        else:
            self.feedforward_target_scheduler = feedforward_target_scheduler

        # create blend dict from schedule OR load from file
        all_targs = sorted(
            set(self.feedback_target_scheduler.target_waveform_dict.keys())
        )
        print("all targets", all_targs)

        print("Shape controller initialised")

    ### OLD blends function
    # defget_shape_blends(self, targets, time_stamp):
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

    def get_inductance_reduced(self, coils=None):
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
        mask = [self.coil_order_dictionary[coil] for coil in coils]
        print("coil ordering mask ", mask)
        inductance_reduced = self.inductance_full[np.ix_(mask, mask)]
        self.inductance_reduced = inductance_reduced

        return inductance_reduced

    ## this function will be replaced by instance of build virtual circuit class.
    def calc_vc_from_eq(self, targets, eq=None, profiles=None, coils=None):
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
        if eq is None:
            eq = self.eq
        if profiles is None:
            profiles = self.profiles

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

    def calculate_gained_target_deltas(
        self,
        eq,
        targets,
        shape_gain_matrix,
        targets_req,
        targets_obs=None,
    ):
        """Calculate the target deltas between the required and observed targets

        Parameters
        ----------
        eq : object
            equilibrium object

        profiles : object
            profiles object

        targets : list
            list of target names

        shape_gain_matrix : np.array() (2d array))
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

        if targets_obs is None:
            # get the targets from the equilibrium
            print("Observed targets not provided, calculating from equilibrium")
            _, targets_obs = self.VCH.calculate_targets(eq, targets)
            print("observed targets", targets_obs)

            # check dimensions of target values
        assert len(targets_req) == len(
            targets_obs
        ), "The target required and observed vectors are not the same length"

        assert (
            shape_gain_matrix.shape[0] == shape_gain_matrix.shape[1] == len(targets_req)
        ), "The gain matrix is not the square or same size as the target vector"
        print("shape gain matrix", shape_gain_matrix)
        # shifts required
        target_deltas = targets_req - targets_obs
        gained_target_deltas = shape_gain_matrix @ target_deltas
        print("requested targets", targets_req)
        print("required target deltas", target_deltas)
        print("gained target deltas", gained_target_deltas)
        return gained_target_deltas

    # THIS IS REDUNDANT NOW I THINK
    # def blended_ff_fb_targs(self, time_stamp, eq):
    #     """
    #     Combine feedback and feed forward targets.

    #     Parameters
    #     ----------
    #     time_stamp : float
    #         time stamp of the target to be retrieved
    #     Returns
    #     -------
    #     combined_targs : list[str]
    #         list of target names
    #     """
    #     # get set of targets being controlled at this time
    #     controlled_targets = self.feedback_target_scheduler.retrieve_controlled_targets(
    #         time_stamp
    #     )
    #     feed_forward_targets = list(
    #         self.feedforward_target_scheduler.target_waveform_dict.keys()
    #     )  # these hard coded, or hard coded in init? or just get all targets from scheduler?
    #     all_targs = sorted(set(controlled_targets + feed_forward_targets))
    #     # ?? control targs should be subset of feedforward targets?
    #     # dictionary of blend vals basaed on if target in controlled or not. ()
    #     blend_vals = {
    #         targ: 1 if targ in controlled_targets else 0 for targ in all_targs
    #     }  # blends for controlled targets 1

    #     ff_gradients = self.feedforward_target_scheduler.feed_forward_gradient(
    #         time_stamp=time_stamp, targets=all_targs
    #     )
    #     ff_grad_dict = dict(zip(all_targs, ff_gradients))

    #     # compute gained control targets
    #     shape_gain_matrix = self.feedback_target_scheduler.vc_scheduler.retrieve_gains(
    #         controlled_targets, time_stamp
    #     )
    #     targets_req = self.feedback_target_scheduler.desired_target_values(time_stamp)
    #     gained_control_targs = self.calculate_gained_target_deltas(
    #         eq,
    #         targets=controlled_targets,
    #         shape_gain_matrix=shape_gain_matrix,
    #         targets_req=targets_req,
    #         targets_obs=None,
    #     )
    #     gained_targs_dict = dict(zip(controlled_targets, gained_control_targs))

    #     blended_dict = {}
    #     for targ in all_targs:
    #         blended_dict[targ] = (
    #             ff_grad_dict[targ] + blend_vals[targ] * gained_targs_dict[targ]
    #         )

    #     # convert to array, ordered according to all_targs ?? maybe this need changing/fixing order?
    #     blended_array = np.array([blended_dict[targ] for targ in all_targs])
    #     return blended_array

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
        targ_values = np.array(
            [
                virtual_circuit.targets_val[i]
                for i in np.array([vc_targ_order_dict[targ] for targ in targets])
            ]
        )

        virtualcircuit = vc.VirtualCircuit(
            name="recomputed_vc",
            eq=virtual_circuit.eq,
            profiles=virtual_circuit.profiles,
            shape_matrix=sens_reduced,
            VCs_matrix=vc_mat_reduced,
            targets=targs_reduced,
            targets_val=targ_values,
            non_standard_targets=None,
            targets_options=None,
            coils=virtual_circuit.coils,
        )

        return virtualcircuit

    def calculate_blended_feedback_current_rate_vc_proportional(
        self,
        eq,
        profiles,
        targets_req,
        targets_blends,
        ff_deltas,
        targets_obs=None,
        targets=None,
        coils=None,
        virtual_circuit: VirtualCircuit = None,
        shape_gain_matrix=None,
    ):
        """
        Compute current given a set of target value shifts and vc matrix, at a given time.

        Assigns attributes in place for feedback voltages, and returns them.

        Parameters
        ----------
        eq : object
            equilibrium object

        profiles : object
            profiles object

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

        shape_gain_matrix : array
            diagonal square matrix of target gains. Defaults to identity matrix

        Returns
        -------
        feedback_voltages : array
            feedback voltages
        """

        # # check coils and targets and update attributes accordingly
        # if targets is not None:
        #     self.targets = targets

        if coils is None:
            print("coils being set to default active coils reduced")
            coils = self.active_coils_reduced
        # build VC object if not provided
        if virtual_circuit is None:
            print("No VC object passed, building one with ")
            # check coils in virtual circuit match those in the tokamak
            print("target names provided ", targets)
            print("control coils", self.control_coils)
            virtual_circuit = self.calc_vc_from_eq(
                eq=eq, profiles=profiles, targets=targets, coils=coils
            )
        else:
            print("Virtual circuit provided with :")
            print("targets", virtual_circuit.targets)
            print("coils", virtual_circuit.coils)

        # assign virtual circuit attribute to class
        # self.virtual_circuit = virtual_circuit

        # if not targets == virtual_circuit.targets:
        #     # raise error ? or recompute VC from sensitivity
        #     print(f"target names provided {targets}")
        #     print(f"virtual circuit targets {virtual_circuit.targets}")
        #     raise ValueError(
        #         "The virtual circuit targets do not match the targets requested. Check the VC and Target sequence"
        #     )

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

        # compute the shape target deltas
        gained_target_deltas = self.calculate_gained_target_deltas(
            eq, targets, shape_gain_matrix, targets_req, targets_obs
        )

        blended_target_deltas = (
            targets_blends * gained_target_deltas + (1 - targets_blends) * ff_deltas
        )
        # do matrix multiplication VC @ G @ delta
        # delta_currents = virtual_circuit.VCs_matrix @ gained_target_deltas
        delta_currents = virtual_circuit.VCs_matrix @ blended_target_deltas
        print("shape current deltas", delta_currents)

        # option 1 reorder currents, fill in zeros and multiply by inductance matrix
        reshaped_currents = np.zeros(len(self.active_coils))
        for i, coil in enumerate(virtual_circuit.coils):
            # voltages_v1[i] = np.dot(inductance_matrix[self.coil_order_dictionary[coil],:], delta_currents[:])
            reshaped_currents[self.coil_order_dictionary[coil]] = delta_currents[i]
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
        # inductance_matrix_controlled = self.get_inductance_reduced(
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

    def feedback_current_rate_timefunc(
        self,
        time_stamp,
        eq,
        profiles,
        target_obs=None,
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

        # shape_gain_matrix : array
        #     diagonal square matrix of target gains. Defaults to identity matrix

        Returns
        -------
        voltage_array : array
            feedback voltages
        """
        controlled_targets = self.feedback_target_scheduler.retrieve_controlled_targets(
            time_stamp
        )
        print("controlled targets are ", controlled_targets)
        if controlled_targets == []:
            print("no controlled targets at time ", time_stamp)
            return np.zeros(len(self.active_coils))

        # if shape_gain_matrix is None:
        #     if self.feedback_target_scheduler.vc_flag == "file":
        #         shape_gain_matrix = (
        #             self.feedback_target_scheduler.vc_scheduler.retrieve_gains(
        #                 targets=controlled_targets, time_stamp=time_stamp
        #             )
        #         )
        #     elif (
        #         self.feedback_target_scheduler.vc_flag == "emulator"
        #         or "emu"
        #         or "Emulator"
        #     ):
        #         # set default gains for emulators - this may want to be updated in future
        #         print("using emulators - gains default to identity matrix ")
        #         shape_gain_matrix = np.identity(len(controlled_targets))
        #         # or gain matrix is the one provided???

        shape_gain_matrix = self.feedback_target_scheduler.get_shape_gains(
            targets=controlled_targets, time_stamp=time_stamp
        )
        print("shape target gains", shape_gain_matrix)
        # get the virtual circuit object
        virtual_circuit = self.feedback_target_scheduler.get_vc(
            eq=eq, profiles=profiles, time_stamp=time_stamp, coils=self.control_coils
        )

        desired_target_values = self.feedback_target_scheduler.desired_target_values(
            time_stamp
        )

        fb_blends_arr = self.feedback_target_scheduler.get_shape_blends(
            controlled_targets, time_stamp
        )
        print("fb blends", fb_blends_arr)
        ff_deltas = self.feedforward_target_scheduler.feed_forward_gradient(
            time_stamp, targets=controlled_targets
        )
        print("ff deltas", ff_deltas)

        # compute the proportional voltages
        current_rate = self.calculate_blended_feedback_current_rate_vc_proportional(
            eq,
            profiles,
            targets=controlled_targets,
            targets_req=desired_target_values,
            targets_blends=fb_blends_arr,
            ff_deltas=ff_deltas,
            targets_obs=target_obs,
            virtual_circuit=virtual_circuit,
            shape_gain_matrix=shape_gain_matrix,
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
