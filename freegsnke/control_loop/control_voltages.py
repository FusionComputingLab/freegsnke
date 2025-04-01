"""
Module for computing feeback control voltages from virtual circuits.

"""

import pickle
from copy import deepcopy

import numpy as np

from .. import machine_config
from .. import virtual_circuits as vc  # import the virtual circuit class
from ..equilibrium_update import Equilibrium
from ..nonlinear_solve import nl_solver
from ..virtual_circuits import VirtualCircuit

from .scheduler import TargetScheduler


class ControlVoltages:
    """
    Class to implement control voltages from virtual circuit, and given a set of observed target values, and a set of requested target values.

    Attributes :
    ------------
    eq : eq ojbect
    profiles : profiles ojbect
    stepping :  stepping ojbect (nl solver)
    active_coils : list of all active coils
    coils : list of coils to be used in controller
    coil_order_dictionary : dictionary mapping coil names to their order in the list
    inductance_full : full inductance matrix for all active coils
    VCH (virtual circuit handling class)
    target_sequencer (target sequencer class)


    Methods :
    get_inductance_reduced : retrieve inductance matrix from machine config, and select rows/columns
    calc_vc_from_eq : retrieve from file or compute a virtual circuit object from freegsnke or NN emulator.
    calculate_voltage_vc_feedback_proportional : compute feedback voltages from a virtual circuit object and a set of target shifts.
    feeback_voltage_control_timefunc : compute feedback voltages from a time provided by retrieving targets at given time from the target sequencer and computing with calculate_voltage_vc_feedback_proportional.
    """

    def __init__(
        self,
        eq: Equilibrium,
        profiles,
        stepping: nl_solver,
        target_sequencer: TargetScheduler,
        coils=None,
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
            target_sequencer : TargetScheduler object
                TargetScheduler object - contains targets and vc schedule for simulation.
            coils : list[str]   (optional)
                list of coil names, defaults to all active coils defined in get_active_coils.
        """
        # assign equi and profiles objects
        self.eq = eq
        self.profiles = profiles

        self.stepping = stepping
        self.n_active_coils = (
            self.stepping.n_active_coils
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
        print(self.active_coils)
        self.active_coils_reduced.remove("Solenoid")
        self.active_coils_reduced.remove("px")
        self.active_coils_reduced.remove("p6")
        print(self.active_coils_reduced)
        # .remove("px").remove("p6")

        # create a dictionary to map coil names to their order in the list
        coil_order_dictionary = {coil: i for i, coil in enumerate(self.active_coils)}
        self.coil_order_dictionary = coil_order_dictionary

        # assign coils to default or
        if coils is None:
            print("initialising with default all active coils")
            self.coils = deepcopy(self.active_coils_reduced)
        else:
            print("initialising with custom coils")
            self.coils = coils

        print("Default targets and current's initialised")
        print("all active", self.active_coils)
        print("control coilds", self.coils)

        # get inductance matrix (full with all active coils)
        # ??Machine config and inductance matrix will come from stepper function later??
        self.inductance_full = machine_config.coil_self_ind[
            : len(self.active_coils), : len(self.active_coils)
        ]
        # initialise a VC handling object
        self.VCH = vc.VirtualCircuitHandling()
        self.VCH.define_solver(self.stepping.NK, target_relative_tolerance=1e-7)

        # initialise a target sequencer object
        self.target_sequencer = target_sequencer
        print("target sequencer flag :", self.target_sequencer.vc_flag)
        if self.target_sequencer.vc_flag == ("Emulator" or "emu" or "emulator"):

            # pre run the emulators so future calls to calculate_voltage_vc_feedback_proportional are quicker
            start_targs = self.target_sequencer.target_schedule_dict[
                self.target_sequencer.target_schedule_times[0]
            ]
            self.target_sequencer.vc_scheduler.build_vc(
                eq=self.eq,
                profiles=self.profiles,
                targets=start_targs,
                coils=self.coils,
            )

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
            print("Inductance matrix for default of all active coils")
            coils = self.coils
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
            coils = self.active_coils_reduced

        print("building virtual circuit from freegsnke")
        self.VCH.calculate_VC(
            eq,
            profiles,
            coils=self.coils,
            targets=targets,
            targets_options=None,
        )

        # get the virtual circuit object
        virtual_circuit = self.VCH.latest_VC

        return virtual_circuit

    def calculate_target_deltas(
        self,
        eq,
        targets,
        gain_matrix,
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

        gain_matrix : np.array() (2d array))
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
            print(targets_obs)

            # check dimensions of target values
        assert len(targets_req) == len(
            targets_obs
        ), "The target required and observed vectors are not the same length"

        assert (
            gain_matrix.shape[0] == gain_matrix.shape[1] == len(targets_req)
        ), "The gain matrix is not the square or same size as the target vector"

        # shifts required
        target_deltas = targets_req - targets_obs
        gained_target_deltas = gain_matrix @ target_deltas
        print("target deltas", target_deltas)
        return gained_target_deltas

    def calculate_voltage_vc_feedback_proportional(
        self,
        eq,
        profiles,
        targets_req,
        targets_obs=None,
        targets=None,
        coils=None,
        virtual_circuit: VirtualCircuit = None,
        gain_matrix=None,
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

        gain_matrix : array
            diagonal square matrix of target gains. Defaults to identity matrix

        Returns
        -------
        feedback_voltages : array
            feedback voltages
        """

        # set default gain matrix if not provided
        if gain_matrix is None:
            gain_matrix = np.identity(len(targets_req))
            print("Gain matrix not provided, using identity matrix")
            print(gain_matrix)

        # # check coils and targets and update attributes accordingly
        # if targets is not None:
        #     self.targets = targets

        if coils is None:
            print("updating coils to", coils)
            coils = self.active_coils_reduced
        # build VC object if not provided
        if virtual_circuit is None:
            print("No VC object passed, building one with ")
            # check coils in virtual circuit match those in the tokamak
            print("target names provided ", targets)
            print("self coils", self.coils)
            virtual_circuit = self.calc_vc_from_eq(
                eq=eq, profiles=profiles, targets=targets, coils=coils
            )
        else:
            print("Virtual circuit provided")
            print("targets", virtual_circuit.targets)
            print("coils", virtual_circuit.coils)

        # assign virtual circuit attribute to class
        # self.virtual_circuit = virtual_circuit

        if not targets == virtual_circuit.targets:
            # raise error ? or recompute VC from sensitivity
            print(f"target names provided {targets}")
            print(f"virtual circuit targets {virtual_circuit.targets}")
            raise ValueError(
                "The virtual circuit targets do not match the targets requested. Check the VC and Target sequence"
            )
        # compute the shape target deltas
        gained_target_deltas = self.calculate_target_deltas(
            eq, targets, gain_matrix, targets_req, targets_obs
        )

        # do matrix multiplication VC @ G @ delta
        delta_currents = virtual_circuit.VCs_matrix @ gained_target_deltas
        print("delta currents", delta_currents)

        # option 1 reorder currents, fill in zeros and multiply by inductance matrix
        reshaped_currents = np.zeros(len(self.active_coils))
        for i, coil in enumerate(virtual_circuit.coils):
            # voltages_v1[i] = np.dot(inductance_matrix[self.coil_order_dictionary[coil],:], delta_currents[:])
            reshaped_currents[self.coil_order_dictionary[coil]] = delta_currents[i]
        print("reshaped currents")
        print(reshaped_currents)
        voltages_v1 = np.dot(self.inductance_full, reshaped_currents)

        print("------------- \n compuiting voltages \n -------------")
        print(
            "voltages v1 : reorder currents, fill in zeros and multiply by full active coil inductance matrix"
        )
        print("voltages v1 : shape", voltages_v1.shape)
        print(voltages_v1)

        # option 2 reshape inductance matrix, multiply by currents and then fill in zeros
        print("doing option 2")
        print("delta currents", delta_currents)
        inductance_matrix_controlled = self.get_inductance_reduced(
            coils=virtual_circuit.coils
        )
        voltages_v2_controlled = np.dot(inductance_matrix_controlled, delta_currents)
        # fill in zeros
        voltages_v2 = np.zeros(len(self.active_coils))
        for i, coil in enumerate(virtual_circuit.coils):
            voltages_v2[self.coil_order_dictionary[coil]] = voltages_v2_controlled[i]

        print(
            "voltages v2 : reshaped inductance matrix, then fill in zeros in voltage vector"
        )
        print("voltages v2 : shape", voltages_v2.shape)
        print(voltages_v2)

        self.feedback_voltages_v1 = voltages_v1
        self.feedback_voltages_v2 = voltages_v2

        return voltages_v1, voltages_v2

    def feeback_voltage_control_timefunc(
        self,
        time_stamp,
        eq,
        profiles,
        gain_matrix=None,
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

        gain_matrix : array
            diagonal square matrix of target gains. Defaults to identity matrix

        Returns
        -------
        voltage_array : array
            feedback voltages
        """
        controlled_targets = self.target_sequencer.retrieve_controlled_targets(
            time_stamp
        )
        print("controlled targets are ", controlled_targets)
        # get the virtual circuit object
        virtual_circuit = self.target_sequencer.get_vc(
            eq=eq, profiles=profiles, time_stamp=time_stamp, coils=self.coils
        )

        desired_target_values = self.target_sequencer.desired_target_values(time_stamp)

        # compute the proportional voltages
        voltages_v1, voltages_v2 = self.calculate_voltage_vc_feedback_proportional(
            eq,
            profiles,
            targets=controlled_targets,
            targets_req=desired_target_values,
            targets_obs=None,
            virtual_circuit=virtual_circuit,
            gain_matrix=gain_matrix,
        )

        # TO DO
        # add in other voltage computations (integral, ip etc here)

        return voltages_v1, voltages_v2


### TESTING ###
# if __name__ == "__main__":

#     pass
