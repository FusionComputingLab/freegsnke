"""
Module to obtain control voltages from virtual circuits.

"""

import numpy as np
from copy import deepcopy

from . import virtual_circuits as vc  # import the virtual circuit class
from . import machine_config
from freegsnke import GSstaticsolver



class ControlVoltages:
    """
    Class to implement control voltages from virtual circuit, and given a set of requested target shifts.

    Attributes :
    ???

    Methods :
    ???

    """

    def __init__(self, targets = None ):
        """Initialize the control voltages class"""
        if targets is None:
            targets = ["R_in","R_out","Rx_lower"]
        self.targets = targets 

    def assign_eqi():

        pass

    def get_active_coils(self, eq):
        """set default coils to be used and set the order according to that in the tokamak description
        get all active ones
        assigne reduced set of coils without solenoid and p6 (these voltages will be set via  different method)
        """

        active_coils = eq.tokamak.coils_list[:12]
        active_coils_reduced = [
            coil for coil in active_coils if coil not in {"Solenoid", "p6"}
        ]

        self.active_coils_all = active_coils
        self.active_coils_reduced = active_coils_reduced

        print("all active coils", self.active_coils_all)
        print("redduced set of active coils", self.active_coils_reduced)

        # create a dictionary to map coil names to their order in the list
        order_dictionary = {coil: i for i, coil in enumerate(active_coils)}
        self.order_dictionary = order_dictionary
        print("order dictionary", self.order_dictionary)

        return active_coils, active_coils_reduced, order_dictionary

    def get_inductance(self, coils=None):
        """retrieve inductance matrix from machine config

        machine_config.coil_self_ind . only want active part
        coils : list of coils to use
        """
        if coils is None:
            # use default of all acitve coils from tokamak
            coils_dict = machine_config.coils_dict
            inductance_active = machine_config.coil_self_ind[:12, :12]

        else:  # use coils provided and select apropriate part of inductance matrix
            mask = [self.order_dictionary[coil] for coil in coils]
            inductance_active = machine_config.coil_self_ind[np.ix_(mask, mask)]
        self.inductance_matrix = inductance_active

        return inductance_active

    def get_vc(self, eq, profiles, targets=None, origin=None):
        """
        Get a virtual circuit object from freegsnke or from a file or NN emulator.

        parameters
        ----------
        eq : object
            equilibrium object

        profiles : object
            profiles object

        origin : str
            origin of the virtual circuit object. Defaults to None, in which case the virtual circuit is computed from the equilibrium.
            options are "file" or "emulator". These methods are not yet implemented.


        returns
        -------
        virtual_circuit : object
            virtual circuit object
        """
        solver = GSstaticsolver.NKGSsolver(eq)  


        # assert hasattr(self,active_coils_reduced) , "coils haven't been set yet"
        _,coils_active, order_dict = self.get_active_coils(eq)
        if origin is None:
            # create virtual circuit object using freegsnke
            print("building virtual circuit from freegsnke")
            vch = vc.VirtualCircuitHandling()
            vch.define_solver(solver)
            vch.calculate_VC(
                eq, profiles, coils=coils_active, targets=targets, targets_options= None ,
            )

            # get the virtual circuit object
            virtual_circuit = vch.latest_VC
        elif origin == "emulator":
            # create virtual circuit object using nn emulator
            print("building VC from emulator")
        elif origin == "file":
            # create virtual circuit object using file
            print("building VC from file")
            pass
        return virtual_circuit

    def calculate_feedback_voltage_vector(
        self,
        eq,
        profiles,
        targets_req,
        targets_obs=None,
        target_names=None,
        virtual_circuit=None,
        gain_matrix=None,
    ):
        """
        Compute current given a set of target value shifs and vc matrix, at a given time.

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


        targets_names : list
            list of target names. Defaults to None, in which case the targets taken to be all active.

        virtual_circuit : object
            virtual circuit object. Defaults to None, in which case the virtual circuit is computed from the equilibrium.
            with default currents of the Tokamak minus p6, and solenoid (these are determined differently)

        gain_matrix : array
            diagonal square matrix of target gains. Defaults to identity matrix

        Returns
        -------
        feedback_current : array
        """
        self.targets = target_names 
        # get active coils and ordering dictionary
        _,active_coils, order_dict = self.get_active_coils(eq)

        # check dimensions
        assert len(targets_req) == len(
            targets_obs
        ), "The target required and observed vectors are not the same length"

        # set default gain matrix if not provided
        if gain_matrix is None:
            gain_matrix = np.identity(len(targets_req))
            print("gain matrix not provided, using identity matrix")
            print(gain_matrix)

        assert gain_matrix.shape[0] == len(
            targets_req
        ), "The gain matrix is not the same length as the target vector"

        # build VC object if not provided
        if virtual_circuit is None:
            print("building virtual circuit")
            virtual_circuit = self.get_vc(eq=eq, profiles=profiles, targets=self.targets,)

        self.virtual_circuit = virtual_circuit

        # check coils in virtual circuit match those in the tokamak
        assert (
            target_names == virtual_circuit.targets
        ), "The virtual circuit targets do not match the targets in the tokamak"

        if targets_obs is None:
            # get the targets from the equilibrium
            targets_obs = vc.VirtualCircuitHandling().calculate_targets(
                eq, self.targets
            )

        # shifts required
        target_deltas = targets_req - targets_obs
        print("target deltas", target_deltas)


        # do matrix multiplication VC @ G @ delta
        delta_currents = virtual_circuit.VCs_matrix @ gain_matrix @ target_deltas

        # bulid inductance matrix
        inductance_matrix = self.get_inductance(coils=self.active_coils_all)
        print("full inductance matrix", inductance_matrix.shape)

        # option 1 reorder currents, fill in zeros and multiply by inductance matrix
        reshaped_currents = np.zeros(len(self.active_coils_all))
        for i, coil in enumerate(virtual_circuit.coils):
            # voltages_v1[i] = np.dot(inductance_matrix[self.order_dictionary[coil],:], delta_currents[:])
            reshaped_currents[self.order_dictionary[coil]] = delta_currents[i]
        voltages_v1 = np.dot(inductance_matrix, reshaped_currents)

        print(
            "volatges v1 : reorder currents, fill in zeros and multiply by full active coil inductance matrix"
        )
        print("voltages v1 : shape", voltages_v1.shape)
        print(voltages_v1)

        # option 2 reshape inductance matrix, muiltply by currents and then fill in zeros
        inductance_matrix_reduced = self.get_inductance(coils=virtual_circuit.coils)
        voltages_v2_temp = np.dot(inductance_matrix_reduced, delta_currents)
        # fill in zeros
        voltages_v2 = np.zeros(len(self.active_coils_all))
        for i, coil in enumerate(virtual_circuit.coils):
            voltages_v2[self.order_dictionary[coil]] = voltages_v2_temp[i]

        print(
            "voltages v2 : rehaped inductance matrix, then fill in zeros in voltage vector"
        )
        print("voltages v2 : shape", voltages_v2.shape)
        print(voltages_v2)

        self.feedback_voltages_v1 = voltages_v1
        self.feedback_voltages_v2 = voltages_v2

        return voltages_v1, voltages_v2

    # def feedback_voltage(feedback_current, inductance_matrix):
    #     """
    #     Compute feedback voltage from feedback current, by multiplying current by inductance matrix.

    #     Notes (to do)
    #     - check that current vector is the same length as the inductance matrix,
    #     - check ordering of currents in inductacne matrix, and in VC.
    #     - multiply current array by inductance matrix

    #     """
    #     # check dimensions
    #     assert inductance_matrix.shape[0] == len(
    #         feedback_current
    #     ), "The inductance matrix is not the same length as the current vector"

    #     voltage_array = np.dot(inductance_matrix, feedback_current)

    #     pass


### TESTING ###
# if __name__ == "__main__":

#     pass
