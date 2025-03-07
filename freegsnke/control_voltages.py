"""
Module to obtain control voltages from virtual circuits.

"""

import numpy as np
from copy import deepcopy

from . import virtual_circuits as vc  # import the virtual circuit class
from . import machine_config



class ControlVoltages:
    """
    Class to implement control voltages from virtual circuit, and given a set of requested target shifts.

    """

    def __init__(self):
        """Initialize the control voltages class"""

    def assign_eqi():

        pass

    def get_active_coils(self, eq): 
        """ set default coils to be used and set the order according to that in the tokamak description 
        get all active ones
        assign reduced set of coils without solenoid and p6 (these voltages will be set via  different method)
        """



    def get_inductance(self, coils = None):
        """retrieve inductance matrix from machine config
        
        comes from machine_config.coil_self_ind . only want active part 
        """ 

        pass
    
    def get_vc():
        """ 
        If a vc ojbject is not provided, compute using freegsnke calculate_VC methods.
        Add option to provide vc object from another source (stored in file or NN emulator)
        """
        pass

    def feedback_current_vector(
        eq,
        profiles,
        time,
        targets_req,
        targets_obs=None,
        virtucal_circuit=None,
        gain_matrix=None,
    ):
        """
        Compute current given a set of target value shifs and vc matrix, at a given time

        Parameters
        ----------
        targets_req : array
            array function of target values for each control voltage as a function of time

        targets_obs : array
            array function of target values for each control voltage as a function of time.
            Defaults to None, in which case the targets are computed from the equilibrium.

        gain_matrix : array
            array function of target gains. Defaults to identiy matrix

        time : float
            time at which to compute feedback current

        virtucal_circuit : object
            virtual circuit object. Defaults to None, in which case the virtual circuit is computed from the equilibrium.
            with default currents of the Tokamak minus p6, and solenoid (these are determined differently)


        Returns
        -------
        feedback_current : array


        Notes (to do)
        - check that target vector is same length as config of virtual circuit
        - check
        """

        pass

    def feedback_voltage(feedback_current):
        """
        Compute feedback voltage from feedback current, by multiplying current by inductance matrix.

        Notes (to do)
        - check that current vector is the same length as the inductance matrix,
        - check ordering of currents in inductacne matrix, and in VC.
        - multiply current array by inductance matrix

        """

        pass
