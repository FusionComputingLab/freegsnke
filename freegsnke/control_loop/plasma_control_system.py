"""Plasma Control System

Bring together all the elements of PCS in the final PF and systems categories.


"""

import time
from copy import deepcopy
from pprint import pprint
from sys import float_info

# imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from .ip_control import ControlSolenoid
from .pf_category import pf_voltage_demands
from .shape_scheduling import ShapeTargetScheduler
from .shape_targets_control import ShapeController
from .system_category import system_approved_currents
from .target_scheduler import TargetScheduler
from .vertical_control import vertical_controller


class PlasmaControlSystem:
    """
    Class to implement control voltages from virtual circuit, and given a set of observed target values, and a set of requested target values.

    Attributes :
    ------------
    ip_controller : ControlSolenoid
        An object that controls the plasma current. Contains schedules and waveforms.
    shape_controller : ShapeController
        An object that controls the shape currents. Contains schedules and waveforms.
    divertor_controller : ShapeController
        An object that controls the divertor currents. Contains schedules and waveforms.
    vertical_controller : vertical_controller
        An object that controls the vertical position of the plasma.
    active_coils : list of all active coils
        A list of all active coils - used for shape, ip and vertical control.
    shaping_coils : list
        A list of control used for shape control. These correspond to coils in VC's.
    vertical_coils : list
        A list of coils used for vertical control.
    pf_schedule : dict
        A dictionary of coil gains.


    Methods :
    ---------
    """

    def __init__(
        self,
        ip_controller: ControlSolenoid,
        shape_controller: ShapeController,
        divertor_controller: ShapeController,
        vertical_controller,
        pf_schedule: TargetScheduler,
        active_coils: list[str],
        shaping_coils: list[str],
        vertical_coils: list[str],
        machine_parameters: dict,
    ):

        # assign controllers
        self.ip_controller = ip_controller
        self.shape_controller = shape_controller
        self.divertor_controller = divertor_controller
        self.vertical_controller = vertical_controller
        self.pf_schedule = pf_schedule
        self.active_coils = active_coils
        self.shaping_coils = shaping_coils
        self.vertical_coils = vertical_coils
        print("plasma control system initialised")
        print("all active", self.active_coils)
        print("shaping coils", self.shaping_coils)
        print("vertical coils", self.vertical_coils)

        # TODO
        # Run consistency checks for controllers (coils labelling across controllers etc.)
        # maybe use load_setup_from_files to create controllers
        # figure out storage/retrieval of various parametesr
        # figure out storage of integral/pi states etc. (and any other 'previous timestep' quantities)

    def load_setup_from_files(self, uda_file, pcs_file):
        """
        Load data from files and create schedulers, controllers etc.

        NB This currently is set to use the formats used for testing. Will update later

        # load files and create schedule/waveofrm dictionaries.
        #  NB this will be removed once input files in the correct format are provided

        Modify class - assign attributes
        """

        # load dictionaries from file

        # construct schedulers

        # construct controllers

        pass

    def compute_control_voltage(
        self,
        time_stamp,
        measured_shapes,
        measured_I,
        other_args,
    ):
        """Compute voltages for control coils

        1) compute current rates from plasma, shape and divertor controllers
        This applies plasma/shape/divertor and circuits category calculations in one step.
        2) combine the current rates and apply systems category clipping to get approved currents/rates
        3) apply PF category and multiply by inductances to obtain voltages


        """
        # 1) current rates
        ip_current_rate = self.ip_controller.ip_control(
            ts=time_stamp,
            Ip_obs_t=measured_I,
            # other_args
        )

        shape_current_rate = self.shape_controller.control_shape_rates(
            time_stamp=time_stamp,
            target_obs=measured_shapes,
            # other args,
        )

        total_current_rate = ip_current_rate + shape_current_rate

        # 2) SYSYTEMS :  apply current clipping and coil pertubations (S)
        approved_current_rates = system_approved_currents(
            total_current_rate,
            ...,
        )

        # 3) PF : convert currents and current rates into voltages
        voltage_request = pf_voltage_demands(
            # input arguments
        )

        pass

    def compute_vertical_voltage(self, time_stamp, other_args):
        """
        Apply vertical control to the vertical coils

        """
        # apply vertical controller to get voltages for
        pass

    def compute_voltages_all(self, time_stamp):

        # shaping voltage,
        # vertical voltage
        # combine these two

        control_voltages = self.compute_control_voltage()
        vertical_voltage = self.compute_vertical_voltage()

        return control_voltages + vertical_voltage


class Simulator:
    """Simulator class to interface PCS control with dynamic solving in freegsnke"""

    def __init__(
        self,
        pcs_controller: PlasmaControlSystem,
        stepping,
    ):
        """initialse simulator

        Parameters:
        pcs  : PlasmaControlSystem
            instance of PCS class to compute voltages
        stepping : FreeGSNKE NL solver
            dynamic forward solver in freegsnke

        """
        self.stepping = stepping
        self.controller = pcs_controller

    def initialise_VCH(
        self,
        stepping,
        target_relative_tolerance: float = 1e-7,
    ):
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

    def get_observed_from_eqi(
        self,
    ):
        """
        Compute or retrieve all appropriate observed quantites from equilibrium
        Shape targets, plasma current, etc.

        Returns
        -------

        """

        pass

    def update_eqi(self, voltages, linear=True):
        """
        update and solve equilibrium with new voltages as inputs
        """
        if linear == True:
            print("updating equilibrium  : LINEAR")
            self.stepping.nlstepper(
                active_voltage_vec=voltages,
                linear_only=True,  # linearise solve only
                verbose=False,
                # custom_coil_resist=coil_resist,   #options for restistances/inductances used in solve
                # custom_self_ind=coil_ind)
            )
        elif linear == False:
            print("updating equilibrium : NON-LINEAR")
            self.stepping.nlstepper(
                active_voltage_vec=voltages,
                linear_only=False,  # Non linear solve
                verbose=False,
                # custom_coil_resist=coil_resist,   #options for restistances/inductances used in solve
                # custom_self_ind=coil_ind)
            )
        print("equi updated")

    def simulate(self, control_times):
        """run simulation ..."""

        # for time in times
        # compute voltage request
        # solve for new equilibrium
        # compute new plasma shape targets/currents
        pass


def get_inductance_resistance(self, stepping):
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
