"""
Module to implement a Plasma Control System (PCS) in FreeGSNKE. 

"""

# imports
import numpy as np
from copy import deepcopy
from .plasma_category import PlasmaController
# from .pf_category import PFController, pf_voltage_demands
# from .shape_scheduling import ShapeTargetScheduler
# from .shape_targets_control import ShapeController
# from .system_category import system_approved_currents
# from .target_scheduler import TargetScheduler
# from .vertical_control import vertical_controller


class PlasmaControlSystem:
    """
    
    ADD DESCRIP.
    
    Attributes :
    ------------
    ip_controller : SolenoidController
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
        plasma_data,
        shape_data,
        circuits_data,
        systems_data,
        pf_data,
        active_coils,
        ctrl_coils,
        solenoid_coils,
        vertical_coils,
    ):

        # # store data dictionaries --> not sure we need to save this here
        # self.plasma_data = plasma_data
        # self.shape_data = shape_data
        # self.circuits_data = circuits_data
        # self.systems_data = systems_data
        # self.pf_data = pf_data

        # coil lists 
        self.active_coils = active_coils
        self.ctrl_coils = ctrl_coils
        self.solenoid_coils = solenoid_coils
        self.vertical_coils = vertical_coils

        # TODO consistency checks --> need to check all the minimum required inputs are present before initialisation
        self.check_data_entry(data=plasma_data, key="ip_fb", label="plasma")
        self.check_data_entry(data=plasma_data, key="vloop_ff", label="plasma")

        # TODO consistency checks --> need to check coils stuff


        # assign data to the individual controllers
        self.PlasmaController = PlasmaController(
            plasma_data=plasma_data,
        )

        # self.ShapeController = ShapeController(
        #     fefe
        # )

        # self.CircuitsController = CircuitsController(
        #     fefe
        # )

        # self.SystemsController = SystemsController(
        #     fefe
        # )

        # self.PfController = PfController(
        #     fefe
        # )


        # TODO
        # figure out storage/retrieval of various parametesr
        # figure out storage of integral/pi states etc. (and any other 'previous timestep' quantities)

    def check_data_entry(
        self,
        data, 
        key, 
        label
        ):
        """
        Check required 'times' and 'vals' fields are present in 'data[key]' dictionary. 

        Parameters
        ----------
        data: dict
            Dictionary of input data. 
        key : str
            Key value for the dictionary. 
        label : str
            Label describing which data dictionary is being checked.

        Returns
        -------
        None

        """

        if key not in data:
            raise ValueError(f"Missing key: '{key}' in {label} data dictionary.")

        if not isinstance(data[key], dict):
            raise ValueError(f"Expected '{key}' to be a dictionary in {label} data dictionary.")

        if "times" not in data[key]:
            raise ValueError(f"Missing key: 'times' in  {label} data dictionary (data['{key}']).")

        if "vals" not in data[key]:
            raise ValueError(f"Missing key: 'vals' in  {label} data dictionary (data['{key}']).")

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

        shape_current_rate = self.shape_controller.control_current_rates(
            time_stamp=time_stamp,
            target_obs=measured_shapes,
            # other args,
        )

        # add divertor here if its done separetly to shape (can put all shape/div targets into one controller if you want)
        divertor_current_rate = self.divertor_controller.control_current_rates(
            time_stamp=time_stamp,
            target_obs=measured_shapes,
            # other args,
        )

        total_current_rate = (
            ip_current_rate + shape_current_rate + divertor_current_rate
        )

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

    def get_fgs_inductance_resistance(self, stepping):
        """
        get inductance matrix and coil resistances from stepping object (Freegsnke)

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
        n_active_coils = (
            stepping.n_active_coils
        )  # could also be eq.tokamak.n_active_coils
        print("number active coils", n_active_coils)
        tok = stepping.eq1.tokamak
        active_coils = tok.coils_list[:n_active_coils]

        inductance_full = tok.coil_self_ind[: len(active_coils), : len(active_coils)]
        coil_resist = tok.coil_resist[: len(active_coils)]
        print(
            "Inductances and resistances retrieved for all active coils :", active_coils
        )
        coil_order_dictionary = {coil: i for i, coil in enumerate(active_coils)}

        return {
            "inductance_full": inductance_full,
            "coil_resist": coil_resist,
            "coils": active_coils,
            "coil_order_dictionary": coil_order_dictionary,
        }

