"""
Module to implement the PF category of MAST-U control loops.
"""

import numpy as np

from .target_scheduler import TargetScheduler


class PFController(TargetScheduler):
    """
    class to impliment the PF coil control

    Attributes:
    -----------
    inductance_full : full inductance matrix for all active coils

    Methods:
    --------
    reshape_inductance : retrieve inductance matrix from machine config, and select rows/columns

    machine_parameters : dict
            dictionary containing full  matrix and coil resistances
    """

    def __init__(
        self,
        coil_schedule: dict,
        coil_order: list[str],
        machine_parameters: dict,
    ):

        self.coil_schedule = coil_schedule
        self.coil_order = coil_order
        self.schedule_times = sorted(list(coil_schedule.keys()))
        # machine parameters - inductances,resitances,
        # self.M_FF = machine_parameters["M_FF"]
        # self.M_FB = machine_parameters["M_FB"]
        # self.R = machine_parameters["R"]
        # self.coil_order = machine_parameters["coil_order"]

        # # set machine parameters (inductance and resistances for coils)
        # self.inductance_full = machine_parameters["inductance_full"]
        # self.coil_resist = machine_parameters["coil_resist"]
        # self.machine_coils = machine_parameters["coils"]
        # self.machine_param_coil_order = machine_parameters["coil_order_dictionary"]

        # # reorder inductance matrix and coil resistances to match coil order
        # # ### ??? inducnace for active coils or control coils ???
        # # self.inductance_full = self.reshape_inductance(coils=self.active_coils)
        # # self.coil_resist = self.reorder_resistance(coils=self.active_coils)
        # self.inductance_full = self.reshape_inductance(coils=self.control_coils)
        # self.coil_resist = self.reorder_resistance(coils=self.control_coils)
        # # reduced inductance matrix for control coils
        # self.inductance_reduced = self.reshape_inductance(coils=self.control_coils)
        # # initialise the VCH object

        # Build coil gains array schedule.
        gain_schedule = {}
        for time in self.coil_schedule.keys():
            gains_arr = np.zeros(len(self.coil_order))
            gains_dict = self.coil_schedule[time]
            if "units" in gains_dict.keys():
                if gains_dict["units"] == "ms":
                    scale_factor = 1e-3  # convert ms to s
                    print("converting milliseconds to seconds")
            else:
                scale_factor = 1
            for i, coil in enumerate(self.coil_order):
                if coil in gains_dict["gains"].keys():
                    tau = (
                        gains_dict["gains"][coil] * scale_factor
                    )  # convert ms to seconds
                    gains_arr[i] = 1 / tau
                else:
                    print(f"No gains provided for coil {coil} - setting to zero")
                    gains_arr[i] = 0
            gain_schedule[time] = gains_arr

        self.schedule = {"coil_gains": gain_schedule}

    # overwrite get gains method
    def get_gains(self, time_stamp):
        """
        Get coil gains. Provided as time scales tau (s or ms) and gain = 1/tau.


        Parameters
        ----------
        time_stamp : float

        Returns
        gains_arr : np.ndarray
            array of coil gains"""
        return self.get_scheduled_params(time_stamp=time_stamp, param_type="coil_gains")

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

    def reorder_resistance(
        self,
        coils: list[str],
    ):
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


# will move this inside the class
def pf_voltage_demands(
    R,
    M_FF,
    M_FB,
    coil_gains,
    approved_dIdt,
    approved_I,
    measured_I,
    voltage_clips,
    slew_rates,
    prev_voltages,
    dt,
    verbose=False,
):
    """
    Calculate the output voltage to apply on the coils, as prescribed in
    the PF category of the PCS.

    Parameters
    ----------
    R : numpy 1D array
        The vector of resistances for the active coils.
    M_FF : numpy 2D array
        The feedforward inductance matrix of the active coils.
    M_FB : numpy 2D array
        The feedback inductance matrix of the active coils.
    coil_gains : numpy 1D array
        The vector of "coil gains" for the active coils (these are timescales in seconds).
    approved_dIdt: numpy 1D array
        Approved rate of change of the coil currents in the active coils (provided by the
        "system" category of the control loops). Input in A.
    approved_I : numpy 1D array
        Approved coil currents in the active coils (provided by the "system" category of
        the control loops). Input in A.
    measured_I : numpy 1D array
        Measured coil currents (from experiment or simulation) in the active coils. Input in A.
    voltage_clips : numpy 1D array
        Final voltage requests are clipped if the they are outside +/- these values
        for each of the active coils. Input in Volts.
    slew_rates : numpy 1D array
        Final voltage requests are clipped again if their derivatives (wrt previous timestep) are outside
        +/- these values for each of the active coils. Input in Volts/s.
    prev_voltages : numpy 1D array
        Voltage demands from the previous time step in the active coils. Input in Volts.
    dt : float
        Time step between current and previous voltage demands. Input in seconds.
    verbose : bool
        Print some output (True) or not (False).

    Returns
    -------
    numpy 1D array
        Voltage to apply to each of the active coils. Units in Volts.
    numpy 1D array
        Difference in currents in the feedback term used to calculate the feedback voltage. Units in Amps.
    """

    # resistive voltages
    v_res = measured_I * R
    if verbose:
        print("---")
        print(f"    Resistive voltage = {v_res}")

    # FF voltages
    v_FF = M_FF @ approved_dIdt
    if verbose:
        print(f"    Feedforward voltage = {v_FF}")

    # FB voltages
    delta_I = approved_I - measured_I
    v_FB = M_FB @ (delta_I / coil_gains)
    if verbose:
        print(f"    Feedback voltage = {v_FB}")

    # initial voltage demands (pre-clipping)
    v_init = v_res + v_FF + v_FB
    if verbose:
        print(f"    Pre-clipping voltage demand (sum of above) = {v_init}")

    # final voltage demands (clipped)
    v_init_clipped_pos = np.minimum(v_init, voltage_clips)
    v_clipped = np.maximum(v_init_clipped_pos, -voltage_clips)
    if verbose and not np.allclose(v_init, v_clipped):
        print(f"    Clipped voltage demand (according to `voltage_clips`) = {v_init}")

    # finally we apply the "slew rates", additive clipping of voltage rate of change
    delta_voltages = v_clipped - prev_voltages
    max_delta = slew_rates * dt
    delta_clipped = np.clip(delta_voltages, -max_delta, max_delta)
    v_final = prev_voltages + delta_clipped
    if verbose and not np.allclose(v_final, v_clipped):
        print(
            f"    Derivative clipped voltage demand (according to `slew_rates`) = {v_final}"
        )

    if verbose:
        print(f"FINAL VOLTAGE DEMANDS = {v_final}")

    return v_final, delta_I
