"""
Module to implement PF control in FreeGSNKE control loops. 

"""

import numpy as np

from freegsnke.control_loop.useful_functions import (
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


class PFController:
    """
    ADD DESCRIP.

    Parameters
    ----------


    Attributes
    ----------

    """

    def __init__(
        self,
        data,
    ):

        # check correct data is input and in correct format
        keys_to_spline = []
        keys_to_step = [
            "R_matrix",
            "M_FF_matrix",
            "M_FB_matrix",
            "coil_gains",
            "coil_voltage_signs",
            "coil_voltage_lims",
            "coil_voltage_slew_lims",
        ]
        for key in keys_to_spline + keys_to_step:
            check_data_entry(data=data, key=key, controller_name="PFController")

        # create an internal copy of the data
        self.data = data

        # create a dictionary to store the spline functions
        self.interpolants = {}

        # interpolate the input data
        for key in keys_to_step:
            self.interpolants[key] = interpolate_step(self.data[key])

    def run_control(
        self,
        t,
        dt,
        I_meas,
        I_approved,
        dI_dt_approved,
        V_approved_prev,
        verbose=False,
    ):
        """
        Compute the coil voltage demands for the current control loop step.

        This method implements a control loop with resistive, feedforward (FF),
        and feedback (FB) voltage components, applies voltage limits and slew rate
        constraints, and returns the final approved voltage demands.

        Parameters
        ----------
        t : float
            Current time (used to interpolate time-dependent system matrices).
        dt : float
            Time step between the current and previous voltage demands, in seconds.
        I_meas : np.ndarray
            Measured coil currents at time `t`, in Amps.
        I_approved : np.ndarray
            Approved coil currents (from system controller), in Amps.
        dI_dt_approved : np.ndarray
            Approved rate of change of coil currents (from system controller), in Amps/sec.
        V_approved_prev : np.ndarray
            Previously approved coil voltage demands, in Volts.
        verbose : bool, optional
            If True, print detailed diagnostic output.

        Returns
        -------
        V_approved : np.ndarray
            Final voltage demand to apply to the active coils, in Volts.
        """
        # extract interpolated data
        R = self.interpolants["R_matrix"](t)
        M_FF = self.interpolants["M_FF_matrix"](t)
        M_FB = self.interpolants["M_FB_matrix"](t)
        coil_gains = self.interpolants["coil_gains"](t)
        voltage_clips = self.interpolants["coil_voltage_lims"](t)
        slew_rates = self.interpolants["coil_voltage_slew_lims"](t)
        voltage_signs = self.interpolants["coil_voltage_signs"](t)

        # resistive voltages
        v_res = R * I_meas

        # FF voltages
        v_FF = M_FF @ dI_dt_approved

        # FB voltages
        delta_I = I_approved - I_meas
        v_FB = M_FB @ (delta_I / coil_gains)

        # initial voltage demands (pre-clipping)
        v_init = v_res + v_FF + v_FB

        # clip voltage to max/min allowed
        v_clipped = np.clip(v_init, -voltage_clips, voltage_clips)

        # apply slew rate constraints
        delta_voltages = v_clipped - (V_approved_prev * 1.0)
        max_delta = slew_rates * dt
        delta_clipped = np.clip(delta_voltages, -max_delta, max_delta)
        V_approved = (V_approved_prev * 1.0) + delta_clipped

        return V_approved

    def extract_values(
        self,
        t,
        targets,
    ):
        """
        Evaluate and extract interpolated values at a given time for specified targets.

        Parameters
        ----------
        t : float
            The time at which to evaluate the interpolants.
        targets : list of str
            A list of target names corresponding to keys in `self.interpolants`.

        Returns
        -------
        np.ndarray
            An array of interpolated values evaluated at time `t`, one for each target.
        """

        return np.array([self.interpolants[target](t) for target in targets])
