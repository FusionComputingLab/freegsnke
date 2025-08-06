"""
Module to implement PF control in FreeGSNKE control loops. 

"""

import numpy as np

from freegsnke.control_loop.useful_functions import interpolate_spline, interpolate_step


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

        # create an internal copy of the data
        self.data = data

        # create a dictionary to store the spline functions
        self.interpolants = {}

        # interpolate the input data
        for key in self.data.keys():
            if key not in ["coil_order"]:
                self.interpolants[key] = interpolate_step(self.data[key])

    # will move this inside the class
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
        FIX.

        Calculate the output voltage to apply on the coils, as prescribed in
        the PF category of the PCS.

        Parameters
        ----------
        approved_dIdt: numpy 1D array
            Approved rate of change of the coil currents in the active coils (provided by the
            "system" category of the control loops). Input in A.
        approved_I : numpy 1D array
            Approved coil currents in the active coils (provided by the "system" category of
            the control loops). Input in A.
        I_meas : numpy 1D array
            Measured coil currents (from experiment or simulation) in the active coils. Input in A.

        V_approved_prev : numpy 1D array
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

        # extract interpolated data
        R = self.interpolants["R_matrix"](t)
        M_FF = self.interpolants["M_FF_matrix"](t)
        M_FB = self.interpolants["M_FB_matrix"](t)
        coil_gains = self.interpolants["coil_gains"](t)
        voltage_clips = self.interpolants["coil_voltage_lims"](t)
        slew_rates = self.interpolants["coil_voltage_slew_lims"](t)

        # resistive voltages
        v_res = R @ I_meas
        if verbose:
            print("---")
            print(f"    Resistive voltage = {v_res}")

        # FF voltages
        v_FF = M_FF @ dI_dt_approved
        if verbose:
            print(f"    Feedforward voltage = {v_FF}")

        # FB voltages
        delta_I = I_approved - I_meas
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
            print(
                f"    Clipped voltage demand (according to `voltage_clips`) = {v_init}"
            )

        # finally we apply the "slew rates", additive clipping of voltage rate of change
        delta_voltages = v_clipped - V_approved_prev
        max_delta = slew_rates * dt
        delta_clipped = np.clip(delta_voltages, -max_delta, max_delta)
        V_approved = V_approved_prev + delta_clipped
        if verbose and not np.allclose(V_approved, v_clipped):
            print(
                f"    Derivative clipped voltage demand (according to `slew_rates`) = {V_approved}"
            )

        if verbose:
            print(f"FINAL VOLTAGE DEMANDS = {V_approved}")

        return V_approved

    def extract_values(
        self,
        t,
        targets,
        derivative=False,
    ):
        """
        Evaluate and extract interpolated values at a given time for specified targets.

        Parameters
        ----------
        t : float
            The time at which to evaluate the interpolants.
        targets : list of str
            A list of target names corresponding to keys in `self.interpolants`.
        derivative : bool
            If True, evaluates the first derivative of the interpolated function.

        Returns
        -------
        np.ndarray
            An array of interpolated values evaluated at time `t`, one for each target.
        """

        if derivative:
            return np.array(
                [self.interpolants[target].derivative()(t) for target in targets]
            )
        else:
            return np.array([self.interpolants[target](t) for target in targets])
