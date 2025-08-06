"""
Module to implement shape control in FreeGSNKE control loops. 

"""

import numpy as np

from freegsnke.control_loop.useful_functions import interpolate_spline, interpolate_step


class ShapeController:
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
        ctrl_targets,
    ):

        # create an internal copy of the data
        self.data = data

        # targets list
        self.ctrl_targets = ctrl_targets

        # create a dictionary to store the spline functions
        self.interpolants = {}

        # interpolate the input data
        waveforms = ["ff", "fb", "blend", "k_prop", "k_int", "damping"]
        for key in self.ctrl_targets:
            self.interpolants[key] = {}
            for wave in waveforms:
                if wave in ["ff", "fb", "blend"]:
                    self.interpolants[key][wave] = interpolate_spline(
                        self.data[key][wave]
                    )
                else:
                    self.interpolants[key][wave] = interpolate_step(
                        self.data[key][wave]
                    )

    def run_control(
        self,
        t,
        dt,
        T_meas,
        T_err_prev,
        T_hist_prev,
    ):
        """
        NEED TO UPDATE.


        Parameters
        ----------
        - Kp : float
            Proportional term used in the Vloop_fb computation.


        Returns
        -------
        - dI_dt : 1D numpy array
            Array of delta currents requests that will be part of the input of
            Circuits category.

        """

        # extract data
        T_fb = self.extract_values(t=t, targets=self.ctrl_targets, key="fb")
        T_ff = self.extract_values(t=t, targets=self.ctrl_targets, key="ff")
        T_blend = self.extract_values(t=t, targets=self.ctrl_targets, key="blend")
        k_prop = self.extract_values(t=t, targets=self.ctrl_targets, key="k_prop")
        k_int = self.extract_values(t=t, targets=self.ctrl_targets, key="k_int")
        alpha_inv = 1 / self.extract_values(
            t=t, targets=self.ctrl_targets, key="damping"
        )

        # proportional term
        T_err = (1 - alpha_inv) * T_err_prev + alpha_inv * (T_fb - T_meas)

        # integral term
        T_int = T_hist_prev + 0.5 * T_err * dt

        # update hist
        T_hist = T_hist_prev + T_err * dt

        # FB term
        T_fb = k_prop * T_err + k_int * T_int

        # time deriv of shape target requests
        dT_dt = T_blend * T_fb + (1 - T_blend) * T_ff

        return dT_dt, T_err, T_hist

    def extract_values(
        self,
        t,
        targets,
        key,
    ):
        """
        Evaluate and extract interpolated values at a given time for specified targets.

        Parameters
        ----------
        t : float
            The time at which to evaluate the interpolants.
        targets : list of str
            A list of target names corresponding to keys in `self.interpolants`.
        key : str
            The dictionary key (e.g., 'fb') used to select the interpolation function for each target.

        Returns
        -------
        np.ndarray
            An array of interpolated values evaluated at time `t`, one for each target.
        """

        return np.array([self.interpolants[target][key](t) for target in targets])
