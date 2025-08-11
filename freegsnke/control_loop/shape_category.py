"""
Module to implement shape control in FreeGSNKE control loops. 

"""

import numpy as np

from freegsnke.control_loop.useful_functions import (
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


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

        # targets list
        self.ctrl_targets = ctrl_targets

        # check correct data is input and in correct format
        keys_to_spline = ["ff", "fb", "blend"]
        keys_to_step = ["k_prop", "k_int", "damping"]
        for targ in self.ctrl_targets:
            if targ not in data:
                raise ValueError(
                    f"{ShapeController}: Key '{targ}' not found in 'data'. "
                    f"Please include waveforms {keys_to_spline+keys_to_step} for '{targ}'."
                )
                for key in keys_to_spline + keys_to_step:
                    check_data_entry(
                        data=data[targ], key=key, controller_name="ShapeController"
                    )

        # create an internal copy of the data
        self.data = data

        # create a dictionary to store the spline functions
        self.interpolants = {}

        # interpolate the input data
        for key in self.ctrl_targets:
            self.interpolants[key] = {}
            for wave in keys_to_spline:
                self.interpolants[key][wave] = interpolate_spline(self.data[key][wave])
            for wave in keys_to_step:
                self.interpolants[key][wave] = interpolate_step(self.data[key][wave])

    def run_control(
        self,
        t,
        dt,
        T_meas,
        T_err_prev,
        T_hist_prev,
    ):
        """
        Runs the shape control PI loop with blending between feedforward and feedback.

        Parameters
        ----------
        t : float
            Current time (s).
        dt : float
            Time step (s).
        T_meas : np.ndarray
            Measured shape targets.
        T_err_prev : np.ndarray
            Previous filtered error signal.
        T_hist_prev : np.ndarray
            Previous integral (history) term.

        Returns
        -------
        dT_dt : np.ndarray
            Derivative of target requests (to be passed to circuit solver).
        T_err : np.ndarray
            Filtered error term at current time.
        T_hist : np.ndarray
            Updated integral term for next step.
        """

        # extract data
        T_fb = self.extract_values(t=t, targets=self.ctrl_targets, key="fb")
        T_ff = self.extract_values(t=t, targets=self.ctrl_targets, key="ff", deriv=True)
        T_blend = self.extract_values(t=t, targets=self.ctrl_targets, key="blend")
        k_prop = self.extract_values(t=t, targets=self.ctrl_targets, key="k_prop")
        k_int = self.extract_values(t=t, targets=self.ctrl_targets, key="k_int")
        alpha_inv = 1.0 / self.extract_values(
            t=t, targets=self.ctrl_targets, key="damping"
        )

        # proportional term
        T_err = (1 - alpha_inv) * T_err_prev + alpha_inv * (T_fb - T_meas)

        # integral term
        T_int = T_hist_prev + (0.5 * T_err * dt)

        # update hist
        T_hist = T_hist_prev + (T_err * dt)

        # FB term
        T_FB = (k_prop * T_err) + (k_int * T_int)

        # time deriv of shape target requests
        dT_dt = ((T_blend * T_FB) + ((1.0 - T_blend) * T_ff)).squeeze()

        return dT_dt, T_err, T_hist

    def extract_values(
        self,
        t,
        targets,
        key,
        deriv=False,
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
        deriv : bool
            Returns first derivative of the interpolant if True.

        Returns
        -------
        np.ndarray
            An array of interpolated values evaluated at time `t`, one for each target.
        """

        if deriv:
            return np.array(
                [self.interpolants[target][key].derivative()(t) for target in targets]
            )
        else:
            return np.array([self.interpolants[target][key](t) for target in targets])
