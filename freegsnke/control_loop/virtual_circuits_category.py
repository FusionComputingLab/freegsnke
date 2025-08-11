"""
Module to implement virtual circuits control in FreeGSNKE control loops. 

"""

import numpy as np

from freegsnke.control_loop.useful_functions import (
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


class VirtualCircuitsController:
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
        plasma_target,
    ):

        # targets list
        self.ctrl_targets = ctrl_targets
        self.plasma_target = plasma_target

        # check correct data is input and in correct format
        keys_to_spline = []
        keys_to_step = self.ctrl_targets + self.plasma_target
        for key in keys_to_spline + keys_to_step:
            check_data_entry(
                data=data, key=key, controller_name="VirtualCircuitsController"
            )

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
        dip_dt,
        dT_dt,
        I_approved_prev,
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

        # extract VC matrix (targets x coils)
        V = self.extract_values(t=t, targets=self.ctrl_targets + self.plasma_target)

        # unapproved coil currents rates of change
        dI_dt_unapproved = np.concatenate((dT_dt, [dip_dt])) @ V

        # unapproved coil currents (by simple Euler integration)
        I_unapproved = I_approved_prev + (dI_dt_unapproved * dt)

        return I_unapproved, dI_dt_unapproved

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
