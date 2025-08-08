"""
Module to implement plasma control in FreeGSNKE control loops. 

"""

# imports
import numpy as np

from freegsnke.control_loop.useful_functions import interpolate_spline, interpolate_step


class PlasmaController:
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
            self.interpolants[key] = {}
            if key in ["ip_fb", "ip_blend", "vloop_ff"]:
                self.interpolants[key] = interpolate_spline(self.data[key])
            elif key in ["k_prop", "k_int", "M_solenoid"]:
                self.interpolants[key] = interpolate_step(self.data[key])

    def run_control(
        self,
        t,
        dt,
        ip_meas,
        ip_hist_prev,
    ):
        """
        NEED TO UPDATE.
        Calculates the vector of current trajectories ΔI/Δt, as prescribed
        in the plasma category of the MAST-U PCS. The equations followed are:

        Ip_error = (Ip_req - Ip_obs)
        integral = internal_state + 0.5 * Ip_error * dt
        internal_state = internal_state + Ip_error * dt
        ΔIsol_fb/Δt = Kp * Ip_error + Ki * integral
        ΔIsol/Δt = ΔIsol_fb/Δt * blend - Vloop_ff * (1 - blend)/M_sp

        It should be noted that the PI controller works at a frequency twice as
        high as the data recording system. This is why the PI controller goes
        through two cycles in this method.

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

        # proportional term
        ip_err = self.interpolants["ip_fb"](t) - ip_meas
        k_prop = self.interpolants["k_prop"](t)
        k_int = self.interpolants["k_int"](t)
        blend = self.interpolants["ip_blend"](t)
        vloop_ff = self.interpolants["vloop_ff"](t)
        M_solenoid = self.interpolants["M_solenoid"](t)

        # integral term
        ip_int = ip_hist_prev + (0.5 * ip_err * dt)

        # update ip_hist
        ip_hist = ip_hist_prev + (ip_err * dt)

        # FB term
        ip_fb = (k_prop * ip_err) + (k_int * ip_int)

        # time deriv of plasma current request
        dip_dt = (blend * ip_fb) + ((1 - blend) * (vloop_ff / M_solenoid))

        return dip_dt, ip_hist
