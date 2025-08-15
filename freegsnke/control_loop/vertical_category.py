"""
Module to implement vertical plasma control in FreeGSNKE control loops. 

"""

import matplotlib.pyplot as plt
import numpy as np

from freegsnke.control_loop.useful_functions import (
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


class VerticalController:
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
        self.keys_to_spline = ["z_ref", "k_prop", "k_deriv"]
        self.keys_to_step = []
        for key in self.keys_to_spline + self.keys_to_step:
            check_data_entry(data=data, key=key, controller_name="VerticalController")

        # create an internal copy of the data
        self.data = data

        # create a dictionary to store the spline functions
        self.interpolants = {}

        # interpolate the input data
        for key in self.data.keys():
            self.interpolants[key] = {}
            if key in self.keys_to_spline:
                self.interpolants[key] = interpolate_spline(self.data[key])
            elif key in self.keys_to_step:
                self.interpolants[key] = interpolate_step(self.data[key])

    def run_control(
        self,
        t,
        dt,
        ip_meas,
        zip_meas,
        zipv_meas,
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

        # extract data
        z_ref = self.interpolants["z_ref"](t)
        k_prop = self.interpolants["k_prop"](t)
        k_deriv = self.interpolants["k_deriv"](t)

        return k_prop * (z_ref * ip_meas - zip_meas) + k_deriv * zipv_meas

    def plot_data(self, tmin=-1.0, tmax=1.0, nt=10001):
        """
        Plot selected time series from interpolated functions alongside their raw data.

        This function takes callable interpolants stored in `self.interpolants` and
        plots them on separate subplots, optionally overlaying the original raw
        data points from `self.data`.

        Parameters
        ----------
        tmin : float, optional
            Minimum time for the evaluation grid (default is -1.0).
        tmax : float, optional
            Maximum time for the evaluation grid (default is 1.0).
        nt : int, optional
            Number of equally spaced time points to evaluate the interpolants over
            between `tmin` and `tmax` (default is 10001).
        """

        # times to plot at
        t = np.linspace(tmin, tmax, nt)
        nplots = len(self.keys_to_spline + self.keys_to_step)  # number of plots

        # start plotting
        fig, axes = plt.subplots(nplots, 1, figsize=(10, 2.5 * nplots), sharex=True)

        if nplots == 1:
            axes = [axes]

        for ax, key in zip(axes, self.data.keys()):
            ax.plot(
                t,
                self.interpolants[key](t),
                color="navy",
                linewidth=0.8,
                label="interpolated",
            )
            ax.scatter(
                self.data[key]["times"],
                self.data[key]["vals"],
                s=3,
                marker=".",
                color="red",
                label=f"raw data",
            )
            ax.grid(True, linestyle="--", alpha=0.6)

            if key == "z_ref":
                ax.set_ylabel(rf"{key} [$m$]")
            # elif key == "k_prop":
            #     ax.set_ylabel(rf"{key} [$1/s$]")
            # elif key == "k_deriv":
            #     ax.set_ylabel(rf"{key} [$1/s^2$]")
            else:
                ax.set_ylabel(key)

        axes[0].legend(loc="best")
        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
