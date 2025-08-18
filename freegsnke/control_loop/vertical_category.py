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

        """

        # extract data
        z_ref = self.interpolants["z_ref"](t)
        k_prop = self.interpolants["k_prop"](t)
        k_deriv = self.interpolants["k_deriv"](t)

        return k_prop * (z_ref * ip_meas - zip_meas) + k_deriv * zipv_meas

    def run_control2(
        self,
        dt,
        target,
        history,
        Ip,
    ):
        """
        PID controller required for plasma vertical position. Computes the required voltage
        in the vertical stability coil to stabilise the plasma.

        Parameters
        ----------
        dt : float
            Time step over which controller should act [s].
        target : float
            Target vertical position [m].
        history : list
            List of previous vertical positions of the effective toroidal current center [m].
        k_prop : float
            Proportional gain controls how strongly the voltage reacts to deviations from the target.
        k_int : float
            Integral gain controls how the controller accumulates error over time (to correct drifts).
        k_deriv : float
            Derivative gain controls how the controller reacts to rapid changes in target.
        prop_exponent : float
            Exoponent in proportional term.
        prop_error : float
            Reference error for the proportional term.
        deriv_threshold : float
            Threshold for derivative action - limits effect of sudden jumps in target.
        int_factor : float
            Exponential decay factor that limits effect of older values on integral term.
        Ip : float
            Total plasma current at current time [Amps].
        Ip_ref : float
            Reference total plasma current [Amps], used to normalise output.
        derivative_lag : int
            Number of historical values over which the derivative term acts.

        Returns
        -------
        float
            Voltage required for the vertical stability coil to stabilise the plasma [Volts].
        """

        k_prop = -20000
        k_int = 0
        k_deriv = -50
        prop_exponent = 1.0
        prop_error = 1e-3
        deriv_threshold = 50
        int_factor = 0.98
        Ip_ref = None
        derivative_lag = 1

        # proportional term
        error = history[-1] - target
        output = (
            k_prop
            * prop_error
            * np.sign(error)
            * (np.abs(error / prop_error) ** prop_exponent)
        )

        # integral and derivative terms only if there's enough history
        if len(history) > derivative_lag:

            # integral term
            memory = (int_factor ** np.arange(len(history)))[::-1]
            integral_term = k_int * np.sum(np.array(history) * memory) * dt

            # derivative term (capped)
            derivative_term = k_deriv * (
                (history[-1] - history[-1 - derivative_lag]) / dt
            )
            derivative_term = np.sign(derivative_term) * min(
                abs(derivative_term), deriv_threshold
            )

            output += integral_term + derivative_term

        # scale by plasma current reference
        if Ip_ref is not None:
            output *= Ip / Ip_ref

        return output

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
