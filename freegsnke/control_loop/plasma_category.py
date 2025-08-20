"""
Module to implement plasma control in FreeGSNKE control loops. 

"""

import matplotlib.pyplot as plt
import numpy as np

from freegsnke.control_loop.useful_functions import (
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


class PlasmaController:
    """
    A controller class for managing plasma-related control parameters using interpolated input data.

    Parameters
    ----------
    data : dict
        A nested dictionary containing control data for each target to be controlled. Each target's
        data must include keys for both spline-based and step-based parameters:
            - Spline keys: "ip_ref", "ip_blend", "vloop_ff"
            - Step keys: "k_prop", "k_int", "M_solenoid"
        Each key should map to a dictionary suitable for interpolation, with keys:
            - 'times': 1D array of time points
            - 'vals': 1D array of values at those time points (same length).

    Attributes
    ----------
    keys_to_spline : list of str
        Keys corresponding to parameters that will be interpolated using splines.

    keys_to_step : list of str
        Keys corresponding to parameters that will be interpolated using step functions.

    data : dict
        Internal copy of the input control data.

    interpolants : dict
        A nested dictionary storing interpolation functions for each control target and parameter.
        Structure: {target: {param: interpolant_function}}

    """

    def __init__(
        self,
        data,
    ):

        # check correct data is input and in correct format
        self.keys_to_spline = ["ip_ref", "ip_blend", "vloop_ff"]
        self.keys_to_step = ["k_prop", "k_int", "M_solenoid"]
        for key in self.keys_to_spline + self.keys_to_step:
            check_data_entry(data=data, key=key, controller_name="PlasmaController")

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
        ip_hist_prev,
    ):
        """
        Computes the time derivative of the plasma current request (`dip_dt`) and updates the
        integral history of the plasma current error (`ip_hist`) using a blended feedback and
        feedforward control strategy.

        Parameters:
        ----------
        t : float
            Current time in seconds.
        dt : float
            Time step in seconds.
        ip_meas : float
            Measured plasma current at time `t`.
        ip_hist_prev : float
            Previous value of the integrated plasma current error.

        Returns:
        -------
        dip_dt : float
            Time derivative of the requested plasma current.
        ip_hist : float
            Updated integral of the plasma current error.

        Notes:
        ------
        - The control law uses time-dependent interpolants for reference current (`ip_ref`),
        proportional gain (`k_prop`), integral gain (`k_int`), blend factor (`ip_blend`),
        feedforward voltage (`vloop_ff`), and solenoid inductance (`M_solenoid`).
        - The blend factor determines the weighting between feedback and feedforward control.
        - The integral term is computed using the trapezoidal rule for numerical integration.
        """

        # proportional term
        ip_err = self.interpolants["ip_ref"](t) - ip_meas
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
        ip_ref = (k_prop * ip_err) + (k_int * ip_int)

        # time deriv of plasma current request
        dip_dt = (blend * ip_ref) + ((1 - blend) * (vloop_ff / M_solenoid))

        return dip_dt, ip_hist

    def plot_data(self, tmin=-1.0, tmax=1.0, nt=10001):
        """
        Visualizes interpolated control data and corresponding raw input for a specified target.

        This method generates subplots for each control parameter (both spline and step types),
        showing the interpolated time series alongside the original data points. It helps verify
        the quality and behavior of the interpolation.

        Parameters
        ----------
        targ : str
            The name of the control target to plot. Must be a key in `self.interpolants` and `self.data`.
        tmin : float, optional
            Start time for the evaluation grid (default is -1.0 seconds).
        tmax : float, optional
            End time for the evaluation grid (default is 1.0 seconds).
        nt : int, optional
            Number of time points to evaluate the interpolants over the interval [tmin, tmax] (default is 10001).

        Notes
        -----
        - Each subplot corresponds to a control parameter (e.g., 'ip_ref', 'ip_blend', 'vloop_ff', 'k_prop', etc.).
        - Interpolated curves are plotted in navy; raw data points are shown in red.
        - Axis labels include units where applicable.
        - Useful for debugging or validating the interpolation quality.
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

            if key == "ip_ref":
                ax.set_ylabel(rf"{key} [$A/s$]")
            elif key == "vloop_ff":
                ax.set_ylabel(rf"{key} [$V$]")
            elif key == "k_prop":
                ax.set_ylabel(rf"{key} [$1/s$]")
            elif key == "k_int":
                ax.set_ylabel(rf"{key} [$1/s^2$]")
            elif key == "M_solenoid":
                ax.set_ylabel(rf"{key} [$V.s/A$]")
            else:
                ax.set_ylabel(key)

        axes[0].legend(loc="best")
        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
