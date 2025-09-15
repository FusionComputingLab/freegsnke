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
    A controller class for managing vertical plasma control.

    Parameters
    ----------
    data : dict
        A nested dictionary containing control waveforms for the vertical controller.
        The required keys for both spline-based and step-based waveforms are:
            - Spline keys: "z_ref", "k_prop", "k_deriv"
            - Step keys:
        Each key should map to a waveform dictionary suitable for interpolation with keys:
            - 'times': 1D array of time points
            - 'vals': 1D array of values at those time points (same length).

    Attributes
    ----------
    keys_to_spline : list of str
        Keys corresponding to waveforms that will be interpolated using splines.

    keys_to_step : list of str
        Keys corresponding to waveforms that will be interpolated using step functions.

    data : dict
        Internal copy of the input control waveforms.

    interpolants : dict
        A nested dictionary storing interpolation functions of each input waveform.
        Structure: {spline/step key: interpolant_function}

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

        # interpolate the input data
        self.update_interpolants()

    def update_interpolants(self):
        """
        Recompute all interpolant functions from the current `self.data`.

        This method clears the existing `self.interpolants` dictionary and
        rebuilds it by applying either `interpolate_spline` or `interpolate_step`
        depending on whether each key belongs to `self.keys_to_spline` or
        `self.keys_to_step`.

        """

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
        Compute the control signal for plasma vertical position regulation using a
        proportional-derivative (PD) control law.

        This method uses interpolated reference and gain values to calculate the control
        output based on the measured plasma current, vertical position, and vertical velocity.

        Parameters
        ----------
        t : float
            Current time [s].

        dt : float
            Time step [s].

        ip_meas : float
            Measured plasma current [A].

        zip_meas : float
            Measured vertical position of the plasma multiplied by measured Ip [A.m].

        zipv_meas : float
            Measured vertical velocity of the plasma multiplied by measured Ip [A.m/s].

        Returns
        -------
        control_signal : float
            Output of the PD controller, representing the voltage command
            for vertical position regulation.
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
        Computes a control signal using a nonlinear proportional-integral-derivative (PID) controller
        for regulating plasma vertical position.

        This method applies a nonlinear proportional term with exponentiation and saturation,
        an exponentially weighted integral term, and a capped derivative term. The output can
        optionally be scaled by the plasma current.

        Parameters
        ----------
        dt : float
            Time step [s].
        target : float
            Desired plasma vertical position [m].
        history : list of float
            Time-ordered list of past measurements of the plasma vertical position [m].
        Ip : float
            Measured plasma current [A].

        Returns
        -------
        output : float
            Control voltage computed from the PID logic, potentially scaled by plasma current.
        """

        # defualt fixed params
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
        Visualizes interpolated control waveforms and corresponding raw inputs.

        This method generates subplots for each control waveform (step types),
        showing the interpolated time series alongside the original data points. It helps verify
        the quality and behavior of the interpolation.

        Parameters
        ----------
        tmin : float, optional
            Start time for the evaluation grid (default is -1.0 seconds).
        tmax : float, optional
            End time for the evaluation grid (default is 1.0 seconds).
        nt : int, optional
            Number of time points to evaluate the interpolants over the interval [tmin, tmax] (default is 10001).

        Notes
        -----
        - Each subplot corresponds to a control waveform.
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
