"""
Module to implement shape control in FreeGSNKE control loops. 

"""

import matplotlib.pyplot as plt
import numpy as np

from freegsnke.control_loop.useful_functions import (
    PID,
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


class ShapeController:
    """
    A controller class for managing shape control waveforms.

    Parameters
    ----------
    data : dict
        A nested dictionary containing waveforms for each target to be controlled. Each target's
        dictionary must include keys for both spline-based and step-based parameters:
            - Spline keys: "ff", "ref", "blend"
            - Step keys: "k_prop", "k_int", "damping"
        Each key should map to a waveform dictionary suitable for interpolation with keys:
            - 'times': 1D array of time points
            - 'vals': 1D array of values at those time points (same length).

    ctrl_targets : list of str
        A list of shape target names (keys in `data`) that the controller will manage.

    mode : str
        Choose the type of controller to use, here the default is an "PI_with_P_damping"
        controller, see "run_control" method for more information.

    Attributes
    ----------
    ctrl_targets : list of str
        The list of shape targets being managed.

    keys_to_spline : list of str
        Keys corresponding to waveforms that will be interpolated using splines.

    keys_to_step : list of str
        Keys corresponding to waveforms that will be interpolated using step functions.

    data : dict
        Internal copy of the input control waveforms.

    interpolants : dict
        A nested dictionary storing interpolation functions of each input waveform for each
        shape target.
        Structure: {target: {spline/step key: interpolant_function}}
    """

    def __init__(
        self,
        data,
        ctrl_targets,
        mode=None,
    ):

        # targets list
        self.ctrl_targets = ctrl_targets

        # create an internal copy of the data
        self.data = data

        # choose controller to use (more can be added)
        if mode is None:
            mode = "PI_with_P_damping"

        if mode == "PI_with_P_damping":
            # select control algorithm
            self.run_control = self.run_control_PI_with_P_damping

            # inputs required for this algorithm
            self.keys_to_spline = ["ff", "ref", "blend"]
            self.keys_to_step = ["k_prop", "k_int", "damping"]

        elif mode == "PID_with_scaled_out_damping":
            # select control algorithm
            self.run_control = self.run_control_PID_with_scaled_out_damping

            # inputs required for this algorithm
            self.keys_to_spline = ["ff", "ref", "blend"]
            self.keys_to_step = ["k_prop", "damping"]

        elif mode == "PID":
            # select control algorithm
            self.run_control = self.run_control_PID

            # inputs required for this algorithm
            self.keys_to_spline = ["ff", "ref", "blend"]
            self.keys_to_step = ["k_prop", "k_int", "k_deriv"]

        # check correct data is input and in correct format
        for targ in self.ctrl_targets:
            for key in self.keys_to_spline + self.keys_to_step:
                check_data_entry(
                    data=data[targ], key=key, controller_name="ShapeController"
                )

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

        # create a dictionary to store the spline funcions
        self.interpolants = {}

        # interpolate the input data
        for targ in self.ctrl_targets:
            self.interpolants[targ] = {}
            for key in self.keys_to_spline:
                self.interpolants[targ][key] = interpolate_spline(self.data[targ][key])
            for key in self.keys_to_step:
                self.interpolants[targ][key] = interpolate_step(self.data[targ][key])

    def run_control_PI_with_P_damping(
        self,
        t,
        dt,
        T_meas,
        T_err_prev,
        T_hist_prev,
    ):
        """
        Computes the time derivative of shape target requests based on measured values,
        reference trajectories, and control gains. It blends feedforward and feedback
        contributions using a time-varying blend factor, and applies damping to the error
        signal.

        Parameters
        ----------
        t : float
            Current time [s].
        dt : float
            Time step [s].
        T_meas : np.ndarray
            Measured values of the shape targets at the current time [m].
        T_err_prev : np.ndarray
            Previously filtered error signal (used for damping) [m].
        T_hist_prev : np.ndarray
            Previous integral term (used for PI control) [m.s].

        Returns
        -------
        dT_dt : np.ndarray
            Time derivative of the shape target requests [m/s].
        T_err : np.ndarray
            Filtered error signal at the current time [m].
        T_hist : np.ndarray
            Updated integral term for use in the next control step [m.s].

        Notes
        -----
        - The error signal is filtered using a damping factor to smooth transitions.
        - The integral term is updated using trapezoidal integration.
        - The final output blends feedforward and feedback derivatives based on a dynamic blend factor.
        """

        # extract data
        T_ref = self.extract_values(t=t, targets=self.ctrl_targets, key="ref")
        T_ff_deriv = self.extract_values(
            t=t, targets=self.ctrl_targets, key="ff", deriv=True
        )
        T_blend = self.extract_values(t=t, targets=self.ctrl_targets, key="blend")
        k_prop = self.extract_values(t=t, targets=self.ctrl_targets, key="k_prop")
        k_int = self.extract_values(t=t, targets=self.ctrl_targets, key="k_int")
        alpha_inv = 1.0 / self.extract_values(
            t=t, targets=self.ctrl_targets, key="damping"
        )

        # proportional term
        T_err = ((1 - alpha_inv) * T_err_prev) + (alpha_inv * (T_ref - T_meas))

        # integral term
        T_int = T_hist_prev + (0.5 * T_err * dt)

        # update hist
        T_hist = T_hist_prev + (T_err * dt)

        # FB term
        T_fb_deriv = PID(
            error_prop=T_err,
            error_int=T_int,
            error_deriv=None,
            k_prop=k_prop,
            k_int=k_int,
            k_deriv=0.0,
        )

        # time deriv of shape target requests
        dT_dt = ((T_blend * T_fb_deriv) + ((1.0 - T_blend) * T_ff_deriv)).squeeze()

        return dT_dt.squeeze(), T_err.squeeze(), T_hist.squeeze()

    def run_control_PID_with_scaled_out_damping(
        self,
        t,
        dt,
        T_meas,
        T_err_prev,
        T_hist_prev,
    ):
        """
        Computes the time derivative of shape target requests based on measured values,
        reference trajectories, and control gains. It blends feedforward and feedback
        contributions using a time-varying blend factor, and applies damping to the error
        signal.

        This function re-formulates "run_control_PI_with_scaled_out_damping" to not include a
        damping term.

        Parameters
        ----------
        t : float
            Current time [s].
        dt : float
            Time step [s].
        T_meas : np.ndarray
            Measured values of the shape targets at the current time [m].
        T_err_prev : np.ndarray
            Previously filtered error signal [m].
        T_hist_prev : np.ndarray
            Previous integral term (used for PI control) [m.s].

        Returns
        -------
        dT_dt : np.ndarray
            Time derivative of the shape target requests [m/s].
        T_err : np.ndarray
            Filtered error signal at the current time [m].
        T_hist : np.ndarray
            Updated integral term for use in the next control step [m.s].

        """

        # extract data
        T_ref = self.extract_values(t=t, targets=self.ctrl_targets, key="ref")
        T_ff_deriv = self.extract_values(
            t=t, targets=self.ctrl_targets, key="ff", deriv=True
        )
        T_blend = self.extract_values(t=t, targets=self.ctrl_targets, key="blend")
        k_prop = self.extract_values(t=t, targets=self.ctrl_targets, key="k_prop")
        alpha_inv = 1.0 / self.extract_values(
            t=t, targets=self.ctrl_targets, key="damping"
        )

        # build PID gains to match damping
        beta = 1 - alpha_inv
        abs_beta = np.abs(beta)

        k_int = alpha_inv * (1 + beta) / (1e-4)
        k_deriv = (abs_beta * k_int * dt - beta) * dt
        k_prop_new = 1 - k_int * dt / 2 - k_deriv / dt

        # rescale
        k_int *= k_prop * alpha_inv
        k_deriv *= k_prop * alpha_inv
        k_prop = k_prop_new * k_prop * alpha_inv

        # proportional term
        T_err = T_ref - T_meas

        # integral term
        T_int = abs_beta ** (dt / 1e-4) * T_hist_prev + (0.5 * T_err * dt)

        # derivative term
        T_deriv = (T_err - T_err_prev) / dt

        # FB term
        T_fb_deriv = PID(
            error_prop=T_err,
            error_int=T_int,
            error_deriv=T_deriv,
            k_prop=k_prop,
            k_int=k_int,
            k_deriv=k_deriv,
        )

        # time deriv of shape target requests
        dT_dt = ((T_blend * T_fb_deriv) + ((1.0 - T_blend) * T_ff_deriv)).squeeze()

        # update hist
        T_hist = T_int + (0.5 * T_err * dt)

        return dT_dt.squeeze(), T_err.squeeze(), T_hist.squeeze()

    def run_control_PID(
        self,
        t,
        dt,
        T_meas,
        T_err_prev,
        T_hist_prev,
    ):
        """
        Computes the time derivative of shape target requests based on measured values,
        reference trajectories, and control gains. It blends feedforward and feedback
        contributions using a time-varying blend factor.

        Parameters
        ----------
        t : float
            Current time [s].
        dt : float
            Time step [s].
        T_meas : np.ndarray
            Measured values of the shape targets at the current time [m].
        T_err_prev : np.ndarray
            Previously filtered error signal [m].
        T_hist_prev : np.ndarray
            Previous integral term (used for PI control) [m.s].

        Returns
        -------
        dT_dt : np.ndarray
            Time derivative of the shape target requests [m/s].
        T_err : np.ndarray
            Filtered error signal at the current time [m].
        T_hist : np.ndarray
            Updated integral term for use in the next control step [m.s].

        Notes
        -----
        - The integral term is updated using trapezoidal integration.
        - The final output blends feedforward and feedback derivatives based on a dynamic blend factor.
        - THIS FUNCTION IS UNTESTED.
        """

        # extract data
        T_ref = self.extract_values(t=t, targets=self.ctrl_targets, key="ref")
        T_ff_deriv = self.extract_values(
            t=t, targets=self.ctrl_targets, key="ff", deriv=True
        )
        T_blend = self.extract_values(t=t, targets=self.ctrl_targets, key="blend")
        k_prop = self.extract_values(t=t, targets=self.ctrl_targets, key="k_prop")
        k_int = self.extract_values(t=t, targets=self.ctrl_targets, key="k_int")
        k_deriv = self.extract_values(t=t, targets=self.ctrl_targets, key="k_deriv")

        # proportional term
        T_err = T_ref - T_meas

        # integral term
        T_int = T_hist_prev + (0.5 * T_err * dt)

        # derivative term
        T_deriv = (T_err - T_err_prev) / dt

        # FB term
        T_fb_deriv = PID(
            error_prop=T_err,
            error_int=T_int,
            error_deriv=T_deriv,
            k_prop=k_prop,
            k_int=k_int,
            k_deriv=k_deriv,
        )

        # time deriv of shape target requests
        dT_dt = ((T_blend * T_fb_deriv) + ((1.0 - T_blend) * T_ff_deriv)).squeeze()

        # update hist
        T_hist = T_hist_prev + (T_err * dt)

        return dT_dt.squeeze(), T_err.squeeze(), T_hist.squeeze()

    def extract_values(
        self,
        t,
        targets,
        key,
        deriv=False,
    ):
        """
        Extracts interpolated values or their derivatives for specified shape targets at a given time.

        This method queries the stored interpolation functions for each target and key, returning either
        the interpolated value or its first derivative depending on the `deriv` flag.

        Parameters
        ----------
        t : float
            Time at which to evaluate the interpolants [s].
        targets : list of str
            List of shape target names. Each must correspond to a key in `self.interpolants`.
        key : str
            The waveform name (e.g., 'ff', 'ref', 'blend', 'k_prop', etc.) used to select the interpolant.
        deriv : bool, optional
            If True, returns the first derivative of the interpolant at time `t`. Default is False.

        Returns
        -------
        np.ndarray
            Array of interpolated values (or derivatives) for each target at time `t`.

        Notes
        -----
        - Assumes that `self.interpolants[target][key]` is a valid `scipy.interpolate` object.
        - If `deriv=True`, the method calls `.derivative()` on the interpolant before evaluation.
        """

        if deriv:
            return np.array(
                [self.interpolants[target][key].derivative()(t) for target in targets]
            )
        else:
            return np.array([self.interpolants[target][key](t) for target in targets])

    def plot_data(self, targ, tmin=-1.0, tmax=1.0, nt=1001):
        """
        Visualizes interpolated control waveforms and corresponding raw inputs for a specified
        shape target.

        This method generates subplots for each control waveform (both spline and step types),
        showing the interpolated time series alongside the original data points. It helps verify
        the quality and behavior of the interpolation.

        Parameters
        ----------
        targ : str
            The name of the shape target waveforms to plot. Must be a key in `self.interpolants`
            and `self.data`.
        tmin : float, optional
            Start time for the evaluation grid (default is -1.0 seconds).
        tmax : float, optional
            End time for the evaluation grid (default is 1.0 seconds).
        nt : int, optional
            Number of time points to evaluate the interpolants over the interval [tmin, tmax]
            (default is 10001).

        Notes
        -----
        - Each subplot corresponds to a control parameter (e.g., 'ff', 'ref', 'blend', 'k_prop', etc.).
        - Interpolated curves are plotted in navy; raw data points are shown in red.
        - Axis labels include units where applicable.
        - Useful for debugging or validating the interpolation quality.
        """

        # times to plot at
        t = np.linspace(tmin, tmax, nt)
        nplots = len(self.keys_to_spline + self.keys_to_step)  # number of plots

        # start plotting
        fig, axes = plt.subplots(nplots, 1, figsize=(6, 2.5 * nplots), sharex=True)

        if nplots == 1:
            axes = [axes]

        for ax, key in zip(axes, self.keys_to_spline + self.keys_to_step):
            ax.scatter(
                self.data[targ][key]["times"],
                self.data[targ][key]["vals"],
                s=10,
                marker="x",
                color="tab:orange",
                label=f"raw data",
            )
            ax.plot(
                t,
                self.interpolants[targ][key](t),
                color="navy",
                linewidth=1.2,
                label="interpolated",
            )
            ax.grid(True, linestyle="--", alpha=0.6)

            if key in ["ref", "ff"]:
                ax.set_ylabel(rf"{key} [$m$]")
            elif key == "k_prop":
                ax.set_ylabel(rf"{key} [$1/s$]")
            elif key == "k_int":
                ax.set_ylabel(rf"{key} [$1/s^2$]")
            else:
                ax.set_ylabel(key)

            # y-scaling inside the window
            times = np.array(self.data[targ][key]["times"])
            mask = (times >= tmin) & (times <= tmax)
            if np.any(mask):
                ydata = np.concatenate(
                    [
                        self.interpolants[targ][key](t),
                        np.array(self.data[targ][key]["vals"])[mask],
                    ]
                )
                ymin, ymax = np.min(ydata), np.max(ydata)
                yrange = ymax - ymin
                if yrange == 0:
                    yrange = 1.0
                ax.set_ylim(ymin - 0.02 * yrange, ymax + 0.02 * yrange)

        fig.suptitle(targ)
        axes[0].legend(loc="best")
        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
