"""
Module to implement coil activation times in FreeGSNKE control loops. 

"""

import matplotlib.pyplot as plt
import numpy as np

from freegsnke.control_loop.useful_functions import (
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


class CoilActivationController:
    """
    A controller class for managing time-dependent coil activation times.

    Parameters
    ----------
    data : dict
        A nested dictionary containing coil activation waveforms for the controller.
        The required keys
        for both spline-based and step-based waveforms are:
            - Spline keys: "<coil>_activation"
            - Step keys:
        Each key should map to a waveform dictionary suitable for interpolation with keys:
            - 'times': 1D array of time points
            - 'vals': 1D array of values at those time points (same length).

    active_coils : list of str
        The list of active coils being used.

    Attributes
    ----------
    active_coils : list of str
        The list of active coils being used.

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
        active_coils,
    ):

        # coils list
        self.active_coils = active_coils

        # check correct data is input and in correct format
        self.keys_to_spline = []
        self.keys_to_step = [coil + "_activation" for coil in self.active_coils]
        for key in self.keys_to_spline + self.keys_to_step:
            check_data_entry(
                data=data, key=key, controller_name="CoilActivationController"
            )

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

    # def run_control(
    #     self,
    #     t,
    #     dt,
    #     zip_meas,
    #     zipv_meas,
    # ):
    #     """
    #     NEED TO UPDATE.
    #     Calculates the vector of current trajectories ΔI/Δt, as prescribed
    #     in the plasma category of the MAST-U PCS. The equations followed are:

    #     Ip_error = (Ip_req - Ip_obs)
    #     integral = internal_state + 0.5 * Ip_error * dt
    #     internal_state = internal_state + Ip_error * dt
    #     ΔIsol_fb/Δt = Kp * Ip_error + Ki * integral
    #     ΔIsol/Δt = ΔIsol_fb/Δt * blend - Vloop_ff * (1 - blend)/M_sp

    #     It should be noted that the PI controller works at a frequency twice as
    #     high as the data recording system. This is why the PI controller goes
    #     through two cycles in this method.

    #     Parameters
    #     ----------
    #     - Kp : float
    #         Proportional term used in the Vloop_fb computation.

    #     Returns
    #     -------
    #     - dI_dt : 1D numpy array
    #         Array of delta currents requests that will be part of the input of
    #         Circuits category.

    #     """

    #     # extract data
    #     z_ref = self.interpolants["z_ref"](t)
    #     ip_ref = self.interpolants["ip_ref"](t)
    #     k_prop = self.interpolants["k_prop"](t)
    #     k_deriv = self.interpolants["k_deriv"](t)

    #     # return k_prop*(zip_meas - z_ref*ip_ref) + k_deriv*zipv_meas
    #     return k_prop*(z_ref*ip_ref - zip_meas) + k_deriv*zipv_meas

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
            ax.set_ylabel(key)

        fig.suptitle("Coil activation schedule (0 = off, 1 = on)")
        axes[0].legend(loc="best")
        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
