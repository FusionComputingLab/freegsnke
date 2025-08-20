"""
Module to implement PF control in FreeGSNKE control loops. 

"""

import matplotlib.pyplot as plt
import numpy as np

from freegsnke.control_loop.useful_functions import (
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


class PFController:
    """
    A controller class for managing coil resistances, inductances, gains, voltage limits, and
    voltage ramp rate limits.

    Parameters
    ----------
    data : dict
        A nested dictionary containing control waveforms for the PF controller.
        The required keys
        for both spline-based and step-based waveforms are:
            - Spline keys:
            - Step keys: "R_matrix", "M_FF_matrix", "M_FB_matrix", "coil_gains",
            "coil_voltage_lims", "coil_voltage_slew_lims"
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
        self.keys_to_spline = []
        self.keys_to_step = [
            "R_matrix",
            "M_FF_matrix",
            "M_FB_matrix",
            "coil_gains",
            "coil_voltage_lims",
            "coil_voltage_slew_lims",
        ]
        for key in self.keys_to_spline + self.keys_to_step:
            check_data_entry(data=data, key=key, controller_name="PFController")

        # create an internal copy of the data
        self.data = data

        # create a dictionary to store the spline functions
        self.interpolants = {}

        # interpolate the input data
        for key in self.keys_to_step:
            self.interpolants[key] = interpolate_step(self.data[key])

    def run_control(
        self,
        t,
        dt,
        I_meas,
        I_approved,
        dI_dt_approved,
        V_approved_prev,
        verbose=False,
    ):
        """
        Computes the approved coil voltage commands based on measured and approved currents,
        while enforcing voltage and slew rate constraints.

        This method calculates the total voltage demand using resistive, feedforward, and
        feedback components. It then clips the voltage according to hardware limits and
        applies slew rate constraints to ensure smooth transitions between time steps.

        Parameters
        ----------
        t : float
            Current time [s].

        dt : float
            Time step [s].

        I_meas : numpy.ndarray
            Measured coil currents at the current time step [A].

        I_approved : numpy.ndarray
            Approved coil currents after applying perturbations and clipping [A].

        dI_dt_approved : numpy.ndarray
            Approved coil current derivatives after clipping [A/s].

        V_approved_prev : numpy.ndarray
            Previously approved coil voltages from the last control step [V].

        verbose : bool, optional
            If True, prints diagnostic information about voltage clipping and slew rate limiting.

        Returns
        -------
        V_approved : numpy.ndarray
            Final coil voltage demands after applying all constraints [V].
        """

        # extract interpolated data
        R = self.interpolants["R_matrix"](t)
        M_FF = self.interpolants["M_FF_matrix"](t)
        M_FB = self.interpolants["M_FB_matrix"](t)
        coil_gains = self.interpolants["coil_gains"](t)
        voltage_clips = self.interpolants["coil_voltage_lims"](t)
        slew_rates = self.interpolants["coil_voltage_slew_lims"](t)

        # resistive voltages
        v_res = R * I_meas

        # FF voltages
        v_FF = M_FF @ dI_dt_approved

        # FB voltages
        delta_I = I_approved - I_meas
        v_FB = M_FB @ (delta_I / coil_gains)

        # initial voltage demands (pre-clipping)
        v_init = v_res + v_FF + v_FB

        # clip voltage to max/min allowed
        v_clipped = np.clip(v_init, -voltage_clips, voltage_clips)

        # apply slew rate constraints
        delta_voltages = v_clipped - V_approved_prev
        max_delta = slew_rates * dt
        delta_clipped = np.clip(delta_voltages, -max_delta, max_delta)
        V_approved = V_approved_prev + delta_clipped

        return V_approved.squeeze()

    def extract_values(
        self,
        t,
        targets,
    ):
        """
        Extracts interpolated values for specified shape targets at a given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the interpolants [s].
        targets : list of str
            List of keys. Each must correspond to a key in `self.interpolants`.

        Returns
        -------
        np.ndarray
            Array of interpolated values (or derivatives) for each target at time `t`.

        Notes
        -----
        - Assumes that `self.interpolants[target]` is a valid `scipy.interpolate` object.
        """

        return np.array([self.interpolants[target](t) for target in targets])

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
        nplots = len(self.keys_to_step[3:6])  # number of plots

        # start plotting
        fig, axes = plt.subplots(nplots, 1, figsize=(10, 2.5 * nplots), sharex=True)

        if nplots == 1:
            axes = [axes]

        for ax, key in zip(axes, self.keys_to_step[3:6]):
            ax.plot(
                t,
                self.interpolants[key](t),
                color="navy",
                linewidth=0.8,
                label="interpolated",
            )

            times = np.asarray(self.data[key]["times"])
            vals_list = self.data[key]["vals"]

            if np.isscalar(vals_list[0]):
                ax.scatter(
                    self.data[key]["times"],
                    self.data[key]["vals"],
                    s=3,
                    marker=".",
                    color="red",
                    label=f"raw data",
                )
            else:
                m = len(vals_list[0])
                times_repeated = np.repeat(times, m)
                vals_flat = np.concatenate(vals_list)
                ax.scatter(
                    times_repeated,
                    vals_flat,
                    s=3,
                    marker=".",
                    color="red",
                    label="raw data",
                )

            ax.grid(True, linestyle="--", alpha=0.6)

            if key == "coil_gains":
                ax.set_ylabel(rf"{key} [$s$]")
            elif key == "coil_voltage_lims":
                ax.set_ylabel(rf"{key} [$V$]")
            elif key == "coil_voltage_slew_lims":
                ax.set_ylabel(rf"{key} [$V/s$]")
            else:
                ax.set_ylabel(key)

        # axes[0].legend(loc='best')
        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
