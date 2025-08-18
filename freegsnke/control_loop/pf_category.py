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
        Compute the coil voltage demands for the current control loop step.

        This method implements a control loop with resistive, feedforward (FF),
        and feedback (FB) voltage components, applies voltage limits and slew rate
        constraints, and returns the final approved voltage demands.

        Parameters
        ----------
        t : float
            Current time (used to interpolate time-dependent system matrices).
        dt : float
            Time step between the current and previous voltage demands, in seconds.
        I_meas : np.ndarray
            Measured coil currents at time `t`, in Amps.
        I_approved : np.ndarray
            Approved coil currents (from system controller), in Amps.
        dI_dt_approved : np.ndarray
            Approved rate of change of coil currents (from system controller), in Amps/sec.
        V_approved_prev : np.ndarray
            Previously approved coil voltage demands, in Volts.
        verbose : bool, optional
            If True, print detailed diagnostic output.

        Returns
        -------
        V_approved : np.ndarray
            Final voltage demand to apply to the active coils, in Volts.
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
