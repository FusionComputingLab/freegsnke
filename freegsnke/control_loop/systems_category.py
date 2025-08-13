"""
Module to implement systems control in FreeGSNKE control loops. 

"""

import matplotlib.pyplot as plt
import numpy as np

from freegsnke.control_loop.useful_functions import (
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


class SystemsController:
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
        ctrl_coils,
    ):

        # coils list
        self.ctrl_coils = ctrl_coils

        # check correct data is input and in correct format
        self.keys_to_spline = [coil + "_pert" for coil in self.ctrl_coils]
        self.keys_to_step = [
            "min_coil_curr_lims",
            "max_coil_curr_lims",
            "max_coil_curr_ramp_lims",
        ]
        for key in self.keys_to_spline + self.keys_to_step:
            check_data_entry(data=data, key=key, controller_name="SystemsController")

        # create an internal copy of the data
        self.data = data

        # create a dictionary to store the spline functions
        self.interpolants = {}

        # interpolate the input data
        for key in self.keys_to_spline:
            self.interpolants[key] = interpolate_spline(self.data[key])
        for key in self.keys_to_step:
            self.interpolants[key] = interpolate_step(self.data[key])

    def run_control(self, t, dt, I_unapproved, dI_dt_unapproved, verbose=False):
        """
        FIX.

        Calculate the output voltage to apply on the coils, as prescribed in
        the PF category of the PCS.

        Parameters
        ----------
        I_unapproved: numpy 1D array
            Coil currents (provided by the "circuits" category of the control loops). Input in Amps.
        dI_dt_unapproved : numpy 1D array
            Coil current rates of change ("deltas") (provided by the "circuits" category of the control
            loops). Input in Amps.
        dt : float
            Time step. Input in seconds.
        verbose : bool
            Print some output if True.

        Returns
        -------
        I_approved : numpy 1D array
            Approved coil currents in the active coils. Input in A.
        dI_dt_approved: numpy 1D array
            Approved rate of change of the coil currents in the active coils. Input in A.

        """

        # extract coil current perturbations
        dI_pert_dt = self.extract_values(t=t, targets=self.ctrl_coils, deriv=True)

        # add perturbations
        I_perturbed = I_unapproved + dI_pert_dt * dt
        dI_dt_perturbed = dI_dt_unapproved + dI_pert_dt

        # extract coil current limits and ramp rate limits
        min_coil_curr_lims = self.interpolants["min_coil_curr_lims"](t)
        max_coil_curr_lims = self.interpolants["max_coil_curr_lims"](t)
        max_coil_curr_ramp_lims = self.interpolants["max_coil_curr_ramp_lims"](t)

        # apply the clipping
        I_approved = np.clip(I_perturbed, min_coil_curr_lims, max_coil_curr_lims)
        dI_dt_approved = np.clip(
            dI_dt_perturbed, -max_coil_curr_ramp_lims, max_coil_curr_ramp_lims
        )

        # print if required
        if verbose:
            print("---")

            if not np.allclose(I_approved, I_perturbed):
                print("    Coil currents clipped (according to `min/max_coil_limits`).")

            if not np.allclose(dI_dt_approved, dI_dt_perturbed):
                print(
                    "    Coil current deltas clipped (according to `max_coil_delta_limits`)."
                )

            print(f"    Approved coil currents = {I_approved}")
            print(f"    Approved delta coil currents = {dI_dt_approved}")

        return I_approved, dI_dt_approved

    def extract_values(
        self,
        t,
        targets,
        deriv=False,
    ):
        """
        Evaluate and extract interpolated values at a given time for specified targets.

        Parameters
        ----------
        t : float
            The time at which to evaluate the interpolants.
        targets : list of str
            A list of target names corresponding to keys in `self.interpolants`.
        deriv : bool
            Returns first derivative of the interpolant if True.

        Returns
        -------
        np.ndarray
            An array of interpolated values evaluated at time `t`, one for each target.
        """

        if deriv:
            return np.array(
                [
                    self.interpolants[target + "_pert"].derivative(n=1)(t)
                    for target in targets
                ]
            )
        else:
            return np.array(
                [self.interpolants[target + "_pert"](t) for target in targets]
            )

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

            if key[-4:] == "pert":
                ax.set_ylabel(rf"{key} [$A$]")
            elif key in ["min_coil_curr_lims", "min_coil_curr_lims"]:
                ax.set_ylabel(rf"{key} [$A$]")
            elif key == "max_coil_curr_ramp_lims":
                ax.set_ylabel(rf"{key} [$A/s$]")
            else:
                ax.set_ylabel(key)

        axes[0].legend(loc="best")
        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
