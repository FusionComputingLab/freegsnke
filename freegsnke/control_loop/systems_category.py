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
    A controller class for managing coil current perturbations, coil current limits, and
    coil current ramp rate limits.

    Parameters
    ----------
    data : dict
        A nested dictionary containing control waveforms for the systems controller.
        The required keys for both spline-based and step-based waveforms are:
            - Spline keys: "<coil>_pert"
            - Step keys: "min_coil_curr_lims", "max_coil_curr_lims", "max_coil_curr_ramp_lims"
        Each key should map to a waveform dictionary suitable for interpolation with keys:
            - 'times': 1D array of time points
            - 'vals': 1D array of values at those time points (same length).

    ctrl_coils : list of str
        The list of active coils being controlled.

    Attributes
    ----------
    ctrl_coils : list of str
        The list of active coils being controlled.

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
        for key in self.keys_to_spline:
            self.interpolants[key] = interpolate_spline(self.data[key])
        for key in self.keys_to_step:
            self.interpolants[key] = interpolate_step(self.data[key])

    def run_control(self, t, dt, I_unapproved, dI_dt_unapproved, verbose=False):
        """
        Applies coil current perturbations to unapproved coil currents and enforce coil current
        constraints to produce approved control signals.

        This method adjusts the unapproved coil currents and their rates of change by applying
        time-dependent perturbations, then clips the results according to current and ramp rate
        limits. It returns the final approved coil currents and their derivatives.

        Parameters
        ----------
        t : float
            Current time [s].

        dt : float
            Time step [s].

        I_unapproved : numpy.ndarray
            Coil currents (not yet approved), computed via Euler integration [A].

        dI_dt_unapproved : numpy.ndarray
            Rate of change of coil currents (not yet approved) [A/s].

        verbose : bool, optional
            If True, prints diagnostic messages about clipping and approved values.

        Returns
        -------
        I_approved : numpy.ndarray
            Coil currents (approved) [A].

        dI_dt_approved : numpy.ndarray
            Rate of change of coil currents (approved) [A/s].

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

        return I_approved.squeeze(), dI_dt_approved.squeeze()

    def extract_values(
        self,
        t,
        targets,
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
            List of keys. Each must correspond to a key in `self.interpolants`.
        deriv : bool, optional
            If True, returns the first derivative of the interpolant at time `t`. Default is False.

        Returns
        -------
        np.ndarray
            Array of interpolated values (or derivatives) for each target at time `t`.

        Notes
        -----
        - Assumes that `self.interpolants[target]` is a valid `scipy.interpolate` object.
        - If `deriv=True`, the method calls `.derivative()` on the interpolant before evaluation.
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

    def plot_data(self, tmin=-1.0, tmax=1.0, nt=1001):
        """
        Visualizes interpolated control waveforms and corresponding raw inputs.

        This method generates subplots for each control waveform (spline types),
        showing the interpolated time series alongside the original data points. It helps verify
        the quality and behavior of the interpolation.

        Parameters
        ----------
        tmin : float, optional
            Start time for the evaluation grid (default is -1.0 seconds).
        tmax : float, optional
            End time for the evaluation grid (default is 1.0 seconds).
        nt : int, optional
            Number of time points to evaluate the interpolants over the interval [tmin, tmax] (default is 1001).

        Notes
        -----
        - Each subplot corresponds to a control waveform (e.g., '<coil>_pert').
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

        for ax, key in zip(axes, self.data.keys()):
            times = np.asarray(self.data[key]["times"])
            vals_list = self.data[key]["vals"]

            # find out which control is ON and when
            if key in self.keys_to_spline:
                FF_reference = self.interpolants[key].derivative()(t)
                FF_mask = np.abs(FF_reference) > 0

                # shade region of FF control
                on_regions = np.where(np.diff(FF_mask.astype(int)) != 0)[0] + 1
                segments = np.split(t, on_regions)
                states = np.split(FF_mask, on_regions)

                for seg_t, seg_state in zip(segments, states):
                    if np.all(seg_state):  # region fully "on"
                        if len(seg_t) > 0:
                            ax.axvspan(seg_t[0], seg_t[-1], color="yellow", alpha=0.25)

            if np.isscalar(vals_list[0]):
                ax.scatter(
                    self.data[key]["times"],
                    self.data[key]["vals"],
                    s=10,
                    marker="x",
                    color="tab:orange",
                    alpha=0.9,
                    label=f"raw data",
                )
            else:
                m = len(vals_list[0])
                times_repeated = np.repeat(times, m)
                vals_flat = np.concatenate(vals_list)
                ax.scatter(
                    times_repeated,
                    vals_flat,
                    s=10,
                    marker="x",
                    color="tab:orange",
                    alpha=0.9,
                    label=f"raw data",
                )

            ax.plot(
                t,
                self.interpolants[key](t),
                color="navy",
                linewidth=1.2,
                label="interpolated",
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

            # y-scaling inside the window
            times = np.array(self.data[key]["times"])
            mask = (times >= tmin) & (times <= tmax)
            if np.any(mask):
                ydata = np.concatenate(
                    [self.interpolants[key](t), np.array(self.data[key]["vals"])[mask]]
                )
                ymin, ymax = np.min(ydata), np.max(ydata)
                yrange = ymax - ymin
                if yrange == 0:
                    yrange = 1.0
                ax.set_ylim(ymin - 0.02 * yrange, ymax + 0.02 * yrange)

        axes[0].legend(loc="best")
        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
