"""
Module to implement systems control in FreeGSNKE control loops. 

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
  
You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from freegsnke.control_loop.useful_functions import (
    Waveform,
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

    interpolant_derivatives : dict
        Derivatives of the coil-perturbation spline interpolants, rebuilt whenever
        `update_interpolants` is called and reused during control steps.

    """

    def __init__(
        self,
        data: dict[str, Waveform],
        ctrl_coils: list[str],
    ) -> None:
        """
        Initialise the systems controller.

        Validates that per-coil perturbation data and coil current/ramp
        limits are present in `data`, stores a reference to the data, and
        builds the spline/step interpolants used to evaluate them at
        arbitrary times.

        Parameters
        ----------
        data : dict
            Dictionary of time-series entries. Must contain the following
            keys, each in the format expected by `check_data_entry` (i.e.
            containing 'times' and 'vals' arrays of matching length):

            - "<coil>_pert" for each coil in `ctrl_coils` : perturbation
            signal for that coil.
            - "min_coil_curr_lims" : minimum coil current limits.
            - "max_coil_curr_lims" : maximum coil current limits.
            - "max_coil_curr_ramp_lims" : maximum coil current ramp-rate
            limits.
        ctrl_coils : list of str
            Names of the coils controlled by this controller. Determines
            which "<coil>_pert" keys are required in `data`.

        Attributes
        ----------
        ctrl_coils : list of str
            Stored copy of `ctrl_coils`.
        keys_to_spline : list of str
            Data keys that will be spline-interpolated: one "<coil>_pert"
            entry per coil in `ctrl_coils`.
        keys_to_step : list of str
            Data keys that will be step-interpolated: "min_coil_curr_lims",
            "max_coil_curr_lims", and "max_coil_curr_ramp_lims".
        data : dict
            Internal reference to the input `data`.

        Raises
        ------
        ValueError
            If a required key is missing from `data` or is not in the
            expected format, as enforced by `check_data_entry`.

        Notes
        -----
        Calls `update_interpolants` at the end of initialisation to build
        the interpolating functions from `data`.
        """
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

    def update_interpolants(self) -> None:
        """
        Recompute all interpolant functions from the current `self.data`.

        This method rebuilds `self.interpolants` by applying either
        `interpolate_spline` or `interpolate_step`, and rebuilds the cached
        derivatives of every spline interpolant.

        """

        # create dictionaries to store the interpolants and spline derivatives
        self.interpolants = {}
        self.interpolant_derivatives = {}

        # interpolate the input data
        for key in self.keys_to_spline:
            self.interpolants[key] = interpolate_spline(self.data[key])
            self.interpolant_derivatives[key] = self.interpolants[key].derivative(n=1)
        for key in self.keys_to_step:
            self.interpolants[key] = interpolate_step(self.data[key])

    def run_control(
        self,
        t: float,
        dt: float,
        I_unapproved: np.ndarray,
        dI_dt_unapproved: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        t: float,
        targets: list[str],
        deriv: bool = False,
    ) -> np.ndarray:
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
        - Spline derivatives are constructed by `update_interpolants` and reused here.
        """

        if deriv:
            return np.array(
                [
                    self.interpolant_derivatives[target + "_pert"](t)
                    for target in targets
                ]
            )
        else:
            return np.array(
                [self.interpolants[target + "_pert"](t) for target in targets]
            )

    def plot_data(self, tmin: float = -1.0, tmax: float = 1.0, nt: int = 1001) -> None:
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
                FF_reference = self.interpolant_derivatives[key](t)
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
            elif key in ["min_coil_curr_lims", "max_coil_curr_lims"]:
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
