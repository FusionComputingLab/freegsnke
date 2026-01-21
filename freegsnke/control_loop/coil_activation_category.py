"""
Module implementing time‑dependent coil activation scheduling for FreeGSNKE
control loops.

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
        active_coil_resists,
    ):
        """
        Compute effective coil resistances at a given time step.

        This function extracts coil activation values at time ``t`` and scales the
        base resistances accordingly. Coils that are inactive (activation ~ 0)
        are assigned a very large resistance to effectively disable them in the
        control model.

        Parameters
        ----------
        t : float
            Current time at which to evaluate coil activations.
        dt : float
            Time step size (currently unused, but kept for interface consistency).
        active_coil_resists : numpy.ndarray
            Array of active coil resistances when coils are switched on [Ohms].

        Returns
        -------
        numpy.ndarray
            Array of effective coil resistances, where inactive coils are set to
            a large resistance value (``1e12``).
        """

        # extract data
        activations = self.extract_values(t=t, targets=self.active_coils, deriv=False)

        # if coil is not active, set very large resistance
        # final_coil_resists = active_coil_resists + (1.0 - activations) * 1e12
        mask = activations.astype(bool)
        final_coil_resists = active_coil_resists.copy()
        final_coil_resists[~mask] = 1e12

        return final_coil_resists

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
                    self.interpolants[target + "_activation"].derivative(n=1)(t)
                    for target in targets
                ]
            )
        else:
            return np.array(
                [self.interpolants[target + "_activation"](t) for target in targets]
            )

    def plot_data(self, tmin=-1.0, tmax=1.0, nt=1001):
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
            Number of time points to evaluate the interpolants over the interval [tmin, tmax] (default is 1001).

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
        fig, axes = plt.subplots(nplots, 1, figsize=(6, 2.5 * nplots), sharex=True)

        if nplots == 1:
            axes = [axes]

        for ax, key in zip(axes, self.data.keys()):
            ax.scatter(
                self.data[key]["times"],
                self.data[key]["vals"],
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

        fig.suptitle("Coil activation schedule (0 = off, 1 = on)")
        axes[0].legend(loc="best")
        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
