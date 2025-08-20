"""
Module to implement virtual circuits control in FreeGSNKE control loops. 

"""

import matplotlib.pyplot as plt
import numpy as np

from freegsnke.control_loop.useful_functions import (
    check_data_entry,
    interpolate_spline,
    interpolate_step,
)


class VirtualCircuitsController:
    """
    A controller class for managing virtual circuit control matrices and coil current reference
    waveforms.

    This class supports both spline-based and step-based interpolation of control signals
    for coils and plasma shaping targets. It optionally integrates with an emulated virtual
    circuit provider for enhanced control capabilities.

    Parameters
    ----------
    data : dict
        A nested dictionary containing control waveforms for each target to be controlled.
        Each target's dictionary must include keys for both spline-based and step-based parameters:
            - Spline keys: typically of the form '<coil>_ref'
            - Step keys: typically shape target and plasma target names
        Each key should map to a dictionary suitable for interpolation, with keys:
            - 'times': 1D array of time points
            - 'vals': 1D array of values at those time points (same length).

    ctrl_coils : list of str
        The list of active coils being controlled.

    ctrl_targets : list of str
        The list of shape targets being managed.

    plasma_target : list of str
        The list of plasma targets being managed.

    emu_vc_provider : object, optional
        An optional provider object for emulated virtual circuits. If provided, it enables
        enhanced control capabilities and diagnostics.
    """

    def __init__(
        self,
        data,
        ctrl_coils,
        ctrl_targets,
        plasma_target,
        emu_vc_provider=None,
    ):

        # coils list
        self.ctrl_coils = ctrl_coils

        # targets list
        self.ctrl_targets = ctrl_targets
        self.plasma_target = plasma_target

        # check correct data is input and in correct format
        self.keys_to_spline = [coil + "_ref" for coil in self.ctrl_coils]
        self.keys_to_step = self.ctrl_targets + self.plasma_target
        for key in self.keys_to_spline + self.keys_to_step:
            check_data_entry(
                data=data, key=key, controller_name="VirtualCircuitsController"
            )

        # create an internal copy of the data
        self.data = data

        # create a dictionary to store the spline functions
        self.interpolants = {}

        # interpolate the input data
        for key in self.keys_to_spline:
            self.interpolants[key] = interpolate_spline(self.data[key])
        for key in self.keys_to_step:
            self.interpolants[key] = interpolate_step(self.data[key])

        self.emu_vc_provider = emu_vc_provider
        if self.emu_vc_provider is not None:
            print("Emulated virtual circuits have been provided.")

    def run_control(
        self,
        t,
        dt,
        dip_dt,
        dT_dt,
        I_approved_prev,
        emu_flag=False,
    ):
        """
        Computes the unapproved coil currents and their rates of change based on feedforward
        coil current references and virtual circuit transformations.

        This method extracts coil current reference derivatives, applies virtual circuit matrices
        (either from an emulator or interpolated data), and computes the unapproved coil
        current updates using Euler integration.

        Parameters
        ----------
        t : float
            Current time at which control values are evaluated [s].

        dt : float
            Time step for Euler integration [s].

        dip_dt : float
            Time derivative of the requested plasma current [A/s].

        dT_dt : np.ndarray
            Time derivative of the shape target requests [m/s].

        I_approved_prev : numpy.ndarray
            Previously approved coil currents [A].

        emu_flag : bool, optional
            If True, use the emulated virtual circuit provider to obtain the VC matrix.
            If False, use interpolated VC data from input.

        Returns
        -------
        I_unapproved : numpy.ndarray
            Coil currents (not yet approved), computed via Euler integration [A].

        dI_dt_unapproved : numpy.ndarray
            Rate of change of coil currents (not yet approved) [A/s].
        """

        # extract (feedforward) current references
        dI_dt_ref = self.extract_values(
            t=t, targets=[coil + "_ref" for coil in self.ctrl_coils], deriv=True
        )

        # extract VC matrix (targets x coils)
        # get shape vc
        if emu_flag == True:
            assert (
                self.emu_vc_provider is not None
            ), "Need to provide an emulator VC provider to the class"
            VC_shape = self.emu_vc_provider.get_vc(
                targets=self.ctrl_targets, coils=self.ctrl_coils
            )
        else:
            VC_shape = self.extract_values(t=t, targets=self.ctrl_targets)

        # get plasma vc
        VC_plas = self.extract_values(t=t, targets=self.plasma_target)

        # unapproved coil currents rates of change
        dI_dt_unapproved = dI_dt_ref + (dT_dt @ VC_shape) + (dip_dt * VC_plas)

        # unapproved coil currents (by simple Euler integration)
        I_unapproved = I_approved_prev + (dI_dt_unapproved * dt)

        return I_unapproved.squeeze(), dI_dt_unapproved.squeeze()

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
                [self.interpolants[target].derivative(n=1)(t) for target in targets]
            )
        else:
            return np.array([self.interpolants[target](t) for target in targets])

    def plot_data(self, tmin=-1.0, tmax=1.0, nt=10001):
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
            Number of time points to evaluate the interpolants over the interval [tmin, tmax] (default is 10001).

        Notes
        -----
        - Each subplot corresponds to a control waveform (e.g., '<coil>_ref').
        - Interpolated curves are plotted in navy; raw data points are shown in red.
        - Axis labels include units where applicable.
        - Useful for debugging or validating the interpolation quality.
        """

        # times to plot at
        t = np.linspace(tmin, tmax, nt)
        nplots = len(self.keys_to_spline)  # number of plots

        # start plotting
        fig, axes = plt.subplots(nplots, 1, figsize=(10, 2.5 * nplots), sharex=True)

        if nplots == 1:
            axes = [axes]

        for ax, key in zip(axes, self.keys_to_spline):
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

            if key[-3:] == "ref":
                ax.set_ylabel(rf"{key} [$A$]")
            else:
                ax.set_ylabel(key)

        axes[0].legend(loc="best")
        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
