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
            print("Emulated VC's provided")

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
        NEED TO UPDATE.


        Parameters
        ----------
        - Kp : float
            Proportional term used in the Vloop_fb computation.


        Returns
        -------
        - dI_dt : 1D numpy array
            Array of delta currents requests that will be part of the input of
            Circuits category.

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
                [self.interpolants[target].derivative(n=1)(t) for target in targets]
            )
        else:
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
