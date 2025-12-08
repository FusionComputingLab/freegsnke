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

    vc_generator : object, optional
        An optional class object for applying emulated virtual circuits. If not
        provided, deafult waveform-defined VCs will be used.

    vc_update_rate : float, optional
        Optional argument to specify how often, in seconds, new VCs are computed with vc_generator.
        If None provided, defaults to zero and new VC computed at every iterration.

    """

    def __init__(
        self,
        data,
        ctrl_coils,
        ctrl_targets,
        plasma_target,
        vc_generator=None,
        vc_update_rate=None,
    ):

        # coils list
        self.ctrl_coils = ctrl_coils
        self.vc_coil_order = data["coil_order"]
        self.vc_coil_order_index = {
            coil: i for i, coil in enumerate(self.vc_coil_order)
        }

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

        # interpolate the input data
        self.update_interpolants()

        # store emulated VCs class if present
        self.vc_generator = vc_generator
        if vc_update_rate is None:
            vc_update_rate = 0.0
            print(
                "Setting default update rate to zero. New VC computed at every time step"
            )
        self.vc_update_rate = vc_update_rate
        # set holders for most recent vcs
        self.latest_vc_time = None
        self.latest_vc = None

        # store emulated VC's that were used
        self.emulated_vc_list = []
        self.emulated_vc_times = []

        # store first vc matrix
        # t0 = min(data)
        # self.present_shape_vc_matrix = self.extract_values(
        #     t=t, targets=self.ctrl_targets
        # )

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

    def run_control(
        self,
        t,
        dt,
        dip_dt,
        dT_dt,
        I_approved_prev,
        emulated_VC_targets=None,
        emulated_VC_targets_calc=None,
        emulator_coils_calc=None,
        emu_inputs=None,
        verbose=False,
    ):
        """
        Computes the unapproved coil currents and their rates of change based on feedforward
        coil current references and virtual circuit transformations.

        This method extracts coil current reference derivatives, applies virtual circuit matrices
        (either from an emulator or interpolated data), and computes the unapproved coil
        current updates using Euler integration.

        There is also the option to provide VCs from an emulator class object.

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

        emulated_VC_targets : list of str , optional
            List of targets to be controlled using the emulated VC's. Must be subset of
            ctrl_targets, and subset/equal to emulated_VC_targets_calc. Those not defined in this list will be taken from waveform-defined
            VCs.

        emulated_VC_targets_calc : list of str , optional
            List of targets to be used when performing pseudoinverse of jacobian when calculating the emulated VC.

        emulator_coils_calc : list of str, optional
            List of coils to use in emulated VC compuation. These are coils to use in computing shape sensitivity matrix.

        emu_inputs : np.ndarray , optional
            Array of input values for all input parameters (currents and other plasma parameters) of the Neural Network emulator.

        verbose : bool
            Print some output if True.

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

        # extract shape target VCs from waveform data (targets x coils)
        VC_shape = self.extract_values(t=t, targets=self.ctrl_targets)
        if verbose:
            print("VC's from file", VC_shape)

        # extract plasma target VC from waveform data (targets x coils)
        VC_plasma = self.extract_values(t=t, targets=self.plasma_target)

        # if emulated VCs to be used, extract the data and overwrite relevant VC
        # matrix columns
        if (
            (self.vc_generator is not None)
            and (emulated_VC_targets is not None)
            and (emulated_VC_targets_calc is not None)
            and (emulator_coils_calc is not None)
        ):
            print("emulated targets", emulated_VC_targets)
            print("emulated targets_calc", emulated_VC_targets_calc)
            # error checks
            assert (
                self.vc_generator is not None
            ), "Need to provide a VC emulator class to `VirtualCircuitsController`."
            assert (
                emulated_VC_targets is not None
            ), "Need to provide targets for the VC emulator."
            assert (
                emulated_VC_targets_calc is not None
            ), "Need to provide targets for calculation in the VC emulator."

            if self.latest_vc is None:
                # compute first emulated VC
                print("First VC computed")
                VC_shape_emu = self.vc_generator.get_vc(
                    targets=emulated_VC_targets,
                    targets_calc=emulated_VC_targets_calc,
                    coils=self.ctrl_coils,
                    coils_calc=emulator_coils_calc,
                    input_data=emu_inputs,  # This may be temporary and removed at some point.
                )
                # update latest vcs/times
                self.latest_vc_time = 1.0 * t
                self.latest_vc = VC_shape_emu

            delta_t_vc = t - self.latest_vc_time
            if delta_t_vc >= self.vc_update_rate:
                # compute new emulated VCs
                print(f"New VC computed at time {t}")
                VC_shape_emu = self.vc_generator.get_vc(
                    targets=emulated_VC_targets,
                    targets_calc=emulated_VC_targets_calc,
                    coils=self.ctrl_coils,
                    coils_calc=emulator_coils_calc,
                    input_data=emu_inputs,  # This may be temporary and removed at some point.
                )
                # update latest vcs/times
                self.latest_vc_time = 1.0 * t
                self.latest_vc = VC_shape_emu
                # store VC's
                self.emulated_vc_list.append(VC_shape_emu)
                self.emulated_vc_times.append(t)

            else:
                print(f"Using latest VC, computed at time {self.latest_vc_time} ")
                VC_shape_emu = self.latest_vc

            print("latest vc", self.latest_vc)
            if verbose:
                print("Emulated VC matrix", VC_shape_emu)
            # fill appropriate columns from emulated vcs
            ctrl_target_order = {
                target: i for i, target in enumerate(self.ctrl_targets)
            }
            for j, emu_targ in enumerate(emulated_VC_targets):
                # expand array as apropriate
                VC_shape[ctrl_target_order[emu_targ], :] = 1.0 * VC_shape_emu[:, j]

            if verbose:
                print("VCs - hybrid emu and file", VC_shape)
        # unapproved coil currents rates of change
        dI_dt_unapproved = dI_dt_ref + (dT_dt @ VC_shape) + (dip_dt * VC_plasma)

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

    def plot_data_FF_currents(self, tmin=-1.0, tmax=1.0, nt=1001):
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
        - Each subplot corresponds to a control waveform (e.g., '<coil>_ref').
        - Interpolated curves are plotted in navy; raw data points are shown in red.
        - Axis labels include units where applicable.
        - Useful for debugging or validating the interpolation quality.
        """

        # times to plot at
        t = np.linspace(tmin, tmax, nt)
        nplots = len(self.keys_to_spline)  # number of plots

        # start plotting
        fig, axes = plt.subplots(nplots, 1, figsize=(6, 2.5 * nplots), sharex=True)

        if nplots == 1:
            axes = [axes]

        for ax, key in zip(axes, self.keys_to_spline):

            # find out which control is ON and when
            FF_reference = self.interpolants[key](t)
            FF_mask = np.abs(FF_reference) > 0

            # shade region of FF control
            on_regions = np.where(np.diff(FF_mask.astype(int)) != 0)[0] + 1
            segments = np.split(t, on_regions)
            states = np.split(FF_mask, on_regions)

            for seg_t, seg_state in zip(segments, states):
                if np.all(seg_state):  # region fully "on"
                    ax.axvspan(seg_t[0], seg_t[-1], color="yellow", alpha=0.25)

            # raw data
            ax.scatter(
                self.data[key]["times"],
                self.data[key]["vals"],
                s=10,
                marker="x",
                color="tab:orange",
                alpha=0.9,
                label=f"raw data",
            )
            # interpolated data
            ax.plot(
                t,
                self.interpolants[key](t),
                color="navy",
                linewidth=1.2,
                label="interpolated",
            )
            ax.grid(True, linestyle="--", alpha=0.6)

            if key[-3:] == "ref":
                ax.set_ylabel(rf"{key} [$A$]")
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

    def plot_data_VCs(self, tmin=-1.0, tmax=1.0, nt=1001):
        """
        Visualizes virtual circuits times and corresponding raw inputs.

        Parameters
        ----------
        tmin : float, optional
            Start time for the evaluation grid (default is -1.0 seconds).
        tmax : float, optional
            End time for the evaluation grid (default is 1.0 seconds).
        nt : int, optional
            Number of time points to evaluate the interpolants over the interval [tmin, tmax] (default is 1001).

        """

        # times to plot at
        t = np.linspace(tmin, tmax, nt)
        nplots = len(self.keys_to_step)  # number of plots

        # start plotting
        fig, axes = plt.subplots(nplots, 1, figsize=(6, 2.5 * nplots), sharex=True)

        if nplots == 1:
            axes = [axes]

        # Convert each array to a hashable form
        def make_key(arr):
            return tuple(arr.tolist())

        for ax, key in zip(axes, self.keys_to_step):

            # Assign a unique ID to each unique array
            state_ids = []

            next_id = 1
            for arr in self.data[key]["vals"]:

                if np.all(np.abs(arr) < 1e-12):
                    state_ids.append(0)
                else:
                    state_ids.append(next_id)
                    next_id += 1

            state_ids = np.array(state_ids)

            # plot different VC times
            ax.step(
                self.data[key]["times"],
                state_ids,
                where="post",
                color="navy",
                label=key,
            )
            ax.set_yticks(sorted(set(state_ids)))
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_ylabel(f"Unique VC ID")
            ax.legend(loc="best")


        axes[-1].set_xlabel(r"Time [$s$]")
        axes[-1].set_xlim([tmin, tmax])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
