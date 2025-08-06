"""
Module to implement systems control in FreeGSNKE control loops. 

"""

import numpy as np

from freegsnke.control_loop.useful_functions import interpolate_spline, interpolate_step


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

        # create an internal copy of the data
        self.data = data

        # coils list
        self.ctrl_coils = ctrl_coils

        # create a dictionary to store the spline functions
        self.interpolants = {}

        # interpolate the input data
        for key in self.ctrl_coils:
            self.interpolants[key] = interpolate_spline(self.data[key])
        for key in [
            "min_coil_curr_lims",
            "max_coil_curr_lims",
            "max_coil_curr_ramp_lims",
        ]:
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
        I_pert = self.extract_values(t=t, targets=self.ctrl_coils, derivative=False)
        dI_pert_dt = self.extract_values(t=t, targets=self.ctrl_coils, derivative=True)

        # add perturbations
        I_perturbed = I_unapproved + I_pert
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
        derivative=False,
    ):
        """
        Evaluate and extract interpolated values at a given time for specified targets.

        Parameters
        ----------
        t : float
            The time at which to evaluate the interpolants.
        targets : list of str
            A list of target names corresponding to keys in `self.interpolants`.
        derivative : bool
            If True, evaluates the first derivative of the interpolated function.

        Returns
        -------
        np.ndarray
            An array of interpolated values evaluated at time `t`, one for each target.
        """

        if derivative:
            return np.array(
                [self.interpolants[target].derivative()(t) for target in targets]
            )
        else:
            return np.array([self.interpolants[target](t) for target in targets])
