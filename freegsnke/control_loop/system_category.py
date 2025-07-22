"""
Module to clip if necessary the currents and changes in currents during a
tokamak shot.

"""

import numpy as np

# from .target_scheduler import TargetScheduler


def system_approved_currents(
    circuits_I,
    circuits_dIdt,
    icoil_perts,
    min_coil_limits,
    max_coil_limits,
    max_coil_delta_limits,
    dt,
    verbose=False,
):
    """
    Calculate the output voltage to apply on the coils, as prescribed in
    the PF category of the PCS.

    Parameters
    ----------
    circuits_I: numpy 1D array
        Coil currents (provided by the "circuits" category of the control loops). Input in Amps.
    circuits_dIdt : numpy 1D array
        Coil current rates of change ("deltas") (provided by the "circuits" category of the control
        loops). Input in Amps.
    icoil_perts : numpy 1D array
        Perturbations to be applied to the coil currents and deltas. Input in Amps.
    min_coil_limits : numpy 1D array
        Coil currents are clipped if the they are below these values for each of the active
        coils. Input in Amps.
    max_coil_limits : numpy 1D array
        Coil currents are clipped if the they are above these values for each of the active
        coils. Input in Amps.
    max_coil_delta_limits : numpy 1D array
        Coil current deltas are clipped if the they are outside +/- these values for each of the
        active coils. Input in Amps/second.
    dt : float
        Time step. Input in seconds.
    verbose : bool
        Print some output (True) or not (False).

    Returns
    -------
    approved_I : numpy 1D array
        Approved coil currents in the active coils. Input in A.
    approved_dIdt: numpy 1D array
        Approved rate of change of the coil currents in the active coils. Input in A.

    """

    # add perturbations
    perturbed_I = circuits_I + icoil_perts * dt
    perturbed_dIdt = circuits_dIdt + icoil_perts

    # apply the clipping
    approved_I = np.clip(perturbed_I, min_coil_limits, max_coil_limits)
    approved_dIdt = np.clip(
        perturbed_dIdt, -max_coil_delta_limits, max_coil_delta_limits
    )

    # print if required
    if verbose:
        print("---")

        if not np.allclose(approved_I, perturbed_I):
            print("    Coil currents clipped (according to `min/max_coil_limits`).")

        if not np.allclose(approved_dIdt, perturbed_dIdt):
            print(
                "    Coil current deltas clipped (according to `max_coil_delta_limits`)."
            )

        print(f"    Approved coil currents = {approved_I}")
        print(f"    Approved delta coil currents = {approved_dIdt}")

    return approved_I, approved_dIdt


# class SystemCategory:
#     """
#     This class simulates the System category in the PCS
#     """

#     def __init__(self, limits):
#         """
#         Initialises the SystemCategory class.

#         """
#         self.limits = limits

#     def apply_system(self):
#         """
#         Apply the limits to the currents and change rates.

#         """

#     def clip_currents(self, currs):
#         """
#         Clip the currents

#         Parameters
#         ----------
#         currs :

#         Returns
#         -------

#         """

#         for coil in self.limits["currents"].keys():
#             lower = self.limits["currents"][coil]["min"]
#             upper = self.limits["currents"][coil]["max"]
#             values = currs[coil.lower()]["vals"]

#             clamped = [min(max(val, lower), upper) for val in values]
#             currs[coil.lower()]["vals"] = clamped

#         return dict(sorted(currs.items()))

#     def clip_deltas(self, deltas):
#         """
#         Clip the trajectories (deltas)

#         Parameters
#         ----------
#         deltas :

#         Returns
#         -------

#         """

#         for coil in self.limits["deltas"].keys():
#             upper = self.limits["deltas"][coil]["max"]
#             lower = -self.limits["deltas"][coil]["max"]
#             values = deltas[coil.lower()]["vals"]

#             clamped = [min(max(val, lower), upper) for val in values]
#             deltas[coil.lower()]["vals"] = clamped

#         return dict(sorted(deltas.items()))
