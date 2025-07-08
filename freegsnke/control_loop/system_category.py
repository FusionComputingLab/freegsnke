"""
Module to clip if necessary the currents and changes in currents during a
tokamak shot.

"""

import numpy as np
from .target_scheduler import TargetScheduler


class SystemCategory:
    """
    This class simulates the System category in the PCS
    """

    def __init__(self, limits):
        """
        Initialises the SystemCategory class.

        """
        self.limits = limits

    def apply_system(self):
        """
        Apply the limits to the currents and change rates.

        """

    def clip_currents(self, currs):
        """
        Clip the currents

        Parameters
        ----------
        currs :

        Returns
        -------

        """

        for coil in self.limits["currents"].keys():
            lower = self.limits["currents"][coil]["min"]
            upper = self.limits["currents"][coil]["max"]
            values = currs[coil.lower()]["vals"]

            clamped = [min(max(val, lower), upper) for val in values]
            currs[coil.lower()]["vals"] = clamped

        return dict(sorted(currs.items()))

    def clip_deltas(self, deltas):
        """
        Clip the trajectories (deltas)

        Parameters
        ----------
        deltas :

        Returns
        -------

        """

        for coil in self.limits["deltas"].keys():
            upper = self.limits["deltas"][coil]["max"]
            lower = -self.limits["deltas"][coil]["max"]
            values = deltas[coil.lower()]["vals"]

            clamped = [min(max(val, lower), upper) for val in values]
            deltas[coil.lower()]["vals"] = clamped

        return dict(sorted(deltas.items()))
