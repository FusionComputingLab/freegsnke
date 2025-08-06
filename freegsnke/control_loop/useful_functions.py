"""
Module full of functions required by the PCS in FreeGSNKE.

"""

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


def interpolate_step(
    data,
):
    """
    Creates a step-wise interpolator for time-series data using 'previous' value interpolation.

    Parameters
    ----------
    data : dict
        Dictionary with keys:
        - 'times': 1D array of time points
        - 'vals': 1D array of values at those time points (same length)

    Returns
    -------
    f_interp : function
        Callable function f(t) that returns the step-wise interpolated value at time t.
        For t < min(times), returns the first value.
        For t > max(times), returns the last value.
    """

    times = np.array(data["times"])
    vals = np.stack(data["vals"])

    # build interpolator
    f_interp = interp1d(
        times,
        vals,
        kind="previous",
        axis=0,
        bounds_error=False,
        fill_value=(vals[0], vals[-1]),  # extrapolate for first and last values
    )

    return f_interp


def interpolate_spline(data):
    """
    Creates a spline interpolator for time-series data in 'data'.

    Parameters
    ----------
    data : dict
        Dictionary with keys:
        - 'times': 1D array of time points
        - 'vals': 1D array of values at those time points (same length)

    Returns
    -------
    f_interp : function
        Callable function f(t) that returns the spline interpolated value at time t.
        For t < min(times), returns the first value.
        For t > max(times), returns the last value.
    """

    times = np.array(data["times"])
    vals = np.array(data["vals"])

    # build interpolator
    f_interp = UnivariateSpline(
        times,
        vals,
        s=0,
        ext=3,
    )

    return f_interp
