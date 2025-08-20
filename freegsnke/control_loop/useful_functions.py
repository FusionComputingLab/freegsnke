"""
Module of functions required by the PCS in FreeGSNKE.

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
        fill_value=(0.0, vals[-1]),  # extrapolate for first and last values
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
        k=1,  # order (linear)
        s=0,  # interpolates points exactly
        ext="zeros",  # extrapolate to zeros outside of boundary points
    )

    return f_interp


def check_data_entry(
    data: dict,
    key: str,
    controller_name: str,
) -> bool:
    """
    Validate that a specified sub-dictionary contains 'times' and 'vals' keys
    of equal length.

    Parameters
    ----------
    data : dict
        A dictionary where each value is expected to be a sub-dictionary
        containing at least 'times' and 'vals'.
    key : str
        The key in `data` corresponding to the sub-dictionary to validate.
    controller_name : str
        A string corresponding to which controller is being checked.

    Returns
    -------
    bool
        True if the checks pass.

    Raises
    ------
    ValueError
        If the specified key is missing from `data`, if 'times' or 'vals'
        is missing from the sub-dictionary, or if 'times' and 'vals'
        are not the same length.
    """

    # key not found
    if key not in data:
        raise ValueError(
            f"{controller_name}: Key '{key}' not found in 'data'. "
            f"Please include {{'times': [], 'vals': []}} for '{key}'."
        )

    subdict = data[key]

    # key found, check for times and values
    for required_key in ["times", "vals"]:
        if required_key not in subdict:
            raise ValueError(
                f"{controller_name}: Missing '{required_key}' in data['{key}']."
            )

    # times and vals found, check equal lengths
    times_len = len(subdict["times"])
    vals_len = len(subdict["vals"])
    if times_len != vals_len:
        raise ValueError(
            f"{controller_name}: Length mismatch in data['{key}']: "
            f"'times' has length {times_len}, 'vals' has length {vals_len}. "
        )
