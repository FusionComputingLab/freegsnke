"""
Module of functions required by the PCS in FreeGSNKE.

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

from typing import Any, Optional, Union

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

# a single time-series entry, e.g. {"times": [...], "vals": [...]}
Waveform = dict[str, Any]


class ConstantInterpolant:
    """
    Callable interpolant for a waveform whose value is constant in time.

    It preserves the output shapes of the SciPy interpolants used for
    non-constant waveforms and provides a compatible `derivative` method.
    """

    def __init__(self, value: Any) -> None:
        self.value = np.asarray(value)

    def __call__(self, t: Any) -> np.ndarray:
        """Return the constant value with the same leading shape as ``t``."""

        result = np.broadcast_to(self.value, np.shape(t) + self.value.shape)
        return np.array(result, copy=True)

    def derivative(self, n: int = 1) -> "ConstantInterpolant":
        """Return this interpolant for order zero, otherwise a zero interpolant."""

        if n < 0:
            raise ValueError("Derivative order must be non-negative.")
        if n == 0:
            return self
        return ConstantInterpolant(np.zeros_like(self.value, dtype=float))


# an interpolant produced by `interpolate_step`/`interpolate_spline`: callable at a
# time `t`, and (for splines only) supports `.derivative()`
Interpolant = Union[ConstantInterpolant, interp1d, UnivariateSpline]


def interpolate_step(
    data: Waveform,
) -> Interpolant:
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

    Notes
    -----
    Constant waveforms use `ConstantInterpolant` to avoid repeated SciPy
    interpolation overhead during control-loop execution.
    """

    times = np.array(data["times"])
    vals = np.stack(data["vals"])

    if np.all(vals == vals[0]):
        return ConstantInterpolant(vals[0])

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


def interpolate_spline(data: Waveform) -> Interpolant:
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

    Notes
    -----
    Constant waveforms use `ConstantInterpolant` to avoid repeated SciPy
    interpolation overhead during control-loop execution.
    """

    times = np.array(data["times"])
    vals = np.array(data["vals"])

    if np.all(vals == vals[0]):
        return ConstantInterpolant(vals[0])

    # build interpolator
    f_interp = UnivariateSpline(
        times,
        vals,
        k=1,  # order (linear)
        s=0,  # interpolates points exactly
        ext="const",  # extrapolate to first/last values outside of boundary points
    )

    return f_interp


def check_data_entry(
    data: dict[str, Waveform],
    key: str,
    controller_name: str,
) -> None:
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


def PID(
    error_prop: Optional[Union[float, np.ndarray]] = None,
    error_int: Optional[Union[float, np.ndarray]] = None,
    error_deriv: Optional[Union[float, np.ndarray]] = None,
    k_prop: Optional[Union[float, np.ndarray]] = 0.0,
    k_int: Optional[Union[float, np.ndarray]] = 0.0,
    k_deriv: Optional[Union[float, np.ndarray]] = 0.0,
) -> Union[float, np.ndarray]:
    """
    Compute a flexible PID controller output.

    Any of the P, I, or D components may be omitted. A component
    contributes only if both its error term and its gain are provided
    (i.e. not None); otherwise it contributes zero to the output.

    Parameters
    ----------
    error_prop : float or array_like, optional
        Proportional error term. If None, the P contribution is zero.
    error_int : float or array_like, optional
        Integral error term. If None, the I contribution is zero.
    error_deriv : float or array_like, optional
        Derivative error term. If None, the D contribution is zero.
    k_prop : float or array_like, optional
        Proportional gain. Default is 0. If explicitly set to None, the P
        contribution is zero regardless of error_prop.
    k_int : float or array_like, optional
        Integral gain. Default is 0. If explicitly set to None, the I
        contribution is zero regardless of error_int.
    k_deriv : float or array_like, optional
        Derivative gain. Default is 0. If explicitly set to None, the D
        contribution is zero regardless of error_deriv.

    Returns
    -------
    float or ndarray
        The PID (or PI, PD, P, I, D, or ID) controller output. Arrays must be
        broadcast-compatible if array inputs are used.

    Notes
    -----
    - A component contributes only if both its gain and error term are provided.
    - This function performs no time integration or differentiation; the caller
        must compute error_int and error_deriv externally.
    """

    out = 0.0

    if error_prop is not None and k_prop is not None:
        out += np.asarray(k_prop) * np.asarray(error_prop)

    if error_int is not None and k_int is not None:
        out += np.asarray(k_int) * np.asarray(error_int)

    if error_deriv is not None and k_deriv is not None:
        out += np.asarray(k_deriv) * np.asarray(error_deriv)

    return out
