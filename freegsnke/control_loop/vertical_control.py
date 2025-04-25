"""Vertical Control

P6 coil is up/down antisymmetric and used for vertical control.
The function vertical_controller is used to compute the required voltage in P6 coil.

This taken from mastu_tools.py in the MAST-U repository (gitlab)
"""

import numpy as np


def vertical_controller(
    dt,
    target,
    history,
    k_prop,
    k_int,
    k_deriv,
    prop_exponent,
    prop_error,
    deriv_threshold,
    int_factor,
    Ip,
    Ip_ref=None,
    derivative_lag=1,
):
    """
    PID controller required for plasma vertical position. Computes the required voltage
    in the vertical stability coil to stabilise the plasma.

    Parameters
    ----------
    dt : float
        Time step over which controller should act [s].
    target : float
        Target vertical position [m].
    history : list
        List of previous vertical positions of the effective toroidal current center [m].
    k_prop : float
        Proportional gain controls how strongly the voltage reacts to deviations from the target.
    k_int : float
        Integral gain controls how the controller accumulates error over time (to correct drifts).
    k_deriv : float
        Derivative gain controls how the controller reacts to rapid changes in target.
    prop_exponent : float
        Exoponent in proportional term.
    prop_error : float
        Reference error for the proportional term.
    deriv_threshold : float
        Threshold for derivative action - limits effect of sudden jumps in target.
    int_factor : float
        Exponential decay factor that limits effect of older values on integral term.
    Ip : float
        Total plasma current at current time [Amps].
    Ip_ref : float
        Reference total plasma current [Amps], used to normalise output.
    derivative_lag : int
        Number of historical values over which the derivative term acts.

    Returns
    -------
    float
        Voltage required for the vertical stability coil to stabilise the plasma [Volts].
    """

    # if not history, no control action
    if not history:
        return 0

    # proportional term
    error = history[-1] - target
    output = (
        k_prop
        * prop_error
        * np.sign(error)
        * np.abs(error / prop_error) ** prop_exponent
    )

    # integral and derivative terms only if there's enough history
    if len(history) > derivative_lag:

        # integral term
        memory = (int_factor ** np.arange(len(history)))[::-1]
        integral_term = k_int * np.sum(np.array(history) * memory) * dt

        # derivative term (capped)
        derivative_term = k_deriv * ((history[-1] - history[-1 - derivative_lag]) / dt)
        derivative_term = np.sign(derivative_term) * min(
            abs(derivative_term), deriv_threshold
        )

        output += integral_term + derivative_term

    # scale by plasma current reference
    if Ip_ref is not None:
        output *= Ip / Ip_ref

    return output
