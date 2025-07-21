"""
Small module to implement the PF category of MAST-U control loops
"""

import numpy as np

def pf_voltage_demands(
    R,
    M_FF,
    M_FB,
    coil_gains,
    approved_dIdt,
    approved_I,
    measured_I,
    voltage_clips,
    slew_rates,
    prev_voltages,
    dt,
    verbose=False,
):
    """
    Calculate the output voltage to apply on the coils, as prescribed in
    the PF category of the PCS. 
    
    Parameters
    ----------
    R : numpy 1D array
        The vector of resistances for the active coils.
    M_FF : numpy 2D array
        The feedforward inductance matrix of the active coils.
    M_FB : numpy 2D array
        The feedback inductance matrix of the active coils.
    coil_gains : numpy 1D array
        The vector of "coil gains" for the active coils (these are timescales in seconds).
    approved_dIdt: numpy 1D array
        Approved rate of change of the coil currents in the active coils (provided by the
        "system" category of the control loops). Input in A. 
    approved_I : numpy 1D array
        Approved coil currents in the active coils (provided by the "system" category of 
        the control loops). Input in A. 
    measured_I : numpy 1D array
        Measured coil currents (from experiment or simulation) in the active coils. Input in A. 
    voltage_clips : numpy 1D array
        Final voltage requests are clipped if the they are outside +/- these values
        for each of the active coils. Input in Volts. 
    slew_rates : numpy 1D array
        Final voltage requests are clipped again if their derivatives (wrt previous timestep) are outside 
        +/- these values for each of the active coils. Input in Volts/s. 
    prev_voltages : numpy 1D array
        Voltage demands from the previous time step in the active coils. Input in Volts.  
    dt : float
        Time step between current and previous voltage demands. Input in seconds.   
    verbose : bool
        Print some output (True) or not (False).

    Returns
    -------
    numpy 1D array
        Voltage to apply to each of the active coils. Units in Volts. 
    numpy 1D array
        Difference in currents in the feedback term used to calculate the feedback voltage. Units in Amps.
    """

    print("---")

    # resistive voltages
    v_res = measured_I * R
    if verbose:
        print(f"    Resistive voltage = {v_res}")

    # FF voltages
    v_FF = M_FF @ approved_dIdt
    if verbose:
        print(f"    Feedforward voltage = {v_FF}")

    # FB voltages
    delta_I = approved_I - measured_I
    v_FB = M_FB @ (delta_I / coil_gains)
    if verbose:
        print(f"    Feedback voltage = {v_FB}")

    # initial voltage demands (pre-clipping)
    v_init = v_res + v_FF + v_FB
    if verbose:
        print(f"    Pre-clipping voltage demand (sum of above) = {v_init}")

    # final voltage demands (clipped)
    v_init_clipped_pos = np.minimum(v_init, voltage_clips)
    v_clipped = np.maximum(v_init_clipped_pos, - voltage_clips)
    if verbose and not np.allclose(v_init, v_clipped):
        print(f"    Clipped voltage demand (according to `voltage_clips`) = {v_init}")

    # finally we apply the "slew rates", additive clipping of voltage rate of change
    delta_voltages = v_clipped - prev_voltages
    max_delta = slew_rates*dt
    delta_clipped = np.clip(delta_voltages, -max_delta, max_delta)
    v_final = prev_voltages + delta_clipped
    if verbose and not np.allclose(v_final, v_clipped):
        print(f"    Derivative clipped voltage demand (according to `slew_rates`) = {v_final}")

    if verbose:
        print(f"FINAL VOLTAGE DEMANDS = {v_final}")

    return v_final, delta_I