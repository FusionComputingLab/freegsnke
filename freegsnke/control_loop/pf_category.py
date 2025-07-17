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

    Returns
    -------
    numpy 1D array
        Voltage to apply to each of the active coils. Output in Volts. 
    """

    # resistive voltages
    v_res = measured_I * R
    # print(f"    The resistive voltage: \n {v_res}")

    # FF voltages
    v_FF = M_FF @ approved_dIdt
    # print(f"    The PF feedforward voltage: \n {v_FF}")

    # FB voltages
    delta_I = approved_I - measured_I
    # v_FB = M_FB @ (delta_I * coil_gains)
    v_FB = M_FB @ (delta_I / coil_gains)
    # print(f"    The PF feedback voltage: \n {v_FB}")

    # initial voltage demands (pre-clipping)
    v_init = v_res + v_FF + v_FB

    # final voltage demands (clipped)
    v_init_clipped_pos = np.minimum(v_init, voltage_clips)
    v_clipped = np.maximum(v_init_clipped_pos, - voltage_clips)
    print(f"---")
    print(f"Absolute clipping? --> {not np.allclose(v_init, v_clipped)})")

    # finally we apply the "slew rates", additive clipping of voltage rate of change
    delta_voltages = v_clipped - prev_voltages
    max_delta = slew_rates*dt
    delta_clipped = np.clip(delta_voltages, -max_delta, max_delta)
    v_final = prev_voltages + delta_clipped
    print(f"Derivative clipping? --> {not np.allclose(delta_clipped, delta_voltages)})")

    print(f"    Voltages = {v_final}")

    return v_final, delta_I