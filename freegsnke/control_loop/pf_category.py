"""Small module to implement the PF category of MAST-U's PCS computer """

import numpy as np


def calculate_voltage_pf(
    R,
    inductances,
    coil_gains,
    approved_dI_dt,
    approved_I,
    measured_I,
    # These clips follow the same coil order as the gains dictionary
    clips=np.array([1820, 800, 900, 800, 800, 800, 800, 800, 800, 800, 800,
                    350])
):
    """
    Calculate the output voltage to apply on the coils, as prescribed in
    the PF category of the PCS. The equations followed are:

    V_ind = (coil_gains * (approved_I - measured_I) + approved_dI_dt) * M_fb
    V_res = measured_I * R
    V = V_ind + V_res

    Parameters
    ----------
    - R : numpy 1D array
        The resistivity vector for the active coils.
    - inductance_ff : numpy 2D array
        The feedforward inductance matrix of the active coils.
    - inductance_fb : numpy 2D array
        The feedback inductance matrix of the active coils.
    - coil_gains : numpy 2D array
        A diagonal matrix with the coil_gains for each coil.
    - approved_dI_dt: numpy 1D array
        Approved change of rate of the coil currents by the system category of
        the PCS.
    - approved_I : numpy 1D array
        Approved coil current by the system category of the PCS.
    - measured_I : numpy 1D array
        Actual measured coil currents.

    Returns
    -------
    numpy 1D array
        Output control voltage of PCS to apply on the coils.

    """

    # Compute the feedback voltage
    # print("R shape,", np.shape(R), R)
    # print("inductances shape,", np.shape(inductances), inductances)
    # print("coil_gains shape,", np.shape(coil_gains), coil_gains)
    print("approved I shape and values:,", np.shape(approved_I), approved_I)
    print("approved dI_dt shape and values:,",
          np.shape(approved_dI_dt), approved_dI_dt)
    print("measured I shape and values:,", np.shape(measured_I), measured_I)

    corrected_I = coil_gains @ (approved_I - measured_I)
    print("PF proportional current rate: \n", corrected_I)

    v_ff = inductances @ approved_dI_dt
    v_fb = inductances @ corrected_I
    print(f"    The PF feedforward voltage: \n {v_ff}")
    print(f"    The PF feedback voltage: \n {v_fb}")

    # Compute the resistive voltage
    v_res = measured_I * R
    print(f"    The resistive voltage: \n {v_res}")

    # Combine all the voltages into the output voltage
    v_out = v_ff + v_fb + v_res
    print(f"    The full voltage: \n {v_out}")

    # We omit pc (the last element of clips)
    v_out_clipped_pos = np.minimum(v_out, clips[:-1])
    v_out_clipped = np.maximum(v_out_clipped_pos, -clips[:-1])

    if not np.allclose(v_out, v_out_clipped):
        print(f"    The full voltage clipped: \n {v_out_clipped}")

    results = {"v_ff": v_ff,
               "v_fb": v_fb,
               "v_res": v_res,
               "v_out": v_out,
               "v_out_cp": v_out_clipped
               }

    # return v_out
    return results
