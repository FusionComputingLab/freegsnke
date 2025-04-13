"""
This script simulates the PCS pipeline that controls the voltages applied to
the active coils of the MAST-U tokamak reactor.

"""

import numpy as np
from sys import float_info

from ip_control import ControlSolenoid


def accumulate_currents(dI, est_I):
    """
    Computes the predicted currents for the active coils so far in the shot, as
    prescribed in the VC subcategory of the circuit category.

    Parameters
    ----------
    - dI : numpy 1D array
        The trajectories of the coil currents that PCS wants to apply
    - est_I : numpy 1D array
        The absolute values of the coil currents that PCS wants to apply

    Returns
    -------
    None (est_I is modified in place)

    """

    est_I += dI


def check_currents(dIvec, Ivec):
    """
    Check the values of the ΔI, I as prescribed in the system
    category of the PCS. At the moment we just check against python
    float limits.

    Parameters
    ----------
    - dIvec : numpy 1D array
        The change of rate estimated for the active coil currents.
    - Ivec : numpy 1D array
        The absolute valued estimated for the active coil currents.

    Returns
    -------
    None (dIvec, Ivec are modified in place)

    """

    dIvec[dIvec > float_info.max] = float_info.max
    Ivec[Ivec > float_info.max] = float_info.max


def calculate_voltage(
    R,
    inductance_ff,
    inductance_fb,
    gains,
    approved_dI,
    approved_I,
    measured_I
):
    """
    Calculate the output voltage to apply on the coils, as prescribed in
    the PF category of the PCS. The equations followed are:

    V_fb = gains * (approved_I - measured_I) * M_fb
    V_ff = approved_dI * M_ff
    V_res = measured_I * R
    V = V_fb + V_ff + V_res

    Parameters
    ----------
    - R : numpy 1D array
        The resistivity vector for the active coils.
    - inductance_ff : numpy 2D array
        The feedfoward inductance matrix of the active coils.
    - inductance_fb : numpy 2D array
        The feedback inductance matrix of the active coils.
    - gains : numpy 2D array
        A diagonal matrix with the gains for each coil.
    - approved_dI : numpy 1D array
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
    Corrected_I = gains @ (approved_I - measured_I)
    V_fb = inductance_fb @ Corrected_I
    print(f"    The feedback voltage: {V_fb}")

    # Compute the feedforward voltage
    V_ff = inductance_ff @ approved_dI
    print(f"    The feedforward voltage: {V_ff}")

    # Compute the resistive voltage
    V_res = measured_I * R
    print(f"    The resistive voltage: {V_res}")

    # Combine all the voltages into the output voltage
    Vout = V_fb + V_ff + V_res

    return Vout


def main():
    """
    Script that simulates the branch of the PCS that controls the active coil
    currents.

    """

    # Initialise the solenoid controller
    ip_controller = ControlSolenoid("ip_sequence.pkl",
                                    "ip_schedule.pkl",
                                    "control_params.pkl")

    # Create necessary objects for plasma control that are supposed to be known
    # at runtime.
    Rp = 0.84                           # Plasma resistivity
    inductances = {"plasma": 3.9,       # Plasma inductance
                   "mutual": 2.7,       # Plasma-Solenoid inductance
                   }

    # Initialise the shape controller
    # TODO to be filled in by Alasdair

    # Create necessary objects for the FP category that are supposed to be
    # known at runtime.
    Rvec = np.rand(12, 1)                   # Vector of resistivities
    inductance_matrix = None                # TODO to be filled in by Alasdair
    gain_matrix = np.diag(np.rand(12, 1))   # Gain matrix
    measured_I = np.rand(12, 1)*1e4         # Vector of measured coil currents

    # Initialise the estimation of the coil currents from the actions applied
    # by the PCS on the current trajectories. The PCS takes over when the
    # currents in the coils are I0.
    I0 = np.rand(12, 1)*1e4
    est_I = I0

    # Execute the PCS pipeline. Here it is assumed that the control is
    # performed over the ticking of some clock (hence the np.arange()).
    for timestamp in np.arange(0.15, 1, 0.1):

        # 1. Plasma control
        sol_dI = ip_controller.ip_control(time_stamp=timestamp,
                                          Rp=Rp,
                                          inductances=inductances
                                          )
        # 2. Shape control
        shp_dI = None    # TODO to be filled in by Alasdair

        # 3. Combine the currents coming from plasma and shape control
        dI = sol_dI + shp_dI

        # 4. Compute the absolute value of the currents.
        accumulate_currents(dI, est_I)

        # 5. System category
        check_currents(dI, est_I)

        # 6. PF category
        # FIXME As of now, inductance_matrix is used for both inductance_ff and
        # inductance_fb.
        V_out = calculate_voltage(R=Rvec,
                                  inductance_ff=inductance_matrix,
                                  inductance_fb=inductance_matrix,
                                  gains=gain_matrix,
                                  approved_dI=dI,
                                  approved_I=est_I,
                                  measured_I=measured_I
                                  )
        print(f"The output voltage is {V_out}")


if __name__ == "__main__":
    main()
