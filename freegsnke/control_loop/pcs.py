"""
This script simulates the PCS pipeline that controls the voltages applied to
the active coils of the MAST-U tokamak reactor.

"""

import numpy as np
from sys import float_info
from copy import deepcopy

import freegsnke
from .ip_control import ControlSolenoid
from .target_scheduler import TargetScheduler
from .vc_scheduler import ShapeTargetScheduler
from .shape_targets_control import ShapeController


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
    R, inductance_ff, inductance_fb, gains, approved_dI, approved_I, measured_I
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
    print("gains shape,", np.shape(gains), gains)
    print("approved I shape,", np.shape(approved_I), approved_I)
    print("measured I shape,", np.shape(measured_I), measured_I)
    Corrected_I = gains @ (approved_I - measured_I)
    print("corrected I", np.shape(Corrected_I), Corrected_I)
    print("inductance_fb", np.shape(inductance_fb), inductance_fb)
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


def main(eq_start, profiles_start, stepper):
    """
    Script that simulates the branch of the PCS that controls the active coil
    currents.

    Inputs :
    --------
    eq_start : eq object
        Equilibrium at start of simulation
    profiles_start : profiles object
        Profiles at start of simulation
    stepper : Stepper object (nl_solver)

    """

    pickle_folder = "/Users/alasdair.ross/Documents/HARTREE/freegsnke/freegsnke/control_loop/control_test_files"

    # Initialise the solenoid controller
    ip_controller = ControlSolenoid(
        f"{pickle_folder}/ip_sequence.pkl",
        f"{pickle_folder}/ip_schedule.pkl",
        f"{pickle_folder}/ip_control_params.pkl",
    )

    # Create necessary objects for plasma control that are supposed to be known
    # at runtime.
    Rp = 0.84  # Plasma resistivity
    inductances = {
        "plasma": 3.9,  # Plasma inductance
        "mutual": 2.7,  # Plasma-Solenoid inductance
    }

    # Initialise the shape controller
    target_fb_scheduler = ShapeTargetScheduler(
        target_waveform_path=f"{pickle_folder}/target_waveform.pkl",
        target_schedule_path=f"{pickle_folder}/target_schedule.pkl",
        vc_flag="file",
        vc_schedule_path=f"{pickle_folder}/test_vc_set.pkl",
    )

    target_ff_scheduler = TargetScheduler(
        target_schedule_path=f"{pickle_folder}/target_schedule.pkl",
        target_waveform_path=f"{pickle_folder}/target_waveform.pkl",
    )

    shape_controller = ShapeController(
        eq=eq_start,
        profiles=profiles_start,
        stepping=stepper,
        feedback_target_scheduler=target_fb_scheduler,
        feedforward_scheduler=target_ff_scheduler,
        coils=None,
    )

    # Create necessary objects for the FP category that are supposed to be
    # known at runtime.
    Rvec = np.random.rand(
        12,
    )  # Vector of resistivities
    inductance_matrix = deepcopy(shape_controller.inductance_full)
    gain_matrix = np.diag(
        np.random.rand(12)
    )  # Gain matrix for coils(what are these gains??)
    measured_I = np.random.rand(12) * 1e4  # Vector of measured coil currents

    # Initialise the estimation of the coil currents from the actions applied
    # by the PCS on the current trajectories. The PCS takes over when the
    # currents in the coils are I0.
    I0 = np.random.rand(12) * 1e4
    est_I = I0
    print("I0", I0, np.shape(I0))

    eq = deepcopy(eq_start)
    profiles = deepcopy(profiles_start)

    # Execute the PCS pipeline. Here it is assumed that the control is
    # performed over the ticking of some clock (hence the np.arange()).
    for timestamp in np.arange(0.15, 1, 0.1):

        # 1. Plasma control
        sol_dI = ip_controller.ip_control(
            time_stamp=timestamp, Rp=Rp, inductances=inductances, eq=eq
        )
        # print("sol dI", sol_dI, np.shape(sol_dI))

        # 2. Shape control
        shp_dI = shape_controller.feedback_current_rate_timefunc(
            time_stamp=timestamp, eq=eq, profiles=profiles, gain_matrix=None
        )
        # print("shp dI", shp_dI, np.shape(shp_dI))

        # 3. Combine the currents coming from plasma and shape control
        dI = sol_dI + shp_dI
        # print("combined dI", dI, np.shape(dI))

        # 4. Estimate the absolute value of the currents.
        est_I += dI
        # print("est I", est_I, np.shape(est_I))

        # 5. System category
        check_currents(dI, est_I)

        # 6. PF category
        # FIXME As of now, inductance_matrix is used for both inductance_ff and
        # inductance_fb.
        V_out = calculate_voltage(
            R=Rvec,
            inductance_ff=inductance_matrix,
            inductance_fb=inductance_matrix,
            gains=gain_matrix,
            approved_dI=dI,
            approved_I=est_I,
            measured_I=measured_I,
        )
        print(f"time {timestamp} ")
        print(f"The requested output voltage is {V_out}")


def simulate_shot(
    eq_start,
    profiles_start,
    stepper,
    time_slices,
    config_kwargs,
    control_kwargs,
):
    """Simulate a single shot
    Starting from an equilibrium and a profile, simulate the plasma and shape control voltages for a set of time slices.

    Parameters
    ----------
    eq : equilibrium object
        equilibrium object
    profiles : profiles object
        profiles object
    stepper : Stepper object (nl_solver)
        Non Linear Solver object
    time_slices : list
        list of time slices to simulate
    config_kwargs : dict
        dictionary of configuration parameters (values for resistances, inductances, etc.)
    control_kwargs : dict
        dictionary of configuration filespaths (schedules, waveforms, etc)

    Returns
    -------
    None

    """
    print(control_kwargs)
    print(config_kwargs)
    # Initialise the solenoid controller
    ip_controller = ControlSolenoid(
        target_sched_path=control_kwargs["ip_schedule"],
        target_waveform_path=control_kwargs["ip_waveform"],
        contr_params_path=control_kwargs["ip_control_params"],
    )

    # Create necessary objects for plasma control that are supposed to be known
    # at runtime.
    Rp = config_kwargs["plasma_resistivity"]  # Plasma resistivity
    inductances = {
        "plasma": config_kwargs["plasma_inductance"],  # Plasma inductance
        "mutual": config_kwargs["plas_sol_inductance"],  # Plasma-Solenoid inductance
    }

    # Initialise the shape controller
    target_fb_scheduler = ShapeTargetScheduler(
        target_waveform_path=control_kwargs["fb_target_waveform"],
        target_schedule_path=control_kwargs["fb_target_schedule"],
        vc_flag="file",
        vc_schedule_path=control_kwargs["vc_schedule"],
    )

    target_ff_scheduler = TargetScheduler(
        target_schedule_path=control_kwargs["ff_target_schedule"],
        target_waveform_path=control_kwargs["ff_target_waveform"],
    )

    shape_controller = ShapeController(
        eq=eq_start,
        profiles=profiles_start,
        stepping=stepper,
        feedback_target_scheduler=target_fb_scheduler,
        feedforward_scheduler=target_ff_scheduler,
        coils=None,
    )

    # Create necessary objects for the FP category that are supposed to be
    # known at runtime.
    Rvec = config_kwargs["R_vec"]  # Vector of resistivities
    inductance_matrix = deepcopy(shape_controller.inductance_full)
    gain_matrix = config_kwargs[
        "coil_gains"
    ]  # Gain matrix for coils(what are these gains??)
    active_coils = list(eq_start.tokamak.coils_dict.keys())[
        : eq_start.tokamak.n_active_coils
    ]
    active_currents = [eq_start.tokamak.getCurrents()[key] for key in active_coils]
    measured_I = deepcopy(active_currents)  # Vector of measured coil currents

    # Initialise the estimation of the coil currents from the actions applied
    # by the PCS on the current trajectories. The PCS takes over when the
    # currents in the coils are I0.
    I0 = deepcopy(active_currents)
    est_I = I0

    eq = deepcopy(eq_start)
    profiles = deepcopy(profiles_start)

    # Execute the PCS pipeline. Here it is assumed that the control is
    # performed over the ticking of some clock (hence the np.arange()).
    for timestamp in time_slices:

        # 1. Plasma control
        sol_dI = ip_controller.ip_control(
            time_stamp=timestamp, Rp=Rp, inductances=inductances, eq=eq
        )
        print("sol dI", sol_dI, np.shape(sol_dI))
        # 2. Shape control
        shp_dI = shape_controller.feedback_current_rate_timefunc(
            time_stamp=timestamp, eq=eq, profiles=profiles, gain_matrix=None
        )
        print("shp dI", shp_dI, np.shape(shp_dI))

        # 3. Combine the currents coming from plasma and shape control
        dI = sol_dI + shp_dI
        print("combined dI", dI, np.shape(dI))

        # 4. Estimate the absolute value of the currents.
        est_I += dI
        print("est I", est_I, np.shape(est_I))

        # 5. System category
        check_currents(dI, est_I)

        # 6. PF category
        # FIXME As of now, inductance_matrix is used for both inductance_ff and
        # inductance_fb.
        V_out = calculate_voltage(
            R=Rvec,
            inductance_ff=inductance_matrix,
            inductance_fb=inductance_matrix,
            gains=gain_matrix,
            approved_dI=dI,
            approved_I=est_I,
            measured_I=measured_I,
        )
        print(f"time {timestamp} ")
        print(f"The requested output voltage is {V_out}")


if __name__ == "__main__":

    # create equi, profiels and stepper
    # use base example in freeegsnke
    from freegs4e.gradshafranov import mu0  # permeability
    from freegsnke import GSstaticsolver, build_machine, equilibrium_update, jtor_update
    from freegsnke import virtual_circuits as vc
    from freegsnke.jtor_update import Lao85
    from freegsnke.nonlinear_solve import nl_solver

    from copy import deepcopy

    def create_equilibrium(plasma_psi=None) -> tuple:

        tokamak = build_machine.tokamak()

        eq = equilibrium_update.Equilibrium(
            tokamak=tokamak,  # provide tokamak object
            Rmin=0.1,
            Rmax=2.0,  # radial range
            Zmin=-2.2,
            Zmax=2.2,  # vertical range
            nx=129,  # number of grid points in the radial direction (needs to be of the form (2**n + 1) with n being an integer)
            ny=129,  # number of grid points in the vertical direction (needs to be of the form (2**n + 1) with n being an integer)
            # psi=plasma_psi
        )

        alpha = np.array([1, 0, -1])
        beta = (1 - 0.3) / 0.3 * alpha * mu0

        profiles = Lao85(
            eq=eq,
            limiter=tokamak.limiter,
            Ip=6e5,
            fvac=0.5,
            alpha=alpha,
            beta=beta,
            alpha_logic=False,
            beta_logic=False,
            Ip_logic=True,
        )

        # Define solver
        GSStaticSolver = GSstaticsolver.NKGSsolver(eq)

        return eq, profiles, GSStaticSolver

    # base equi and profiles and solver - contains machine informatoin and greens functions
    eq_base, profiles_base, solver = create_equilibrium()
    GSStaticSolver = GSstaticsolver.NKGSsolver(eq_base)

    eq_start = deepcopy(eq_base)
    profiles_start = deepcopy(profiles_base)

    stepper = nl_solver(
        profiles=profiles_start,
        eq=eq_start,
        mode_removal=False,
        linearize=False,
    )

    test_control_kwargs = {
        "ip_schedule": "../freegsnke/control_loop/control_test_files/ip_schedule.pkl",
        "ip_control_params": "../freegsnke/control_loop/control_test_files/ip_control_params.pkl",
        "ip_waveform": "../freegsnke/control_loop/control_test_files/ip_waveform.pkl",
        "fb_target_waveform": "../freegsnke/control_loop/control_test_files/target_waveform.pkl",
        "fb_target_schedule": "../freegsnke/control_loop/control_test_files/target_schedule.pkl",
        "ff_target_waveform": "../freegsnke/control_loop/control_test_files/target_waveform.pkl",
        "ff_target_schedule": "../freegsnke/control_loop/control_test_files/target_schedule.pkl",
        "vc_schedule": "../freegsnke/control_loop/control_test_files/test_vc_set.pkl",
    }
    test_config_kwargs = {
        "plasma_resistivity": 0.84,
        "plasma_inductance": 3.9,
        "plas_sol_inductance": 2.7,
        "Rp": 0.84,
        "R_vec": np.random.rand(12),
        "gain_matrix": np.diag(np.random.rand(12)),
    }

    tiemslices = np.arange(0.15, 1, 0.1)
    main(eq_start=eq_start, profiles_start=profiles_start, stepper=stepper)
