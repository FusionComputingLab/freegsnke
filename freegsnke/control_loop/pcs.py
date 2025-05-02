"""
This script simulates the PCS pipeline that controls the voltages applied to
the active coils of the MAST-U tokamak reactor.

"""

import numpy as np
import pickle
from sys import float_info
from copy import deepcopy

import freegsnke
from .ip_control import ControlSolenoid
from .target_scheduler import TargetScheduler
from .shape_scheduling import ShapeTargetScheduler
from .shape_targets_control import ShapeController
from .vertical_control import vertical_controller

from freegs4e.plotting import plotEquilibrium as plot_eqi


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


def calculate_voltage_pf(
    R,
    inductance_ff,
    inductance_fb,
    gains,
    approved_dI_dt,
    approved_I,
    measured_I,
    Ipert=None,
    dIpert_dt=None,
):
    """
    Calculate the output voltage to apply on the coils, as prescribed in
    the PF category of the PCS. The equations followed are:

    V_fb = gains * (approved_I - measured_I) * M_fb
    V_ff = approved_dI_dt* M_ff
    V_res = measured_I * R
    V = V_fb + V_ff + V_res

    Parameters
    ----------
    - R : numpy 1D array
        The resistivity vector for the active coils.
    - inductance_ff : numpy 2D array
        The feedforward inductance matrix of the active coils.
    - inductance_fb : numpy 2D array
        The feedback inductance matrix of the active coils.
    - gains : numpy 2D array
        A diagonal matrix with the gains for each coil.
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

    # # Compute the feedback voltage
    # print("gains shape,", np.shape(gains), gains)
    # print("approved I shape,", np.shape(approved_I), approved_I)
    # print("measured I shape,", np.shape(measured_I), measured_I)
    approved_I += Ipert
    Corrected_I = gains @ (approved_I - measured_I)
    # print("corrected I", np.shape(Corrected_I), Corrected_I)
    # print("inductance_fb", np.shape(inductance_fb), inductance_fb)
    V_fb = inductance_fb @ Corrected_I
    print(f"    The feedback voltage: {V_fb}")

    # Compute the feedforward voltage
    approved_dI_dt += dIpert_dt
    V_ff = inductance_ff @ approved_dI_dt
    print(f"    The feedforward voltage: {V_ff}")

    # Compute the resistive voltage
    V_res = measured_I * R
    print(f"    The resistive voltage: {V_res}")

    # Combine all the voltages into the output voltage
    Vout = V_fb + V_ff + V_res

    return Vout


def voltage_request(
    ip_controller,
    shape_controller,
    eq,
    profiles,
    timestamp,
    Rp,
    inductacnes_pl,
    inductance_matrix,
    gain_matrix,
    est_I,
    measured_I,
    coil_perturbation,
    Rvec=None,
    targ_obs=None,
    Ip_obs=None,
):
    """
    Computes the voltages required for the plasma, shape and coil perturbation categories and combines in pf category.
    Need to provide schedulers, controllers, values for control parameters

    Parameters
    ----------
    ip_controller : ControlSolenoid
        An object that controls the plasma current.
    shape_controller : ShapeController
        An object that controls the shape currents.
    eq : equilibrium object
        equilibrium object
    profiles : profiles object
        profiles object
    timestamp : float
        time stamp of the target to be retrieved
    Rp : float
        plasma resistivity
    inductacnes_pl : dict
        A dictionary with all the required plasma inductacnes.
    Rvec : numpy 1D array
        The resistivity vector for the active coils.
    inductance_matrix : numpy 2D array
        The inductance matrix for the active coils.
    gain_matrix : numpy 2D array
        A diagonal matrix with the gains for each coil.
    est_I : numpy 1D array
        The estimated coil currents.
    measured_I : numpy 1D array
        The measured coil currents.
    coil_perturbation : TargetScheduler
        An object that controls the coil perturbation.
    targ_obs : numpy 1D array
        The observed coil currents.   Defaults to None, in which case the observed targets are computed from the equilibrium.
    Ip_obs : float
        The observed plasma current. Defaults to None, in which case the observed plasma current is taken from the equilibrium.

    Returns
    -------
    numpy 1D array
        The calculated voltages for the all the active coils.
    """

    print(f"Computing Voltage request at time: {timestamp} ")

    # 1. Plasma control
    print("\n Plasma Control")
    sol_dI_dt = ip_controller.ip_control(
        time_stamp=timestamp, Rp=Rp, inductacnes_pl=inductacnes_pl, Ip_obs=Ip_obs, eq=eq
    )
    # print("sol dI", sol_dI, np.shape(sol_dI))
    # 2. Shape control
    print("\n Shape Control")
    shp_dI_dt = shape_controller.feedback_current_rate_timefunc(
        time_stamp=timestamp,
        eq=eq,
        profiles=profiles,
        gain_matrix=None,
        target_obs=targ_obs,
    )
    # print("shp dI", shp_dI_dt, np.shape(shp_dI_dt))
    # 2.1. Coil perturbation
    Ipert = coil_perturbation.desired_target_values(time_stamp=timestamp)
    dIpert_dt = coil_perturbation.feed_forward_gradient(
        time_stamp=timestamp, targets=None
    )
    print("coil pert value ", Ipert, np.shape(Ipert))
    ## add this coil currents to ....????

    # 3. Combine the currents coming from plasma and shape control
    dI_dt = sol_dI_dt + shp_dI_dt
    # print("combined dI", dI, np.shape(dI))

    # 4. Estimate the absolute value of the currents.
    est_I += dI_dt
    # todo Add coil perturbation here ??
    # print("est I", est_I, np.shape(est_I))

    # 5. System category
    check_currents(dI_dt, est_I)

    # 6. PF category
    # FIXME As of now, inductance_matrix is used for both inductance_ff and
    # inductance_fb.
    print("\n Calculating Voltage Requests")
    V_out = calculate_voltage_pf(
        R=Rvec,
        inductance_ff=inductance_matrix,
        inductance_fb=inductance_matrix,
        gains=gain_matrix,
        approved_dI_dt=dI_dt,
        approved_I=est_I,
        measured_I=measured_I,
        Ipert=Ipert,
        dIpert_dt=dIpert_dt,
    )
    print(f"The requested output voltage is {V_out}")
    return V_out


def get_measured_shape_vals(measured_dict, names, pos):
    """
    Extracts the values of the shape targets from the measured values dictionary.
    dict of form {"times" : [vals], "targ_1" : [vals], "targ_2" : [vals]}

    Parameters
    ----------
    dict : dict
        Dictionary containing the measured values.
    names : list
        List of target names.
    pos : int
        Position of the target in the list.

    Returns
    -------
    list
        List of target values.
    """
    return np.array([measured_dict[name][pos] for name in names])


def validate_shot(
    config_kwargs,
    control_kwargs,
    eq_start,
    profiles_start,
    stepping,
    # ip_vals,
    # shape_vals,
    # timestamps,
    measured_vals,
):
    """
    Run validation of control with historic measurements, rather than equi simulation.

    Provide the simulation inputs (schedules, control parameters), eqi, profiles, stepping (provides machine description)
    provide also the measured values for the plasma current and shape currents, along with associated timestamps.

    Parameters
    ----------
    config_kwargs : dict
        dictionary of configuration parameters (values for resistances, inductances, etc.)
    control_kwargs : dict
        dictionary of configuration filespaths (schedules, waveforms, etc)
    eq_start : eq object
        equilibrium object
    profiles_start : profiles object
        profiles object
    stepping : stepping object (nl_solver)
        Non Linear Solver object
    ip_vals : numpy 1D array
        measured plasma current values
    shape_vals : numpy 1D array
        measured shape current values
    timestamps : numpy 1D array
        timestamps for the measured values

    Returns
    -------
    None
    """

    # Load schedulers and controllers
    # Load configuration parameters needed at runtime.
    Rp = config_kwargs["plasma_resistivity"]  # Plasma resistivity
    inductances_pl = {
        "plasma": config_kwargs["plasma_inductance"],  # Plasma inductance
        "mutual": config_kwargs["plas_sol_inductance"],  # Plasma-Solenoid inductance
    }

    if "inductance_matrix" in config_kwargs.keys():
        inductance_matrix = config_kwargs["inductance_matrix"]
        print("Using user provided inductance matrix")
    else:
        inductance_matrix = (
            None  # default inductance matrix (initialized below in shape controller)
        )
    gain_matrix = config_kwargs[
        "coil_gains"
    ]  # Gain matrix for coils(what are these gains??)
    active_coils = list(eq_start.tokamak.coils_dict.keys())[
        : eq_start.tokamak.n_active_coils
    ]

    # Load Schedulers
    target_ff_scheduler = TargetScheduler(
        target_schedule_path=control_kwargs["ff_target_schedule"],
        target_waveform_path=control_kwargs["ff_target_waveform"],
    )
    target_fb_scheduler = ShapeTargetScheduler(
        target_waveform_path=control_kwargs["fb_target_waveform"],
        target_schedule_path=control_kwargs["fb_target_schedule"],
        vc_flag=control_kwargs["vc_flag"],
        vc_schedule_path=control_kwargs["vc_schedule"],
    )

    # Initialise controllers
    # Initialise the solenoid controller
    ip_controller = ControlSolenoid(
        target_sched_path=control_kwargs["ip_schedule"],
        target_waveform_path=control_kwargs["ip_waveform"],
        contr_params_path=control_kwargs["ip_control_params"],
    )

    shape_controller = ShapeController(
        eq=eq_start,
        profiles=profiles_start,
        stepping=stepping,
        feedback_target_scheduler=target_fb_scheduler,
        feedforward_scheduler=target_ff_scheduler,
        coils=None,
        inductance_matrix=inductance_matrix,
        coil_resist=None,
    )

    inductance_matrix = deepcopy(
        shape_controller.inductance_full
    )  ### needst tidying as code (inductance matrix recreated multiple times)
    # TODO I_Coil_Pert category ....
    coil_perturbation = TargetScheduler(
        target_schedule_path=control_kwargs["coil_pert_schedule"],
        target_waveform_path=control_kwargs["coil_pert_waveform"],
    )
    currents_start = [eq_start.tokamak.getCurrents()[key] for key in active_coils]
    if "Rvec" in config_kwargs.keys():
        Rvec = config_kwargs["Rvec"]
    else:
        Rvec = shape_controller.coil_resist  # Vector of resistivities

    for i, timestamp in enumerate(measured_vals["time_stamps"]):
        # Ip_val = ip_vals[i]
        # shape_vals = shape_vals[i]
        print(
            f" --------------- \n TIMESLICE COMPUTATION  {timestamp}, pos {i} \n ---------------"
        )
        controlled_targs = (
            shape_controller.feedback_target_scheduler.retrieve_controlled_targets(
                timestamp
            )
        )
        Ip_val = get_measured_shape_vals(measured_vals, ["Ip_vals"], i)[0]
        shape_vals = get_measured_shape_vals(measured_vals, controlled_targs, i)

        voltage_request(
            ip_controller,
            shape_controller,
            eq=eq_start,
            profiles=profiles_start,
            timestamp=timestamp,
            Rp=Rp,
            inductacnes_pl=inductances_pl,
            Rvec=Rvec,
            inductance_matrix=inductance_matrix,
            est_I=currents_start,
            measured_I=currents_start,  # ?is this what we want?
            gain_matrix=gain_matrix,
            coil_perturbation=coil_perturbation,
            targ_obs=shape_vals,
            Ip_obs=Ip_val,
        )


def simulate_shot(
    eq_start,
    profiles_start,
    stepping,
    t_start,
    n_iter,
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
    stepping : stepping object (nl_solver)
        Non Linear Solver object
    t_start : float
        starting time for simulation
    n_iter : int
        number of iterations to simulate
    config_kwargs : dict
        dictionary of configuration parameters (values for resistances, inductances, etc.)
    control_kwargs : dict
        dictionary of configuration filepaths (schedules, waveforms, etc)

    Returns
    -------
    None

    """

    # Load configuration parameters needed at runtime.
    Rp = config_kwargs["plasma_resistivity"]  # Plasma resistivity
    inductances_pl = {
        "plasma": config_kwargs["plasma_inductance"],  # Plasma inductance
        "mutual": config_kwargs["plas_sol_inductance"],  # Plasma-Solenoid inductance
    }

    if "inductance_matrix" in config_kwargs.keys():
        inductance_matrix = config_kwargs["inductance_matrix"]
        print("Using user provided inductance matrix")
    else:
        inductance_matrix = (
            None  # default inductance matrix (initialized below in shape controller)
        )
    gain_matrix = config_kwargs[
        "coil_gains"
    ]  # Gain matrix for coils(what are these gains??)
    active_coils = list(eq_start.tokamak.coils_dict.keys())[
        : eq_start.tokamak.n_active_coils
    ]

    # Load Schedulers
    target_ff_scheduler = TargetScheduler(
        target_schedule_path=control_kwargs["ff_target_schedule"],
        target_waveform_path=control_kwargs["ff_target_waveform"],
    )
    target_fb_scheduler = ShapeTargetScheduler(
        target_waveform_path=control_kwargs["fb_target_waveform"],
        target_schedule_path=control_kwargs["fb_target_schedule"],
        vc_flag=control_kwargs["vc_flag"],
        vc_schedule_path=control_kwargs["vc_schedule"],
    )

    # Initialise controllers
    # Initialise the solenoid controller
    ip_controller = ControlSolenoid(
        target_sched_path=control_kwargs["ip_schedule"],
        target_waveform_path=control_kwargs["ip_waveform"],
        contr_params_path=control_kwargs["ip_control_params"],
    )

    shape_controller = ShapeController(
        eq=eq_start,
        profiles=profiles_start,
        stepping=stepping,
        feedback_target_scheduler=target_fb_scheduler,
        feedforward_scheduler=target_ff_scheduler,
        coils=None,
        inductance_matrix=inductance_matrix,
    )

    inductance_matrix = deepcopy(
        shape_controller.inductance_full
    )  ### needst tidying as code (inductance matrix recreated multiple times)
    coil_perturbation = TargetScheduler(
        target_schedule_path=control_kwargs["coil_pert_schedule"],
        target_waveform_path=control_kwargs["coil_pert_waveform"],
    )

    active_currents = [eq_start.tokamak.getCurrents()[key] for key in active_coils]

    if "Rvec" in config_kwargs.keys():
        Rvec = config_kwargs["Rvec"]
    else:
        Rvec = shape_controller.coil_resist  # Vector of resistivities

    measured_I = deepcopy(active_currents)  # Vector of measured coil currents

    # Initialise the estimation of the coil currents from the actions applied
    # by the PCS on the current trajectories. The PCS takes over when the
    # currents in the coils are I0.
    I0 = deepcopy(active_currents)
    est_I = I0

    eq = deepcopy(eq_start)
    profiles = deepcopy(profiles_start)

    # instatiate NL solver stuff....
    t = t_start
    dt = stepping.dt_step
    t_stop = t_start + n_iter * dt
    print(f"------\n Simulation at t={t_start} \n --------")
    print(f"Simulation will run for {n_iter} iterations")
    plot_eqi(eq_start, show=True)

    history_jz = [
        np.mean(stepping.profiles1.jtor / stepping.profiles1.Ip * eq.Z)
    ]  # for vertical controller

    # initialise storage list for data collection (for plotting later )
    history_times = [t]
    history_eqs = [deepcopy(stepping.eq1)]
    history_full_currents = [stepping.currents_vec[:-1]]
    history_Ip = [stepping.profiles1.Ip]
    history_voltages = []
    history_plasma_resistivity = [stepping.plasma_resistivity]
    # separatrix = stepping.eq1.separatrix(ntheta=100)
    # history_width = [np.amax(separatrix[:, 0]) - np.amin(separatrix[:, 0])]
    history_o_points = [stepping.eq1.opt[0]]
    # history_elongation = [
    #     (np.amax(separatrix[:, 1]) - np.amin(separatrix[:, 1])) / history_width[0]
    # ]
    xpts = stepping.eq1.xpt[0:2, 0:2]
    x_point_ind = np.argmin(xpts[:, 1])
    Zx = xpts[x_point_ind, 1]
    Rx = xpts[x_point_ind, 0]
    history_xpoints = [[Rx, Zx]]
    history_Rin = [stepping.eq1.innerOuterSeparatrix()[0]]
    history_Rout = [stepping.eq1.innerOuterSeparatrix()[1]]
    # for timestamp in time_slices:  # do around 1000hz
    # do as while loop

    while t < t_stop:
        print(f"------\n Simulation at t={t} \n --------")
        t += dt
        ##compute voltage request
        v_requested = voltage_request(
            ip_controller,
            shape_controller,
            eq=eq_start,
            profiles=profiles_start,
            timestamp=t,
            Rp=Rp,
            inductacnes_pl=inductances_pl,
            Rvec=Rvec,
            inductance_matrix=inductance_matrix,
            est_I=est_I,
            measured_I=measured_I,
            gain_matrix=gain_matrix,
            coil_perturbation=coil_perturbation,
        )
        #### update equi

        v_requested[-1] = vertical_controller(
            dt=stepping.dt_step,
            target=0.0,
            history=history_jz,
            k_prop=-20000,
            k_int=0,
            k_deriv=-50,
            prop_exponent=1.0,
            prop_error=1e-3,
            deriv_threshold=50,
            int_factor=0.98,
            Ip=stepping.profiles1.Ip,
            Ip_ref=750e3,
            derivative_lag=1,
        )

        # copy from example 6c.

        # update equilibrium
        # carry out the time step
        print("updating equilibrium with nolinear solve")
        stepping.nlstepper(
            active_voltage_vec=v_requested,
            linear_only=True,  # linearise only
            # linear_only=False,  # linearise only
            verbose=False,
            # custom_coil_resist=coil_resist,   #options for restistances/inductances used in solve
            # custom_self_ind=coil_ind)
        )
        print("equi updated")
        # #   # store inputs/outputs
        xpts = stepping.eq1.xpt[0:2, 0:2]
        x_point_ind = np.argmin(xpts[:, 1])
        Zx = xpts[x_point_ind, 1]
        Rx = xpts[x_point_ind, 0]
        history_xpoints.append([Rx, Zx])
        history_Rin.append(stepping.eq1.innerOuterSeparatrix()[0])
        history_Rout.append(stepping.eq1.innerOuterSeparatrix()[1])

        history_times.append(t)
        history_Ip.append(stepping.profiles1.Ip)
        history_full_currents.append(stepping.currents_vec[:-1])
        history_voltages.append(v_requested)
        history_plasma_resistivity.append(stepping.plasma_resistivity)
        history_jz.append(
            np.mean(stepping.profiles1.jtor / stepping.profiles1.Ip * eq.Z)
        )
        # separatrix = stepping.eq1.separatrix(ntheta=100)
        # history_width.append(np.amax(separatrix[:, 0]) - np.amin(separatrix[:, 0]))
        history_o_points = np.append(history_o_points, [stepping.eq1.opt[0]], axis=0)
        # history_elongation.append(
        #     (np.amax(separatrix[:, 1]) - np.amin(separatrix[:, 1])) / history_width[-1]
        # )
    # lists to arrays
    history_Ip = np.array(history_Ip)
    history_full_currents = np.array(history_full_currents)
    history_voltages = np.array(history_voltages)
    history_plasma_resistivity = np.array(history_plasma_resistivity)
    history_times = np.array(history_times)
    history_o_points = np.array(history_o_points)
    history_xpoints = np.array(history_xpoints)

    # save the history to file
    history_dict = {
        "times": history_times,
        "equilibrium": history_eqs,
        "full_currents": history_full_currents,
        "Ip": history_Ip,
        "voltages": history_voltages,
        "plasma_resistivity": history_plasma_resistivity,
        "jz": history_jz,
        # "width": history_width,
        "o_points": history_o_points,
        # "elongation": history_elongation,
        "xpoints": history_xpoints,
        "Rin": history_Rin,
        "Rout": history_Rout,
    }
    # with open("history.pkl", "wb") as fp:
    #     pickle.dump(history_dict, fp)
    return history_dict


if __name__ == "__main__":

    pass

    test_control_kwargs = {
        "ip_schedule": "../freegsnke/control_loop/control_test_files/ip_schedule.pkl",
        "ip_control_params": "../freegsnke/control_loop/control_test_files/ip_control_params.pkl",
        "ip_waveform": "../freegsnke/control_loop/control_test_files/ip_waveform.pkl",
        "fb_target_waveform": "../freegsnke/control_loop/control_test_files/target_waveform.pkl",
        "fb_target_schedule": "../freegsnke/control_loop/control_test_files/target_schedule.pkl",
        "ff_target_waveform": "../freegsnke/control_loop/control_test_files/target_waveform.pkl",
        "ff_target_schedule": "../freegsnke/control_loop/control_test_files/target_schedule.pkl",
        "vc_schedule": "../freegsnke/control_loop/control_test_files/test_vc_set.pkl",
        "vc_flag": "file",
        "coil_pert_schedule": "../freegsnke/control_loop/control_test_files/test_coil_pert_set.pkl",
        "coil_pert_waveform": "../freegsnke/control_loop/control_test_files/test_coil_pert_set.pkl",
    }
    test_config_kwargs = {
        "plasma_resistivity": 0.84,
        "plasma_inductance": 3.9,
        "plas_sol_inductance": 2.7,
        "Rp": 0.84,
        "R_vec": np.random.rand(12),
        "gain_matrix": np.diag(np.random.rand(12)),
        "inductance_matrix": np.random.rand(12, 12),
    }
