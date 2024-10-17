import os
import time
from copy import deepcopy

import freegs4e
import numpy as np
import pytest
from IPython.display import clear_output, display

os.environ["ACTIVE_COILS_PATH"] = "./machine_configs/test/active_coils.pickle"
os.environ["PASSIVE_COILS_PATH"] = "./machine_configs/test/passive_coils.pickle"
os.environ["WALL_PATH"] = "./machine_configs/test/wall.pickle"
os.environ["LIMITER_PATH"] = "./machine_configs/test/limiter.pickle"
os.environ["PROBE_PATH"] = "./machine_configs/test/magnetic_probes.pickle"

# from freegs4e import faster_shape

from freegsnke import build_machine


@pytest.fixture()
def create_machine():
    tokamak = build_machine.tokamak()

    # Creates equilibrium object and initializes it with
    # a "good" solution
    # plasma_psi = np.loadtxt('plasma_psi_example.txt')
    from freegsnke import equilibrium_update

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        # domains can be changed
        Rmin=0.1,
        Rmax=2.0,  # Radial domain
        Zmin=-2.2,
        Zmax=2.2,  # Height range
        # grid resolution can be changed
        nx=65,
        ny=129,  # Number of grid points
        # psi=plasma_psi[::2,:])
    )

    # Sets desired plasma properties for the 'starting equilibrium'
    # values can be changed
    from freegsnke.jtor_update import ConstrainPaxisIp

    profiles = ConstrainPaxisIp(
        eq,
        tokamak.limiter,
        8.1e3,  # Plasma pressure on axis [Pascals]
        6.2e5,  # Plasma current [Amps]
        0.5,  # vacuum f = R*Bt
        alpha_m=1.8,
        alpha_n=1.2,
    )

    from freegsnke import GSstaticsolver

    NK = GSstaticsolver.NKGSsolver(eq)
    currents = np.array(
        [
            40000,
            623.1330076232998,
            15761.113413087669,
            6218.6648587680265,
            10169.401670695957,
            -1913.7157252356117,
            2440.9195954337097,
            -5349.68745069716,
            -1786.696839741781,
            93.17532977532858,
            -4057.3992383452764,
            -100,
        ]
    )
    keys = list(eq.tokamak.getCurrents().keys())
    for i in np.arange(12):
        eq.tokamak[keys[i]].current = currents[i]
    NK.solve(eq, profiles, target_relative_tolerance=1e-8)

    # Initialize the evolution object
    # This uses the starting equilibrium to get all the geometric constraints/grids etc
    from freegsnke import nonlinear_solve

    stepping = nonlinear_solve.nl_solver(
        profiles=profiles,
        eq=eq,
        max_mode_frequency=10**2.5,
        full_timestep=3e-3,
        max_internal_timestep=1,
        plasma_resistivity=5e-7,
        automatic_timestep=False,
        mode_removal=True,
        min_dIy_dI=0.1,
    )
    return tokamak, eq, profiles, stepping


def test_linearised_growth_rate(create_machine):
    tokamak, eq, profiles, stepping = create_machine
    true_GR = 0.06767
    # check that
    assert (
        abs((stepping.linearised_sol.instability_timescale[0] - true_GR) / true_GR)
        < 1e-3
    ), f"Growth rate deviates { abs((stepping.linearised_sol.growth_rates[0]-true_GR)/true_GR)}% from baseline"


def test_linearised_stepper(create_machine):
    tokamak, eq, profiles, stepping = create_machine
    U_active = (stepping.vessel_currents_vec * stepping.evol_metal_curr.R)[
        : stepping.evol_metal_curr.n_active_coils
    ]

    # Example of evolution with constant applied voltages
    t = 0
    history_times = [t]
    t_per_step = []
    # use the following to reset stepping.eq1 to a new IC
    stepping.initialize_from_ICs(eq, profiles)  # _level=0)
    #  noise_level=.001,
    #  noise_vec=None,
    #  update_linearization=False,
    #  update_n_steps=12,
    #  threshold_svd=.15)
    # eqs = deepcopy(stepping.eq1)

    # history_currents = [stepping.currents_vec]
    # history_equilibria = [deepcopy(stepping.eq1)]
    # shapes = faster_shape.shapes_f(stepping.eq1, stepping.profiles1)
    # history_width = [shapes[0]]
    # history_o_points = shapes[1]
    # history_o_points = [stepping.eq1.opt[0]]
    # history_elongation = [shapes[2]]
    # history_dJs = [stepping.dJ]

    history_o_points = [eq.opt[0]]
    history_x_points = [eq.xpt[0]]

    counter = 0
    max_count = 10
    while counter < max_count:
        clear_output(wait=True)
        display(f"Step: {counter}/{max_count-1}")
        display(f"current time t = {t}")
        display(f"current time step dt = {stepping.dt_step}")

        t_start = time.time()

        stepping.nlstepper(
            active_voltage_vec=U_active,
            target_relative_tol_currents=0.001,
            target_relative_tol_GS=0.001,
            verbose=False,
            linear_only=True,
        )

        t_end = time.time()
        t_per_step.append(t_end - t_start)

        t += stepping.dt_step
        history_times.append(t)
        # shapes = faster_shape.shapes_f(stepping.eq2, stepping.profiles2)

        # history_currents.append(stepping.currents_vec)
        # history_equilibria.append(deepcopy(stepping.eq2))
        # history_width.append(shapes[0])
        # history_o_points.append(stepping.eq1.opt[0])
        # history_o_points = np.array(history_o_points)
        # history_elongation.append(shapes[2])
        # history_dJs.append(stepping.dJ)
        history_o_points = np.append(history_o_points, [stepping.eq1.opt[0]], axis=0)
        history_x_points = np.append(history_x_points, [stepping.eq1.xpt[0]], axis=0)
        counter += 1

    # history_currents = np.array(history_currents)
    # history_times = np.array(history_times)
    # history_o_points = np.array(history_o_points)

    leeway = (
        np.array(
            [
                (stepping.eqR[-1, -1] - stepping.eqR[0, 0]) / stepping.nx,
                (stepping.eqZ[-1, -1] - stepping.eqZ[0, 0]) / stepping.ny,
            ]
        )
        / 2
    )  # 1/2 of the pixel size

    true_o_point = np.array([9.49e-01, 0.03])
    true_x_point = np.array([0.599, 1.08])

    assert np.all(
        np.abs((stepping.eq1.opt[0, :2] - true_o_point)) < leeway
    ), "O-point location deviates more than 1/2 of pixel size."
    assert np.all(
        np.abs((stepping.eq1.xpt[0, :2] - true_x_point)) < leeway
    ), "X-point location deviates more than 1/2 of pixel size."


def test_non_linear_stepper(create_machine):
    tokamak, eq, profiles, stepping = create_machine
    U_active = (stepping.vessel_currents_vec * stepping.evol_metal_curr.R)[
        : stepping.evol_metal_curr.n_active_coils
    ]

    # Example of evolution with constant applied voltages
    t = 0
    history_times = [t]
    t_per_step = []
    # use the following to reset stepping.eq1 to a new IC
    stepping.initialize_from_ICs(eq, profiles)  # , noise_level=0)
    #  noise_level=.001,
    #  noise_vec=None,
    #  update_linearization=False,
    #  update_n_steps=12,
    #  threshold_svd=.15)
    # eqs = deepcopy(stepping.eq1)

    # history_currents = [stepping.currents_vec]
    # history_equilibria = [deepcopy(stepping.eq1)]
    # # shapes = faster_shape.shapes_f(stepping.eq1, stepping.profiles1)
    # # history_width = [shapes[0]]
    # # history_o_points = shapes[1]
    history_o_points = [stepping.eq1.opt[0]]
    history_x_points = [stepping.eq1.xpt[0]]
    # history_elongation = [shapes[2]]
    # history_dJs = [stepping.dJ]

    counter = 0
    max_count = 10
    while counter < max_count:
        clear_output(wait=True)
        display(f"Step: {counter}/{max_count-1}")
        display(f"current time t = {t}")
        display(f"current time step dt = {stepping.dt_step}")

        t_start = time.time()

        stepping.nlstepper(
            active_voltage_vec=U_active,
            target_relative_tol_currents=0.001,
            target_relative_tol_GS=0.001,
            verbose=False,
            linear_only=False,
        )

        t_end = time.time()
        t_per_step.append(t_end - t_start)

        t += stepping.dt_step
        history_times.append(t)
        # shapes = faster_shape.shapes_f(stepping.eq2, stepping.profiles2)

        # history_currents.append(stepping.currents_vec)
        # history_equilibria.append(deepcopy(stepping.eq2))
        # # history_width.append(shapes[0])
        # history_o_points.append(stepping.eq1.opt[0])
        # history_o_points = np.array(history_o_points)
        # history_elongation.append(shapes[2])
        # history_dJs.append(stepping.dJ)
        history_o_points = np.append(history_o_points, [stepping.eq1.opt[0]], axis=0)
        history_x_points = np.append(history_x_points, [stepping.eq1.xpt[0]], axis=0)
        counter += 1

    # history_currents = np.array(history_currents)
    # history_times = np.array(history_times)
    # history_o_points = np.array(history_o_points)

    leeway = (
        np.array(
            [
                (stepping.eqR[-1, -1] - stepping.eqR[0, 0]) / stepping.nx,
                (stepping.eqZ[-1, -1] - stepping.eqZ[0, 0]) / stepping.ny,
            ]
        )
        / 2
    )  # 1/2 of the pixel size

    true_o_point = np.array([9.49e-01, 0.03])
    true_x_point = np.array([0.599, 1.08])

    assert np.all(
        np.abs((history_o_points[-1, :2] - true_o_point)) < leeway
    ), "O-point location deviates more than 1/2 of pixel size."
    assert np.all(
        np.abs((stepping.eq1.xpt[0, :2] - true_x_point)) < leeway
    ), "X-point location deviates more than 1/2 of pixel size."
