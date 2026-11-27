"""Tests for the dedicated even-only evolutive-solver interface."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from freegsnke.nonlinear_solve import nl_solver
from freegsnke.symmetry_reduced_solve import SymmetryReducedSolver


def test_reduced_solver_enforces_even_configuration():
    profiles = object()
    equilibrium = object()
    static_solver = object()
    reflection = np.eye(2)

    with patch.object(nl_solver, "__init__", return_value=None) as initialise:
        solver = SymmetryReducedSolver(
            profiles,
            equilibrium,
            static_solver,
            passive_reflection_operator=reflection,
            full_timestep=2.5e-3,
        )

    initialise.assert_called_once_with(
        profiles=profiles,
        eq=equilibrium,
        GSStaticSolver=static_solver,
        passive_reflection_operator=reflection,
        force_up_down_symmetric=True,
        remove_odd_passive_modes=True,
        full_timestep=2.5e-3,
    )


@pytest.mark.parametrize(
    "option",
    [
        {"force_up_down_symmetric": False},
        {"remove_odd_passive_modes": False},
    ],
)
def test_reduced_solver_rejects_non_even_configuration(option):
    """Users cannot disable either invariant through inherited options."""
    with pytest.raises(ValueError):
        SymmetryReducedSolver(object(), object(), object(), **option)


def _symmetry_test_equilibrium(z_values):
    """Provide the real constructor's inputs up to its first GS solve."""
    z_values = np.asarray(z_values, dtype=float)
    nx, ny = 2, len(z_values)
    return SimpleNamespace(
        nx=nx,
        ny=ny,
        dR=1.0,
        dZ=z_values[1] - z_values[0],
        R=np.broadcast_to(np.array([[1.0], [2.0]]), (nx, ny)).copy(),
        Z=np.tile(z_values, (nx, 1)),
        _vgreen=np.arange(nx * ny, dtype=float).reshape(1, nx, ny),
        limiter_handler=SimpleNamespace(
            mask_inside_limiter=np.ones((nx, ny), dtype=bool)
        ),
        tokamak=SimpleNamespace(
            n_coils=1,
            n_active_coils=1,
            coils_dict={"active": {}},
            coil_resist=np.ones(1),
            coil_self_ind=np.eye(1),
        ),
    )


@pytest.mark.parametrize("use_subclass", [False, True])
@pytest.mark.parametrize(
    "invalid_input", ["off_center", "interior", "nonfinite", "mask"]
)
def test_invalid_symmetry_inputs_fail_before_modifying_equilibrium(
    use_subclass, invalid_input
):
    """Reject bad geometry before changing Greens or calling the static solver."""
    z_values = {
        "off_center": [0.0, 1.0, 2.0],
        "interior": [-1.0, 0.1, 1.0],
        "nonfinite": [-1.0, np.nan, 1.0],
        "mask": [-1.0, 0.0, 1.0],
    }[invalid_input]
    eq = _symmetry_test_equilibrium(z_values)
    if invalid_input == "mask":
        eq.limiter_handler.mask_inside_limiter[0, -1] = False
    greens = eq._vgreen.copy()
    resistance = eq.tokamak.coil_resist.copy()
    inductance = eq.tokamak.coil_self_ind.copy()
    static_solver = Mock()
    solver = SymmetryReducedSolver if use_subclass else nl_solver
    options = {} if use_subclass else {"force_up_down_symmetric": True}
    message = (
        "limiter mask symmetric" if invalid_input == "mask" else "Z grid symmetric"
    )
    with pytest.raises(ValueError, match=message):
        solver(object(), eq, static_solver, **options)
    static_solver.forward_solve.assert_not_called()
    np.testing.assert_array_equal(eq._vgreen, greens)
    np.testing.assert_array_equal(eq.tokamak.coil_resist, resistance)
    np.testing.assert_array_equal(eq.tokamak.coil_self_ind, inductance)


@pytest.mark.parametrize(
    "z_values", [[-1.0, 0.0, 1.0], [-1.0, -0.2, 0.2, 1.0], [-1.0, 2e-16, 1.0]]
)
def test_valid_symmetric_grid_and_mask_reach_static_solver(z_values):
    """Accept odd/even grid sizes and floating-point roundoff at the midplane."""
    eq = _symmetry_test_equilibrium(z_values)
    static_solver = Mock()
    static_solver.forward_solve.side_effect = RuntimeError("Reached static solver")
    with pytest.raises(RuntimeError, match="Reached static solver"):
        SymmetryReducedSolver(object(), eq, static_solver)
    static_solver.forward_solve.assert_called_once()


def test_unrestricted_solver_accepts_asymmetric_grid_and_mask():
    """The symmetry guards must not constrain ordinary asymmetric evolution."""
    eq = _symmetry_test_equilibrium([0.0, 1.0, 2.0])
    eq.limiter_handler.mask_inside_limiter[0, -1] = False
    greens = eq._vgreen.copy()
    static_solver = Mock()
    static_solver.forward_solve.side_effect = RuntimeError("Reached static solver")
    with pytest.raises(RuntimeError, match="Reached static solver"):
        nl_solver(object(), eq, static_solver)
    static_solver.forward_solve.assert_called_once()
    np.testing.assert_array_equal(eq._vgreen, greens)


@pytest.mark.parametrize(
    "reflection",
    [
        np.eye(3),
        np.array([[1.0, 0.1], [0.0, 1.0]]),
        np.full((2, 2), 0.5),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
    ],
)
def test_invalid_reflection_fails_before_mutation_or_gs(reflection):
    """Reflection validation happens before changing any part of the machine."""
    eq = _symmetry_test_equilibrium([-1.0, 0.0, 1.0])
    eq.tokamak.n_coils = 3
    eq.tokamak.coils_dict = {"active": {}, "upper": {}, "lower": {}}
    eq.tokamak.coil_resist = np.ones(3)
    eq.tokamak.coil_self_ind = np.eye(3)
    eq._vgreen = np.arange(18.0).reshape(3, 2, 3)
    original = eq._vgreen.copy()
    static_solver = Mock()
    with pytest.raises(ValueError, match="passive_reflection_operator"):
        SymmetryReducedSolver(
            object(), eq, static_solver, passive_reflection_operator=reflection
        )
    np.testing.assert_array_equal(eq._vgreen, original)
    static_solver.forward_solve.assert_not_called()


def test_nonlinear_state_acceptance_uses_the_solved_grid_flux():
    """An old named Green cache cannot change the accepted plasma distribution."""
    solver = object.__new__(nl_solver)
    solver.force_up_down_symmetric = True
    flux = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 3.0]])
    tokamak = SimpleNamespace(
        set_all_coil_currents=Mock(),
        getPsitokamak=lambda greens: greens[0],
        calcPsiFromGreens=Mock(side_effect=AssertionError("Used obsolete cache")),
    )
    solver.eq1 = SimpleNamespace(
        tokamak=tokamak,
        _vgreen=flux[None],
        _pgreen={"obsolete": True},
        set_plasma_psi=Mock(),
    )
    solver.eq2 = SimpleNamespace(tokamak=tokamak)
    solver.vessel_currents_vec = np.array([1.0])
    solver.assign_currents = Mock()
    solver.trial_currents = np.array([1.0, 2.0])
    solver.trial_plasma_psi = np.zeros_like(flux)
    solver.plasma_norm_factor = 1.0
    solver.eqR, solver.eqZ = np.ones_like(flux), np.zeros_like(flux)
    solver.profiles1 = SimpleNamespace(jtor=flux.copy(), Jtor=Mock())
    solver.NK = SimpleNamespace(port_critical=Mock())
    solver.limiter_handler = SimpleNamespace(
        Iy_from_jtor=lambda j: j.ravel(), normalize_sum=lambda j: j / j.sum()
    )
    solver.assign_trial_solution_state()
    solver.eq1.set_plasma_psi.assert_called_once_with(solver.trial_plasma_psi)
    np.testing.assert_array_equal(solver.profiles1.Jtor.call_args.args[2], flux)
    tokamak.calcPsiFromGreens.assert_not_called()


@pytest.mark.parametrize("linear_only", [True, False])
def test_symmetry_reduced_solver_matches_full_solver_with_gs_and_relinearisation(
    linear_only,
):
    """End-to-end SymmetryReducedSolver matches full nl_solver on an even machine."""
    from freegsnke import GSstaticsolver, build_machine, equilibrium_update
    from freegsnke.jtor_update import ConstrainPaxisIp
    from freegsnke.up_down_symmetry import prepare_up_down_symmetric_machine

    active = {
        "solenoid": {
            "R": [0.22, 0.22, 0.22, 0.22],
            "Z": [-0.45, -0.15, 0.15, 0.45],
            "dR": 0.04,
            "dZ": 0.04,
            "resistivity": 1.5e-8,
            "polarity": 1,
            "multiplier": 20,
        },
        "PF1_upper": {
            "R": [0.55],
            "Z": [0.55],
            "dR": 0.05,
            "dZ": 0.05,
            "resistivity": 1.5e-8,
            "polarity": 1,
            "multiplier": 40,
        },
        "PF1_lower": {
            "R": [0.55],
            "Z": [-0.55],
            "dR": 0.05,
            "dZ": 0.05,
            "resistivity": 1.5e-8,
            "polarity": 1,
            "multiplier": 40,
        },
        "PF2_upper": {
            "R": [1.45],
            "Z": [0.35],
            "dR": 0.05,
            "dZ": 0.05,
            "resistivity": 1.5e-8,
            "polarity": 1,
            "multiplier": 40,
        },
        "PF2_lower": {
            "R": [1.45],
            "Z": [-0.35],
            "dR": 0.05,
            "dZ": 0.05,
            "resistivity": 1.5e-8,
            "polarity": 1,
            "multiplier": 40,
        },
    }
    passive = [
        {
            "name": "shell_upper",
            "element": "shell",
            "R": [1.25, 1.33, 1.33, 1.25],
            "Z": [0.20, 0.20, 0.28, 0.28],
            "resistivity": 5.5e-7,
        },
        {
            "name": "shell_lower",
            "element": "shell",
            "R": [1.25, 1.33, 1.33, 1.25],
            "Z": [-0.28, -0.28, -0.20, -0.20],
            "resistivity": 5.5e-7,
        },
        {
            "name": "midplane_ring",
            "element": "ring",
            "R": [1.34, 1.40, 1.40, 1.34],
            "Z": [-0.05, -0.05, 0.05, 0.05],
            "resistivity": 5.5e-7,
        },
    ]
    theta = np.linspace(0.0, 2 * np.pi, 48, endpoint=True)
    limiter = [
        {"R": float(0.90 + 0.36 * np.cos(a)), "Z": float(0.36 * np.sin(a))}
        for a in theta
    ]
    prepared = prepare_up_down_symmetric_machine(
        active,
        passive,
        limiter_data=limiter,
        wall_data=limiter,
        verbose=False,
    )

    def make_state():
        tok = build_machine.tokamak(
            active_coils_data=prepared.even_active_coils_data,
            passive_coils_data=prepared.passive_coils_data,
            limiter_data=prepared.limiter_data,
            wall_data=prepared.wall_data,
        )
        eq = equilibrium_update.Equilibrium(
            tokamak=tok, Rmin=0.1, Rmax=1.7, Zmin=-0.8, Zmax=0.8, nx=33, ny=33
        )
        eq._vgreen = prepared.symmetrise_even_machine_greens(eq._vgreen)
        tok.coil_resist = prepared.symmetrise_even_machine_resistances(tok.coil_resist)
        tok.coil_self_ind = prepared.symmetrise_even_machine_square_operator(
            tok.coil_self_ind
        )
        tok.set_coil_current("solenoid", 1200.0)
        tok.set_coil_current("PF1", 400.0)
        tok.set_coil_current("PF2", -900.0)
        prof = ConstrainPaxisIp(
            eq=eq, paxis=3.0e3, Ip=1.2e5, fvac=0.4, alpha_m=1.8, alpha_n=1.2
        )
        static = GSstaticsolver.NKGSsolver(eq)
        static.solve(
            eq=eq,
            profiles=prof,
            constrain=None,
            target_relative_tolerance=1e-9,
            force_up_down_symmetric=True,
            suppress=True,
        )
        return eq, prof, static

    eq_red, prof_red, static_red = make_state()
    eq_full, prof_full, static_full = make_state()

    reduced = SymmetryReducedSolver(
        profiles=prof_red,
        eq=eq_red,
        GSStaticSolver=static_red,
        passive_reflection_operator=prepared.passive_reflection_operator,
        full_timestep=1e-3,
        max_internal_timestep=1e-3,
        plasma_resistivity=1e-6,
        max_mode_frequency=np.inf,
        verbose=False,
    )
    full = nl_solver(
        profiles=prof_full,
        eq=eq_full,
        GSStaticSolver=static_full,
        passive_reflection_operator=prepared.passive_reflection_operator,
        force_up_down_symmetric=True,
        remove_odd_passive_modes=False,
        full_timestep=1e-3,
        max_internal_timestep=1e-3,
        plasma_resistivity=1e-6,
        max_mode_frequency=np.inf,
        verbose=False,
    )
    assert reduced.n_passive_modes == 2
    assert full.n_passive_modes == 3

    voltages = np.array([250.0, -40.0, 60.0])
    for step in range(2):
        for solver in (reduced, full):
            solver.nlstepper(
                active_voltage_vec=voltages,
                linear_only=linear_only,
                no_GS=False,
                relinearise_threshold=0.0 if (linear_only and step == 1) else None,
                target_relative_tol_currents=1e-3,
                target_relative_tol_GS=1e-3,
                working_relative_tol_GS=5e-4,
                verbose=0,
            )
        rtol = 1e-7 if linear_only else 1e-5
        atol = 1e-5 if linear_only else 1e-3
        np.testing.assert_allclose(
            reduced.vessel_currents_vec,
            full.vessel_currents_vec,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            reduced.profiles1.Ip, full.profiles1.Ip, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            reduced.eq1.plasma_psi,
            full.eq1.plasma_psi,
            rtol=rtol,
            atol=1e-7 if linear_only else 1e-6,
        )
        np.testing.assert_allclose(
            reduced.eq1.plasma_psi,
            reduced.eq1.plasma_psi[:, ::-1],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            prepared.even_machine_reflection_operator @ reduced.vessel_currents_vec,
            reduced.vessel_currents_vec,
            rtol=0.0,
            atol=1e-10,
        )
