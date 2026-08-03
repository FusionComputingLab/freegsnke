import numpy as np

from freegsnke.circuit_eq_metal import metal_currents
from freegsnke.linear_solve import linear_solver


class _Tokamak:
    n_active_coils = 1
    n_coils = 5
    coil_resist = np.ones(n_coils)

    reflection = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    parity_basis = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, -1.0],
        ]
    ) / np.sqrt(2.0)
    passive_inductance = parity_basis @ np.diag([1.0, 2.0, 3.0, 4.0]) @ parity_basis.T
    coil_self_ind = np.zeros((n_coils, n_coils))
    coil_self_ind[0, 0] = 1.0
    coil_self_ind[1:, 1:] = passive_inductance


class _Equilibrium:
    tokamak = _Tokamak()


def test_odd_passive_modes_are_removed_before_mode_selection():
    currents = metal_currents(
        eq=_Equilibrium(),
        flag_vessel_eig=True,
        flag_plasma=False,
        max_mode_frequency=np.inf,
        max_internal_timestep=1e-3,
        full_timestep=1e-3,
        passive_reflection_operator=_Tokamak.reflection,
    )

    assert np.count_nonzero(currents.normal_modes.passive_mode_parity > 0) == 2
    assert currents.n_independent_vars == 3
    np.testing.assert_allclose(
        _Tokamak.reflection @ currents.P[1:],
        currents.P[1:],
        atol=1e-12,
    )

    greens = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    mode_greens = currents.normal_modes.normal_modes_greens(greens, currents.P)
    np.testing.assert_allclose(
        mode_greens,
        np.einsum("im,irz->mrz", currents.P, greens),
    )


def test_odd_passive_modes_can_be_retained_for_full_mode_reference():
    currents = metal_currents(
        eq=_Equilibrium(),
        flag_vessel_eig=True,
        flag_plasma=False,
        max_mode_frequency=np.inf,
        max_internal_timestep=1e-3,
        full_timestep=1e-3,
        passive_reflection_operator=_Tokamak.reflection,
        remove_odd_passive_modes=False,
    )

    assert np.count_nonzero(currents.normal_modes.passive_mode_parity < 0) == 2
    assert currents.n_independent_vars == _Tokamak.n_coils
    np.testing.assert_allclose(currents.P, currents.normal_modes.Pmatrix)


def test_even_reduced_dynamics_match_full_symmetric_dynamics():
    """Even initial data evolve identically with or without odd passive modes."""

    def metal_model(remove_odd):
        return metal_currents(
            eq=_Equilibrium(),
            flag_vessel_eig=True,
            flag_plasma=False,
            max_mode_frequency=np.inf,
            max_internal_timestep=1e-3,
            full_timestep=1e-3,
            passive_reflection_operator=_Tokamak.reflection,
            remove_odd_passive_modes=remove_odd,
            verbose=False,
        )

    full = metal_model(remove_odd=False)
    reduced = metal_model(remove_odd=True)

    metal_reflection = np.eye(_Tokamak.n_coils)
    metal_reflection[1:, 1:] = _Tokamak.reflection
    plasma_reflection = np.eye(4)[::-1]
    rng = np.random.default_rng(4)

    mey_seed = rng.normal(scale=0.02, size=(_Tokamak.n_coils, 4))
    mey = 0.5 * (
        mey_seed + metal_reflection @ mey_seed @ plasma_reflection
    )
    response_seed = rng.normal(scale=0.02, size=(4, _Tokamak.n_coils))
    physical_response = 0.5 * (
        response_seed
        + plasma_reflection @ response_seed @ metal_reflection
    )
    ip_response = np.array([0.01, 0.02, 0.02, 0.01])
    hat_iy = np.array([0.3, 0.2, 0.2, 0.3])
    myy_hat_iy = np.array([0.1, 0.05, 0.05, 0.1])

    def coupled_solver(model):
        solver = linear_solver(
            coil_numbers=(_Tokamak.n_active_coils, _Tokamak.n_coils),
            Lambdam1=model.Lambdam1,
            P=model.P,
            Pm1=model.Pm1,
            Rm1=np.diag(model.Rm1),
            Mey=mey,
            plasma_norm_factor=1.0,
            plasma_resistance_1d=np.ones(4),
            max_internal_timestep=1e-3,
            full_timestep=1e-3,
        )
        transformed_response = np.column_stack(
            (physical_response @ model.P, ip_response)
        )
        solver.set_linearization_point(
            transformed_response,
            np.empty((4, 0)),
            hat_iy,
            myy_hat_iy,
        )
        return solver

    full_solver = coupled_solver(full)
    reduced_solver = coupled_solver(reduced)
    physical_initial = np.array([0.3, 1.0, 1.0, -0.5, -0.5])
    full_state = np.r_[full.Pm1 @ physical_initial, 0.7]
    reduced_state = np.r_[reduced.Pm1 @ physical_initial, 0.7]

    for step in range(20):
        voltage = np.array([0.2 * np.cos(step)])
        full_state = full_solver.stepper(full_state, voltage, np.empty(0))
        reduced_state = reduced_solver.stepper(
            reduced_state, voltage, np.empty(0)
        )

        full_physical = full.P @ full_state[:-1]
        reduced_physical = reduced.P @ reduced_state[:-1]
        np.testing.assert_allclose(full_physical, reduced_physical, atol=2e-14)
        np.testing.assert_allclose(full_state[-1], reduced_state[-1], atol=2e-14)
        np.testing.assert_allclose(
            metal_reflection @ full_physical,
            full_physical,
            atol=2e-14,
        )
