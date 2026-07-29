import numpy as np

from freegsnke.circuit_eq_metal import metal_currents


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
