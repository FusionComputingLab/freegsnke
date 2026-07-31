import concurrent.futures.process
import multiprocessing
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from freegsnke.nonlinear_solve import nl_solver


def bare_solver():
    """Construct only the state needed by the perturbation helpers."""
    solver = nl_solver.__new__(nl_solver)
    solver.NK = SimpleNamespace(rng=np.random.default_rng(seed=0))
    return solver


def test_update_starting_dI_uses_previously_accepted_amplitudes():
    solver = bare_solver()
    solver.final_dI_record = np.array([1e-3, -2e-3])
    solver.starting_dI = np.array([100.0, 200.0])

    updated = solver.update_starting_dI()

    np.testing.assert_array_equal(updated, [True, True])
    np.testing.assert_allclose(solver.starting_dI, [1e-3, 2e-3])


def test_update_starting_dI_preserves_invalid_columns():
    solver = bare_solver()
    solver.final_dI_record = np.array([1e-3, 0.0, np.nan])
    solver.starting_dI = np.array([100.0, 200.0, 300.0])

    updated = solver.update_starting_dI()

    np.testing.assert_array_equal(updated, [True, False, False])
    np.testing.assert_allclose(solver.starting_dI, [1e-3, 200.0, 300.0])


def test_update_starting_dI_ignores_incompatible_record():
    solver = bare_solver()
    solver.final_dI_record = np.array([1e-3])
    solver.starting_dI = np.array([100.0, 200.0])

    updated = solver.update_starting_dI()

    assert not np.any(updated)
    np.testing.assert_allclose(solver.starting_dI, [100.0, 200.0])


def test_starting_dI_rescaling_guard():
    limit = nl_solver._MAX_STARTING_DI_RATIO

    assert not nl_solver.starting_dI_requires_rescaling(10.0, 10.0 * limit * 0.99)
    assert not nl_solver.starting_dI_requires_rescaling(10.0, 10.0 / limit / 0.99)
    assert nl_solver.starting_dI_requires_rescaling(10.0, 10.0 * limit * 1.01)
    assert nl_solver.starting_dI_requires_rescaling(10.0, 10.0 / limit / 1.01)
    assert nl_solver.starting_dI_requires_rescaling(0.0, 1.0)
    assert nl_solver.starting_dI_requires_rescaling(10.0, np.nan)


def test_reused_starting_dI_uses_tighter_rescaling_guard():
    legacy_limit = nl_solver._MAX_STARTING_DI_RATIO
    reused_limit = nl_solver._MAX_REUSED_STARTING_DI_RATIO
    scaled_dI = 10.0 * 0.5 * (legacy_limit + reused_limit)

    assert not nl_solver.starting_dI_requires_rescaling(10.0, scaled_dI)
    assert nl_solver.starting_dI_requires_rescaling(
        10.0,
        scaled_dI,
        max_ratio=reused_limit,
    )


def fake_column_builder(self, column, *args):
    """Return enough information to identify one dispatched column."""
    return int(column)


class FakeProfiles:
    """Minimal conventional profile object for dispatcher tests."""

    alpha_m = 1.0
    alpha_n = 2.0
    betap = 0.5

    def copy(self):
        return SimpleNamespace(
            alpha_m=self.alpha_m,
            alpha_n=self.alpha_n,
            betap=self.betap,
        )


def configure_profile_dispatch(solver, workers):
    """Populate only the state needed by the profile-column dispatcher."""
    solver.n_linearization_workers = workers
    solver.n_profiles_parameters = 3
    solver.profiles_param = "betap"
    solver._build_dIydtheta_column = MethodType(fake_column_builder, solver)
    solver.check_and_change_profiles = lambda _: None


def test_linearization_solve_state_resets_flux_profiles_and_rng():
    solver = bare_solver()
    solver.eq1 = SimpleNamespace(plasma_psi=np.arange(4.0).reshape(2, 2))
    solver.eq2 = SimpleNamespace(plasma_psi=np.zeros((2, 2)))
    solver.profiles1 = FakeProfiles()
    solver.profiles2 = None
    solver._linearization_rng_state = solver.NK.rng.bit_generator.state
    expected_random_value = np.random.default_rng(seed=0).random()
    solver.NK.rng.random(5)

    solver._reset_linearization_solve_state()

    np.testing.assert_array_equal(solver.eq2.plasma_psi, solver.eq1.plasma_psi)
    assert solver.profiles2 is not solver.profiles1
    assert solver.NK.rng.random() == expected_random_value


def test_dIydI_columns_use_serial_dispatch_by_default():
    solver = bare_solver()
    solver.n_linearization_workers = 1
    solver.arange_currents = np.array([2, 4])
    solver._build_dIydI_column = MethodType(fake_column_builder, solver)

    results = solver._build_dIydI_columns(
        1e-8,
        False,
        np.zeros(5, dtype=bool),
        lambda _: np.array([0.0]),
    )

    assert results == [2, 4]


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="Parallel linearization requires the fork start method.",
)
def test_dIydI_columns_preserve_order_with_multiple_workers(monkeypatch):
    solver = bare_solver()
    solver.n_linearization_workers = 2
    solver.arange_currents = np.array([4, 2, 3])
    solver._build_dIydI_column = MethodType(fake_column_builder, solver)
    monkeypatch.setattr(
        concurrent.futures.process,
        "_check_system_limits",
        lambda: None,
    )

    results = solver._build_dIydI_columns(
        1e-8,
        False,
        np.zeros(5, dtype=bool),
        lambda _: np.array([0.0]),
    )

    assert results == [4, 2, 3]


def test_profile_parameter_shift_preserves_lao_constraints():
    solver = bare_solver()
    solver.profiles_param = None
    solver.n_profiles_parameters_alpha = 2
    profiles = SimpleNamespace(
        alpha=np.array([1.0, 2.0, -3.0]),
        beta=np.array([4.0, -4.0]),
        alpha_logic=True,
        beta_logic=True,
    )

    alpha_shift = solver._profile_parameters_for_column(profiles, 1, 0.25)
    beta_shift = solver._profile_parameters_for_column(profiles, 2, 0.5)

    np.testing.assert_allclose(alpha_shift["alpha"], [1.0, 2.25, -3.25])
    np.testing.assert_allclose(alpha_shift["beta"], profiles.beta)
    np.testing.assert_allclose(beta_shift["alpha"], profiles.alpha)
    np.testing.assert_allclose(beta_shift["beta"], [4.5, -4.5])
    np.testing.assert_allclose(profiles.alpha, [1.0, 2.0, -3.0])
    np.testing.assert_allclose(profiles.beta, [4.0, -4.0])


def test_profile_parameter_shift_for_conventional_profile():
    solver = bare_solver()
    solver.profiles_param = "betap"
    profiles = FakeProfiles()

    shifted = solver._profile_parameters_for_column(profiles, 2, 0.05)

    assert shifted == {"alpha_m": 1.0, "alpha_n": 2.0, "betap": 0.55}


def test_dIydtheta_columns_use_serial_dispatch_by_default():
    solver = bare_solver()
    configure_profile_dispatch(solver, workers=1)

    results = solver._build_dIydtheta_columns(
        FakeProfiles(),
        1e-8,
        np.ones(3),
        lambda _: np.array([0.0]),
    )

    assert results == [0, 1, 2]


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="Parallel linearization requires the fork start method.",
)
def test_dIydtheta_columns_preserve_order_with_multiple_workers(monkeypatch):
    solver = bare_solver()
    configure_profile_dispatch(solver, workers=2)
    monkeypatch.setattr(
        concurrent.futures.process,
        "_check_system_limits",
        lambda: None,
    )

    results = solver._build_dIydtheta_columns(
        FakeProfiles(),
        1e-8,
        np.ones(3),
        lambda _: np.array([0.0]),
    )

    assert results == [0, 1, 2]
