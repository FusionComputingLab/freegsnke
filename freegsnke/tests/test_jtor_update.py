import os
from types import SimpleNamespace

import freegs4e
import numpy as np
import pytest

import freegsnke.jtor_update as jtor
from freegsnke import build_machine, equilibrium_update, nonlinear_solve


@pytest.fixture()
def create_machine():

    # build machine
    tokamak = build_machine.tokamak(
        active_coils_path=f"./machine_configs/test/active_coils.pickle",
        passive_coils_path=f"./machine_configs/test/passive_coils.pickle",
        limiter_path=f"./machine_configs/test/limiter.pickle",
        wall_path=f"./machine_configs/test/wall.pickle",
        magnetic_probe_path=f"./machine_configs/test/magnetic_probes.pickle",
    )

    # Creates equilibrium object and initializes it with
    # a "good" solution
    # plasma_psi = np.loadtxt('plasma_psi_example.txt')
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
    return eq, tokamak


def test_profiles_PaxisIp(create_machine):
    """Tests that the profiles have the xpt, opt and jtor attributes."""
    eq, tokamak = create_machine

    profiles = jtor.ConstrainPaxisIp(
        eq,
        8.1e3,  # Plasma pressure on axis [Pascals]
        6.2e5,  # Plasma current [Amps]
        0.5,  # vacuum f = R*Bt
        alpha_m=1.8,
        alpha_n=1.2,
    )

    profiles.Jtor(eq.R, eq.Z, eq.psi())
    assert (
        hasattr(profiles, "xpt")
        and hasattr(profiles, "opt")
        and hasattr(profiles, "jtor")
    ), "The profiles object does not have the xpt, opt and jtor attributes"


def test_profiles_BetapIp(create_machine):
    """Tests that the profiles have the xpt, opt and jtor attributes."""
    eq, tokamak = create_machine

    profiles = jtor.ConstrainBetapIp(
        eq,
        8.1e3,  # Plasma pressure on axis [Pascals]
        6.2e5,  # Plasma current [Amps]
        0.5,  # vacuum f = R*Bt
    )

    profiles.Jtor(eq.R, eq.Z, eq.psi())
    assert (
        hasattr(profiles, "xpt")
        and hasattr(profiles, "opt")
        and hasattr(profiles, "jtor")
    ), "The profiles object does not have the xpt, opt and jtor attributes"


class _MinimalLimiterHandler:
    """Limiter data required when instantiating a Lao85 profile."""

    def __init__(self, shape):
        self.mask_inside_limiter = np.ones(shape, dtype=bool)
        self.limiter_mask_out = np.zeros(shape, dtype=bool)

    def make_layer_mask(self, mask, layer_size=1):
        return np.zeros_like(mask, dtype=bool)


class _MinimalEquilibrium:
    """Small equilibrium-like object for testing Lao beta_p/li fitting."""

    def __init__(self):
        r = np.linspace(0.8, 1.2, 3)
        z = np.linspace(-0.2, 0.2, 3)
        self.R, self.Z = np.meshgrid(r, z, indexing="ij")
        self.R_1D = self.R[:, 0]
        self.Z_1D = self.Z[0, :]
        self.limiter_handler = _MinimalLimiterHandler(self.R.shape)
        self.plasma_psi = np.zeros_like(self.R)
        self.solved = False
        self._betap = 0.0
        self._li = 0.0

    def poloidalBeta1(self):
        return self._betap

    def internalInductance2(self):
        return self._li


class _LinearProfileMetricSolver:
    """Static-solver stand-in with deterministic profile-dependent metrics."""

    def forward_solve(self, eq, profiles, *args, **kwargs):
        alpha = profiles.alpha[:2]
        beta = profiles.beta[:2]
        eq._betap = alpha[0] + 2.0 * beta[0]
        eq._li = alpha[1] - beta[1]


class _LaoProfileLike:
    """Small Lao-like profile object used by the evolutive conversion test."""

    def __init__(self):
        self.Ip = 1.0
        self.fvac = 1.0
        self.Raxis = 1.0
        self.alpha_logic = True
        self.beta_logic = True
        self.alpha = np.array([1.0, -0.5, -0.5])
        self.beta = np.array([0.5, -0.25, -0.25])

    def initialize_profile(self):
        if self.alpha_logic and len(self.alpha) == 2:
            self.alpha = np.concatenate((self.alpha, [-np.sum(self.alpha)]))
        if self.beta_logic and len(self.beta) == 2:
            self.beta = np.concatenate((self.beta, [-np.sum(self.beta)]))


def test_fit_lao85_betap_li_ip_handles_logic_flags():
    """Fits two Lao alpha and beta coefficients before optional logic terms."""
    eq = _MinimalEquilibrium()
    solver = _LinearProfileMetricSolver()

    profiles, result = jtor.fit_lao85_betap_li_ip(
        eq,
        solver,
        Ip=1.0,
        fvac=1.0,
        betap=1.4,
        li=0.4,
        alpha=[0.0, 0.0],
        beta=[0.0, 0.0],
        alpha_logic=True,
        beta_logic=False,
        regularization_weight=0.0,
        use_metric_jacobian=True,
        optimizer_kwargs={"max_nfev": 20},
    )

    assert np.isclose(result.final_betap, result.target_betap, rtol=1e-3)
    assert np.isclose(result.final_li, result.target_li, rtol=1e-3)
    assert len(result.alpha) == 2
    assert len(result.beta) == 2
    assert len(result.effective_alpha) == 3
    assert len(result.effective_beta) == 2
    assert np.isclose(result.effective_alpha[-1], -np.sum(result.alpha))
    assert np.allclose(profiles.alpha, result.effective_alpha)
    assert np.allclose(profiles.beta, result.effective_beta)


def test_fit_lao85_betap_li_ip_requires_two_input_coefficients():
    """Rejects Lao coefficient vectors that already include logic terms."""
    eq = _MinimalEquilibrium()
    solver = _LinearProfileMetricSolver()

    with pytest.raises(ValueError, match="exactly two coefficients"):
        jtor.fit_lao85_betap_li_ip(
            eq,
            solver,
            Ip=1.0,
            fvac=1.0,
            betap=1.0,
            li=1.0,
            alpha=[1.0, 0.0, -1.0],
            beta=[1.0, 0.0],
        )


def test_nl_solver_converts_lao85_betap_li_targets(monkeypatch):
    """Converts physical Lao targets before the existing coefficient update path."""
    solver = object.__new__(nonlinear_solve.nl_solver)
    solver.profiles_type = "Lao85"
    solver.profiles1 = _LaoProfileLike()
    solver.profiles2 = _LaoProfileLike()
    solver.n_profiles_parameters_alpha = 2
    solver.n_profiles_parameters_beta = 2
    solver.eq1 = _MinimalEquilibrium()
    solver.eq2 = _MinimalEquilibrium()
    solver.eq2.plasma_psi[:] = 3.0
    solver.eq2.solved = True
    solver.NK = object()
    solver.fvac = 1.0
    solver.lao_betap_li_fit_results = []
    solver._lao_betap_li_metric_jacobian = None

    def fake_fit(eq, static_solver, **kwargs):
        eq.plasma_psi[:] = 9.0
        result = SimpleNamespace(
            alpha=np.array([2.0, -0.25]),
            beta=np.array([0.3, -0.1]),
            metric_jacobian=np.ones((2, 4)),
            target_betap=kwargs["betap"],
            target_li=kwargs["li"],
        )
        return object(), result

    monkeypatch.setattr(nonlinear_solve, "fit_lao85_betap_li_ip", fake_fit)

    solver.check_and_change_profiles(
        profiles_parameters={"betap": 0.7, "li": 0.8},
        profile_constraint_options={"optimizer_kwargs": {"max_nfev": 2}},
    )

    assert np.allclose(solver.profiles1.alpha, [2.0, -0.25, -1.75])
    assert np.allclose(solver.profiles1.beta, [0.3, -0.1, -0.2])
    assert np.allclose(solver.profiles2.alpha, solver.profiles1.alpha)
    assert np.allclose(solver.eq2.plasma_psi, 3.0)
    assert solver.eq2.solved is True
    assert solver.profiles_change_flag == 1
    assert len(solver.lao_betap_li_fit_results) == 1
    assert np.allclose(solver._lao_betap_li_metric_jacobian, np.ones((2, 4)))


def test_nl_solver_can_reuse_cached_metric_jacobian(monkeypatch):
    """Uses a cached beta_p/li metric Jacobian without another static fit."""
    solver = object.__new__(nonlinear_solve.nl_solver)
    solver.profiles_type = "Lao85"
    solver.profiles1 = _LaoProfileLike()
    solver.profiles2 = _LaoProfileLike()
    solver.n_profiles_parameters_alpha = 2
    solver.n_profiles_parameters_beta = 2
    solver.eq1 = _MinimalEquilibrium()
    solver.eq1._betap = 0.2
    solver.eq1._li = 0.6
    solver.eq2 = _MinimalEquilibrium()
    solver.NK = object()
    solver.fvac = 1.0
    solver.lao_betap_li_fit_results = []
    solver._lao_betap_li_metric_jacobian = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    def fake_fit(*args, **kwargs):
        raise AssertionError("The cached linearized metric path should be used.")

    monkeypatch.setattr(nonlinear_solve, "fit_lao85_betap_li_ip", fake_fit)

    solver.check_and_change_profiles(
        profiles_parameters={"betap": 0.25, "li": 0.55},
        profile_constraint_options={
            "linearized_metric_update": True,
            "linearized_metric_regularization": 0.0,
            "linearized_metric_step_fraction": 0.5,
        },
    )

    assert np.allclose(solver.profiles1.alpha, [1.025, -0.525, -0.5])
    assert np.allclose(solver.profiles1.beta, [0.5, -0.25, -0.25])
    assert np.allclose(solver.profiles2.alpha, solver.profiles1.alpha)
    assert np.allclose(solver.profiles2.beta, solver.profiles1.beta)
    assert solver.lao_betap_li_fit_results == []
    assert solver.profiles_change_flag == 1
