import os

import freegs4e
import numpy as np
import pytest

import freegsnke.jtor_update as jtor
from freegsnke import build_machine, equilibrium_update


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
    """Small equilibrium-like object for testing the static Lao fitter."""

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


def test_lao85_copy_handles_optional_solved_state():
    """Copies Lao85 profiles before and after GS-derived state is available."""
    eq = _MinimalEquilibrium()
    profiles = jtor.Lao85(
        eq,
        Ip=1.0,
        fvac=1.0,
        alpha=[1.0, -0.5],
        beta=[1.0, -0.5],
    )

    unsolved_copy = profiles.copy()
    assert not hasattr(unsolved_copy, "inputs")

    profiles.inputs = [0.1, 0.2, np.ones(eq.R.shape, dtype=bool)]
    profiles.psi_axis = 0.1
    profiles.psi_bndry = 0.2
    profiles.L = 3.0

    solved_copy = profiles.copy()
    assert solved_copy.inputs[0] == profiles.inputs[0]
    assert solved_copy.psi_axis == profiles.psi_axis
    assert solved_copy.psi_bndry == profiles.psi_bndry
    assert solved_copy.L == profiles.L


def test_sync_profile_flux_normalisation_uses_inputs():
    """Restores flux-normalisation attributes needed by Lao85 pressure()."""
    eq = _MinimalEquilibrium()
    profiles = jtor.Lao85(
        eq,
        Ip=1.0,
        fvac=1.0,
        alpha=[1.0, -0.5],
        beta=[1.0, -0.5],
    )
    profiles.inputs = [0.1, 0.2, np.ones(eq.R.shape, dtype=bool)]
    profiles.L = 3.0
    eq._profiles = profiles.copy()

    jtor._sync_profile_flux_normalisation(eq, profiles)

    assert profiles.psi_axis == 0.1
    assert profiles.psi_bndry == 0.2
    assert eq._profiles.psi_axis == 0.1
    assert eq._profiles.psi_bndry == 0.2
