import numpy as np

from freegsnke import virtual_circuits


class _FakeSolver:
    def __init__(self):
        self.calls = 0

    def forward_solve(self, eq, profiles, target_relative_tolerance, suppress=True):
        self.calls += 1
        eq.solved = True
        profiles.jtor = np.ones((2, 2))


class _FakeEquilibrium:
    def __init__(self):
        self.solved = False

    def poloidalBeta1(self):
        return 0.2

    def internalInductance2(self):
        return 0.8


class _FakeLaoProfile:
    Ip = 1.5
    alpha = np.array([10.0, -2.0, -8.0])
    beta = np.array([0.4, -0.1, -0.3])
    alpha_logic = True
    beta_logic = True
    Raxis = 1.0

    def fvac(self):
        return 0.5


def test_profile_adjuster_can_mark_equilibrium_as_solved():
    """A VC profile adjuster may solve the equilibrium itself."""
    handler = virtual_circuits.VirtualCircuitHandling()
    solver = _FakeSolver()
    handler.define_solver(solver)
    eq = _FakeEquilibrium()
    profiles = _FakeLaoProfile()

    def adjuster(eq, profiles, solver, target_relative_tolerance):
        eq.solved = True
        profiles.adjusted = True
        return profiles, True

    returned_profiles = handler.solve_GS_with_profile_adjuster(
        eq,
        profiles,
        target_relative_tolerance=1e-6,
        profile_adjuster=adjuster,
    )

    assert returned_profiles is profiles
    assert returned_profiles.adjusted
    assert eq.solved
    assert solver.calls == 0


def test_profile_adjuster_returning_profile_is_solved_by_handler():
    """If the adjuster only updates profiles, the VC handler still solves GS."""
    handler = virtual_circuits.VirtualCircuitHandling()
    solver = _FakeSolver()
    handler.define_solver(solver)
    eq = _FakeEquilibrium()
    profiles = _FakeLaoProfile()

    def adjuster(eq, profiles, solver, target_relative_tolerance):
        profiles.adjusted = True
        return profiles

    returned_profiles = handler.solve_GS_with_profile_adjuster(
        eq,
        profiles,
        target_relative_tolerance=1e-6,
        profile_adjuster=adjuster,
    )

    assert returned_profiles.adjusted
    assert hasattr(returned_profiles, "jtor")
    assert eq.solved
    assert solver.calls == 1


def test_make_lao85_betap_li_profile_adjuster_uses_reference_targets(monkeypatch):
    """The Lao fixed-beta_p/li adjuster forwards captured targets to the fitter."""
    recorded = []
    reusable_jacobian = np.arange(8.0).reshape(2, 4)

    class _FakeFitResult:
        metric_jacobian = reusable_jacobian

    def fake_fit_lao85_betap_li_ip(eq, solver, **kwargs):
        recorded.append(kwargs)
        return "fitted-profile", _FakeFitResult()

    monkeypatch.setattr(
        virtual_circuits,
        "fit_lao85_betap_li_ip",
        fake_fit_lao85_betap_li_ip,
    )

    reference_eq = _FakeEquilibrium()
    adjuster = virtual_circuits.make_lao85_betap_li_profile_adjuster(
        reference_eq,
        optimizer_kwargs={"max_nfev": 3},
    )

    profile, solved = adjuster(
        _FakeEquilibrium(),
        _FakeLaoProfile(),
        _FakeSolver(),
        vc_target_relative_tolerance=1e-5,
    )
    adjuster(
        _FakeEquilibrium(),
        _FakeLaoProfile(),
        _FakeSolver(),
        vc_target_relative_tolerance=1e-5,
    )

    assert profile == "fitted-profile"
    assert solved is True
    assert adjuster.betap == reference_eq.poloidalBeta1()
    assert adjuster.li == reference_eq.internalInductance2()
    assert len(adjuster.fit_results) == 2
    assert np.allclose(adjuster.reusable_metric_jacobian[0], reusable_jacobian)
    assert recorded[0]["metric_jacobian"] is None
    assert np.allclose(recorded[1]["metric_jacobian"], reusable_jacobian)
    assert recorded[0]["betap"] == reference_eq.poloidalBeta1()
    assert recorded[0]["li"] == reference_eq.internalInductance2()
    assert recorded[0]["li_method"] == "internalInductance2"
    assert recorded[0]["alpha"].shape == (2,)
    assert recorded[0]["beta"].shape == (2,)
    assert recorded[0]["target_relative_tolerance"] == 1e-5
    assert recorded[0]["use_metric_jacobian"] is True
