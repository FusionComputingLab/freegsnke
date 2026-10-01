"""Behavioral checks of acceleration, recovery and state restoration."""

import numpy as np
import pytest

from freegsnke.anderson import AndersonOptions, safeguarded_solve


def run(residual, newton, **kwargs):
    """Solve a two-component fixture through the real recovery policy."""
    settings = dict(
        options=AndersonOptions(),
        tolerance=1e-9,
        max_iterations=100,
        handover=0.0,
        max_rel_update_size=10.0,
    )
    settings.update(kwargs)
    return safeguarded_solve(
        np.array([0.0, 1.0]), residual, newton, lambda f, x: np.max(abs(f)), **settings
    )


def test_accelerates_slow_fixed_point():
    """Finish a contraction that plain damped Picard cannot in this budget."""
    target = np.array([1.0, 2.0])
    x, f, info = run(lambda x: 0.01 * (x - target), lambda x, f: -f)
    assert info["reason"] == "converged"
    np.testing.assert_allclose(x, target, atol=1e-7)
    assert len(info["events"]) < 30


def test_failed_newton_recovers():
    """Reject an uphill Newton step and recover without a Jacobian."""
    target = np.array([1.0, 2.0])
    x, f, info = run(lambda x: x - target, lambda x, f: f, handover=10.0)
    assert info["reason"] == "converged"
    assert any(e["event"] == "anderson_recovery" for e in info["events"])
    np.testing.assert_allclose(x, target, atol=1e-8)


def test_geometry_failure_restores_and_stops():
    """A rejected invalid geometry cannot remain in derived profile state."""
    state = {}
    start = np.array([0.0, 1.0])

    def residual(x):
        state["x"] = x.copy()
        if not np.array_equal(x, start):
            raise ValueError("No O-point")
        return np.ones(2)

    x, f, info = run(
        residual,
        lambda x, f: -f,
        handover=10.0,
        options=AndersonOptions(max_backtracks=3),
    )
    assert info["reason"] == "stalled"
    np.testing.assert_array_equal(x, start)
    np.testing.assert_array_equal(state["x"], start)
    assert info["evaluations"] < 30


def test_unexpected_error_propagates_after_restoration():
    """Do not disguise programming errors as geometry failures."""
    state = {}

    def residual(x):
        state["x"] = x.copy()
        if x[0] != 0:
            raise KeyError("broken callback")
        return np.ones(2)

    with pytest.raises(KeyError):
        run(residual, lambda x, f: -f)
    np.testing.assert_array_equal(state["x"], [0.0, 1.0])


def test_newton_probe_state_restored():
    """The final callback state belongs to the accepted solution."""
    state = {}
    target = np.array([1.0, 2.0])

    def residual(x):
        state["x"] = x.copy()
        return x - target

    def newton(x, f):
        residual(x + 100.0)
        return -f

    x, f, info = run(residual, newton, handover=10.0)
    assert info["reason"] == "converged"
    np.testing.assert_array_equal(state["x"], x)


def test_projection_cannot_hide_full_residual():
    """Even restriction must not certify a nonzero odd residual."""
    x, f, info = run(
        lambda x: np.array([1.0, -1.0]),
        lambda x, f: -f,
        project=lambda x: np.full_like(x, x.mean()),
        max_iterations=4,
    )
    assert info["reason"] != "converged"
    assert info["relative_error"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history": 0},
        {"history": 1.5},
        {"damping": 0},
        {"damping": np.nan},
        {"max_backtracks": -1},
        {"regularization": -1},
        {"stagnation_ratio": 1.1},
    ],
)
def test_bad_settings(kwargs):
    """Invalid settings fail before any physical state is touched."""
    with pytest.raises(ValueError):
        AndersonOptions(**kwargs)


def test_scipy_history_can_start_with_uphill_picard_step():
    """A noncontractive fixed point needs history before a descent step exists."""
    target = np.array([1.0, 2.0])
    x, f, info = run(lambda x: target - x, lambda x, f: -f, handover=10.0)
    assert info["reason"] == "converged"
    assert any(e["event"] == "anderson_recovery" for e in info["events"])
    assert any(e.get("method") == "scipy_anderson" for e in info["events"])
    assert info["norm_history"][1] > info["norm_history"][0]
    np.testing.assert_allclose(x, target, atol=1e-8)


def test_anderson_budget_restores_best_valid_state():
    """An exhausted uphill phase must not replace a better retained field."""
    state = {}

    def residual(x):
        state["x"] = x.copy()
        return np.array([1.0, 2.0]) - x

    x, f, info = run(residual, lambda x, f: f, max_iterations=1)
    assert info["reason"] == "iteration_limit"
    assert info["iterations"] == 1
    np.testing.assert_array_equal(x, [0.0, 1.0])
    np.testing.assert_array_equal(state["x"], x)


def test_newton_resumes_after_bounded_anderson_phase():
    """A one-step recovery phase returns control to a now usable NK step."""
    calls = []
    target = np.array([1.0, 2.0])

    def newton(x, f):
        calls.append(x.copy())
        return f if len(calls) == 1 else -f

    x, f, info = run(
        lambda x: x - target,
        newton,
        handover=10.0,
        options=AndersonOptions(recovery_iterations=1),
    )
    assert info["reason"] == "converged"
    assert len(calls) == 2
    methods = [e.get("method") for e in info["events"] if e["event"] == "accepted"]
    assert methods == ["scipy_anderson", "newton"]
