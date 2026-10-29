import numpy as np
import pytest

from freegsnke.inverse import _solve_regularized_lstsq


def test_regularized_lstsq_matches_normal_equations_when_well_conditioned():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(30, 6))
    target = rng.normal(size=30)
    regularization = np.geomspace(1e-8, 1e-3, matrix.shape[1])

    expected = np.linalg.solve(
        matrix.T @ matrix + np.diag(regularization), matrix.T @ target
    )
    result = _solve_regularized_lstsq(matrix, target, regularization)

    assert np.allclose(result, expected, rtol=1e-12, atol=1e-14)


def test_regularized_lstsq_is_accurate_for_ill_conditioned_pairwise_system():
    rng = np.random.default_rng(8)
    n_points = 20
    n_variables = 8
    basis, _ = np.linalg.qr(rng.normal(size=(n_points, n_variables)))
    basis, _ = np.linalg.qr(basis - np.mean(basis, axis=0))
    rotation, _ = np.linalg.qr(rng.normal(size=(n_variables, n_variables)))
    point_response = basis @ np.diag(np.geomspace(1.0, 1e-8, n_variables)) @ rotation.T

    row, column = np.triu_indices(n_points, 1)
    pairwise_response = point_response[row] - point_response[column]
    expected = rng.normal(size=n_variables)
    target = pairwise_response @ expected

    result = _solve_regularized_lstsq(pairwise_response, target, np.zeros(n_variables))

    assert np.linalg.cond(pairwise_response) > 1e7
    assert np.allclose(result, expected, rtol=1e-7, atol=1e-8)


def test_regularized_lstsq_returns_minimum_norm_rank_deficient_solution():
    matrix = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]])
    target = np.array([1.0, 2.0, -1.0])

    result = _solve_regularized_lstsq(matrix, target, np.zeros(2))

    assert np.allclose(result, [0.5, 0.5])
    assert np.allclose(matrix @ result, target)


@pytest.mark.parametrize("value", [-1e-12, np.nan, np.inf])
def test_regularized_lstsq_rejects_invalid_regularization(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _solve_regularized_lstsq(np.eye(2), np.ones(2), np.array([value, 0.0]))
