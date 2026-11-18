import unittest

import numpy as np

from freegsnke.gs_response import solve_odd_response


class ResponseTest(unittest.TestCase):
    def test_dense_reference(self):
        """Check plasma and current responses against an independent dense solve."""
        # Tolerance for comparison
        atol = 1e-12
        G = np.array([[0.2, 0.03, 0.01], [0.03, 0.2, 0.03], [0.01, 0.03, 0.2]])
        c = np.array([0.4, 0.7, 0.4])
        g = np.array([-1.0, 0.0, 1.0])
        p = np.array([2, 1, 0])
        r = solve_odd_response(lambda j: G @ j, c, g, p)
        # Independently solve
        #   (I - G diag(c))v = G(cg)
        # and check that the result is identical to within the specified tolerance
        v = np.linalg.solve(np.eye(3) - G @ np.diag(c), G @ (c * g))
        np.testing.assert_allclose(r["plasma_response"], v, atol=atol)
        np.testing.assert_allclose(r["current_response"], c * (v + g), atol=atol)

    def test_rejects_even_forcing(self):
        """Check that an even forcing term raises a ValueError."""
        with self.assertRaises(ValueError):
            solve_odd_response(
                lambda j: j * 0.2, np.ones(2), np.ones(2), np.array([1, 0])
            )

    def test_singular_tangent_fails(self):
        """Check that a singular tangent equation raises a RuntimeError."""
        with self.assertRaises(RuntimeError):
            solve_odd_response(
                lambda j: j,
                np.ones(2),
                np.array([-1.0, 1.0]),
                np.array([1, 0]),
                maxiter=1,
            )
