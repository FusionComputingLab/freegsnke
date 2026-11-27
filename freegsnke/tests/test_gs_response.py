import unittest
from unittest.mock import patch

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

    def test_invalid_shapes_rejected(self):
        """Check that an exception is raised for inputs that aren't 1D arrays of equal length."""
        cases = (
            ("scalar coefficient", 1.0, [-1.0, 1.0], [1, 0]),
            ("matrix coefficient", [[1.0, 1.0]], [-1.0, 1.0], [1, 0]),
            ("forcing length mismatch", [1.0, 1.0], [-1.0], [1, 0]),
            ("matrix forcing", [1.0, 1.0], [[-1.0, 1.0]], [1, 0]),
            ("reflection length mismatch", [1.0, 1.0], [-1.0, 1.0], [0]),
            ("matrix reflection", [1.0, 1.0], [-1.0, 1.0], [[1, 0]]),
        )
        for name, c, g, p in cases:
            with (
                self.subTest(case=name),
                self.assertRaisesRegex(
                    ValueError, "Expected equal-length one-dimensional arrays"
                ),
            ):
                solve_odd_response(lambda j: j * 0.2, c, g, p)

    def test_invalid_reflections_rejected(self):
        """Check that an exception is raised for invalid reflection indices."""
        cases = (
            ("non-integer indices", [2.0, 1.0, 0.0]),
            ("negative index", [-1, 1, 0]),
            ("out-of-range index", [3, 1, 0]),
            ("repeated index", [0, 0, 2]),
            ("non-involutive permutation", [1, 2, 0]),
        )
        for name, p in cases:
            with (
                self.subTest(case=name),
                self.assertRaisesRegex(
                    ValueError, "Reflection must be an involutive permutation"
                ),
            ):
                solve_odd_response(lambda j: j * 0.2, np.ones(3), [-1.0, 0.0, 1.0], p)

    def test_nonfinite_inputs_rejected(self):
        """Check that an exception is raised for non-finite coefficients and/or vacuum forcing."""
        for field in ("coefficient", "vacuum_response"):
            for nonfinite_value in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field, nonfinite_value=nonfinite_value):
                    inputs = {
                        "coefficient": np.ones(2),
                        "vacuum_response": np.array([-1.0, 1.0]),
                    }
                    # Overwrite the specified field with a non-finite value.
                    inputs[field][0] = nonfinite_value
                    with self.assertRaisesRegex(ValueError, "Nonfinite response input"):
                        solve_odd_response(
                            lambda j: j * 0.2, reflection=[1, 0], **inputs
                        )

    def test_uneven_coefficient_rejected(self):
        """Check that an exception is raised if the source coefficients differ at reflected points."""
        uneven_coefficients = [1.0, 2.0]
        with self.assertRaisesRegex(ValueError, "Source coefficient must be even"):
            solve_odd_response(
                lambda j: j * 0.2, uneven_coefficients, [-1.0, 1.0], [1, 0]
            )

    def test_even_forcing_rejected(self):
        """Check that an exception is raised if the forcing term is even."""
        even_forcing = np.ones(2)
        with self.assertRaisesRegex(ValueError, "Vacuum forcing must be odd"):
            solve_odd_response(
                lambda j: j * 0.2, np.ones(2), even_forcing, np.array([1, 0])
            )

    def test_gmres_failures_rejected(self):
        """Check that an exception is raised for GMRES failures."""
        failure_statuses = (-1, 1)
        coefficient = np.ones(2)
        vacuum_response = np.array([-1.0, 1.0])
        reflection = np.array([1, 0])
        response = np.array([-0.25, 0.25])
        for status in failure_statuses:
            with (
                self.subTest(status=status),
                patch(
                    "freegsnke.gs_response.gmres",
                    return_value=(response, status),
                ),
                self.assertRaisesRegex(
                    RuntimeError, f"Odd tangent did not converge: info={status},"
                ),
            ):
                solve_odd_response(
                    lambda j: j * 0.2, coefficient, vacuum_response, reflection
                )

    def test_nonfinite_residual_rejected(self):
        """Check that an exception is raised if the response is non-finite."""
        nonfinite_response = np.full(2, np.nan)
        with (
            patch(
                "freegsnke.gs_response.gmres",
                return_value=(nonfinite_response, 0),
            ),
            self.assertRaisesRegex(RuntimeError, "info=0, residual=nan"),
        ):
            solve_odd_response(lambda j: j * 0.2, np.ones(2), [-1.0, 1.0], [1, 0])

    def test_excessive_residual_rejected(self):
        """Check that an exception is raised if the equation residual is excessive (independent of the GMRES success status)."""
        inaccurate_response = np.zeros(2)
        with (
            patch("freegsnke.gs_response.gmres", return_value=(inaccurate_response, 0)),
            self.assertRaisesRegex(RuntimeError, r"info=0, residual=1\.0"),
        ):
            solve_odd_response(lambda j: j * 0.2, np.ones(2), [-1.0, 1.0], [1, 0])

    def test_singular_tangent_fails(self):
        """Check that an exception is raised if the tangent equation is singular."""
        # G = I and C = I make the odd-sector tangent (I - G C) singular
        identity_inverse = lambda j: j
        unit_coefficient = np.ones(2)
        with self.assertRaises(RuntimeError):
            solve_odd_response(
                identity_inverse,
                unit_coefficient,
                np.array([-1.0, 1.0]),
                np.array([1, 0]),
                maxiter=1,
            )
