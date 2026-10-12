"""Check an odd GS tangent against independent signed nonlinear solves.

All inputs are manufactured here. FreeGS4E provides the spatial inverse;
the response solver sees only a callable mapping current density to flux.
See README.md for equations, units and the limits of this verification.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import root
from freegs4e.gradshafranov import Greens, mu0
from freegs4e.gs_solver import GSDSTSolver
from freegsnke.gs_response import solve_odd_response


def main(output_dir):
    r, z = np.linspace(1, 3, 17), np.linspace(-1, 1, 17)
    R, Z = np.meshgrid(r, z, indexing="ij")
    shape = R.shape
    reflection = np.arange(R.size).reshape(shape)[:, ::-1].ravel()
    odd = lambda field: (field - field[reflection]) / 2
    area = (r[1] - r[0]) * (z[1] - z[0])
    boundary = np.zeros(shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    interior = ~boundary
    solver = GSDSTSolver(R, Z, order=2)
    # The source vanishes at the computational boundary. The full Green sum
    # below supplies free-boundary flux; it makes no symmetry approximation.
    green = (
        Greens(
            R[boundary, None],
            Z[boundary, None],
            R[interior][None, :],
            Z[interior][None, :],
        )
        * area
    )

    def inverse(current):
        j = np.asarray(current).reshape(shape)
        # Upstream uses +Delta*: negative interior source, positive edge flux.
        rhs = -mu0 * R * j
        rhs[boundary] = green @ j[interior]
        return solver(np.zeros(shape), rhs).ravel()

    envelope = np.exp(-(((R - 2) / 0.35) ** 2) - (Z / 0.4) ** 2)
    envelope[boundary] = 0
    envelope = envelope.ravel()
    j0 = 1e6 * envelope
    psi0 = inverse(j0)
    # Opposite external unit ring currents give odd applied flux per ampere.
    g = odd((Greens(3.5, 0.8, R, Z) - Greens(3.5, -0.8, R, Z)).ravel())
    coefficient = 1e5 * envelope
    cubic = 0.1 * coefficient / np.ptp(psi0) ** 2

    def current(psi, amplitude):
        # The manufactured source has equilibrium (psi0,j0) at amplitude zero.
        # Its derivative there is coefficient; the cubic term exercises the
        # difference between a finite perturbation and the implicit tangent.
        displacement = psi + amplitude * g - psi0
        return j0 + coefficient * displacement + cubic * displacement**3

    def residual(psi, amplitude):
        return psi - inverse(current(psi, amplitude))

    response = solve_odd_response(inverse, coefficient, g, reflection)
    dj = response["current_response"]
    checks = []
    for step in (4000.0, 2000.0, 1000.0):
        currents = []
        for sign in (1, -1):
            amplitude = sign * step
            guess = psi0 + amplitude * response["plasma_response"]
            solved = root(
                lambda psi: residual(psi, amplitude),
                guess,
                method="hybr",
                options={"xtol": 1e-10},
            )
            error = np.linalg.norm(residual(solved.x, amplitude)) / np.linalg.norm(psi0)
            if not np.isfinite(error) or error > 1e-11:
                raise RuntimeError("Nonlinear perturbed solve failed")
            currents.append(current(solved.x, amplitude))
        # Central differences have O(step**2) truncation error for a smooth
        # source. Explicit odd projection matches the sector solved above.
        fd = odd((currents[0] - currents[1]) / (2 * step))
        checks.append(
            dict(
                step_A=step,
                relative_current_error=float(
                    np.linalg.norm(fd - dj) / np.linalg.norm(dj)
                ),
            )
        )
    if checks[-1]["relative_current_error"] >= checks[0]["relative_current_error"] / 8:
        raise RuntimeError("Signed nonlinear checks did not approach the tangent")
    # Reciprocity converts per-radian flux to linkage with the factor 2*pi.
    # This is an illustrative coupling, without a conductor dynamics model.
    result = dict(
        tangent_relative_residual=response["relative_residual"],
        linkage_H=float(2 * np.pi * area * np.dot(g, dj)),
        finite_differences=checks,
        scope="Manufactured fixed-grid differentiation check; no spatial convergence claim.",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "response.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("response-output"))
    main(parser.parse_args().output_dir)
