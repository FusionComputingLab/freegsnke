"""Response calculations for Grad–Shafranov equilibria."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def solve_odd_response(
    inverse, coefficient, vacuum_response, reflection, rtol=1e-9, maxiter=20
):
    """Solve the static odd response about a reflection-symmetric equilibrium.

    Solve ``(I - P_o G C P_o)v = P_o G C g`` and return
    ``dj = P_o C(v + g)``, where ``P_o`` projects onto odd fields,
    ``G`` is the linear free-boundary inverse, ``C`` multiplies by
    ``dj/dpsi`` and ``g`` is the vacuum-flux response.

    Inputs must describe one frozen equilibrium satisfying the full
    equilibrium residual, with fixed mesh and coil geometry. ``G`` must
    commute with reflection, and ``C`` must represent the full first-order
    current-density variation for odd total-flux perturbations.
    Symmetry eliminates first-order scalar variations only for differentiable,
    reflection-invariant quantities; non-local terms, moving current
    interfaces and critical-point switching require separate treatment.

    Note also that:

    - Uniqueness requires an invertible odd tangent operator.
    - This computes a static response; it doesn't (on its own) determine growth rates.

    Args:
        inverse (Callable): Map 1D current density (A m^-2) to plasma
            flux (Wb/rad) on the same mesh.
        coefficient (array_like): Finite, even, 1D ``dj/dpsi`` array,
            in A m^-2 per (Wb/rad).
        vacuum_response (array_like): Finite, odd vacuum-flux response
            in Wb/rad per forcing ampere, matching coefficient's shape.
        reflection (array_like): Matching integer index permutation
            that restores the original ordering when applied twice.
        rtol (float): Relative GMRES tolerance. Defaults to 1e-9.
        maxiter (int): Maximum GMRES restart cycles, each with at most
            ``min(40, n)`` iterations for ``n`` vertices. Defaults to 20.

    Returns:
        dict: Responses per forcing ampere and convergence diagnostic:
            plasma_response: Odd plasma flux ``v`` (Wb/rad/A).
            current_response: Odd current density ``dj`` (A m^-2/A).
            relative_residual:
                ``norm(operator @ v - rhs) / max(norm(rhs), 1e-300)``.

    Raises:
        ValueError: Invalid input shapes, reflection, finiteness or parity.
        RuntimeError: GMRES failure, non-finite relative residual, or
            relative residual exceeding ``10 * rtol``.
    """
    # C is the local source derivative at one frozen equilibrium. The caller
    # supplies its profile physics; this solver only couples C to the inverse G.
    c = np.asarray(coefficient, dtype=float)
    g = np.asarray(vacuum_response, dtype=float)
    p = np.asarray(reflection)
    if c.ndim != 1 or g.shape != c.shape or p.shape != c.shape:
        raise ValueError("Expected equal-length one-dimensional arrays")
    # Applying reflection twice must recover the original vertex ordering.
    # Otherwise Po below is not a projection onto odd fields.
    n = len(c)
    if (
        not np.issubdtype(p.dtype, np.integer)
        or np.any(p < 0)
        or np.any(p >= n)
        or not np.array_equal(p[p], np.arange(n))
    ):
        raise ValueError("Reflection must be an involutive permutation")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(g)):
        raise ValueError("Nonfinite response input")
    if not np.allclose(c, c[p], rtol=1e-12, atol=0):
        raise ValueError("Source coefficient must be even")
    if np.linalg.norm(g + g[p]) > 1e-12 * max(np.linalg.norm(g), 1e-300):
        raise ValueError("Vacuum forcing must be odd")
    # Po removes even roundoff components on every inverse application. For
    # even C and a reflection-equivariant G, the odd subspace is closed.
    odd = lambda x: (x - x[p]) / 2
    # Linearise v = G[C(v+g)] and move the unknown plasma term to the left.
    # Matrix-free actions avoid storing a dense susceptibility matrix.
    operator = LinearOperator(
        (n, n), matvec=lambda x: x - odd(inverse(c * odd(x))), dtype=float
    )
    rhs = odd(inverse(c * g))
    # Restart limits the Krylov basis storage; maxiter counts restart cycles.
    # Check the equation ourselves as well as the library convergence flag.
    v, info = gmres(
        operator, rhs, rtol=rtol, atol=0, restart=min(40, n), maxiter=maxiter
    )
    v = odd(v)
    residual = np.linalg.norm(operator @ v - rhs) / max(np.linalg.norm(rhs), 1e-300)
    if info or not np.isfinite(residual) or residual > 10 * rtol:
        raise RuntimeError(
            f"Odd tangent did not converge: info={info}, residual={residual}"
        )
    # v is plasma flux only; the current responds to total flux v+g. Both
    # responses are per applied ampere, with no circuit model imposed here.
    return dict(
        plasma_response=v,
        current_response=odd(c * (v + g)),
        relative_residual=float(residual),
    )
